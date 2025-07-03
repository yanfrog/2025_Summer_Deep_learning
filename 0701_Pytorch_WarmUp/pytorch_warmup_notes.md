# Lab 0: PyTorch Warm-up
## 為何使用 PyTorch？

- 自動計算梯度 (`.backward()`)（相較於numpy）
- 易於構建複雜神經網路
- 可直接在 GPU 上運算（相較於numpy）
- 開發快速

## 基本觀念：Computational Graph

神經網路可用有向無環圖（DAG）表示  
PyTorch 會幫你追蹤運算圖並自動微分。

## 訓練流程總覽

```
Prepare Data → Create Model → Forward Pass → Backward Pass → Update Weights
(repeat until convergence)
```

---

## 用 PyTorch 手動實作 2-layer NN

- Input: 64 × 1000  
- Hidden: 100  
- Output: 10  
- Activation: ReLU  
- Loss: Mean Squared Error  

### Step 1: 準備資料
```python
x = torch.randn(64, 1000, device=device)
y = torch.randn(64, 10, device=device)
```

### Step 2: 初始化權重
```python
w1 = torch.randn(1000, 100, device=device, requires_grad=True)
w2 = torch.randn(100, 10, device=device, requires_grad=True)
```

### Step 3: Forward
```python
h = x.mm(w1).clamp(min=0)  # ReLU
y_pred = h.mm(w2)
loss = (y_pred - y).pow(2).sum()
```

### Step 4: Backward
```python
loss.backward()
```

### Step 5: 更新權重
```python
with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad
    w1.grad.zero_()
    w2.grad.zero_()
```

---

## 使用 PyTorch 高階 API

### Step 1: `torch.utils.data`
- `Dataset`：定義你自己的資料集類別
- `DataLoader`：提供 mini-batch、shuffle、多執行緒讀取

### Step 2: `torch.nn`
- `nn.Module`：神經網路模組基底類別
- 在 `__init__()` 中定義子模組
- 在 `forward()` 中定義前向傳播

### Step 3: `torch.nn.functional`
- 包含常用 activation/loss 函式
- ex: `F.relu`, `F.cross_entropy`

### Step 4: `torch.autograd`
- 自動建立運算圖，自動計算梯度
- 所有 `requires_grad=True` 的張量會被追蹤

### Step 5: `torch.optim`
- 常用 Optimizer: `SGD`, `Adam`, `Adadelta`, ...
- 使用 `.step()` 更新參數，`.zero_grad()` 清除梯度

---

## 實戰：MNIST CNN 分類器

範例程式碼：
https://github.com/pytorch/examples/tree/master/mnist

### 訓練流程

1. 設定超參數
2. 載入資料 (`torchvision.datasets`)
3. 定義模型 (`nn.Module`)
4. 定義 Loss 與 Optimizer
5. 加入 learning rate scheduler
6. 進行訓練與測試
7. 儲存模型

---

## 延伸閱讀

- PyTorch 官方文件: https://pytorch.org/docs/
- 自訂資料集範例: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
- Learning Rate Scheduler: https://wiki.cloudfactory.com/docs/mp-wiki/scheduler/steplr
