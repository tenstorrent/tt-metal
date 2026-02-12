# **TTML Onboarding Guide**

**TTML** is a high-performance C++/Python ML framework for Tenstorrent hardware.

---

# **Part 1: Install TTML**

## **Prerequisites**

| Requirement | Version | Notes |
| ----- | ----- | ----- |
| Ubuntu / WSL | 20.04+ | Tested on 22.04 |
| Python | 3.9+ |  |
| CMake | ≥ 3.20 | 3.30 recommended |
| tt-metal | Latest | **Must be installed first** |

## **Quick Install**

**Prerequisites:** Ensure `tt-metal` is installed first (see [INSTALLING.md](../INSTALLING.md)).

```bash
# 1. Activate tt-metal Python environment
cd /path/to/tt-metal
source python_env/bin/activate

# 2. Install tt-train
cd tt-train
pip install --no-build-isolation -e .

# 3. Verify installation
python -c "import ttml; print('TTML installed successfully')"
```

For development installation with C++ components, see [Development Installation](#development-installation) below.

# **Part 2: User Guide**

## **Overview**

TTML provides a complete training framework with automatic differentiation, optimized operations, and distributed training support—all designed specifically for Tenstorrent's tile-based scaleout architecture.

## **TTML vs PyTorch**

TTML provides a PyTorch-like API optimized for Tenstorrent hardware:

| Feature | PyTorch | TTML |
|---------|---------|------|
| **API Style** | Python-first | C++/Python hybrid |
| **Autograd** | ✅ Dynamic graphs | ✅ Dynamic graphs |
| **Device** | CUDA/CPU | Tenstorrent hardware |
| **Tensor Layout** | Row-major | Tile-based (optimized for TT) |
| **Distributed** | DDP, FSDP | DDP, TP (FSDP-style), PP (native) |

**Key Differences:**
- **Tensor shapes**: TTML expects `[B, 1, 1, features]` format for many operations (tile-aligned)
- **Graph reset**: Must call `ctx.reset_graph()` after each backward pass
- **Loss retrieval**: Get loss value AFTER `backward()` using `loss.get_value().item()`
- **Layout awareness**: Operations respect `ttnn.Layout.TILE` vs `ttnn.Layout.ROW_MAJOR`
- **Device mesh**: Native support for scale-up (multiple devices on same host) and scale-out (multiple hosts) via unified mesh API
- **Process model**: **1 process per host** (unlike PyTorch which spawns 1 process per device/GPU)

**Similarities:**
TTML aims for a **very similar API and developer experience** to PyTorch:
- Autograd tensor API (`requires_grad`, `backward()`)
- Module-based architecture (`nn.Module` → `ModuleBase`)
- Optimizer interface (`zero_grad()`, `step()`)
- Training/eval modes (`model.train()`, `model.eval()`)

**Note:** Most differences from PyTorch arise from TTML's **native support for scale-out training** (unified mesh API, single-process-per-host architecture, and built-in distributed strategies).

## **Supported Features**

### Optimizers

* `AdamW` \- AdamW with weight decay
* `MorehAdamW` \- Moreh-optimized AdamW
* `SGD` \- Stochastic Gradient Descent
* *Remote*  \- Used for multihost training

### Schedulers

* `identity` \- Constant learning rate
* `warmup_linear` \- Linear warmup \+ linear decay

### Distributed Training

| Strategy | Support | Description |
| ----- | ----- | ----- |
| Data Parallel (DDP) | ✅ | Replicate model, shard data, synchronize gradients |
| Tensor Parallel (TP) | ✅ | FSDP-style: Shard parameters across devices, gather/reduce as needed |
| Pipeline Parallel (PP) | ✅ | Shard layers sequentially across devices |

### WanDB Integration

TTML supports Weights & Biases (wandb) for experiment tracking. Training metrics are automatically logged during training.

**Setup:**

1. **Install wandb** (if not already installed):
```bash
pip install wandb
```

2. **Login to wandb** (first time only):
```bash
wandb login
```

3. **Configure in training config**:
```yaml
training_config:
  project_name: "my_training_project"  # wandb project name
```

4. **Disable wandb** (if desired):
```bash
# Option 1: Use offline mode
wandb offline

# Option 2: Disable in code (C++ examples)
# Pass -w 0 flag to disable wandb logging
```

**What gets logged:**
- Training loss per step
- Learning rate (if using scheduler)
- Model parameters count
- Training metrics (samples/sec, tokens/sec)

**Example wandb dashboard:**
See the [NanoGPT training example](https://wandb.ai/tenstorrent-ml/tt_train_nano_gpt) for a live dashboard.


## **Getting Started**

**Core Concepts:**

1. **AutoContext**: Global singleton managing device, graph, and distributed state
2. **Tensor**: Autograd-enabled tensor wrapper around `ttnn.Tensor`
3. **Graph**: Computation graph for automatic differentiation
4. **Modules**: Composable layers (Linear, Attention, etc.)
5. **Optimizers**: Parameter update algorithms (AdamW, SGD)

**Typical Training Flow:**

```python
# 1. Initialize context
ctx = ttml.autograd.AutoContext.get_instance()
ctx.open_device()

# 2. Create model
model = create_model(...)

# 3. Setup optimizer
optimizer = ttml.optimizers.AdamW(model.parameters(), config)

# 4. Training loop
for batch in dataloader:
    optimizer.zero_grad()
    pred = model(input)
    loss = compute_loss(pred, target)
    loss.backward()
    optimizer.step()
    ctx.reset_graph()  # Important: reset after each iteration
```

See the examples below for complete working code.

### **Quick Start: Run TinyLlama Inference**

Get started quickly by running an existing model. This example loads TinyLlama from Hugging Face and runs inference:

```python
import os
import sys
import numpy as np
import ttml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from ttml.common.config import load_config
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import initialize_device, set_seed

# ============================================================
# 1. Setup
# ============================================================
set_seed(42)
os.chdir(os.path.join(os.environ.get("TT_METAL_HOME", "."), "tt-train"))

# ============================================================
# 2. Load Model Configuration
# ============================================================
model_config_path = "configs/model_configs/tinyllama.yaml"
model_yaml = load_config(model_config_path)
print(f"Loaded model config: {model_config_path}")

# ============================================================
# 3. Initialize Device
# ============================================================
device_config = {"device_config": {"mesh_shape": [1, 1]}}  # Single device
initialize_device(device_config)
ctx = ttml.autograd.AutoContext.get_instance()
ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)  # Inference mode
print("Device initialized")

# ============================================================
# 4. Create and Load Model
# ============================================================
model_factory = TransformerModelFactory(model_yaml)
model = model_factory.create_model()
model.eval()

# Download and load weights from Hugging Face
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
safetensors_dir = hf_hub_download(repo_id=model_path, filename="model.safetensors")
safetensors_dir = safetensors_dir.replace("model.safetensors", "")
model.load_from_safetensors(safetensors_dir)
print(f"Model loaded from: {model_path}")

# ============================================================
# 5. Load Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

# ============================================================
# 6. Run Inference
# ============================================================
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="np")[0]

# Convert to TTML tensor format [B, 1, 1, seq_len]
input_tensor = ttml.autograd.Tensor.from_numpy(
    input_ids.reshape(1, 1, 1, len(input_ids)),
    layout=ttml.Layout.ROW_MAJOR,
    new_type=ttml.DataType.UINT32
)

# Create causal mask
seq_len = len(input_ids)
mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
mask = ttml.autograd.Tensor.from_numpy(
    mask_np.reshape(1, 1, seq_len, seq_len),
    layout=ttml.Layout.TILE,
    new_type=ttml.DataType.BFLOAT16
)

# Forward pass
logits = model(input_tensor, mask)
output_ids = ttml.argmax(logits.get_value(), dim=-1).to_numpy(ttml.DataType.UINT32)

# Decode output
generated_text = tokenizer.decode(output_ids.flatten()[:len(input_ids)])
print(f"\nPrompt: {prompt}")
print(f"Generated: {generated_text}")

# ============================================================
# 7. Cleanup
# ============================================================
ctx.close_device()
```

**Quick Command (C++):**
```bash
# Build the inference example
cd $TT_METAL_HOME/tt-train
cmake --build build --target llama_inference

# Run with pre-trained weights
./build/sources/examples/nano_gpt/llama_inference \
  --model-path path/to/model.msgpack \
  --prompt "1,2,3" \
  --max-tokens 50
```

**Next Steps:**
- See [llm_inference.ipynb](../sources/examples/llm_inference/llm_inference.ipynb) for a complete Jupyter notebook example
- Try training your own model (see examples below)
- Explore distributed training with multiple devices

### **Example 1: Linear Regression**

[Example link](/tt-train/sources/examples/linear_regression/linear_regression.py)

To run the regression example: `python3 tt-train/sources/examples/linear_regression/linear_regression.py`

A minimal example to understand the TTML programming model:

```py
import numpy as np
import ttnn
import ttml

# ============================================================
# 1. Initialize Device Context
# ============================================================
ctx = ttml.autograd.AutoContext.get_instance()
ctx.open_device()
ctx.set_seed(42)

# ============================================================
# 2. Create Model
# ============================================================
n_features = 2
n_outputs = 1
model = ttml.models.linear_regression.create_linear_regression_model(n_features, n_outputs)

# ============================================================
# 3. Configure Optimizer
# ============================================================
opt_cfg = ttml.optimizers.SGDConfig.make(
    lr=0.1,
    momentum=0.0,
    weight_decay=0.0,
    dampening=0.0,
    nesterov=False
)
optimizer = ttml.optimizers.SGD(model.parameters(), opt_cfg)

# ============================================================
# 4. Prepare Data
# ============================================================
# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(256, n_features).astype(np.float32)
y_train = (X_train @ np.array([[2.0], [-1.0]])).astype(np.float32)

# ============================================================
# 5. Training Loop
# ============================================================
model.train()
batch_size = 32
epochs = 10

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        if len(batch_idx) < batch_size:
            continue

        # Reshape to [B, 1, 1, features] - TTML's expected format
        x_batch = X_train[batch_idx].reshape(batch_size, 1, 1, n_features)
        y_batch = y_train[batch_idx].reshape(batch_size, 1, 1, n_outputs)

        # Convert to TTML tensors
        tt_x = ttml.autograd.Tensor.from_numpy(x_batch)
        tt_y = ttml.autograd.Tensor.from_numpy(y_batch)

        # Forward pass
        optimizer.zero_grad()
        pred = model(tt_x)
        loss = ttml.ops.loss.mse_loss(pred, tt_y, ttml.ops.ReduceType.MEAN)

        # Backward pass
        loss.backward(False)  # retain graph = False

        # Get loss value AFTER backward
        loss_val = float(loss.get_value().item())
        epoch_loss += loss_val
        n_batches += 1

        # Update weights
        optimizer.step()

        # Reset computation graph
        ctx.reset_graph()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.6f}")

# ============================================================
# 6. Inference
# ============================================================
model.eval()
test_x = np.array([[1.0, 1.0]], dtype=np.float32).reshape(1, 1, 1, n_features)
tt_test = ttml.autograd.Tensor.from_numpy(test_x)
prediction = model(tt_test).to_numpy(ttnn.DataType.FLOAT32)
print(f"Prediction for [1, 1]: {prediction.flatten()[0]:.4f}")  # Should be ~1.0

# ============================================================
# 7. Cleanup
# ============================================================
ctx.close_device()
```

### **Example 2: NanoGPT Training**

[Full Python nanogpt training example](/tt-train/sources/examples/nano_gpt/train_nanogpt.py).

To run the nanogpt Python training example: `python3 tt-train/sources/examples/nano_gpt/train_nanogpt.py -c training_shakespeare_nanogpt.yaml`

[Full C++ nanogpt training example](/tt-train/sources/examples/nano_gpt/main.cpp).

To run the nanogpt C++ training example:
```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal
cd tt-train
./build/sources/examples/nano_gpt/nano_gpt -c ./configs/training_configs/training_shakespeare_nanogpt.yaml
```

A complete language model training example:

```py
import numpy as np
import ttnn
import ttml
from ttml.models.nanogpt import create_nanogpt, NanoGPTConfig
from ttml.common.data import CharTokenizer

# ============================================================
# 1. Initialize
# ============================================================
ctx = ttml.autograd.AutoContext.get_instance()
ctx.open_device()
ctx.set_seed(42)

# ============================================================
# 2. Load Data & Create Tokenizer
# ============================================================
with open("data/shakespeare.txt", "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
tokens = tokenizer.encode(text)

# Create sequences
sequence_length = 128
dataset = []
for i in range(len(tokens) - sequence_length):
    seq = tokens[i:i+sequence_length]
    target = tokens[i+1:i+sequence_length+1]
    dataset.append((seq, target))

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Dataset size: {len(dataset)} sequences")

# ============================================================
# 3. Create Model
# ============================================================
# Round vocab to tile boundary (multiple of 32)
vocab_size = ((tokenizer.vocab_size + 31) // 32) * 32

config = NanoGPTConfig(
    vocab_size=vocab_size,
    block_size=sequence_length,
    n_embd=384,
    n_layer=6,
    n_head=6,
    dropout=0.2,
    bias=True
)
model = create_nanogpt(config)

# ============================================================
# 4. Configure Optimizer
# ============================================================
adamw_config = ttml.optimizers.AdamWConfig.make(
    lr=3e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01
)
optimizer = ttml.optimizers.AdamW(model.parameters(), adamw_config)

# ============================================================
# 5. Create Causal Mask
# ============================================================
mask_np = np.tril(np.ones((sequence_length, sequence_length), dtype=np.float32))
mask_np = mask_np.reshape(1, 1, sequence_length, sequence_length)
causal_mask = ttml.autograd.Tensor.from_numpy(
    mask_np,
    layout=ttnn.Layout.TILE,
    new_type=ttnn.DataType.BFLOAT16
)

# ============================================================
# 6. Training Loop
# ============================================================
batch_size = 4
max_steps = 1000
gradient_accumulation_steps = 4

model.train()

for step in range(max_steps):
    # Sample random batch
    batch_indices = np.random.randint(0, len(dataset), batch_size)

    # Prepare batch
    input_ids = []
    target_ids = []
    for idx in batch_indices:
        seq, tgt = dataset[idx]
        input_ids.extend(seq)
        target_ids.extend(tgt)

    input_np = np.array(input_ids, dtype=np.uint32).reshape(batch_size, 1, 1, sequence_length)
    target_np = np.array(target_ids, dtype=np.uint32).reshape(batch_size, sequence_length)

    input_tensor = ttml.autograd.Tensor.from_numpy(
        input_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )
    target_tensor = ttml.autograd.Tensor.from_numpy(
        target_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    # Zero gradients at accumulation boundary
    if step % gradient_accumulation_steps == 0:
        optimizer.zero_grad()

    # Forward pass
    logits = model(input_tensor, causal_mask)

    # Compute loss
    loss = ttml.ops.loss.cross_entropy_loss(
        logits, target_tensor, reduce=ttml.ops.ReduceType.MEAN
    )

    # Scale loss for gradient accumulation
    if gradient_accumulation_steps > 1:
        loss = ttml.ops.binary.mul(loss, 1.0 / gradient_accumulation_steps)

    # Get loss value
    loss_val = float(loss.get_value().item())

    # Backward pass
    loss.backward(False)
    ctx.reset_graph()

    # Optimizer step at accumulation boundary
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        print(f"Step {step+1}, Loss: {loss_val * gradient_accumulation_steps:.4f}")

# ============================================================
# 7. Text Generation
# ============================================================
model.eval()

def generate(prompt, max_tokens=100):
    prompt_ids = tokenizer.encode(prompt)

    # Pad to sequence length
    if len(prompt_ids) < sequence_length:
        padding = [0] * (sequence_length - len(prompt_ids))
        prompt_ids = padding + prompt_ids

    generated = []
    running = list(prompt_ids[-sequence_length:])

    for _ in range(max_tokens):
        # Prepare input
        input_np = np.array(running, dtype=np.uint32).reshape(1, 1, 1, sequence_length)
        input_tensor = ttml.autograd.Tensor.from_numpy(
            input_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward
        logits = model(input_tensor, causal_mask)

        # Sample from last position (greedy)
        last_logits = logits.get_value()
        next_id = int(ttnn.argmax(last_logits, dim=-1).item()) % tokenizer.vocab_size

        generated.append(next_id)
        running = running[1:] + [next_id]
        ctx.reset_graph()

    return tokenizer.decode(generated)

print("\nGenerated text:")
print(generate("ROMEO: ", max_tokens=200))

ctx.close_device()
```

## **Build your own model**

### Implementing a Module
You can implement your own model as a Module consisting of submodules (either self-written or provided by TTML) and Parameters, similar to PyTorch.

You need to implement initialization of Modules/Parameters and a forward pass. TTML's autograd engine will automatically implement the backward pass.

#### Python module

Here's an example of the [LayerNorm implementation in Python](/tt-train/sources/ttml/ttml/models/nanogpt/gpt_block.py). It creates weights (Parameters) for gamma (scale) and beta (shift/bias) of LayerNorm manually in the Module's `__init__`. In the forward pass, it simply calls `ttml.ops.layernorm.layernorm`. `ttml.ops.layernorm.layernorm` is responsible for building the autograd graph and propagating gradients to inputs and weights.

```python

class LayerNorm(AbstractModuleBase):
    """Layer normalization module with gamma and beta parameters."""

    def __init__(self, embedding_dim: int, bias: bool = True) -> None:
        """Initialize layer norm.

        Args:
            embedding_dim: Dimension of embeddings
            bias: Whether to use bias (beta) parameter
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Layer norm requires gamma (scale) and beta (shift) parameters
        ln_shape = (1, 1, 1, embedding_dim)
        gamma_np = np.ones(ln_shape, dtype=ml_dtypes.bfloat16)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(
            gamma_np, layout=ttnn.Layout.TILE
        )
        self.gamma = Parameter(gamma_tensor)

        if bias:
            beta_np = np.zeros(ln_shape, dtype=ml_dtypes.bfloat16)
            beta_tensor = ttml.autograd.Tensor.from_numpy(
                beta_np, layout=ttnn.Layout.TILE
            )
            self.beta = Parameter(beta_tensor)
        else:
            self.beta = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of layer norm.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        return ttml.ops.layernorm.layernorm(
            x, self.gamma.tensor, self.beta.tensor if self.beta else None
        )
```

Here's the [implementation of the GPTMLP module in Python](/tt-train/sources/ttml/ttml/models/nanogpt/gpt_mlp.py), which uses the Linear submodule (`ttml.modules.LinearLayer`) as well as the ttml operation (`ttml.ops.dropout`):

```python
class GPTMLP(AbstractModuleBase):
    """GPT-style MLP (feed-forward) layer."""

    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        """Initialize GPT MLP layer.

        Args:
            embedding_dim: Dimension of embeddings
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout
        # Note: RunMode is managed by AbstractModuleBase (defaults to TRAIN)

        # First linear: embedding_dim -> embedding_dim * 4
        self.fc1 = LinearLayer(embedding_dim, embedding_dim * 4, True)

        # Second linear: embedding_dim * 4 -> embedding_dim
        self.fc2 = LinearLayer(embedding_dim * 4, embedding_dim, True)

    # train() and eval() are inherited from AbstractModuleBase

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor after MLP
        """
        x = self.fc1(x)
        x = ttml.ops.unary.gelu(x)
        x = self.fc2(x)

        # Note: It's better to just use Dropout module here
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob)

        return x
```

#### C++ module
C++ modules are fairly similar to Python modules, with one small difference: they require manual `register_module`, `register_tensor`, and `create_name` calls.
[linear_module.hpp](/tt-train/sources/ttml/modules/linear_module.hpp), [linear_module.cpp](/tt-train/sources/ttml/modules/linear_module.cpp), [gpt_block.hpp](/tt-train/sources/ttml/modules/gpt_block.hpp), [gpt_block.cpp](/tt-train/sources/ttml/modules/gpt_block.cpp)

```cpp
// gpt_block.hpp
class GPTMLP : public modules::ModuleBase {
    std::shared_ptr<LinearLayer> fc1;
    std::shared_ptr<LinearLayer> fc2;
    std::shared_ptr<DropoutLayer> dropout;

public:
    GPTMLP(uint32_t embedding_size, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;
};

// gpt_block.cpp
GPTMLP::GPTMLP(uint32_t embedding_size, float dropout_prob) {
    fc1 = std::make_shared<LinearLayer>(embedding_size, embedding_size * 4);
    fc2 = std::make_shared<LinearLayer>(embedding_size * 4, embedding_size);
    dropout = std::make_shared<DropoutLayer>(dropout_prob);

    create_name("gpt_mlp");
    register_module(fc1, "fc1");
    register_module(fc2, "fc2");
    register_module(dropout, "dropout");
}

autograd::TensorPtr GPTMLP::operator()(const autograd::TensorPtr& input) {
    auto x = (*fc1)(input);
    x = ops::gelu(x);
    x = (*fc2)(x);
    x = (*dropout)(x);
    return x;
}
```

### Implementing a new operation

#### Python operation
Unfortunately, there is currently no way to implement operations in Python, but this feature is coming soon.

#### C++ operation
TTML operations are responsible for:
1. Forward pass
2. Lifetime of activations
3. Backward pass
4. Populating the autograd graph

TTML operations rely on TTNN operations.

Example: Element-wise multiplication:

```cpp
namespace ttml::ops {

autograd::TensorPtr mul(
    const autograd::TensorPtr& a,
    const autograd::TensorPtr& b
) {
    // 1. Forward pass
    // Create output tensor
    auto out = autograd::create_tensor();

    // Compute output value
    out->set_value(ttnn::multiply(a->get_value(), b->get_value()));

    // 2. Gradient callback lambda captures activations that are required for the backward pass.
    // Make sure that you're not capturing anything not needed, since this might increase memory usage.
    autograd::GradFunction grad = [a, b, out]() {
        // 3. Backward implementation
        // d(a*b)/da = b, d(a*b)/db = a
        // Chain rule: grad_a = upstream_grad * b
        auto a_grad = ttnn::multiply(out->get_grad(), b->get_value());
        auto b_grad = ttnn::multiply(out->get_grad(), a->get_value());

        // Accumulate gradients (supports multi-use tensors)
        a->add_grad(a_grad);
        b->add_grad(b_grad);
    };

    // 4. Register in computation graph
    auto links = autograd::get_links(a, b);
    // Note: add_backward_node returns std::optional<NodeId> (nullopt if grads disabled)
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
```


## **Scale Up**

Scale up refers to optimizing training on a single device or multiple devices on the **same host** by:
- Increasing batch size (within device memory limits)
- Using gradient accumulation to simulate larger batches
- Enabling memory-efficient training (gradient checkpointing)
- Optimizing tensor layouts and operations
- **Tensor Parallel (TP)**: Shard model weights across devices (FSDP-style)
- **Distributed Data Parallel (DDP)**: Replicate model, shard data across devices

**Key Architecture Benefit:**
TTML uses a **unified mesh API** where a single process manages all devices on a host. Unlike PyTorch (which spawns 1 process per device), TTML spawns **1 process per host**, simplifying multi-device coordination and reducing overhead.

**Example: Single Device Optimization**

```python
# Single device (N150/N300)
device_config:
  mesh_shape: [1, 1]  # Single device
  enable_ddp: false
  enable_tp: false

# Use gradient accumulation for effective larger batch
training_config:
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch size = 32

# Enable memory-efficient mode for larger models
transformer_config:
  runner_type: "memory_efficient"  # Gradient checkpointing
```

**Data Parallel Example:**

```python
# 4-device DDP (e.g., LoudBox with 4 N300s)
device_config:
  mesh_shape: [1, 4]  # 4 devices in a row
  enable_ddp: true
  enable_tp: false

training_config:
  batch_size: 16  # Will be split across 4 devices (4 per device)
```

**Memory Optimization Tips:**
- Use `memory_efficient` runner type to reduce activation memory
- Reduce batch size and increase gradient accumulation
- Monitor memory usage with `MemoryUsageTracker` (see [Profiling](#profiling))

## **Scale Out**

Scale out refers to distributed training across **multiple hosts** (each host can have multiple devices) using:
- **Pipeline Parallel (PP)**: Shard layers sequentially across devices
- **Multi-host**: Scale across multiple machines with unified mesh topology

**Architecture:**
- **Scale-up** (same host): A single process manages all devices via `mesh_shape: [1, N]` (e.g., `[1, 8]` for LoudBox)
- **Scale-out** (multiple hosts): One process per host, coordinated via MPI or Fabric communication
- **Unified API**: The same mesh configuration works for both single-host and multi-host scenarios


**2/3-tier Architecture**

Currently, we employ a 2- or 3-tier architecture when performing multi-host training.

All hosts are split into 2 or 3 roles: Worker / AggregatorOptimizer (2-tier) or Worker / Aggregator / Optimizer (3-tier).

In the 3-tier architecture, the Aggregator gathers and reduces gradients from every worker. Then it sends the gathered gradients to the Optimizer, which performs the optimizer step and sends new weights back to the Aggregator, which broadcasts them back to workers. In the 2-tier architecture, the Aggregator and Optimizer are merged into a single node.

For more details and instructions on how to run, see [this doc](/tt-train/sources/examples/python/multihost/hierarchical_parallel/README.md)

**Tensor Parallel Example (FSDP-style):**

Tensor Parallel in TTML implements FSDP (Fully Sharded Data Parallel) semantics:
- **Parameter sharding**: Model weights are sharded across devices using `shard_tensor_to_mesh_mapper`
- **Gather on demand**: Parameters are gathered when needed during the forward pass (via `all_gather`)
- **Reduce gradients**: Gradients are reduced across devices during the backward pass (via `reduce_scatter`)

```python
# 32-device TP (e.g., Galaxy)
device_config:
  mesh_shape: [1, 32]  # 32 devices
  enable_ddp: false
  enable_tp: true  # Enables FSDP-style parameter sharding

# Vocab size automatically rounded to (32 * 32) = 1024 boundary
transformer_config:
  vocab_size: 32000  # Will be padded to 32768
```

**Key Differences from DDP:**
- **DDP**: Full model replication → higher memory usage, only gradients synchronized
- **TP (FSDP)**: Parameter sharding → lower memory usage per device, parameters gathered/reduced as needed

**Pipeline Parallel Example:**

```python
# Multi-host pipeline parallel
multihost_config:
  enabled: true
  num_workers: 4
  socket_type: "fabric"  # or "mpi"
  pipeline_parallel_config:
    num_blocks: 24
    blocks_per_rank:
      0: 6  # First 6 blocks on rank 0
      1: 6  # Next 6 blocks on rank 1
      2: 6  # Next 6 blocks on rank 2
      3: 6  # Last 6 blocks on rank 3
```

**Device Mesh Shapes:**
- **Single device**: `[1, 1]` (N150, P150) - 1 process, 1 device
- **Dual device**: `[1, 2]` (N300, P300) - 1 process, 2 devices
- **LoudBox**: `[1, 8]` (8 devices) - 1 process, 8 devices
- **Galaxy**: `[1, 32]` (32 devices) - 1 process, 32 devices
- **Multi-host**: Multiple processes (1 per host), each managing its own device mesh

**Process Model Comparison:**
| Scenario | PyTorch | TTML |
|----------|---------|------|
| 8 GPUs on 1 host | 8 processes | **1 process** |
| 32 devices on 1 host | 32 processes | **1 process** |
| 5 hosts, 8 devices each | 40 processes | **5 processes** (1 per host) |

See [configs/README.md](../configs/README.md) for detailed configuration options.

## **Profiling**

TTML provides comprehensive profiling tools for performance analysis and memory tracking.

### **Performance Profiling (Tracy)**

Profile kernel execution times and identify bottlenecks:

**Build with profiling:**
```bash
./build_metal.sh -b Release --build-tt-train
```

**Run with Tracy:**
```bash
env -u TT_METAL_DPRINT_CORES \
TT_METAL_WATCHER_NOINLINE=1 \
python -m tracy -r -v -p ./build/sources/examples/nano_gpt/nano_gpt
```

**Output:**
- `ops_perf_results_<timestamp>.csv`: Per-kernel performance data
- `tracy_profile_log_host.tracy`: Tracy GUI file for visualization

**Analyze results:**
```bash
jupyter lab notebooks/profiler_results.ipynb
```

See [PROFILER.md](./PROFILER.md) for detailed profiling guide.

### **Memory Tracking**

Track memory usage across training phases:

```python
# Access via ttml.core.utils
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker

# Begin tracking
guard = MemoryUsageTracker.begin_capture()

# Take snapshots at key points
MemoryUsageTracker.snapshot("MODEL_CREATION")
MemoryUsageTracker.snapshot("FORWARD_PASS")
MemoryUsageTracker.snapshot("BACKWARD_PASS")

# Print results and cleanup
MemoryUsageTracker.end_capture("ITERATION_COMPLETE")
MemoryUsageTracker.print_memory_usage()
```

**Analyze memory logs:**
```bash
python scripts/analyze_memory.py \
  --logs memory.log \
  --visualize_peak \
  --title "Training Memory Analysis"
```

See [MEMORY_TRACKING.md](./MEMORY_TRACKING.md) for detailed memory tracking guide.

### **Performance Metrics**

Track training performance:

```python
from ttml.common.utils import PerformanceMeter

meter = PerformanceMeter(cfg)
# ... training loop ...
samples_per_second, tokens_per_second = meter.get_metrics()
print(f"Samples/s: {samples_per_second:.2f}, Tokens/s: {tokens_per_second:.2f}")
```

# **Part 3: Contributing**

## **Development Installation**

### **Step 1: Clone and Setup tt-metal**

```shell
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git submodule update --init --recursive
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
```

### **Step 2: Install TT-Train (Editable)**

```shell
cd tt-train
pip install --no-build-isolation -e .
```

The `-e` flag enables editable mode—changes to Python code are reflected immediately.

### **Step 3: Build C++ Components**

```shell
cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja
cmake --build build
```

## **Project Structure**

```
tt-train/
├── sources/
│   ├── ttml/                    # Core library
│   │   ├── autograd/            # Computation graph & tensors
│   │   │   ├── auto_context.hpp # Global context singleton
│   │   │   ├── graph.hpp        # Backward graph
│   │   │   ├── tensor.hpp       # Autograd tensor wrapper
│   │   │   └── autocast_tensor.hpp
│   │   ├── core/                # Utilities
│   │   │   ├── tt_tensor_utils.hpp  # Tensor conversions
│   │   │   └── clip_grad_norm.hpp
│   │   ├── ops/                 # Operations with forward/backward
│   │   │   ├── binary_ops.cpp   # +, -, *, /
│   │   │   ├── matmul_op.cpp
│   │   │   ├── losses.cpp       # MSE, CE, NLL
│   │   │   └── distributed/     # all_reduce, all_gather, etc.
│   │   ├── modules/             # Composable layers
│   │   │   ├── linear_layer.hpp
│   │   │   ├── attention.hpp
│   │   │   └── distributed/     # TP modules
│   │   ├── models/              # Full model implementations
│   │   │   ├── gpt2.hpp
│   │   │   └── llama.hpp
│   │   ├── optimizers/          # AdamW, SGD
│   │   ├── schedulers/          # LR schedulers
│   │   ├── metal/               # Custom TT kernels
│   │   └── nanobind/            # Python bindings
│   └── examples/                # Example applications
├── tests/                       # Test suite
├── configs/                     # YAML configurations
└── docs/                        # Documentation
```

## **C++ Architecture Deep Dive**

### **Autograd System**

**1. AutoContext** (`autograd/auto_context.hpp`)

The global singleton managing TTML's runtime state:

```cpp
class AutoContext {
public:
    static AutoContext& get_instance();

    // Device management
    void open_device();
    void close_device();
    ttnn::MeshDevice& get_device();

    // Graph management
    Graph& get_graph();  // Access to underlying graph (advanced usage)
    void reset_graph();

    // Gradient tracking
    void set_gradient_mode(GradMode mode);
    GradMode get_gradient_mode() const;

    // RNG
    void set_seed(uint32_t seed);

    // Backward node registration
    std::optional<NodeId> add_backward_node(GradFunction&& fn, std::span<NodeId> links);
};

// Convenience accessor
inline AutoContext& ctx() { return AutoContext::get_instance(); }
```

**2. Graph** (`autograd/graph.hpp`)

Stores backward functions and dependencies:

```cpp
using GradFunction = std::function<void()>;

struct GraphNode {
    GradFunction grad_function;
};

class NodeId {
public:
    NodeId(size_t node_id, Graph* graph);
    size_t get_id() const;
    Graph& get_graph() const;
private:
    size_t m_node_id;
    Graph* m_graph;
};

class Graph {
    std::vector<GraphNode> m_graph_nodes;
    std::vector<std::vector<size_t>> m_links;  // Adjacency list

public:
    NodeId add_node(GradFunction&& grad_function, std::span<NodeId> links);
    void reset();  // Clear for next iteration
};
```

**3. Tensor** (`autograd/tensor.hpp`)

The core tensor abstraction with autograd support:

```cpp
class Tensor : public std::enable_shared_from_this<Tensor> {
    AutocastTensor m_value;           // Forward value
    tt::tt_metal::Tensor m_grad;      // Accumulated gradient
    bool m_requires_grad = true;
    std::optional<NodeId> m_node_id;  // Link to backward graph

public:
    // Value access
    const tt::tt_metal::Tensor& get_value(
        PreferredPrecision precision = PreferredPrecision::HALF) const;

    // Gradient operations
    void set_grad(const tt::tt_metal::Tensor& grad);
    void add_grad(const tt::tt_metal::Tensor& grad);  // Accumulates!
    const tt::tt_metal::Tensor& get_grad() const;
    tt::tt_metal::Tensor& get_grad();  // Non-const version

    // Backward pass
    void backward(bool retain_graph = false);
};

using TensorPtr = std::shared_ptr<Tensor>;
```

### **Implementing a New Operation**

Every operation must implement both forward and backward. [See example](#implementing-a-new-operation)

### **Implementing a New C++ Module**

Modules encapsulate parameters and forward logic:

```cpp
// Example: Custom layer
class MyLayer : public ModuleBase {
    LinearLayer m_linear;
    RMSNormLayer m_norm;

public:
    MyLayer(uint32_t in_features, uint32_t out_features)
        : m_linear(in_features, out_features)
        , m_norm(out_features)
    {
        // Register sub-modules (for parameter collection)
        register_module("linear", m_linear);
        register_module("norm", m_norm);
    }

    autograd::TensorPtr forward(const autograd::TensorPtr& input) override {
        auto x = m_linear(input);
        x = m_norm(x);
        x = ops::gelu(x);
        return x;
    }
};
```

See [more examples](#implementing-a-module)

### **Memory-Efficient Training**

Use gradient checkpointing for large models:

```cpp
#include <ttml/models/common/transformer_common.hpp>

// Wrap block forward in memory_efficient_runner
for (auto& block : blocks) {
    if (use_checkpointing) {
        out = common::transformer::memory_efficient_runner(*block, out, mask);
    } else {
        out = (*block)(out, mask);
    }
}
```

## **Testing**

### **Running Tests**

```shell
# All tests
ctest --test-dir build

# Specific test
./build/tests/ops/test_matmul

# With verbose output
ctest --test-dir build -V
```

### **Writing Tests**

Tests use Google Test:

```cpp
#include <gtest/gtest.h>
#include <ttml/ttml.hpp>

TEST(OpsTest, MatmulForward) {
    auto& ctx = ttml::autograd::ctx();
    ctx.open_device();

    // Create inputs
    auto a = create_random_tensor({32, 64});
    auto b = create_random_tensor({64, 128});

    // Forward
    auto c = ttml::ops::matmul(a, b);

    // Verify shape
    auto shape = c->get_shape();
    EXPECT_EQ(shape[0], 32);
    EXPECT_EQ(shape[1], 128);

    ctx.close_device();
}

TEST(OpsTest, MatmulBackward) {
    // ... test gradient computation
}
```

### **Nightly Tests**

Long-running tests are prefixed with `NIGHTLY_`:

```shell
export ENABLE_NIGHTLY_TT_TRAIN_TESTS=1
ctest --test-dir build -R NIGHTLY_
```

## **Profiling**

### **Tracy Profiler**

```shell
# Build with profiling
./build_metal.sh -b Release --build-tt-train

# Run with Tracy
python -m tracy -r -v -p ./build/sources/examples/nano_gpt/nano_gpt
```

### **Memory Tracking**

```cpp
#include <ttml/utils/memory_utils.hpp>

// Start tracking
auto guard = ttml::utils::MemoryUsageTracker::begin_capture();

// ... training code ...
ttml::utils::MemoryUsageTracker::snapshot("FORWARD");
// ... more code ...
ttml::utils::MemoryUsageTracker::snapshot("BACKWARD");

// Print results
ttml::utils::MemoryUsageTracker::end_capture("COMPLETE");
ttml::utils::MemoryUsageTracker::print_memory_usage();
```

Analyze with:

```shell
python scripts/analyze_memory.py --logs memory.log --visualize_peak
```

---
