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

```py

```

# **Part 2: User Guide**

## **Overview**

TT-Train provides a complete training framework with automatic differentiation, optimized operations, and distributed training support—all designed specifically for Tenstorrent's tile-based scaleout architecture.

## **TTML vs PyTorch**

…

## **Supported Features**

### Optimizers

* `AdamW` \- AdamW with weight decay  
* `MorehAdamW` \- Moreh-optimized AdamW  
* `SGD` \- Stochastic Gradient Descent

### Schedulers

* `identity` \- Constant learning rate  
* `warmup_linear` \- Linear warmup \+ linear decay

### Distributed Training

| Strategy | Support | Description |
| ----- | ----- | ----- |
| Data Parallel (DDP) | ✅ | Replicate model, shard data |
| Tensor Parallel (TP) | ✅ | Shard layers across devices |
| Pipeline Parallel (PP) | ✅ | Shard layers sequentially |

### WanDB Integration

## **Known Issues**

## **Getting Started**

…

### **Example 1: Linear Regression**

A minimal example to understand the TT-Train programming model:

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
            
        # Reshape to [B, 1, 1, features] - TT-Train's expected format
        x_batch = X_train[batch_idx].reshape(batch_size, 1, 1, n_features)
        y_batch = y_train[batch_idx].reshape(batch_size, 1, 1, n_outputs)
        
        # Convert to TT-Train tensors
        tt_x = ttml.autograd.Tensor.from_numpy(x_batch)
        tt_y = ttml.autograd.Tensor.from_numpy(y_batch)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(tt_x)
        loss = ttml.ops.loss.mse_loss(pred, tt_y, ttml.ops.ReduceType.MEAN)
        
        # Backward pass
        loss.backward(False)
        
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

## **Scale Up**

## **Scale Out**

## **Profiling**

Perf / Memory / Tracy links

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

**1\. AutoContext** (`autograd/auto_context.hpp`)

The global singleton managing TT-Train's runtime state:

```c
class AutoContext {
public:
    static AutoContext& get_instance();
    
    // Device management
    void open_device();
    void close_device();
    ttnn::MeshDevice& get_device();
    
    // Graph management
    Graph& get_graph();
    void reset_graph();
    
    // Gradient tracking
    void set_gradient_mode(GradMode mode);
    bool is_gradient_enabled() const;
    
    // RNG
    void set_seed(uint32_t seed);
    
    // Backward node registration
    NodeId add_backward_node(GradFunction&& fn, std::span<NodeId> links);
};

// Convenience accessor
inline AutoContext& ctx() { return AutoContext::get_instance(); }
```

**2\. Graph** (`autograd/graph.hpp`)

Stores backward functions and dependencies:

```c
using GradFunction = std::function<void()>;

struct GraphNode {
    GradFunction grad_function;
};

class Graph {
    std::vector<GraphNode> m_graph_nodes;
    std::vector<std::vector<size_t>> m_links;  // Adjacency list
    
public:
    NodeId add_node(GradFunction&& grad_function, std::span<NodeId> links);
    void reset();  // Clear for next iteration
};
```

**3\. Tensor** (`autograd/tensor.hpp`)

The core tensor abstraction with autograd support:

```c
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
    tt::tt_metal::Tensor& get_grad();
    
    // Backward pass
    void backward(bool retain_graph = false);
};

using TensorPtr = std::shared_ptr<Tensor>;
```

### **Implementing a New Operation**

Every operation must implement both forward and backward:

```c
// Example: Element-wise multiplication
namespace ttml::ops {

autograd::TensorPtr mul(
    const autograd::TensorPtr& a,
    const autograd::TensorPtr& b
) {
    // 1. Create output tensor
    auto out = autograd::create_tensor();
    
    // 2. Forward: compute output value
    out->set_value(ttnn::multiply(a->get_value(), b->get_value()));
    
    // 3. Define backward function (captures inputs by shared_ptr)
    autograd::GradFunction grad = [a, b, out]() {
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
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    
    return out;
}

}  // namespace ttml::ops
```

### **Implementing a New C++ Module**

Modules encapsulate parameters and forward logic:

```c
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

### **Memory-Efficient Training**

Use gradient checkpointing for large models:

```c
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

```c
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

```c
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

