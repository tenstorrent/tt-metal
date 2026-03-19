# Distributed Training Guide

This guide explains how to configure and run distributed training with TTML, covering Mesh Graph Descriptors (MGD files), device configuration, and parallelism strategies.

## Overview

TTML supports four types of distributed parallelism:
- **Data Parallel (DP/DDP)**: Replicate model across devices, shard data, synchronize gradients
- **Tensor Parallel (TP)**: Shard model parameters across devices, gather/reduce as needed
- **Context Parallel (CP)**: Shard sequence length across devices for long-context training
- **Pipeline Parallel (PP)**: Shard layers sequentially across devices (multi-host)

These can be combined (e.g., TP + DDP, or CP + TP) to scale training across large device meshes.

---

## Parallelism Context

### What is ParallelismContext?

`ParallelismContext` is the central configuration object that manages distributed training in TTML. It determines:
- Which parallelism strategies are active (DDP, TP, CP)
- Which mesh axis each strategy uses
- How many devices participate in each parallelism dimension

### Why Do We Need It?

When training on multiple devices, different operations need to know:
1. **How to shard tensors**: Should weights be split across devices (TP) or replicated (DDP)?
2. **How to communicate**: Which devices need to synchronize gradients? Which need to all-gather parameters?
3. **Which axis to use**: In a 2D mesh `[4, 8]`, does axis 0 represent DP groups or TP groups?

`ParallelismContext` provides a single source of truth for these decisions.

### How It Works

```cpp
// ParallelismContext stores:
struct ParallelismContext {
    std::optional<uint32_t> m_ddp_axis;  // Which mesh axis for data parallelism
    std::optional<uint32_t> m_tp_axis;   // Which mesh axis for tensor parallelism
    std::optional<uint32_t> m_cp_axis;   // Which mesh axis for context parallelism
    uint32_t m_num_ddp_devices;          // Number of DDP replicas
    uint32_t m_num_tp_devices;           // Number of TP shards
    uint32_t m_num_cp_devices;           // Number of CP shards
};
```

### Initialization

Initialize the parallelism context **after** opening the device but **before** creating any distributed tensors:

```cpp
// C++
auto& ctx = ttml::autograd::ctx();
ctx.open_device({4, 8});  // 4×8 = 32 devices
ctx.initialize_parallelism_context({
    .enable_ddp = true,   // Data parallelism on axis 0
    .enable_tp = true,    // Tensor parallelism on axis 1
    .enable_cp = false    // No context parallelism
});

// Now you can query:
auto& pctx = ctx.get_parallelism_context();
uint32_t dp_size = pctx.get_ddp_size();  // Returns 4
uint32_t tp_size = pctx.get_tp_size();   // Returns 8
```

```python
# Python
autograd_ctx = ttml.autograd.AutoContext.get_instance()
autograd_ctx.open_device([4, 8])
autograd_ctx.initialize_parallelism_context(
    DistributedConfig(enable_ddp=True, enable_tp=True)
)
```

---

## Parallelism Strategies Explained

### Data Parallelism (DDP)

**Concept**: Replicate the entire model on each device, split the data batch across devices, and synchronize gradients after each backward pass.

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Parallelism (DDP)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Batch = [B0, B1, B2, B3]                                  │
│              ↓                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Device 0    Device 1    Device 2    Device 3       │   │
│   │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐      │   │
│   │  │Model  │   │Model  │   │Model  │   │Model  │      │   │
│   │  │(full) │   │(full) │   │(full) │   │(full) │      │   │
│   │  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘      │   │
│   │      │           │           │           │          │   │
│   │      ↓           ↓           ↓           ↓          │   │
│   │   B0→Loss     B1→Loss     B2→Loss     B3→Loss       │   │
│   │      │           │           │           │          │   │
│   │      └───────────┴─────┬─────┴───────────┘          │   │
│   │                        │                            │   │
│   │                   All-Reduce                        │   │
│   │                   (gradients)                       │   │
│   │                        │                            │   │
│   │                        ↓                            │   │
│   │                 Averaged Gradients                  │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation in TTML**:
```cpp
// After backward pass, synchronize gradients across DDP devices
loss->backward();
ttml::core::distributed::synchronize_gradients(model->parameters());
optimizer.step();
```

**When to Use**:
- Model fits in single device memory
- Want to increase effective batch size
- Simple to implement and debug

**Tradeoffs**:
| Pros | Cons |
|------|------|
| Simple implementation | Full model on each device (memory inefficient) |
| Linear speedup with devices | Communication scales with model size |
| No model code changes needed | Gradient sync can be bottleneck |

---

### Tensor Parallelism (TP)

**Concept**: Shard model parameters across devices. Each device holds a slice of the weights and computes a partial result, then devices communicate to combine results.

```
┌─────────────────────────────────────────────────────────────┐
│                   Tensor Parallelism (TP)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Weight Matrix W [4096 × 4096]                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  W₀        W₁        W₂        W₃                   │   │
│   │ [4096×    [4096×    [4096×    [4096×                │   │
│   │  1024]     1024]     1024]     1024]                │   │
│   └─────────────────────────────────────────────────────┘   │
│              ↓           ↓           ↓           ↓          │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Device 0    Device 1    Device 2    Device 3       │   │
│   │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐      │   │
│   │  │  W₀   │   │  W₁   │   │  W₂   │   │  W₃   │      │   │
│   │  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘      │   │
│   │      │           │           │           │          │   │
│   │      ↓           ↓           ↓           ↓          │   │
│   │   X @ W₀ᵀ     X @ W₁ᵀ     X @ W₂ᵀ     X @ W₃ᵀ       │   │
│   │      │           │           │           │          │   │
│   │      └───────────┴─────┬─────┴───────────┘          │   │
│   │                        │                            │   │
│   │                   All-reduce                        │   │
│   │                   (outputs)                         │   │
│   │                        │                            │   │
│   │                        ↓                            │   │
│   │              Y = [Y₀, Y₁, Y₂, Y₃]                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Two Sharding Strategies**:

1. **Column Parallel** (`ColumnParallelLinear`): Shard output features
   - Each device computes `Y_local = X @ W_local^T` where `W_local` has `out_features/tp_size` rows
   - Output is sharded; optionally `all_gather` to get full output

2. **Row Parallel** (`RowParallelLinear`): Shard input features
   - Input must be sharded (or scatter it first)
   - Each device computes partial output, then `all_reduce` to sum

```cpp
// Column parallel: shard along output dimension
auto weight_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(
    *device,
    /*dim=*/2,      // out_features dimension
    shard_dim       // which mesh axis to shard across
);

// Row parallel: shard along input dimension
auto weight_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(
    *device,
    /*dim=*/3,      // in_features dimension
    shard_dim
);
```

**When to Use**:
- Model too large for single device memory
- Want to reduce memory footprint per device
- Training very large models (billions of parameters)

**Tradeoffs**:
| Pros | Cons |
|------|------|
| Enables training larger models | More complex implementation |
| Lower memory per device | Communication overhead (all-gather, reduce-scatter) |
| Can combine with DDP | Requires model code changes |

---

### Context Parallelism (CP)

**Concept**: Shard the sequence length across devices. Each device processes a chunk of the sequence, and devices communicate during attention computation using **Ring Attention**.

This is essential for training with very long sequences (e.g., 128K+ tokens) that don't fit in single device memory.

```
┌─────────────────────────────────────────────────────────────┐
│                  Context Parallelism (CP)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sequence: [Token₀ ... Token₁₀₂₃ | Token₁₀₂₄ ... Token₂₀₄₇│
│              | Token₂₀₄₈ ... Token₃₀₇₁ | Token₃₀₇₂ ... ]   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Device 0    Device 1    Device 2    Device 3       │   │
│   │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐      │   │
│   │  │Q₀,K₀, │   │Q₁,K₁, │   │Q₂,K₂, │   │Q₃,K₃, │      │   │
│   │  │V₀     │   │V₁     │   │V₂     │   │V₃     │      │   │
│   │  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘      │   │
│   │      │           │           │           │          │   │
│   │      └───────────┴─────┬─────┴───────────┘          │   │
│   │                        │                            │   │
│   │              Ring Attention Algorithm               │   │
│   │      ┌─────────────────────────────────────┐        │   │
│   │      │  Step 0: Compute local attention    │        │   │
│   │      │  Step 1: Ring-shift K,V → compute   │        │   │
│   │      │  Step 2: Ring-shift K,V → compute   │        │   │
│   │      │  Step 3: Ring-shift K,V → compute   │        │   │
│   │      │  Combine with online softmax        │        │   │
│   │      └─────────────────────────────────────┘        │   │
│   │                        │                            │   │
│   │                        ↓                            │   │
│   │              Full Attention Output                  │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Ring Attention Algorithm**:
```
For each device d holding Q_local, K_local, V_local:
    Initialize: O = 0, L = -inf (log-sum-exp accumulator)
    For step in 0..cp_size:
        # Compute attention with current K, V chunk
        local_attn = softmax(Q_local @ K_current^T) @ V_current

        # Online softmax: combine with running accumulator
        O, L = online_softmax_combine(O, L, local_attn)

        # Ring-shift K and V to next device
        K_current, V_current = ring_shift(K_current, V_current)
    Return O
```

**Causal Masking in Ring Attention**:
- Step where source == current device: Apply causal mask (diagonal chunk)
- Step where source < current device: Full attention (earlier tokens)
- Step where source > current device: Skip computation (future tokens masked)

**When to Use**:
- Training with very long sequences (32K, 64K, 128K+ tokens)
- Sequence length exceeds single device memory
- Models with attention mechanisms

**Tradeoffs**:
| Pros | Cons |
|------|------|
| Enables very long sequences | Ring communication overhead |
| Memory scales with 1/cp_size | More complex attention implementation |
| Mathematically equivalent to full attention | Limited to attention-based models |

---

### Pipeline Parallelism (PP)

**Concept**: Shard model layers sequentially across devices (typically across hosts). Each device processes a subset of layers, passing activations to the next device.

```
┌─────────────────────────────────────────────────────────────┐
│                  Pipeline Parallelism (PP)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Model: 24 Transformer Blocks                              │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Host 0       Host 1       Host 2       Host 3      │   │
│   │  (Rank 0)     (Rank 1)     (Rank 2)     (Rank 3)    │   │
│   │  ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐   │   │
│   │  │Embed  │    │Block  │    │Block  │    │Block  │   │   │
│   │  │Block  │    │6-11   │    │12-17  │    │18-23  │   │   │
│   │  │0-5    │    │       │    │       │    │LM Head│   │   │
│   │  └───┬───┘    └───┬───┘    └───┬───┘    └───┬───┘   │   │
│   │      │            │            │            │       │   │
│   │      │   Send     │   Send     │   Send     │       │   │
│   │      │───────────→│───────────→│───────────→│       │   │
│   │      │ activations│ activations│ activations│       │   │
│   │      │            │            │            │       │   │
│   │      │←───────────│←───────────│←───────────│       │   │
│   │      │   Send     │   Send     │   Send     │       │   │
│   │      │ gradients  │ gradients  │ gradients  │       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Forward: Input → Rank 0 → Rank 1 → Rank 2 → Rank 3 → Loss │
│   Backward: Loss → Rank 3 → Rank 2 → Rank 1 → Rank 0        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Configuration**:
```yaml
multihost_config:
  enabled: true
  num_workers: 4
  socket_type: "fabric"
  pipeline_parallel_config:
    num_blocks: 24
    blocks_per_rank:
      0: 6   # Embedding + Blocks 0-5
      1: 6   # Blocks 6-11
      2: 6   # Blocks 12-17
      3: 6   # Blocks 18-23 + LM Head
```

**When to Use**:
- Model too large for single host (multiple hosts needed)
- Want to scale across many hosts
- Can tolerate pipeline bubbles

**Tradeoffs**:
| Pros | Cons |
|------|------|
| Scales across hosts | Pipeline bubbles reduce efficiency |
| Each host has subset of layers | Complex gradient flow |
| Can combine with TP within host | Requires careful load balancing |

---

## Tradeoffs Summary

| Strategy | Memory Efficiency | Communication | Complexity | Best For |
|----------|-------------------|---------------|------------|----------|
| **DDP** | Low (full model replicated) | Gradient all-reduce | Low | Small-medium models, scaling batch size |
| **TP** | High (params sharded) | All-gather, reduce-scatter | Medium | Large models, single-host multi-device |
| **CP** | High (sequence sharded) | Ring communication | Medium | Very long sequences |
| **PP** | High (layers sharded) | Activation/gradient transfer | High | Multi-host, very large models |

### Combining Strategies

Common combinations:
- **TP + DDP** (`mesh_shape: [4, 8]`): 4 DP groups × 8 TP devices
  - Use when model needs TP for memory, but want more parallelism
- **CP + TP** (`mesh_shape: [4, 8]` with CP on axis 0): 4 CP devices × 8 TP devices
  - Use for long sequences with large models
- **PP + TP** (multi-host): Pipeline across hosts, TP within each host
  - Use for very large models across many hosts

### Decision Guide

```
Is your model too large for one device?
├── Yes → Use Tensor Parallelism (TP)
│         └── Still too large? → Add Pipeline Parallelism (PP)
└── No  → Is your sequence too long?
          ├── Yes → Use Context Parallelism (CP)
          └── No  → Use Data Parallelism (DDP)
                    └── Want more throughput? → Combine with TP
```

## Mesh Graph Descriptors (MGD Files)

### What are MGD Files?

Mesh Graph Descriptors (MGD) are `.textproto` files that define the hardware topology and connectivity of your distributed system. They specify:
- Device topology (how devices are arranged in a mesh)
- Host topology (how hosts are connected)
- Inter-device connectivity and channels

### Setting MGD Files via Environment Variable

MGD files are specified using the `TT_MESH_GRAPH_DESC_PATH` environment variable:

```bash
export TT_MESH_GRAPH_DESC_PATH="/path/to/your/mesh_graph_descriptor.textproto"
```

### Where to Find MGD Files

Default MGD files are located in the tt-metal repository:

```
$TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mesh_descriptors/
```

Common examples include:
- `galaxy_1x32_mesh_graph_descriptor.textproto` - For 32-device Galaxy setups (1×32 mesh)
- `t3k_1x8_mesh_graph_descriptor.textproto` - For 8-device T3K setups (1×8 mesh)

You can also find additional MGD files in:
- `$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/` - Standard mesh descriptors
- `$TT_METAL_HOME/tt-train/sources/examples/python/multihost/*/configurations/` - Example configurations for multihost training

### Automatic MGD Selection

If `TT_MESH_GRAPH_DESC_PATH` is not set, the training script will automatically select an MGD file based on the number of devices:
- **8 devices**: Uses `t3k_1x8_mesh_graph_descriptor.textproto`
- **32 devices**: Uses `galaxy_1x32_mesh_graph_descriptor.textproto`

This automatic selection requires `TT_METAL_HOME` to be set.

### MGD File Format

MGD files use Protocol Buffer text format. Here's an example for a 1×32 Galaxy mesh:

```proto
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 32 ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 4 policy: RELAXED }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
```

You can also specify if you want wrap around connections to be visible to ccls like this:

```
device_topology { dims: [ 1, 32 ], dim_types: [RING, LINE] }
```

**Important**: The `device_topology` dimensions in the MGD file define the physical device arrangement, which must be compatible with your logical mesh shape specified in the device config.

## Device Configuration

### Mesh Shape Matching

**Critical**: The mesh device shape specified in your MGD file must match the mesh shape you use when opening the device. The mesh shape in your training config's `device_config` determines how devices are logically organized.

For example, if your MGD defines a `device_topology { dims: [ 4, 8 ] }` (32 devices total), your `mesh_shape` should be compatible with this arrangement, such as:
- `[4, 8]` - 4 DP groups × 8 TP devices is fine
- `[1, 32]` - Single group with 32 TP devices will hang or crash with an error at open_device
- `[32, 1]` - 32 DP groups × 1 device each will also hang or crash with an error at open_device

### Device Config in Training YAML

In your training configuration YAML file, specify the device configuration:

```yaml
device_config:
  enable_tp: true      # Enable tensor parallelism
  enable_ddp: true     # Enable data parallelism
  mesh_shape: [4, 8]   # 4 DP groups × 8 TP devices = 32 devices, should be the same as in MGD file
  device_ids: []       # Optional: specific device IDs to use
```

### Parallelism Strategy Selection

In the device config, you can specify which parallelism strategies to use:

- **`enable_tp`**: Enable tensor parallelism (shard model parameters)
- **`enable_ddp`**: Enable data parallelism (replicate model, shard data)
- **`enable_pp`**: Enable pipeline parallelism (shard layers sequentially)
- **`enable_cp`**: Enable context parallelism (shard input along sequence dimension)

### Mesh Shape Determination

In the NanoGPT training script (and other training scripts), the mesh shape is automatically determined from the `device_config` section of your YAML configuration:

```cpp
// From device_config in YAML
DeviceConfig config = parse_device_config(yaml_config);
MeshShape mesh_shape = config.mesh_shape;  // e.g., [4, 8]

// Open device with this mesh shape
initialize_device(mesh_shape, device_ids);
```

The mesh shape must match the total number of devices available and be compatible with your MGD file's device topology.

## Parallelism Axis Assignment

### Automatic Axis Determination

The parallelism axes are **automatically determined** based on the order you enable parallelism strategies in your device config:

1. **Data Parallel (DP) axis**: Always assigned to **axis 0** (first dimension) if `enable_ddp` is true
2. **Context Parallel (CP) axis**: Assigned to **axis 1** (second dimension) if `enable_cp` is true
2. **Tensor Parallel (TP) axis**: Assigned to **axis 2** (third dimension) if `enable_tp` is true

Please keep an mind mesh is not more than 2 dimensional, so maximum two of those parallelism strategies can be enabled at the same time and they will take axis according to the arrangement above.

This means:
- For `mesh_shape: [4, 8]` with both DDP and TP enabled:
  - DP uses axis 0 → 4 DP groups
  - TP uses axis 1 → 8 TP devices per group
- For `mesh_shape: [1, 32]` with only TP enabled:
  - TP uses axis 1 → 32 TP devices
- For `mesh_shape: [32, 1]` with only DDP enabled:
  - DP uses axis 0 → 32 DP groups
- For `mesh_shape: [8, 4]` with both CP and DDP enabled:
  - DP uses axis 0 → 8 DP groups
  - CP uses axis 1 → 4 CP groups

### Implementation Details

The axis assignment happens in `ParallelismContext`:

```cpp
uint32_t axis = 0;
if (config.enable_ddp && mesh_shape[axis] > 1U) {
    m_ddp_axis = axis++;
    m_num_ddp_devices = mesh_shape[m_ddp_axis.value()];
}
if (config.enable_cp && mesh_shape[axis] > 1U) {
    m_cp_axis = axis++;
    m_num_cp_devices = mesh_shape[m_cp_axis.value()];
}
if (config.enable_tp && mesh_shape[axis] > 1U) {
    m_tp_axis = axis++;
    m_num_tp_devices = mesh_shape[m_tp_axis.value()];
}
```

**Key constraint**: The number of enabled parallelism strategies must equal the number of mesh shape dimensions:
- 1D mesh `[N]`: Can use either DP or TP (but not both)
- 2D mesh `[M, N]`: Can use both DP and TP together

## Complete Example

Here's a complete example for training Llama 8B with TP + DDP on a 32-device Galaxy:

### 1. Training Configuration (`training_llama8b_tp_ddp_galaxy.yaml`)

```yaml
training_config:
  project_name: "tt_train_llama8b_tp_dp"
  batch_size: 4  # Total batch size across all DP groups
  model_config: "configs/model_configs/llama8b_galaxy_tp_dp.yaml"

device_config:
  enable_tp: true
  enable_ddp: true
  mesh_shape: [4, 8]  # 4 DP groups × 8 TP devices = 32 devices
```

### 2. Set MGD File

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto
```

Or let it auto-select (for 32 devices):
```bash
export TT_METAL_HOME=/path/to/tt-metal
# TT_MESH_GRAPH_DESC_PATH will be auto-set to galaxy_1x32_mesh_graph_descriptor.textproto
```

### 3. Run Training

```bash
./build/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/training_llama8b_tp_ddp_galaxy.yaml
```

### 4. What Happens

1. The script reads `mesh_shape: [4, 8]` from `device_config`
2. It opens a device with mesh shape `[4, 8]` (32 devices total)
3. The MGD file must define a compatible device topology (e.g., `device_topology { dims: [ 4, 8 ] }`)
4. Parallelism context is initialized:
   - DP axis = 0 → 4 DP groups
   - TP axis = 1 → 8 TP devices per group
5. Model parameters are sharded across TP devices
6. Data batches are distributed across DP groups

## Troubleshooting

### MGD Shape Mismatch

**Error**: Device initialization fails or mesh shape doesn't match

**Solution**: Ensure the `device_topology` in your MGD file is compatible with your `mesh_shape`. For a `mesh_shape: [4, 8]`, your MGD should define a topology that can accommodate 32 devices arranged as 4×8.

### Parallelism Axis Mismatch

**Error**: "The number of parallelization axes must be equal to the number of mesh shape dimensions"

**Solution**:
- For 1D mesh `[N]`: Enable only one of `enable_ddp` or `enable_tp`
- For 2D mesh `[M, N]`: You can enable both `enable_ddp` and `enable_tp`

### MGD File Not Found

**Error**: MGD file path not found or `TT_METAL_HOME` not set

**Solution**:
- Set `TT_METAL_HOME` to your tt-metal repository root
- Explicitly set `TT_MESH_GRAPH_DESC_PATH` to your MGD file path
- Or ensure you're using 8 or 32 devices for automatic selection

### ParallelismContext Not Initialized

**Error**: `"ParallelismContext is not initialized."`

**Cause**: Code is trying to access the parallelism context (e.g., calling `get_parallelism_context()`, `get_tp_size()`, `get_ddp_size()`) before it has been initialized.

**Solution**:
1. Ensure you call `initialize_parallelism_context()` **after** opening the device but **before** using any distributed features:
   ```cpp
   // Correct order:
   initialize_device(mesh_shape, device_ids);
   ctx.initialize_parallelism_context({.enable_ddp = true, .enable_tp = true});
   // Now you can use get_parallelism_context()
   ```

2. In training scripts, the parallelism context should be initialized right after device initialization:
   ```cpp
   auto *device = &ttml::autograd::ctx().get_device();
   ttml::autograd::ctx().initialize_parallelism_context(
       {.enable_ddp = device_config.enable_ddp, .enable_tp = device_config.enable_tp});
   ```

3. If using distributed models (e.g., `DistributedLlama`), ensure the parallelism context is initialized before model construction, as the model may query parallelism settings during initialization.

### ParallelismContext Already Initialized

**Error**: `"ParallelismContext is already initialized."`

**Cause**: `initialize_parallelism_context()` is being called more than once. The parallelism context can only be initialized once per training session.

**Solution**:
1. **Check for duplicate initialization**: Ensure `initialize_parallelism_context()` is called exactly once in your code flow. Common causes:
   - Calling it in both initialization and model construction
   - Calling it in a loop or multiple code paths
   - Calling it in both C++ and Python if using mixed code

2. **Typical pattern**: Initialize once after device opening:
   ```cpp
   // Open device
   initialize_device(mesh_shape, device_ids);

   // Initialize parallelism context ONCE
   ttml::autograd::ctx().initialize_parallelism_context(config);

   // Use context throughout training
   // ... rest of training code ...
   ```

4. **In Python**: Same principle applies:
   ```python
   autograd_ctx = ttml.autograd.AutoContext.get_instance()
   autograd_ctx.open_device([mesh_rows, mesh_cols])

   # Initialize ONCE
   autograd_ctx.initialize_parallelism_context(
       DistributedConfig(enable_ddp=True, enable_tp=True)
   )
   ```

### Physical Chip ID Not Found / Topology Not Configured

**Error**:
```
TT_FATAL: Physical chip id X not found in control plane chip mapping.
You are calling for a chip outside of the fabric cluster.
Check that your mesh graph descriptor specifies the correct topology
```

**Possible cause**: The hardware topology has not been configured on the system. This error occurs when the fabric control plane cannot find the physical chips specified in your MGD file or mesh configuration.

**Solution**:
1. **Install and configure topology**:
   ```bash
   pip install tt-topology
   tt-topology -l mesh
   ```
   This configures the hardware topology to match your mesh configuration.

2. **Verify topology matches your mesh**: Ensure the topology configuration matches the number of chips you're trying to use. For example:
   - If using 8 chips with `mesh_shape: [1, 8]`, ensure all 8 chips are configured
   - If using 32 chips with `mesh_shape: [4, 8]`, ensure all 32 chips are configured

3. **Check device configuration**: Some device configurations (e.g., n300) may default to using fewer chips than available. If you want to use all chips:
   - Check your device config YAML file
   - Update `mesh_shape` to match the number of chips you want to use
   - Example: Change `mesh_shape: [1, 2]` to `mesh_shape: [1, 8]` to use all 8 chips on an n300 configuration

**Important Notes**:
- **Graceful shutdown**: When terminating a training process (e.g., with Ctrl+C), the fabric may not exit gracefully. If you encounter errors on subsequent runs, reset the cards:
  ```bash
  tt-smi -r
  ```
  or
  ```bash
  tt-smi -glx_reset
  ```
  on a galaxy

- **Topology persistence**: The topology configuration should persist across reboots, but may need to be reconfigured if hardware changes or after system updates.

## Python Distributed Infrastructure (Rules, Dispatch, ParallelStyle)

This section describes the **rule-based layout and dispatch layer** used for Python distributed training (e.g. Llama TP). You use it to shard models by name pattern, and extend it with custom **op rules**, **module rules**, and **parallel styles**.

### Overview

1. **Layout**: Describes how a tensor is placed across the mesh (`Shard(dim)` or `Replicate()` per mesh axis). Built with `Layout(ndim=..., axis_placements={axis: Shard(dim), ...})`.
2. **Dispatch**: When you call a patched op (e.g. `ttml.ops.linear.linear`), the call goes through `dispatch()`. Dispatch reads layouts from tensor metadata, looks up a **sharding rule** by op name, gets a `ShardingPlan`, redistributes inputs as needed, runs optional **pre-collectives** (e.g. broadcast), calls the raw op, runs optional **post-collectives** (e.g. all_reduce, all_gather), and stamps the output layout.
3. **Parallelize**: `parallelize_module(model, mesh_device, parallelize_plan, tp_axis=..., cp_axis=...)` walks the module tree, matches module names to the plan (exact or regex), and applies a **ParallelStyle** (e.g. `ColwiseParallel`, `RowwiseParallel`) or a **module rule** for composite modules (e.g. GQA).
4. **Op rules** and **module rules** are registered with decorators; the same op name used in the rule is used when patching `ttml.ops.*` so dispatch finds the rule.

### How to Use the Infrastructure

1. **Activate dispatch** (once, before creating the model):

   ```python
   import ttml.distributed
   from ttml.distributed import init_ops, parallelize_module, ColwiseParallel, RowwiseParallel

   init_ops()  # Patch ttml.ops.* so linear, matmul, etc. go through dispatch
   ```

2. **Define a parallelize plan**: map module name patterns to a style. Use exact names or regex; regex is matched with `re.fullmatch` against the module path (e.g. `"layers.0.attention.q_linear"`).

   ```python
   LLAMA_TP_PLAN = {
       r".*\.(q_linear|kv_linear|w1|w3)": ColwiseParallel(),
       r".*\.(out_linear|w2)": RowwiseParallel(),
       "fc": ColwiseParallel(gather_output=True),  # LM head
   }
   ```

3. **Parallelize the model** after construction:

   ```python
   mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
   model = parallelize_module(model, mesh_device, LLAMA_TP_PLAN, tp_axis=1, cp_axis=0)
   ```

4. **Forward/backward**: Patched ops run through dispatch; rules decide input/output layouts and collectives. No layout code in the model itself.

See `tt-train/sources/examples/llama_tp/train_llama_tp.py` for a full example.

### Column vs row linears, broadcast / all_reduce / all_gather, and gradients

**Megatron pairing (expected pattern)**
In a TP MLP (or FFN), you almost always alternate:

- **`ColwiseParallel`**: weights sharded on **output** features; forward wrapper **`broadcast`**s activations on `tp_axis` so each rank sees the full input; linear output is **sharded** on the last dim.
- **`RowwiseParallel`**: weights sharded on **input** features; expects that **sharded** activation; forward wrapper **`all_reduce`**s the partial sums so the block output is **replicated** again.


**LM head / full replicated output**
For the last linear (e.g. logits over vocab), use `ColwiseParallel(gather_output=True)`. The style wraps forward with **`all_gather`** on the output dim and uses **`GradOutputType.REPLICATED`** on that gather’s backward so that **replicated loss gradients** (same on each TP rank) are handled correctly—analogous to C++ column-parallel with `gather_output=true`. See `style.py` (`ColwiseParallel._wrapped_forward`) and `rules/matmul.py` / custom rules using `AllGather(..., gather_grad_replicated=True)`.

**Where this appears in dispatch**
Op rules return `ShardingPlan`s with optional **`Broadcast`** (pre), **`AllReduce`**, **`AllGather`** (post). `parallelize_module` + `ColwiseParallel`/`RowwiseParallel` attach the collectives by wrapping modules; dispatch still stamps layouts for `linear`/`matmul`.

### Layout

- **Construction**: `Layout(ndim=N)` (all Replicate) or `Layout(ndim=N, axis_placements={mesh_axis: Shard(dim), ...})`. Unspecified axes are Replicate.
- **Placements**: `layout.placements` is a tuple of `Shard(dim)` or `Replicate()` per mesh dimension.
- **Helpers**: `layout.with_placement(mesh_axis, placement)` returns a new Layout with that axis updated. `layout.ndim`, `layout.is_replicated()`, `layout.is_sharded_on(axis)`.

### Op Rules (What They Are For)

An **op rule** is a function that, given the **layouts of the tensor arguments** to an op, returns a **ShardingPlan**: required input layouts, output layout, and optional **pre-** and **post-collectives** per input/output. Dispatch uses this to:

- Redistribute inputs to the required layouts (all_gather/scatter as needed),
- Run pre-collectives (e.g. `Broadcast(mesh_axis)` on an input),
- Call the raw C++ op,
- Run post-collectives (e.g. `AllReduce(mesh_axis)`, `AllGather(dim, mesh_axis)`),
- Attach the output layout to the result.

Rules are **per op name** (e.g. `"linear"`, `"matmul"`). The op name in `@register_rule("op_name")` must match the name you use with `register_op("op_name", raw_callable)` when wrapping your op. You do **not** patch ttml.ops; built-in ops are patched by the lib, and your ops use your own names and the wrapper from `register_op`.

### How to Create a Custom Op Rule (all in your code, no lib changes)

1. **Implement the rule** (e.g. in your training script or a plugin module). The rule receives layouts as positional args (one per tensor input), then `runtime=None` and `**kwargs`. Return a `ShardingPlan`. Use the same op name as in step 3.

   ```python
   from ttml.distributed import register_rule, register_op
   from ttml.distributed.rules.registry import ShardingPlan

   @register_rule("my_custom_op")
   def my_custom_op_rule(
       input_layout,
       *extra_layouts,
       runtime=None,
       **kwargs,
   ):
       return ShardingPlan(
           input_layouts=[input_layout],
           output_layout=input_layout,
       )
   ```

2. **CCL types** (from `ttml.distributed.rules.registry`):
   - **Pre-collectives**: `Broadcast(mesh_axis)` — broadcast tensor on that axis (e.g. align replicated activations with column-sharded weights).
   - **Post-collectives**: `AllReduce(mesh_axis, noop_backward=False)` — sum partial results (typical row-parallel forward); backward behavior depends on `noop_backward` when the input was already sharded.
   - **Post-collectives**: `AllGather(dim, mesh_axis, gather_grad_replicated=False)` — gather shards along `dim`; set **`gather_grad_replicated=True`** when upstream loss provides **replicated** gradients w.r.t. the gathered tensor (same as `GradOutputType.REPLICATED` on the C++ `all_gather` path—required for correct LM-head-style training).

3. **Register your op under the same name**: Wrap your raw callable with `register_op(op_name, raw_callable)`. Use the returned wrapper as your op entry point (no patching of ttml.ops).

   ```python
   def my_raw_op(*args, **kwargs):
       # Your implementation (e.g. ttnn call or Python logic).
       return ...

   my_custom_op = register_op("my_custom_op", my_raw_op)
   # From here on, call my_custom_op(...) so it goes through dispatch.
   ```

### Module Rules (What They Are For)

A **module rule** applies to a **composite module type** (e.g. `GroupedQueryAttention`). When `parallelize_module` walks the tree and finds a module whose type has a registered module rule, it:

1. Calls the **module rule** with `(module, mesh_device, tp_axis, cp_axis)` — axes are passed through from `parallelize_module`, no policy or prefix.
2. **Recurses into children** so that submodules (e.g. `q_linear`, `kv_linear`, `out_linear`) are visited with their full path and get the matching **ParallelStyle** from the plan. Those children then get weight sharding and forward collectives (broadcast, all_reduce) from `style._apply()`.

So the rule should **not** call `distribute_linear` on sub-linears that are matched by the plan; recursion will apply the style to them (weight + collectives). The rule only handles composite-specific logic (e.g. GQA: `num_heads`/`num_groups` for TP using `tp_axis`, and when `cp_axis` is set: rope_params and ring_sdpa swap).

### How to Create a Custom Module Rule

1. **Register the rule** for your module type with `@register_module_rule(MyModuleClass)`.

2. **Implement the rule** with signature `(module, mesh_device, tp_axis, cp_axis=None)`. You receive `tp_axis` and `cp_axis` from `parallelize_module`; no policy or prefix. Do **composite-only** work: e.g. get `tp_size = mesh_device.shape[tp_axis]` and adjust `module.num_heads` / `module.num_groups` for TP; when `cp_axis` is set, rebuild `rope_params` and swap to `ring_attention_sdpa`. Do **not** call `distribute_linear` on sub-linears—after the rule returns, `parallelize_module` recurses into children and applies the plan’s styles to them (weight sharding + broadcast/all_reduce).

   ```python
   from ttml.distributed.rules.registry import register_module_rule

   @register_module_rule(MyCompositeModule)
   def distribute_my_composite(module, mesh_device, tp_axis, cp_axis=None):
       # Composite-only: adjust head/group counts for TP; optional CP (rope, ring_sdpa).
       mesh_shape = mesh_device.shape
       tp_size = mesh_shape[tp_axis]
       if tp_size > 1:
           module.num_heads = module.num_heads // tp_size
           module.num_groups = module.num_groups // tp_size
       if cp_axis is not None:
           # Rebuild rope_params, swap to ring_attention_sdpa, etc.
           ...
       return module
   ```

3. **Ensure the rule is loaded**: `ttml.distributed` imports `module_rules`, so any `@register_module_rule` in that package runs at import time. For a new rule in a new file, add an import in `ttml/distributed/__init__.py` or in `module_rules.py`.

4. **Use the plan**: In the parallelize plan, use name patterns that match the **submodule** paths (e.g. `r".*\.q_linear"`, `r".*\.out_linear"`). `parallelize_module` calls your rule then recurses so those children receive the matching styles.

### Custom ParallelStyle

To define a new style (e.g. a variant of column/row parallel):

1. Subclass `ParallelStyle` and implement `_apply(module, mesh_device, tp_axis)` (mutate the module’s parameters with `distribute_tensor` and the layout you want).
2. Implement `get_layout(mesh_device, tp_axis) -> Layout` when callers need the style’s weight layout without applying it to a module (e.g. tests, tooling). `parallelize_module` does **not** use this to detect composites; detection is **`get_module_rule(type(module))`** only.
3. Add your style to the parallelize plan by name or regex, e.g. `{"my_layer": MyCustomStyle()}`.

### Summary

| Concept | Purpose |
|--------|---------|
| **Layout** | Describes sharding/replication per mesh axis; used for redistribution and plan output. |
| **Op rule** | Maps op name + input layouts → ShardingPlan (layouts + optional pre/post collectives). |
| **Module rule** | For composite modules: called with `(module, mesh_device, tp_axis, cp_axis)`; handles TP/CP logic (e.g. head counts, rope, ring_sdpa); children get styles on recursion. |
| **ParallelStyle** | Assignable by name pattern; applies to leaf modules (e.g. LinearLayer). Optional `get_layout` for introspection / tests. |
| **parallelize_module** | Walks the model, applies styles by pattern and module rules by type; no layout code in user model. |

## Additional Resources

- **MGD Format Documentation**: `$TT_METAL_HOME/tt_metal/fabric/MGD_README.md`
- **Multihost Training Examples**: `$TT_METAL_HOME/tt-train/sources/examples/python/multihost/`
- **Device Config Examples**: `$TT_METAL_HOME/tt-train/configs/training_configs/`
- **TP+DP Example (Python)**: `$TT_METAL_HOME/tt-train/sources/examples/linear_regression_tp_dp/linear_regression_tp_dp.py` - Complete example demonstrating combined Tensor Parallelism and Data Parallelism with proper mesh configuration, parallelism context initialization, and distributed tensor mapping
- **Llama TP (Python, rules/dispatch)**: `$TT_METAL_HOME/tt-train/sources/examples/llama_tp/train_llama_tp.py` - Full example using `parallelize_module`, `ColwiseParallel`/`RowwiseParallel`, and the rule-based dispatch layer
