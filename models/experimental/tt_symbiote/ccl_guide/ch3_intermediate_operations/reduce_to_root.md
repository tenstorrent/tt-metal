# 3.3 ReduceToRoot and MeshPartition

---

## Part A: ReduceToRoot

### Concept

ReduceToRoot answers: *every device holds partial intermediate tensors — how does one designated "root" device accumulate the fully reduced result, while all other devices' outputs are discarded?*

After ReduceToRoot, only the device identified by `root_coord` holds valid output tensors. All non-root devices' output tensors are allocated but contain undefined values. This differs from AllReduce (which gives the same result to every device) and ReduceScatter (which gives each device one reduced shard).

ReduceToRoot is specialized for **Scaled Dot Product Attention (SDPA) tree reduction**. It does not reduce a single generic tensor; it reduces three tensors simultaneously that together represent the flash-attention running statistics:

```
input_tensor_l   —  values tensor (L component of flash-attention output)
input_tensor_s   —  sum-of-softmax-exp tensor (S / denominator)
input_tensor_m   —  running max tensor (M / max-so-far for numerical stability)
```

All three must be reduced together because the numerically stable recombination formula requires all three components at each step.

```
Before ReduceToRoot   (4 devices, root_coord = (1, 0))

  Device (0,0): [L0, S0, M0]   ← partial SDPA outputs from attention head shard 0
  Device (0,1): [L1, S1, M1]
  Device (1,0): [L2, S2, M2]   ← root
  Device (1,1): [L3, S3, M3]

After ReduceToRoot

  Device (0,0): [undefined, undefined, undefined]
  Device (0,1): [undefined, undefined, undefined]
  Device (1,0): [L_combined, S_combined, M_combined]   ← only root holds result
  Device (1,1): [undefined, undefined, undefined]
```

The combination formula at each reduction step is the standard flash-attention merge:
```
M_new = max(M_a, M_b)
S_new = exp(M_a - M_new) * S_a + exp(M_b - M_new) * S_b
L_new = exp(M_a - M_new) * L_a + exp(M_b - M_new) * L_b
```

The `scale_fp32` parameter scales the values before reduction, used to apply the attention scale factor `(1 / sqrt(d_k))` as part of the combine.

### When to use ReduceToRoot

| Situation | Why ReduceToRoot |
|-----------|-----------------|
| Context-parallel SDPA: heads sharded across devices, final output needed only on host | Avoid AllReduce overhead when only the root (host-connected) device needs the result |
| Pipeline-parallel models: loss computation on one device before backward pass | Aggregate partial losses from all pipeline stages to the root for logging |
| Tree-parallel attention across 4 devices | Specialized path that merges the three flash-attention tensors together rather than requiring three separate AllReduce calls |

> **Gotcha:** ReduceToRoot is currently optimized for a 4-device configuration (as stated in the nanobind docstring: "Performs sdpa tree reduction across 4 devices"). The `root_coord` is expected to be `(1, 0)` in the typical 4-device layout. The operation may function with other configurations but is tested in this specific setup.

---

### Python API

Source: `ttnn/cpp/ttnn/operations/ccl/reduce_to_root/reduce_to_root.hpp`

```python
output_tensor_l, output_tensor_s, output_tensor_m = ttnn.reduce_to_root(
    input_tensor_l,                  # ttnn.Tensor — values (L) per device
    input_tensor_s,                  # ttnn.Tensor — sum-of-exp (S) per device
    input_tensor_m,                  # ttnn.Tensor — running max (M) per device
    root_coord,                      # ttnn.MeshCoordinate — which device receives the result
    scale_fp32=1.0,                  # float — attention scale applied during reduction
    output_tensor_l=None,            # Optional[ttnn.Tensor] — preallocated output for L
    output_tensor_s=None,            # Optional[ttnn.Tensor] — preallocated output for S
    output_tensor_m=None,            # Optional[ttnn.Tensor] — preallocated output for M
    intermediate_tensor=None,        # Optional[ttnn.Tensor] — preallocated intermediate buffer
    input_mux_cores=None,            # Optional[List[ttnn.CoreCoord]] — input mux core assignments
    topology=ttnn.Topology.Linear,   # ttnn.Topology — topology; default Linear
)
# Returns: Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
#   [0] output_tensor_l: reduced L, valid only on root device
#   [1] output_tensor_s: reduced S, valid only on root device
#   [2] output_tensor_m: reduced M, valid only on root device
```

**Return value note:** The C++ return type is `std::vector<ttnn::Tensor>`. Python receives a list that is unpacked by the three-variable assignment shown above.

### Parameter notes

**`root_coord`** (required): A `ttnn.MeshCoordinate` identifying the device that receives the combined result. For a 4-device 2×2 mesh, `root_coord = ttnn.MeshCoordinate((1, 0))` is the expected configuration per the docstring.

> **Gotcha:** `root_coord` is a `ttnn.MeshCoordinate`, not a plain tuple. Passing `(1, 0)` raises a type error. Construct it explicitly: `ttnn.MeshCoordinate((1, 0))`.

**`scale_fp32`** (default `1.0`): Scaling factor applied to the L tensor during the reduction combine. Set to `1.0 / math.sqrt(head_dim)` to incorporate the attention scale directly in the reduction step.

**`topology`** (default `ttnn.Topology.Linear`): Unlike AllGather and AllReduce which default to `None`, ReduceToRoot defaults to `Linear`. This is consistent with the typical 4-device pipeline where no wrap-around cable exists. Set to `ttnn.Topology.Ring` if ring cabling is available.

**`input_mux_cores`**: Advanced parameter. When provided, specifies which Tensix cores act as input multiplexers for the reduction tree. Leave as `None` for standard usage; the program factory computes the assignment automatically.

**`intermediate_tensor`**: Pre-allocate the intermediate buffer used during the reduction tree traversal. Leave as `None` to let the runtime allocate. Pre-allocating eliminates allocation overhead in repeated-inference loops.

---

### Illustrative Example

```python
import math
import torch
import ttnn

# 2×2 mesh (4 devices)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))

head_dim = 128
scale = 1.0 / math.sqrt(head_dim)

# Each device produces partial SDPA outputs
# input_tensor_l shape: [num_heads, seq_len]    — values
# input_tensor_s shape: [num_heads, head_dim/4] — sum-of-exp
# input_tensor_m shape: [num_heads, head_dim/4] — running max
# (exact shapes depend on model configuration; use ttnn.from_torch to place on mesh)

l_data = torch.zeros((8, 128), dtype=torch.bfloat16)
s_data = torch.zeros((8, 32),  dtype=torch.bfloat16)
m_data = torch.zeros((8, 32),  dtype=torch.bfloat16)

input_l = ttnn.from_torch(l_data, device=mesh,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
input_s = ttnn.from_torch(s_data, device=mesh,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
input_m = ttnn.from_torch(m_data, device=mesh,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                           dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

root = ttnn.MeshCoordinate((1, 0))

output_l, output_s, output_m = ttnn.reduce_to_root(
    input_l, input_s, input_m,
    root_coord=root,
    scale_fp32=scale,
    topology=ttnn.Topology.Linear,
)
# output_l, output_s, output_m are valid ONLY on device (1, 0)
# Other devices' outputs contain undefined values
```

---

### Under the Hood

### Kernel structure

The `ReduceToRoot` program factory builds a three-stage pipeline:

1. **Send kernels** (non-root devices): `send_unary_reader_kernel_id` reads the three input tensors from each non-root device's L1 and forwards them toward the root via ERISC. The `forward_coord` and `backward_coord` optional mesh coordinates in `create_at` determine the reduction tree path — two devices send to the root, which then combines four partial inputs.

2. **Root stage 1** (`root1_reader_kernel_id`, `root1_writer_kernel_id`): the root receives from its two immediate neighbors, performs the flash-attention merge formula using `compute_kernel_id`, and stores the intermediate result.

3. **Root stage 2** (`root2_reader_kernel_id`, `root2_writer_kernel_id`): the root combines the two intermediate results into the final output tensors.

The `semaphores` vector in `shared_variables_t` synchronizes these stages. The `input_mux_cores` parameter controls which Tensix cores handle the multiplexing between root stage 1 and root stage 2.

The `reduce_to_root_program_factory()` free function (declared in `reduce_to_root_op.hpp`) is the entry point for per-device program creation, called with `forward_coord` and `backward_coord` computed from the mesh topology.

### Output validity

Only the root device's output tensors contain valid data. The C++ return type `std::array<std::vector<ttnn::Tensor>, 2>` contains two vectors:
- Index 0: output tensors for the root device
- Index 1: empty (zero-length) tensors for non-root devices

At the Python level, the three-way unpack receives the root's outputs. Reading the returned tensors on non-root devices produces undefined values — they are allocated but not written.

---

### Common Errors (ReduceToRoot)

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError` on `root_coord` | Passing plain tuple `(1, 0)` instead of `ttnn.MeshCoordinate((1, 0))` | Wrap: `ttnn.MeshCoordinate((1, 0))` |
| Wrong values in output | Reading output on a non-root device | Only access output on the device matching `root_coord` |
| Shape mismatch between L, S, M tensors | Tensors were not produced by the same attention head computation | Verify all three tensors share the same batch/head dimensions |
| Incorrect reduction with `scale_fp32 != 1.0` | Scale applied twice (once in attention, once in reduce_to_root) | Use `scale_fp32=1.0` if attention already scaled, or `1/sqrt(d_k)` if scaling in the reduction |

---

## Part B: MeshPartition

### Concept

MeshPartition answers: *a single replicated tensor is on all devices — how does each device extract the slice that belongs to it based on its position in the mesh?*

MeshPartition is a **local** operation: no data crosses device boundaries. Each device independently computes which slice of the input tensor it owns — based on `dim`, `cluster_axis`, and its own mesh coordinate — and returns that slice.

```
Before MeshPartition   (4 devices, input shape [1, 1, 32, 256], dim=3, cluster_axis=1)

  Device (0,0): [1, 1, 32, 256]  ← full tensor, replicated
  Device (0,1): [1, 1, 32, 256]
  Device (1,0): [1, 1, 32, 256]
  Device (1,1): [1, 1, 32, 256]

After MeshPartition   (each device keeps its slice)

  Device (0,0): [1, 1, 32, 64]   ← columns   0..63   (column index 0 of 4)
  Device (0,1): [1, 1, 32, 64]   ← columns  64..127  (column index 1 of 4)
  Device (1,0): [1, 1, 32, 64]   ← columns 128..191  (column index 2 of 4)
  Device (1,1): [1, 1, 32, 64]   ← columns 192..255  (column index 3 of 4)
```

MeshPartition is conceptually the inverse of AllGather: AllGather assembles shards into a full tensor, MeshPartition disassembles a full tensor into shards. Unlike AllGather, MeshPartition requires no cross-device communication.

### When to use MeshPartition

| Situation | Why MeshPartition |
|-----------|------------------|
| Distribute a freshly loaded weight matrix to each device's tensor-parallel shard | Efficient local slice without any network traffic |
| After AllBroadcast: each device received all N tensors, now each needs just its own segment | Slice by mesh position instead of a fixed index |
| KV-cache initialization: each device initializes only its head-shard of the cache | Each device independently computes which cache rows it owns |
| Undo a previous AllGather to return to a sharded state without reloading from host | The inverse of `ttnn.all_gather` for on-device resharding |

MeshPartition is preferred over manual slicing (e.g., `ttnn.slice`) when the slice boundaries are determined by mesh position, because it encapsulates the mesh-coordinate calculation and avoids hard-coded index arithmetic.

---

### Python API

Source: `ttnn/cpp/ttnn/operations/ccl/mesh_partition/mesh_partition.hpp`

```python
output_tensor = ttnn.mesh_partition(
    input_tensor,                    # ttnn.Tensor — full tensor, replicated on all devices
    dim,                             # int — dimension to partition along
    cluster_axis,                    # Optional[int] — mesh axis; determines partition count and device index
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout; None = match input
)
# Returns: ttnn.Tensor
#   Shape: input_shape with input_shape[dim] / N on dim, where N = number of devices along cluster_axis
#   Each device receives the slice corresponding to its position along cluster_axis
```

### Parameter notes

**`dim`** (required, positional): The dimension to partition. The input must be divisible by N (the number of devices along `cluster_axis`) on this dimension. The output shape on `dim` is `input_shape[dim] / N`.

**`cluster_axis`** (required, positional): The mesh axis that determines how many partitions to create and which partition each device owns. `cluster_axis=0` partitions across rows; `cluster_axis=1` partitions across columns.

> **Gotcha:** Unlike most CCL operations where `cluster_axis` is `Optional`, `mesh_partition`'s `cluster_axis` parameter is typed `Optional[int]` in the header but the `detail::get_cluster_axis_size()` helper is always called with it. On a 1-D mesh (`MeshShape(1, N)`), pass `cluster_axis=1` to partition across the N devices.

**`memory_config`**: The output inherits the input's memory config when `None`. Specify explicitly when the downstream op requires a specific layout.

---

### Illustrative Examples

### Basic MeshPartition on a 1-D mesh

```python
import torch
import ttnn

# 1×4 mesh
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

# Full weight matrix replicated on all devices (e.g., just loaded from host)
weight = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
weight_distributed = ttnn.from_torch(
    weight,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)

# Each device keeps its own 1/4 slice (no network traffic)
weight_shard = ttnn.mesh_partition(
    weight_distributed,
    dim=3,           # partition hidden dimension
    cluster_axis=1,  # partition across 4 columns
)
# Device (0,0): weight[:, :, :, 0:64]
# Device (0,1): weight[:, :, :, 64:128]
# Device (0,2): weight[:, :, :, 128:192]
# Device (0,3): weight[:, :, :, 192:256]
```

### MeshPartition as the inverse of AllGather

```python
# AllGather assembled shards into a full tensor; mesh_partition reassembles the shard
# (Illustrates round-trip; in practice you would not AllGather just to repartition)

full_tensor = ttnn.all_gather(shard, dim=3, topology=ttnn.Topology.Ring)
# full_tensor.shape == [1, 1, 32, 256] on every device

restored_shard = ttnn.mesh_partition(full_tensor, dim=3, cluster_axis=1)
# restored_shard.shape == [1, 1, 32, 64] — back to original per-device shape
```

### 2-D mesh partition along one axis

```python
# 2×4 mesh; partition along columns (axis=1 = 4-way partition within each row)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Partition hidden dim across 4 column devices; row axis is unaffected
weight_shard = ttnn.mesh_partition(
    weight_distributed,
    dim=3,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Each device in a row gets 1/4 of the hidden dimension
# Devices in different rows get the same slice (they share column index)
```

---

### Under the Hood

### Program structure

MeshPartition's device operation (`MeshPartitionDeviceOperation`) is implemented using `SliceDeviceOperation` internally. The `shared_variables_t` in `MeshPartition::shared_variables_t` wraps a `SliceSharedVariables` variant that holds the compiled Slice program's state.

At program creation (`create_at`), the operation:
1. Calls `detail::get_cluster_axis_size(input_tensor, cluster_axis)` to determine N — the number of devices along the axis.
2. Determines this device's index within the axis from its `MeshCoordinate`.
3. Computes the slice start/end: `[dim_index * (input_shape[dim] / N), (dim_index + 1) * (input_shape[dim] / N)]`.
4. Dispatches to the appropriate `SliceProgramFactory` variant based on tensor layout and shape.

Because MeshPartition is purely local (no ERISC or EDM involvement), it completes in compute time only — no Ethernet latency. The bandwidth is bounded by DRAM (for interleaved inputs) or L1 NOC (for sharded inputs).

### No cross-device synchronization

MeshPartition does not use `GlobalSemaphore` or EDM channels. There is no cross-device synchronization. All devices execute their slice independently. This also means MeshPartition is safe to call on sub-meshes or individual devices without the full mesh being synchronized.

### Program caching

The cache key is `(dim, cluster_axis, output_mem_config)`. Changing the tensor shape invalidates the Slice sub-program's cache as well. Keep shapes constant across calls.

---

### Common Errors (MeshPartition)

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: dim not divisible by cluster axis size` | `input_shape[dim]` is not divisible by N | Pad the dimension to the nearest multiple of N |
| Output shape unexpected | `cluster_axis` selects the wrong axis size | Verify `cluster_axis` matches the axis you intend to partition across; check `mesh.shape` |
| Each device produces the same slice | `cluster_axis` points to an axis of size 1 | For a 1×N mesh, use `cluster_axis=1`; `cluster_axis=0` is the row axis with size 1 |
| Input tensor changes between calls cause recompilation | Tensor shape varies | Keep shapes constant |

---

## Relationship Between ReduceToRoot and MeshPartition

These two operations are often used in complementary roles in a pipeline-parallel attention model:

1. **MeshPartition** distributes the query, key, value projections to each device's attention head shard at the start of the attention layer (no network traffic).

2. Devices compute local SDPA attention, producing partial `(L, S, M)` tuples.

3. **ReduceToRoot** collects all partial `(L, S, M)` tuples on the root device and combines them with the flash-attention merge formula, producing the final attention output at the root.

If the attention output is then needed by all devices (for subsequent layers), a `ttnn.broadcast` from the root distributes it back, and `ttnn.mesh_partition` reshards it for the next attention layer.

---

*Back to [Chapter 3 Index](index.md)*

*Back to [3.2 AllToAllDispatch and AllToAllCombine](all_to_all.md)*

*Next: [Chapter 4 — Async Overlap](../ch4_async_overlap/index.md)*
