# 3.1 ReduceScatter

## Concept

ReduceScatter answers: *each device holds a full-sized partial result tensor — how do all devices end up each holding one reduced shard, without any device holding the full result?*

After ReduceScatter, device `i` holds the element-wise sum of shard `i` from every device's input. The output is `1/N` the size of the input along the scatter dimension.

```
Before ReduceScatter   (4 devices, input shape [1, 1, 32, 256], scatter dim=3)

  Device 0: [P0_A | P0_B | P0_C | P0_D]   (P0_X = columns for quarter X)
  Device 1: [P1_A | P1_B | P1_C | P1_D]
  Device 2: [P2_A | P2_B | P2_C | P2_D]
  Device 3: [P3_A | P3_B | P3_C | P3_D]

After ReduceScatter   (each device holds one reduced shard, shape [1, 1, 32, 64])

  Device 0: [P0_A + P1_A + P2_A + P3_A]   ← shard A, fully reduced
  Device 1: [P0_B + P1_B + P2_B + P3_B]   ← shard B
  Device 2: [P0_C + P1_C + P2_C + P3_C]   ← shard C
  Device 3: [P0_D + P1_D + P2_D + P3_D]   ← shard D
```

Output shape on the scatter dimension: `input_shape[dim] / N` where N is the number of participating devices.

ReduceScatter is the *inverse* of AllGather on the scatter dimension and the *first phase* of AllReduce:

```
AllReduce  =  ReduceScatter  +  AllGather
              (reduce → shard)  (distribute → replicate)
```

When the downstream operation can consume a shard directly — e.g., a column-parallel matmul that only needs its own slice — stopping after ReduceScatter avoids the AllGather bandwidth cost entirely.

### When to use ReduceScatter

| Situation | Why ReduceScatter |
|-----------|-------------------|
| Row-parallel linear layer output, next layer is column-parallel | Next layer only needs one shard; full AllReduce would waste bandwidth on AllGather |
| Gradient synchronization in data-parallel training, gradients are then sharded for optimizer | Optimizer shards work on one gradient slice per device; no need to replicate |
| First phase of a custom AllReduce | Build AllReduce explicitly when you need to insert computation between the two phases |
| Ring-attention: reduce partial attention outputs before the next sequence chunk | Each device contributes partial softmax numerators/denominators; ReduceScatter collects one shard before moving to the next chunk |

ReduceScatter is bandwidth-optimal when the downstream layer can directly consume the scattered output. If all devices ultimately need the full result, prefer `ttnn.all_reduce` (Chapter 2) which handles both phases.

---

## Python API

Source: `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp`

```python
output_tensor = ttnn.reduce_scatter(
    input_tensor,                    # ttnn.Tensor — partial result on each device
    dim,                             # int — scatter dimension; output_shape[dim] = input_shape[dim] / N
    cluster_axis=None,               # Optional[int] — mesh axis (0=rows, 1=cols); None = all devices
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin; leave None normally
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout; None = match input
    output_tensor=None,              # Optional[ttnn.Tensor] — preallocated output buffer
    num_links=None,                  # Optional[int] — Ethernet links to use; None = auto
    topology=None,                   # Optional[ttnn.Topology] — Ring or Linear; None = auto
)
# Returns: ttnn.Tensor with output_shape[dim] = input_shape[dim] / num_devices
```

From the nanobind docstring:
> "When the layout is row-major or the scatter breaks apart tiles, we use the composite reduce-scatter implementation that falls back to all-broadcast."

### Parameter notes

**`dim`** (required, positional): The dimension to reduce over and scatter along. The input is conceptually divided into N equal slices on `dim`; device `i` receives the reduced version of slice `i`. The scatter must divide the dimension evenly — `input_shape[dim]` must be divisible by N.

> **Gotcha:** `dim` must divide the input dimension cleanly into N tile-aligned chunks. If `input_shape[dim] / N` is not a multiple of 32 (the tile width/height), ReduceScatter falls back to the AllBroadcast-based composite implementation, which has higher bandwidth overhead. Pad to the nearest tile multiple before calling.

**`cluster_axis`**: Restricts ReduceScatter to one axis of a 2-D mesh. `cluster_axis=1` scatters across all devices in the same row (column-axis devices). When unset, scatters across the entire flattened device list.

**`output_tensor`**: Pre-allocate the output buffer to avoid allocation overhead. The pre-allocated tensor must have shape `[..., input_shape[dim] // N, ...]` where `N` is the device count along the relevant axis.

**`topology`**: As in Chapter 2 — must match physical cabling. See [Section 2.1 Gotcha](../ch2_basic_operations/all_gather.md#data-flow-ring-allgather).

**`num_links`**, **`memory_config`**, **`subdevice_id`**: Same semantics as in Chapter 2 (see [Section 2.1 Parameter notes](../ch2_basic_operations/all_gather.md#parameter-notes)).

---

## Illustrative Examples

### Basic ReduceScatter on a 1-D mesh

```python
import torch
import ttnn

# 1×4 mesh (4 devices in a row)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

# Each device holds a [1, 1, 32, 256] partial result (e.g., from a local matmul)
# Shape must be divisible by 4 on the scatter dim (256 / 4 = 64 ✓)
partial = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)

distributed = ttnn.from_torch(
    partial,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),  # each device starts with same shape
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# ReduceScatter along dim=3: each device receives [1, 1, 32, 64] (256/4 = 64)
scattered = ttnn.reduce_scatter(distributed, dim=3, topology=ttnn.Topology.Ring)
# scattered.shape == [1, 1, 32, 64] on every device
# Device 0 holds the reduced sum of columns 0..63 across all devices
# Device 1 holds columns 64..127, etc.
```

### Row-parallel linear with ReduceScatter (instead of AllReduce)

```python
# Tensor-parallel row-linear: each device holds a row-shard of the weight matrix
# After local matmul, partial outputs are [batch, seq, hidden]
# Next op is column-parallel so only needs one shard of the hidden dimension

partial_output = ttnn.matmul(input_shard, weight_row_shard)

# ReduceScatter: reduce across devices, keep one shard per device
# Input [batch, seq, hidden] → output [batch, seq, hidden/N]
shard_output = ttnn.reduce_scatter(
    partial_output,
    dim=3,                           # scatter hidden dimension
    topology=ttnn.Topology.Ring,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# shard_output can go directly into a column-parallel matmul without AllGather
```

### 2-D mesh ReduceScatter along one axis

```python
# 2×4 mesh: tensor-parallel across columns, data-parallel across rows
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Scatter only within each row (cluster_axis=1 = same row, different columns)
scattered = ttnn.reduce_scatter(
    partial_output,
    dim=3,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
)
# Output shape[dim] = input_shape[dim] / 4 (4 devices per row)
# Row axis is unaffected
```

### Pre-allocated output buffer

```python
# Pre-allocate to avoid per-call allocation overhead
import ttnn

N = 4  # number of devices
input_shape = list(partial_output.shape)
output_shape = input_shape.copy()
output_shape[3] //= N

output_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape(output_shape),
    ttnn.bfloat16,
    ttnn.TILE_LAYOUT,
    mesh,
    ttnn.DRAM_MEMORY_CONFIG,
)

scattered = ttnn.reduce_scatter(
    partial_output,
    dim=3,
    output_tensor=output_buf,
    topology=ttnn.Topology.Ring,
)
```

---

## Data Flow: Ring ReduceScatter

With `topology=ttnn.Topology.Ring` and N devices, ReduceScatter runs N-1 rounds of send-and-accumulate. Each round:

1. Each device sends its *current accumulation* shard to the next device in the ring.
2. Each device receives the arriving shard from the previous device.
3. The arriving shard is element-wise added to the local partial shard.

```
Initial state (each device holds the full [P_A | P_B | P_C | P_D], N=4):

  Dev 0: [P0_A | P0_B | P0_C | P0_D]
  Dev 1: [P1_A | P1_B | P1_C | P1_D]
  Dev 2: [P2_A | P2_B | P2_C | P2_D]
  Dev 3: [P3_A | P3_B | P3_C | P3_D]

Round 1 — each device sends its own starting shard and accumulates the arriving shard:
  Dev 0 → Dev 1: sends P0_D    Dev 0 receives P3_C from Dev 3  →  accumulates P0_C + P3_C
  Dev 1 → Dev 2: sends P1_A    Dev 1 receives P0_D from Dev 0  →  accumulates P1_D + P0_D
  Dev 2 → Dev 3: sends P2_B    Dev 2 receives P1_A from Dev 1  →  accumulates P2_A + P1_A
  Dev 3 → Dev 0: sends P3_C    Dev 3 receives P2_B from Dev 2  →  accumulates P3_B + P2_B

  (partial sums rotate around the ring for N-1 total rounds)

After round N-1:
  Dev 0: [P0_A + P1_A + P2_A + P3_A]  ← fully reduced shard A
  Dev 1: [P0_B + P1_B + P2_B + P3_B]  ← fully reduced shard B
  Dev 2: [P0_C + P1_C + P2_C + P3_C]  ← fully reduced shard C
  Dev 3: [P0_D + P1_D + P2_D + P3_D]  ← fully reduced shard D
```

Total data transferred per device: `(N-1) * shard_size = (N-1) * input_size / N`. The ring keeps all N links active simultaneously.

### Linear ReduceScatter

With `topology=ttnn.Topology.Linear`, the algorithm runs a sequential sweep from device 0 to device N-1. Each device accumulates the partial sums arriving from its left neighbour and forwards the updated partial to its right neighbour. No wrap-around link is required. This is simpler but serialized — total latency scales with N.

---

## Under the Hood

### Program structure

The ReduceScatter device operation (`ReduceScatterDeviceOperation`) shares the same `ReduceScatterProgramArtifacts` infrastructure as the experimental `reduce_scatter_minimal_async` operation (imported from `ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/`). Each device runs the same Reader and Writer kernels as AllGather (see [Ch2 §2.1 — Under the Hood](../ch2_basic_operations/all_gather.md#under-the-hood-kernel-internals)) plus a **Compute kernel** that performs element-wise `add_tiles` on each arriving chunk — this is the key difference from AllGather, where received data is stored but not reduced. GlobalSemaphore usage is identical to AllGather — see Section 2.1.

### Fallback to AllBroadcast composite

The composite path transfers `N * (N-1) * tensor_size` bandwidth (N× worse than ring) by:
1. AllBroadcasting all inputs to all devices
2. Each device locally reducing its assigned shard

Avoid by ensuring `input_shape[dim] / N` is a multiple of 32.

### Program caching

The program factory caches on `(dim, topology, num_links, memory_config, cluster_axis, subdevice_id)`. Changing any of these — including `dim` between calls — forces recompilation.

> **Gotcha:** Changing the tensor shape between calls (e.g., varying sequence length) invalidates the cache. See [Section 2.1 — Program caching](../ch2_basic_operations/all_gather.md#program-caching) for the general rule.

---

## Memory Layout and Performance

ReduceScatter inherits the same input layout sensitivity as AllGather (see [Section 2.1 — Memory Layout](../ch2_basic_operations/all_gather.md#memory-layout-and-performance)). L1-sharded inputs produce the largest contiguous NOC transfers and best link utilization. The L1-sharding setup is identical; substitute `partial_output` for `input_tensor` and call `ttnn.reduce_scatter`.

### Output shape planning

The output shape on `dim` is `input_shape[dim] // N`. When chaining into AllGather to reconstruct AllReduce, this output feeds directly into `ttnn.all_gather` without shape adjustment:

```python
# Manual AllReduce via ReduceScatter + AllGather
# (use ttnn.all_reduce in practice; this is for illustration)
shard = ttnn.reduce_scatter(partial_output, dim=3, topology=ttnn.Topology.Ring)
# shard.shape[3] == partial_output.shape[3] // N

full = ttnn.all_gather(shard, dim=3, topology=ttnn.Topology.Ring)
# full.shape[3] == partial_output.shape[3]  (restored)
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: scatter dim not divisible by num_devices` | `input_shape[dim]` is not divisible by N | Pad the scatter dimension to the nearest multiple of N |
| Fallback to AllBroadcast (high bandwidth usage) | `input_shape[dim] / N` is not tile-aligned (not a multiple of 32) | Pad to make the per-device shard a tile multiple |
| Output shape mismatch | Caller expected unchanged shape | ReduceScatter shrinks `dim` by factor N; check the downstream op's input shape expectation |
| Program recompiles every iteration | Tensor shape or `dim` changes between calls | Keep shapes and `dim` constant; use padding |

*See also: [Section 2.1 error table](../ch2_basic_operations/all_gather.md#common-errors) for `cluster_axis` and topology errors common to all ops.*

---

*Back to [Chapter 3 Index](index.md)*

*Next: [3.2 AllToAllDispatch and AllToAllCombine](all_to_all.md)*
