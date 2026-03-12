# 2.2 AllReduce

## Concept

AllReduce answers: *each device holds a partial result tensor — how do all devices end up with the fully reduced (summed) result?*

After AllReduce every participating device holds an identical tensor whose values are the element-wise sum of the corresponding tensors on all devices. The tensor shape is **unchanged** — AllReduce is an in-place semantic operation from the caller's perspective.

```
Before AllReduce   (4 devices, each holds a [1, 1, 32, 256] partial sum)

  Device 0: P0   (partial result from device 0's compute)
  Device 1: P1
  Device 2: P2
  Device 3: P3

After AllReduce

  Device 0: P0 + P1 + P2 + P3
  Device 1: P0 + P1 + P2 + P3
  Device 2: P0 + P1 + P2 + P3
  Device 3: P0 + P1 + P2 + P3
```

The output shape is identical to the input shape on every device.

### How it works internally: ReduceScatter + AllGather

AllReduce in tt-metal is implemented as a two-phase composite:

1. **ReduceScatter**: Each device scatters a different slice of the output to one peer, accumulating partial sums as they arrive. After ReduceScatter, each device holds a fully-reduced shard (a contiguous slice of the full reduced result), but only that shard.

2. **AllGather**: Each device's fully-reduced shard is gathered to all other devices. After AllGather every device holds the complete reduced tensor.

```
ReduceScatter phase (4 devices):

  Before:
    Dev 0: [P0_A | P0_B | P0_C | P0_D]
    Dev 1: [P1_A | P1_B | P1_C | P1_D]
    Dev 2: [P2_A | P2_B | P2_C | P2_D]
    Dev 3: [P3_A | P3_B | P3_C | P3_D]

  After ReduceScatter (each device owns one reduced shard):
    Dev 0: [P0_A + P1_A + P2_A + P3_A]  ← shard A, fully reduced
    Dev 1: [P0_B + P1_B + P2_B + P3_B]  ← shard B
    Dev 2: [P0_C + P1_C + P2_C + P3_C]  ← shard C
    Dev 3: [P0_D + P1_D + P2_D + P3_D]  ← shard D

AllGather phase:

  After AllGather (every device holds the full reduced tensor):
    Dev 0..3: [A_full | B_full | C_full | D_full]
```

This two-phase decomposition keeps bandwidth usage at `2 * (N-1) / N * tensor_size` per device — the theoretical optimum for ring-based AllReduce.

### When to use AllReduce

| Situation | Why AllReduce |
|-----------|---------------|
| Row-parallel linear layer: each device computes `Y_local = X_shard @ W_row` | Partial matrix products must be summed across all devices to get the full output |
| Attention output projection in tensor-parallel mode | Output of each head shard must be summed across devices |
| Loss aggregation across data-parallel replicas | Each replica computes a local loss; AllReduce produces the global mean |
| Embedding lookup with partitioned embedding table | Each device fetches some rows; the scatter-reduce produces the combined embedding |

AllReduce is bandwidth-optimal only when every device needs the full reduced result. If the next operation can work on a shard, prefer ReduceScatter (covered in Chapter 3) to avoid the AllGather phase.

---

## Python API

Source: `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.hpp`

```python
output_tensor = ttnn.all_reduce(
    input_tensor,                    # ttnn.Tensor — partial result on each device
    cluster_axis=None,               # Optional[int] — mesh axis (0=rows, 1=cols); None = all devices
    subdevice_id=None,               # Optional[ttnn.SubDeviceId] — sub-device pin; leave None normally
    memory_config=None,              # Optional[ttnn.MemoryConfig] — output layout; None = match input
    num_links=None,                  # Optional[int] — Ethernet links to use; None = auto
    topology=None,                   # Optional[ttnn.Topology] — Ring or Linear; None = auto
)
# Returns: ttnn.Tensor with the same shape as input_tensor, containing the element-wise sum
```

> **Gotcha:** AllReduce has **no `dim` parameter**. The reduction is always element-wise over the entire tensor — there is no dimension along which to reduce because the operation sums matching elements across devices, not within a single tensor.

### Parameter notes

**`cluster_axis`**: On a 2-D mesh, specifies which axis AllReduce runs along. `cluster_axis=0` reduces across all devices sharing the same column index; `cluster_axis=1` reduces across all devices sharing the same row index. For 1-D meshes, omit this parameter.

**`num_links`**: Each link uses ERISC L1 for EDM channel buffers. With `num_links=None`, the runtime picks the maximum available links. Explicitly limiting links reduces L1 pressure at the cost of bandwidth.

**`memory_config`**: When `None`, the output inherits the input's memory config. Specify explicitly when the downstream operation requires a different layout to avoid a separate layout conversion step.

**`topology`**: Accepts `ttnn.Topology.Ring` or `ttnn.Topology.Linear`. Must match physical cabling — see [Section 2.1 Gotcha](all_gather.md#data-flow-ring-allgather).

---

## Illustrative Examples

### Basic AllReduce on a 1-D mesh

```python
import torch
import ttnn

# 1×4 mesh (4 devices in a row)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

# Simulate each device holding a different partial result [1, 1, 32, 256]
# In practice these come from a preceding matmul on each device
partial = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)

# Replicate onto all devices (each device starts with the same value here;
# in real usage each device holds its own partial product)
distributed = ttnn.from_torch(
    partial,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Reduce: every device gets the sum of all partials
reduced = ttnn.all_reduce(distributed, topology=ttnn.Topology.Ring)
# reduced.shape == [1, 1, 32, 256] on every device
```

### Row-parallel linear with AllReduce

```python
# Tensor-parallel row-linear: each device holds a row shard of the weight matrix
# After the local matmul, all devices hold a partial output that must be summed.

# Local matmul on each device (produces partial output [batch, seq, hidden])
partial_output = ttnn.matmul(input_shard, weight_row_shard)

# AllReduce sums partials across all devices
full_output = ttnn.all_reduce(
    partial_output,
    topology=ttnn.Topology.Ring,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# full_output is identical on all devices and has shape [batch, seq, hidden]
```

### 2-D mesh AllReduce along one axis

```python
# 2×4 mesh: 2 rows, 4 columns (8 devices total)
# Tensor-parallel across columns (axis=1), data-parallel across rows (axis=0)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

# Reduce across the column axis only (cluster_axis=1 = same row, different columns)
# Row-parallel linear partitioned across 4 columns; reduce within each row
reduced = ttnn.all_reduce(
    partial_output,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
)
# Each device gets the sum of the 4 partials in its row; row axis stays independent
```

---

## Data Flow

AllReduce executes **ReduceScatter** (N-1 rounds, each device accumulates one reduced shard) followed by **AllGather** (N-1 rounds, each device's fully-reduced shard is distributed to all others). The round-by-round rotation diagram is shown in [Section 2.1 — Data Flow](all_gather.md#data-flow-ring-allgather); AllReduce's ReduceScatter phase uses the same ring rotation with an element-wise addition step at each receive.

Total data transferred per device: `2 * (N-1)/N * tensor_size`. For large N this approaches `2 * tensor_size`.

With `topology=ttnn.Topology.Linear`, both phases run as sequential sweeps (ReduceScatter: device 0 → N-1; AllGather: device N-1 → 0). Each hop is serialized. Use only when no wrap-around link exists.

---

## Under the Hood

### Program structure

`ttnn.all_reduce` is a composite operation. The program factory decomposes it into:

1. A **ReduceScatter** subprogram that runs ring/linear rotation with element-wise addition at each receive step. The addition happens in the Tensix compute kernel; the result of each accumulated chunk is written to a designated scratch region in L1 or DRAM.

2. An **AllGather** subprogram (identical to `ttnn.all_gather`) that redistributes the fully-reduced shards.

Both subprograms share the same ERISC EDM infrastructure. The EDM channels are configured once; the ReduceScatter and AllGather phases reuse the same Ethernet links without re-setup.

### No `dim` in operation_attributes_t

Unlike AllGather and ReduceScatter, AllReduce does not expose a scatter/gather dimension to the caller. Internally the ReduceScatter phase selects the scatter dimension automatically based on tensor shape. If you need explicit control over which dimension is scattered, call `ttnn.reduce_scatter` + `ttnn.all_gather` explicitly (Chapter 3).

### Program caching

Program cache behaviour is identical to AllGather — keep tensor shapes constant across calls (see [Section 2.1 — Program caching](all_gather.md#program-caching)).

---

## Memory Layout and Performance

### Input layout impact

AllReduce inherits the same layout sensitivity as AllGather (see [Section 2.1 — Memory Layout and Performance](all_gather.md#memory-layout-and-performance)). Convert to L1-sharded layout before calling AllReduce when the input comes from DRAM; substitute `partial_output` for `input_tensor` in the L1-sharding snippet shown there.

### Output layout

The output has the same shape as the input and uses the input's memory config by default. Specify `memory_config` explicitly when the downstream operation requires a different layout to avoid an extra conversion step.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Shape mismatch after AllReduce | Caller expected a different dimension to be modified | AllReduce never changes shape; check that the preceding layer produced the expected shape |
| Program recompiles every iteration | Tensor shape changes between calls | Fix tensor shape; use padding to keep shapes constant |
| Higher-than-expected memory usage | Internal ReduceScatter allocates scratch tensors | Pre-allocate in L1-sharded config to reduce DRAM traffic during the scatter phase |

*See also: [Section 2.1 error table](all_gather.md#common-errors) for `cluster_axis` and topology errors common to all ops.*

---

*Back to [Chapter 2 Index](index.md)*

*Next: [2.3 Broadcast and AllBroadcast](broadcast.md)*
