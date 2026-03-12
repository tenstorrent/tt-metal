# 5.2 Fused Ops

This section covers the architecture-neutral fused ops: `all_gather_matmul_async`, `matmul_reduce_scatter_async`, and `strided_all_gather_minimal_matmul_async`. These use the standard AllGather/ReduceScatter kernel infrastructure with a `FusedOpSignaler` bridge. See [Section 5.1](why_fusion.md) for the conceptual foundation.

All three are under `ttnn::experimental`.

---

## `ttnn.experimental.all_gather_matmul_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async.hpp`

Fuses an AllGather with an immediately following matmul. The AllGather gathers `input_tensor` across the ring, and the matmul computes `gathered × weight_tensor`. The matmul begins processing the first gathered chunk before the AllGather has finished receiving all shards. See [§5.1](why_fusion.md) for the DRAM round-trip motivation and tile-level pipeline diagram. The weight tensor must be sharded so that each ring step maps to the corresponding column block — the fused program computes `chunk_i × W_col_i` and accumulates.

### API

```python
output_tensors = ttnn.experimental.all_gather_matmul_async(
    input_tensor,                        # ttnn.Tensor — local shard (each device holds 1/N of dim)
    weight_tensor,                       # ttnn.Tensor — weight matrix (full or sharded)
    persistent_output_buffer,           # Optional[ttnn.Tensor] — pre-allocated AG output; None = allocate
    dim,                                 # int — AllGather dimension
    multi_device_global_semaphore,       # List[GlobalSemaphore] — per-chunk sync between CCL and matmul
    all_gather_core_grid_offset,         # CoreCoord — grid offset for AllGather worker cores
    bias=None,                           # Optional[ttnn.Tensor] — matmul bias
    num_links=1,                         # int — Ethernet links for AllGather
    memory_config_ag=None,               # Optional[ttnn.MemoryConfig] — AllGather output memory
    topology=ttnn.Topology.Ring,         # ttnn.Topology — default Ring
    barrier_semaphore=None,              # Optional[GlobalSemaphore] — completion signal
    subdevice_id=None,                   # Optional[ttnn.SubDeviceId]
    memory_config_mm=None,               # Optional[ttnn.MemoryConfig] — matmul output memory
    transpose_a=False,                   # bool
    transpose_b=False,                   # bool
    dtype=None,                          # Optional[ttnn.DataType]
    program_config=None,                 # Optional[ttnn.MatmulProgramConfig]
    activation=None,                     # Optional[str] — e.g. "relu"
    compute_kernel_config=None,          # Optional[ttnn.DeviceComputeKernelConfig]
    core_grid=None,                      # Optional[ttnn.CoreGrid]
    chunks_per_sync=None,                # Optional[int] — chunk granularity for signaler
    num_workers_per_link=None,           # Optional[int]
    num_buffers_per_channel=None,        # Optional[int]
)
# Returns: List[ttnn.Tensor] — [gathered_tensor, matmul_output]
```

### Return value

Returns two tensors as a list:
1. `[0]` — the fully gathered tensor (written to `persistent_output_buffer` if provided, otherwise allocated)
2. `[1]` — the matmul output

Returning the gathered tensor allows downstream ops to use it directly without re-gathering.

### Parameter notes

**`all_gather_core_grid_offset`**: Required. A `CoreCoord` that shifts the AllGather worker cores away from the matmul worker cores on the same chip. Without this offset, the AllGather helper kernels and the matmul kernels would attempt to occupy overlapping Tensix core coordinates, causing a core allocation failure. Typical value: `CoreCoord(0, 4)` to shift AllGather workers to rows 4+ if matmul occupies rows 0–3.

**`persistent_output_buffer`**: Optional. If provided, the AllGather writes its full output here (for downstream use). If `None`, the runtime allocates internally. For the fused path, the matmul reads from the L1 CB regardless of whether a persistent buffer is provided — the persistent buffer is the *secondary* output, not the primary communication channel to the matmul.

**`multi_device_global_semaphore`**: A `List[GlobalSemaphore]` (not a single semaphore). Each element in the list corresponds to one direction of the ring. The AllGather worker signals the matmul worker via these semaphores each time a chunk is ready in the CB.

**`chunks_per_sync`**: How many tile chunks the AllGather accumulates before signaling the matmul. Smaller = finer-grained pipelining; larger = less signaling overhead. Default `None` auto-selects based on tensor geometry.

> **Gotcha:** `all_gather_core_grid_offset` is a required positional argument, not optional. Passing `CoreCoord(0, 0)` (no offset) may work if the matmul grid does not start at row 0, but will silently produce wrong results or crash if there is a core conflict. Always explicitly plan the core layout before invoking this op.

### Illustrative example

```python
import ttnn

# Assume 4-device ring; each device holds shard [1, 1, 4096, 1024]
# Weight tensor is [1, 1, 4096, 4096] (full, replicated on each device)

sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
ag_sems = [ttnn.create_global_semaphore(mesh, sem_cores, 0) for _ in range(2)]  # 2 ring directions

# Pre-allocate gathered output buffer (optional but recommended for reuse)
gathered_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 4096, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
    mesh, ttnn.L1_MEMORY_CONFIG
)

# Fused dispatch — returns before matmul is complete
results = ttnn.experimental.all_gather_matmul_async(
    input_shard,
    weight,
    gathered_buf,
    dim=3,
    multi_device_global_semaphore=ag_sems,
    all_gather_core_grid_offset=ttnn.CoreCoord(0, 4),  # AllGather on rows 4+
    num_links=1,
    topology=ttnn.Topology.Ring,
    subdevice_id=ccl_sub_id,
    chunks_per_sync=2,
)
gathered_tensor = results[0]
mm_output      = results[1]

# Wait for completion (both AG and matmul are in the same program; one sync covers both)
ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)

# Reset semaphores for next iteration
for sem in ag_sems:
    ttnn.reset_global_semaphore_value(sem, 0)
```

### Under the Hood

`AllGatherMatmulAsyncMeshWorkloadFactory` builds a single Metal program using the `MULTI`-mode `FusedOpSignaler` handoff described in [§5.1 — The FusedOpSignaler Mechanism](why_fusion.md#the-fusedopsignaler-mechanism). The AllGather writer simultaneously writes each chunk to the L1 CB (for the matmul) and to `persistent_output_buffer` (for downstream use).

---

## `ttnn.experimental.matmul_reduce_scatter_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async.hpp`

Fuses a matmul with an immediately following ReduceScatter. The matmul computes `input_tensor × weight_tensor`, then the ReduceScatter reduces and scatters the output across the ring — tile by tile, without writing the full matmul result to DRAM first.

See [§5.1](why_fusion.md) for the DRAM round-trip motivation. In tensor-parallel feed-forward layers, this op replaces:

```python
# Unfused:
partial = ttnn.linear(x, w_col_shard)
shard   = ttnn.reduce_scatter(partial, dim=3)

# Fused:
shard = ttnn.experimental.matmul_reduce_scatter_async(x, w_col_shard, ...)
```

### API

```python
output_tensors = ttnn.experimental.matmul_reduce_scatter_async(
    input_tensor,                        # ttnn.Tensor — input activation
    weight_tensor,                       # ttnn.Tensor — weight matrix (column shard)
    persistent_intermediate_buffer,      # ttnn.Tensor — pre-allocated intermediate (REQUIRED, mutable)
    persistent_output_buffer,            # ttnn.Tensor — pre-allocated RS output (REQUIRED, mutable)
    dim,                                 # int — ReduceScatter scatter dimension
    multi_device_global_semaphore,       # List[GlobalSemaphore] — matmul → RS signaling
    reduce_scatter_core_grid_offset,     # CoreCoord — grid offset for ReduceScatter workers
    barrier_semaphore=None,              # Optional[GlobalSemaphore] — completion signal
    bias=None,                           # Optional[ttnn.Tensor]
    num_links=1,                         # int
    memory_config_rs=None,               # Optional[ttnn.MemoryConfig] — RS output memory
    intermediate_memory_config_rs=None,  # Optional[ttnn.MemoryConfig] — RS scratch buffer memory
    topology=ttnn.Topology.Ring,         # ttnn.Topology
    subdevice_id=None,                   # Optional[ttnn.SubDeviceId]
    memory_config_mm=None,               # Optional[ttnn.MemoryConfig] — matmul output memory
    transpose_a=False,
    transpose_b=False,
    dtype=None,
    program_config=None,
    activation=None,
    compute_kernel_config=None,
    core_grid=None,
)
# Returns: List[ttnn.Tensor] — [matmul_output, reduce_scatter_shard]
```

### Return value

Returns two tensors:
1. `[0]` — the full matmul output (may be in `persistent_intermediate_buffer`'s memory)
2. `[1]` — the final ReduceScatter shard (`output_shape[dim] = input_shape[dim] / N`)

### Parameter notes

**`persistent_intermediate_buffer`** (required, mutable): The buffer the matmul writes its output into before the ReduceScatter reads it. Unlike `all_gather_matmul_async` where the intermediate is a L1 CB, here the matmul output may be larger than L1, so this buffer is explicitly managed. Must be pre-allocated to match the full matmul output shape.

**`persistent_output_buffer`** (required, mutable): The buffer the ReduceScatter writes its final reduced shard into. Both persistent buffers are mutable references in C++ — changes to them are visible to the caller after completion.

**`reduce_scatter_core_grid_offset`**: Same purpose as `all_gather_core_grid_offset` — separates ReduceScatter worker cores from matmul worker cores.

**`intermediate_memory_config_rs`**: Memory config for the ReduceScatter's internal accumulation scratch region. If `None`, uses `memory_config_rs`. Setting this to L1 while `memory_config_rs` is DRAM can improve accumulation throughput (accumulation happens in fast L1; final output spills to DRAM).

> **Gotcha:** Both `persistent_intermediate_buffer` and `persistent_output_buffer` are required (no optional equivalent unlike `all_gather_matmul_async`). Failing to pre-allocate them causes a runtime error at dispatch time. Allocate both once before the inference loop and reuse.

### Illustrative example

```python
import ttnn

# 4-device ring; each device computes a column-parallel partial result
# input: [1, 1, 4096, 4096]; weight column shard: [1, 1, 4096, 1024]
# matmul output (before RS): [1, 1, 4096, 1024]
# RS output (shard): [1, 1, 1024, 1024]

sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
rs_sems = [ttnn.create_global_semaphore(mesh, sem_cores, 0) for _ in range(2)]
barrier = ttnn.create_global_semaphore(mesh, sem_cores, 0)

# Pre-allocate both persistent buffers
mm_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 4096, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
    mesh, ttnn.L1_MEMORY_CONFIG
)
rs_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 1024, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
    mesh, ttnn.L1_MEMORY_CONFIG
)

results = ttnn.experimental.matmul_reduce_scatter_async(
    activation,
    weight_shard,
    mm_buf,
    rs_buf,
    dim=3,
    multi_device_global_semaphore=rs_sems,
    reduce_scatter_core_grid_offset=ttnn.CoreCoord(0, 4),
    barrier_semaphore=barrier,
    num_links=1,
    topology=ttnn.Topology.Ring,
    subdevice_id=ccl_sub_id,
)
mm_output = results[0]
rs_shard  = results[1]

ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
for sem in rs_sems:
    ttnn.reset_global_semaphore_value(sem, 0)
ttnn.reset_global_semaphore_value(barrier, 0)
```

### Under the Hood

Internally uses the `SINGLE`-mode `ReduceScatterFusedOpSignaler` as described in [§5.1 — The FusedOpSignaler Mechanism](why_fusion.md#the-fusedopsignaler-mechanism) — the matmul writer wakes one privileged RS core at `reduce_scatter_core_grid_offset`, which fans out to the full RS grid. The matmul simultaneously writes to `persistent_intermediate_buffer` for downstream inspection.

---

## `ttnn.experimental.strided_all_gather_minimal_matmul_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/strided_all_gather_minimal_matmul_async.hpp`

A variant of AllGather+Matmul that uses a *strided* AllGather kernel path and a *minimal* matmul kernel path. The strided AllGather uses `StridedAllGatherFusedOpSignaler` and the `MinimalMatmulFusedOpSignaler`, which is lighter-weight than the full `MatmulFusedOpSignaler`. This variant is tuned for cases where the input tensor has a strided memory layout (non-contiguous tiles in the AllGather dimension).

### API

```python
output_tensors = ttnn.experimental.strided_all_gather_minimal_matmul_async(
    input_tensor,                        # ttnn.Tensor — local shard (strided layout)
    weight_tensor,                       # ttnn.Tensor
    persistent_output_buffer,           # Optional[ttnn.Tensor] — pre-allocated AG output
    dim,                                 # int
    multi_device_global_semaphore,       # List[GlobalSemaphore]
    strided_all_gather_core_grid_offset, # CoreCoord — required
    num_links=1,                         # int
    memory_config_ag=None,               # Optional[ttnn.MemoryConfig]
    topology=ttnn.Topology.Ring,         # default Ring
    cluster_axis=None,                   # Optional[int]
    bias=None,                           # Optional[ttnn.Tensor]
    fused_activation=None,               # Optional[UnaryWithParam] — e.g. relu/gelu
    config=None,                         # Optional[MinimalMatmulConfig]
    memory_config_mm=None,               # Optional[ttnn.MemoryConfig]
    compute_kernel_config=None,          # Optional[ttnn.DeviceComputeKernelConfig]
    num_workers_per_link=None,           # Optional[int]
    num_buffers_per_channel=None,        # Optional[int]
    read_local_slice_from_input=None,    # Optional[bool] — read local shard from input_tensor directly
)
# Returns: List[ttnn.Tensor] — [gathered_tensor, matmul_output]
```

### Key differences from `all_gather_matmul_async`

| Feature | `all_gather_matmul_async` | `strided_all_gather_minimal_matmul_async` |
|---------|--------------------------|------------------------------------------|
| AG kernel | Standard AllGather | Strided AllGather (`StridedAllGatherFusedOpSignaler`) |
| Matmul kernel | Full `MatmulProgramConfig` path | `MinimalMatmulConfig` — reduced-overhead path |
| Signaler | `MatmulFusedOpSignaler` | `MinimalMatmulFusedOpSignaler` |
| Activation | `activation: Optional[str]` | `fused_activation: Optional[UnaryWithParam]` |
| `chunks_per_sync` | Yes | No (handled by `MinimalMatmulFusedOpSignaler`) |
| Input layout | Contiguous | Strided (non-contiguous in AllGather dimension) |
| `read_local_slice_from_input` | N/A | Optional: if True, the local shard is read directly from `input_tensor` rather than being sent through the ring for the local slot |

**`MinimalMatmulConfig`**: A reduced-parameter matmul config (`ttnn::experimental::minimal_matmul::MinimalMatmulConfig`) that specifies core grid, block sizes, and CB allocations without the full `MatmulProgramConfig` schema. Used when the caller wants tighter control over the matmul micro-architecture.

**`read_local_slice_from_input`**: When True, the local device's shard is read directly from `input_tensor` (avoiding one round-trip around the ring for the local element). Default `None` auto-selects based on ring size.

> **Gotcha:** The `fused_activation` parameter type is `UnaryWithParam`, not a plain string. Construct it as `ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)` — not `"relu"` as in the full matmul API.

---

*Back to [Chapter 5 Index](index.md)*

*Next: [5.3 Llama Fused Ops](llama_fused_ops.md)*
