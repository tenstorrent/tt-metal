# 5.3 Llama Fused Ops

This section covers the Llama-specific fused operations: `llama_all_gather_matmul_async`, `llama_rs_matmul`, `all_gather_concat`, and `fused_rms_1_1_32_8192`. These are hand-tuned for the Llama model's tensor layout, head configuration, and core grid arrangement. They use `GlobalCircularBuffer` for zero-copy device-to-device tile delivery and Llama-specific `MatmulFusedOpSignaler` variants.

All are under `ttnn::experimental`.

---

## `ttnn.experimental.llama_all_gather_matmul_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/llama_all_gather_matmul_async.hpp`

Llama-optimized fused AllGather + Matmul. Differs from the generic `all_gather_matmul_async` in several ways:

- Uses `MatmulFusedOpSignalerType::LLAMA_ALL_GATHER` — a dedicated signaler path with a `start_cb_index` field for the GlobalCircularBuffer slot
- Takes an explicit `intermediate_tensor` (the pre-allocated AllGather output buffer) as a positional argument, not an optional parameter
- Accepts a `GlobalCircularBuffer` (`global_cb`) for direct L1-to-L1 tile transfer without DRAM
- `topology` is a required positional argument (caller must always specify Ring or Linear explicitly)
- Returns a single `ttnn.Tensor` (matmul output only), not a list of two tensors

### API

```python
mm_output = ttnn.experimental.llama_all_gather_matmul_async(
    input_tensor0,                       # ttnn.Tensor — local shard (input to AllGather and to matmul as in0)
    input_tensor1,                       # ttnn.Tensor — weight matrix (matmul in1)
    intermediate_tensor,                 # ttnn.Tensor — pre-allocated AllGather output (REQUIRED)
    dim,                                 # int — AllGather dimension
    cluster_axis,                        # int — mesh axis for multi-axis mesh
    mesh_device,                         # MeshDevice — explicit device handle
    topology,                            # ttnn.Topology — REQUIRED positional (Ring or Linear)
    multi_device_global_semaphore,       # GlobalSemaphore — single semaphore (not a list)
    num_links=None,                      # Optional[int] — defaults to None (auto-select)
    ag_memory_config=None,               # Optional[ttnn.MemoryConfig]
    mm_memory_config=None,               # Optional[ttnn.MemoryConfig]
    subdevice_id=None,                   # Optional[ttnn.SubDeviceId]
    program_config=None,                 # Optional[ttnn.MatmulProgramConfig]
    compute_kernel_config=None,          # Optional[ttnn.DeviceComputeKernelConfig]
    dtype=None,                          # Optional[ttnn.DataType]
    global_cb=None,                      # Optional[GlobalCircularBuffer] — zero-copy L1 path
)
# Returns: ttnn.Tensor — matmul output
```

### Key differences from `all_gather_matmul_async`

| Feature | `all_gather_matmul_async` | `llama_all_gather_matmul_async` |
|---------|--------------------------|--------------------------------|
| Return type | `List[ttnn.Tensor]` (2 tensors) | `ttnn.Tensor` (matmul output only) |
| Intermediate tensor | Optional `persistent_output_buffer` | Required `intermediate_tensor` positional |
| Semaphore type | `List[GlobalSemaphore]` | Single `GlobalSemaphore` |
| `topology` | Keyword, default Ring | Required positional |
| `mesh_device` | N/A | Required positional |
| `cluster_axis` | Keyword, optional | Required positional |
| `global_cb` | N/A | Optional (enables zero-copy L1 path) |
| Signaler | `MatmulFusedOpSignalerType::ALL_GATHER` | `MatmulFusedOpSignalerType::LLAMA_ALL_GATHER` |
| Core grid offset | `all_gather_core_grid_offset` (explicit) | Managed internally by Llama factory |
| `chunks_per_sync` | Optional | N/A (Llama path uses CB slot signaling) |

### Under the Hood

Uses the `LLAMA_ALL_GATHER` signaler path (see [§5.1 — The FusedOpSignaler Mechanism](why_fusion.md#the-fusedopsignaler-mechanism)). The `start_cb_index` field of `MatmulFusedOpSignaler` specifies the `GlobalCircularBuffer` slot for each ring step; when `global_cb=None`, falls back to DRAM-backed intermediate.

> **Gotcha:** `intermediate_tensor` must be pre-allocated to the full gathered shape `[batch, seq, heads × head_dim]` before the first call. Unlike `all_gather_matmul_async` where `persistent_output_buffer=None` triggers internal allocation, here there is no fallback — a missing or incorrectly shaped `intermediate_tensor` raises a validation error.

### Illustrative example

```python
import ttnn

# Llama-70B, 8-device tensor parallel, attention projection
# input_tensor0: [1, 1, 32, 1024]  (local Q shard per device; 8 heads × 128 head_dim)
# input_tensor1: [1, 1, 8192, 4096] (weight Wq; column-parallel shard)
# After AllGather on dim=3: [1, 1, 32, 8192]
# Matmul output: [1, 1, 32, 4096]

sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
ag_sem = ttnn.create_global_semaphore(mesh, sem_cores, 0)

# Intermediate buffer for AllGather output
ag_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 32, 8192]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
    mesh, ttnn.L1_MEMORY_CONFIG
)

mm_output = ttnn.experimental.llama_all_gather_matmul_async(
    q_shard,
    weight_q,
    ag_buf,
    dim=3,
    cluster_axis=1,
    mesh_device=mesh,
    topology=ttnn.Topology.Linear,
    multi_device_global_semaphore=ag_sem,
    subdevice_id=ccl_sub_id,
    global_cb=global_circular_buf,      # optional but recommended for Llama
)

ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
ttnn.reset_global_semaphore_value(ag_sem, 0)
```

---

## `ttnn.experimental.llama_rs_matmul`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul.hpp`

Registered Python name: `ttnn.experimental.llama_rs_matmul`

Llama-optimized fused ReduceScatter + Matmul. The ReduceScatter reduces and scatters the first input, then feeds the result into a matmul against the weight. This is the inverse of `llama_all_gather_matmul_async`: it appears in the feed-forward layer's output projection path where ReduceScatter precedes the next attention layer's matmul.

### Concept

```
Unfused:
  [ReduceScatter: partial → shard (DRAM)] → [Matmul: shard × W → out]

Fused (Llama path):
  [ReduceScatter: shard arrives in L1 CB] → [Matmul: consume from CB, output → out]
  (privileged core orchestrates via LLAMA_REDUCE_SCATTER signaler)
```

The Llama RS path uses a privileged matmul core that atomically counts down `matmul_semaphore_target`. When all ReduceScatter workers have written their tiles into the matmul CB, the privileged core signals the rest of the matmul grid to begin.

### API

```python
output_tensors = ttnn.experimental.llama_rs_matmul(
    input_tensor,                        # ttnn.Tensor — partial result to scatter+reduce
    weight_tensor,                       # ttnn.Tensor — weight matrix
    intermediate_packet_buffer,          # ttnn.Tensor — pre-allocated RS packet buffer (REQUIRED, mutable)
    dim,                                 # int — scatter dimension (positional)
    cross_device_semaphore,              # GlobalSemaphore — single semaphore (not a list)
    cluster_axis,                        # int — required positional
    mesh_device,                         # MeshDevice — required positional
    num_links,                           # int — required positional (no default)
    subdevice_id,                        # tt.SubDeviceId — required positional (no default)
    second_weight_tensor=None,           # Optional[ttnn.Tensor] — second matmul weight (two-matmul variant)
    rs_tensor=None,                      # Optional[ttnn.Tensor] — alternative RS input
    topology=ttnn.Topology.Linear,       # default Linear
    memory_config_rs=None,               # Optional[ttnn.MemoryConfig]
    memory_config_mm=None,               # Optional[ttnn.MemoryConfig]
    compute_kernel_config=None,          # Optional[ttnn.DeviceComputeKernelConfig]
    global_cb=None,                      # Optional[GlobalCircularBuffer]
    core_grid=None,                      # Optional[ttnn.CoreGrid]
    transpose_a=False,
    transpose_b=False,
    dtype=None,
    program_config=None,
    activation=None,
    output_tile=None,
    optional_output_tensor=None,
    use_noc1_only=False,                 # bool — restrict traffic to NOC1
)
# Returns: List[ttnn.Tensor] — [rs_output, matmul_output] (or [rs_output, mm0_output, mm1_output] if second_weight_tensor is used)
```

### Required positional arguments

Unlike most CCL ops where `num_links` and `subdevice_id` have defaults, `llama_rs_matmul` requires both as positional arguments (no keyword default in nanobind). Always pass explicit values.

### `second_weight_tensor` — two-matmul variant

When `second_weight_tensor` is provided, `llama_rs_matmul` runs two matmuls:
1. `rs_shard × weight_tensor` → `mm0_output`
2. `rs_shard × second_weight_tensor` → `mm1_output`

Both matmuls share the same ReduceScatter output shard. This matches the Llama attention layer where the scattered KV shard is multiplied against both the key and value projection weights in the same fused pass. Returns three tensors: `[rs_output, mm0_output, mm1_output]`.

### `rs_tensor` — alternative ReduceScatter input

When `rs_tensor` is provided, the ReduceScatter operates on `rs_tensor` instead of `input_tensor`. `input_tensor` is then passed directly as the matmul's `in0`. This decouples the ReduceScatter data path from the matmul data path — useful when the matmul input has already been partially computed and the RS input is a separate tensor.

### `use_noc1_only`

When True, all CCL traffic is restricted to NOC1 (the second Network-on-Chip). NOC0 is left exclusively for matmul-related traffic. This prevents bandwidth contention between the ReduceScatter NOC traffic and the matmul's weight tensor reads on the same NOC.

> **Gotcha:** `num_links` and `subdevice_id` are required positional arguments with no defaults in the nanobind binding. The C++ default values shown in the header are not exposed to Python. Always pass explicit values.

> **Gotcha:** `intermediate_packet_buffer` is a mutable reference — the ReduceScatter writes packet-format data into it during the collective. The format is RS-specific and not directly interpretable as a standard tensor. Do not read from `intermediate_packet_buffer` after the call; use the returned `rs_output` instead.

### Illustrative example

```python
import ttnn

# Llama-70B feedforward output projection
# input_tensor: [1, 1, 32, 4096] (partial projection result, column-parallel)
# weight_tensor: [1, 1, 1024, 4096] (transposed weight shard)
# After RS on dim=2: shard [1, 1, 4, 4096]
# Matmul output: [1, 1, 4, hidden_dim]

sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
rs_sem = ttnn.create_global_semaphore(mesh, sem_cores, 0)

# Intermediate packet buffer for ReduceScatter internal state
rs_packet_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 32, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
    mesh, ttnn.DRAM_MEMORY_CONFIG   # RS packets can be DRAM; matmul CB is L1
)

results = ttnn.experimental.llama_rs_matmul(
    partial_projection,
    weight_shard,
    rs_packet_buf,
    dim=2,
    cross_device_semaphore=rs_sem,
    cluster_axis=1,
    mesh_device=mesh,
    num_links=1,
    subdevice_id=ccl_sub_id,
    topology=ttnn.Topology.Linear,
    use_noc1_only=True,
    global_cb=global_circular_buf,
)
rs_output  = results[0]
mm_output  = results[1]

ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
ttnn.reset_global_semaphore_value(rs_sem, 0)
```

### Under the Hood

The `LLAMA_REDUCE_SCATTER` signaler path in `MatmulFusedOpSignaler` uses a privileged-core protocol:

1. Each ReduceScatter worker core writes its reduced tile to the matmul CB and decrements a shared counter at `matmul_privilaged_semaphore` on the privileged matmul core
2. The privileged core polls the counter; when it reaches `matmul_semaphore_target` (all RS workers done), it sends a semaphore increment to all matmul worker cores via `rs_semaphore`
3. The matmul workers were blocking on `rs_semaphore`; they wake up and begin their computation

This fan-in → privileged broadcast pattern is more efficient than each RS worker independently signaling all matmul cores (`MULTI` mode) for large matmul grids where the per-chunk signaling overhead would dominate.

---

## `ttnn.experimental.all_gather_concat`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat.hpp`

Registered Python name: `ttnn.experimental.all_gather_concat`

Fuses an AllGather with a head-concatenation operation for multi-head attention (NLP). In attention, the query, key, and value tensors are often gathered across devices, then the heads from all devices are concatenated into the full multi-head layout. `all_gather_concat` performs both steps in a single kernel pass.

### Concept

```
Unfused:
  [AllGather: shard → gathered] → [concat_heads: gathered → (batch, seq, num_heads, head_dim)]

Fused:
  AllGather arrives in CB → head concat applied per tile → output in final head layout
```

The `num_heads` parameter tells the kernel how many attention heads to concatenate. The kernel interleaves the incoming tiles into the output in head-major order as they arrive.

### API

```python
output = ttnn.experimental.all_gather_concat(
    input_tensor,                        # ttnn.Tensor — local head shard
    buffer_tensor,                       # ttnn.Tensor — pre-allocated intermediate buffer (REQUIRED, mutable)
    dim,                                 # int — AllGather dimension
    cluster_axis,                        # int — required positional
    mesh_device,                         # MeshDevice — required positional
    multi_device_global_semaphore,       # GlobalSemaphore — single semaphore (Python name: multi_device_global_semaphore)
    num_heads,                           # int — number of attention heads to concat (required, no default)
    memory_config,                       # ttnn.MemoryConfig — required (no default)
    use_noc1_only=False,                 # bool
    num_links=1,                         # int — default 1
    topology=ttnn.Topology.Linear,       # default Linear
    subdevice_id=None,                   # Optional[ttnn.SubDeviceId]
)
# Returns: ttnn.Tensor — head-concatenated gathered tensor
```

### Parameter notes

**`buffer_tensor`** (required, mutable): Intermediate buffer used by the AllGather stage. Must be pre-allocated before the call. Unlike the output which is in head-concatenated layout, `buffer_tensor` holds the gathered-but-not-yet-concatenated intermediate.

**`multi_device_global_semaphore`**: Note the Python argument name is `multi_device_global_semaphore` (from the nanobind binding, line `nb::arg("multi_device_global_semaphore")`), even though the C++ parameter name is `global_semaphore`.

**`num_heads`**: Registered with `.noconvert()` in nanobind — must be passed as a plain Python `int`, not a numpy integer or tensor.

**`memory_config`**: Required (no default). Specifies the output tensor's memory layout. For attention workloads, typically `ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG` or `ttnn.L1_MEMORY_CONFIG`.

**Topology**: Defaults to `Linear` (unlike the generic ops which default to `Ring`). Llama's tensor-parallel dimension is typically a linear axis in the mesh.

> **Gotcha:** `num_heads` is registered with `nb::arg("num_heads").noconvert()`. Passing a value of type `numpy.int64` instead of Python `int` will raise a type error. Convert with `int(num_heads)` if loading from numpy arrays.

> **Gotcha:** `memory_config` is a required argument with no default — unlike most CCL ops where `memory_config=None` is the default. Omitting it causes a `TypeError` at the Python call site.

### Use in Llama attention

In Llama's multi-head attention, after the QKV projection each device holds a shard of Q, K, and V with `num_heads / num_devices` heads per device. After AllGather, all devices need the full `num_heads` heads for rotary embedding and attention. `all_gather_concat` handles both:

```python
# Each device holds: q_shard [batch, seq, local_heads, head_dim]
# After all_gather_concat: [batch, seq, total_heads, head_dim] on each device

q_full = ttnn.experimental.all_gather_concat(
    q_shard,
    q_buf,
    dim=2,
    cluster_axis=1,
    mesh_device=mesh,
    multi_device_global_semaphore=ag_sem,
    num_heads=int(total_heads),
    memory_config=ttnn.L1_MEMORY_CONFIG,
    topology=ttnn.Topology.Linear,
    subdevice_id=ccl_sub_id,
)
```

### Under the Hood

The fused program combines:
1. An AllGather ring reader/writer kernel that gathers tiles from all N devices
2. A head-permutation kernel that rearranges the gathered tiles into head-major order as they arrive in the CB

The head layout transformation is: for each incoming shard from device `d`, its heads are interleaved into slots `[d * local_heads, (d+1) * local_heads)` in the output tensor. This is equivalent to `torch.cat([s.split(local_heads, dim=-2) for s in shards], dim=-2)` but executed tile-by-tile without materializing the concatenated intermediate.

---

## `ttnn.fused_rms_1_1_32_8192`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp`

Registered Python name: `ttnn.fused_rms_1_1_32_8192`

Note: This op is registered directly under `ttnn.` (not `ttnn.experimental.`) — see the `register_operation` call in the header.

A fused RMS normalization + AllGather operation. Specifically optimized for the tensor shape `(1, 1, 32, 8192)` sharded on 1 core. The AllGather gathers the normalization statistics (mean, variance) across devices so each device can perform the final scale-and-shift with consistent statistics.

This is a pre/post normalization primitive for Llama's MoE (Mixture of Experts) or standard attention layers where the hidden state is gathered before RMSNorm application.

### Scope note

Unlike the other fused ops in this chapter, `fused_rms_1_1_32_8192` is a normalization+AllGather fusion rather than a CCL+matmul fusion. It is included here because it uses the same AllGather infrastructure and is typically used alongside `llama_all_gather_matmul_async` in the Llama pipeline.

### API

```python
output = ttnn.fused_rms_1_1_32_8192(
    input_tensor,                        # ttnn.Tensor — shape MUST be (1, 1, 32, 8192), sharded on 1 core
    program_config,                      # LayerNormProgramConfig — required
    cluster_axis,                        # int — required positional
    mesh_device,                         # MeshDevice — required positional
    global_semaphore,                    # GlobalSemaphore — required positional
    persistent_output_tensor=None,       # Optional[ttnn.Tensor]
    num_links=None,                      # Optional[int] — default None (auto)
    topology=ttnn.Topology.Linear,       # default Linear
    subdevice_id=None,                   # Optional[ttnn.SubDeviceId]
    dtype=None,                          # Optional[ttnn.DataType]
    compute_kernel_config=None,          # Optional[ttnn.DeviceComputeKernelConfig]
    memory_config=None,                  # Optional[ttnn.MemoryConfig]
    residual_input_tensor=None,          # Optional[ttnn.Tensor] — add residual before norm (pre-norm only)
    epsilon=1e-12,                       # float — normalization epsilon (post-norm only)
    weight=None,                         # Optional[ttnn.Tensor] — RMSNorm weight (gamma)
    stats=None,                          # Optional[ttnn.Tensor] — pre-computed statistics (bypasses local stat computation)
    use_noc1_only=False,                 # bool
)
# Returns: ttnn.Tensor — normalized output with AllGather-consistent statistics
```

### Pre-norm vs Post-norm mode

The op supports two modes selected by which parameters are used:

**Pre-norm mode** (compute stats, gather, apply norm):
- `residual_input_tensor`: if provided, `input = input + residual` before computing stats
- `stats`: should be `None` (stats are computed locally, then gathered)
- `epsilon`: ignored (constant `1e-12` in pre-norm mode per the nanobind comment)
- Use: positioned before the attention/FFN block

**Post-norm mode** (use pre-gathered stats, apply scale):
- `stats`: pre-computed statistics tensor (the gathered stats from pre-norm pass)
- `weight`: RMSNorm scale vector (gamma)
- `epsilon`: applied to stabilize normalization
- Use: positioned after reduction, applying the final scale

> **Gotcha:** The shape constraint `(1, 1, 32, 8192)` is hard-coded in the kernel. This op cannot handle other shapes — use `ttnn.rms_norm` for general shapes. The name encodes the constraint: `fused_rms_1_1_32_8192`.

> **Gotcha:** `program_config` is a `ttnn.operations.normalization.LayerNormProgramConfig`, not the matmul `MatmulProgramConfig`. The two are different types.

---

## Summary of Llama Fused Op Differences

| Op | Python name | Topology default | Semaphore type | Returns | Key constraint |
|----|-------------|-----------------|----------------|---------|----------------|
| `llama_all_gather_matmul_async` | `ttnn.experimental.llama_all_gather_matmul_async` | Required positional | Single `GlobalSemaphore` | `ttnn.Tensor` | `intermediate_tensor` required |
| `llama_rs_matmul` | `ttnn.experimental.llama_rs_matmul` | `Linear` | Single `GlobalSemaphore` | `List[ttnn.Tensor]` | `num_links`, `subdevice_id` required (no default) |
| `all_gather_concat` | `ttnn.experimental.all_gather_concat` | `Linear` | Single `GlobalSemaphore` | `ttnn.Tensor` | `memory_config` required; `num_heads` int only |
| `fused_rms_1_1_32_8192` | `ttnn.fused_rms_1_1_32_8192` | `Linear` | Single `GlobalSemaphore` | `ttnn.Tensor` | Shape `(1,1,32,8192)` hard-coded |

All Llama ops use a single `GlobalSemaphore` (not a list), `cluster_axis` + `mesh_device` instead of implicit topology discovery, and default to `Linear` topology. The generic ops in Section 5.2 use a list of semaphores and default to `Ring`.

---

## Related: `strided_all_gather_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/strided_all_gather_async.hpp`

This is not a fused (CCL+matmul) op, but a standalone AllGather variant for strided input tensors. It is listed here because it is the communication primitive underlying `strided_all_gather_minimal_matmul_async` (Section 5.2) and is commonly used in Llama-like models with non-standard tensor layouts.

```python
output = ttnn.experimental.strided_all_gather_async(
    input_tensor,                        # ttnn.Tensor — strided input
    persistent_output_buffer,           # Optional[ttnn.Tensor]
    dim,                                 # int — AllGather dimension
    multi_device_global_semaphore,       # List[GlobalSemaphore]
    num_links=1,
    memory_config=None,
    topology=ttnn.Topology.Ring,
    cluster_axis=None,
    tiles_per_chunk=None,                # Optional[int] — chunk size in tiles (strided-specific)
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    mm_cores_y=None,                     # Optional[int] — matmul core grid Y dimension hint
    mm_block_ht=None,                    # Optional[int] — matmul block height in tiles
    mm_block_wt=None,                    # Optional[int] — matmul block width in tiles
)
# Returns: ttnn.Tensor
```

The `mm_cores_y`, `mm_block_ht`, `mm_block_wt` parameters hint at the downstream matmul grid configuration so the strided AllGather can size its chunks to match the matmul's tile access pattern — enabling tight pipelining when used with `strided_all_gather_minimal_matmul_async`.

---

*Back to [Chapter 5 Index](index.md)*

*Back to [5.2 Fused Ops](fused_ops.md)*
