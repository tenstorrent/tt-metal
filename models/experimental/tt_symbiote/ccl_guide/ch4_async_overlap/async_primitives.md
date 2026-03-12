# 4.2 Async Primitives

All async operations live under `ttnn::experimental` and are registered at `ttnn.experimental.<op_name>`. They require persistent output buffers, externally managed `GlobalSemaphore` objects, and (optionally) a `SubDeviceId` to separate CCL helper cores from compute cores. See [Section 4.1](why_async.md) for the conceptual foundation before using these APIs.

### Sync vs async comparison

| Sync op | Async op | Blocks | Semaphores | Persistent buffer | Pipeline tuning |
|---------|----------|--------|------------|-------------------|-----------------|
| `ttnn.all_gather` | `ttnn.experimental.all_gather_async` | Sync: yes; Async: no | Sync: internal; Async: caller-supplied `GlobalSemaphore` | Sync: allocated internally; Async: optional `persistent_output_buffer` | `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel` |
| `ttnn.all_reduce` | `ttnn.experimental.all_reduce_async` | Sync: yes; Async: no | Sync: internal; Async: 3 sets (`barrier`, `rs`, `ag`) in basic overloads | Sync: N/A; Async: optional `buffer_tensor` in Llama overload | (same tuning params when using cluster_axis overloads) |
| `ttnn.reduce_scatter` | `ttnn.experimental.reduce_scatter_minimal_async` | Sync: yes; Async: no | Sync: internal; Async: caller-supplied `multi_device_global_semaphore` list | Sync: optional single tensor; Async: optional list (`persistent_output_buffers`) | `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`, `intermediate_memory_config` |

Additional async-only distinctions: `all_reduce_async` requires an explicit `math_op` (`ttnn.ReduceType.Sum` for the standard case) — `ttnn.all_reduce` always sums and has no `math_op`. `all_reduce_async` Overload 3 adds `use_noc1_only` and `use_optimal_ccl_for_llama` for advanced Llama deployments.

---

## `ttnn.experimental.all_gather_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.hpp`

The async variant of AllGather. Equivalent semantic result to `ttnn.all_gather` (see [Section 2.1](../ch2_basic_operations/all_gather.md)), but returns without waiting for the Ethernet transfers to complete. The caller is responsible for waiting on `barrier_semaphore` before reading the output.

`all_gather_async_reversed` is also available: identical but the assembled output has reversed device ordering. The tensor width must be divisible by `32 * num_devices` when using the reversed API.

### Overloads

The operation has three Python-visible overloads (from the nanobind binding):

**Overload 1 — simple ring** (no persistent buffer, no cluster_axis):

```python
output = ttnn.experimental.all_gather_async(
    input_tensor,                        # ttnn.Tensor — local shard
    dim,                                 # int — gather dimension
    multi_device_global_semaphore,       # GlobalSemaphore or List[GlobalSemaphore] — per-round sync
    num_links=1,                         # int — Ethernet links
    memory_config=None,                  # Optional[ttnn.MemoryConfig]
    topology=ttnn.Topology.Ring,         # ttnn.Topology — default Ring
    subdevice_id=None,                   # Optional[ttnn.SubDeviceId]
    use_optimal_ccl_for_llama=False,     # bool — enable Llama-specific kernel path
    barrier_semaphore=None,              # Optional[GlobalSemaphore] — completion signal
    sub_core_grids=None,                 # Optional[CoreRangeSet] — restrict worker cores
)
# Returns: ttnn.Tensor
```

**Overload 2 — ring with persistent buffer and cluster_axis** (most common for async use):

```python
output = ttnn.experimental.all_gather_async(
    input_tensor,                        # ttnn.Tensor — local shard
    persistent_output_buffer=None,       # Optional[ttnn.Tensor] — pre-allocated output; None = allocate
    dim,                                 # int — gather dimension
    multi_device_global_semaphore,       # GlobalSemaphore or List[GlobalSemaphore]
    num_links=1,
    memory_config=None,
    topology=ttnn.Topology.Ring,
    subdevice_id=None,
    cluster_axis=None,                   # Optional[int] — mesh axis
    use_optimal_ccl_for_llama=False,
    barrier_semaphore=None,
    chunks_per_sync=None,                # Optional[int] — how many chunks before each sync point
    num_workers_per_link=None,           # Optional[int] — Tensix helper cores per link
    num_buffers_per_channel=None,        # Optional[int] — EDM pipeline depth per channel
    sub_core_grids=None,
)
# Returns: ttnn.Tensor
```

**Overload 3 — linear topology with explicit mesh_device** (for Linear topology only):

```python
output = ttnn.experimental.all_gather_async(
    input_tensor,
    dim,
    cluster_axis,                        # int — required for linear overload
    mesh_device,                         # MeshDevice — explicit device handle
    topology,                            # ttnn.Topology — required, typically Linear
    multi_device_global_semaphore,
    persistent_output_tensor=None,
    num_links=None,
    memory_config=None,
    subdevice_id=None,
    use_optimal_ccl_for_llama=False,
    barrier_semaphore=None,
    sub_core_grids=None,
)
# Returns: ttnn.Tensor
```

### Tuning parameters

**`chunks_per_sync`**: The number of tensor chunks that are transferred between successive `multi_device_global_semaphore` sync points. Smaller values give finer-grained overlap (compute can start sooner) but add per-chunk semaphore overhead. Larger values amortize semaphore cost but require the consumer to wait longer. Default `None` selects a value based on tensor size.

**`num_workers_per_link`**: Number of Tensix cores assigned as reader/writer helpers per Ethernet link. More workers can sustain higher link saturation for large tensors. Each worker occupies one core in the CCL SubDevice partition. Default `None` auto-selects.

**`num_buffers_per_channel`**: EDM pipeline depth — the number of in-flight buffer slots per ERISC channel. More buffers hide Ethernet round-trip latency. Each slot consumes `eth_buffer_size_bytes` of ERISC L1. Default `None` auto-selects from the `EriscDatamoverConfig` computed size.

### Illustrative example

```python
import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
N = 4  # devices

# Setup: semaphore and persistent buffer (do once before inference loop)
sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
ag_sem    = ttnn.create_global_semaphore(mesh, sem_cores, 0)
barrier   = ttnn.create_global_semaphore(mesh, sem_cores, 0)

out_shape = list(input_shard.shape)
out_shape[3] *= N
persistent_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG
)

# Async dispatch — returns before transfers complete
output = ttnn.experimental.all_gather_async(
    input_shard,
    persistent_output_buffer=persistent_buf,
    dim=3,
    multi_device_global_semaphore=ag_sem,
    topology=ttnn.Topology.Ring,
    barrier_semaphore=barrier,
    subdevice_id=ccl_sub_id,
)

# ← here you can dispatch compute on independent data ←

# Wait for AllGather to complete before reading output
ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
# OR: the barrier_semaphore is polled inside the next kernel that consumes `output`
```

---

## `ttnn.experimental.all_reduce_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp`

The async variant of AllReduce. Internally decomposes into ReduceScatter + AllGather, with three separate semaphore sets for the barrier, ReduceScatter phase, and AllGather phase. Requires a persistent fabric to be enabled.

### Overloads

**Overload 1 — explicit semaphore sets with `num_devices`**:

```python
output = ttnn.experimental.all_reduce_async(
    input_tensor,                        # ttnn.Tensor
    num_devices,                         # int — total device count
    barrier_semaphores,                  # List[GlobalSemaphore] — global completion
    rs_global_semaphores,                # List[GlobalSemaphore] — ReduceScatter phase
    ag_global_semaphores,                # List[GlobalSemaphore] — AllGather phase
    math_op,                             # ttnn.ReduceType — e.g. ttnn.ReduceType.Sum
    memory_config=None,
    topology=ttnn.Topology.Linear,       # default Linear
    num_links=None,
    subdevice_id=None,
)
# Returns: ttnn.Tensor
```

**Overload 2 — `cluster_axis` + `mesh_device` with optional semaphore sets**:

```python
output = ttnn.experimental.all_reduce_async(
    input_tensor,
    cluster_axis=None,                   # Optional[int]
    mesh_device=mesh,                    # MeshDevice
    barrier_semaphores=None,             # Optional[List[GlobalSemaphore]]
    rs_global_semaphores=None,
    ag_global_semaphores=None,
    math_op,                             # ttnn.ReduceType — required, no default
    memory_config=None,
    topology=ttnn.Topology.Linear,
    num_links=None,
    subdevice_id=None,
)
```

**Overload 3 — buffer_tensor + single semaphore (Llama-optimized path)**:

```python
output = ttnn.experimental.all_reduce_async(
    input_tensor,
    buffer_tensor,                       # ttnn.Tensor — persistent intermediate buffer (required)
    cluster_axis,                        # int — required
    mesh_device,                         # MeshDevice
    multi_device_global_semaphore,       # GlobalSemaphore — single semaphore
    dtype=None,                          # Optional[ttnn.DataType]
    memory_config=None,
    topology=ttnn.Topology.Linear,
    num_links=None,
    subdevice_id=None,
    use_noc1_only=False,                 # bool — restrict to NOC1 for isolation
    use_optimal_ccl_for_llama=False,     # bool — Llama-specific kernel optimization
)
```

### Illustrative example

```python
import ttnn

# Three semaphore sets for AllReduce async (one per phase + barrier)
sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
barrier_sems = ttnn.create_global_semaphore(mesh, sem_cores, 0)
rs_sems      = ttnn.create_global_semaphore(mesh, sem_cores, 0)
ag_sems      = ttnn.create_global_semaphore(mesh, sem_cores, 0)

output = ttnn.experimental.all_reduce_async(
    partial_output,
    num_devices=4,
    barrier_semaphores=barrier_sems,
    rs_global_semaphores=rs_sems,
    ag_global_semaphores=ag_sems,
    math_op=ttnn.ReduceType.Sum,
    topology=ttnn.Topology.Linear,
    subdevice_id=ccl_sub_id,
)

# Dispatch independent compute here while AllReduce is in flight
# ...

ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
```

---

## `ttnn.experimental.reduce_scatter_minimal_async`

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp`

The async variant of ReduceScatter. The "minimal" designation indicates it uses a streamlined kernel path designed for overlap: it accepts a pre-allocated `persistent_output_buffers` list and optional intermediate memory config for the scratch region.

```python
output = ttnn.experimental.reduce_scatter_minimal_async(
    input_tensor,                        # ttnn.Tensor — partial result on each device
    persistent_output_buffers=None,      # Optional[List[ttnn.Tensor]] — pre-allocated output(s)
    dim,                                 # int — scatter dimension
    multi_device_global_semaphore,       # List[GlobalSemaphore] — per-round sync
    barrier_semaphore=None,              # Optional[GlobalSemaphore] — completion signal
    num_links=1,                         # int
    memory_config=None,                  # Optional[ttnn.MemoryConfig] — output layout
    intermediate_memory_config=None,     # Optional[ttnn.MemoryConfig] — scratch buffer layout
    topology=ttnn.Topology.Ring,         # default Ring
    subdevice_id=None,
    cluster_axis=None,
    chunks_per_sync=None,                # Optional[int] — chunk granularity for overlap
    num_workers_per_link=None,           # Optional[int]
    num_buffers_per_channel=None,        # Optional[int]
)
# Returns: ttnn.Tensor with output_shape[dim] = input_shape[dim] / num_devices
```

**`intermediate_memory_config`**: The ReduceScatter accumulation scratch region. If `None`, uses the same config as `memory_config`. Setting this to L1 while the output goes to DRAM can improve accumulation throughput.

**`persistent_output_buffers`**: A list because the scatter produces multiple intermediate shards internally. In most use cases, pass `None` (let the runtime allocate) or pass a list with a single pre-allocated tensor matching the expected output shape.

### Illustrative example

```python
# Async ReduceScatter: dispatch, then overlap with next layer's compute
rs_sems = ttnn.create_global_semaphore(mesh, sem_cores, 0)
barrier = ttnn.create_global_semaphore(mesh, sem_cores, 0)

# Pre-allocate output
out_shape = list(partial.shape)
out_shape[3] //= N
rs_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG
)

shard = ttnn.experimental.reduce_scatter_minimal_async(
    partial,
    persistent_output_buffers=[rs_buf],
    dim=3,
    multi_device_global_semaphore=rs_sems,
    barrier_semaphore=barrier,
    topology=ttnn.Topology.Ring,
    subdevice_id=ccl_sub_id,
)

# Overlap: run next layer's input preparation while scatter is in flight
# ...

ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
# shard is now ready — shape is partial.shape[3] // N on dim 3
```

---

## `ttnn.experimental.send_async` and `ttnn.experimental.recv_async`

Sources:
- `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp`

Point-to-point async send and receive operations using a `MeshSocket`. Unlike the collective ops, these are directed: one device sends to exactly one receiver, using a pre-configured socket connection.

```python
# Send: dispatch a tensor over the mesh socket
output_tensors = ttnn.experimental.send_async(
    input_tensor,     # ttnn.Tensor — tensor to send
    mesh_socket,      # tt.tt_metal.distributed.MeshSocket — pre-configured socket
)
# Returns: List[ttnn.Tensor] (typically empty; side-effect is transmission)

# Receive: write incoming data into output_tensor
output_tensors = ttnn.experimental.recv_async(
    output_tensor,    # ttnn.Tensor — pre-allocated buffer to receive into
    mesh_socket,      # tt.tt_metal.distributed.MeshSocket — pre-configured socket
)
# Returns: List[ttnn.Tensor]
```

### MeshSocket concept

A `MeshSocket` is a pre-configured point-to-point channel between two specific devices. It specifies:
- Sender device coordinate and core
- Receiver device coordinate and core
- FIFO buffer size (`socket_mem_config.fifo_size`) — must be ≥ the tensor's aligned page size

The socket is created at the metal layer and passed in. Unlike the collective ops that automatically discover device topology from the `MeshDevice`, `send_async`/`recv_async` require explicit socket configuration.

### Validation requirements

From `send_recv_utils.hpp`:
- Exactly one input tensor
- Tensor must be on a device
- `mesh_socket.get_socket_endpoint_type()` must match the operation (SENDER for `send_async`, RECEIVER for `recv_async`)
- `fifo_size >= input_tensor.buffer()->aligned_page_size()`

> **Gotcha:** The FIFO size constraint is checked at dispatch time, not at socket creation time. If your tensor page size changes between calls (e.g., different sequence lengths), re-validate or resize the socket.

### When to use send_async / recv_async

These primitives are appropriate for:
- Pipeline-parallel models where each stage sends activations to the next stage device
- Custom ring-attention implementations that need fine-grained per-step control over data movement
- Any pattern where only two specific devices communicate, not a full collective

For mesh-wide collectives, prefer `all_gather_async` / `reduce_scatter_minimal_async` — they handle topology routing automatically.

---

## Synchronizing After Async Ops

After dispatching one or more async ops, use one of two synchronization mechanisms:

### Device-level sync

```python
# Wait for all outstanding ops on the CCL subdevice to complete
ttnn.experimental.synchronize_devices(mesh_device, subdevice_id=ccl_sub_id)
```

This is a host-side blocking call. After it returns, all tensors produced by async ops on `ccl_sub_id` are safe to read.

### Semaphore-based sync (within a kernel)

The `barrier_semaphore` parameter on each async op can be consumed inside a downstream kernel — the kernel waits on the semaphore before reading the output tensor, without requiring a host-side sync. This is the mechanism used in traced overlap patterns where host-side sync calls would break the trace.

---

*Back to [Chapter 4 Index](index.md)*

*Next: [4.3 Overlap Patterns](overlap_patterns.md)*
