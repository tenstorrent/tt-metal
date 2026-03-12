# 4.3 Overlap Patterns

This section covers the concrete patterns used to structure compute/communication overlap in production workloads. All patterns build on the concepts from [Section 4.1](why_async.md) and the APIs from [Section 4.2](async_primitives.md).

---

## Pattern 1: AllGather → Compute (Standard Tensor-Parallel)

The simplest overlap pattern appears in column-parallel linear layers. The previous layer produced a shard on each device; the current layer needs the full tensor. With synchronous AllGather the shard is transmitted, then the matmul starts. With async AllGather, the two overlap:

```
Synchronous (no overlap):
  Time →
  Tensix: [matmul layer N] [idle] [matmul layer N+1]
  ERISC:  [idle] [AllGather] [idle]

Async with overlap:
  Time →
  Tensix: [matmul layer N][matmul on partial (no-dep work)] [matmul layer N+1]
  ERISC:  [idle]          [AllGather overlapping]
                                       ↑ barrier_semaphore consumed by layer N+1 matmul
```

### Implementation

Setup (semaphore creation and persistent buffer allocation) follows the same pattern as [§4.2 Async Primitives — Illustrative example](async_primitives.md#illustrative-example): create semaphores and pre-allocate the persistent buffer once before the loop.

```python
# ----- Inference loop -----
for step in range(num_steps):
    # 1. Dispatch AllGather for layer N's shard — returns immediately
    gathered = ttnn.experimental.all_gather_async(
        shard,
        persistent_output_buffer=ag_buf,
        dim=3,
        multi_device_global_semaphore=ag_sem,
        topology=ttnn.Topology.Ring,
        barrier_semaphore=barrier,
        subdevice_id=ccl_sub_id,
    )

    # 2. While AllGather is in flight, run compute that does NOT need `gathered`
    norm_out = ttnn.rms_norm(embedding, weight)

    # 3. Wait for AllGather to complete, then run the matmul
    ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
    matmul_out = ttnn.matmul(gathered, weight_shard)

    # 4. Reset semaphores for next iteration
    ttnn.reset_global_semaphore_value(ag_sem, 0)
    ttnn.reset_global_semaphore_value(barrier, 0)
```

> **Gotcha:** If step 2 has no independent compute — or if independent compute is shorter than the AllGather — there is no benefit. Profile first with synchronous ops to measure actual CCL time before investing in async.

---

## Pattern 2: Microbatch Pipeline (AllGather → Compute → ReduceScatter)

For larger batch sizes the most effective pattern pipelines *two microbatches* so that one microbatch's AllGather overlaps the other's matmul which overlaps the other's ReduceScatter:

```
Pipeline with 2 microbatches (µB0, µB1):

  Time →  T0       T1       T2       T3       T4       T5
  ERISC:  [AG µB0] [RS µB0] [AG µB1] [RS µB1]
  Tensix: [setup]  [MM µB0] [MM µB0] [MM µB1] [MM µB1] [out]
          ↑               ↑ AG µB1 dispatched before MM µB0 finishes
          AG µB0 dispatched before MM µB0 starts
```

In steady state, each timestep does one matmul while simultaneously running one AllGather from the previous step and one ReduceScatter for the step before that. Compute and both communication phases are in flight at the same time.

### Pseudocode structure

```python
# Setup: two sets of semaphores and buffers for double-buffering
def make_semaphores():
    return (
        ttnn.create_global_semaphore(mesh, sem_cores, 0),
        ttnn.create_global_semaphore(mesh, sem_cores, 0),
    )

ag_sems_0, ag_barrier_0 = make_semaphores()
ag_sems_1, ag_barrier_1 = make_semaphores()
rs_sems_0, rs_barrier_0 = make_semaphores()
rs_sems_1, rs_barrier_1 = make_semaphores()

# Two sets of persistent buffers
ag_buf_0, ag_buf_1 = alloc_ag_buffer(), alloc_ag_buffer()
rs_buf_0, rs_buf_1 = alloc_rs_buffer(), alloc_rs_buffer()

for step in range(num_steps):
    # Select which buffer set to use this step (alternate)
    ag_buf   = ag_buf_0   if step % 2 == 0 else ag_buf_1
    ag_sem   = ag_sems_0  if step % 2 == 0 else ag_sems_1
    ag_bar   = ag_barrier_0 if step % 2 == 0 else ag_barrier_1

    # Dispatch AllGather for this step
    gathered = ttnn.experimental.all_gather_async(
        shard[step], persistent_output_buffer=ag_buf,
        dim=3, multi_device_global_semaphore=ag_sem,
        barrier_semaphore=ag_bar, subdevice_id=ccl_sub_id,
    )

    if step > 0:
        # Wait for previous step's AllGather (already partially complete)
        ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)

        # Run matmul using previous step's gathered tensor
        prev_gathered = ag_buf_1 if step % 2 == 0 else ag_buf_0
        mm_out = ttnn.matmul(prev_gathered, weight)

        # Dispatch ReduceScatter for previous step's matmul output
        rs_buf   = rs_buf_0   if step % 2 == 0 else rs_buf_1
        rs_sem   = rs_sems_0  if step % 2 == 0 else rs_sems_1
        shard[step] = ttnn.experimental.reduce_scatter_minimal_async(
            mm_out, persistent_output_buffers=[rs_buf], dim=3,
            multi_device_global_semaphore=rs_sem, subdevice_id=ccl_sub_id,
        )

    # Reset semaphores for this step's buffers
    ttnn.reset_global_semaphore_value(ag_sem, 0)
```

> **Gotcha:** Double-buffering doubles the L1/DRAM consumption for persistent buffers. For large models with tight memory budgets, the persistent buffer allocation may fail. Profile L1 usage before enabling this pattern.

---

## Pattern 3: Ring Attention Overlap

Ring attention distributes the attention computation across devices by rotating key-value blocks around a ring. Each device computes attention for its own query against the KV block it currently holds, then passes the KV block to the next device while receiving the next KV block from the previous device.

The operation `ttnn.experimental.ring_attention_all_gather_async` is purpose-built for this pattern:

```python
# Source: ring_attention_all_gather_async.hpp
output_tensors = ttnn.experimental.ring_attention_all_gather_async(
    input_tensors,                       # List[ttnn.Tensor] — current KV block (one per device)
    persistent_output_buffer,            # List[ttnn.Tensor] — pre-allocated output (mutable)
    dim,                                 # int — gather dimension
    multi_device_global_semaphore,       # List[GlobalSemaphore]
    cluster_axis,                        # int — required
    mesh_device,                         # MeshDevice
    topology,                            # ttnn.Topology
    num_links=None,
    memory_config=None,
    subdevice_id=None,
)
# Returns: List[ttnn.Tensor] — one gathered tensor per device
```

Note: `persistent_output_buffer` is a `List[ttnn.Tensor]` here and is non-optional (mutable reference in C++). Pre-allocate matching the full gathered KV shape.

### Ring attention overlap timeline

```
4-device ring, 4 KV rotation steps:

Step 0: Each device holds KV_local
  ERISC: [dispatch: send KV_local → next device, recv KV from prev]
  Tensix: [Attn(Q_local, KV_local)]

Step 1: Each device holds KV from step 0's sender
  ERISC: [dispatch: rotate KV again]
  Tensix: [Attn(Q_local, KV_received_step0)]  ← overlaps with step 1 rotation

Step 2: Similarly overlapping...

After N-1 rotations: each device has computed attention against all N KV blocks
```

The key insight: each attention compute step is independent of the *next* rotation. The current KV block is being processed while the next KV block is arriving over Ethernet.

```python
# Ring attention loop pattern
kv_current = kv_local
attn_accum = None

for ring_step in range(num_devices):
    # Dispatch async rotation: send current KV to next, receive next KV
    kv_next = ttnn.experimental.ring_attention_all_gather_async(
        [kv_current],
        persistent_output_buffer=kv_buf_next,
        dim=1,
        multi_device_global_semaphore=ring_sems,
        cluster_axis=1,
        mesh_device=mesh,
        topology=ttnn.Topology.Ring,
        subdevice_id=ccl_sub_id,
    )

    # Compute attention against current KV while next KV is arriving
    attn_partial = flash_attention(query, kv_current)
    attn_accum = merge_attention(attn_accum, attn_partial)

    # Wait for rotation to complete
    ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
    ttnn.reset_global_semaphore_value(ring_sems[0], 0)
    kv_current = kv_next[0]
```

---

## Pattern 4: SubDevice Dispatch

When using SubDevices for overlap, the dispatch order matters. Both SubDevices must have their kernels dispatched before either starts executing — otherwise the hardware may deadlock waiting for a partner that has not been dispatched yet.

```python
# CORRECT: dispatch CCL first, then compute, before either is waited on
ccl_output = ttnn.experimental.all_gather_async(
    shard, persistent_output_buffer=ag_buf, dim=3,
    multi_device_global_semaphore=ag_sem, subdevice_id=ccl_sub_id,
)

# Dispatch compute (different SubDevice) — both now in-flight simultaneously
compute_output = ttnn.matmul(
    independent_tensor, weight, compute_with_storage_grid_size=...,
)

# NOW wait for CCL
ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)
# CCL output is ready; compute output may still be running if it was longer
```

```python
# INCORRECT: waiting for CCL before dispatching compute eliminates overlap
ccl_output = ttnn.experimental.all_gather_async(...)
ttnn.experimental.synchronize_devices(mesh, subdevice_id=ccl_sub_id)  # blocks here
compute_output = ttnn.matmul(...)   # now sequential, no overlap
```

> **Gotcha:** TTNN's host-side dispatch is non-blocking for most ops, but `synchronize_devices` is explicitly a blocking call. Any `synchronize_devices` call in the critical path ends the overlap window for all subsequent ops.

---

## Pattern 5: Traced Overlap

TTNN Trace mode captures a sequence of Metal programs and replays them with minimal host interaction. When the overlap pattern is inside a trace, the host-side sync calls must either be removed or captured correctly.

### Structure for traced overlap

```python
import ttnn

# 1. Allocate all persistent buffers and semaphores BEFORE trace capture
ag_buf = ttnn.allocate_tensor_on_device(...)
ag_sem = ttnn.create_global_semaphore(mesh, sem_cores, 0)

# 2. Capture the trace
tid = ttnn.begin_trace_capture(mesh, cq_id=0)

# Inside the trace: semaphore reset must be explicit
ttnn.reset_global_semaphore_value(ag_sem, 0)   # captured in trace

output = ttnn.experimental.all_gather_async(
    shard, persistent_output_buffer=ag_buf,
    dim=3, multi_device_global_semaphore=ag_sem,
    barrier_semaphore=ag_sem,               # reuse as barrier for simplicity
    subdevice_id=ccl_sub_id,
)
matmul_out = ttnn.matmul(other_tensor, weight)
# NOTE: no synchronize_devices inside the trace — use semaphore-based sync
# The matmul kernel polls ag_sem before reading `output`, handled by the program

ttnn.end_trace_capture(mesh, tid, cq_id=0)

# 3. Replay
for _ in range(num_iterations):
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
```

> **Gotcha:** `ttnn.experimental.synchronize_devices` cannot be captured in a trace — it is a host function, not a Metal program. Replace it with semaphore-polling barriers inside the consumer kernel, or use `blocking=True` on `ttnn.execute_trace` to synchronize after the entire trace completes.

---

## Common Pitfalls

### Race condition: reading output before completion

```python
# WRONG: reading output immediately after async dispatch
output = ttnn.experimental.all_gather_async(shard, ...)
result = ttnn.to_torch(output)  # garbage: transfer not complete
```

Always synchronize or poll the barrier semaphore before reading.

### Semaphore not reset between iterations

See the reset Gotcha in [§4.1 Semaphores](why_async.md#semaphores-in-async-operations).

### Buffer aliasing

```python
# WRONG: same persistent buffer used for two in-flight async ops
output_a = ttnn.experimental.all_gather_async(..., persistent_output_buffer=buf)
output_b = ttnn.experimental.all_gather_async(..., persistent_output_buffer=buf)
# output_a and output_b both write to buf — data corruption
```

Use separate buffers for each concurrently in-flight async op (hence double-buffering in Pattern 2).

### Wrong SubDevice for consumer kernel

```python
# WRONG: consumer matmul dispatched to CCL sub-device
gathered = ttnn.experimental.all_gather_async(..., subdevice_id=ccl_sub_id)
matmul_out = ttnn.matmul(gathered, weight, subdevice_id=ccl_sub_id)
# Matmul and AllGather compete for the same Tensix cores → serialized or OOM
```

The CCL SubDevice must be exclusively for CCL helper kernels. Use the compute SubDevice (or no `subdevice_id`) for matmuls.

### Persistent buffer in wrong memory region

```python
# Suboptimal: persistent buffer in DRAM increases AllGather latency
ag_buf = ttnn.allocate_tensor_on_device(..., ttnn.DRAM_MEMORY_CONFIG)
```

For latency-critical inference, allocate persistent buffers in L1. Monitor L1 fragmentation across iterations if using many small persistent buffers.

---

## Code Structure Recommendations

1. **Separate setup from inference loop.** All semaphore creation, persistent buffer allocation, and SubDevice manager setup should happen once before the loop.

2. **Profile with sync ops first.** Measure the synchronous baseline before implementing overlap. If CCL time is < 5% of wall time, async complexity is not justified.

3. **Use one SubDevice per resource type.** One SubDevice for CCL helper kernels, one for compute. Mixing them produces unpredictable serialization.

4. **Reset semaphores inside the trace when using Trace mode.** Do not rely on out-of-trace host calls to reset semaphores between trace replays.

5. **Pre-warm before measuring.** The first iteration after trace capture or after program cache miss is slower. Run several warm-up iterations before benchmarking.

6. **Test correctness with sync ops first.** Implement the model with synchronous ops, verify numerical correctness, then port to async. Async bugs (race conditions, wrong barriers) are hard to debug because the failure is non-deterministic.

---

## Related Experimental Operations

The experimental CCL directory contains additional fused overlap primitives beyond the core async ops:

| Operation | Purpose |
|-----------|---------|
| `all_gather_matmul_async` | Fused AllGather + matmul: starts matmul as soon as the first chunk arrives |
| `matmul_reduce_scatter_async` | Fused matmul + ReduceScatter: scatters chunks as they are produced |
| `llama_all_gather_matmul_async` | Llama-specific fused path with Llama head layout |
| `llama_reduce_scatter_matmul` | Llama-specific fused reduce-scatter + matmul |
| `strided_all_gather_async` | AllGather with strided tensor layout |
| `slice_reshard_async` | Async slice + reshard for pipeline-parallel boundary |
| `neighbor_pad_async` | Async padding to neighboring device |

These fused ops achieve tighter overlap than the two-step "dispatch async, then compute" pattern by running the matmul and the collective within a single Metal program that pipelines chunk-by-chunk. See the respective directories under `ttnn/cpp/ttnn/operations/experimental/ccl/` for headers and usage.

---

*Back to [Chapter 4 Index](index.md)*

*Back to [4.2 Async Primitives](async_primitives.md)*

*Next: [Chapter 5 — Op Fusion (CCL + Matmul)](../ch5_op_fusion/index.md)*
