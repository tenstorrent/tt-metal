# 4.1 Why Async Matters

## The Synchronous Blocking Problem

Every synchronous CCL operation — `ttnn.all_gather`, `ttnn.reduce_scatter`, `ttnn.all_reduce` — blocks all Tensix compute cores until the collective is complete. The host runtime dispatches the CCL program, the ERISC cores perform the Ethernet transfers, and only when the last shard has arrived on every device does control return to the next operation.

The result is a staircase of idle time:

```
Synchronous execution timeline (single device, 4-device ring):

  Tensix  ████████▓▓▓▓▓▓▓▓████████▓▓▓▓▓▓▓▓████████
           │ matmul │ idle │ matmul │ idle │ matmul │
                    │  AllGather  │     │  AllGather  │
  ERISC    ░░░░░░░░░████████░░░░░░░░████████░░░░░░░░
           │ idle   │ Ethernet│ idle   │ Ethernet│ idle│

  ████ = active    ░ = idle    ▓ = waiting for CCL
```

For a model like Llama-70B running tensor-parallel inference across 8 devices, the AllGather after the attention projection and the ReduceScatter after the feedforward projection together can represent 20–40% of end-to-end latency on small batch sizes. The synchronous model throws away that time.

### Why this is hard to fix

The fundamental constraint is that the *synchronous* API gives the runtime no opportunity to schedule other work during a collective. When you call `ttnn.all_gather(shard, dim=3)`, the TTNN runtime:

1. Dispatches the AllGather Metal program to all devices
2. Waits for all ERISC transfers to complete (via a barrier semaphore)
3. Returns the gathered tensor

Between steps 1 and 3, no other TTNN operation can be dispatched to the Tensix cores on those devices. The Tensix cores are not literally stalled by the hardware — they just have no work queued.

---

## The Overlap Model

Async CCL operations decouple *dispatch* from *completion*. Instead of blocking until the collective is done, the async call:

1. Dispatches the communication program to the ERISC cores
2. Returns immediately
3. Leaves a *semaphore* that the caller can wait on when the result is needed

Between dispatch and the semaphore wait, the Tensix cores are free to do other compute. The ideal overlap looks like:

```
Async execution timeline (pipelined compute + communication):

  Tensix  ████████████████████████████████████████
           │ matmul[i] │ matmul[i+1] │ matmul[i+2]│
                    ↑                 ↑
               dispatch AG       dispatch RS
  ERISC    ░░░░████████████████████████████████░░░░
           │idle│      AllGather[i]     │ idle      │
                         │ ReduceScatter[i+1]       │

  ████ = active    ░ = idle
```

In this model, the ERISC and Tensix cores are rarely idle simultaneously. The theoretical throughput approaches `max(matmul_time, CCL_time)` rather than `matmul_time + CCL_time`.

### Conditions for overlap

Overlap is only achievable when:

1. **The communication does not depend on the immediately preceding compute result.** For example, AllGather on the previous layer's output can overlap with the current layer's matmul only if the all-gather result is not needed until that matmul completes.

2. **The output buffer for the async CCL is pre-allocated** (a *persistent output buffer*). Async ops cannot allocate their output dynamically because the runtime would need to wait for the allocation to be visible before dispatching. Pre-allocating pins the memory location.

3. **A SubDevice is used to separate the CCL workers from the compute workers**, so the CCL program's Tensix helper kernels do not interfere with the compute kernel's core reservations.

---

As covered in [Ch1 §1.2 — Hardware Topology](../ch1_introduction/hardware_topology.md#inter-chip-communication-erisc-and-edm), ERISC and Tensix cores are physically separate. Async CCL exploits this by running ERISC transfers concurrently with Tensix compute. Async CCL helper kernels (the reader/writer Tensix kernels that feed ERISC outbox buffers) occupy a reserved subset of Tensix cores — see SubDevice below.

---

## SubDevice: Partitioning Tensix Cores

A **SubDevice** is a named partition of Tensix cores within a single device. The sub_device_id parameter that appears in every CCL API tells the runtime which SubDevice partition should host the CCL helper kernels.

Without SubDevice:
```
All 120 Tensix cores → CCL helper kernels claim some
                      → matmul cannot start (core conflict)
                      → serialized
```

With SubDevice:
```
SubDevice 0: cores 0..7    → CCL helper kernels run here
SubDevice 1: cores 8..119  → matmul runs here
→ both run simultaneously
```

### Creating and using SubDevices

```python
import ttnn

# Reserve 8 cores for CCL helpers, leave the rest for compute
ccl_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
compute_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(11, 8))])

# Create two sub-devices on the mesh
sub_device_manager = mesh_device.create_sub_device_manager(
    [ttnn.SubDevice([ccl_cores]), ttnn.SubDevice([compute_cores])],
    local_l1_size=0
)
mesh_device.load_sub_device_manager(sub_device_manager)

# SubDeviceId(0) = CCL partition, SubDeviceId(1) = compute partition
ccl_sub_id   = ttnn.SubDeviceId(0)
compute_sub_id = ttnn.SubDeviceId(1)
```

Pass `subdevice_id=ccl_sub_id` to async CCL operations to confine their Tensix helpers to the CCL partition. The compute kernel dispatched to `compute_sub_id` runs concurrently.

> **Gotcha:** The SubDevice core layout must leave enough Tensix cores for the CCL helper kernels. AllGather async uses approximately `num_links * num_workers_per_link` Tensix cores per device. With `num_links=1` and the default `num_workers_per_link`, 4–8 cores is typically sufficient. Too few cores causes a validation failure at program creation time.

---

## Semaphores in Async Operations

Async ops communicate completion through `GlobalSemaphore` objects. Unlike the implicit semaphores used inside synchronous ops, these must be created and passed explicitly by the caller.

### Role of each semaphore

All async CCL ops take at least one `multi_device_global_semaphore` (sometimes a vector):

| Semaphore parameter | Purpose |
|---------------------|---------|
| `multi_device_global_semaphore` | Per-round synchronization: signals that a chunk of data has been received and is ready for the next step |
| `barrier_semaphore` | Global completion: signals that the entire collective is done — safe to read the output |
| `rs_global_semaphores` (AllReduce) | Semaphore set for the ReduceScatter phase |
| `ag_global_semaphores` (AllReduce) | Semaphore set for the AllGather phase |

### Creating semaphores for async ops

```python
import ttnn

# All participating cores on each device need a semaphore
# Typically use the same CoreRangeSet as the CCL SubDevice partition
semaphore_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])

# Create one semaphore per device (initialized to 0)
# Returns a GlobalSemaphore that is valid across all devices in the mesh
multi_device_sem = ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0)

# For AllReduce async, three separate semaphore sets are needed
barrier_sems = ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0)
rs_sems      = ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0)
ag_sems      = ttnn.create_global_semaphore(mesh_device, semaphore_cores, 0)
```

> **Gotcha:** Semaphores must be **reset** between calls if the same semaphore object is reused across inference iterations. Failing to reset causes the second invocation to see the semaphore already at its signaled value and skip the wait, producing incorrect results. Use `ttnn.reset_global_semaphore_value(semaphore, 0)` before each call, or create fresh semaphore objects per iteration (more allocation overhead).

---

## Persistent Output Buffers

Async ops often take a `persistent_output_buffer` (or `persistent_output_buffers`) parameter. This is a pre-allocated tensor that the async op writes its output into directly, bypassing runtime allocation.

Persistent buffers serve two purposes:

1. **Enable dispatch without host sync**: The runtime can dispatch the async program immediately because the output address is known at dispatch time.

2. **Avoid allocation jitter**: In steady-state inference, the same buffers are reused every iteration. This eliminates allocation overhead and prevents L1 fragmentation.

```python
# Allocate the output buffer once, before the inference loop
output_shape = list(input_shard.shape)
output_shape[3] *= num_devices   # AllGather dimension

persistent_buf = ttnn.allocate_tensor_on_device(
    ttnn.Shape(output_shape),
    ttnn.bfloat16,
    ttnn.TILE_LAYOUT,
    mesh_device,
    ttnn.L1_MEMORY_CONFIG,          # L1 for lowest latency
)

# Reuse every iteration
for _ in range(num_iterations):
    output = ttnn.experimental.all_gather_async(
        input_shard,
        persistent_output_buffer=persistent_buf,
        dim=3,
        multi_device_global_semaphore=multi_device_sem,
        topology=ttnn.Topology.Ring,
    )
    # output and persistent_buf refer to the same underlying L1 allocation
```

> **Gotcha:** The persistent buffer must not be written by any other operation while the async CCL is in flight. Reading from it before the `barrier_semaphore` signals completion produces undefined data.

---

## Sync vs Async: Decision Guide

Not all workloads benefit from async CCL. The decision depends on the ratio of compute time to CCL time and the complexity budget.

| Criterion | Use Synchronous | Use Async |
|-----------|----------------|-----------|
| Collective latency < 10% of total layer time | Yes | Marginal benefit |
| Collective latency ≥ 20% of total layer time | No | Strong benefit |
| Batch size is large (throughput-bound) | Often yes — compute dominates | Less critical |
| Batch size is 1 (latency-bound inference) | No | Yes — CCL is large fraction |
| Multi-layer pipeline with independent data | No | Yes — overlap across layers |
| Single collective with no adjacent compute | Yes — simpler | No benefit |
| Trace mode required | Either (see below) | Check Trace compatibility |
| First implementation / prototyping | Yes — simpler API | After profiling |

### Trace mode interaction

TTNN Trace mode pre-captures a sequence of Metal programs and replays them without host intervention — enabling the lowest possible dispatch latency. Async CCL operations are compatible with Trace mode, but with constraints:

1. **Semaphores must be pre-reset inside the trace**, or the trace must start with semaphores in their initial state. Do not rely on host-side `reset_global_semaphore_value` calls between trace replays without verifying that the reset is captured in the trace.

2. **Persistent buffers must be allocated before the trace is captured**. Allocation cannot happen inside a trace.

3. **Subdevice configuration must be fixed** before the trace begins. Changes to SubDevice managers invalidate the trace.

When these conditions are met, async CCL inside a trace achieves the lowest possible end-to-end latency for latency-critical inference serving.

---

*Back to [Chapter 4 Index](index.md)*

*Next: [4.2 Async Primitives](async_primitives.md)*
