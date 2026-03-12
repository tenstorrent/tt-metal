# §7.2 — SubDevice and Semaphores

This section explains the SubDevice mechanism for partitioning the Tensix core grid, how `subdevice_id` routes CCL operations to a specific partition, the full lifecycle of `GlobalSemaphore` objects, and how to safely overlap CCL with compute using async execution.

---

## SubDevice and subdevice_id

A SubDevice partitions the Tensix core grid so CCL and compute can be dispatched concurrently. For the concept, creation code (`create_sub_device_manager`, `load_sub_device_manager`, `get_sub_device_ids`), and partitioning diagram, see [Ch4 §4.1 — SubDevice: Partitioning Tensix Cores](../ch4_async_overlap/why_async.md#subdevice-partitioning-tensix-cores). This section focuses on `subdevice_id` parameter semantics and the common mistakes specific to advanced use.

## subdevice_id parameter

Most CCL ops accept an optional `subdevice_id` keyword argument:

```python
ttnn.all_gather(
    input_tensor,
    dim=1,
    subdevice_id=ccl_sub_id,   # SubDeviceId(0)
    num_links=1,
)

ttnn.matmul(
    a, b,
    subdevice_id=compute_sub_id,  # SubDeviceId(1)
)
```

When `subdevice_id` is provided:

1. The operation is dispatched only to cores within the designated SubDevice's core range.
2. The Metal program's program descriptor is associated with that SubDevice's command queue slot.
3. Program completion is signaled only when the SubDevice's cores finish — not when the entire device is idle.

When `subdevice_id=None` (the default), the operation uses all cores on the device and the dispatch system blocks on whole-device completion.

---

## GlobalSemaphore

`GlobalSemaphore` objects are created with `ttnn.create_global_semaphore(device, cores, initial_value)` and must be reset with `ttnn.reset_global_semaphore_value(sem, 0)` between iterations. For the full creation and reset pattern, see [Ch4 §4.1 — Semaphores in Async Operations](../ch4_async_overlap/why_async.md#semaphores-in-async-operations). The reset rules below apply regardless of which op is using the semaphore.

### How many semaphores does each op need?

| Op family | Semaphores needed | Roles |
|-----------|-------------------|-------|
| `all_gather` (sync) | 2 (internal) | `multidevice_semaphores` (ring) + `barrier_semaphore` |
| `all_gather_async` | 1 (caller provides) | Ring coordination semaphore |
| `all_reduce_async` | 3 (caller provides) | AllGather sems (2) + ReduceScatter sem (1) |
| `reduce_scatter_minimal_async` | 1 (caller provides) | Ring coordination semaphore |
| `all_gather_matmul_async` | 1 (caller provides) | Shared AG+MM coordination semaphore |
| `all_to_all_dispatch` | 2 (internal) | `init_semaphore` + `cross_device_semaphore` |
| `all_to_all_combine` | 2 (internal) | Same pattern as dispatch |

"Internal" semaphores are created automatically inside the op's program factory and do not need to be provided by the caller. "Caller provides" semaphores must be created once before the loop and reset each iteration.

---

## Multi-semaphore patterns

### AllReduce async: three semaphore sets

`all_reduce_async` is internally implemented as AllGather + local reduce + ReduceScatter, pipelined. It requires three semaphore objects: two for the AllGather ring (bidirectional) and one for the ReduceScatter ring.

```python
# Create once, before the training loop
ag_sem_a = ttnn.create_global_semaphore(mesh, ccl_cores, 0)
ag_sem_b = ttnn.create_global_semaphore(mesh, ccl_cores, 0)
rs_sem   = ttnn.create_global_semaphore(mesh, ccl_cores, 0)

for step in range(num_steps):
    output = ttnn.all_reduce_async(
        grad,
        ag_semaphore_a=ag_sem_a,
        ag_semaphore_b=ag_sem_b,
        rs_semaphore=rs_sem,
        subdevice_id=ccl_sub_id,
    )
    # ... compute ...

    # Reset all three semaphores before the next step
    ttnn.reset_global_semaphore_value(ag_sem_a, 0)
    ttnn.reset_global_semaphore_value(ag_sem_b, 0)
    ttnn.reset_global_semaphore_value(rs_sem, 0)
```

The three-semaphore design prevents a second ring traversal from interfering with the acks of the first when bidirectional pipeline stages overlap.

### Fused op (AllGather + Matmul): shared semaphore

In `all_gather_matmul_async` (see Ch5), one `GlobalSemaphore` (passed as `multi_device_global_semaphore`) serves as the FusedOpSignaler bridge. The AllGather kernel increments it after each chunk; the Matmul kernel waits on it before processing each chunk. This is the same semaphore object passed to both the CCL side and the compute side — the caller creates it once and provides it to both.

---

## Semaphore reset: when and how

### Timing rules

1. Reset **after** the op completes on all devices — use `ttnn.synchronize_device(mesh)` or equivalent barrier first if running async.
2. Reset **before** the next invocation begins.
3. Never reset **during** an op — the ERISC kernel may be mid-flight checking the semaphore value.

```
Correct order:
  step N:  dispatch CCL (async) → barrier → reset semaphore
  step N+1: dispatch CCL (async) → ...

Wrong order:
  step N:  dispatch CCL (async)
  step N+1: dispatch CCL without reset → HANG (semaphore never reaches expected value)
```

### Core range for reset

`reset_global_semaphore_value` writes the new value to the L1 address used by the semaphore. The write applies to all devices that own a copy of the semaphore (for `MeshDevice` semaphores, this is all devices). There is no need to reset per-device manually when using the `MeshDevice` creation API.

---

## Async CCL + SubDevice: overlapping CCL with compute

For the overlap timeline and dispatch pattern, see [Ch4 §4.3 — Pattern 4: SubDevice Dispatch](../ch4_async_overlap/overlap_patterns.md#pattern-4-subdevice-dispatch). The key rule is that both SubDevice ops must be dispatched before either is waited on — waiting between dispatches eliminates overlap.

---

## Common mistakes

**Mistake 1: Creating a new semaphore every iteration.**

`create_global_semaphore` allocates L1 memory. Creating one per iteration leaks L1 allocations and increases host-side overhead. Create once, reset between iterations.

**Mistake 2: Using SubDevice without matching core ranges to the CCL op's worker grid.**

If the `subdevice_id` routes the op to SubDevice 0 but the CCL op's internal worker core range (set via its device operation) includes cores outside SubDevice 0's `CoreRangeSet`, program dispatch fails with a core-out-of-range error. Use the `sub_core_grid` parameter (for `all_gather`) or `all_gather_core_grid_offset` / `reduce_scatter_core_grid_offset` (for fused ops, see Ch5) to keep CCL workers within the SubDevice's bounds.

---

*Back to [Chapter 7 Index](index.md)*
