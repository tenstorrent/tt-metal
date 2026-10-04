---
description: 'PR review rules for device-kernel NoC operations and helper code'
applyTo: '**/kernels/**,**/kernel_common/**,tt_metal/hw/inc/api/dataflow/**,tt_metal/hw/inc/experimental/**,tt_metal/fabric/hw/**,tt_metal/fabric/impl/kernels/**'
excludeAgent: "cloud-agent"
---

# Device-Kernel NoC Review

## 🔴 CRITICAL

- **Drain non-posted atomics before kernel exit**: a finite device kernel must drain each NoC on which it issued a non-posted atomic transaction. A matching `noc_async_atomic_barrier()` or `noc_async_full_barrier()` must execute after the last possible atomic issue and before every return from `kernel_main()`.
- **Trace helper calls**: inspect callees, wrappers, object methods, and included headers. Do not limit the review to raw atomic calls in the changed kernel. Treat `noc_semaphore_inc()`, `noc_semaphore_inc_multicast()`, non-posted `noc_fast_atomic_*` calls, remote semaphore methods such as `Semaphore::up(noc, ...)` and `inc_multicast()`, remote circular-buffer credit updates, and every helper that can reach them as atomic issuers.
- **Check every control-flow path**: verify sender and receiver roles, master and worker roles, coordinator and non-coordinator roles, early returns, compile-time branches, and feature macros. A barrier under one feature macro does not protect a path where that macro is disabled.
- **Match the NoC**: the barrier must drain the same NoC that issued the atomic transaction. A barrier on NoC 0 does not drain an atomic transaction issued on NoC 1.
- **Place the barrier after the final issue**: a barrier before the final atomic transaction does not protect kernel exit. A callee barrier is sufficient only when no later caller or callee can issue another atomic transaction on that NoC.
- **Do not substitute unrelated synchronization**: write barriers, read barriers, `noc_async_writes_flushed()`, local semaphore waits, destination-side acknowledgements, and evidence that the remote core observed the semaphore do not drain the source core's non-posted atomic acknowledgement queue.
- **Verify posted semantics per architecture**: exempt a posted atomic only after the implementation for every target architecture confirms that it remains posted. Blackhole currently forces low-level atomic increments to non-posted mode even when the caller requests posted mode.
- **Make helper ownership explicit**: if a reusable helper can issue a non-posted atomic but does not drain it, its contract must assign the final drain to the caller. Review all changed callers and all exit paths when this contract changes.

Do not report a violation only because an atomic call and barrier are in different files. Review the complete call path. Do not treat local semaphore updates, such as `Semaphore::up(value)` or `noc_semaphore_set()`, as remote atomic transactions.

## 🟡 IMPORTANT

- **Add a branch-specific regression test**: a fix must execute the affected helper, role, compile-time branch, and kernel exit path. A nearby test that does not select that path is not sufficient.
- **Use the NoC debug dump when possible**: in a Tracy-enabled build, run the affected test with `TT_METAL_NOC_DEBUG_DUMP=1`. It records instrumented semaphore atomic issues and their matching atomic or full barriers. Its pending-atomic state clears only on a recorded atomic or full barrier, not when enough time passes for the hardware acknowledgement to arrive. Use Watcher as hardware confirmation. A clean Watcher run alone does not prove that the barrier exists because the acknowledgement can arrive before the kernel epilogue check.
- **Cover uninstrumented low-level calls separately**: if a raw low-level atomic bypasses the NoC event instrumentation, require a targeted check or Watcher coverage in addition to source inspection.

## Review Checklist

- [ ] All direct and helper-issued remote atomics are identified
- [ ] Each issuing path drains the same NoC after its final atomic transaction
- [ ] Every finite `kernel_main()` exit is protected
- [ ] All roles, early returns, compile-time branches, and feature macros are checked
- [ ] Posted-mode assumptions are valid on every target architecture
- [ ] The regression test executes the affected path and uses NoC debug-dump or Watcher coverage where practical
