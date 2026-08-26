# reader_argmax_interleaved_multicore.cpp — REVERTED (was migrated at API v14)

**Status:** `deferred` · reverted to the `llk_helper_library` baseline on 2026-08-26
**Unit:** `argmax-multicore-control` (kernel + `argmax_multi_core_program_factory.cpp`)
**Blocker flag:** `blocked:needs-pipe-semaphore-restore`

## History

1. Originally deferred on three v7-era design gaps (arbitrary-value monotone set, per-rect
   loopback modes in one send, mixed counters) — see git history of this file.
2. Migrated at API v14 (commit `5aaaf5b5aa5`, tier 2.14) once typed signals landed: two fixed
   no-handshake **Counter** control wires replacing the `start_sem` flag broadcast. Rectangle 0
   contains and excludes the reducer; rectangle 1 uses the reducer as a separate sender. The
   operation-owned `done_sem` worker-arrival counter stayed independent.
3. **Reverted 2026-08-26** during the rebase from `dc9282be7d5` onto `e6d0562cfaa`.

## Why it was reverted

Upstream added an end-of-kernel semaphore restore:

```cpp
if (is_reduce_core) {
    done_sem.set(0);
    start_sem.set(0);
    if constexpr (num_cores > 1) {
        start_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(noc, /* rect 0 */ ..., num_cores0);
        if (num_cores1 > 0) { start_sem.set_multicast(noc, /* rect 1 */ ..., num_cores1); }
        noc.async_write_barrier();
    }
}
```

Trace replay does not re-run the dispatcher's semaphore initialization. Without the restore the
previous run's final count is still present, the reducer's `k == 0` wait passes immediately, and it
collates stale partials — observed upstream as argmax indices from the wrong core's slice on the
first outer row.

Git auto-merged that new tail onto the migrated kernel **with no textual conflict**, because the
migration had deleted the `start_sem` declaration and the eight rectangle-coordinate CT args the
tail refers to. Result: a kernel that does not compile.

```
reader_argmax_interleaved_multicore.cpp:523:9: error: 'start_sem' was not declared in this scope
reader_argmax_interleaved_multicore.cpp:526:22: error: 'start_core_x0' was not declared in this scope
... (+8 more: start_core_y0, end_core_x0/y0, start_core_x1/y1, end_core_x1/y1, num_cores1)
TT_THROW: ncrisc build failed
```

Deleting the tail was rejected: **upstream's defect applies to the migrated form too.** A Counter is
monotone and reset-free *by design*, and `ReceiverPipe`'s round counter is kernel-local, so on a
replayed trace the `data_ready` semaphore keeps its final count while every receiver restarts at
round 0 — the same staleness class. The migration simply predates the fix and carries the latent bug.

## What is needed to migrate it again

**The gap:** v14 gives a sender no way to restore its `data_ready` semaphore across the multicast
rectangle. `send_signal()` advances the wire; nothing rewinds it.

A by-hand reset *is* expressible today — `McastArgs::rect()` exposes the rectangle and
`McastArgs::data_ready` the semaphore id — but it reintroduces raw `set_multicast` into a migrated
kernel and therefore violates the rollout's own audit invariant
(`test_mcast_pipe_source_audit.py` asserts `set_multicast` is absent from migrated kernels). Do not
take that shortcut to close this entry.

Pick one of:

1. **Add a sender-side restore to the helper** (preferred). Something in the shape of
   `SenderPipe::reset_signal()` / `restore()` that writes the signal cell's initial value to every
   receiver cell over the same rectangle `send_signal()` uses, honouring the same loopback mode.
   This is an API addition and a version bump — a `tune-dm-helper` decision, **not** a rebase or
   apply-time edit. Note argmax needs it on **two disjoint rectangles** that share one semaphore, so
   the operation must be callable per-wire.
2. **Make Counter readiness replay-safe** by having the receiver derive its expectation from the
   absolute semaphore value rather than a kernel-local round, so a leftover count cannot be misread.
   This changes the Counter contract itself and affects every Counter call site — strictly bigger.

**Re-verification required when it is retried:**

- `tests/ttnn/unit_tests/operations/reduce/test_argmax.py` — full file, fresh JIT and warm cache.
- `test_argmax_multicore_two_rectangles_cached[dim=-1|None]` specifically: it runs argmax **twice**,
  so the second launch hits the cached program and is the cheapest reproduction of leftover
  semaphore state. (Retained in the test file through the revert precisely for this reason; the
  helper-specific env-gated perf test was removed with the migration.)
- A trace-replay path, which is the actual regression upstream fixed and which the cached-program
  test only approximates.
- `--dev` / Watcher on the two-rectangle reduce-all route, plus the source audit.

## Related

The audit assertion `test_argmax_multicore_composes_two_counter_wires_and_keeps_done_fanin` was
removed with this revert; restore it together with the migration.
