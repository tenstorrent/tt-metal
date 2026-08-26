# reader_single_row_multi_core.cpp — MIGRATED API v10

> **REVERTED 2026-08-26** — during the rebase from `dc9282be7d5` onto `e6d0562cfaa`.
> Upstream Metal-2.0-ported the whole `SortProgramFactorySingleRowMultiCore` (#52528): positional
> `get_arg_val` / `get_compile_time_arg_val` became declarative named args via
> `experimental/kernel_args.h` (`get_arg(args::name)`), and the factory now builds `SemaphoreSpec` /
> `TensorParameter` / `DFBBinding` groups. This migration was authored against the old positional
> API, so bringing it back is a **re-authoring, not a conflict resolution** — the `McastArgs` decoder
> currently assumes positional CT/RT slots.
> Ledger status is `deferred`, flag `blocked:needs-metal2-named-args`. Re-migrate once the rollout
> supports the Metal 2.0 named-argument surface. The audit assertion
> `test_sort_row_start_readiness_is_pipe_owned` was removed with the revert; restore it with the
> migration.

Tier 5 atomic unit: `sort-single-row-control`. Code: `7337302b564`.

## Migrated role

The reader constructs two receiver faces from chained `Mcast2D` wires. Handshaked
`receive_signal()` acknowledges readiness before waiting for row start; the separate no-handshake
Pipe receives sub-stage events. Both replace the old inverted level doorbell and raw row-ready path.

The helper owns data-ready IDs 0/2 and row-ready ID 1. The writer-done counter remains op-owned at ID
3. The old runtime semaphore IDs and level reset are removed.

## Validation

- Host build passed.
- Exact fresh-cache `--dev` long-tensor node passed with this reader JIT artifact confirmed.
- Ht=2 deadlock regression: 2 passed.
- Full long-tensor inventory: 7 passed.
- Helper suite: 77 passed.
- Three-run performance median: +1.195124% versus baseline, within the 1.5% gate.

Helper API is v10.
