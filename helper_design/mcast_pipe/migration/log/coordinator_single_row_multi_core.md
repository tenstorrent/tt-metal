# coordinator_single_row_multi_core.cpp — MIGRATED API v10

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

The coordinator constructs two sender faces over the dense full worker grid. The handshaked Counter
Pipe publishes row start only after every reader acknowledges readiness; a separate no-handshake
Counter Pipe publishes sub-stage events, preserving back-to-back events without requiring an ACK.

Reader readiness is now helper-owned. The writer-done counter remains independent operation protocol
because it counts completed compare/write pairs rather than receiver availability.

Runtime args fell from 11 to 7. The host owns the multicast rectangle and sender-coordinate wire.

## Validation

- `./build_metal.sh`: passed.
- Exact `test_sort_long_tensor[shape=[1, 524288]-dim=-1-descending=False]` under `--dev` from a fresh
  isolated JIT cache: passed; coordinator, reader, and writer artifacts confirmed.
- `test_sort_multi_row_multi_core_no_deadlock`: 2 passed (`descending=False/True`, `Ht=2`).
- Complete `test_sort_long_tensor`: 7 passed.
- Complete `test_mcast_pipe.py`: 77 passed, including handshaked and no-handshake control-only cells.
- Three-run performance median: +1.195124% versus baseline, within the 1.5% gate.

Helper API is v10.
