# writer_single_row_multi_core.cpp — DEFERRED (helper-neutral companion)

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

Tier 5 atomic unit: `sort-single-row-control`. Coupled ABI cleanup: `7337302b564`.

## Disposition

The writer is not a Pipe caller. It emits the operation-owned writer-done counter after each
processed pair; the coordinator waits and resets that counter independently from the helper-owned
coordinator-to-reader control channel. Counting this file as migrated would overstate helper reach.

The atomic production change removed dead doorbell and semaphore-ID runtime words, reducing the
writer runtime block from six words to four. Its done semaphore is constexpr ID 2. The functional
counter leg remains raw/object API by design.

## Coupled validation

- Host build passed.
- Exact fresh-cache `--dev` long-tensor node passed with the writer JIT artifact confirmed.
- Ht=2 deadlock regression: 2 passed.
- Full long-tensor inventory: 7 passed.

Standalone helper status remains deferred; no API gap is claimed.
