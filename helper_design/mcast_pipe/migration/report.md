# mcast_pipe rollout report — API v9, reconciled 2026-07-30

## Run header

- Source migration branch: `origin/sjovic/mcast-helpers-july`
- Source migration commit: `acafdfcc6c4`
- Current branch baseline: `origin/llk_helper_library` at `54d8dfb7bef`
- Current helper/rollback commit: `307951cc8dc`
- Helper contract: `MCAST_PIPE_API_VERSION 9`
- Device validation: single-chip Blackhole p100a
- Test runner: repository environment plus `scripts/run_safe_pytest.sh`

## Outcome

The current production ledger was semantically reconsidered rather than
rebased. Ten are current v9 migrations. Twelve were either rejected before edit
or reverted after review/device isolation because the current helper cannot
own their entire multicast protocol.

| State | Count |
|---|---:|
| current migrated | 10 |
| deferred on a documented blocker | 12 |
| partial migrations left in the worktree | 0 |

Every deferred production kernel is byte-for-byte equal to the confirmed
target baseline. Every migrated multicast channel delegates its multicast
semaphore ownership to `SenderPipe` / `ReceiverPipe`. The raw
`reserve_done_sem` and `write_done_sem` instances remaining in the 2D Conv
pair coordinate a separate split-reader circular-buffer protocol.

## Validation headline

- Helper baseline: 68 passed.
- Post-allgather LayerNorm rollback: the baseline-restored pair passed all 136
  mapped cases on 2026-07-30. This inventory exercises only `mcast_1d`, so it
  validates the rollback but does not unblock the helper migration for the
  accepted non-1D sender geometry.
- GroupNorm legacy: 108 passed, 2 expected skips; fixed/default routing: 19 passed, 6 expected skips.
- GroupNorm Welford: 108 passed, 2 expected skips; fixed/default routing shares the same 19 passed, 6 expected skips.
- Matmul in1 inventory: 302 passed, 188 expected skips.
- Conv height direct inventory: 49 passed, 16 expected skips (including the migrated 1D sender).
- Conv block direct inventory: 49 passed, 16 expected skips.
- Conv shared DRAM regressions: 14 passed.
- Host rebuild after Conv factory changes: passed.

The rotating width-sharded Conv migration passed its initial smoke but failed
25 of the full 65 selected cases numerically. It was restored to baseline and
is not counted as migrated. The rotating block-sharded matmul migration also
failed an exact numeric regression that passed after restoring the baseline.

## Missing helper capabilities

1. Acknowledged signal-only send/receive.
2. One-gate, multi-block mixed flag/counter streaming.
3. Race-free host-owned initialization for no-handshake flag receivers.
4. Typed/custom control values such as `IGNORE_BATCH`.
5. Independent data and signal loopback modes.
6. Explicit INCLUDE-source loopback independent of rectangle membership.
7. Explicit multicast fan-out when host population differs from rectangle
   area, or a checked dense-rectangle invariant.

Detailed semantics and failed/passing test evidence are recorded in
`ledger.json`, `test_map.json`, `tiers.md`, and the per-kernel migration logs.

## Documentation state

The skill-owned `helper_design/mcast_pipe` documentation tree is present and
reconciled through 2026-07-30. The current tree contains 128 artifacts,
including two host-I/O annotations renamed to their live kernels. The following
are current:

- `migration/ledger.json`
- `migration/ledger.md`
- `migration/report.md`
- `migration/reconcile_2026-07-29.md`
- `migration/test_map.json`
- `migration/tiers.md`
- `migration/log/*.md`

`test_map.json` owns the complete definitions behind production inventory
labels, compile/JIT coverage rules, reusable routes for additional
single-device census candidates, and the known multi-device or binary-only
coverage gaps.

The 2026-06-19 and 2026-06-27 reconcile reports, original per-kernel logs, and
pre-migration audits remain historical records. Current-status banners on
historical files take precedence over their retained execution text.
