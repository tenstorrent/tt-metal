# mcast_pipe rollout report — API v9, host rollout updated 2026-07-30

## Run header

- Source migration branch: `origin/sjovic/mcast-helpers-july`
- Source migration commit: `acafdfcc6c4`
- Current branch baseline: `origin/llk_helper_library` at `54d8dfb7bef`
- Current helper/rollback commit: `307951cc8dc`
- Host rollout plan commit: `bb66c3a25fc`
- Conv2D height-sharded host migration commit: `75b977e1a04`
- Helper contract: `MCAST_PIPE_API_VERSION 9`
- Entry mode: re-entry
- Invocation mode: `run-all`
- Selected unit approved at Gate A: `conv2d-weights-single-sender-rect`
- Re-entry breakdown: 0 stale kernels, 0 stale host bindings, 10 newly
  inventoried host integrations across 4 atomic units, 0 net-new kernel units
- Device validation: single-chip Blackhole p100a
- Test runner: repository environment plus `scripts/run_safe_pytest.sh`

## Outcome

The selected Conv2D height-sharded **weights binding** is fully current end to
end at v9. This status applies to that protocol channel, not to every Conv2D
multicast path. Height-sharded Conv2D reads activations locally from each input
shard and has no activation-multicast binding. No selected unit failed or was
quarantined. The other three host-integration units were outside the
user-approved scope for this run and remain pending.

| Rollout state at v9 | Count |
|---|---:|
| kernel-current | 10 |
| host-binding-current | 1 |
| fully end-to-end current | 1 binding / 2 kernels |
| host-pending | 9 |
| quarantined | 0 |
| deferred | 82 kernels / 0 host bindings |

Every deferred production kernel is byte-for-byte equal to the confirmed
target baseline. Every migrated multicast channel delegates its multicast
semaphore ownership to `SenderPipe` / `ReceiverPipe`. The raw
`reserve_done_sem` and `write_done_sem` instances remaining in the 2D Conv
pair coordinate a separate split-reader circular-buffer protocol.

The migrated Conv2D host binding constructs the full height-sharded rectangle
with `Mcast2D` while preserving `total_active_num_cores - 1` as the acknowledged
receiver subset. Its sender and receiver kernels consume the helper-owned
five-word compile-time and four-word runtime wire through `McastArgs`.

## This run by tier

| Tier | Migrated | Failed/quarantined | Skipped/deferred |
|---|---:|---:|---:|
| 1 — Conv2D height-sharded host binding | 1 | 0 | 0 |
| Selected-scope total | 1 | 0 | 0 |

Nine required host bindings across the Conv2D fixed-line, matmul in1, and
GroupNorm v2 units remain pending outside this selected scope. They are not
counted as failures or deferrals. Conv2D activation multicast is used by the
block- and width-sharded paths; those activation kernels remain deferred at the
kernel-helper stage and therefore are not part of the host-binding worklist.

## Per-kernel and binding result

| Site | Status | Validation | Production deletions | Perf |
|---|---|---|---:|---|
| Conv2D 1D weights sender | migrated, fully end-to-end | exact JIT hit; height and shared regressions passed | 19 | not measured |
| Conv2D 1D weights receiver | migrated, fully end-to-end | exact JIT hit; height and shared regressions passed | 11 | not measured |
| Height-sharded factory binding | migrated at v9 | host build, host oracle, and device wire passed | 21 | not measured |

The atomic production change contains 48 insertions and 51 deletions, for a net
reduction of 3 lines. The factory refactor preserves the full multicast
rectangle, the smaller active ACK subset, buffer bindings, activation-reuse
offsets, and the `SKIP_MCAST` path. No in-context performance run was requested,
so this report makes no performance-delta claim.

## Coverage gaps

None for the selected unit. Both participating kernels were observed in an
isolated JIT cache, and focused plus regression validation passed.

## Validation headline

- Helper baseline: 68 passed.
- Post-allgather LayerNorm rollback: the baseline-restored pair passed all 136
  mapped cases on 2026-07-30. This inventory exercises only `mcast_1d`, so it
  validates the rollback but does not unblock the helper migration for the
  accepted non-1D sender geometry.
- GroupNorm legacy: 108 passed, 2 expected skips; fixed/default routing: 19 passed, 6 expected skips.
- GroupNorm Welford: 108 passed, 2 expected skips; fixed/default routing shares the same 19 passed, 6 expected skips.
- Matmul in1 inventory: 302 passed, 188 expected skips.
- Conv height direct inventory after the host migration: 49 passed, 16 expected
  skips. The exact compile-focused case passed under `--dev` and produced both
  sender and receiver JIT artifacts.
- Conv block direct inventory: 49 passed, 16 expected skips.
- Conv shared DRAM regressions after the host migration: 14 passed.
- Host helper tests after the host migration: 19 passed.
- Device helper-wire tests after the host migration: 68 passed.
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
