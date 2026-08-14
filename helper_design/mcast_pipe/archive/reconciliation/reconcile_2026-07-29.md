# Archived: Reconcile report — mcast_pipe API v9 — updated 2026-07-30

## Why this reconcile exists

The July branch recorded 19 production kernels as migrated at API v8. The
target branch has since advanced to API v9 and contains substantial TT-Metal
and kernel changes. The migrations were therefore ported semantically onto
`origin/llk_helper_library` at `54d8dfb7bef`, without rebasing.

## Reconciliation rules

- Preserve the current kernel structure and host argument layouts.
- Do not retain a migration that leaves pipe-owned multicast handshake
  operations raw.
- Require one exact compile-focused parameter before the complete mapped unit
  inventory.
- Require exact JIT-path evidence for every promoted kernel.
- Restore unsupported kernels exactly to the target baseline.

## Reconciled result through `307951cc8dc`

| Current production candidate state | Count |
|---|---:|
| migrated and fully validated | 10 |
| deferred/rolled back; baseline restored | 12 |

The current 22-row production set is larger than the 19 source migrations
because remediation added the previously raw Conv 1D sender and both
GroupNorm v2 senders so each protocol pair is tracked atomically.

The current migrated rows are the matmul in1 pair, both fixed Conv
weights-multicast pairs, and both GroupNorm v2 sender/receiver pairs.

The post-allgather LayerNorm pair was restored on 2026-07-30. Its baseline
protocol requires INCLUDE-source loopback even when the sender is outside the
receiver rectangle, and its explicit host fan-out may differ from the
rectangle area. The baseline-restored pair passed all 136 mapped cases, but
that inventory exercises only `mcast_1d` and therefore validates the rollback,
not the helper migration.

`SenderPipe` now waits for ACKed completion when it emits a real in-rectangle
loopback copy. Remote-only sends still stop at SENT because the receiver flag
wait proves arrival. This internal change keeps API version 9 and is covered by
the 68-case helper suite, including a same-core compute consumer.

The deferred production rows require one or more of:

- typed control values;
- acknowledged signal-only traffic;
- one-gate/multi-block mixed-mode streaming;
- race-free no-handshake initialization;
- independent data and signal loopback behavior;
- explicit out-of-rectangle loopback; and
- explicit fan-out independent of rectangle area, or a checked dense geometry.

The width-sharded Conv reader remains baseline-restored pending a fresh
re-port/retest. Its earlier 25 numeric failures predated ACKed loopback
completion and are not evidence against the current helper.

## Current sources of truth

- Machine-readable status and flags: `ledger.json`
- Human status mirror: `ledger.md`
- Complete unit-test inventory: `test_map.json`
- Execution evidence: `log/*.md`
- Concise rollout result: `report.md`

The 2026-06 reconcile files and original per-kernel log bodies describe the v8
rollout and must not be used as current branch status. Current-status banners
on those logs take precedence over their historical text.
