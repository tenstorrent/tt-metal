# `mcast_pipe` migration ledger — reconciled after experimental rollback 2026-08-14

Machine source of truth: `ledger.json`. Test dispatch is in `test_map.json`; per-unit evidence is in
`log/`. The last pre-rollback static audit is archived at
`../archive/reconciliation/reconcile_2026-08-14.md`; Round 29 of `../changelog.md` records the rollback.

- Branch: `sjovic/mcast-migration` at `9d870bf2da9` after rollback.
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`.
- Ledger API: v10.
- Materialized helper API: v11.

The version mismatch is intentional at this checkpoint. `reconcile-dm-helper` aligned the ledger
inventory with the source tree but does not perform Tier-0 device validation or API-version write-back
owned by `apply-dm-helper`.

## Current paper state

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated at ledger API v10 | 17 | 14 |
| pending | 3 | 9 |
| deferred | 71 | 0 |
| quarantined | 0 | 0 |

All 91 inventoried kernel paths exist. Before the separate text inventory was removed, its path set
matched these entries exactly. No migrated kernel was removed, renamed, or clobbered. Two migrated
kernels were edited after the last ledger write-back and carry `needs_recheck`.

## Migrated units awaiting API-v11 re-entry

| Unit | Kernels | Bindings | Existing evidence |
|---|---:|---:|---|
| `conv2d-weights-single-sender-rect` | 2 | 1 | Conv height inventory, DRAM routes, host/helper tests |
| `conv2d-weights-fixed-line` | 2 | 1 | Conv block inventory, PerRow/PerColumn, DRAM routes |
| `matmul-in1-mcast-padding-host` | 2 | 4 | `MM-IN1-ALL` 302 passed / 188 expected skips |
| `groupnorm-sharded-v2-mcast-host` | 4 | 4 | legacy/Welford inventories and matched performance |
| `sort-single-row-control` | 2 | 1 | exact JIT, long 7/7, deadlock 2/2, matched performance |
| `conv2d-activation-width-sharded` | 1 | 1 | exact JIT, features 48/16, DRAM route |
| `topk-multicore-final-readiness` | 2 | 1 | exact JIT, 14 passed / 12 expected xfails |
| `layernorm-sharded-pre-allgather` | 2 | 1 | pre 126, post 136, sharded 208 |

These rows remain stamped v10 until the API-v11 apply gate is green.

## Open `needs_recheck`

| Kernel | Reason |
|---|---|
| `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | Matmul remediation and multicast naming cleanup changed the migrated source |
| `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | Multicast naming cleanup changed the migrated source |

`apply-dm-helper` must run their mapped verify-only coverage and clear the flags when green. API-v11
Tier-0 coverage may satisfy both obligations when it exercises the same complete inventories.

## Source-integrated pending work

| Area | Kernel/binding rows | Remaining work |
|---|---|---|
| Matmul in0 interleaved | sender and receiver; five host bindings | API-v11 validation, exact fresh-JIT evidence, complete mapped inventories, performance evidence, ledger write-back |
| Matmul in0 block-sharded | hybrid reader; four legacy/descriptor 1D/2D bindings | Validate rotating sender topology and the exact block-sharded routes, then write back atomically |
The Matmul API-007 and block-sharded topology blockers are resolved in source. Pending status is
retained because reconciliation does not substitute source inspection for build/device evidence.
Block-sharded Conv activation is deferred: its producer-overlapped streaming multicast remains the R4
design gap and continues to use the established raw primitive path.

## Deferred backlog

Seventy-one entries remain deferred. Their exact reasons and flags are authoritative in `ledger.json`.
The major classes are:

- genuine capability gaps such as chain relay, runtime role/count, and multi-phase protocols;
- coverage gaps or binary-only routes;
- helper-neutral or non-multicast entries retained for atomic-unit context;
- experimental/CCL/Quasar entries intentionally deferred as groups.

Before changing the helper for any deferred kernel, state the required behavior independently of its
current implementation and verify that the existing helper cannot express it through a different
factory, ABI, channel split, or data flow.

## Next action

Run `apply-dm-helper` from the reconciled state:

1. validate and stamp the v10 fleet at API v11;
2. clear the two `needs_recheck` flags through mapped verify-only coverage;
3. validate and write back the two interleaved Matmul kernels and five bindings;
4. validate and write back the block-sharded Matmul kernel and four bindings;
5. update logs and the live report after each atomic unit completes.
