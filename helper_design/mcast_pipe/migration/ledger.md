# `mcast_pipe` migration ledger — inventory reconciled 2026-08-16

Machine source of truth: `ledger.json`. Test dispatch is in `test_map.json`; per-unit evidence is in
`log/`. The current static audit is archived at
`../archive/reconciliation/reconcile_2026-08-16-plan-inventory.md`; the preceding rebase audit is
`../archive/reconciliation/reconcile_2026-08-14-rebase-dc9282.md`.

- Branch: `sjovic/mcast-migration`; approved rollout plan materialized at `830190c9721`.
- Baseline: `origin/llk_helper_library` at `dc9282be7d5`.
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
| deferred | 84 | 0 |
| quarantined | 0 | 0 |

All 104 inventoried kernel paths exist. No migrated kernel was removed, renamed, or clobbered. Twelve
migrated kernels changed across the baseline move or conflict-composed rebase and carry
`needs_recheck`. The approved plan audit added 13 previously omitted call-site/receiver companions: two
Matmul Decode two-hub readers, three programming/lab example receivers, four Quasar Matmul receivers,
and four Quasar Conv receivers. The production and Quasar `conv_reader_common.hpp` files are recorded as
atomic-scope support dependencies, not false call-site rows. Deferred factories are mapped in the reconcile
report; `host_bindings` retains its convention of migrated or source-integrated pending bindings only.

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

| Unit | Kernels | Reason |
|---|---|---|
| Matmul in1 | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`, `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | baseline DFB/global-CB churn plus conflict-composed geometry and naming changes |
| GroupNorm v2 | legacy sender/receiver and Welford sender/receiver | baseline DFB/fp32 changes plus helper/control-ABI conflict composition |
| LayerNorm pre-allgather | sender and receiver | baseline runtime-argument vector changes composed with the helper prefix |
| TopK | `reader_final_topk.cpp`, `writer_local_topk.cpp` | baseline DFB changes composed with helper-owned readiness |
| Sort | `coordinator_single_row_multi_core.cpp`, `reader_single_row_multi_core.cpp` | baseline UInt16 and partial-grid hang fixes composed with the split helper channels |

`apply-dm-helper` must run their complete mapped verify-only coverage and clear the flags when green.
The enclosing rebase workflow already passed focused post-rebase probes, but reconciliation keeps the
flags until the complete mapped inventories are recorded by the apply workflow.

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

Eighty-four entries remain deferred. Their exact reasons and flags are authoritative in `ledger.json`.
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
2. clear the 12 `needs_recheck` flags through complete mapped verify-only coverage;
3. validate and write back the two interleaved Matmul kernels and five bindings;
4. validate and write back the block-sharded Matmul kernel and four bindings;
5. update logs and the live report after each atomic unit completes.
