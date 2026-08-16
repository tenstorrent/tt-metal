# `mcast_pipe` migration ledger — inventory reconciled 2026-08-16

Machine source of truth: `ledger.json`. Test dispatch is in `test_map.json`; per-unit evidence is in
`log/`. The current static audit is archived at
`../archive/reconciliation/reconcile_2026-08-16-plan-inventory.md`; the preceding rebase audit is
`../archive/reconciliation/reconcile_2026-08-14-rebase-dc9282.md`.

- Branch: `sjovic/mcast-migration`; approved rollout plan materialized at `830190c9721`.
- Baseline: `origin/llk_helper_library` at `dc9282be7d5`.
- Ledger API: v11.
- Materialized helper API: v11.

Tier 0 API-v11 verification/write-back completed on 2026-08-16. Tier 0.1 remains pending only because
its plan-mandated historical matched performance baseline requires a separately authorized checkout.

## Current paper state

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated at ledger API v11 | 18 | 18 |
| pending | 2 | 5 |
| deferred | 84 | 0 |
| quarantined | 0 | 0 |

All 104 inventoried kernel paths exist. No migrated kernel was removed, renamed, or clobbered. The 12
post-rebase `needs_recheck` flags were cleared after complete mapped API-v11 verification. The approved
plan audit added 13 previously omitted call-site/receiver companions: two
Matmul Decode two-hub readers, three programming/lab example receivers, four Quasar Matmul receivers,
and four Quasar Conv receivers. The production and Quasar `conv_reader_common.hpp` files are recorded as
atomic-scope support dependencies, not false call-site rows. Deferred factories are mapped in the reconcile
report; `host_bindings` retains its convention of migrated or source-integrated pending bindings only.

## API-v11 verified migrated units

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

These units are stamped at API v11. Tier 0.2 `matmul-in0-mcast-block-sharded` is also migrated at v11:
its exact zero-hit-cache probe, complete mapped inventory, and inherited matched performance evidence
passed on 2026-08-16.

## Closed `needs_recheck`

| Unit | Kernels | Reason |
|---|---|---|
| Matmul in1 | `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`, `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | baseline DFB/global-CB churn plus conflict-composed geometry and naming changes |
| GroupNorm v2 | legacy sender/receiver and Welford sender/receiver | baseline DFB/fp32 changes plus helper/control-ABI conflict composition |
| LayerNorm pre-allgather | sender and receiver | baseline runtime-argument vector changes composed with the helper prefix |
| TopK | `reader_final_topk.cpp`, `writer_local_topk.cpp` | baseline DFB changes composed with helper-owned readiness |
| Sort | `coordinator_single_row_multi_core.cpp`, `reader_single_row_multi_core.cpp` | baseline UInt16 and partial-grid hang fixes composed with the split helper channels |

All 12 flags are cleared. The 2026-08-16 apply verification passed the build, exact route probes,
complete mapped operation inventories, 80 helper device tests, 17 source-audit tests, and 32 host
fixture tests. Claude C3 independently approved the write-back.

## Source-integrated pending work

| Area | Kernel/binding rows | Remaining work |
|---|---|---|
| Matmul in0 interleaved | sender and receiver; five host bindings | Route-specific historical matched performance baseline and ledger write-back |
The Matmul API-007 and block-sharded topology blockers are resolved in source. Interleaved pending
status is retained because the approved plan requires a matched pre-unit baseline at `45033178088b`,
and the separately gated historical checkout was not authorized.
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

Proceed with approved Tier 1 unit 6. Keep the two interleaved Matmul kernels and five bindings pending
until a future instruction separately authorizes the historical performance checkout.
