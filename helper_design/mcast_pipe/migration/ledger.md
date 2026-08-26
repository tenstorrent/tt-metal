# `mcast_pipe` migration ledger — rollout state updated 2026-08-23

Machine source of truth: `ledger.json`. Test dispatch is in `test_map.json`; per-unit evidence is in
`log/`. The current static audit is archived at
`../archive/reconciliation/reconcile_2026-08-23-document-audit.md`; the preceding plan-inventory and
rebase audits are `../archive/reconciliation/reconcile_2026-08-16-plan-inventory.md` and
`../archive/reconciliation/reconcile_2026-08-14-rebase-dc9282.md`.

- Branch: `sjovic/mcast-migration`; feedback intake head `cea14afbea9`.
- Baseline: `origin/llk_helper_library` at `e6d0562cfaa` (rebased 2026-08-26; was `dc9282be7d5`).
- Ledger API: v14.
- Materialized helper API: v14.

The 2026-08-23 feedback pass verified and wrote back the existing migrated fleet
at API v14. It did not change the pending/deferred inventory dispositions.

## Current paper state

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated at ledger API v14 | 31 | 27 |
| pending | 2 | 5 |
| deferred | 75 | 0 |
| quarantined | 0 | 0 |

All 108 inventoried kernel paths exist. No migrated kernel was removed, renamed, or clobbered. The 12
post-rebase `needs_recheck` flags were cleared after complete mapped API-v11 verification. The approved
plan audit added 13 previously omitted call-site/receiver companions: two
Matmul Decode two-hub readers, three programming/lab example receivers, four Quasar Matmul receivers,
and four Quasar Conv receivers. The production and Quasar `conv_reader_common.hpp` files are recorded as
atomic-scope support dependencies, not false call-site rows. Deferred factories are mapped in the reconcile
report; `host_bindings` retains its convention of migrated or source-integrated pending bindings only.

API-v14 verification removed the dynamic runtime-base constructor and made the
`RT_BASE` template argument the only source of truth. Five layouts with
runtime-sized operation data and one layout with a genuinely optional
compile-time tail now place those tails after the opaque helper and derive their
starts from the corresponding next offset; all matching producers use the same
order. The final Conv feedback pass also restored original terminal drains,
removed migration-only source-lifetime synchronization, clarified independent
input ownership, and audited dense versus divergent ACK populations. Every
migrated-kernel offset chain now uses a named constexpr helper object, and
aliases remain only where nested pipe types need them;
static API and wire semantics are unchanged. The host build, 36/36 host gtests,
80/80 helper device tests under `--dev`, 33/33 source
audits, and sequential focused Matmul, Conv2D, Conv3D, Move, GroupNorm, and
LayerNorm gates passed. The
API-v13 tagged optional ABI and its present/absent/chained coverage remain
intact. Detailed evidence, known Watcher exceptions, and the isolated unrelated
BFLOAT16 block-Conv hang are recorded in
`../migration_feedback_tracker.md`.

## API-v14 verified migrated units

> **Rebase 2026-08-26 (`dc9282be7d5` → `e6d0562cfaa`).** Two units were reverted to the baseline:
> `sort-single-row-control` (upstream Metal 2.0 port #52528) and `argmax-multicore-control`
> (needs a pipe semaphore restore for trace replay). Both are `deferred` with blocker flags.
> Full detail: `archive/reconciliation/reconcile_2026-08-26-rebase-e6d0562.md`.

| Unit | Kernels | Bindings | Existing evidence |
|---|---:|---:|---|
| `conv2d-weights-single-sender-rect` | 2 | 1 | Conv height inventory, DRAM routes, host/helper tests |
| `conv2d-weights-fixed-line` | 2 | 1 | Conv block inventory, PerRow/PerColumn, DRAM routes |
| `matmul-in1-mcast-padding-host` | 2 | 4 | `MM-IN1-ALL` 302 passed / 188 expected skips |
| `groupnorm-sharded-v2-mcast-host` | 4 | 4 | legacy/Welford inventories and matched performance |
| ~~`sort-single-row-control`~~ | 2 | 1 | **REVERTED 2026-08-26** — upstream Metal 2.0 port (#52528); re-migration is a re-authoring |
| `conv2d-activation-width-sharded` | 1 | 1 | exact JIT, features 48/16, DRAM route |
| `topk-multicore-final-readiness` | 2 | 1 | exact JIT, 14 passed / 12 expected xfails |
| `layernorm-sharded-pre-allgather` | 2 | 1 | pre 126, post 136, sharded 208 |

These units are stamped at API v14. Tier 0.2 `matmul-in0-mcast-block-sharded` is also migrated at v14:
its exact zero-hit-cache probe, complete mapped inventory, and inherited matched performance evidence
passed on 2026-08-16.

Tier 1.6 `deepseek-b1-sampling-loop-barrier` is stamped at v14; its original migration commit is
`2840fc28361`. Four 101-core
argmax nodes passed, including cold and warm JIT paths. The Blackhole top-k test remains skipped for
its pre-existing selection mismatch; when temporarily unskipped, its 100-iteration barrier completed
and reproduced the raw implementation's exact `p_scores` failure signature. Matched device-kernel
durations were +0.25% for argmax and -0.05% for top-k.

Tier 2.7 DRAM-sharded Matmul remains deferred because API v14 still cannot preserve both its forced
sender-only EXCLUDE-source data path and its type-2 signal-only INCLUDE-source path; API expansion was
not authorized. Tier 2.8 `group-attn-matmul-rotating-mcast` is stamped at v14; its original migration
commit is `6e8eb763885`.
Fresh-JIT and complete correctness passed (322 passed / 132 categorized expected skips), as did the
helper/source/host guards. Matched 800 MHz q16 and q48 device-kernel medians improved 32.20% and 27.31%.

Tier 2.9 `conv3d-weight-sharing-mcast` is stamped at v14; its original migration commit is
`a290ce20281`. The fixed-sender group
strips now use independent `Mcast2D` objects with an unconditional four-word runtime ABI; Chain and
Disabled paths remain unchanged. Fresh-JIT, focused and complete correctness, and all helper guards
passed. Matched 800 MHz non-grouped and grouped medians improved 0.815% and 0.298%.

Tier 2.10 `layernorm-sharded-post-allgather` is stamped at v14; its original migration commit is
`6cc49825476`. Dense `mcast_1d`
uses helper loopback; each non-1D line uses an outside-sender remote pipe plus an operation-owned local
copy. Build, exact fresh JIT, all 136 post, 126 pre, and 208 plain-sharded cases, plus helper and host
guards passed. Matched 800 MHz LayerNorm and RMSNorm medians improved 3.2% and 10.0%. The operation's
non-1D route remains unexercised because its existing one-core stats contract cannot supply every sender
line; the outside-sender geometry is host-tested and required no helper API expansion.

Tier 2.11 plain sharded LayerNorm remains deferred after its migrated single-stage path exceeded the
mandatory performance gate by 0.086 percentage points; the experiment was reverted. Tier 2.12
interleaved GroupNorm is stamped at v14 (original migration commit `40e209daad9`), Tier 2.13
SDPA-decode `read_k` in
`f760425fe06`, Tier 2.14 Argmax control in `5aaaf5b5aa5`, and Tier 2.15 Move overlap in
`a25603ae2c0`. Their complete correctness, fresh-JIT, helper/source guards, production-LOC, and matched
performance evidence is recorded in the per-unit logs and
`../migration_feedback_tracker.md`.

Tier 2.16 reached terminal deferrals without production edits. Routed-expert FFN needs two linked data
stages under one ACK and one final signal, which API v14 still cannot express. Persistent H2D/D2H target
cross-program GlobalSemaphore L1 addresses that API-v14 program-semaphore binding cannot address;
H2D additionally separates metadata and worker-ready publication across its completion boundary. The
service twins do not satisfy the unrelated-family API-extension gate, and their worker-sync tests
require Galaxy/UBB hardware unavailable on the current single-chip machine. Helper API is v14.

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

Seventy-five entries remain deferred. Their exact reasons and flags are authoritative in `ledger.json`.
The major classes are:

- genuine capability gaps such as chain relay, runtime role/count, and multi-phase protocols;
- coverage gaps or binary-only routes;
- helper-neutral or non-multicast entries retained for atomic-unit context;
- experimental/CCL/Quasar entries intentionally deferred as groups.
- four TT-Train kernel faces added by repository-wide recall, deferred pending
  a helper dependency-boundary decision and mapped C++ test execution.

Before changing the helper for any deferred kernel, state the required behavior independently of its
current implementation and verify that the existing helper cannot express it through a different
factory, ABI, channel split, or data flow.

## Scoped rollout complete

Tier 0, Tier 1.6, and Tier 2.7-2.16 have reached their approved terminal dispositions. Keep the two
interleaved Matmul kernels and five bindings pending until a future instruction separately authorizes
the historical performance checkout.
