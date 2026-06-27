# Reconcile report — mcast_pipe @ v8 — 2026-06-27

`reconcile-dm-helper helper_design/mcast_pipe/` after a rebase of `sjovic/mcast-helpers-june26` onto a
fresh `llk_helper_library` (`df38b5abc67`). STATIC audit; no device, no migration, no build. Helper API
unchanged (`MCAST_PIPE_API_VERSION` = 8, untouched).

**Base discriminator.** No `--base` was passed, but the reflog records the exact pre-rebase rollout tip
(`7ae140745b9`, replayed onto `llk_helper_library` → `acafdfcc6c4`). Diffing `7ae140745b9 → HEAD`
restricted to the 69 census files yields the PRECISE set of census files the rebase actually altered —
strictly tighter than the conservative "flag-all-migrated" fallback the 2026-06-19 run used.

## Summary

| bucket | count | action |
|---|---|---|
| `unchanged` | 68 | none (18 migrated byte-identical pre↔post + 50 deferred) |
| `removed` | 0 | — |
| `renamed` | 0 | — |
| `clobbered` | 0 | all 19 migrated still reference SenderPipe/ReceiverPipe |
| `rebase-touched` (migrated) | 1 | `reader_mcast_receiver_unary_sharded_ln.cpp` → `flags+=["needs_recheck"]` (edit is comment-only) |
| `added` | 23 | +23 census lines, +23 ledger entries, +grouped annotations |

Ledger: **69 → 92 entries** — 19 migrated (1 `needs_recheck`) · 3 pending · 70 deferred.

## Phase 1 — existence + usage (69 census entries)
- **Existence: 0 missing / 69.** No deletions, no renames. census.txt ↔ ledger.json agree on all 69 paths.
- **Usage: all 19 migrated still reference the helper** (`SenderPipe`/`ReceiverPipe`/`mcast_pipe.hpp`,
  2–5 hits each) → **0 clobbered**.
- **Rebase-touched:** `git diff 7ae140745b9 HEAD` touched 9 census files. 8 are `deferred` (no migrated
  record to disturb → no action). The 1 **migrated** one — `reader_mcast_receiver_unary_sharded_ln.cpp` —
  received a **3-line explanatory comment** before an existing `pop_front` (helper logic untouched).
  Flagged `needs_recheck` per the rule; the verify-only re-run is expected to be a trivial pass (a comment
  cannot change JIT/runtime). The other 18 migrated kernels are byte-identical pre↔post → `unchanged`,
  no flag. (06-19 conservatively flagged 13 with no `--base`; the precise diff narrows it to 1.)

## Phase 2 — recall sweep (multicast recognition family from primitive_contracts.md)
82 files matched the family; 48 already in census → 34 not-in-census → **11 excluded** + **23 added**.

### Excluded (11) — not migration call sites
- 7 substrate API headers — `tt_metal/hw/inc/api/dataflow/{noc,noc_semaphore,endpoints,dataflow_api,
  circular_buffer,dataflow_buffer}.h` + `internal/dataflow/dataflow_api_addrgen.h`. These DEFINE the
  recognition family (e.g. `noc.h:386` is the `async_write_multicast` definition). Substrate, not call sites.
- `tt_metal/impl/emulation/emulated_program_runner.cpp` — host emulation; comment-only match (06-19 too).
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.cpp` — host
  program factory; comment-only match (06-19 too).
- `conv_reader_common.hpp` ×2 (conv2d + quasar copy) — `McastDst` type-alias header, comment-only match.

### Added (23) — each read-and-classified by 3 parallel subagents against the existing patterns/hazards

**A. New `experimental/quasar/` Metal-2.0 tree (18) → DEFERRED-AS-GROUP** (gate decision). Absent at the
pre-rebase tip; brought in by the rebase (upstream commits "port experimental/quasar ops to metal 2.0
api", #47797). All object-API. Each is a port-fork of a production twin already in census. Recorded with
ledger `status=deferred` + flag `quasar-metal2-port` (+`hang:#47797` on the 4 forks carrying live
hang-debug scaffolding). The per-file `tag` is the intrinsic eventual target for when the port stabilizes:

| file (basename) | op | role | tag | flags |
|---|---|---|---|---|
| reader_bmm_tile_layout_in0_sender_dram_sharded.cpp | matmul | hybrid | refactor | quasar-metal2-port |
| reader_bmm_tile_layout_in0_sender_padding.cpp | matmul | sender | clean | quasar-metal2-port |
| reader_bmm_tile_layout_in0_sender_padding_metal2.cpp | matmul | sender | clean | quasar-metal2-port |
| reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp | matmul | hybrid | refactor | quasar-metal2-port |
| reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp | matmul | hybrid | refactor | quasar-metal2-port, hang:#47797 |
| reader_bmm_tile_layout_in1_ring_all_gather.cpp | matmul | sender | defer | quasar-metal2-port |
| reader_bmm_tile_layout_in1_sender_writer_padding.cpp | matmul | hybrid | clean | quasar-metal2-port |
| reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp | matmul | hybrid | clean | quasar-metal2-port |
| activation_reader_width_sharded.cpp | conv | hybrid | refactor | quasar-metal2-port |
| activation_reader_width_sharded_metal2.cpp | conv | hybrid | refactor | quasar-metal2-port |
| reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp | conv | hybrid | refactor | quasar-metal2-port |
| reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp | conv | hybrid | defer | quasar-metal2-port, hang:#47797 |
| reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | conv | sender | clean | quasar-metal2-port |
| reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp | conv | sender | clean | quasar-metal2-port, hang:#47797 |
| writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp | conv | sender | clean | quasar-metal2-port |
| writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp | conv | sender | clean | quasar-metal2-port, hang:#47797 |
| move_interleaved_with_overlap.cpp | move | hybrid | refactor | quasar-metal2-port |
| move_stick_layout_interleaved_with_overlap.cpp | move | hybrid | refactor | quasar-metal2-port |

Rationale for defer-as-group (both quasar subagents converged independently): (1) **actively churning**
mid-flight Metal-2.0 port; (2) every kernel **duplicates** a production pattern already in census (P1/P2/P3/
P4 + conv/move twins — zero new hazard coverage); (3) 4 `_metal2` forks carry live `#47797` hang-debug
scaffolding, one (`reader_conv_activations_2d…v2_metal2.cpp`) with an UNRESOLVED act-mcast deadlock on the
exact handshake a migration would touch. Census grouped under a new `# ===== quasar (experimental metal
2.0 port) =====` section (gate decision — a coherent distinct experimental tree with its own factories).

**B. New non-quasar call sites (5)**

| file | op-family | role | tag | status | flags | note |
|---|---|---|---|---|---|---|
| persistent_d2h_sender.cpp (models/…/micro_ops/host_io) | ccl/deepseek/examples | sender | refactor | pending | — | intra-chip `inc_multicast`-to-workers in scope; PCIe bulk leg oos. Twin `persistent_h2d_receiver` ended up deferred+coverage-gap → apply may defer this too |
| persistent_d2d_receiver.cpp (ttnn/core/tensor/kernels) | ccl/deepseek/examples | sender | refactor | pending | — | intra-chip data mcast + counter in scope; fabric data leg oos |
| persistent_d2d_sender.cpp (ttnn/core/tensor/kernels) | ccl/deepseek/examples | sender | refactor | pending | — | one intra-chip `inc_multicast(consumed_sem)` in scope; fabric leg oos |
| reader_indexer_score.cpp | transformer+sdpa | hybrid | refactor | deferred | design-gap | needs CHAIN/`relay_multicast`+linked=true — the GAP=CHAIN the helper can't express yet |
| lab_multicast/mcast_sender.cpp | ccl/deepseek/examples | sender | ref | deferred | — | didactic example (raw API); previously-missed (existed pre-rebase) |

## Hand-off to apply-dm-helper (NOT executed here)
Re-invoke `apply-dm-helper helper_design/mcast_pipe/ --mode=<halt|run-all>`:
- **verify-only** the 1 `needs_recheck` kernel (`reader_mcast_receiver_unary_sharded_ln.cpp`) — re-run its
  mapped test, clear the flag on green. Trivial pass expected (comment-only edit).
- **migrate** the 3 new `pending` (refactor) persistent kernels as net-new work. NOTE: the censused twin
  `persistent_h2d_receiver` ended up `deferred`+`coverage-gap`; apply may defer these once it builds the
  device-verified test map and finds no single-chip coverage.
- the 20 new `deferred` (18 quasar + `reader_indexer_score` + `lab_multicast`) stay out of scope until
  their gates clear — quasar: port stabilizes + `#47797` closed + DEBUG stripped; indexer_score: helper
  gains the CHAIN/relay path.

No `<HELPER>_API_VERSION` change — this was a tree-move reconcile, not an API bump.
