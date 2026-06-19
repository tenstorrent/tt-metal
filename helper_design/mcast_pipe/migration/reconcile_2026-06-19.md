# Reconcile report — mcast_pipe @ v4 — 2026-06-19

`reconcile-dm-helper helper_design/mcast_pipe/` (no `--base`). STATIC audit; no device, no migration.
Re-aligns `census.txt` + `migration/ledger.json` with the post-rebase tree. Helper API unchanged
(`MCAST_PIPE_API_VERSION` = 4, untouched).

## Summary

| bucket | count | action taken |
|---|---|---|
| `unchanged` | 53 | none |
| `removed` | 0 | — |
| `renamed` | 0 | — |
| `clobbered` | 0 | — |
| `rebase-touched` | 13 | set `flags:["needs_recheck"]` on all migrated entries (conservative — no `--base`) |
| `added` | 3 | +3 census lines, +3 ledger entries, +3 annotations |

Ledger: 66 → **69 entries** — 13 migrated (all `needs_recheck`) · 48 pending · 8 deferred.

## Phase 1 — existence + usage (66 census entries)
- All 66 files present in the tree. No deletions, no renames.
- All 13 `migrated` kernels still reference the helper (`SenderPipe`/`ReceiverPipe`/`mcast_pipe.hpp`) →
  none clobbered.
- **No `--base` supplied** → cannot prove which migrated kernels the rebase actually touched, so all 13
  are conservatively flagged `needs_recheck`. `apply-dm-helper` will **verify-only** (re-run mapped
  tests, no rewrite) and clear the flag on green. Supply the pre-rebase ref on a future run to narrow.

## Phase 2 — recall sweep (multicast family from primitive_contracts.md)
55 files matched the multicast recognition family; 48 already in census. **3 real new candidates**
(below) + **3 excluded** (matched the family but are not migration call sites).

### Added (3)
| kernel | family | role | tag | annotation |
|---|---|---|---|---|
| `…/deepseek_prefill/unified_routed_expert_ffn/…/unified_routed_expert_ffn_reader.cpp` | ccl/deepseek | hybrid | refactor | `kernel_annotations/unified_routed_expert_ffn_reader.md` |
| `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_receiver.cpp` | ccl/deepseek | sender | refactor | `kernel_annotations/persistent_h2d_receiver.md` |
| `…/ccl/llama_all_gather_matmul_async/…/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | ccl | sender | defer | `kernel_annotations/llama_all_gather_matmul_async_in1_ring_all_gather.md` |

(New or previously-missed — both in experimental dirs; action is the same either way.)

### Excluded (3) — not call sites
- `tt_metal/impl/emulation/emulated_program_runner.cpp` — host emulation infra; tokens appear only in comments.
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.cpp` — host program factory; comment-only match.
- `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp` — shared read-template header; its only multicast token is a `McastDst` type alias used *by* the already-migrated conv kernels — no handshake block of its own (infra, like `kernel_lib`).

## Hand-off to apply-dm-helper (NOT executed here)
Re-invoke `apply-dm-helper helper_design/mcast_pipe/ --mode=<halt|run-all>`:
- **verify-only** the 13 `needs_recheck` migrated kernels (re-run mapped tests, clear flag on green; a red
  one regressed under the rebase → escalate);
- **migrate** the 2 new `pending` (refactor) candidates as net-new work in their tier;
- the 1 new `deferred` (llama ring all-gather) stays out of scope.

No `<HELPER>_API_VERSION` change — this was a tree-move reconcile, not an API bump.
