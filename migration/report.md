# `mcast_pipe` rollout — migration report

Helper: `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` (`dataflow_kernel_lib::Pipe`). Mode: **run-all**
(every failure reverted, tree stays green, one report). Host machine: **single-chip Blackhole p150a**.
Baseline commit: `90120bf` (Phase 1). All migration commits are atomic, per-kernel, and bisectable.

---

## 1. Headline

| | count |
|---|---|
| Helper materialized + unit test | ✅ 45/45 cells PASS |
| Kernels device-verified (Phase 2) | 20 |
| **Kernels migrated** (committed, mapped test green) | **13** |
| Kernels failed (reverted, tree green) | 7 |
| Call-site lines removed (net across 13 commits) | **310 deletions / 235 insertions** (`git diff 90120bf..HEAD -- ':!migration'`) |
| Coverage gaps (not migratable on this machine) | ~24 kernels (see §5) |
| Deferred by invocation | R6, R4 streaming, legacy move/sort, fabric/ring CCL |

## 2. Per-tier summary

| Tier | scope | migrated | failed |
|---|---|---|---|
| 0 | helper unit test + F3 degenerate + PRE_HANDSHAKE | proof (green) | – |
| 1 | clean spine (canonical send/receive) | **10** | 1 |
| 2 | refactor-low (multi-rect / loopback / scatter) | **2** | 3 |
| 3 | refactor-high (counter / chain-link / chunked) | **1** | 3 |
| **Σ** | | **13** | **7** |

## 3. Per-kernel results

### Migrated (13) — committed, mapped test PASS
| kernel | tier | commit | lines removed | note |
|---|---|---|---|---|
| reader_bmm_tile_layout_in0_sender_padding | 1 | (in `4751ee8`*) | ~42 | canonical `send()`; sparsity flag-only block left raw |
| reader_bmm_tile_layout_in0_receiver | 1 | c9ac10b | ~7 | canonical `receive()`; sparsity wait_min left raw |
| reader_bmm_tile_layout_in1_sender_writer_padding | 1 | 9370e0a | ~83 | 2× `send()` (in1 + bias) |
| reader_bmm_tile_layout_in1_receiver_writer_padding | 1 | 8a13611 | ~14 | 2× `receive()` (in1 + bias); matmul_2d |
| reader_writer_tiled_out_1d_mcast_receiver_conv_weights… | 1 | a18af0f | ~14 | 2× `receive()`; count-independent → fits |
| writer_tiled_out_2d_mcast_sender_conv_weights… | 1 | 07ec88c | ~64 | 2× `send()`; 2D num_dests==num_cores |
| writer_tiled_out_2d_mcast_receiver_conv_weights… | 1 | d8aaa9e | ~14 | 2× `receive()` |
| reader_mcast_receiver_unary_sharded_gn_v2 | 1 | b07ce75 | ~3 | `receive()`; GN sem naming flipped |
| reader_final_topk | 1 | da0ae51 | ~5 | PARTIAL — readiness → `send_signal()`; fan-in counter left raw (INV9 multi-producer) |
| sampling_kernel (deepseek) | 1 | 9f93e58 | ~4 | flag-only loop barrier → `send_signal(1)`; R5 NOC1-swap validated |
| welford_reader_mcast_receiver_unary_sharded_gn_v2 | 2 | e7bc619 | 5 | `receive()`; twin of Tier-1 receiver |
| activation_reader_width_sharded (conv) | 2 | 967cb69 | 43 | PARTIAL — INCLUDE_SRC loopback data+flag → `send()` (PRE_HANDSHAKE=false); counter half left raw; PCC 0.9999992 |
| reader_mcast_sender_unary_sharded_ln | 3 | 26540b5 | 8 | PARTIAL — phase-1 control flag → `send_signal(VALID)`; phase-2 monotone counter left raw |

`*` `reader_bmm_tile_layout_in0_sender_padding` was the first kernel migrated; a `git commit --amend`
during Tier 1 folded its change into the Phase-2/3 commit `4751ee8` instead of a standalone commit.
The code change is intact in HEAD and was device-verified PASS — only the commit boundary is cosmetic.

### Failed (7) — reverted, tree green, all clean structural determinations (no hidden breakage)
| kernel | tier | validation | reason |
|---|---|---|---|
| reader_writer_tiled_out_1d_mcast_sender_conv_weights… | 1 | hang | **divergent count**: 1D conv handshake count (`active_cores-1`) ≠ mcast geometry count (`num_cores-1`); single `num_dests` can't express both |
| reader_mcast_sender_unary_sharded_gn_v2 | 2 | n/a | **multi-rect** (mid/first/last ×3 rectangles, each own num_dests) |
| welford_reader_mcast_sender_unary_sharded_gn_v2 | 2 | n/a | **multi-rect** ×3 per-group loop |
| writer_local_topk | 2 | n/a | **not rectangle-mcast** — unicast scatter to one final core + up(counter) |
| reader_mcast_receiver_unary_sharded_ln | 3 | n/a | **H11 reset-ordering mismatch**: phase-1 needs clear-before-wait on a counter-contaminated shared sem cell; Pipe pins clear-after-wait. Phase-2 `wait_min(base+2)` ≠ Pipe's `++round_` |
| reader_interleaved (sdpa) | 3 | n/a | **legacy raw API + R6 ring**: raw `noc_*` free functions + KV-chain ring/role-flip forwarding (deferred ring legs) |
| reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2 | 3 | n/a | **R4 streaming-only**: the sole data mcast is the chunked `mcast_block_chunked`; flag inseparable from it (INV4 single-VC). R4 is deferred this round |

## 4. API generality findings (feedback for a Pipe v2 — these are the load-bearing limits the rollout hit)

1. **Single-rect `McastRect` (INV8) is the #1 blocker.** It cannot express (a) a send to a **list of
   rectangles** with per-rect `num_dests` (group_norm/welford senders — 2 kernels), nor (b) a topology
   where the **handshake ack-count differs from the mcast rectangle population** (conv 1D weights
   sender — `active_cores-1` vs `num_cores-1`, hung). Matmul + conv-2D happen to have these equal, so
   they migrated; the divergent-count cases cannot without a two-count / multi-rect API.
2. **H11 reset ordering is pinned to clear-after-wait**, which collides with kernels that share one sem
   cell across a counter phase and a flag phase and must clear-before-wait (sharded LN receiver).
3. **Counter staging is `++round_` (per-side monotone)**, which can't match a kernel that waits an
   externally-computed threshold (`wait_min(block+2)`).
4. **The helper assumes object-API spelling and rectangle-mcast.** Legacy raw-`noc_*` kernels (sdpa
   reader_interleaved) and unicast-scatter (topk writer) are out of its shape by construction.
5. **Receivers migrate far more readily than senders** — `receive()` is count-independent, so a
   receiver migrates even when its paired sender can't (gn_v2, welford, conv-1D all show this).

## 5. Coverage gaps (risk list)

### Migrated but PARTIAL (only part of the block uses the Pipe — the rest stays raw)
`reader_final_topk` (fan-in counter raw), `activation_reader_width_sharded` (R→S counter raw),
`reader_mcast_sender_unary_sharded_ln` (phase-2 counter + consumed-drain raw),
`reader_bmm_tile_layout_in0_sender_padding` / `in0_receiver` (sparsity sub-block raw). These passed
their mapped tests, so the migrated portion is validated — but the kernel is a hybrid, not a full
conversion.

### Verified kernels NOT migrated (API mismatch) — 7, see §3 failed.

### Kernels with NO validatable coverage on this single-chip machine (NOT attempted)
- **Multi-device only (~11):** ln_pre/post_allgather ×4; all CCL legs (rms_allgather, llama
  AG-matmul, deepseek_prefill dispatch/combine, moe_gpt ×2, selective_reduce ×2, all_gather_concat,
  all_to_all). Migrating these blind would defeat the run-all safety net — they need a multi-device
  host to validate.
- **Build-seen but failing test here (3):** `dm1` (moe_gate_mm), `moe_compute` tilize ×2.
- **No test coverage at all (sweep-only, 4):** interleaved (non-v2) group_norm mcast kernels.
- **Unverified / mode-dependent:** argmax multicore reader, conv3d writer, matmul dram_sharded,
  3 matmul didactic examples.

## 6. Deferred by invocation (correctly excluded, not attempted)
R6 same-core role-flip (matmul block-sharded `_in0_sender_receiver`, group_attn); R4 streaming
chunked-send; legacy-API move/sort (need a Noc/Semaphore port first); fabric/ring CCL legs;
`chain_link.hpp` (prior-art) and deepseek `mcast.hpp` (preprogram-state). Note: the conv halo reader
and sdpa reader_interleaved landed in the failed list because, once examined, their *only* mcast was
the deferred R4/ring shape — consistent with the deferral, just surfaced at migration time.

## 7. Provisional items confirmed (Phase 1)
- **Consumer-wait-inside-`send()` ordering** (the matmul-in0 concern): confirmed green by
  `test_pre_handshake` (reused-dest, multi-round, PRE_HANDSHAKE=true, 1×2 & 1×8, N=1 & 8). The
  consumed-wait at the start of `send()` correctly gates the mcast without gating the source fill.
- **F4 linked** correctness confirmed across the full coverage matrix (`flag_linked` variant, 24
  cells). The bake-off perf gaps (flush −27%, flag −29%, linked −36%) are **embedded** in the helper;
  every migrated call site now rides those measured-fastest variants.
- **F3 degenerate guard** (self-only num_dests==1 → local copy) confirmed by `test_f3_degenerate`.

## 8. Default decisions made (autonomous run — no human gates)
1. **Gates 0 and A skipped** per the invocation (run fully autonomously).
2. **Migratable universe = the 20 device-verified kernels only.** Kernels with no runnable test on
   this single-chip machine were NOT migrated blindly (run-all's revert safety net can only protect a
   kernel with a green mapped test). This is the single biggest scoping decision.
3. **`receive_signal()` value-carrying payload** (moe_gpt) left as a documented migration-time concern
   (the op reads its own sem cell) rather than threading the sem address through the helper — keeps the
   doorbell path simple.
4. **Partial migration counts as migrated** when the migrated verb is a clean Pipe call and the mapped
   test passes; the raw remainder is noted per kernel.
5. **Per-kernel perf was NOT separately re-measured** (validation-focused run). The helper bakes in the
   bake-off's measured-fastest variants, so migrated sites inherit them; a tracy perf sweep of the 13
   migrated kernels is a sensible follow-up but was out of scope for the green-tree rollout.
6. **The bake-off harness was preserved** as `test_mcast_pipe_bakeoff.py` (raw baseline); the helper's
   unit test is the ported `test_mcast_pipe.py`.

## 9. Artifacts
- Helper: `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` + unit test `tests/.../kernel_lib/test_mcast_pipe.py`
  + driver kernels `tests/.../kernel_lib/kernels/pipe_{sender,receiver,f3_sender}.cpp`.
- Map: `migration/test_map.{md,json}` + `migration/test_map_<family>.md` (kept + reusable).
- Tiers: `migration/tiers.md`. Per-kernel detail: `migration/log/<basename>.md`.
- 13 atomic migration commits on top of `90120bf`.
