# Upstream tt-metal scan for LTX-2.5 (2026-09-30 .. 2026-10-05), task #132

How I checked (off-device; no gains measured):
- `git fetch origin`; origin/main tip 53de0ba10ec (2026-10-05 01:56 UTC).
- ltx-rt and t48 (`ttp/t48-ltx25-integrated` @ c4409b1fa24) both fork main at c2a4d40104d (2026-09-29 16:32).
  261 main commits since then; 109 touch conv3d / neighbor_pad / SDPA / matmul / CCL / norm / tilize / trace / LLK / tt_dit paths.
- Conflict risk: files each commit touches vs files t48 changed since the base (`git diff --name-only c2a4d40104d t48`),
  plus "drift" (same files changed on main between the base and the commit's parent = prerequisite commits).
  git 2.34 here has no `merge-tree --write-tree`, so no real trial merge was done.
- LTX SDPA facts from t48 `attention_ltx.py`: every attention uses `ring_joint_scaled_dot_product_attention` or
  `scaled_dot_product_attention` with HiFi2, `fp32_dest_acc_en=False`, BH 8x4 chunks q128/k512.
  With bf16 dest both ops run the streaming kernel (`can_use_streaming_compute`), whose main softmax exp is hard-coded
  approximate (`compute_streaming.hpp:525`).
- Time model: S2 ≈ 2.5 s = 3 steps × 48 blocks × ~16.4 ms; ring SDPA 5.37 ms/block → ~0.77 s of S2.
  S1 ≈ 2.25 s at lower resolution, so the SDPA share is smaller (guess: ~0.2-0.3 s).

Nothing merged in this window speeds up conv3d, neighbor_pad, all_gather_matmul, the DiT rmsnorm, tilize/untilize or trace.

## Ranked top 5

| # | PR / commit | What | Est. saving on LTX-2.5 1080p 6s | Conflict risk with t48 | Bit-identical? |
|---|---|---|---|---|---|
| 1 | [#58223](https://github.com/tenstorrent/tt-metal/pull/58223) a819297d50b | ring_joint_sdpa `SDPAProgramConfig.matmul_math_fidelity` (QK^T and P·V at their own fidelity, e.g. LoFi, while exp/max stay HiFi2) + opt-in `segmented_accumulation` | **~150-350 ms** if LoFi passes quality. Gemma4 BH galaxy: one global layer 24.9→16.8 ms (-33%). LTX at HiFi2→LoFi: est. -20 to -30% of ring SDPA → S2 -0.15 to -0.25 s, S1 -0.05 to -0.1 s. | **Medium.** It touches `sdpa/.../compute_common.hpp` and `transformer_nanobind.cpp`, which t48 also changed (na-integration merge). 9 of its 10 files changed on main since the base, so it needs #57979 (6427f420091) and #58032 (30cbb5c8014) first. | Default yes (opt-in, unset = unchanged). With LoFi: **no**. Needs PSNR, VBench and 5 seeds. |
| 2 | [#56023](https://github.com/tenstorrent/tt-metal/pull/56023) 27cecf3f7c1 | BH NoC: stop writing `NOC_TARG_ADDR_MID`/`NOC_RET_ADDR_MID` on every non-PCIe read/write (~5.5% faster NoC issue); PCIe paths move to explicit helpers | **~20-80 ms** (guess). Helps data-movement-bound ops: conv3d reader, layout, halo, CCL workers, dispatch. Most DiT time is compute-bound, so the gain is small. | **Low.** No file overlap with t48. 4 of its 40 files drifted on main (check they apply). No t48-changed kernel uses PCIe NoC addresses. Rebuilds all kernels (dataflow_api.h, dispatch). | **Yes.** Removes redundant register writes, no math change. |
| 3 | [#58032](https://github.com/tenstorrent/tt-metal/pull/58032) 30cbb5c8014 | ring_joint_sdpa K split (`max_k_splits`, opt-in), packed latent V; `nlp_create_qkv_heads`/`nlp_concat_heads` wait once per 8 tiles instead of per tile | **~0-60 ms.** LTX S2 per device: ~8 heads × ~38 q-chunks ≈ 304 work units on ~120 cores = 2.53 waves; a K split might even out the last wave (0-5% of SDPA). LTX calls `nlp_create_qkv_heads` (4 files), which gets the barrier batching by default: maybe 5-20 ms. Gemma4: -1.7 to -3.8% per chunk. | **Medium.** Overlaps t48 in `compute_common.hpp`, `transformer_nanobind.cpp` and gtests `sources.cmake`. Needs #57979 first (drift 4). | Head-op batching: yes. K split: no (different softmax merge order), but opt-in. |
| 4 | [#57979](https://github.com/tenstorrent/tt-metal/pull/57979) 6427f420091 | Multicast multi-hop sliding halo, larger halo packets | **0 ms direct.** LTX DiT attention is global, not sliding. Bring it in only as the base for #1 and #3. | **Low.** No overlap, no drift. | Yes for the global path. |
| 5 | [#56502](https://github.com/tenstorrent/tt-metal/pull/56502) e38fdfea4a9 | `untilize_with_unpadding` device hang fix | **0 ms.** LTX model code does not call it directly, but `to_layout(ROW_MAJOR)` on padded tensors does, e.g. VAE output/export. Safety only. | **Low.** No overlap, no drift, 3 files. | Yes for cases that did not hang. |

Recommended order: #2 on its own first (cheap, bit-identical, eager-vs-trace A/B on a 2x4 submesh). Then the SDPA chain
#57979 → #58032 → #58223 as one cherry-pick task, run with `matmul_math_fidelity=LoFi` on the ring SDPA configs, and judge with PSNR + VBench + 5 seeds.

## Checked and rejected (no LTX gain or not applicable)
- [#58634](https://github.com/tenstorrent/tt-metal/pull/58634) 4dd0c5a36c1 (Motif: flip SDPA `exp_approx_mode=True`; #57180 made accurate exp ~24% slower per Motif denoise step).
  LTX asks for `exp_approx_mode=False` too, but its SDPA runs the streaming kernel, whose main exp is always approximate,
  so expect little or no gain. Flipping the flag on the LTX configs is a free 1-run check, not a cherry-pick.
- [#57878](https://github.com/tenstorrent/tt-metal/pull/57878) 4c680a02ab5 ring MLA batched V matmul and softmax/V overlap: only on the in-place latent-V (MLA) path (`static_assert` in `compute_streaming.hpp`). LTX has separate V. 0 ms.
- [#57998](https://github.com/tenstorrent/tt-metal/pull/57998) 701f5c04142 block-cyclic sliding Q: sliding windows only.
- [#58654](https://github.com/tenstorrent/tt-metal/pull/58654) 8ff9ba539e4 conv3d unpacker reconfig fix: fixes `matmul_blocks_split` from #56922 (MiniMax fp32 operand split). t48 does not have that function. N/A unless #56922 is ported.
- [#54786](https://github.com/tenstorrent/tt-metal/pull/54786) f140930eb08 Welford two-pass statistics: 107 files, changes numerics of groupnorm/layernorm Welford. The LTX DiT uses `dit_fused_distributed_{rms,layer}norm`; the LTX conv VAE uses no GroupNorm. No hot-path gain; high conflict cost.
- [#58086](https://github.com/tenstorrent/tt-metal/pull/58086) 2460c2f6b6d norm defect fixes: `layernorm_distributed` Welford and sharded groupnorm, which LTX does not use.
- [#58615](https://github.com/tenstorrent/tt-metal/pull/58615) f30c017d815 GroupNorm/RMSNorm unity-build collision: build-only. It touches `dit_fused_distributed_rmsnorm_program_factory.cpp`, which t48 changed (81a9b8f8963). Take it only if a unity build breaks.
- BH SFPU: [#58182](https://github.com/tenstorrent/tt-metal/pull/58182) 383aefb8b99 log constant hoist and [#58188](https://github.com/tenstorrent/tt-metal/pull/58188) 668bbbb7de9 SFPU reduce are bit-identical with no conflicts, but neither log nor SFPU reduce is on the LTX hot path. ~0 ms.
- [#55638](https://github.com/tenstorrent/tt-metal/pull/55638) e5a9e9b41aa BH B2D datacopy MOP: hang fix for tiny tiles, a no-op at 4 faces.
- [#56108](https://github.com/tenstorrent/tt-metal/pull/56108) 6908c16404d Kimi fused RMSNorm: model-local (deepseek_v3_d_p). As an idea it overlaps t48's own dit rmsnorm fusion.
- #58211 trace API cleanup, #58381/#58378 SDPA/conv argument validation, Quasar and LLK test commits: no perf.

## Ideas worth borrowing (not cherry-picks)
- Gemma4 [#58224](https://github.com/tenstorrent/tt-metal/pull/58224) / dbece40ccde: LoFi global attention + LoFi projections with segmented accumulation passed their PCC gate. The same LoFi sweep on LTX DiT projections (QKV/out/FF matmuls) follows the #84 VAE LoFi result (-9% conv decode, PSNR ≥45 dB).
- #57395 (BH SDPA roofline set, from the 09-30 scan) is still not on main.
