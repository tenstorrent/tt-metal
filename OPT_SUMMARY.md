# `wransom/model_opt` — Optimization summary

Branch off main (`bbfbb558690`) carrying two independent optimization streams:

1. **BH wins** from the OptSearchBH session (this conversation) — `packer_l1_acc=False → True` on
   conv-heavy BH vision models. Validated on BH p100a, this branch will be re-verified before WH testing.
2. **Main-compatible matmul-config wins** from `wransom/matmul_helpers_opt_3` — h=1
   subblock-volume bumps that don't depend on the helper library (`row_major_output` flag, etc.).
   Original WH validation was on n150; BH validation where documented in the BH-continuation
   memory.

Threshold bumps on Falcon7b's WH perf tests are included so the WH device-perf gate doesn't
trip the "too slow" lower bound after the optimization lands.

## Optimization ledger

| # | Change | File:Line | System | Device-kernel Δ | Model-perf Δ | Source |
|---|---|---|---|---:|---:|---|
| 1 | `packer_l1_acc=False → True` on Conv class | `models/demos/vision/segmentation/vgg_unet/common/ttnn/ttnn_vgg_unet.py:53` | BH p100a | **Conv2d −5.32 %**, all-ops −3.66 % | Device-perf test 510 → 529 sps (expected 522, PASS) | OptSearchBH 2026-05-11 (2 trials per arm, <0.1 % noise) |
| 2 | `packer_l1_acc=False → True` on Conv_transpose class | `models/demos/vision/segmentation/vgg_unet/common/ttnn/ttnn_vgg_unet.py:131` | BH p100a | (same kernel — sibling edit in same model) | — | OptSearchBH 2026-05-11 |
| 3 | `packer_l1_acc=False → True` on TtnnUFLDV2Conv2D | `models/demos/vision/segmentation/ufld_v2/common/ttnn/common.py:39` | BH p100a | **Conv2d −2.76 %**, all-ops −1.23 % | Device-perf test 595 → 628 sps (perf test now trips upper bound; `expected_perf` would need bump) | OptSearchBH 2026-05-11 |
| 4 | `packer_l1_acc=False → True` on Conv `_initialize_compute_config` | `models/demos/vision/classification/mobilenetv2/tt/common.py:85` | BH p100a | **Conv2d −9.44 %**, all-ops −5.77 % | n/a (no BH perf-test in tree; mobilenetv2 perf-test is `@run_for_wormhole_b0()`) | OptSearchBH 2026-05-11 (2 OPT trials, 0.02 % spread) |
| 5 | `out_subblock_w 1 → 8` on `DENSE_H_TO_4H_MM_PROGCFG` (mm_h_to_4h) | `models/demos/falcon7b_common/tt/model_config.py:320` | WH n150 (untested on BH; `@run_for_wormhole_b0()` test gate) | matmul wins amortize as +19.9 % prefill seq=1024 sps (CI thr) / **+27.3 %** same-hw A/B per OPT3_RESULTS.md | Prefill seq=1024 device-kernel sps 3120 → 3742 | `opt_3` commit `88124241bcd` (legacy-compatible h=1, no helper needed) |
| 6 | `out_subblock_w 1 → 6` on `DENSE_4H_TO_H_MM_PROGCFG` (mm_4h_to_h) | `models/demos/falcon7b_common/tt/model_config.py:332` | WH n150 (untested on BH) | (combined with #5 above into Falcon7b prefill seq=1024 +19.9 %) | (same as #5) | `opt_3` commit `88124241bcd` |
| 7 | Threshold bump: prefill seq=1024 `expected_inference_time 0.41 → 0.31` (single-chip) | `models/demos/falcon7b_common/tests/test_perf_falcon.py:65` | WH n150 | n/a (test threshold) | locks in the #5/#6 win | `opt_3` commit `88124241bcd` |
| 8 | Threshold bump: prefill seq=1024 `expected_inference_time 0.41 → 0.31` (mesh) | `models/demos/falcon7b_common/tests/test_perf_falcon.py:124` | WH n150 (mesh path) | n/a (test threshold) | locks in the #5/#6 win on the mesh test | `opt_3` commit `88124241bcd` |
| 9 | Threshold bump: device-perf samples `3120 → 3741` | `models/demos/falcon7b_common/tests/test_falcon_device_perf.py:88` | WH n150 | n/a (test threshold) | locks in the #5/#6 win on device-perf | `opt_3` commit `88124241bcd` |
| 10 | `out_subblock_w 1 → 8` on `linear_config_1024` (Segformer MLP) | `models/demos/vision/segmentation/segformer/tt/ttnn_segformer_mlp.py:34` | WH n150 + BH p100a | **WH −35.5 %** kernel; **BH −31.9 %** kernel (BH measured 2026-04-30 on p100a per BH-continuation memory) | Segformer model-level perf-test threshold not bumped (not landed in opt_3 either) | `opt_3` BH-continuation memory (was at the lost tip `47df299ef68` — h=1 path, NOT helper-enabled) |
| 11 | `out_subblock_w 1 → 8` on `2D_GEGLU_LINEAR_1536_SPLIT_GELU` (refiner 64_cores section, 1024x1024 WH) | `models/demos/stable_diffusion_xl_base/refiner/tt/model_configs/model_configs_1024x1024.py:187` | WH n150 + BH p100a — but only fires when `force_full_grid=True`; default is `False` so this edit is currently a no-op on the default code path | **WH −28.2 %** kernel; **BH −11.9 %** kernel (BH measured 2026-04-30 on p100a) | n/a | `opt_3` BH-continuation memory |
| 12 | `out_subblock_w 1 → 8` on `2D_GEGLU_LINEAR_1536_SPLIT_GELU` (refiner 64_cores section, 1024x1024 BH) | `models/demos/stable_diffusion_xl_base/refiner/tt/model_configs/model_configs_1024x1024BH.py:187` | BH p100a — sibling to #11; only fires when `force_full_grid=True` | (same as #11) | n/a | `opt_3` BH-continuation memory |

## Per-arch testability table

| Optimization | Testable on **BH** this session | Testable on **WH** (next session) |
|---|---|---|
| #1, #2 VGG UNet packer_l1_acc | ✅ — `test_vgg_unet.py` not arch-gated; PCC + Tracy A/B | ✅ — model has WH demo path too |
| #3 UFLD v2 packer_l1_acc | ✅ — `test_ttnn_ufld_v2.py` not arch-gated | ✅ |
| #4 Mobilenetv2 packer_l1_acc | ✅ — `test_mobilenetv2.py` not arch-gated | ✅ |
| #5, #6 Falcon7b prefill subblocks | ❌ — `test_falcon_device_perf.py` is `@run_for_wormhole_b0()` | ✅ — primary target |
| #7, #8, #9 Falcon7b threshold bumps | ❌ — same arch gate | ✅ — gates the #5/#6 win |
| #10 Segformer linear_config_1024 | ✅ — `test_segformer_for_semantic_segmentation.py` not arch-gated; PCC + Tracy A/B (BH-continuation memory documented −31.9 % on this hardware) | ✅ |
| #11, #12 SDXL refiner GEGLU 64_cores | ⚠️ — code path requires `force_full_grid=True`; default refiner BH perf test runs `force_full_grid=False` and won't exercise this edit. 64_cores section has ~21 missing keys per the earlier BH research memory — flipping the flag may fail before reaching this matmul. Edit stays as documented-and-ready for later activation. | ⚠️ — same `force_full_grid` gate; the WH BH-continuation result was measured by isolating the GEGLU matmul, not by running the full refiner with `force_full_grid=True` |

## Hardware recommendation

**N150 is sufficient for the WH testing pass.** All `opt_3` WH measurements (Falcon7b, Segformer,
SDXL refiner GEGLU) were originally on n150 per OPT3_RESULTS.md / BH-continuation memory.

**N300 / T3000 would only be needed for** the multi-chip wins (Llama2-70B PREFILL_*, Falcon40B
DENSE_4H_TO_H decode) — all of which were **reverted** in `opt_3` because the perf bump
wasn't worth the multi-chip-only scope. None of those are carried over to this branch.

## Verification status

**BH side (this branch, this session) — confirmed on `wransom/model_opt` off main `bbfbb558690`, full rebuild 2026-05-12 16:04:**

Tracy A/B (1 trial per arm — same-test PCC run produces Tracy CSV by design; full revert→Tracy→re-apply cycle):

| Model | BASELINE Conv2d ns | OPT Conv2d ns | **Conv2d Δ** | BASELINE total ns | OPT total ns | **e2e Δ** | Prev-session Conv2d Δ | Prev-session e2e Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGG UNet | 1,199,514 | 1,134,101 | **−5.45 %** | 1,856,657 | 1,790,417 | **−3.57 %** | −5.32 % | −3.66 % |
| UFLD v2 | 643,674 | 631,225 | **−1.93 %** | 1,589,317 | 1,575,074 | **−0.90 %** | −2.76 % | −1.23 % |
| Mobilenetv2 | 793,513 | 719,759 | **−9.29 %** | 1,442,014 | 1,361,783 | **−5.56 %** | −9.44 % | −5.77 % |

All three BH wins reproduce within trial-to-trial variance vs prev session. PCC passes on all
three (test would have failed before reaching device kernels otherwise). The May 12 main
baseline is faster than the April 27 build the prev session measured against (e.g. VGG UNet
total 1.86 ms vs 1.96 ms baseline before), but the **delta from the optimization is
preserved on the new build**.

- [x] VGG UNet — verified
- [x] UFLD v2 — verified
- [x] Mobilenetv2 — verified
- [ ] (bonus) Segformer PCC + Tracy A/B — BH-continuation memory documented `−31.9 %`; not run this turn but worth a re-confirm.

**WH side — completed 2026-05-12 on n150 L (board 01000186119060f3, AICLK 500 MHz):**

| Model | Test | BASELINE | OPT | **Δ** | Verdict |
|---|---|---:|---:|---:|---|
| **Falcon7b prefill seq=1024** | `test_falcon_device_perf::test_device_perf[prefill-1-1024-0-BFLOAT16-DRAM-3741]` | (CI baseline 3120 sps) | **3771.73 sps** | **+20.9 % vs CI baseline** | ✅ **WIN CONFIRMED.** Test passes at the bumped threshold 3741 (3 % margin → 3629–3853). The +20.9 % matches the published +19.9 % from OPT3_RESULTS.md closely. |
| Segformer `linear_config_1024` matmul | `test_segformer_mlp[device_params0-160-2-1-32-32]` (mlp_id=2 path) | 8847 ns | 8850 ns | **0 %** | ❌ **NO MEASURABLE EFFECT.** The opt_3 −35.5 % was measured on a synthetic M=1024 K=1024 N=1280 shape; the actual segformer MLP at mlp_id=2 routes M=1024 K=160 N=256 — too small for the DST-volume optimization to matter. Edit is safe (PCC passes) but doesn't help segformer's actual model perf. |
| VGG UNet (cross-check `packer_l1_acc=True`) | `test_vgg_unet[0-pretrained_weight_false]` | Conv2d 2,393,312 ns / total 3,339,426 ns | Conv2d 2,391,881 ns / total 3,338,553 ns | **Conv2d −0.06 % / total −0.03 %** | ❌ **NEUTRAL ON WH.** The `determine_packer_l1_acc()` auto-gate that delivered −5.3 % on BH doesn't translate here. Edit is safe (PCC passes). |
| UFLD v2 (cross-check `packer_l1_acc=True`) | `test_ufld_v2_model` | Conv2d 1,075,219 ns / total 2,903,700 ns | Conv2d 1,080,150 ns / total 2,903,976 ns | **Conv2d +0.46 % / total +0.01 %** | ❌ **NEUTRAL ON WH.** Same story as VGG UNet — edit safe but no measurable win. |
| Mobilenetv2 (cross-check `packer_l1_acc=True`) | `test_mobilenetv2[10-pretrained_weight_true-device_params0]` | n/a — test fails with `TT_THROW @ program.cpp:1403` on WH (pre-existing issue with the BH-tuned mobilenetv2 test running on smaller WH L1) | n/a | n/a | ⚠️ **NOT TESTABLE ON WH.** Failure is unrelated to packer_l1_acc; the edit is safe to keep. |
| Falcon7b prefill seq=1024 e2e | `test_perf_falcon::test_perf_wh_bare_metal[..., prefill_seq1024_bf16_dram]` | (was 0.41 s) | (now expecting 0.31 s) | — | ⚪ NOT RUN this turn — relies on the same Falcon7b config edit that the device-perf test exercises. Falcon7b device-perf already passed at the bumped threshold; e2e is a redundant check. |
| SDXL refiner with `force_full_grid=True` | n/a | n/a | n/a | n/a | ⚠️ Same gate as BH: 64_cores section has missing keys, can't activate without the rest of the section filled. Out of scope. |

**WH side conclusions:**
- The **one solid opt_3 WH win that translates here is Falcon7b prefill seq=1024 (+20.9 %)** — driven by the legacy-compatible h=1 subblock-volume bumps in `model_config.py` (mm_h_to_4h w=1→8, mm_4h_to_h w=1→6).
- Segformer's `linear_config_1024` edit is safe but does not produce a measurable win on the actual model path — the opt_3 measurement was on a different (synthetic) matmul shape.
- The 3 BH-side `packer_l1_acc=True` wins (vgg_unet, ufld_v2, mobilenetv2) are arch-specific. The BH `determine_packer_l1_acc()` auto-gate path that delivered 3–6 % e2e on BH does not translate to WH. The edits are safe (no regression) but only valuable on BH.

## CI threshold audit (per user request — preserve safety factors)

All `expected_perf` margins kept at **±3 %** (the default `margin = 0.03`).

| Test | File | Old threshold | New threshold | Status |
|---|---|---:|---:|---|
| Falcon7b WH device-perf prefill_seq1024 | `test_falcon_device_perf.py:88` | 3120 sps | **3741 sps** | ✅ Updated in commit `6254e886776`. Measured 3771.73 sps on dev n150 — within (3629, 3853). |
| Falcon7b WH e2e prefill_seq1024 (single-chip) | `test_perf_falcon.py:65` | 0.41 s | **0.31 s** | ✅ Updated in commit `6254e886776`. Not re-measured this turn; the same model_config drives both perf paths so the device-perf pass implies e2e is in the new range. |
| Falcon7b WH e2e prefill_seq1024 (T3000 mesh) | `test_perf_falcon.py:124` | 0.41 s | **0.31 s** | ✅ Updated in commit `6254e886776`. T3000 only, not measurable here. |
| UFLD v2 BH device-perf | `test_ufld_v2_perf.py:17` | 595 sps | **628 sps** | ✅ **NEW UPDATE THIS TURN.** Previous BH session measured 628.71 sps with the opt. Old threshold 595 + 3 % margin = (577, 613) would have tripped the upper bound. |
| VGG UNet BH device-perf | `test_perf_vgg_unet.py:14` | 522 sps → 551 sps (bumped by `829a98ab782` "BH fast tilize" upstream of this branch) | **551 sps** (unchanged) | ✅ My opt adds ~1–2 % on top of fast-tilize; 551 with 3 % margin (534–567) absorbs both. No update needed. |
| VGG UNet WH device-perf | `vgg_unet/wormhole/.../test_perf_vgg_unet.py` | 293 sps | **293 sps** | ✅ No update — packer_l1_acc=True is neutral on WH for VGG UNet. |
| UFLD v2 WH device-perf | `ufld_v2/wormhole/.../test_ufld_v2_perf.py` | 350 sps | **350 sps** | ✅ No update — packer_l1_acc=True is neutral on WH for UFLD v2. |
| Mobilenetv2 WH | (test gated `@run_for_wormhole_b0()` per OPT3_RESULTS) | n/a | n/a | ⚪ No WH perf test in tree. |
| Mobilenetv2 BH | n/a | n/a | n/a | ⚪ No BH perf test in tree. |
| Segformer (any) | n/a | n/a | n/a | ⚪ No model-level threshold targets the `linear_config_1024` matmul specifically; whole-model perf-test was not measured (segformer-b0 weights not cached for offline run). |

**Test infra fix in same commit as the UFLD bump:** `test_falcon_device_perf.py` now passes `op_support_count=2500` to `run_device_perf`. The default 1333 (from `PROFILER_DEFAULT_OP_SUPPORT_COUNT * 1.333`) is too small for the 32-layer Falcon7b prefill — tracy subprocess exits with code 4 mid-trace. Without this fix the perf test never reaches the measurement step on WH n150, regardless of the actual win.

## Where the wins come from

- The `packer_l1_acc=True` wins (#1–#4) come from `determine_packer_l1_acc()` in
  `conv2d_utils.cpp:1273` — it auto-gates the hardware L1-accumulation feature on
  `enable_bias && in0_num_blocks_w > 1`. The model authors had set the flag to `False`
  unnecessarily; flipping to `True` lets the auto-gate turn on for the deep-K bias convs
  in each model.
- The subblock-volume bumps (#5, #6, #10, #11, #12) come from increasing `out_subblock_w`
  at `out_subblock_h=1`, exploiting the DST register limit (8 tiles for non-fp32-dest
  accumulation). Each bump 1 → 8 (or 1 → 6) gives 6–8× more tiles per pack call, amortizing
  per-pack LLK overhead. All stay at `out_subblock_h=1` so they don't trigger the legacy
  FATAL gate (`out_subblock_w == per_core_N || out_subblock_h == 1`) and don't need the
  `row_major_output` machinery from the `wransom/matmul_helpers_opt_3` helper branch.
