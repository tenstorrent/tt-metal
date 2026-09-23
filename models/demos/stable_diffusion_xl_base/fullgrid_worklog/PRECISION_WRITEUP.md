# SDXL 1024x1024 on Blackhole: where the precision goes, and why a worse GroupNorm made a better model

Companion to `WORKLOG.md` (Parts 4 and 5, 2026-09-17/18). Everything below was measured on a p150a with the
`mstaletovic/sdxl-gn-fullgrid` worktree; probe scripts are `test_precision_probe.py` (synthetic single-op gains)
and `stage_gains.py` (per-stage gains inside a transformer block from `SDXL_UPBLOCK_DUMP` dumps).

## 1. Summary

- Swapping the reference `ttnn.group_norm` for the agent-generated `groupnorm_sc_N_1_HW_C` made every module test
  MORE accurate, yet the 50-step UNet loop got WORSE (PCC vs torch fp32 0.915 -> 0.867, gate 0.905) and the image
  drifted (red visor, lost background detail).
- The GroupNorm op was never the problem. The model carries two **systematic gain errors** in other ops:
  1. Scaled-dot-product attention in LoFi shrinks the self-attention output by ~4% (operand truncation).
  2. `conv_in` / `conv_out` with bf16 DEST accumulation (fp32 off, packer_l1_acc off) inflate their output by ~3.7%.
- The reference GroupNorm happens to shrink its output by 3.8%. That shrinkage partially cancelled the inflation of
  the noise prediction, so the less accurate op produced a more accurate model. Removing the shrinkage (generated GN)
  exposed the uncompensated errors.
- Fix: SDPA HiFi2 + `conv_in`/`conv_out` on `CONV_HIFI2_NO_FP32_COMPUTE_CONFIG` (packer L1 accumulation on).
  Loop PCC 0.867 -> **0.962**, UNet step cost 54.01 -> 54.49 ms (+0.9%). Generated GN + fixes beats every other
  combination, including reference GN + fixes (0.899).

## 2. Metric: gain, not just PCC

PCC is insensitive to a uniform scale error. A tensor that is exactly right but 4% too small still has PCC 1.0.
Most of the errors here are exactly that kind, so the probes report the least-squares **gain**
`g = <out, ref> / <ref, ref>` (how much the device output is scaled relative to torch fp32), the std ratio, and the
mean bias. A random (unbiased) rounding error shows up as low rms but gain ~1; a systematic error shows up as gain
away from 1 even when rms is tiny. In a 50-step denoising loop the two behave completely differently: random error
averages out across steps, while a gain error on the noise prediction compounds every step.

## 3. Op-by-op findings

### 3.1 Generated GroupNorm (`groupnorm_sc_N_1_HW_C`): accurate, no bias

On the 46 real GN calls of one UNet step, compared against torch fp32:

| op | rms err | pcc | gain / bias |
|---|---|---|---|
| generated GN | 0.0014 | 0.999998 | ~1.000, no bias |
| reference `ttnn.group_norm` | 0.017 | 0.99988 | **std ratio 0.962** (3.8% too small) |

The generated op computes its statistics in fp32 DEST / HiFi4 by construction and its rms error is ~12x lower.
Every module test improves with it (resnets rms 0.007-0.028 vs 0.016-0.059; resnet output PCC 0.99984 vs 0.99958,
attention GN 0.99983 vs 0.99950, proj_in 0.99984 vs 0.99960). This op is good; no change recommended.

### 3.2 Reference `ttnn.group_norm`: 3.8% systematic shrinkage

Measured on the real UNet GN inputs: std ratio 0.962 with the reference op. It is a scale error, not noise, and it
sits on the input of every resnet conv and every transformer stack (the attention GN feeds `proj_in`). A scaled
input to a GN-free chain of linear ops scales the whole chain's output, so a 3.8% smaller GN output means a ~3.8%
smaller transformer/resnet contribution to the residual stream. This is what "compensated" the errors below.

**Root-caused 2026-09-21 (worklog Part 8, `test_gn_precision_isolation.py`): the approximate SFPU rsqrt.**
`ttnn.group_norm` defaults `math_approx_mode=true` and the reference kernels use the legacy-compat `rsqrt_tile<true>`,
whose APPROX path is biased ~2-3% low; with `math_approx_mode=False` the same kernel's gain is 1.003-1.006 (legacy) /
1.001 (Welford). The DEST mode is not the cause (fp32 DEST + approx still shrinks 2.6-3.2%). The earlier hypothesis of
an over-estimated variance in the bf16 statistics path was wrong. Confirmed experimentally by the other direction: multiplying the generated GN
output by 0.9624 (`SDXL_GN_SCALE_HACK`) recovers most of the up_blocks.0 gap (0.949 -> 0.965, reference 0.971).

### 3.3 Error 1: SDPA in LoFi shrinks self-attention output by 4-6%

Synthetic probe, random q/k/v, model shapes (1,20,1024,64) and (1,10,4096,64):

| SDPA fidelity | gain 1024 | gain 4096 |
|---|---|---|
| LoFi (shipped) | 0.941 | 0.940 |
| HiFi2 | 1.003 | 1.000 |
| HiFi4 | 1.008 | 1.006 |

Independent of `fp32_dest_acc_en` and `exp_approx_mode`. On real data inside transformer block 0 of up_blocks.0
the attn1 stage gain is 0.962 (LoFi) vs 0.998 (HiFi2). Cross-attention (96 encoder keys, flat softmax) is nearly
unaffected (attn2 gain 1.000), which is why only self-attention showed it.

Mechanism (tech_reports/matrix_engine): the FPU multiplier is 5b x 7b. LoFi feeds 1 hidden + 4 MSBs of SrcA (in1)
and 1 hidden + 6 MSBs of SrcB (in0) and truncates the rest toward zero, no rounding. Truncation toward zero is
biased: every product is slightly too small in magnitude. Verified on a plain matmul: RNE-rounding in1 to 4
mantissa bits and in0 to 5-6 bits on the host before a LoFi matmul makes the result exact (gain 1.00001); in1
alone accounts for ~1.95% loss, in0 ~0.28%. In SDPA in1 is K for Q.K^T and V for P.V, so scores come out ~2% too
small (flatter softmax) and P.V another ~2% too small: ~4% on real data, 6% on random data. This is a property of
the fidelity, not a kernel bug. Host pre-rounding of K and V to 4 bits (`SDXL_QKV_PREROUND=8,4,4`) gets up_blocks.0
from 0.9488 to 0.979 but stays below HiFi2 because Q and P keep their 6-bit truncation.

Cost of HiFi2 on the model shapes: 1024x1024 SDPA 83.0 -> 86.5 us, 4096x4096 422 -> 431 us, cross-attention
+0.2-0.4 us. The op is not FPU-bound; +3-4% of SDPA time, ~+0.45 ms per UNet step. HiFi4 would cost +12% at 1024
rows for no accuracy gain over HiFi2 (up_blocks.0 0.9912 vs 0.9897).

### 3.4 Error 2: bf16 DEST accumulation inflates `conv_in` / `conv_out` by ~3.7%

`conv_in` and `conv_out` shipped on `CONV_HIFI2_NO_FP32_NO_L1_COMPUTE_CONFIG` (HiFi2, fp32 DEST off, packer L1
accumulation off). Synthetic conv 3x3 320->32 @128x128 (the conv_out shape class):

| config | gain | bias |
|---|---|---|
| HiFi2, fp32 off, l1acc off (shipped) | **1.037** | +0.003 |
| HiFi2, fp32 off, l1acc on | 0.9998 | |
| HiFi2, fp32 on | 0.9976 | |
| HiFi4, fp32 off, l1acc off | 1.038 | |

Fidelity is irrelevant here; the inflation comes from accumulating in the 16-bit DEST register. A plain matmul with
K=1280 shows gain 1.018 for ANY number of K blocks when l1acc is off, and with l1acc on the gain tracks the number
of tiles accumulated inside DEST per block (in0_block_w 40/10/5/2/1 -> 1.018 / 1.011 / 1.005 / 1.0006 / 0.999).
So every MAC result rounded into the bf16 accumulator carries a magnitude-increasing bias of ~0.05-0.15% per
accumulated tile; the spill/reload of partials is not the cause. fp32 DEST avoids it. Packer L1 accumulation also
avoids most of it, not by extra precision (the L1 intermediate is bf16 when fp32 DEST is off) but because DEST is
reset every in0_block_w tiles and the packer's L1 add is unbiased in practice. Root cause in 3.6.

`conv_out` produces the noise prediction, so the shipped config scaled the UNet output by ~2% on every denoising
step. Switching both convs to `CONV_HIFI2_NO_FP32_COMPUTE_CONFIG` (l1acc on) is free in device time
(Conv2d 8.73 -> 8.73 ms).

### 3.5 Remaining, documented, not fixed

- **Matmul DEST inflation.** All model matmuls run HiFi2 / fp32 off / packer_l1_acc on with in0_block_w 2..16,
  giving +0.1% to +1.5% gain per matmul (FF stage gain 1.013-1.017 in block 0; transformer block outputs
  std ratio 1.007-1.019 vs torch). Tried fp32 DEST (`SDXL_MM_FP32=1`): FF gain 1.017 -> 1.003 but attention gain
  0.998 -> 0.990 (HiFi2 + fp32 DEST shrinks 0.28% per matmul via SrcB truncation), up_blocks.0 0.9897 -> 0.9869,
  subblock rules break for some sharded configs and DEST capacity halves. Not worth it. Cheaper lever if ever
  needed: smaller in0_block_w (more L1-acc rounds, each in fp32).
- **LayerNorm HiFi2**: gain 0.996 (HiFi4 0.9999). LN HiFi4 + fp32 adds +0.004 PCC on up_blocks.0 on top of SDPA
  HiFi2. LN is 2.6 ms/step; HiFi4 cost not measured.
- **Fast GELU**: gain 0.9976 with +0.005 mean bias.
- **Matmul fidelity and weight dtype are NOT the issue**: matmul HiFi4 (0.9494) and bf16 attention/FF weights
  (0.9492) did not move up_blocks.0 from the shipped 0.9488.

### 3.6 Root cause of the DEST inflation, measured (probe `test_dest_rounding_probe.py`, 2026-09-18)

A deterministic matmul on one core (in0 rows = multipliers, in1 columns = a base in K-tile 0 plus a delta in every
later K-tile) places the discarded bits of every accumulate step at an exact bf16 tie, just above or just below it,
for both signs, and compares the device against simulated accumulators. fp32 DEST off, packer L1 acc off, HiFi4
(LoFi identical). Results, all reproduced at bases 1, 256, 4096 and both signs:

| observation | evidence |
|---|---|
| **Ties round away from zero, not to even.** | 256 + 1.0 -> 258 (RNE: 256); -256 - 1.0 -> -258; 40 tie adds: 256 -> 336 = 256 + 40 x 2 (RNE stays 256). Ties-away model matches 964/1024 cells vs RNE 869. |
| **The K-tile is accumulated into DEST in two steps of 16 K-rows** (rows 0-15, 16-31), each rounded. | 16 x 1/16 in rows 0-15 or 16-31 -> tie -> up; the same 16 products in rows 8-23 -> 0.5 per half -> flat; 32 x 1/32 -> flat; 8 x 1/8 in each half -> two ups per tile. |
| **Products are rounded onto a grid 6 bits below the bf16 ULP of the DEST value before summation.** | Single product 0.984375 (31.5 grid units) rounds as a tie -> up; 0.96875 (31 units) is exact -> down. 16 x 0.046875 (1.5 units each, exact sum 0.75) -> up (each product became 2 units, sum 1.0 = tie); 16 x 0.0390625 (1.25 units) -> flat; 12 x 0.0625 (on grid) -> flat. Same at base 4096 with grid 0.5. |
| **Grid ties round toward +inf, final ties away from zero.** | 16 x +0.046875 -> up, 16 x -0.046875 -> flat (-1.5 units -> -1); but -256 - 0.9921875 (a non-tie, -31.75 units) -> -258, so negatives do round to nearest. |
| **fp32 DEST accumulates exactly; the pack to bf16 also rounds ties away from zero.** | fp32 on: exact-sum model matches 972/1024; every miss is a tie (4096 + 8 x 2 = 4112 -> 4128, RNE 4096). |
| Bias reproduced on the same one-core path with random data. | gain 1.0011 / 1.0045 / 1.0172 at 2 / 8 / 40 K-tiles (fp32 off), 1.0000 (fp32 on). |

Interpretation: every rounding in the 16-bit path is implemented as "add half an ULP, then truncate". On the
sign-magnitude final result that is ties-away-from-zero, which inflates magnitude symmetrically (gain > 1, mean
bias ~0). On the two's-complement alignment grid it is ties-toward-+inf, which adds the small positive mean bias we
saw (+0.003 on the conv). Neither is RNE, so the documented "round once, RNE" flow does not describe the 16-bit
DEST accumulate. Per K-tile there are 2 DEST roundings plus 32 grid roundings; with random bf16 data a tie at the
6-bit grid occurs about 1/64 of the time, each worth half a bf16 ULP of magnitude, which lands in the measured
0.05-0.15% per tile. fp32 DEST removes it because the accumulate is exact and only the final pack rounds.
Packer L1 accumulation removes most of it because DEST holds only in0_block_w tiles at a time and the packer's
L1 add is unbiased in practice (gain 0.999 at in0_block_w = 1).

## 4. Why the less precise GroupNorm gave the more precise model

Think of one UNet step as a product of gains on the noise prediction. Simplified, per transformer stack:

```
input -> GN (g_gn) -> proj_in -> [LN -> attn1 (g_attn) -> LN -> attn2 -> LN -> FF (g_ff)] x N -> proj_out -> residual
...                                                                              -> conv_out (g_conv) -> eps
```

Measured stage gains in block 0 of up_blocks.0 and the two edge convs:

| stage | reference GN path | generated GN, pre-fix | generated GN + fixes |
|---|---|---|---|
| GN output | 0.962 | 1.000 | 1.000 |
| attn1 (SDPA) | 0.962 | 0.962 | 0.998 |
| FF | ~1.013 | 1.013 | 1.017 |
| conv_out | 1.037 | 1.037 | 0.9998 |
| **UNet step std ratio vs torch** | **0.975** | **1.018** | **0.9965** |

On the reference path two shrinkages (GN 0.962, SDPA 0.962) fight two inflations (FF, conv_out DEST). The result
is a net 2.5% too small noise prediction. On the generated-GN path one shrinkage disappears and the net becomes
1.8% too LARGE. Neither is correct, and the reference GN adds 12x more random error on top, but the reference
path's net gain error happens to be a little smaller in magnitude and in the opposite direction.

Why a small gain difference dominates the loop: the scheduler subtracts the predicted noise from the latents every
step, so a systematic 1.8% over-prediction removes too much signal each step. Over 50 steps the generated-GN
latents developed a negative mean bias (-0.03 by step 3, -0.09 at the end) and shrank (std ratio 0.85 vs 0.92
reference). The per-step curves are identical for steps 0-5 and then diverge as the errors compound. Random
per-op rounding error, which is where the generated GN wins by 12x, does not compound this way; it is averaged out
by the denoising process. So "more precise op" (lower rms) and "more precise model" (smaller compounded gain error)
were measuring different things, and the reference GN's flaw happened to be a partial correction for two flaws
elsewhere: lucky quantization, not design.

This is also why "reference GN + both fixes" scores WORSE (0.899) than the reference path without the fixes
(0.915): once the SDPA and conv_out errors are gone, the reference GN's own shrinkage is uncompensated too.

## 5. End-to-end numbers

50-step UNet loop vs torch fp32, seed 0, `test_unet_loop`, gate 0.905:

| configuration | loop PCC |
|---|---|
| reference GN (main), shipped configs | 0.915 |
| generated GN, shipped configs | 0.867 (FAIL) |
| generated GN + SDPA HiFi2 | 0.900 |
| **generated GN + SDPA HiFi2 + conv io l1acc** | **0.962** |
| reference GN + SDPA HiFi2 + conv io l1acc | 0.899 |

up_blocks.0 module PCC (gate 0.968): reference 0.9707, generated shipped 0.9488, generated + SDPA HiFi2 0.9897,
+ SDPA HiFi4 0.9912, + LN HiFi4 fp32 0.9934. up_blocks.1 with both fixes 0.9965 (gate 0.993).

Image-level: the fixed image restores the reference's white visor, backpack colour, star and cloud detail
(`comparison_astronaut_4way.png`). Device cost of the fixes: UNet step 54.01 -> 54.49 ms, still 29% faster than
the 77.07 ms reference model.

## 6. How to enable / defaults

Both fixes are env knobs, not defaults, because the original constraint forbade fidelity / fp32 / l1acc changes
without sign-off:

```
SDXL_SDPA_FIDELITY=HiFi2 SDXL_CONVIO_COMPUTE=CONV_HIFI2_NO_FP32_COMPUTE_CONFIG
```

To adopt permanently: change the default in `tt_attention.py` (`sdpa_compute_kernel_config`) and in
`get_conv_compute_config` for conv_in / conv_out in `model_configs_1024x1024BH.py`.

## 7. Lessons

- Report gain and bias next to PCC for any op that sits on a residual or accumulating path; PCC alone hides a
  uniform scale error completely.
- Truncation-based fidelities (LoFi) are biased, not just noisy. Where the truncated operand is data (K, V), expect
  a magnitude loss proportional to the dropped bits, and do not expect fp32 accumulation to help.
- bf16 DEST accumulation is biased upward because its roundings are ties-away-from-zero (and ties-toward-+inf on
  the alignment grid), not RNE; long in-DEST accumulation without fp32 DEST or packer L1 accumulation will inflate
  the output. Prefer l1acc with short in-DEST runs, or fp32 DEST. The pack fp32 -> bf16 is ties-away too.
- When replacing an op with a more accurate one makes an end-to-end metric worse, look for a cancellation
  elsewhere before blaming the new op. Per-stage gain dumps found it in one afternoon.
