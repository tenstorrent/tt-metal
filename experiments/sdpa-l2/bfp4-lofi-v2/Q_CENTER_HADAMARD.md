# Q centering controls H16's common-query regression

Centering Q **before** H16 largely repairs the common-Q regression while
preserving H16's scattered-outlier benefit. It does not repair H16's failure
on reciprocal channel imbalance; diagonal smoothing addresses a different
error mechanism. These are CPU numerical results, not device qualification.

## Contract and implementation options

`q_center_hadamard_models.py` generated `q-center-hadamard-v1.jsonl`:
96 cases,18.0 s,four CPU threads,H1/Q128/N32,768/D128,seeds1240/1241.
Original BF16 Q/K/V and FP64 attention define the reference. K uses native
group16 RNE BFP4; V uses either RNE BFP4 or RNE5→native RNA BFP8→LoFi5.
Q is finally RNE7. Native FP32-grid exp and trunc7 P use the same represented
P in the denominator; matmul, subtraction, online correction, and recurrent
state are FP64; final output is BF16. There is no device FPU alignment,
mean-reduction error, HiFi4 correction error, or timing model.

Let `mu = BF16(mean_FP64(Q))`. Every centered variant uses this same rounded
mean for subtraction and high-precision correction `mu @ ORIGINAL_K.T`.
Never substitute quantized K in that correction. With signed unnormalized
H16 transform T, `TT.T=16I`, the combined unscaled score is
`((Q-mu)T)(KT).T/16 + mu K.T`. Mean rounding cancels algebraically because
the same represented mean appears in both terms.

The primary combined option, center→H16, includes **two actual BF16 spills**:
the centered Q input to the BF16-input rotation matmul, and its transformed
output. Rotated K also spills BF16. A second control replaces the first
spill with RNE7, matching reuse of our existing centered-Q producer; final
RNE7 still occurs after rotation. Both are realizable storage contracts,
though the CPU transform omits device matmul arithmetic errors.

An after-rotation diagnostic first spills rotated Q/K BF16, then subtracts
the **exact transformed original BF16 mean** before Q7 and adds the same
original-K correction. This transformed bias is checked representable in
FP32, but need not be BF16: it is **not** directly supported by the current
BF16-bias centering helper. Rounding that bias independently while keeping
the old correction would be an inconsistent comparison.

Inputs are the existing normal, scattered-outlier, common-Q(+32), and
structured-balanced-channel cases. The latter adds four shared cosine
channel modes before reciprocal power-of-two Q/K channel scaling, as
documented in `DIAGONAL_K_SMOOTHING.md`; it is not a captured activation.
All24 uncentered normal/outlier/common-Q baseline/H16 measurements exactly
reproduce the earlier Hadamard JSON.

## Results

Output L2 percent, seeds1240 /1241, generated from JSON.

K4/V8:

| Input | None | Center only | H16 only | Center→H16, BF16 input | Center→H16, RNE7 input | H16→center, exact-bias diagnostic |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 11.912 / 12.248 | 11.872 / 12.228 | 12.069 / 12.236 | 12.025 / 12.205 | 12.022 / 12.212 | 12.017 / 12.197 |
| Outliers | 46.427 / 21.026 | 45.879 / 20.964 | 5.217 / 9.032 | 5.238 / 8.826 | 5.286 / 8.733 | 5.264 / 8.965 |
| Common Q | 21.115 / 1.489 | 4.443 / 1.483 | 60.439 / 1.484 | 5.230 / 1.483 | 5.230 / 1.483 | 5.519 / 1.482 |
| Structured balanced | 88.599 / 88.630 | 89.073 / 87.794 | 204.823 / 202.400 | 203.211 / 203.912 | 202.581 / 204.525 | 203.433 / 203.027 |

K4/V4:

| Input | None | Center only | H16 only | Center→H16, BF16 input | Center→H16, RNE7 input | H16→center, exact-bias diagnostic |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 16.006 / 16.622 | 16.019 / 16.646 | 16.295 / 16.419 | 16.244 / 16.421 | 16.243 / 16.433 | 16.232 / 16.414 |
| Outliers | 48.138 / 24.104 | 47.592 / 24.067 | 17.340 / 15.151 | 17.379 / 15.060 | 17.396 / 14.995 | 17.389 / 15.120 |
| Common Q | 24.455 / 14.091 | 11.702 / 14.090 | 62.043 / 14.090 | 11.998 / 14.090 | 11.998 / 14.090 | 12.112 / 14.090 |
| Structured balanced | 88.819 / 88.682 | 89.284 / 87.859 | 205.417 / 203.258 | 203.873 / 204.790 | 203.230 / 205.406 | 204.088 / 203.893 |

## Interpretation

Common Q amplifies K quantization error through `mu * deltaK.T`. Centering
routes this sensitive term through original K instead. For seed1240,
row-centered score L2 drops11.591% with H16 alone to0.369% when centering
before H16. The large common component also makes the **order of spills**
important: preprocessing-only centered-score error is0.005% when centering
first versus0.175% for the after-rotation exact-bias diagnostic. Subtracting
the mean after an already-rounded large rotation cannot undo that spill.

Common-Q seed1241 has almost one-hot attention: centered K4/V8 mean maximum
weight is0.9996, versus0.789 for seed1240's combined path. Its low1.483% output
L2 therefore does not establish uniformly accurate logits or common-mode
robustness. V4 leaves an11.998/14.090% output floor even after the Q-side
repair. Raw JSON contains PCC, gain, rowwise error, and entropy diagnostics.

The structured reciprocal imbalance has little removable token-constant Q
component. Centering cannot fix cancellation and shared-exponent error
introduced by mixing its inverse-conditioned channels. Combined H16 remains
over200% L2, compared with16.398–18.880% for diagonal RMS K4/V8 in the previous
study. These synthetic conditions are deliberately different; selecting a
transform from K norm alone is not justified.

## Recommendation

If pursuing H16 for observed scattered outliers, center Q before rotation
when a large block-common query component is present, and retain the
high-precision original-K correction. Reusing the existing RNE7 centering
producer is numerically plausible here; its extra quantization changes only
the last part of the tested accuracy band. It still needs end-to-end device
qualification and measured costs for both rotations, mean generation,
correction, and correction injection. No free preprocessing is assumed.

Do not make H16+centering universal or claim SageAttention-like model quality
from this suite. Keep diagonal scaling and K centering as separate controls,
then validate actual layer/head activation captures and downstream quality.
