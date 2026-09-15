# Full-head H128 versus H16: no compelling universal improvement

The missing CPU comparisons do not show a strong enough H128 advantage to
justify a new device implementation yet. Plain H128 modestly improves the
two scattered-outlier cases over the existing signed H16, but channel-K
outliers are mixed and common-Q remains dangerous. Adding one fixed random
sign diagonal to H128 is not uniformly beneficial.

## Evidence and scope

`hadamard_width_models.py` generated `hadamard-width-v1.jsonl`. It computed
only20 missing output cases in6.6 s with four CPU threads and reused44
completed cases from the source-pinned `qk-hadamard-native-models-v1.jsonl`
and `diagonal-k-smoothing-v1.jsonl`. No existing experiments were rerun.
Every record identifies whether it was computed or reused, and from where.

N32,768,Q128,H1,D128,seeds1240/1241. Original BF16 Q/K/V and FP64 attention
define the reference. Each rotation transforms **both** Q/K with the same
unnormalized Sylvester Hadamard, explicitly casts transformed tensors to
BF16, and divides the attention scale by the transform width. Signed variants
multiply both operands by the same fixed±1 diagonal first, seed20260915.
H128 means the full128-channel head, not mixing different tokens or heads.
The existing H16 comparison is **signed H16**, not plain H16.

Q is RNE7; K is native-group16 RNE BFP4. V is either RNE BFP4 or
RNE5→native RNA BFP8→effective LoFi5. The model uses native exp, trunc7 P
with a matched denominator, FP64 QK/subtraction/online correction/state,
and BF16 output. No mean centering, device FPU alignment, or hardware timing
is included. These are synthetic operator tests, not model-quality results.

## Results

Original-reference L2 percent, seeds1240 /1241, generated from raw JSON.

K4/V8:

| Input | None | Signed H16 | Plain H128 | Signed H128 |
|---|---:|---:|---:|---:|
| Normal | 11.912 / 12.248 | 12.069 / 12.236 | 11.582 / 11.551 | 11.493 / 11.576 |
| Outliers | 46.427 / 21.026 | 5.217 / 9.032 | 4.796 / 8.869 | 7.159 / 8.772 |
| Channel K | 55.643 / 48.452 | 32.850 / 33.208 | 29.943 / 36.716 | 33.184 / 38.987 |
| Common Q | 21.115 / 1.489 | 60.439 / 1.484 | 69.820 / 1.490 | 78.026 / 1.490 |

K4/V4:

| Input | None | Signed H16 | Plain H128 | Signed H128 |
|---|---:|---:|---:|---:|
| Normal | 16.006 / 16.622 | 16.295 / 16.419 | 15.826 / 16.026 | 15.844 / 16.001 |
| Outliers | 48.138 / 24.104 | 17.340 / 15.151 | 16.985 / 14.971 | 18.333 / 15.120 |
| Channel K | 56.551 / 49.849 | 34.606 / 35.022 | 32.011 / 38.345 | 35.160 / 40.500 |
| Common Q | 24.455 / 14.091 | 62.043 / 14.090 | 70.316 / 14.091 | 77.422 / 14.091 |

Channel-K means every16th K channel multiplied by32, Q unchanged. Outliers
are the earlier scattered per-element stress distribution. Common Q adds32.

## Interpretation and recommendation

Plain H128 retains the substantial rotation benefit on scattered outliers,
but improves over signed H16 by only0.421/0.163 percentage points for K4/V8.
With V4 the remaining V error dominates: gains shrink to0.355/0.180 points.
The channel-K comparison changes direction between seeds; it is not a robust
win. Normal improvements are small relative to the existing quantization
floor. This is not evidence that a full-head rotation achieves a new,
substantially better accuracy band on our hardware-friendly BFP4 format.

Common-Q seed1241 nearly saturates to one attention winner for every variant,
masking K-side errors; plain H128's mean maximum weight is0.999999992.
Seed1240 exposes the failure instead: both H128 versions are worse than the
already problematic H16. A single fixed random-sign realization neither
guarantees improvement nor exhausts possible rotations; tuning signs on
these two seeds would not establish generality.

Do not implement another rotation kernel based on this result alone. If real
activation captures have scattered outliers, plain H128 is a reasonable
additional CPU control, but compare its marginal gain with existing H16,
mean-Q correction, and diagonal K scaling on held-out captures. The separate
`Q_CENTER_HADAMARD.md` study shows why centering before rotation matters for
common Q; this width study does not test that combination or replace it.
No claim about SageAttention's end-to-end fidelity or CUDA performance
follows from these BFP4/LoFi simulations.
