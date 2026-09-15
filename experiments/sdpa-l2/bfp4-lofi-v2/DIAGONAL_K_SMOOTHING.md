# Reversible diagonal K smoothing: conditional benefit, not a universal policy

Power-of-two channel equalization fixes a meaningful shared-exponent failure
mode without a mean-Q correction matmul. It strongly improves synthetic
inverse-scaled Q/K channel imbalance, including correlated low-rank inputs.
It does **not** uniformly improve genuinely salient Q/K channels, and does
not lower the ordinary-normal quantization floor. Prefer it as a candidate
for activation-conditioned qualification, not a default enabled by these data.

## Method and numerical contract

For each channel j, choose `s[j] = 2^e[j]`, where
`e = clip(round(log2(RMS_token(K) / median_channel_RMS)), -L, L)`.
Evaluate `Q' = Q*s`, `K' = K/s`, for L=3 and6. Then `Q'K'^T = QK^T`:
no score-scale change, output correction, or mean-Q correction is required.
The diagonal equivalence resembles the smoothing principle in
[SmoothQuant §4](https://arxiv.org/html/2211.10438v4), but this is our own
RMS/power-of-two attention experiment, **not** the paper's W8A8 linear-layer
method or its calibrated optimization recipe.

Power-of-two shifts preserve the original finite BF16 values in this tested
exponent range. Every transformed BF16 spill is explicitly performed and
checked to reconstruct the original Q/K exactly. We also assert
`RNE7(Q*s)/s == RNE7(Q)` exactly. Thus Q's effective quantization error in
original coordinates is unchanged; the benefit comes from changing the
relative sizes within the existing16-channel groups and their chosen exponents.
Overflow, subnormal flushing, and a hardware scaling primitive are not tested.

`diagonal_k_smoothing_models.py` generated
`diagonal-k-smoothing-v1.jsonl`: 160 cases, 22.1 s, four CPU threads,
H1/Q128/N32,768/D128, seeds1240/1241. No device jobs. Reference is original
BF16 Q/K/V with FP64 attention. The modeled candidate uses:

- Q rounded to seven significant bits with RNE.
- K/V B4: host-RNE native groups of16, three magnitude bits.
- K/V B8: per-value RNE5, native RNA BFP8 storage, then effective LoFi
  truncation to five significant bits. **B8 here is not full-precision
  seven-magnitude-bit BFP8 consumption.**
- Native FP32 exponent grid, represented P truncated to seven significant
  bits, and a denominator summing exactly that same P.
- FP64 QK accumulation, subtraction, online rescaling, and PV recurrence;
  BF16 final output. No device FPU alignment/accumulation, cheap subtraction,
  BF16 recurrence, or compensation effects are modeled.

The H16 control uses the previous fixed signed unnormalized transform on
both Q/K, real BF16 transformed spills, and score scale divided by16. No K
centering. All eight ordinary-normal K4 baseline/H16 controls exactly
reproduce the previous Hadamard JSON's L2 values.

## Inputs

All inputs start from independent normal BF16 Q/K/V; V stays normal.

| Name | Change to Q/K |
|---|---|
| Normal | None |
| Channel-outlier K | Every16th K channel multiplied by32; Q unchanged |
| Balanced channels | Same K×32 channels, inverse Q÷32; original logits equal the normal control |
| Joint salient channels | Every16th channel in both Q and K multiplied by4 |
| Structured balanced | Add four shared cosine channel modes with independent Gaussian row amplitudes, BF16 spill; then K×`2^(j%7-3)` and inverse Q scaling |

The structured case has correlated channels and non-IID logits, but its
channel imbalance is intentionally constructed. It is not a captured model
activation. Likewise, a channel-outlier K test is distinct from scattered
per-element outliers or common-mode offsets.

## Operator L2 results

Percent L2, seeds1240 /1241. Tables are generated from the JSON; raw results
also contain PCC, gain, rowwise L2, score error, and channel-scale metadata.

K4/V8 isolates the most important K4 sensitivity:

| Input | No transform | RMS clip3 | RMS clip6 | H16 BF16 |
|---|---:|---:|---:|---:|
| Normal | 11.912 / 12.248 | 11.912 / 12.248 | 11.912 / 12.248 | 12.069 / 12.236 |
| Channel-outlier K | 55.643 / 48.452 | 47.938 / 40.217 | 45.254 / 47.823 | 32.850 / 33.208 |
| Balanced channels | 69.238 / 70.352 | 22.058 / 22.507 | 11.912 / 12.248 | 55.551 / 57.165 |
| Joint salient channels | 48.211 / 34.843 | 42.715 / 40.204 | 42.715 / 40.204 | 18.343 / 17.837 |
| Structured balanced | 88.599 / 88.630 | 18.880 / 16.398 | 18.880 / 16.398 | 204.823 / 202.400 |

K8/V8 retains the same qualitative distinction:

| Input | No transform | RMS clip3 | RMS clip6 | H16 BF16 |
|---|---:|---:|---:|---:|
| Normal | 2.733 / 2.847 | 2.733 / 2.847 | 2.733 / 2.847 | 2.738 / 2.798 |
| Channel-outlier K | 12.044 / 12.916 | 9.520 / 10.609 | 9.406 / 10.784 | 7.463 / 9.701 |
| Balanced channels | 11.223 / 11.530 | 2.979 / 3.063 | 2.733 / 2.847 | 11.246 / 11.620 |
| Joint salient channels | 7.961 / 7.331 | 7.879 / 7.309 | 7.879 / 7.309 | 3.749 / 3.678 |
| Structured balanced | 23.136 / 24.076 | 3.210 / 3.011 | 3.210 / 3.011 | 30.856 / 31.859 |

V4 retains an independent error floor; diagonal K scaling cannot remove it:

| Input | K4/V4 baseline | K4/V4 RMS clip6 | K8/V4 baseline | K8/V4 RMS clip6 |
|---|---:|---:|---:|---:|
| Normal | 16.006 / 16.622 | 16.006 / 16.622 | 11.262 / 11.630 | 11.262 / 11.630 |
| Channel-outlier K | 56.551 / 49.849 | 46.499 / 49.054 | 16.653 / 17.490 | 14.883 / 15.907 |
| Balanced channels | 69.630 / 70.915 | 16.006 / 16.622 | 15.634 / 16.209 | 11.262 / 11.630 |
| Joint salient channels | 49.204 / 36.357 | 44.113 / 41.520 | 13.965 / 13.715 | 13.938 / 13.699 |
| Structured balanced | 88.819 / 88.682 | 22.226 / 19.939 | 26.264 / 26.522 | 12.253 / 11.950 |

## Why K representation L2 is not the right optimization objective

With balanced channels, the large K channel forces the shared BFP exponent
up while contributing no more to logits because its matching Q channel is
small. Small K channels that are important after Q weighting lose bits.
Full equalization recovers the ordinary-normal baseline exactly in this
constructed case. Partial equalization leaves a measurable penalty.

Conversely, when Q is also large in those channels, these channels really
matter. Equalization can surrender their locally favorable resolution. For
seed1241 joint-salient K4/V8, output L2 increases34.843→40.204%; H16 instead
achieves17.837%. Stronger equalization is also not monotonic: channel-outlier
seed1241 gives40.217% with clip3 versus47.823% with clip6.

For seed1240 balanced channels, untransformed K representation L2 is11.331%
but row-centered score L2 is77.740%; full equalization changes these to
11.739% and11.726%. The nominal K norm gets slightly worse while the useful
logit error collapses. The relevant first-order term is `Q * deltaK^T`, not
`norm(deltaK)/norm(K)`. Softmax sensitivity and near-tied winners then matter.

H16 is not a universal replacement either: in structured-balanced inputs it
reduces K representation L2 yet worsens centered logit error and creates
spurious sharp attention. For seed1240 K4/V8, mean maximum attention weight
is0.320 with H16 versus0.112 after RMS equalization; this is finite numerical
distortion, not a NaN failure. Rotations mix inverse-conditioned operands
and can expose cancellation to quantization.

## Engineering recommendation

Keep the diagonal option separate from Hadamard and mean centering. Its
application is O((Nq+Nk)D), potentially fusible into the existing Q/K
quantizers, and has no extra QK/PV matmul or score correction. Computing the
per-channel RMS is an additional reduction over K, with synchronization
and scale distribution; that cost is **excluded here**. Static calibrated
scales could avoid per-call reductions but require validation across layers,
heads, prompt lengths, and deployment distributions. Cached K also requires
a stable scale convention: changing scales mid-cache requires requantizing
old keys or supporting segmented conventions.

Before a device implementation or model-quality claim, collect real Q/K
channel RMS/max, Q-weighted quantization error, softmax margins/entropy, and
cross-layer/head scale stability. Compare identity, clipped diagonal,
rotation, and centering on the same captures; use held-out captures for
selection. These operator-L2 results neither establish SageAttention-like
model quality nor predict device speed. They establish a cheap *algebraic*
transformation worth evaluating where reciprocal channel conditioning is
observed, and an explicit counterexample to enabling K-only equalization
universally.
