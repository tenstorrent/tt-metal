# Signed Q/K Hadamard smoothing with native exp

**H16 is a promising conditional sparse-outlier treatment for K4, including
real BF16 transformed-input spills. It is not a universal accuracy fix.**
Common K must be centered; common Q can still require the more expensive
centering-plus-score-correction path. No device performance is measured here.

## Contract

[Driver](qk_hadamard_native_models.py),
[raw JSON](qk-hadamard-native-models-v1.jsonl): 116 CPU output records in
20.6 seconds, four threads. N32768/H1/Q128/D128, seeds1240/1241, normal,
sparse-outlier, scaled-QK, common-K and common-Q inputs. Original BF16 Q/K/V
and FP64 attention define the reference; final output is BF16.

Apply the same fixed random signs and unnormalized Hadamard to Q and K:
`T = diag(signs) H`, where `TTᵀ = hI`. H16 is block-diagonal over the 128
channels; H128 mixes the whole head. Q′=QT and K′=KT therefore give
`Q′K′ᵀ / (h sqrt(D)) = QKᵀ / sqrt(D)` before rounding. All matrix entries are
±1 (or zero between H16 blocks). No approximate 1/sqrt(h) transform coefficient
is used; the additional score scale 1/h is an exact power of two.

FP32 butterflies are followed by **BF16 spills**, then Q RNE7 and native-group
K RNE BFP4. FP32-spill controls are included for normal/outlier inputs. Optional
K centering is after the transformed spill: subtract the token-column mean
into FP32 before BFP4 encoding, without a second BF16 spill. This matches the
numerical shape of fused center/quantize, not a timed mean implementation.
No Q centering/correction is included.

V8 means RNE5 → native RNA BFP8 → LoFi trunc5; V4 is native-group RNE BFP4.
Native eight-bit exp includes the 1/h scale in its FP32 grid coefficient.
P trunc7 and its matched denominator are unchanged. QK, subtraction, online
correction exponentials and PV/state use FP64. The model excludes BF16 recurrence,
device cheap subtraction and additional transform-kernel arithmetic errors.
It is not a model of an arbitrary TTNN matmul configuration.

All ten seed1240 unrotated V-format/distribution controls reproduce the previous
asymmetric native-exp L2 exactly. The unnormalized transform/inverse identity
is checked in FP64 for both widths. Tables are generated from the linked JSON;
full precision remains there.

## K4/V8 output L2%, including BF16 transformed spills

Each cell is seed1240 / seed1241.

| Input | No rotation | H16 | H16 + center K | H128 | H128 + center K |
|---|---:|---:|---:|---:|---:|
| Normal | 11.912 / 12.248 | 12.069 / 12.236 | 12.102 / 12.257 | 11.493 / 11.576 | 11.568 / 11.588 |
| Sparse outliers | 46.427 / 21.026 | 5.217 / 9.032 | 5.185 / 9.133 | 7.159 / 8.772 | 6.753 / 10.163 |
| Q/K each ×2 | 29.361 / 30.381 | 31.775 / 30.129 | 31.326 / 30.190 | 27.623 / 29.237 | 27.311 / 29.404 |
| Common K +32 | 77.690 / 78.888 | 90.367 / 91.163 | 12.644 / 12.796 | 5374.502 / 5380.058 | 11.824 / 11.913 |
| Common Q +32 | 21.115 / 1.489 | 60.439 / 1.484 | 60.443 / 6.046 | 78.026 / 1.490 | 78.116 / 2.328 |

The extreme H128/common-K values are finite, not missing-data or percentage
formatting errors. For seed1240, mean maximum attention weight is 0.402 without
K centering versus 0.00124 with it; entropy is 2.595 versus 9.883. The reference
output RMS is only 0.00953, so false sharp winners cause enormous normalized
error. A plausible mechanism is token-dependent shared-exponent/rounding changes
in the rotated common component; that specific exponent-boundary mechanism is
not separately proven here. K centering removes the failure.

The common-Q case is strongly seed-sensitive. It is unsafe to declare success
from the seed1241 near-saturated attention case alone. K centering cannot remove
a key-dependent Q-mean score contribution; a corrected Q-centering experiment
would be a separate algorithm.

## K4/V4: V quantization remains after QK improves

| Input | No rotation, seeds1240 / 1241 | H16/BF16, seeds1240 / 1241 | H128/BF16, seeds1240 / 1241 |
|---|---:|---:|---:|
| Normal | 16.006 / 16.622 | 16.295 / 16.419 | 15.844 / 16.001 |
| Sparse outliers | 48.138 / 24.104 | 17.340 / 15.151 | 18.333 / 15.120 |
| Q/K each ×2 | 31.575 / 32.333 | 33.965 / 32.166 | 30.054 / 31.380 |

Thus QK smoothing can remove much of the extra outlier error, but cannot restore
V information. Even H16's improved K4/V8 outlier result remains 5–9% here; it is
not a sub-0.5% candidate. These results do not overturn the safer unrotated
K8/V4 asymmetric option identified in the [risk matrix](NUMERICAL_RISK_MATRIX.md).

## BF16 spill is measured, not assumed free

K4/V8 sparse-outlier L2%:

| Seed | H16 FP32 spill | H16 BF16 spill | H128 FP32 spill | H128 BF16 spill |
|---|---:|---:|---:|---:|
| 1240 | 5.168 | 5.217 | 6.708 | 7.159 |
| 1241 | 9.361 | 9.032 | 8.008 | 8.772 |

Pre-quantization inverse-transform roundtrip L2 is at most 0.176% for Q and
0.174% for K with BF16 spills. Nevertheless, H128's outlier output error changes
by up to 0.764 percentage points: downstream attention sensitivity can amplify
small input changes. Occasional improvements are error cancellation, not
evidence that BF16 is intrinsically more accurate than FP32.

## Suggested implementation experiment

Start with **H16 and existing K centering**, retaining Q7/K4 and V8 as the
cleanest K-side test. Compare against unrotated centered K4/V8 and unrotated
K8/V4 on actual model-derived activations before selecting it globally.
Normal inputs gain little; scaled-QK remains poor; common-Q requires an explicit
policy or a separately tested high-precision correction.

A HiFi4/FP32-accumulating ±1 transform followed by BF16 output is a reasonable
device candidate for this model. Validate transformed tensors against the CPU
oracle before attention: lower-fidelity transforms may discard input bits that
the current model preserves. Maintain channel orientation and the identical
sign pattern for Q and K; transpose K only after the logical transform. Retain
the 1/h score scale in both score exponentials and online rescale corrections.

H16 requires four butterfly stages versus H128's seven. A dense matmul
implementation may have different costs, especially if it fails to exploit the
H16 block structure. Include transform, BF16 spill, mean generation, quantization,
and synchronization in end-to-end timing; no speed claim follows from this CPU
study or from the mathematical O(N) scaling at fixed D. Keep the four locked
implementations unchanged while this remains exploratory.
