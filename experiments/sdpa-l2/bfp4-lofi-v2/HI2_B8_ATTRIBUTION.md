# HiFi2 BFP8 quantization attribution

Native BFP8 tie rounding explains a substantial positive gain, but not the whole
full-chip discrepancy. Replacing native nearest-ties-away (RNA) with host
nearest-ties-even (RNE) removes nearly all packing-induced gain in this model.
It does **not** make single-component BFP8 K/V a 0.5%-L2 solution: the remaining
normal-input error is about 1.2–1.3%.

## Contract

CPU only, four threads, 6.2 seconds. Input generator:
`REPRO.make_inputs(1, 128, N, 128, 1240, "normal")`. Original BF16 inputs are
the reference, Q is per-value RNE7, K/V use native row-wise groups of 16 along D.
All matmuls, online-softmax exponentials, correction factors, recurrent state,
and final division use FP64. Output is rounded to BF16 before the reported L2.

The effective HiFi2 model consumes all BF16/BFP8 K/V bits, truncates P to seven
significant bits, and uses those same P weights in numerator and denominator.
Online updates use K chunks of 512. P truncation first converts exp output to
FP32, matching the existing significand-model convention. The exact-P rows
below deliberately bypass this hardware operand restriction.

This is **not a device simulation**: it excludes cheap-exp coefficients,
finite-precision score subtraction, unpack effects, accumulation rounding,
and kernel scheduling. H1/Q128 rows differ from the H10/full-square sampled
rows in the full-chip results; comparisons across those two runs are indicative,
not an exact same-output error decomposition. All CPU variants within this
experiment do use identical original inputs and references.

Tables below are generated from [raw JSON](hi2-b8-attribution-v1.jsonl).
The [driver](hi2_b8_attribution_models.py) records source hashes and command
arguments. Native RNA and host RNE conversion rules come from the previously
device-validated [v1 packer probes](../bfp4-lofi-v1/REPORT.md).

## Attention: P truncated to seven bits, matched denominator

| K/V storage conversion | N | L2% | Gain error% | Gain-corrected L2% |
|---|---:|---:|---:|---:|
| BF16 | 4,096 | 0.591 | 0.047 | 0.589 |
| Native RNA BFP8 | 4,096 | 1.399 | 0.632 | 1.240 |
| Host RNE BFP8 | 4,096 | 1.253 | 0.063 | 1.250 |
| BF16 | 32,768 | 0.565 | 0.014 | 0.565 |
| Native RNA BFP8 | 32,768 | 1.331 | 0.539 | 1.211 |
| Host RNE BFP8 | 32,768 | 1.216 | 0.015 | 1.215 |
| BF16 | 262,144 | 0.584 | -0.009 | 0.584 |
| Native RNA BFP8 | 262,144 | 1.351 | 0.583 | 1.211 |
| Host RNE BFP8 | 262,144 | 1.264 | -0.004 | 1.264 |

Gain is the least-squares scalar `dot(actual, reference) / dot(reference, reference)`.
Gain-corrected L2 divides actual by that scalar before comparison; it is a
diagnostic, not a proposed runtime correction.

## 32K attribution

| Quantized inputs; others BF16 | L2% | Gain error% |
|---|---:|---:|
| Neither | 0.565 | 0.014 |
| RNA K only | 1.015 | 0.289 |
| RNA V only | 0.941 | 0.262 |
| RNA K and V | 1.331 | 0.539 |
| RNE K only | 0.960 | 0.009 |
| RNE V only | 0.942 | 0.021 |
| RNE K and V | 1.216 | 0.015 |

Native K and V each contribute positive gain. K changes attention temperature;
V changes weighted-output scale. These effects approximately add in the
combined normal-input experiment, but nonlinear softmax prevents an exact
additive attribution.

### K/V representation itself at 32K

| Conversion | Tensor | Representation L2% | Gain error% |
|---|---|---:|---:|
| RNA | K | 0.764 | 0.282 |
| RNA | V | 0.764 | 0.281 |
| RNE | K | 0.764 | -0.004 |
| RNE | V | 0.764 | -0.003 |

RNA and RNE have identical representation L2 here: only exact halfway ties
change direction, and either choice has equal absolute error at a tie.
RNE removes their correlated radial bias; it does not remove their variance.

### Removing P operand quantization at 32K

| K/V | P trunc7 L2% | Exact-P L2% | Exact-P gain error% |
|---|---:|---:|---:|
| BF16 | 0.565 | 0.443 | -0.008 |
| Native RNA BFP8 | 1.331 | 1.266 | 0.526 |
| Host RNE BFP8 | 1.216 | 1.159 | -0.003 |

With BF16 K/V and exact P, the remaining error is Q RNE7 plus BF16 output
rounding. With BFP8 K/V, removing P truncation helps only modestly: K/V
representation remains the dominant limit.

## Consequences for the full-chip control

The native BFP8 CPU model predicts roughly +0.5–0.6% attention gain, whereas
the device control measured approximately +1.06%. Packing bias is therefore
real and material, but it is not a complete explanation of the device gain.
The exact-P ablation retains the native-BFP8 gain, ruling out P truncation as
its source within this model.

Next isolate the remaining gap with identical sampled Q rows and device
intermediate/input checks, then compare full-FP32 subtraction and unbiased
exp plus explicit P RNE7. Keep the matched LoFi denominator: ordinary HiFi2
phases 0+1 still consume only high-seven-bit SrcB P, while refining SrcA V.
Changing the denominator to full P would introduce a different error.
See the [ISA fidelity definitions](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md)
and the private streaming header's denominator-phase comment.

An unbiased BFP8 device packer is a sound bias reduction, but these results
do not support promising residual-4+8-like accuracy from single-component
BFP8. No TT device was opened, and no production or frozen variant was changed.

