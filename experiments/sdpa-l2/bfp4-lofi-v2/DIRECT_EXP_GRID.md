# Native/direct exp grid with matched P denominator

Native approximate exp is a credible **coarse BFP4** numerical tradeoff: its
output error is almost identical to the degree-1 refiner tested earlier.
It is not a negligible change at the roughly 2%-L2 per-value-five-bit K/V point.
Widening the unrefined grid from 8 to 10 fraction bits buys very little.

## Contract and validation

72 CPU configurations, four threads, 5.4 seconds.
H1/Q128/D128, N4096/32768, seed 1240. Original BF16 inputs are the reference;
normal, scaled-QK, sparse-outlier, and common-K-centered distributions match
[the earlier exp-refiner study](EXP_REFINER_DESIGN.md).

Same operand settings and P rule: exact QK/BF16 V, Q7/KV5, or Q7/native-group
RNE BFP4 K/V; P truncated to seven significant bits; represented-P denominator.
QK, subtraction, online correction factors, PV/state, and division use FP64.
Grid operations use the earlier IEEE FP32 emulation. Output is BF16.
The driver reruns the compensated cubic control for every input/operand
setting and asserts exact agreement with the previous L2 results; all 24
controls passed.

New approximants:

- Native grid: the current
  [Blackhole approximate exp initialization](../../../tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h)
  uses `A=256*log2(e)*scale`, `B=32500.818359375`, nearest-away signed INT16,
  then shifts magnitude bits left 15 and restores the sign. No refiner.
  Negative outputs are ReLU-clamped.
- Direct 10-bit grid: existing specialized grid with shift 13, followed only
  by multiplication by2^96 to restore its exponent range. No polynomial.

This is a **FP32-state numerical model**, not a BF16 FAST simulation. It does
not include BF16 recurrent accumulation, device subtraction, SFPU-specific
rounding quirks, or a BF16 intermediate P pack. No device jobs or shared-source
edits were made.

Tables are generated from [raw JSON](direct-exp-grid-v1.jsonl); see the
[isolated driver](direct_exp_grid_models.py). One seed and eight input cases
do not establish qualification.

## Normal input: output L2%

| N | Operands | Cubic | Quadratic | Linear | Native, no refiner | Direct 10-bit |
|---|---|---:|---:|---:|---:|---:|
| 4,096 | exact_qk | 0.386 | 0.414 | 1.851 | 1.899 | 1.889 |
| 4,096 | q7_kv5 | 2.054 | 2.062 | 2.746 | 2.775 | 2.764 |
| 4,096 | q7_kv4 | 16.392 | 16.404 | 16.466 | 16.466 | 16.464 |
| 32,768 | exact_qk | 0.368 | 0.405 | 1.758 | 1.806 | 1.799 |
| 32,768 | q7_kv5 | 1.979 | 1.988 | 2.634 | 2.662 | 2.654 |
| 32,768 | q7_kv4 | 15.968 | 15.972 | 16.001 | 16.006 | 16.005 |

## Stress inputs: cubic versus native grid, output L2%

| N | Input | Operands | Cubic | Native, no refiner | Difference, percentage points |
|---|---|---|---:|---:|---:|
| 4,096 | scaled_qk | q7_kv5 | 4.244 | 4.314 | 0.070 |
| 4,096 | scaled_qk | q7_kv4 | 28.120 | 28.008 | -0.112 |
| 4,096 | outliers | q7_kv5 | 4.451 | 4.265 | -0.186 |
| 4,096 | outliers | q7_kv4 | 21.456 | 21.767 | 0.310 |
| 4,096 | common_k_centered | q7_kv5 | 1.695 | 2.542 | 0.847 |
| 4,096 | common_k_centered | q7_kv4 | 16.224 | 16.296 | 0.072 |
| 32,768 | scaled_qk | q7_kv5 | 4.210 | 4.420 | 0.210 |
| 32,768 | scaled_qk | q7_kv4 | 31.753 | 31.575 | -0.179 |
| 32,768 | outliers | q7_kv5 | 3.833 | 3.789 | -0.044 |
| 32,768 | outliers | q7_kv4 | 48.024 | 48.138 | 0.114 |
| 32,768 | common_k_centered | q7_kv5 | 1.458 | 2.284 | 0.826 |
| 32,768 | common_k_centered | q7_kv4 | 15.747 | 15.813 | 0.066 |

For BFP4, the largest total-L2 increase is 0.311 percentage points on the
4K outlier case. Some stress totals decrease through error cancellation.
The native output still differs from the cubic output by roughly 0.9–1.9% L2:
similar total accuracy bands do not mean numerically identical predictions.

For KV5, the normal-input increase is about 0.7 percentage points, and the
common-K-centered increase about 0.8 percentage points. Retain the quadratic
or cubic if preserving that more accurate band matters.

## Why degree 1 adds little here

After factoring out its common relative gain over logits[-20,0]:

| Approximation | Scalar shape RMS error% | Max shape error% | Common relative gain |
|---|---:|---:|---:|
| native_grid8 | 1.764 | 4.053 | 1.009796 |
| direct_grid10 | 1.762 | 3.955 | 1.009794 |

A matched denominator cancels the constant gain, but cannot cancel the
mantissa-dependent curvature error. The fitted degree-1 reciprocal-mantissa
polynomial is nearly constant, so it only slightly improves this native-grid
shape error. The extra mantissa-grid resolution also barely helps because
uncorrected curvature, not grid rounding, is the dominant error.

## Recommendation and implementation implications

1. At the coarse BFP4 point, test native approximate exp with FP32 state and
   matched represented-P sums before spending more effort on a degree-1
   replay. It potentially removes the entire refiner and its constant setup;
   only a device benchmark can establish the actual gain.
2. At the per-value KV5 point, quadratic remains the stronger numerical
   tradeoff. Native exp materially changes the normal accuracy band.
3. Keep the denominator matched to what PV consumes. Reducing exp arithmetic
   does not justify reverting to unmatched full-P sums.
4. Existing BF16 FAST already uses native approximate exp. Its prior
   degree-1/degree-2 flag smokes were no-ops because the FP32 refiner entrypoint
   is unreachable in that call graph. This proposal concerns removing the
   refiner from **FP32**, not making BF16 FAST's current exp cheaper.
5. This model uses accurate online rescale factors. Do not replace those
   small correction exponentials with uncorrected native exp: a constant
   standalone exp bias there does not cancel and can compound across chunks.

No performance improvement is claimed until the corresponding kernel is
implemented, compiled, timed, and checked on device.

