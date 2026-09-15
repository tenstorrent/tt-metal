# Lower-degree exp refiner: CPU numerical design

A quadratic refiner is the first device candidate. In this model it removes
one of three dependent MADs while barely changing normal-input total L2 at
the LoFi five-bit-K/V and BFP4-K/V points. A linear refiner is plausible only
for a deliberately coarser BFP4 accuracy band: its approximately 1.75% exp
error is material against the roughly 2% five-bit-K/V baseline.

These are numerical results, **not measured speedups**. Existing kernels,
production modes, and frozen variants were not changed. A subsequent isolated
experimental header implements the candidates; its audit is documented below.

**Applicability correction after call-graph review:** the existing BF16 FAST
path does not execute this cubic refiner. Its non-FP32 branch calls
`exp_packthread_tile`, whose previously parsed API definition calls native
`calculate_exponential`. Refiner call sites are in the FP32/diagnostic branches.
Consequently, the first FAST degree-1/degree-2 smoke outputs were bit-identical
because the replacement was unreachable; they are inapplicable controls, not
successful FAST polynomial experiments. Degree reduction applies to the FP32
refiner path. Adding a refiner to FAST would create a new, more expensive exp
scheme, not optimize an existing FAST cubic.

## Actual grid and source provenance

This is not a polynomial fitted directly to ideal exp over the logits. The
driver reads the existing cubic coefficients from
[the frozen SDPA SFPU source](../hybrid-mixed-v1/candidate/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h)
(`init_sdpa_exp_grid`, `calculate_sdpa_exp_stream_effective`) and reproduces
the existing relative least-squares cubic fit to within 1e-9 relative coefficient
tolerance. The original fit is documented in
[check_exp_model.py](../perf-v3/check_exp_model.py).

For unscaled score-minus-maximum Δ at D128:

```
t = FP32_FMA(130.57785034179688, Δ, 31699.2734375)
j = nearest-ties-away signed-INT16(t), saturated to ±32767
linear = float_bitcast(abs(j) << 13), with sign restored
m = SETEXP(linear, 127)
result = linear * Horner(m, coefficients_folded_by_2^96)
result = max(result, 0)
```

The integer conversion matches the specified
[Blackhole signed-INT16 nearest-away conversion](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatInt.md).
For positive normal `linear`, m has 1024 fractional-grid points on [1,2);
the effective exponent after multiplication by the folded coefficients is
`floor(j/1024)-31`.

The deliberate grid bias introduces a common factor
0.970178232816 relative to exp. This factor cancels between
softmax numerator and denominator; it is not a 3% attention error. The
grid-only ideal-refiner ablation preserves this bias and INT16 quantization.

The CPU arithmetic emulates IEEE FP32 FMA with a double intermediate and one
FP32 rounding, then FP32 Horner MADs and final MUL. It does not claim
bit-for-bit SFPU rounding, unpack behavior, or exceptional/denormal semantics.
Negative grid values are ReLU-clamped; subnormal intermediate fractions are
recorded in JSON. The largest true attention mass below the negative-grid
cutoff was5.954e-8 in this suite. Device validation must still check
the positive subnormal-intermediate boundary.

## Targets and polynomial fits

Fit `m * poly(m)` by uniform-m relative least squares on 65536 points in [1,2]:

- Unbiased: `2^(m-1)`.
- Existing P7-compensated: `0.995 * 2^(m-1) + 1/128`.
  This anticipates truncating the resulting P to six fraction bits.
  It is not the right default for a full-P or explicitly RNE7-P recipe.

Coefficients are rounded to FP32 after folding in 2^96. Error here is relative
to the specified target, before exp-grid and P quantization.

| Target | Degree | Relative RMS% | Max relative error% | Per-value arithmetic |
|---|---:|---:|---:|---|
| Unbiased | 1 | 1.742 | 4.469 | 1 MAD + MUL |
| Unbiased | 2 | 0.201 | 0.657 | 2 MAD + MUL |
| Unbiased | 3 | 0.039 | 0.141 | 3 MAD + MUL |
| P7 compensated | 1 | 1.751 | 4.495 | 1 MAD + MUL |
| P7 compensated | 2 | 0.203 | 0.664 | 2 MAD + MUL |
| P7 compensated | 3 | 0.040 | 0.143 | 3 MAD + MUL |

Including the actual grid over logits [-20,0], the ideal refiner itself has
0.020% RMS error and 0.034% max error versus the common-scaled exact exponential.
The unbiased cubic/quadratic/linear have 0.044%/0.201%/1.733% RMS respectively.
P7-compensated fits intentionally include a small nonconstant bias, so their
pre-truncation error relative to true exp is not their target-fit residual.

All fits preserve the sign of underflow-encoded negative `linear`.
However, the new quadratic's refined mantissa can exceed 2:
2.006 for the compensated fit. Keep the final MUL, which handles
normalization. Do not replace it with an exponent overwrite that assumes
the polynomial output is confined to [1,2).

## Coefficient proposal

First candidate: compensated quadratic, descending coefficients, already
folded by 2^96:

```cpp
constexpr float a = 1.79879414e28f;
constexpr float b = -5.34102955e28f;
constexpr float c = 1.14346173e29f;
// y = linear * ((m * a + b) * m + c);
```

Unbiased quadratic, for an explicitly unbiased exp/P recipe:

```cpp
constexpr float a = 1.78788937e28f;
constexpr float b = -5.27842419e28f;
constexpr float c = 1.13613290e29f;
```

Coarse BFP4-only candidate: compensated linear:

```cpp
constexpr float a = 5.52623980e26f;
constexpr float b = 7.53273083e28f;
// y = linear * (m * a + b);
```

Nine significant digits preserve the proposed FP32 constants. Full stored
coefficients for every fit are authoritative in [raw JSON](exp-refiner-v1.jsonl).

## Attention experiment

192 configurations, four CPU threads, 15.5 seconds.
H1/Q128/D128; N4096/32768; seed 1240; normal, scaled-QK, sparse-outlier,
and common-K-centered inputs. Original BF16 inputs are the FP64 reference.

Three operand settings:

- Exact QK with original BF16 V.
- Q RNE7, K/V per-value RNE5 in BF16 storage.
- Q RNE7, K/V host/native-group RNE BFP4.

For common-K input, subtract the per-channel K mean in FP64, then store
centered K in FP32 before the selected encoding. This removes a per-query
constant from logits, preserving exact softmax; it does not insert BF16
centering spill error.

QK, subtraction, online maxima/correction factors, PV, recurrent state,
and division use FP64. P is truncated to seven significant bits and shared
between numerator and denominator. Final output is BF16. Thus this isolates
exp/refiner effects from the device's cheap subtraction and BF16 recurrence.

### Normal input, output L2%

| N | Operands | Exact exp | Current compensated cubic | Compensated quadratic | Compensated linear |
|---|---|---:|---:|---:|---:|
| 4,096 | exact_qk | 0.408 | 0.386 | 0.414 | 1.851 |
| 4,096 | q7_kv5 | 2.059 | 2.054 | 2.062 | 2.746 |
| 4,096 | q7_kv4 | 16.413 | 16.392 | 16.404 | 16.466 |
| 32,768 | exact_qk | 0.387 | 0.368 | 0.405 | 1.758 |
| 32,768 | q7_kv5 | 1.989 | 1.979 | 1.988 | 2.634 |
| 32,768 | q7_kv4 | 15.977 | 15.968 | 15.972 | 16.001 |

### Stress inputs at 32K, output L2%

| Input | Operands | Current compensated cubic | Compensated quadratic | Compensated linear |
|---|---|---:|---:|---:|
| scaled_qk | exact_qk | 0.280 | 0.260 | 1.385 |
| scaled_qk | q7_kv5 | 4.210 | 4.217 | 4.409 |
| scaled_qk | q7_kv4 | 31.753 | 31.820 | 31.572 |
| outliers | exact_qk | 0.193 | 0.190 | 0.539 |
| outliers | q7_kv5 | 3.833 | 3.855 | 3.787 |
| outliers | q7_kv4 | 48.024 | 48.019 | 48.132 |
| common_k_centered | exact_qk | 0.370 | 0.408 | 1.754 |
| common_k_centered | q7_kv5 | 1.458 | 1.469 | 2.257 |
| common_k_centered | q7_kv4 | 15.747 | 15.755 | 15.816 |

Across both lengths and all four distributions, quadratic-minus-current-cubic
changes total L2 by:

| Operands | Minimum change, percentage points | Maximum change, percentage points |
|---|---:|---:|
| exact_qk | -0.037 | 0.038 |
| q7_kv5 | 0.007 | 0.127 |
| q7_kv4 | -0.095 | 0.067 |

Occasional lower L2 with a poorer approximant is error cancellation, not
evidence that the approximation is intrinsically more accurate. The largest
KV5 quadratic regression is the 4K sparse-outlier case: 4.451%→4.578%.
The current cubic retains value in stricter modes.

The quadratic can change output by roughly 0.2–0.5% relative to the cubic
even when their total errors against reference are nearly identical. P
quantization thresholds move, so total-error similarity does not imply
bitwise or elementwise numerical equivalence.

## Implementation tradeoffs and next step

Keep the existing two-chain replay structure and grid setup. A quadratic
changes the refiner from 14 to 12 instructions per two vectors
(2 loads, 2 SETEXP, 4 MAD, 2 MUL, 2 stores), versus the current 6 MAD.
It also needs one fewer coefficient. Linear reduces this to 10 instructions.
These counts exclude setup, loop control, stalls, SFPU load macros, and all
other SDPA work; they are **not proportional kernel speedup estimates**.

Recommended first benchmark is compensated quadratic on the FP32 LoFi 5-bit-K/V
and FP32 coarse BFP4 paths, with existing P-format and denominator choices held
fixed. Test normal plus the 4K outlier regression. Only then consider linear
as an explicit coarser BFP4 tradeoff. Do not change the established ACCURATE
mode on the strength of this CPU model.

[Driver](exp_refiner_models.py) includes coefficient-reproduction assertions,
underflow-sign checks, and exact grid-bit reconstruction checks. No device
jobs were launched and existing sources remained unchanged.

## Isolated implementation and static audit

[exp_refiner.hpp](exp_refiner.hpp) is a new experimental header awaiting
device validation. The integration contract is to include it after the selected
frozen `compute_common.hpp` and before the streaming header; then define
`calculate_lofi_exp_refiner<iterations>` and install the call-site alias last.
The initial prototype explicitly included the hybrid SFPU header. Review found
that FAST selects a physically different frozen SFPU file, so that explicit
include must be removed when wiring FAST: `#pragma once` does not prevent
duplicate definitions from different physical files. The selected common
header already provides the shared SFPU helpers.

- `SDPA_LOFI_EXP_DEGREE=2`: quadratic, 12 replay instructions.
- `SDPA_LOFI_EXP_DEGREE=1`: linear, 10 replay instructions.
- No flag: original cubic remains untouched; invalid flagged degrees fail.
- The coefficient rule exactly matches the original stream refiner:
  `SDPA_DIAG_EXP_MODE` 1 or 4 selects unbiased coefficients, otherwise the
  P7-compensated target. `SDPA_MATCH_HIFI2` is not substituted for this rule.
- Only MATH/PACK threads define the replacement and alias. UNPACK is
  unaffected. Separate fused and load-macro accurate refiners are not aliased.
- The FAST non-FP32 exp branch is also unaffected. Driver guards must reject
  degree flags for that path rather than label a no-op as a measurement.

Register/replay audit:

- Replay slots 0–7 remain reserved for the grid; the new refiner occupies
  8–19 for degree 2 or 8–17 for degree 1.
- L0/L1 retain linear values, L2/L3 mantissas, L4/L5 independent Horner chains,
  L6 the highest coefficient. Degree 2 initializes constants L12/L13; degree 1
  initializes only L12. The next grid invocation must restore its own constants
  as it does for the existing cubic.
- Every dependent MAD/MUL is separated by the other chain's instruction;
  final MUL-to-STORE spacing matches the original. This satisfies the specified
  two-cycle MAD dependency spacing; no new LOADMACRO scheduling is introduced.
- Address modifiers 6/7 and load/store offsets 0/2 match the existing function.
  Only the second store advances DST by four; subsequent pairs replay exactly
  the recorded length. The caller's SFPU/PACK fence remains required.

A host-Clang preprocessing audit covered 54 combinations of thread, degree,
and diagnostic mode, with the dependency include stripped. It checked exact
proposed FP32 coefficient bits against JSON, replay instruction counts, and
alias/no-flag behavior; it did not test cross-snapshot includes. This is **not** a
Tenstorrent kernel compilation or device test. Compile and validate through the
experimental compute wrapper before using the candidate in measurements.
