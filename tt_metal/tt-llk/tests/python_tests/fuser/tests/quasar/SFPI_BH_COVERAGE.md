# Quasar SFPI coverage after YAML consolidation

**The final BH-source-copy revision passed all 278 parity variants and three
controls.** The results below are from the fresh full-suite rerun, not the
historical pre-cleanup run. YAML cases/counts and input tensors are unchanged.
See [source-copy details and required adaptations](SFPI_BH_SOURCE_PARITY.md).

Validated on 2026-09-17 in `codex/quasar-sfpi-fuser-sweep-current`, based on
`marko/fuser-sweep` commit `1b1ac4875571ceb230ba62705d32a252ddb3e657`.
Scope is the 62 header families marked Blackhole SFPI / Quasar missing in the
2026-09-10 parity-dashboard snapshot (commit `3125b4ba98f`), not every SFPU
kernel or the public TTNN API.

## Result and interpretation

**All 278 parity-suite variants passed on the Quasar simulator**, covering
762 pipeline stages. Three existing fuser control variants also passed.
There were no missing, duplicate, or nonpassing parity IDs in the runtime-log
audit against the migration manifest.

The 168 YAML files use the branch's existing operation-list expansion.
No sweep implementation, generator, allocator, or comparison-policy changes
were made. Kernel registration, dispatch/init, and golden references were added.
See [authoring and consolidation notes](SFPI_BH_PARITY.md).

The correct coverage claim is **61 explicitly targeted families plus one
helper-only family (`conversions`)**. This is broad functional coverage, not
100% function, branch, template, numerical-domain, or hardware coverage.
Of the 278 variants, 276 / 760 stages target the dashboard families; two / two
stages are additional Gelu regression checks. There are 952 SFPU node invocations
and 83 distinct operation names when preparation and Boolean guards are included;
those are not 952 independent tests or 83 independently covered families.

Host verification additionally established exactly the original 278 configurations,
762 stages, and **762 bit-identical original input tensors**, including NaNs.
Consolidation preserves coverage rather than increasing it:
109 multi-stage files are unchanged; 169 single-stage configurations are expressed
through 59 operation-list templates. Files fell from the preceding 222-file
layout to 168 (24.3% fewer), or 39.6% fewer than the original 278-file layout.

## Formats, execution paths, and input breadth

| Storage / Dest mode | Variants | Stages | Direct-unpack stages | Ordinary Datacopy stages |
| --- | ---: | ---: | ---: | ---: |
| BF16 / 16-bit | 94 | 335 | 30 | 305 |
| FP32 / 32-bit | 106 | 349 | 49 | 300 |
| Int32 / 32-bit | 78 | 78 | 78 | 0 |
| Total | 278 | 762 | 157 | 605 |

All variants use homogeneous storage formats and default SyncHalf. All SFPU
nodes use 32 iterations and approximation mode false/default. There are
603 one-tile, 153 three-tile, and six two-tile stages. These are full-tile tests,
not partial-face, alternate-slot, multiblock, or synchronization stress tests.

53 variants use random inputs only, 130 constants only, and 95 both:
148 random-input stages and 614 constant-input stages. The default seed is 42
and resets per test. Default floating random values are approximately U(0.1,1.1);
integer random values are nonnegative, roughly half-range. Explicit constants
and signed preparation add negative cases, including -2147483647.

No FP16A storage, UInt16/UInt32 storage, or mixed input/output storage formats
are tested. The FP32-to-FP16A-mantissa cast tests still transport FP32 values.

### Important precision qualification

**300 of the 349 FP32 stages use ordinary SrcA ingress, which narrows the
mantissa in a TF32-style conversion.** Only 49 use the direct-unpack route,
including the discriminating cast guards. FP32 storage does not imply
full-precision FP32 input coverage at every tested SFPU.

The unchanged floating master policy is absolute tolerance 0.05, relative
tolerance 0.05, and PCC threshold 0.99. The secondary L1 comparison uses 0.1
absolute/relative tolerances. All 78 Int32 variants use exact master comparisons.
Floating passes establish functional agreement, not tight ULP accuracy or
bitwise Blackhole-versus-Quasar equivalence.

PCC is not an independent safeguard for uniform constants: undefined constant
correlations return 1, and tiny values can bypass PCC. Elementwise closeness
still applies. The absolute allowance can hide small missing correction terms
or errors in late, small tiled-product prefixes. Floating input-preservation
checks are tolerance-based, not bitwise. Matching NaNs do not check payloads;
positive and negative zero compare equal.

Evidence: [source-format model](../../../helpers/golden_generators.py),
[comparison policy](../../../helpers/utils.py),
[fuser checks](../../golden_check.py),
[unpack selection](../../quasar/unpacker/unpack_a.py),
[Datacopy path](../../quasar/fpu/datacopy.py), and
[input generation](../../../helpers/stimuli_generator/generator.py).

## What is well exercised

- **Signed integer behavior:** exact comparison for 78 variants, including
  bitwise operations, shifts, signed division/modulo, and partial reductions.
  Large negative inputs are included, though this is not exhaustive boundary coverage.
- **Classification:** five predicates × six input cases × two formats = 60
  variants. Cases are random, -1, zero, NaN, +Inf, and -Inf.
- **Selected activation boundaries:** shrink at +/-0.5, Hardtanh at +/-1,
  Threshold at/below/above 5, and piecewise sigmoid at both sides of +/-1 and
  +/-2 plus +/-4 saturation.
- **Discriminating cast checks:** direct FP32 values 0.5001220703125 and
  0.5003662109375 followed by UnaryEq(0.5) reject both no-op and truncating casts.
  These are quarter/three-quarter-ULP cases, not halfway-tie tests.
- **Output placement and preservation:** binary cases pack all three tiles;
  ternary adapters use tiles 0/1/2 -> 0 and check preserved tiles 1/2.
- **Independent references:** most numerical goldens use Torch/Python formulas;
  integer modulo uses sign-aware integer arithmetic, and structural goldens
  model tensor gather geometry instead of reproducing SFPI instruction loops.
  Sigmoid has separate mathematical-quality and BH-piecewise-contract tests.

## Per-family inventory

Counts are primary-target variants/stages, excluding incidental preparation
uses. Each YAML link is one representative file, not the complete family.
BF16/FP32 entries usually have separate YAMLs to retain correlated formats
without accidental Cartesian-product expansion.

| Dashboard family | Variants | Stages | Formats | Tested operations | Example |
| --- | ---: | ---: | --- | --- | --- |
| activations | 2 | 12 | BF16, FP32 | `Hardsigmoid` | [YAML](sfpi_bh_hardsigmoid_bf16.yaml) |
| add1 | 2 | 8 | BF16, FP32 | `Add1` | [YAML](sfpi_bh_add1_bf16.yaml) |
| addcdiv | 2 | 10 | BF16, FP32 | `ParityAddcdiv` | [YAML](sfpi_bh_ternary_addcdiv_bf16.yaml) |
| addcmul | 2 | 10 | BF16, FP32 | `ParityAddcmul` | [YAML](sfpi_bh_ternary_addcmul_bf16.yaml) |
| alt_complex_rotate90 | 2 | 12 | BF16, FP32 | `ParityAltComplexRotate90` | [YAML](sfpi_bh_structural_alt_complex_rotate90_bf16.yaml) |
| binary_bitwise | 15 | 15 | Int32 | `SfpuBitwiseAnd`, `SfpuBitwiseOr`, `SfpuBitwiseXor` | [YAML](sfpi_bh_sweep_binary_int32_negative_large.yaml) |
| binary_fmod | 8 | 15 | BF16, FP32, Int32 | `SfpuBinaryFmod`, `SfpuFmodInt32` | [YAML](sfpi_bh_binary_fmod_bf16.yaml) |
| binary_pow | 2 | 10 | BF16, FP32 | `SfpuElwpow` | [YAML](sfpi_bh_binary_pow_bf16.yaml) |
| binary_remainder | 8 | 15 | BF16, FP32, Int32 | `SfpuBinaryRemainder`, `SfpuRemainderInt32` | [YAML](sfpi_bh_binary_remainder_bf16.yaml) |
| bitwise | 12 | 12 | Int32 | `ScalarBitwiseAnd`, `ScalarBitwiseOr`, `ScalarBitwiseXor` | [YAML](sfpi_bh_sweep_bitwise_int32_negative_257.yaml) |
| bitwise_not | 6 | 6 | Int32 | `BitwiseNot` | [YAML](sfpi_bh_sweep_bitwise_int32_negative_257.yaml) |
| cast_fp32_to_fp16a | 3 | 7 | FP32 | `CastFp32ToFp16a` | [YAML](sfpi_bh_cast_fp32_to_fp16a_fp32.yaml) |
| cbrt | 2 | 12 | BF16, FP32 | `Cbrt` | [YAML](sfpi_bh_cbrt_bf16.yaml) |
| celu | 2 | 12 | BF16, FP32 | `Celu` | [YAML](sfpi_bh_celu_bf16.yaml) |
| conversions | 0 | 0 | Indirect | Helper only | Via BF16 power |
| digamma | 2 | 12 | BF16, FP32 | `Digamma` | [YAML](sfpi_bh_digamma_bf16.yaml) |
| div_int32 | 7 | 7 | FP32 | `ParityDivInt32Float` | [YAML](sfpi_bh_sweep_div_int32_float_near_half_fp32.yaml) |
| div_int32_floor | 10 | 10 | Int32 | `SfpuDivInt32`, `SfpuDivInt32Floor` | [YAML](sfpi_bh_sweep_binary_int32_negative_large.yaml) |
| elu | 2 | 12 | BF16, FP32 | `Elu` | [YAML](sfpi_bh_elu_bf16.yaml) |
| erf | 2 | 12 | BF16, FP32 | `Erf` | [YAML](sfpi_bh_erf_bf16.yaml) |
| erfc | 2 | 12 | BF16, FP32 | `Erfc` | [YAML](sfpi_bh_erfc_bf16.yaml) |
| erfinv | 2 | 10 | BF16, FP32 | `Erfinv` | [YAML](sfpi_bh_erfinv_bf16.yaml) |
| exp2 | 2 | 12 | BF16, FP32 | `Exp2` | [YAML](sfpi_bh_exp2_bf16.yaml) |
| expm1 | 2 | 12 | BF16, FP32 | `Expm1` | [YAML](sfpi_bh_expm1_bf16.yaml) |
| fmod | 2 | 16 | BF16, FP32 | `Fmod` | [YAML](sfpi_bh_fmod_bf16.yaml) |
| hardmish | 2 | 12 | BF16, FP32 | `Hardmish` | [YAML](sfpi_bh_hardmish_bf16.yaml) |
| hardshrink | 2 | 12 | BF16, FP32 | `Hardshrink` | [YAML](sfpi_bh_hardshrink_bf16.yaml) |
| hardtanh | 2 | 12 | BF16, FP32 | `Hardtanh` | [YAML](sfpi_bh_hardtanh_bf16.yaml) |
| heaviside | 2 | 8 | BF16, FP32 | `Heaviside` | [YAML](sfpi_bh_heaviside_bf16.yaml) |
| i0 | 2 | 12 | BF16, FP32 | `I0` | [YAML](sfpi_bh_i0_bf16.yaml) |
| i1 | 2 | 12 | BF16, FP32 | `I1` | [YAML](sfpi_bh_i1_bf16.yaml) |
| identity | 2 | 8 | BF16, FP32 | `Identity` | [YAML](sfpi_bh_identity_bf16.yaml) |
| int_sum | 8 | 8 | Int32 | `ParityIntSumCol`, `ParityIntSumRow` | [YAML](sfpi_bh_sweep_structural_int_sum_int32_negative.yaml) |
| isclose | 2 | 6 | BF16, FP32 | `SfpuIsclose` | [YAML](sfpi_bh_binary_isclose_bf16.yaml) |
| isinf_isnan | 60 | 60 | BF16, FP32 | `Isfinite`, `Isinf`, `Isnan`, `Isneginf`, `Isposinf` | [YAML](sfpi_bh_sweep_classification_bf16_nan.yaml) |
| lerp | 2 | 10 | BF16, FP32 | `ParityLerp` | [YAML](sfpi_bh_ternary_lerp_bf16.yaml) |
| lgamma | 2 | 10 | BF16, FP32 | `Lgamma` | [YAML](sfpi_bh_lgamma_bf16.yaml) |
| logical_not | 2 | 8 | BF16, FP32 | `LogicalNot` | [YAML](sfpi_bh_logical_not_unary_bf16.yaml) |
| logsigmoid | 2 | 10 | BF16, FP32 | `SfpuLogsigmoid` | [YAML](sfpi_bh_binary_logsigmoid_bf16.yaml) |
| mac | 2 | 10 | BF16, FP32 | `ParityMac` | [YAML](sfpi_bh_ternary_mac_bf16.yaml) |
| mask | 6 | 6 | BF16, FP32 | `SfpuMask` | [YAML](sfpi_bh_sweep_extra_mask_keep_mask_bf16.yaml) |
| polygamma | 2 | 10 | BF16, FP32 | `Polygamma` | [YAML](sfpi_bh_polygamma_bf16.yaml) |
| prelu | 2 | 8 | BF16, FP32 | `Prelu` | [YAML](sfpi_bh_prelu_bf16.yaml) |
| rdiv | 2 | 10 | BF16, FP32 | `Rdiv` | [YAML](sfpi_bh_rdiv_bf16.yaml) |
| remainder | 2 | 16 | BF16, FP32 | `Remainder` | [YAML](sfpi_bh_remainder_bf16.yaml) |
| rpow | 2 | 12 | BF16, FP32 | `Rpow` | [YAML](sfpi_bh_rpow_bf16.yaml) |
| rsub_int32 | 5 | 5 | Int32 | `SfpuRsubInt32` | [YAML](sfpi_bh_sweep_binary_int32_negative_large.yaml) |
| selu | 2 | 12 | BF16, FP32 | `Selu` | [YAML](sfpi_bh_selu_bf16.yaml) |
| sigmoid_appx | 4 | 32 | BF16, FP32 | `SigmoidAppx`, `SigmoidAppxParity` | [YAML](sfpi_bh_sigmoid_appx_parity_bf16.yaml) |
| sign | 2 | 8 | BF16, FP32 | `Sign` | [YAML](sfpi_bh_sign_bf16.yaml) |
| situ_glu | 10 | 10 | BF16, FP32 | `SfpuSituGlu` | [YAML](sfpi_bh_sweep_binary_modulo_situ_glu_fp32_random.yaml) |
| snake_beta | 2 | 10 | BF16, FP32 | `ParitySnakeBeta` | [YAML](sfpi_bh_ternary_snake_beta_bf16.yaml) |
| softcap | 2 | 12 | BF16, FP32 | `Softcap` | [YAML](sfpi_bh_softcap_bf16.yaml) |
| softshrink | 2 | 12 | BF16, FP32 | `Softshrink` | [YAML](sfpi_bh_softshrink_bf16.yaml) |
| softsign | 2 | 12 | BF16, FP32 | `Softsign` | [YAML](sfpi_bh_softsign_bf16.yaml) |
| tanhshrink | 2 | 12 | BF16, FP32 | `Tanhshrink` | [YAML](sfpi_bh_tanhshrink_bf16.yaml) |
| threshold | 2 | 8 | BF16, FP32 | `Threshold` | [YAML](sfpi_bh_threshold_bf16.yaml) |
| tiled_prod | 2 | 12 | BF16, FP32 | `ParityTiledProd` | [YAML](sfpi_bh_structural_tiled_prod_bf16.yaml) |
| unary_comp | 12 | 48 | BF16, FP32 | `UnaryEq`, `UnaryNe`, `UnaryGt`, `UnaryLt`, `UnaryGe`, `UnaryLe` | [YAML](sfpi_bh_unary_eq_bf16.yaml) |
| unary_power | 2 | 12 | BF16, FP32 | `UnaryPower` | [YAML](sfpi_bh_power_bf16.yaml) |
| unary_shift | 12 | 12 | Int32 | `LeftShift`, `RightShift` | [YAML](sfpi_bh_sweep_bitwise_shift_int32_negative_one.yaml) |
| xielu | 2 | 12 | BF16, FP32 | `Xielu` | [YAML](sfpi_bh_xielu_bf16.yaml) |

The `conversions` family contributes no standalone variants:
`_float_to_int32_positive_` is exercised by BF16 power paths, without an
independent conversion oracle.

`ParityDivInt32Float` calls the real `calculate_div_int32_float_body`,
not the production outer advancing wrapper. It transports FP32 values,
constructs small signed integers in Dest with a guaranteed nonzero denominator,
then restores input tiles and writes the quotient to tile 2. This is arithmetic
body coverage, not general mixed-format or complete division-wrapper coverage.

Structural tests cover the implemented partial row/column sums, lane-wise
prefix product, and adjacent-column complex rotation. MAC now uses the BH replay
source; its five/six-instruction recordings were checked in Quasar disassembly,
and both focused MAC runtime variants passed. This is not performance parity.

## Remaining gaps and recommended priorities

These are limitations of the tests, not failures demonstrated in the kernels.

1. **Make Isclose tests discriminate tolerance behavior.** Current random,
   exactly equal (2,2), and unequal (2,3) pairs could plausibly pass with an
   equality-only implementation. Add unequal-but-close and just-outside pairs
   through direct FP32 ingress. Dispatch currently fixes rtol=1e-5,
   atol=1e-8, equal_nan=false.
2. **Strengthen rounding and precision boundaries.** Add exact halfway cases
   with even/odd mantissas, negative counterparts, and exponent carry.
   Current cast guards do not distinguish ties-to-even from nearest-away.
   Expand direct-FP32 numerical checks before claiming FP32 accuracy.
3. **Cover scalar/mode branches deliberately.** All approximation modes are
   false. Dispatch fixes, for example, shift=3, bitwise mask=0x55,
   UnaryPower exponent=2, Rpow base=2, scalar modulo divisor=2,
   Polygamma order=1, and Addcdiv/Addcmul scale=0.5. Shift amounts 0/31/32+,
   unsigned types, and alternative scalar values are not tested.
   Operation lists alone do not add these branches.
4. **Add exceptional arithmetic and domain boundaries.** All 30 NaN/Inf-input
   variants are classification tests. Division avoids zero; the large-negative
   integer division case uses equal numerator/denominator, producing the easy
   quotient 1. There is no systematic near-limit quotient/remainder, reduction
   overflow, or zero-divisor coverage. Establish the legal SM32 range before
   introducing INT_MIN blindly.
5. **Expand nonlinear domains.** Gamma-family constants are positive;
   poles and negative nonintegers are absent. Erfinv tests stop at +/-0.9,
   omitting +/-1 and out-of-domain values. Binary power uses positive bases and
   exponents. There are no designed subnormal, tiny-normal, signed-zero, or
   floating-overflow-region cases.
6. **Cover alternate entrypoints and helpers.** The explicit omissions below
   prevent any claim of complete header/function coverage.
7. **Broaden seed and layout coverage after deterministic gaps.** Additional
   LLK_TEST_SEED runs require no fuser changes. They complement, rather than
   replace, boundary cases. Alternate tile assignments, partial iterations,
   larger shapes, synchronization stress, physical-hardware validation,
   performance, and public API integration remain outside this run.

| Family | Untested entrypoints/helpers |
| --- | --- |
| identity | `calculate_identity_uint` |
| mask | `calculate_int_mask`, `calculate_mask_posinf` |
| lgamma | `calculate_lgamma_adjusted`, `calculate_lgamma_stirling_fp32` |
| unary_power | `calculate_unary_power_iterative` |
| binary_remainder | `calculate_remainder_uint32` |
| remainder | `calculate_remainder_uint32_scalar` |
| rsub_int32 | `calculate_rsub_scalar_int32` |
| int_sum | `add_int` |
| conversions | Global `float32_to_bf16_rne`, distinct from the namespaced helper used by ternary kernels |

Recommendation: keep the existing fuser unchanged and treat this suite as the
functional baseline. First improve Isclose and cast discriminators; next add
explicit supported-domain boundaries and remaining entrypoints. Reuse an
operation-list template only when input domains, preparation, layout, formats,
and checking requirements genuinely match. Some remaining scalar/mode coverage
may need dispatch/reference exposure; do not infer YAML support for a parameter
that the current operation registration hard-codes.

## Runtime evidence

All batches ran on the same frozen YAML/kernel inputs and branch base.
The log audit matched exact expanded test IDs, not just aggregate pass counts.

| Batch | Parity variants passed | Pytest elapsed seconds |
| --- | ---: | ---: |
| b01 | 8 | 50.16 |
| b02 | 40 | 98.28 |
| b03 | 40 | 99.00 |
| b04 | 38 | 88.22 |
| b05 | 40 | 77.53 |
| b06 | 39 | 80.07 |
| b07 | 33 | 71.47 |
| b08 | 40 | 85.89 |
| Total | 278 | 650.62 |

b09 separately passed three controls in 49.43 seconds: the existing
`sfpu_unary` baseline and the upstream direct-FP32, approximation-false Exp
and Square sweep variants. These do not count toward the 278 parity variants.
The broad upstream `operation: all` sweep was not run.

Local logs: `/tmp/llk-sfpi-carbon-copy.atUjG9/b01/run.log` through
`b09/run.log`. Migration metadata, exact batch selectors, and the runtime audit are in
`/tmp/llk-sfpi-sweeps.rWeZkb/`; these are local verification artifacts, not
repository dependencies. The migrated YAMLs need no migration script to run.
