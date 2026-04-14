# Acceptable Error Thresholds for Activation Function Approximations

## Overview

This document defines strict and acceptable numerical error thresholds for activation function implementations on Tenstorrent hardware (Wormhole/Blackhole). Thresholds are derived from hardware precision characteristics, industry standards, and empirical data from the tt-metal test suite.

## Hardware Precision Context

### SFPU (Vector Unit) — Full IEEE 754 FP32

The SFPU is a 32-lane SIMD engine with 8 general-purpose 32-bit registers (LREG0-7) per lane. It operates at **genuine IEEE 754 single-precision** (1 sign + 8 exponent + 23 mantissa bits). SFPI instructions (`vFloat`, `vInt`, `dst_reg`) compute at full FP32 precision.

When `fp32_dest_acc_en=true` and data is unpacked directly to dest (`UnpackToDestMode::UnpackToDestFp32`), the data path bypasses SrcA/SrcB entirely:

```
L1 → Unpack → Dest (FP32) → SFPU reads dst_reg (FP32) → SFPU computes (FP32) → writes dst_reg (FP32) → Pack → L1
```

Evidence: `ckernel_sfpu_sqrt.h:122-129` — when `fp32_dest_acc_en=true`, full FP32 result is written to dest; when false, `float_to_fp16b()` explicitly truncates to bfloat16 before writing.

Sources:
- `METALIUM_GUIDE.md:450`: "Wormhole and Blackhole generations use 32-element vectors with 32-bit floating point operations"
- `runtime/sfpi/include/sfpi.h`: `vFloat` wraps `__xtt_vector` — standard IEEE 754 FP32

### FPU (Matrix Unit) — TF32-Limited

The FPU uses 5b x 7b multipliers (SrcA × SrcB). Even at HiFi4, maximum achievable precision is ~TF32 (19 active bits).

Source: `tech_reports/matrix_engine/matrix_engine.md:62-71`

### bfloat16 (FP16B) Format

- Machine epsilon: 2^-7 = 0.0078125 (~0.78%)
- Decimal precision: ~2-3 significant digits
- Maximum meaningful ULP: 2^7 = 128

Source: `tests/ttnn/utils_for_testing.py:239`

## Recommended Thresholds

### FP32 (fp32_dest_acc_en=true, APPROXIMATION_MODE=false)

Since the SFPU computes at genuine FP32, the right comparison point is NVIDIA's FP32 ULP targets.

| Level | ULP | rtol | atol | PCC | Use for |
|-------|-----|------|------|-----|---------|
| **Strict** | ≤ 2 | 1.3e-6 | 1e-5 | ≥ 0.9999 | Simple eltwise: exp, sigmoid, tanh, relu |
| **Acceptable** | ≤ 4 | 1e-3 | 1e-4 | ≥ 0.9996 | Composites: GELU, SiLU, softplus, atanh |

Rationale:
- NVIDIA achieves ≤2 ULP for exp, tanh, erf in FP32 (CUDA Math Appendix §17: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- rtol=1.3e-6 is PyTorch's FP32 default (https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py)
- 4 ULP matches NVIDIA's bound for erfc, pow, tan — reasonable for composed operations

### FP32 (APPROXIMATION_MODE=true)

Approximate mode uses lower-degree polynomials (~10-bit accuracy) for speed. Precision degrades deliberately.

| Level | ULP | rtol | atol | PCC | Use for |
|-------|-----|------|------|-----|---------|
| **Strict** | ≤ 8192 | 1e-3 | 1e-3 | ≥ 0.999 | Simple eltwise with fast approx |
| **Acceptable** | N/A | 1.6e-2 | 1e-2 | ≥ 0.99 | Composites with fast approx |

Rationale:
- 10-bit accuracy → ~3 decimal digits, comparable to NVIDIA intrinsics (__expf, __tanhf)
- ULP becomes less meaningful at this precision level; prefer allclose for acceptable tier

### FP16B / bfloat16 (fp32_dest_acc_en=false)

The SFPU still computes at FP32 internally, but results are explicitly truncated to bfloat16 via `float_to_fp16b()` before writing to dest. Error is dominated by this output truncation.

| Level | ULP | rtol | atol | PCC | Use for |
|-------|-----|------|------|-----|---------|
| **Strict** | ≤ 2 | 1.6e-2 | 1e-3 | ≥ 0.999 | Simple eltwise: exp, sigmoid, tanh, relu |
| **Acceptable** | ≤ 4 | 1.6e-2 | 1e-2 | ≥ 0.99 | Composites: GELU, SiLU, softplus, atanh |

Rationale:
- 1 ULP from FP32→bfloat16 truncation + 1 ULP from computation = 2 ULP strict
- rtol=1.6e-2 is PyTorch's bfloat16 default (~2× machine epsilon)
- atol=1e-2 matches TensorFlow's bfloat16 default (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/test_util.py)
- PCC 0.99 is the bfloat16 sweep test floor in tt-metal

## Hard Failure Boundaries

Below these thresholds, model quality measurably degrades:

| Metric | Threshold | Consequence |
|--------|-----------|-------------|
| PCC < 0.99 | Training divergence, inference degradation |
| Max absolute error > 0.1 | Gradient flow corruption in backprop |
| bfloat16 ULP > 128 | Values differ by more than an order of magnitude |

Source: Timmons & Rice, "Approximating Activation Functions" (https://arxiv.org/abs/2001.06370)

## Choosing Between ULP and Allclose

ULP and allclose measure fundamentally different things and are **not interchangeable**.

### ULP (`assert_with_ulp`)

Measures: |actual - expected| / ULP(expected)

This is a purely relative measure that scales with value magnitude:

| expected | 1 ULP (FP32) | 1 ULP (bfloat16) |
|----------|-------------|-------------------|
| 1.0 | 1.19e-7 | 0.0078 |
| 1000.0 | 6.1e-5 | 8.0 |
| 0.001 | 1.16e-10 | 9.5e-6 |

**Problem near zero**: ULP becomes astronomically strict. For expected=1e-30 (FP32), 2 ULP ≈ 2.8e-45. Most implementations cannot achieve this near the origin.

### Allclose (`assert_allclose`)

Measures: |actual - expected| ≤ atol + rtol × |expected|

Hybrid measure where atol dominates near zero and rtol dominates at large magnitudes.

**Problem at large magnitudes**: Fixed atol becomes irrelevant; if rtol is too loose, large-magnitude errors slip through.

### Recommendation: Use Both, Split by Value Range

```python
# Large/mid values: ULP catches relative precision regressions
mask = torch.abs(expected) > 1e-30
assert_with_ulp(expected[mask], actual[mask], ulp_threshold=2)

# Near-zero values: allclose with atol catches absolute deviations
mask_zero = torch.abs(expected) <= 1e-30
assert_allclose(expected[mask_zero], actual[mask_zero], atol=1e-5, rtol=0)
```

This pattern is already used in `tests/ttnn/unit_tests/operations/eltwise/test_swish.py`.

## Summary Table

| | FP16B Strict | FP16B Acceptable | FP32 Strict | FP32 Acceptable |
|---|---|---|---|---|
| **ULP** | ≤ 2 | ≤ 4 | ≤ 2 | ≤ 4 |
| **rtol** | 1.6e-2 | 1.6e-2 | 1.3e-6 | 1e-3 |
| **atol** | 1e-3 | 1e-2 | 1e-5 | 1e-4 |
| **PCC** | ≥ 0.999 | ≥ 0.99 | ≥ 0.9999 | ≥ 0.9996 |
| **Bottleneck** | bfloat16 output truncation | same | polynomial degree | polynomial degree |

## Sources

| # | Source | Location |
|---|--------|----------|
| 1 | PyTorch `_DTYPE_PRECISIONS` | https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py |
| 2 | NVIDIA CUDA Math Functions Appendix | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html (§17) |
| 3 | TensorFlow `assertAllCloseAccordingToType` | https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/test_util.py |
| 4 | Timmons & Rice, "Approximating Activation Functions" | https://arxiv.org/abs/2001.06370 |
| 5 | tt-metal `assert_with_ulp` | `tests/ttnn/utils_for_testing.py:163` |
| 6 | tt-metal `assert_with_pcc` | `tests/ttnn/utils_for_testing.py:88` |
| 7 | tt-metal `assert_allclose` | `tests/ttnn/utils_for_testing.py:122` |
| 8 | SFPU sqrt kernel (fp32 branching evidence) | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sqrt.h:115-132` |
| 9 | Matrix Engine tech report | `tech_reports/matrix_engine/matrix_engine.md:62-71` |
| 10 | Metalium Guide — SFPU precision | `METALIUM_GUIDE.md:450` |
