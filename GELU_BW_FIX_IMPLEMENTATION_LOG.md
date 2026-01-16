# GELU Backward Fix Implementation Log

**Issue:** GitHub #35971 - ttnn.gelu_bw() severe ULP errors
**Branch:** `ivoitovych/issue-35971-gelu-bw-ulp-fix`
**Date:** 2026-01-16

## Goal

Implement `gelu_bw` with similar qualities as `ttnn::tanh`:
- **Max ULP = 1** (accuracy)
- **Similar performance** (single polynomial evaluation, no expensive transcendental calls)

| Metric | ttnn::tanh | ttnn::gelu_bw (current) | Target |
|--------|------------|-------------------------|--------|
| Max ULP | 1 | 32,460 | 1 |
| Implementation | Sollya polynomial | Formula (erf, exp) | Sollya polynomial |
| Performance | Fast (single poly eval) | Slow (multiple transcendentals) | Fast |

## Current Status: NOT FIXED

The erfc-based fix does NOT work. Max ULP remains at 32,460.

### ULP Errors by Region (tested over all 65,026 valid BF16 values, DAZ+FTZ model)

| Region | Count | Mean ULP | Max ULP | Worst x |
|--------|------:|--------:|--------:|--------:|
| Deep negative (x < -5) | 16,095 | 6,255.61 | 32,460 | -3.376e+38 |
| Moderate negative [-5, -2] | 160 | 194.54 | 29,098 | -4.219 |
| Near negative [-2, -0.5] | 256 | 62.95 | 14,795 | -0.750 |
| Near zero [-0.5, 0.5] | 32,003 | 0.42 | 3 | -0.385 |
| Near positive [0.5, 2] | 256 | 0.27 | 1 | 0.504 |
| Moderate positive [2, 5] | 160 | 0.19 | 1 | 2.000 |
| Large positive (x > 5) | 16,096 | 2,748.22 | 16,203 | 3.376e+38 |
| **OVERALL** | **65,026** | 2,229.57 | **32,460** | -3.376e+38 |

## Root Cause Analysis

### Original Problem
The formula-based implementation uses:
```cpp
cdf = 0.5 * (1 + erf(x/√2))
pdf = exp(-x²/2) / √(2π)
GELU'(x) = cdf + x * pdf
```

When `erf(x/√2) ≈ -1` for large negative x, `1 + erf()` causes catastrophic cancellation.

### Why erfc() Fix Doesn't Work

Attempted fix: Use `erfc(-x/√2)` instead of `1 + erf(x/√2)`.

However, **ttnn::erfc() internally uses `1 - erf()`** (see `ckernel_sfpu_erf_erfc.h`):
```cpp
// For positive x (which erfc(-negative_x) becomes):
x = 1.0 - (calculate_erf_body<APPROXIMATION_MODE>(x));  // Same cancellation!
```

For `erfc(2.62)` when x = -3.7:
- `erf_body(2.62) ≈ 0.99997...`
- `erfc(2.62) = 1.0 - 0.99997 ≈ 0.00003` ← **Same catastrophic cancellation**

## How ttnn::tanh Achieves Max ULP = 1

Reference: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h`

**Key techniques:**
1. **Sollya-derived polynomial coefficients** (line 49-53):
   ```cpp
   // Polynomial coefficients found using Sollya
   sfpi::vFloat result = PolynomialEvaluator::eval(
       val, sfpi::vConst0,
       0.999004364013671875,
       3.0897438526153564453125e-2,
       -0.4890659749507904052734375,
       ...);
   ```

2. **PolynomialEvaluator::eval()** for Horner's method - numerically stable

3. **Clamping to valid range** after polynomial:
   ```cpp
   sfpi::vFloat threshold_value = sfpi::vConst1;
   sfpi::vec_min_max(result, threshold_value);  // Clamp to [-1, 1]
   ```

4. **Sign restoration** for odd function:
   ```cpp
   result = sfpi::setsgn(result, x);  // tanh(-x) = -tanh(x)
   ```

## Required Fix for GELU'(x)

To achieve Max ULP = 1, implement Sollya-derived polynomial approximation:

### 1. Derive Coefficients
Use Sollya to find minimax polynomial for GELU'(x):
```sollya
f = 0.5 * erfc(-x/sqrt(2)) + x * exp(-x^2/2) / sqrt(2*pi);
p = fpminimax(f, 6, [|SG...|], [-4, 4], floating);
```

### 2. Implement SFPU Kernel
```cpp
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_gelu_derivative_polynomial_(sfpi::vFloat x) {
    sfpi::vFloat result;

    // Asymptotic regions
    v_if(x < -5.0f) { result = sfpi::vConst0; }  // GELU'(x) → 0
    v_elseif(x > 5.0f) { result = sfpi::vConst1; }  // GELU'(x) → 1
    v_else {
        // Sollya-derived polynomial
        result = PolynomialEvaluator::eval(x, c0, c1, c2, c3, c4, c5, c6);

        // Clamp to valid range [-0.17, 1.0]
        v_if(result < -0.17f) { result = -0.17f; } v_endif;
        v_if(result > 1.0f) { result = sfpi::vConst1; } v_endif;
    }
    v_endif;

    return result;
}
```

### 3. Challenge: GELU'(x) is NOT Symmetric

Unlike tanh (odd function), GELU'(x) has different behavior on positive/negative sides:
- Positive x: GELU'(x) → 1 (monotonically)
- Negative x: GELU'(x) has minimum at x ≈ -0.751, then → 0

May require:
- **Piecewise polynomial** with multiple segments
- OR **Separate polynomials** for positive/negative regions
- OR **Higher degree** single polynomial

## Files Modified (Experimental - Not Working)

These files contain experimental SFPU polynomial code that is NOT currently used:

| File | Status |
|------|--------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Modified (unstaged) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Modified (unstaged) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` | Modified (unstaged) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` | Modified (unstaged) |
| `tt_metal/include/compute_kernel_api/eltwise_unary/gelu.h` | Modified (unstaged) |
| `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp` | Modified (unstaged) |

**Main composite (committed):**
| File | Status |
|------|--------|
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` | Committed (documents bug) |

## Test Files

- **C++ tests:** `tests/ttnn/unit_tests/gtests/test_gelu_bw_ulp_bug.cpp` (11 tests)
- **Python tests:** `tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp_bug.py` (33 tests)

## Build and Test

```bash
# Rebuild after kernel changes
cd ~/tt/tt-metal
touch ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp
~/tt/rebuild_unit_tests_ttnn.sh
~/tt/clear_kernel_cache.sh

# Run tests
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluBwUlp*"
```

## Next Steps

1. **Derive Sollya coefficients** for GELU'(x) over [-5, 5]
2. **Implement piecewise polynomial** if single polynomial insufficient
3. **Test with comprehensive BF16 sweep** to verify Max ULP = 1
4. **Consider fixing ttnn::erfc()** to not use `1 - erf()` internally

## References

- GitHub Issue: #35971
- tanh implementation: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h`
- erfc implementation: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erf_erfc.h`
- tanh_bw research (parallel work): `ivoitovych/bugfix-issue-35885-ttnn-tanh-bw-ulp-precision-draft`
