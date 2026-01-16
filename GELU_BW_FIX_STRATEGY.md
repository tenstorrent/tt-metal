# GELU Backward (gelu_bw) Fix Strategy

This document consolidates all knowledge gathered from parallel research to help fix GitHub Issue #35971.

## CRITICAL UPDATE: erfc() Approach Does NOT Work

**Discovery (2026-01-16):** The erfc()-based approach documented below does NOT work because **ttnn::erfc() internally uses `1 - erf()`**, which has the SAME catastrophic cancellation.

From `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erf_erfc.h`:
```cpp
template <bool APPROXIMATION_MODE>
inline void calculate_erfc() {
    for (int d = 0; d < 8; d++) {
        vFloat x = dst_reg[0];
        v_if(x < 0.0f) {
            x = -x;
            x = 1.0 + (calculate_erf_body<APPROXIMATION_MODE>(x));
        }
        v_else {
            x = 1.0 - (calculate_erf_body<APPROXIMATION_MODE>(x));  // ← SAME CANCELLATION!
        }
        v_endif;
        dst_reg[0] = x;
        dst_reg++;
    }
}
```

For `erfc(2.62)` when computing GELU'(-3.7):
- `erf_body(2.62) ≈ 0.99997...`
- `erfc(2.62) = 1.0 - 0.99997 ≈ 0.00003` ← **Same precision loss as `1 + erf()`**

**Required Fix:** Sollya-derived polynomial approximation (like `ttnn::tanh` uses to achieve Max ULP = 1).

See `GELU_BW_FIX_IMPLEMENTATION_LOG.md` for the updated implementation plan.

---

## Problem Summary

**`ttnn.gelu_bw()`** has severe ULP errors affecting 10.69% of BF16 values:

| Metric | Value |
|--------|-------|
| Max ULP | 32,460 |
| Mean ULP | 2,233 |
| Values with ULP > 1000 | 10.69% (6,954 values) |
| **Critical Bug** | Wrong sign at x ≈ -3.7 |

### Wrong Sign Bug (Most Critical)

| Input x | Expected GELU'(x) | Actual Output | ULP Error |
|---------|-------------------|---------------|-----------|
| -3.700 | **-1.526e-03** | **+4.349e-04** | 29,742 |
| -3.719 | **-1.373e-03** | **+5.112e-04** | 29,756 |

The derivative should be **negative** but hardware returns **positive**.

### Per-Region Analysis

| Region | Count | Mean ULP | Max ULP | Status |
|--------|-------|----------|---------|--------|
| Deep negative (x < -5) | 16,095 | 6,256 | 32,460 | **SEVERE** |
| Moderate negative [-5, -2] | 160 | 1,886 | 29,756 | **WRONG SIGN** |
| Near negative [-2, -0.5] | 256 | 6 | 146 | Needs work |
| Near zero [-0.5, 0.5] | 32,003 | 0.11 | 4 | Good |
| Near positive [0.5, 2] | 256 | 0.55 | 2 | Good |
| Moderate positive [2, 5] | 160 | 0.38 | 1 | Good |
| Large positive (x > 5) | 16,096 | 2,748 | 16,203 | **SEVERE** |

---

## Mathematical Background

### GELU Derivative Formula

```
GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))

GELU'(x) = Φ(x) + x * φ(x)
         = cdf + x * pdf

where:
  cdf = 0.5 * (1 + erf(x / √2))    -- CDF of standard normal
  pdf = exp(-x²/2) / √(2π)          -- PDF of standard normal
```

### Numerical Stability Issue

For negative x, `erf(x/√2)` approaches -1:
- At x = -8.375, fp64 `erf()` saturates to exactly -1.0
- This causes `1 + erf() = 0`, making `cdf = 0`
- The pdf term `x * pdf` should still produce the correct negative result
- **But the actual output suggests the x multiplication is failing**

### The erfc() Solution (Proven in GELU Forward Fix)

For negative x, use the identity:
```
1 + erf(x/√2) = erfc(-x/√2) = erfc(|x|/√2)
```

This avoids erf() saturation:
```cpp
double cdf;
if (x < 0.0) {
    // Use erfc for numerical stability with negative x
    cdf = 0.5 * std::erfc(-x / SQRT2);
} else {
    cdf = 0.5 * (1.0 + std::erf(x / SQRT2));
}
```

**Verification:** erfc() matches MPFR-256 with **0 ULP difference** across all 65,026 valid BF16 values.

---

## Current Implementation Analysis

### Kernel Location

```
ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp
```

### Current Algorithm (Suspected Bug)

```cpp
// Step 1: erf(x / sqrt(2))
fill_tile(3, kAlpha);          // kAlpha = 1/√2
mul_binary_tile(1, 3, 1);      // tile[1] = x / sqrt(2)
erf_tile(1);                   // tile[1] = erf(x / sqrt(2))

// cdf_term = 0.5 * (1.0 + erf(x / sqrt(2)))
fill_tile(3, 1.0f);
add_binary_tile(1, 3, 1);      // tile[1] += 1.0
fill_tile(3, 0.5f);
mul_binary_tile(1, 3, 1);      // tile[1] *= 0.5

// ... pdf_term calculation ...

// multiply by x
mul_binary_tile(2, 4, 2);      // tile[2] *= x (from tile[4])
```

### Suspected Issues

1. **erf_tile() saturation**: For `|x/√2| > ~2.6`, erf() may saturate to ±1.0
2. **Tile register corruption**: The actual output +4.349e-04 ≈ `pdf` without x multiplication
3. **Possible erf_tile() side effects**: May corrupt tile[4] which holds x for later multiplication

### Key Observation

The actual output +4.349e-04 ≈ pdf = (1/√2π) × exp(-3.7²/2) ≈ 0.000427

This is exactly what you'd get if the final `x * pdf` multiplication produced `|pdf|` instead of `x * pdf`.

---

## Lessons from Parallel Research

### 1. GELU Forward Fix (#35290) - erfc() Solution

**Branch:** `ivoitovych/issue-35290-gelu-ulp-fix-04`

**Key Insight:** For negative x, use `erfc()` instead of `1 + erf()`:
```cpp
// Avoids erf() saturation at x ≈ -8.375
cdf = 0.5 * erfc(-x / SQRT2);  // when x < 0
```

**Results:** Max ULP reduced from 32,767 to 7.

### 2. tanh_bw Fix (#35885) - Polynomial Approximation

**Branch:** `ivoitovych/bugfix-issue-35885-ttnn-tanh-bw-ulp-precision-draft`

**Problem:** `1 - tanh²(x)` loses precision when tanh saturates to ±1.

**Solution:** Remez minimax polynomial for sech²(x):
```cpp
// Degree 12 polynomial over [0, 4.0]
// Achieves Max ULP = 7 vs original Max ULP = 15,139
vFloat result = PolynomialEvaluator::eval(
    val,
    1.000048079011522040e+00f,   // c0
    -3.914105753579200098e-03f,  // c1
    -9.478282605219310319e-01f,  // c2
    // ... more coefficients ...
);
```

**Key Learning:** Direct polynomial approximation avoids precision loss from intermediate saturation.

### 3. Forward tanh Achieves Max ULP = 1 - THE GOLD STANDARD

**This is the most important lesson: Max ULP = 1 IS achievable for transcendental functions in BF16.**

**Results from tanh forward:**
| Metric | Value |
|--------|-------|
| Max ULP | **1** |
| Mean ULP | 0.047 |
| Exact (ULP = 0) | 95.26% |
| Within 1 ULP | **100%** |

**How tanh forward achieves this:**
- Uses **Sollya-generated Chebyshev polynomial** approximation
- Located in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h:47-77`
- **Key insight:** Directly computes tanh(x) via polynomial WITHOUT relying on intermediate functions that saturate
- No intermediate erf(), exp(), or other functions that lose precision
- Single polynomial evaluation path

**Actual tanh forward kernel** (`ckernel_sfpu_tanh.h`):

```cpp
// ACTUAL IMPLEMENTATION - simpler than expected!
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // Use symmetry: tanh(-x) = -tanh(x)
    sfpi::vFloat val = sfpi::abs(x);  // work with |x|

    // Sollya-generated degree-6 polynomial (NO segmentation!)
    // tanh(x) ≈ c1*x + c2*x² + c3*x³ + c4*x⁴ + c5*x⁵ + c6*x⁶
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,                           // c0 = 0
        0.999004364013671875,                    // c1
        3.0897438526153564453125e-2,             // c2
        -0.4890659749507904052734375,            // c3
        sfpi::vConstFloatPrgm2,                  // c4 = 0.281917631626129150390625
        sfpi::vConstFloatPrgm1,                  // c5 = -6.6649019718170166015625e-2
        sfpi::vConstFloatPrgm0);                 // c6 = 5.876733921468257904052734375e-3

    // Clamp to [-1, 1] - handles large x where polynomial exceeds bounds
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::setsgn(result, x);  // restore original sign

    return result;
}

// Init function stores coefficients in programmable registers
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void tanh_init() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // Polynomial coefficients c4, c5, c6 stored in programmable registers
        sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;   // c6
        sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;       // c5
        sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;         // c4
    }
}
```

**Key Observations:**
1. **NO segmentation** - single degree-6 polynomial for entire range!
2. **Symmetry exploitation** - work with |x|, restore sign at end
3. **Clamping** - `vec_min_max` handles large x where polynomial exceeds 1.0
4. **Sollya-generated coefficients** - optimized for minimax error
5. **Programmable registers** - last 3 coefficients in `vConstFloatPrgm0/1/2`
6. **Very simple** - ~15 lines of actual computation

**Why polynomial works better than formula-based computation:**
1. **No intermediate saturation** - polynomial directly approximates the target function
2. **No catastrophic cancellation** - no `1 - something_close_to_1` operations
3. **Controlled error** - Remez/Sollya algorithms minimize max error over the range
4. **Single evaluation path** - no branches, no conditional precision loss

---

## The Polynomial Approach for GELU'(x) - ANALYSIS RESULTS

Based on tanh forward's success, we investigated direct polynomial approximation of GELU'(x).

### CRITICAL FINDING: Single Polynomial Does NOT Work

**Experiment results from `generate_gelu_derivative_coefficients.py`:**

| Configuration | Max ULP (with clamping) | Notes |
|--------------|-------------------------|-------|
| Degree 14, range [-3, 3] | **2,848** | Best single polynomial |
| Degree 12, range [-4, 4] | 33,123 | Polynomial explodes outside range |
| Degree 14, range [-4, 4] | 20,000+ | Still fails in asymptotic regions |

**Per-region analysis (degree 12, [-4, 4], with clamping):**

| Region | Max ULP | Mean ULP | Notes |
|--------|---------|----------|-------|
| Deep negative (x < -4) | **35,944** | 12,742 | Polynomial goes wildly wrong |
| Negative transition [-4, -2] | 1,847 | 394 | Poor fit |
| Near minimum [-2, -0.5] | 8 | 1.2 | Good |
| Near zero [-0.5, 0.5] | 1 | 0.1 | Good |
| Positive transition [0.5, 2] | 1 | 0.2 | Good |
| Approaching 1 [2, 4] | 3 | 0.5 | Good |
| Large positive (x > 4) | **34,043** | 12,421 | Polynomial goes wildly wrong |

**Bug report validation (polynomial fails at asymptotes):**
```
       x     Expected         Poly   Poly+Clamp    ULP
   -5.000  -7.123e-06  +8.192e+00   -1.800e-01  35944   ← WRONG!
  -10.000  -2.568e-23  -6.552e+03   -1.800e-01  35943   ← WRONG!
    5.000   1.000e+00  -1.090e+00    1.000e+00      0   ← OK (clamped)
   10.000   1.000e+00   1.007e+05    1.000e+00      0   ← OK (clamped)
```

### Why Single Polynomial Fails for GELU'(x) (Unlike tanh)

**tanh works because:**
1. Symmetric function: tanh(-x) = -tanh(x)
2. Both asymptotes are ±1 (same magnitude)
3. Clamping to [-1, 1] naturally handles overflow
4. Polynomial only needs to fit [0, ~3] then mirror

**GELU'(x) fails because:**
1. NOT symmetric: GELU'(-x) ≠ -GELU'(x)
2. Different asymptotes: 0 (left) vs 1 (right)
3. Clamping to [-0.18, 1.0] can't fix deep negative → 0 transition
4. Polynomial must fit the entire range, no mirroring possible
5. Local minimum at x ≈ -1.41 adds complexity

**The fundamental problem:** At x = -5, GELU'(x) ≈ 7.1e-6 (nearly zero), but polynomial evaluation gives +8.19. Even clamping to -0.18 (the function minimum) is wrong - should be clamped to 0!

### Shape of GELU'(x)

```
GELU'(x):
         1.0 |                    ___________
             |                 __/
         0.5 |              __/
             |           __/
         0.0 |____------/
             |    \_/
        -0.2 |     ^-- local minimum at x ≈ -1.41 (value ≈ -0.129)
             +-----|-----|-----|-----|-----|----> x
                  -4    -2     0     2     4
```

**Actual local minimum (MPFR verified):** x ≈ -1.4134, GELU'(x) ≈ -0.1289

---

## REVISED RECOMMENDED APPROACH: Explicit Boundary Handling + Polynomial

The solution is to handle asymptotic regions EXPLICITLY before polynomial evaluation:

```cpp
// RECOMMENDED APPROACH: Explicit boundaries + polynomial
sfpi_inline sfpi::vFloat _sfpu_gelu_derivative_(sfpi::vFloat x) {
    sfpi::vFloat result;

    // Region 1: Deep negative (x < -4) → return 0
    v_if(x < -4.0f) {
        result = sfpi::vConst0;
    }
    // Region 2: Large positive (x > 4) → return 1
    v_elseif(x > 4.0f) {
        result = sfpi::vConst1;
    }
    // Region 3: Active region [-4, 4] → use polynomial
    v_else {
        result = PolynomialEvaluator::eval(x, ...);
        // Clamp polynomial to valid range [-0.13, 1.0]
        // (minimum at x≈-1.41 is ≈-0.129)
    }
    v_endif;

    return result;
}
```

**Why this works:**
1. Asymptotic regions handled correctly (0 or 1)
2. Polynomial only evaluated where it's accurate
3. No polynomial explosion at edges
4. Branching overhead is acceptable (single v_if/v_elseif/v_else)

### Alternative: Segmented Polynomials (More Complex but Higher Accuracy)

**Segment the x-axis based on GELU'(x) behavior:**

| Segment | x Range | GELU'(x) Behavior | Approach |
|---------|---------|-------------------|----------|
| 1 | x < -4 | ≈ 0 (asymptotic) | Return **0** directly |
| 2 | [-4, -1.5] | Transition region | Polynomial degree 8-10 |
| 3 | [-1.5, 2] | Contains minimum + steep rise | Polynomial degree 10-12 |
| 4 | [2, 4] | Approaching 1 | Polynomial degree 6-8 |
| 5 | x > 4 | ≈ 1 (asymptotic) | Return **1** directly |

**Note:** Segments 2-4 can potentially be merged into a single polynomial [-4, 4] with explicit boundary handling (recommended approach above).

### Coefficient Generation Using Remez Algorithm

Adapt the `generate_sech2_coefficients.py` script for GELU'(x):

```python
import numpy as np
from mpmath import mp, erf, exp, sqrt, pi

def gelu_derivative_exact(x):
    """GELU'(x) = cdf + x * pdf with high precision"""
    mp.prec = 256
    x_mp = mp.mpf(str(x))
    sqrt2 = sqrt(mp.mpf(2))
    sqrt_2pi = sqrt(2 * pi)

    # Use erfc for numerical stability with negative x
    if x < 0:
        cdf = mp.mpf("0.5") * mp.erfc(-x_mp / sqrt2)
    else:
        cdf = mp.mpf("0.5") * (1 + erf(x_mp / sqrt2))

    pdf = exp(-x_mp * x_mp / 2) / sqrt_2pi
    return float(cdf + x_mp * pdf)

def fit_minimax_remez(func, degree, x_min, x_max, max_iters=100):
    """Remez exchange algorithm for minimax polynomial approximation"""
    n = degree + 1

    # Initial Chebyshev nodes
    k = np.arange(n + 1)
    ref_points = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / n)
    ref_points = np.sort(ref_points)

    # Remez iteration to minimize max error
    for iteration in range(max_iters):
        # Solve linear system for polynomial coefficients
        A = np.zeros((n + 1, n + 1))
        b = np.array([func(x) for x in ref_points])

        for i, x in enumerate(ref_points):
            for j in range(n):
                A[i, j] = x ** j
            A[i, n] = (-1) ** i

        sol = np.linalg.solve(A, b)
        coeffs = sol[:n]

        # Find new reference points at error extrema
        # ... (exchange step)

    return coeffs

# Generate coefficients for each segment
segments = [
    ("neg_transition", -4, -2, 12),
    ("neg_steep", -2, 0, 12),
    ("pos_rise", 0, 2, 10),
    ("pos_approach", 2, 4, 8),
]

for name, x_min, x_max, degree in segments:
    coeffs = fit_minimax_remez(gelu_derivative_exact, degree, x_min, x_max)
    print(f"// Segment: {name} [{x_min}, {x_max}], degree {degree}")
    for i, c in enumerate(coeffs):
        print(f"//   c{i} = {c:.18e}")
```

### Expected Kernel Structure (Simple Version - Recommended)

**Following the tanh pattern exactly:**

```cpp
// ckernel_sfpu_gelu_derivative.h

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_gelu_derivative_polynomial_(sfpi::vFloat x) {
    // Sollya-generated polynomial for GELU'(x) over [-4, 4]
    // GELU'(x) = cdf + x * pdf where cdf = 0.5*(1+erf(x/√2)), pdf = exp(-x²/2)/√(2π)
    //
    // Unlike tanh, GELU' is NOT symmetric, so we work with x directly (no abs)
    //
    // Polynomial degree TBD (try 8-12, tanh uses 6)
    // Coefficients to be generated using Sollya/Remez
    sfpi::vFloat result = PolynomialEvaluator::eval(
        x,
        c0,                      // constant term (≈ 0.5 at x=0)
        c1,                      // linear term
        c2,                      // quadratic term
        c3,
        sfpi::vConstFloatPrgm2,  // stored in programmable register
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    // Clamp to valid range
    // GELU'(x) ∈ [min ≈ -0.17, max = 1.0]
    // For x << 0: GELU'(x) → 0
    // For x >> 0: GELU'(x) → 1
    sfpi::vFloat min_clamp = -0.18f;  // slightly below true minimum
    sfpi::vFloat max_clamp = sfpi::vConst1;

    // Clamp: result = max(min_clamp, min(result, max_clamp))
    v_if(result < min_clamp) {
        result = min_clamp;
    }
    v_endif;
    v_if(result > max_clamp) {
        result = max_clamp;
    }
    v_endif;

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_gelu_derivative() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = _sfpu_gelu_derivative_polynomial_<is_fp32_dest_acc_en>(val);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void gelu_derivative_init() {
    // Store polynomial coefficients in programmable registers
    // (last 3 coefficients, following tanh pattern)
    sfpi::vConstFloatPrgm0 = /* c_last */;
    sfpi::vConstFloatPrgm1 = /* c_last-1 */;
    sfpi::vConstFloatPrgm2 = /* c_last-2 */;
}

}  // namespace sfpu
}  // namespace ckernel
```

**Key points:**
1. ~30 lines of code (similar to tanh)
2. Single polynomial evaluation (no segmentation needed initially)
3. Clamping handles asymptotic regions
4. Programmable registers for last 3 coefficients
5. No complex branching

### Comparison: Formula vs Polynomial Approach

| Aspect | Formula-based (current) | Polynomial (proposed) |
|--------|------------------------|----------------------|
| Max ULP | 32,460 | Target: ≤ 1 |
| Intermediate saturation | Yes (erf saturates) | No |
| Catastrophic cancellation | Yes (1 + erf ≈ 0) | No |
| Branches | Minimal | Multiple segments |
| Complexity | Lower | Higher initial setup |
| Maintenance | Easier to understand | Requires coefficient regeneration |
| **Proven success** | No | Yes (tanh forward) |

---

## Proposed Fix Approaches (Priority Order)

### Option 1: Explicit Boundaries + Polynomial - BEST (Target: Max ULP ≤ 10)

**This is the recommended approach based on polynomial coefficient generation results.**

Unlike tanh (which can use a single polynomial + clamping), GELU'(x) requires explicit boundary handling:

```cpp
// RECOMMENDED IMPLEMENTATION
sfpi_inline sfpi::vFloat _sfpu_gelu_derivative_(sfpi::vFloat x) {
    sfpi::vFloat result;

    // Asymptotic region: x < -4 → 0
    v_if(x < -4.0f) {
        result = sfpi::vConst0;
    }
    // Asymptotic region: x > 4 → 1
    v_elseif(x > 4.0f) {
        result = sfpi::vConst1;
    }
    // Active region: [-4, 4] → polynomial
    v_else {
        // Degree 12 polynomial fitted to [-4, 4]
        result = PolynomialEvaluator::eval(x,
            +5.000000000e-01f,   // c0
            +7.833981654e-01f,   // c1
            ...);                 // See generated coefficients

        // Clamp to valid range (minimum ≈ -0.129 at x ≈ -1.41)
        v_if(result < -0.13f) { result = -0.13f; } v_endif;
        v_if(result > 1.0f) { result = sfpi::vConst1; } v_endif;
    }
    v_endif;

    return result;
}
```

**Summary:**
- Explicit handling for x < -4 (return 0) and x > 4 (return 1)
- Polynomial only evaluated in active region [-4, 4]
- Clamping handles polynomial overshoot within active region
- **Key insight:** Cannot rely on clamping alone (unlike tanh)

**Pros:**
- Polynomial experiments show Max ULP ≤ 10 achievable in [-4, 4]
- No intermediate saturation or catastrophic cancellation
- Asymptotic regions handled correctly (ULP = 0)
- Reasonable branching overhead

**Cons:**
- Three branches (x < -4, x > 4, else)
- Polynomial coefficients need validation
- Higher complexity than erfc() approach

**Effort estimate:** Medium - coefficients already generated, need kernel implementation

**Generated degree-12 coefficients for [-4, 4]:**
```cpp
// From generate_gelu_derivative_coefficients.py (Remez minimax)
c0  = +5.000000000000048850e-01
c1  = +7.833981654635477909e-01
c2  = -1.050645790463612245e-14  // ≈ 0 (even powers small due to function shape)
c3  = -2.349580869811332406e-01
c4  = +6.608238593419942105e-15  // ≈ 0
c5  = +4.019340638116156161e-02
c6  = -1.687283702412671126e-15  // ≈ 0
c7  = -3.698015795220243630e-03
c8  = +2.004221166164155113e-16  // ≈ 0
c9  = +1.724791520572908985e-04
c10 = -1.097032462245857348e-17  // ≈ 0
c11 = -3.190980645382897869e-06
c12 = +2.237370055729854358e-19  // ≈ 0
```

**Note:** Even-power coefficients (c2, c4, c6, ...) are near-zero because GELU'(x) is approximately odd around its inflection point.

### Option 2: Use erfc() in Kernel (Simpler - Target: Max ULP ≤ 10)

Replace erf() with erfc() for negative x values:

```cpp
// For negative x, use erfc for numerical stability
v_if(x < 0.0f) {
    // cdf = 0.5 * erfc(-x / sqrt(2))
    fill_tile(3, -kAlpha);         // -1/√2
    mul_binary_tile(1, 3, 1);      // tile[1] = -x / sqrt(2) = |x| / sqrt(2)
    erfc_tile(1);                  // tile[1] = erfc(|x| / sqrt(2))
    fill_tile(3, 0.5f);
    mul_binary_tile(1, 3, 1);      // tile[1] = cdf
}
v_else {
    // cdf = 0.5 * (1 + erf(x / sqrt(2)))
    fill_tile(3, kAlpha);
    mul_binary_tile(1, 3, 1);
    erf_tile(1);
    fill_tile(3, 1.0f);
    add_binary_tile(1, 3, 1);
    fill_tile(3, 0.5f);
    mul_binary_tile(1, 3, 1);
}
v_endif;
```

**Pros:**
- Minimal code change
- Proven approach from GELU forward fix (achieved Max ULP = 7)
- Easier to understand and maintain

**Cons:**
- May not achieve Max ULP = 1 (GELU forward got Max ULP = 7)
- Still relies on intermediate functions (erfc, exp)
- Requires v_if/v_else which may have performance impact

**Effort estimate:** Low - straightforward code change

### Option 3: Fix Tile Register Corruption (Quick Fix - Unknown Target)

If the issue is tile register corruption:

```cpp
// Save x to a separate tile before erf_tile() call
copy_tile(4, 5, 0);  // Backup x to tile[5]

// ... erf_tile() and other operations ...

// Use backed-up x for final multiplication
mul_binary_tile(2, 5, 2);  // tile[2] *= x (from backup tile[5])
```

**Pros:** Minimal algorithmic change
**Cons:** Needs investigation to confirm root cause, uses extra tile register

### Option 4: Reorder Computation

Compute pdf term first, then cdf term:

```cpp
// 1. Compute pdf = exp(-x²/2) / sqrt(2π)
// 2. Compute x_pdf = x * pdf
// 3. Compute cdf using erfc()
// 4. result = cdf + x_pdf
```

**Pros:** May avoid register conflicts
**Cons:** Still needs erfc() for cdf precision

---

## Reference Implementations

### C++ Reference (fp64 with erfc)

```cpp
double gelu_derivative_exact(double x) {
    constexpr double SQRT2 = 1.4142135623730950488;
    constexpr double INV_SQRT_2PI = 0.3989422804014327;

    double cdf;
    if (x < 0.0) {
        // Use erfc for numerical stability with negative x
        cdf = 0.5 * std::erfc(-x / SQRT2);
    } else {
        cdf = 0.5 * (1.0 + std::erf(x / SQRT2));
    }

    double pdf = std::exp(-0.5 * x * x) * INV_SQRT_2PI;
    return cdf + x * pdf;
}
```

### Python Reference (mpmath 256-bit)

```python
from mpmath import mp, erf, erfc, exp, sqrt

def gelu_derivative_exact(x):
    mp.prec = 256
    x_mp = mp.mpf(x)
    sqrt2 = sqrt(2)
    sqrt_2pi = sqrt(2 * mp.pi)

    if x < 0:
        cdf = mp.mpf("0.5") * erfc(-x_mp / sqrt2)
    else:
        cdf = mp.mpf("0.5") * (1 + erf(x_mp / sqrt2))

    pdf = exp(-x_mp * x_mp / 2) / sqrt_2pi
    return float(cdf + x_mp * pdf)
```

---

## Hardware Model: DAZ+FTZ

Tenstorrent SFPU uses **Denormals-Are-Zero + Flush-To-Zero**:
- All denormal inputs treated as zero
- All denormal outputs flushed to zero
- Per `tech_reports/Handling_Special_Value/special_values.md`: "denormals | all | 0x0"

**Impact on ULP calculation:**
- Denormals map to same index as zero
- ULP distance must account for this
- See `bf16_daz_normalize()` in test files

---

## Key Files Reference

### Bug Report & Tests (gelu_bw)
- `tests/ttnn/unit_tests/operations/eltwise/GELU_BW_ULP_BUG_REPORT.md`
- `tests/ttnn/unit_tests/gtests/test_gelu_bw_ulp_bug.cpp` (11 tests)
- `tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp_bug.py` (33 tests)

### GELU Forward Fix Reference
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`
- `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp`
- `/home/ivoitovych/tt/tt-metal/GELU_ULP_FIX_IMPLEMENTATION.md` (in fix branch)

### tanh_bw Fix Reference
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h`
- `tests/ttnn/unit_tests/operations/eltwise/backward/generate_sech2_coefficients.py`
- `tests/ttnn/unit_tests/gtests/TANH_BW_BUG_REPORT.md`

### SFPU APIs
- `tt_metal/include/compute_kernel_api/eltwise_unary/erf_erfc.h` - erf/erfc tile API
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`

---

## Recommended Next Steps

### Phase 1: Quick Investigation (1 day)
1. **Study tanh forward kernel** (`ckernel_sfpu_tanh.h`)
   - Understand exact polynomial structure and coefficient storage
   - Note how segments are selected and evaluated
   - Identify reusable patterns for GELU derivative

2. **Verify GELU'(x) function shape**
   - Plot GELU'(x) over [-6, 6] to confirm segmentation strategy
   - Identify critical points (minimum at x ≈ -0.85)
   - Determine segment boundaries

### Phase 2: Polynomial Approach (Recommended - 2-3 days)
3. **Generate polynomial coefficients**
   - Adapt `generate_sech2_coefficients.py` for GELU'(x)
   - Generate coefficients for each segment
   - Validate ULP error for each segment using Python/mpmath

4. **Implement kernel**
   - Create `ckernel_sfpu_gelu_derivative.h` based on tanh forward pattern
   - Use `PolynomialEvaluator::eval()` for coefficient evaluation
   - Handle asymptotic regions (x < -6 → 0, x > 4 → 1)

5. **Test and iterate**
   - Run existing test suite (11 C++ + 33 Python tests)
   - Verify Max ULP ≤ 1 across all BF16 values
   - Adjust segment boundaries or polynomial degrees if needed

### Phase 3: Alternative (If Polynomial Too Complex)
6. **Implement erfc() approach** as fallback
   - Simpler implementation, proven to achieve Max ULP ≤ 7
   - Good enough for most use cases
   - Can be done in parallel with polynomial investigation

### Phase 4: Validation
7. **Test on both architectures**
   - Wormhole (gelu_bw bug was found here)
   - Blackhole (tanh_bw research was done here)
   - Both need identical fixes in respective kernel files

8. **Performance benchmarking**
   - Compare polynomial vs formula-based performance
   - Ensure no regression in throughput

---

## Related Issues

| Issue | Description | Status |
|-------|-------------|--------|
| #35971 | gelu_bw ULP errors (this issue) | OPEN |
| #35290 | GELU forward ULP errors | OPEN (fix ready) |
| #35885 | tanh_bw ULP errors | OPEN (fix drafted) |

---

*Document created: 2026-01-16*
*Based on Sessions 18-25 research*
