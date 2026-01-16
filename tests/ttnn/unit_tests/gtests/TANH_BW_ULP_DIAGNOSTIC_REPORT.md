# Tanh Backward BFloat16 ULP Precision Diagnostic Report

## Executive Summary

This report documents a comprehensive precision analysis of `ttnn::tanh_bw` (tanh backward/derivative) function on Tenstorrent hardware. The analysis tests **all 65,025 normal BFloat16 values** to measure ULP (Units in Last Place) error against IEEE-754 reference.

**Key Finding: `ttnn::tanh_bw` has a severe implementation bug causing catastrophic precision loss in the transition and saturation regions (|x| > 3).**

| Metric | Value |
|--------|-------|
| Total values tested | 65,025 |
| Maximum ULP error | **15,139** |
| Mean ULP error | 155.59 |
| Exact results (ULP = 0) | 93.60% |
| Results within 1 ULP | 97.97% |
| Results within 2 ULP | 98.11% |

**This is NOT a BF16 format limitation.** The forward `ttnn::tanh` achieves **Max ULP = 1** across the entire BF16 range, proving correct implementation is achievable.

## Comparison: tanh vs tanh_bw

| Operation | Max ULP | Mean ULP | % Within 1 ULP | % Within 2 ULP |
|-----------|---------|----------|----------------|----------------|
| **tanh (forward)** | **1** | **0.047** | **100%** | **100%** |
| tanh_bw (backward) | 15,139 | 155.59 | 97.97% | 98.11% |

The forward tanh demonstrates that excellent BF16 precision IS achievable for tanh-family functions.

## Per-Segment Analysis

### TANH (FORWARD) - Reference for Correct Implementation

```
Segment                  Count   Max ULP    Mean ULP     ULP=0    ULP<=1    ULP<=2   Worst Input
------------------------------------------------------------------------------------------------
x < -10                  15967         0        0.00    100.0%    100.0%    100.0%        0.0000
-10 <= x < -5              128         1        0.88     12.5%    100.0%    100.0%       -9.0000
-5 <= x < -4                32         1        1.00      0.0%    100.0%    100.0%       -5.0000
-4 <= x < -3                64         1        0.75     25.0%    100.0%    100.0%       -4.0000
-3 <= x < -2                64         1        0.45     54.7%    100.0%    100.0%       -2.7656
-2 <= x < -1               128         1        0.52     47.7%    100.0%    100.0%       -2.0000
-1 <= x < -0.5             128         1        0.41     58.6%    100.0%    100.0%       -1.0000
-0.5 <= x < 0            16001         1        0.07     92.5%    100.0%    100.0%       -0.4980
x == 0                       1         0        0.00    100.0%    100.0%    100.0%        0.0000
0 < x < 0.5              16000         1        0.07     92.5%    100.0%    100.0%        0.0000
0.5 <= x < 1               128         1        0.41     59.4%    100.0%    100.0%        0.5078
1 <= x < 2                 128         1        0.52     47.7%    100.0%    100.0%        1.0000
2 <= x < 3                  64         1        0.47     53.1%    100.0%    100.0%        2.0000
3 <= x < 4                  64         1        0.73     26.6%    100.0%    100.0%        3.0469
4 <= x < 5                  32         1        1.00      0.0%    100.0%    100.0%        4.0000
5 <= x < 10                128         1        0.88     11.7%    100.0%    100.0%        5.0000
x >= 10                  15968         0        0.00    100.0%    100.0%    100.0%        0.0000
------------------------------------------------------------------------------------------------
```

### TANH_BW (BACKWARD) - Shows Implementation Bug

```
Segment                  Count   Max ULP    Mean ULP     ULP=0    ULP<=1    ULP<=2   Worst Input
------------------------------------------------------------------------------------------------
x < -10                  15967     12666      136.48     98.3%     98.3%     98.3%      -10.0625
-10 <= x < -5              128     14515    13738.42      0.0%      0.0%      0.0%       -5.0312
-5 <= x < -4                32     14886    14709.56      0.0%      0.0%      0.0%       -4.0312
-4 <= x < -3.5              32     15080    14988.47      0.0%      0.0%      0.0%       -3.5156
-3.5 <= x < -3              32     15139     5225.31      0.0%      0.0%      3.1%       -3.3438
-3 <= x < -2                64        95       22.30      4.7%      4.7%      7.8%       -3.0000
-2 <= x < -1               128        17        3.67     12.5%     39.8%     54.7%       -1.7734
-1 <= x < -0.5             128         3        1.04     16.4%     81.2%     98.4%       -0.9336
-0.5 <= x < 0            16001         1        0.08     91.9%    100.0%    100.0%       -0.5000
x == 0                       1         0        0.00    100.0%    100.0%    100.0%        0.0000
0 < x < 0.5              16000         1        0.08     91.9%    100.0%    100.0%        0.0002
0.5 <= x < 1               128         3        1.04     16.4%     81.2%     98.4%        0.8984
1 <= x < 2                 128        17        3.68     11.7%     39.8%     54.7%        1.7734
2 <= x < 3                  64        90       20.81      6.2%      6.2%      9.4%        2.9844
3 <= x < 3.5                32     15139     4756.81      0.0%      0.0%      3.1%        3.3438
3.5 <= x < 4                32     15087    14994.44      0.0%      0.0%      0.0%        3.5000
4 <= x < 5                  32     14896    14721.09      0.0%      0.0%      0.0%        4.0000
5 <= x < 10                128     14527    13752.80      0.0%      0.0%      0.0%        5.0000
x >= 10                  15968     12686      137.27     98.3%     98.3%     98.3%       10.0000
------------------------------------------------------------------------------------------------
```

## Bug Evidence

### Specific Failure Examples

| Input x | Expected Output | Actual Output | ULP Error | Note |
|---------|-----------------|---------------|-----------|------|
| -3.3438 | 0.0049 | 0.0000 | 15,139 | **0.0049 is representable in BF16** |
| -5.0312 | 0.0002 | 0.0000 | 14,515 | **0.0002 is representable in BF16** |
| 3.3438 | 0.0049 | 0.0000 | 15,139 | **0.0049 is representable in BF16** |
| 4.0000 | 0.0013 | 0.0000 | 14,896 | **0.0013 is representable in BF16** |

**The expected values (0.0049, 0.0013, 0.0002) are all perfectly representable in BF16.** The implementation is producing incorrect zeros.

### Why This Is NOT a BF16 Limitation

1. **tanh forward achieves Max ULP = 1** - proving correct BF16 implementation is possible
2. **Expected derivative values are representable** - 0.0049, 0.0013 are valid BF16 values
3. **The implementation produces 0.0000 when it should produce non-zero** - this is a bug

## Mathematical Background

### Tanh Backward (Derivative)

```
d/dx tanh(x) = 1 - tanh(x)^2 = sech^2(x)
```

For the backward pass:
```
tanh_bw(grad_output, input) = grad_output * (1 - tanh(input)^2)
```

### Reference Implementation

The test uses **MPFR 256-bit precision** for authoritative reference values:
```cpp
mpfr_tanh(tanh_result, mpfr_x, MPFR_RNDN);     // tanh(x)
mpfr_mul(tanh_squared, tanh_result, tanh_result, MPFR_RNDN);  // tanh(x)^2
mpfr_sub(result, one, tanh_squared, MPFR_RNDN);  // 1 - tanh(x)^2
```

## Root Cause Hypothesis

The bug likely occurs because the implementation computes `1 - tanh(x)^2` by:
1. Computing tanh(x) in BF16 (saturates to ±1.0 for |x| > ~3.3)
2. Squaring the result: 1.0^2 = 1.0
3. Subtracting: 1 - 1 = 0

**The fix**: Compute sech^2(x) directly using a polynomial approximation, similar to how the forward tanh uses a polynomial. Do not rely on `1 - tanh(x)^2` which amplifies saturation error.

## Denormal Behavior (DAZ Verification)

All 127 positive denormal inputs correctly produce derivative = 1.0:
- Denormals are treated as zero under DAZ
- tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
- **100% compliance with DAZ policy**

## Implementation

### Files

| File | Description |
|------|-------------|
| `tests/ttnn/unit_tests/gtests/test_tanh_bw_ulp_diagnostic.cpp` | C++ tests |
| `tests/ttnn/unit_tests/operations/eltwise/backward/test_tanh_bw_ulp_diagnostic.py` | Python tests |
| `tests/ttnn/unit_tests/gtests/TANH_BW_BUG_REPORT.md` | Bug report |

### Running the Tests

```bash
# Build
~/tt/rebuild_ttnn_tests.sh

# Run all tanh ULP tests
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhUlp*:*TanhBwUlp*"
```

## Conclusions

1. **tanh_bw has a severe implementation bug** - Max ULP = 15,139 is unacceptable
2. **This is NOT a BF16 limitation** - tanh forward proves Max ULP = 1 is achievable
3. **Near-zero precision is good** - Max ULP = 1 for |x| < 0.5
4. **Transition/saturation regions are broken** - 0% within 2 ULP for |x| > 3.5
5. **Impact**: Incorrect gradients affect training accuracy for tanh layers

## Priority

**P1** - Training accuracy is affected by incorrect gradients.

---

*Report generated: January 2026*
*Hardware: Blackhole P150a*
*Software: TT-Metal (Debug build)*
