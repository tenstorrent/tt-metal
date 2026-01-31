# ttnn.gelu_bw() severe ULP errors: 10.69% of BF16 values have ULP > 1000, Max ULP = 32,460

### Component / Area

TTNN / Eltwise Operations / Unary Backward / GELU Backward

### Issue Type (optional)

Bad Outputs

### Observed

`ttnn.gelu_bw()` has severe accuracy issues affecting approximately 50% of the BF16 input range:

1. **Wrong sign bug** at x ≈ -3.7: derivative should be negative but returns positive
2. **High ULP errors** in deep negative (x < -5) and large positive (x > 5) regions
3. **10.69% of all BF16 values have ULP > 1000**

**Wrong Sign Bug (most critical):**

| Input x | Expected GELU'(x) | Actual Output | ULP Error |
|---------|-------------------|---------------|-----------|
| -3.700 | **-1.526e-03** | **+4.349e-04** | 29,742 |
| -3.719 | **-1.373e-03** | **+5.112e-04** | 29,756 |

The derivative should be **negative** but the hardware returns **positive**.

**Full ULP Analysis by Region (65,026 BF16 values):**

| Region | Count | Mean ULP | Max ULP | Worst x | Status |
|--------|-------|----------|---------|---------|--------|
| Deep negative (x < -5) | 16,095 | 6,255.61 | 32,460 | -3.376e+38 | **SEVERE** |
| Moderate negative [-5, -2] | 160 | 1,885.86 | 29,756 | -3.719 | **WRONG SIGN** |
| Near negative [-2, -0.5] | 256 | 5.98 | 146 | -0.754 | Needs work |
| Near zero [-0.5, 0.5] | 32,003 | 0.11 | 4 | -0.330 | Good |
| Near positive [0.5, 2] | 256 | 0.55 | 2 | 0.586 | Good |
| Moderate positive [2, 5] | 160 | 0.38 | 1 | 2.0 | Good |
| Large positive (x > 5) | 16,096 | 2,748.22 | 16,203 | 3.376e+38 | **SEVERE** |
| **OVERALL** | **65,026** | **2,233.36** | **32,460** | -3.376e+38 | |

**Cumulative ULP Distribution:**

| ULP Threshold | Count | Percentage |
|---------------|-------|------------|
| ULP = 0 | 53,950 | 82.97% |
| ULP ≤ 1 | 57,435 | 88.33% |
| ULP ≤ 2 | 57,668 | 88.68% |
| ULP ≤ 5 | 57,790 | 88.87% |
| ULP ≤ 10 | 57,893 | 89.03% |
| ULP ≤ 100 | 58,051 | 89.27% |
| ULP ≤ 1000 | 58,072 | 89.31% |
| **ULP > 1000** | **6,954** | **10.69%** |

### Expected

GELU backward should return the derivative of GELU:

```
GELU'(x) = cdf + x * pdf
where:
  cdf = 0.5 * (1 + erf(x / sqrt(2)))  -- CDF of standard normal
  pdf = exp(-x²/2) / sqrt(2π)          -- PDF of standard normal
```

For numerical stability with negative x, use `erfc()`:
```
cdf = 0.5 * erfc(-x / sqrt(2))  when x < 0
```

At x = -3.7:
- cdf ≈ 0.0001076
- pdf ≈ 0.0004405
- x * pdf ≈ -0.001630
- **GELU'(-3.7) = cdf + x*pdf ≈ -0.001522** (negative!)

The reference implementation with DAZ+FTZ modeling gives -1.526e-03, and the hardware should match this within a few ULP.

### 1. Steps (exact commands)

**Prerequisites:** tt-metal must be cloned and built per standard instructions.

```bash
cd tt-metal

# Cherry-pick the reproducer tests (test files + bug report)
git fetch https://github.com/ivoitovych/tt-metal.git ivoitovych/bug-report-gelu-bw-ulp
git cherry-pick FETCH_HEAD  # Cherry-pick commit with test files

# Rebuild only the C++ test target (incremental, ~30 seconds)
cmake --build build_Debug --target unit_tests_ttnn

# Run C++ ULP tests (11 tests)
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluBwUlp*"

# Run specific test showing wrong sign bug
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluBwUlp*ModerateNegativeRegionBugAnalysis"

# Run Python tests (33 tests)
pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp_bug.py -v
```

**Reproducer files:**
- C++ tests: `tests/ttnn/unit_tests/gtests/test_gelu_bw_ulp_bug.cpp`
- Python tests: `tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp_bug.py`
- This report: `tests/ttnn/unit_tests/operations/eltwise/GELU_BW_ULP_BUG_REPORT.md`

**Branch URL:** https://github.com/ivoitovych/tt-metal/tree/ivoitovych/bug-report-gelu-bw-ulp

### 2. Input data / link or description

**Minimal reproducer (Python):**
```python
import ttnn
import torch
import math

device = ttnn.open_device(0)

# Test x = -3.7 where wrong sign occurs
x_val = -3.7
input_tensor = ttnn.from_torch(
    torch.full((1, 1, 32, 32), x_val, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT, device=device
)
grad_tensor = ttnn.from_torch(
    torch.full((1, 1, 32, 32), 1.0, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT, device=device
)

result = ttnn.gelu_bw(grad_tensor, input_tensor, approximate="none")
actual = ttnn.to_torch(result[0])[0, 0, 0, 0].item()

# Reference calculation
SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
cdf = 0.5 * math.erfc(-x_val / SQRT2)
pdf = math.exp(-0.5 * x_val * x_val) * INV_SQRT_2PI
expected = cdf + x_val * pdf

print(f"x = {x_val}")
print(f"Expected: {expected:.6e} (NEGATIVE)")
print(f"Actual:   {actual:.6e} (POSITIVE - WRONG SIGN!)")

ttnn.close_device(device)
```

**Reference Implementation (fp64 with erfc for stability):**
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

### 3. Frequency

100% reproducible. The bug occurs for all BF16 values in the affected regions.

### 1. Software Versions

- **tt-metal version:** main branch (tested on commit 50b633663b)
- **Python:** 3.10
- **OS:** Ubuntu 22.04.5 LTS, Kernel 6.8.0-87-generic

### 2. Hardware Details

- **Device:** Wormhole n150 L (single card)
- **Board ID:** 0100018611902024
- **Driver:** TT-KMD 2.2.0
- **FW Bundle:** 18.5.0

### Is this a regression?

Unknown

### Regression Details

Not tested against previous versions.

### Logs & Diagnostics

**C++ Test Output (ModerateNegativeRegionBugAnalysis):**
```
========================================
MODERATE NEGATIVE REGION BUG ANALYSIS
(Critical region for training: [-5, -2])
========================================
         x       Expected         Actual       ULP     Abs Error
-----------------------------------------------------------------
    -2.000     -8.496e-02     -8.398e-02         2      9.766e-04
    -2.500     -3.760e-02     -3.589e-02         7      1.709e-03
    -3.000     -1.190e-02     -1.135e-02         9      5.493e-04
    -3.500     -2.808e-03     -1.099e-03       168      1.709e-03
    -3.700     -1.526e-03      4.349e-04     29742      1.961e-03
    -3.719     -1.373e-03      5.112e-04     29756      1.884e-03
    -3.750     -1.228e-03     -1.328e-03        13      9.918e-05
    -3.800     -1.045e-03     -1.114e-03         9      6.866e-05
    -4.000     -5.035e-04     -5.341e-04         8      3.052e-05
    -4.500     -6.819e-05     -7.200e-05         8      3.815e-06
    -5.000     -7.123e-06     -7.451e-06        11      3.278e-07
-----------------------------------------------------------------
Worst ULP: 29756 at x = -3.719e+00
```

**C++ Test Output (ComprehensiveULPByRegion):**
```
============================================================
GELU BACKWARD ULP ANALYSIS BY REGION (DAZ+FTZ MODEL)
============================================================
                        Region     Count    Mean ULP     Max ULP       Worst x
-------------------------------------------------------------------------------
        Deep negative (x < -5)     16095     6255.61       32460     -3.376e+38
    Moderate negative [-5, -2]       160     1885.86       29756     -3.719e+00
      Near negative [-2, -0.5]       256        5.98         146     -7.539e-01
         Near zero [-0.5, 0.5]     32003        0.11           4     -3.301e-01
        Near positive [0.5, 2]       256        0.55           2      5.859e-01
      Moderate positive [2, 5]       160        0.38           1      2.000e+00
        Large positive (x > 5]     16096     2748.22       16203      3.376e+38
-------------------------------------------------------------------------------
                       OVERALL     65026     2233.36       32460     -3.376e+38
============================================================
```

#### Root Cause Analysis

The GELU backward kernel is located at:
`ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp`

```cpp
// Step 1: erf(x / sqrt(2))
fill_tile(3, kAlpha);
mul_binary_tile(1, 3, 1);  // tile[1] = x / sqrt(2)
erf_tile(1);               // tile[1] = erf( x / sqrt(2) )

// cdf_term = 0.5 * (1.0 + erf(x / sqrt(2)))
fill_tile(3, 1.0f);
add_binary_tile(1, 3, 1);  // tile[1] += 1.0
fill_tile(3, 0.5f);
mul_binary_tile(1, 3, 1);  // tile[1] *= 0.5

// ... pdf_term calculation ...

// multiply by x
mul_binary_tile(2, 4, 2);  // tile[2] *= x (from tile[4])
```

**Suspected Issue:** The `erf_tile()` implementation may have issues for inputs where `|x/√2| > ~2.6`:

For x = -3.7:
- `erf(-3.7/√2) = erf(-2.616)` should be ≈ -0.99978
- If `erf_tile()` saturates to exactly -1.0, then `1 + erf() = 0` and `cdf_term = 0`
- The pdf_term calculation should still produce a negative result via `x * pdf`
- **But the actual output (+4.349e-04) is suspiciously close to just `pdf` without the `x` multiplication**

This suggests either:
1. The `erf_tile()` function has side effects that corrupt tile[4] (which holds x for later multiplication)
2. The `mul_binary_tile(2, 4, 2)` operation fails silently for certain input ranges
3. There's a precision/overflow issue in the intermediate calculations

**Observation:** The actual output +4.349e-04 ≈ pdf = (1/√2π) × exp(-3.7²/2) ≈ 0.000427

This is exactly what you'd get if the final `x * pdf` multiplication produced `|pdf|` instead of `x * pdf`.

#### Proposed Fixes

**Option 1: Use erfc() for negative inputs (recommended)**
```cpp
// For negative x, use erfc for numerical stability
v_if(x < 0.0f) {
    // cdf = 0.5 * erfc(-x / sqrt(2))
    fill_tile(3, -kAlpha);
    mul_binary_tile(1, 3, 1);  // tile[1] = -x / sqrt(2) = |x| / sqrt(2)
    erfc_tile(1);              // tile[1] = erfc(|x| / sqrt(2))
    fill_tile(3, 0.5f);
    mul_binary_tile(1, 3, 1);  // tile[1] = cdf
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

**Option 2: Investigate and fix the tile corruption issue**
- Add debug prints to verify tile[4] contains x before the final multiplication
- Check if `erf_tile()` modifies registers beyond tile[1]
- Verify the tile register allocation doesn't have conflicts

**Option 3: Use a different computation order**
- Save x to a separate tile that's not adjacent to tiles used by erf_tile()
- Or compute pdf_term first, then cdf_term

### Priority

P1

### Impact

**Training Accuracy:** The wrong sign bug causes gradients to flow in the **wrong direction** during backpropagation. For inputs around x ≈ -3.7:
- The gradient should push the value toward zero (negative derivative)
- Instead, the gradient pushes the value away from zero (positive derivative)

**Affected Use Cases:**
- Any model using GELU activation with backpropagation
- BERT, GPT, and other transformer models
- The moderate negative region [-5, -2] contains commonly encountered activation values during training

**Severity:**
- 10.69% of all BF16 values have ULP > 1000
- Wrong sign is the most severe form of numerical error
- May cause training instability or convergence issues

### Related Issues

- #35290: ttnn.gelu() ULP errors (GELU forward bug, similar erf/erfc numerical stability issue)

### References

- **GELU backward kernel:** `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp`
- **erf/erfc API:** `tt_metal/include/compute_kernel_api/eltwise_unary/erf_erfc.h`
- **Hardware special values:** `tech_reports/Handling_Special_Value/special_values.md`
