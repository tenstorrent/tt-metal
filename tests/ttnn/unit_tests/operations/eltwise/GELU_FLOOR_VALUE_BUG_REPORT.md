# ttnn.gelu() has catastrophic ULP errors in multiple regions (up to 32,767 ULP)

### Component / Area

TTNN / Eltwise Operations / GELU (Accurate/Chebyshev mode)

### Issue Type (optional)

Bad Outputs

### Observed

`ttnn.gelu()` in accurate mode (default, `fast_and_approximate_mode=False`) has **THREE** problematic regions with catastrophic ULP errors:

#### Region 1: Deep Negative Tail (x < -5.5) — WORST

| Input | Expected | Actual | ULP Error |
|-------|----------|--------|-----------|
| -13.5 | -1.06e-40 | 0.0 | **32,767** |
| -12.0 | -2.13e-32 | 0.0 | 29,987 |
| -10.0 | -7.62e-24 | 0.0 | 24,481 |
| -7.0 | -8.96e-12 | 0.0 | 21,219 |
| -5.5625 | -7.40e-08 | 0.0 | 19,554 |

**Cause**: Hardware returns exactly 0.0 for x < -5.5, but exact GELU has tiny negative values.

#### Region 2: Near-Zero (|x| < ~1e-4)

| Input | Expected (0.5×x) | Actual | ULP Error |
|-------|------------------|--------|-----------|
| 1e-38 | 5e-39 | 2.98e-05 | **14,276** |
| 1e-30 | 5e-31 | 2.98e-05 | 10,968 |
| 1e-20 | 5e-21 | 2.98e-05 | 6,718 |
| 1e-10 | 5e-11 | 2.98e-05 | 2,463 |

**Cause**: Chebyshev polynomial c0 coefficient (2.98e-05) dominates for tiny inputs.

#### Region 3: Transition Region (-5.5 to ~-4.0)

| Input | Expected | Actual | ULP Error |
|-------|----------|--------|-----------|
| -5.5 | -1.04e-07 | -3.11e-04 | 1,475 |
| -5.375 | -2.06e-07 | -1.06e-04 | 1,155 |
| -5.0 | -1.43e-06 | -1.26e-04 | 836 |

**Cause**: Polynomial is poorly fitted near the -5.5 threshold boundary.

#### Overall Statistics (Full BFloat16 Sweep)

```
Total values tested: 65,278
Max ULP error: 32,767 (maximum possible for BF16!)
Mean ULP error: 3,266
Values with ULP > 1000: 27,089 (41.5%)
Values with ULP > 100: 29,243 (44.8%)
Values with ULP <= 1: 34,254 (52.5%)
```

### Expected

For all inputs, GELU should have ULP error ≤ 10. Research shows Max ULP ≤ 1 is achievable with proper implementation (see https://github.com/ivoitovych/bf16_gelu_research).

### 1. Steps (exact commands)

**Prerequisites:** tt-metal must be cloned and built per standard instructions.

```bash
cd tt-metal

# Cherry-pick the reproducer tests (single commit with all files)
git fetch https://github.com/ivoitovych/tt-metal.git ivoitovych/bug-report-gelu-floor-value-ulp-03
git cherry-pick FETCH_HEAD

# Rebuild only the C++ test target (incremental, ~30 seconds)
cmake --build build_Debug --target unit_tests_ttnn

# Run Python reproducer (24 tests)
pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py -v -s

# Run C++ reproducer (14 tests: 10 ULP verification + 4 GELU bug reproduction)
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluUlp*:*BFloat16Ulp*"
```

**Reproducer files (all in single commit):**
- Python test: `tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py`
- C++ test: `tests/ttnn/unit_tests/gtests/test_gelu_ulp_bug.cpp`
- This report: `tests/ttnn/unit_tests/operations/eltwise/GELU_FLOOR_VALUE_BUG_REPORT.md`

**Branch URL:** https://github.com/ivoitovych/tt-metal/tree/ivoitovych/bug-report-gelu-floor-value-ulp-03

### 2. Input data / link or description

**Minimal reproducer for Region 1 (Deep Negative Tail):**
```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Deep negative value — hardware returns 0.0, should return tiny negative
input_val = -13.5
torch_input = torch.tensor([[input_val]], dtype=torch.bfloat16)
tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
actual = ttnn.to_torch(result).item()

print(f"Input:    {input_val}")
print(f"Expected: ~-1e-40 (tiny negative)")
print(f"Actual:   {actual}")  # Will print 0.0

ttnn.close_device(device)
```

**Minimal reproducer for Region 2 (Near-Zero):**
```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Tiny positive value — hardware returns 2.98e-05, should return 0.5*x
input_val = 1e-20
torch_input = torch.tensor([[input_val]], dtype=torch.bfloat16)
tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
actual = ttnn.to_torch(result).item()

print(f"Input:    {input_val:.2e}")
print(f"Expected: {0.5 * input_val:.2e}")
print(f"Actual:   {actual:.6e}")  # Will print ~2.98e-05

ttnn.close_device(device)
```

### 3. Frequency

100% reproducible. The full BFloat16 sweep shows:
- Region 1 (x < -5.5): 172 values affected with ULP > 1000
- Region 2 (near-zero): 26,917 values affected with ULP > 1000
- Region 3 (transition): ~200 values with ULP 100-1500

### 1. Software Versions

- tt-metal: main branch (commit 38355f582c or later)
- Python: 3.10
- OS: Ubuntu 22.04

### 2. Hardware Details

- Device: Wormhole n150 L
- Board ID: 0100018611902024
- FW Bundle: 18.5.0

**Additional Verification — Blackhole P150 (FW 19.1.0):**

| Input | Wormhole ULP | Blackhole ULP | Notes |
|-------|--------------|---------------|-------|
| -13.5 | 32,767 | 32,767 | ✓ Match |
| -12.0 | 29,987 | 29,987 | ✓ Match |
| -10.0 | 24,481 | 25,928 | ~6% variance |
| -5.5625 | 19,554 | 19,554 | ✓ Match |
| Near-zero | exact | exact | ✓ Match |
| Transition | exact | exact | ✓ Match |

Bug pattern and severity consistent across both architectures.

### Is this a regression?

Unknown - likely present since GELU Chebyshev implementation was added.

### Regression Details

N/A

### Logs & Diagnostics

**Full test output with all regions:**
```
====================================================================================================
TOP 50 WORST ULP ERRORS
====================================================================================================
         Value |       Expected |         Actual |  ULP Error |   BF16 Hex
----------------------------------------------------------------------------------------------------
 -1.350000e+01 |  -1.055539e-40 |   0.000000e+00 |     32,767 | 0xC158
 -1.343750e+01 |  -2.449291e-40 |   0.000000e+00 |     32,766 | 0xC157
 -1.337500e+01 |  -5.661216e-40 |   0.000000e+00 |     32,762 | 0xC156
 ...
 -5.531250e+00 |  -8.793492e-08 |   0.000000e+00 |     19,524 | (threshold)
 -5.500000e+00 |  -1.044426e-07 |  -3.108978e-04 |      1,475 | (transition)
 ...
  1.000000e-38 |   5.000000e-39 |   2.980232e-05 |     14,276 | (near-zero)

====================================================================================================
REGIONS WITH ULP > 1000
====================================================================================================

Found 2 distinct region(s) with ULP > 1000:

Region 1: DEEP NEGATIVE TAIL
  Range: [-13.5, -5.375]
  Max ULP: 32,767
  Count: 172 values

Region 2: NEAR-ZERO
  Range: [-6e-05, 2.7e-07]
  Max ULP: 14,330
  Count: 26,917 values
```

#### Root Cause Analysis

The GELU accurate mode uses `calculate_gelu_chebyshev()` defined in:
`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    // ...
    v_if(in == 0.0f) { result = 0.0f; }
    v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); }
    v_endif;
    // ...
}

inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {
        result = POLYVAL15(
            // ... coefficients ...
            2.98325768482e-05,    // c0  <-- FLOOR VALUE BUG
            val);
        result = setsgn(result, val);
    }
    v_endif;  // <-- Returns 0.0 for val < -5.5  (DEEP TAIL BUG)
    return result;
}
```

**Bug 1 (Deep Negative Tail):** The `v_if(val >= -5.5f)` returns 0.0 for x < -5.5, but exact GELU has tiny negative values that are representable in BF16.

**Bug 2 (Near-Zero):** The constant term c0 = 2.98e-05 dominates for tiny inputs since all higher-order polynomial terms become negligible.

**Bug 3 (Transition):** The polynomial is poorly fitted near the -5.5 boundary.

#### Proposed Fixes

**For Region 1 (Deep Negative Tail):**
The threshold of -5.5 is too aggressive. For BFloat16, GELU approaches 0 only at x ≈ -13.5625 (saturation threshold). Options:
1. Lower the threshold to -13.5
2. Use asymptotic expansion: `GELU(x) ≈ -φ(x)·(1 - 1/x² + 3/x⁴)` for deep negatives
3. Accept returning 0.0 for x < -13.5 only

**For Region 2 (Near-Zero):**
Add special handling for tiny inputs:
```cpp
v_if(abs_val < 1e-6f) {
    result = val * 0.5f;  // Taylor: GELU(x) ≈ 0.5*x for small x
}
```

**For Region 3 (Transition):**
Refit the Chebyshev polynomial with better boundary handling, or use a piecewise approach with separate polynomial for the transition region.

### Priority

P2

### Impact

- **44.8% of all BFloat16 values** have ULP error > 100
- **41.5% of all BFloat16 values** have ULP error > 1000
- Max ULP of 32,767 is catastrophic (maximum possible error)
- Affects any model using GELU with:
  - Deep negative activations (e.g., after failed training, bad initialization)
  - Near-zero activations (residual connections, normalized values)
  - Training stability in edge cases

#### References

- BFloat16 GELU approximation research: https://github.com/ivoitovych/bf16_gelu_research
- Research shows Max ULP ≤ 1 is achievable with adaptive polynomial approach
- Source file: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`
