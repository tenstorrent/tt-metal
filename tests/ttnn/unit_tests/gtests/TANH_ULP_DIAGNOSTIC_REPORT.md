# Tanh BFloat16 ULP Precision Diagnostic Report

## Executive Summary

This report documents a comprehensive precision analysis of `ttnn::tanh` activation function on Tenstorrent hardware. The analysis tests **all 65,025 normal BFloat16 values** plus **254 denormal values** to measure ULP (Units in Last Place) error against IEEE-754 reference.

**Key Finding: `ttnn::tanh` achieves excellent precision with Max ULP = 1 across the entire BFloat16 range.**

| Metric | Value |
|--------|-------|
| Total normal values tested | 65,025 |
| Total denormal values tested | 254 |
| Maximum ULP error | 1 |
| Mean ULP error | 0.0474 |
| Exact results (ULP = 0) | 95.26% |
| Results within 1 ULP | 100% |

## Background

### BFloat16 Format

BFloat16 is a 16-bit floating-point format with:
- 1 sign bit
- 8 exponent bits
- 7 mantissa bits

This provides the same dynamic range as IEEE float32 but with reduced precision.

### ULP (Units in Last Place)

ULP measures the distance between two floating-point values in terms of representable numbers. For example:
- ULP = 0: Exact match
- ULP = 1: Adjacent representable values (best possible approximation)
- ULP = 2: One representable value between actual and expected

### DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)

Tenstorrent SFPU hardware uses DAZ+FTZ mode:
- **DAZ**: Denormal inputs are treated as zero
- **FTZ**: Denormal outputs are flushed to zero

This is documented in `tech_reports/Handling_Special_Value/special_values.md`: "denormals | all | 0x0"

## Methodology

### BFloat16 Value Space

The complete BFloat16 value space under DAZ+FTZ:

| Category | Bit Range | Count | Notes |
|----------|-----------|-------|-------|
| Negative normals | 0x8080 - 0xFF7F | 32,512 | Excludes -inf (0xFF80) |
| Zero | 0x0000 | 1 | All denormals map here |
| Positive normals | 0x0080 - 0x7F7F | 32,512 | Excludes +inf (0x7F80) |
| **Total finite** | - | **65,025** | |
| Positive denormals | 0x0001 - 0x007F | 127 | Treated as zero |
| Negative denormals | 0x8001 - 0x807F | 127 | Treated as zero |
| **Total denormals** | - | **254** | |

### ULP Calculation

Two independent ULP calculation methods were implemented for cross-verification:

#### Method 1: Bitwise Formula

Maps BF16 bits to a linear index representing value order:

```
Index Layout:
  0xFF7F (-max)      -> index 0
  0x8080 (-min_norm) -> index 32511
  0x0000 (zero)      -> index 32512
  0x0080 (+min_norm) -> index 32513
  0x7F7F (+max)      -> index 65024
```

Formula:
```cpp
int32_t bf16_index_bitwise(uint16_t bits) {
    bits = bf16_daz_normalize(bits);  // Apply DAZ
    if (bits == 0x0000) return 32512;  // Zero
    if (bits & 0x8000) {
        // Negative: magnitude 0x0080-0x7F7F maps to 32511-0
        return 0x7F7F - (bits & 0x7FFF);
    } else {
        // Positive: bits 0x0080-0x7F7F maps to 32513-65024
        return 32513 + bits - 0x0080;
    }
}
```

#### Method 2: Sorted Index Lookup

Builds an explicit sorted list of all 65,025 normal BF16 values, then uses binary search to find indices. This serves as an independent verification of the bitwise method.

### Reference Calculation

For each BF16 input value:
1. Apply DAZ normalization (denormals -> 0)
2. Convert to double precision
3. Calculate `tanh()` using MPFR with 256-bit precision
4. Convert result to float, then truncate to BF16
5. Apply FTZ normalization (flush denormal outputs to zero)
6. Compare with device output using ULP distance

The C++ tests use MPFR-256 for authoritative reference values. A verification test confirms fp64 and mpfr-256 produce identical BF16 results for tanh.

## Implementation

### Files Created

| File | Description |
|------|-------------|
| `tests/ttnn/unit_tests/gtests/test_tanh_ulp_diagnostic.cpp` | C++ tests with dual ULP calculators |
| `tests/ttnn/unit_tests/operations/eltwise/test_tanh_ulp_diagnostic.py` | Python tests |
| `tests/ttnn/unit_tests/gtests/CMakeLists.txt` | Updated to include C++ test |

### C++ Test Suite

#### ULP Calculator Verification Tests

| Test | Purpose |
|------|---------|
| `BitwiseAndSortedMethodsAgree` | Verify both ULP methods produce identical results |
| `AdjacentValuesHaveUlpOne` | Verify adjacent values have ULP distance = 1 |
| `SpecificValuesVerification` | Verify known index values at boundaries |
| `DenormalsMapToZero` | Verify all 254 denormals normalize to zero |
| `SortedIndexSize` | Verify sorted index contains exactly 65,025 values |

#### Device Tests

| Test | Purpose |
|------|---------|
| `ExhaustiveBf16Sweep` | Test all 65,025 normal BF16 values |
| `AllPositiveDenormalsProduceZero` | Verify 127 positive denormals -> 0 |
| `AllNegativeDenormalsProduceZero` | Verify 127 negative denormals -> 0 |
| `CrossVerifyUlpMethods` | Verify both ULP methods agree on device results |

### Python Test Suite

| Test | Purpose |
|------|---------|
| `test_exhaustive_bf16_sweep` | Test all 65,025 normal BF16 values |
| `test_positive_denormals_produce_zero` | Verify positive denormals -> 0 |
| `test_negative_denormals_produce_zero` | Verify negative denormals -> 0 |
| `test_ulp_calculator_self_consistency` | Verify ULP calculator correctness |

## Results

### Exhaustive BF16 Sweep

```
========================================
EXHAUSTIVE BF16 SWEEP RESULTS - tanh()
========================================
Total values tested: 65025
Max ULP: 1
Mean ULP: 0.0474
ULP = 0: 61943 (95.2603%)
ULP <= 1: 65025 (100.0000%)
ULP <= 2: 65025 (100.0000%)
```

### ULP Distribution

| ULP | Count | Percentage |
|-----|-------|------------|
| 0 | 61,943 | 95.26% |
| 1 | 3,082 | 4.74% |
| 2+ | 0 | 0% |

### Sample Values with ULP = 1

| Input | Device Output | Expected | ULP |
|-------|---------------|----------|-----|
| -5.0 | -1.0000 | -0.9999 | 1 |
| -2.0 | -0.9648 | -0.9640 | 1 |
| -1.0 | -0.7617 | -0.7616 | 1 |
| 1.0 | 0.7617 | 0.7616 | 1 |
| 2.0 | 0.9648 | 0.9640 | 1 |
| 5.0 | 1.0000 | 0.9999 | 1 |

### Denormal Behavior (DAZ Verification)

```
Positive denormals (0x0001 - 0x007F): 127 tested
  Non-zero outputs: 0/127
  Result: All produce zero (DAZ verified)

Negative denormals (0x8001 - 0x807F): 127 tested
  Non-zero outputs: 0/127
  Result: All produce zero (DAZ verified)
```

### ULP Calculator Cross-Verification

Both calculation methods (bitwise formula and sorted index lookup) produce identical results for all value pairs tested, confirming the correctness of the ULP measurement methodology.

## Running the Tests

### C++ Tests

```bash
# Build
cmake --build build_Debug --target unit_tests_ttnn

# Run all tanh ULP tests
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhUlp*"

# Run only ULP calculator verification (no device needed)
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhUlpCalculator*"

# Run only device tests
./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*TanhUlpDevice*"
```

### Python Tests

```bash
pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_ulp_diagnostic.py -v
```

## Conclusions

1. **Excellent Precision**: `ttnn::tanh` achieves Max ULP = 1, meaning every result is either exact or the best possible BF16 approximation.

2. **Consistent Quality**: 95.26% of results are exact (ULP = 0), with the remaining 4.74% being off by exactly 1 ULP.

3. **DAZ/FTZ Compliance**: All 254 denormal inputs correctly produce zero output, confirming proper DAZ behavior.

4. **Robust Verification**: Dual ULP calculation methods provide independent verification of measurement correctness.

5. **Complete Coverage**: All 65,025 normal BF16 values tested - no sampling or statistical estimation.

## Comparison with tanh_bw

For context, the same methodology was applied to `ttnn::tanh_bw` (backward/derivative). The forward tanh demonstrates superior precision compared to the backward pass.

| Operation | Max ULP | % Exact (ULP=0) | % Within 1 ULP |
|-----------|---------|-----------------|----------------|
| tanh (forward) | 1 | 95.26% | 100% |
| tanh_bw (backward) | 15,139 | 93.60% | 97.97% |

The high Max ULP in tanh_bw occurs in the saturation region where the derivative approaches zero.

## Files Reference

- C++ Tests: `tests/ttnn/unit_tests/gtests/test_tanh_ulp_diagnostic.cpp`
- Python Tests: `tests/ttnn/unit_tests/operations/eltwise/test_tanh_ulp_diagnostic.py`
- This Report: `tests/ttnn/unit_tests/gtests/TANH_ULP_DIAGNOSTIC_REPORT.md`

---

*Report generated: January 2026*
*Hardware: Blackhole P150a*
*Software: TT-Metal (Debug build)*
