# SFPU Kernel Analysis: atanh

## Overview

The `atanh` operation computes the inverse hyperbolic tangent function: `atanh(x) = 0.5 * ln((1+x)/(1-x))`. This analysis focuses on the SFPU (Special Function Processing Unit) kernel implementation within the TT-NN unary operations framework using the UnaryProgramFactory.

## Mathematical Definition

```
atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
```

**Domain**: (-1, 1) - strictly less than 1 in absolute value
**Range**: (-∞, ∞)
**Key Properties**:
- Odd function: atanh(-x) = -atanh(x)
- Undefined at x = ±1 (asymptotic behavior)
- Well-behaved for small |x| near zero

## SFPU Kernel Architecture

### Kernel Location and Structure

**Primary SFPU API Header**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
- Provides `atanh_tile(uint32_t idst)` and `atanh_tile_init()` functions
- Uses template parameter `APPROX` for precision control

**Hardware-Specific Implementations**:
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- **Note**: Both implementations are identical, indicating stable algorithm across generations

### Core Algorithm Implementation

The SFPU kernel decomposes atanh into logarithmic computations using IEEE 754 floating-point decomposition:

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    // For each tile element:
    // 1. Compute a = 1 + x, b = 1 - x
    sfpi::vFloat a = x + sfpi::vConst1;
    sfpi::vFloat b = -x + sfpi::vConst1;

    // 2. IEEE decomposition: y = 2^e * m, where m ∈ [1, 2)
    sfpi::vInt ea = sfpi::exexp(a);
    sfpi::vFloat ma = sfpi::setexp(a, 127);

    // 3. Cubic polynomial approximation for ln(m)
    // P(m) = c0 + m*(c1 + m*(c2 + m*c3))
    sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;
    pa = pa * ma + sfpi::vConstFloatPrgm1;
    pa = pa * ma + sfpi::vConstFloatPrgm0;

    // 4. Complete logarithm: ln(y) = e*ln(2) + P(m)
    sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa;

    // 5. Repeat for b = 1 - x
    // [similar decomposition for ln(b)]

    // 6. Final result: atanh(x) = 0.5 * (ln(a) - ln(b))
    sfpi::vFloat result = (ln_a - ln_b) * 0.5f;
}
```

### Polynomial Coefficients

The kernel uses a **cubic minimax polynomial approximation** for ln(m) on the interval [1, 2):

```cpp
// Initialized in atanh_init()
constexpr float c3 = 0x2.44734p-4f;         // ~0.1416 (hardcoded)
sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;   // c0 ~ -1.5828
sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;    // c1 ~  2.3110
sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;    // c2 ~ -0.8691
constexpr float ln2 = 0.6931471805599453f;  // ln(2) constant
```

**Polynomial Form**: `P(m) = c0 + m*(c1 + m*(c2 + m*c3))` (Horner's method)

## SFPU Instruction Analysis

### Core SFPU Operations Used

1. **`sfpi::exexp(a)`**: Extracts IEEE 754 exponent from floating-point value
2. **`sfpi::setexp(a, 127)`**: Sets exponent to 127 (bias for normalized mantissa)
3. **`sfpi::int32_to_float(ea, 0)`**: Converts exponent integer to float for scaling
4. **Vector arithmetic**: `+`, `-`, `*` operations on `sfpi::vFloat` SIMD vectors
5. **Constant loading**: `sfpi::vConst1`, `sfpi::vConstFloatPrgm[0-2]` for coefficient access

### Computational Complexity

- **SFPU instructions per element**: ~20-25 operations
- **Critical path**: 2× IEEE decomposition + 2× polynomial evaluation + arithmetic
- **Parallelism**: 32×32 tile (1024 elements) processed simultaneously
- **Memory access**: Minimal (coefficient constants pre-loaded)

## Integration with Unary Operations Framework

### Enum Registration

```cpp
// ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:59
enum class UnaryOpType {
    // ... other operations
    ATANH,  // Line 59
    // ... more operations
};
```

### Python API Registration

```cpp
// ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:136
REGISTER_UNARY_OPERATION(atanh, ATANH)
```

This generates the Python-accessible `ttnn.atanh()` function with standard unary operation signature:
- Input tensor (tile layout required)
- Optional memory config
- Optional output tensor
- Optional core grid specification

### LLK Integration

**Low-Level Kernel (LLK) Wrapper**: `llk_math_eltwise_unary_sfpu_atanh.h`

```cpp
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```

**Key Parameters**:
- `APPROXIMATE`: Controls precision vs. performance (template parameter)
- `ITERATIONS`: Default 8 (processes 8 destination registers per call)
- `vector_mode`: VectorMode::RC (Row-Column processing)

## Accuracy and Precision Analysis

### Numerical Challenges

1. **Catastrophic Cancellation**: For small |x|, computing ln(1+x) - ln(1-x) subtracts nearly equal values
2. **Domain Boundaries**: Values near x = ±1 require careful handling
3. **Polynomial Accuracy**: Cubic approximation provides ~10-bit effective precision

### Test Tolerance Requirements

From `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py`:

**bfloat16**:
- `rtol=1.6e-2, atol=1e-2` (general tolerance)
- ULP threshold: 4 for |expected| > 0.25

**fp32**:
- `rtol=1.6e-2, atol=2e-3` (wider atol due to polynomial limitations)
- No ULP testing (cubic polynomial too coarse for fine-grained ULP analysis)

### Domain Filtering

The test implementation filters invalid domain values:
```cpp
// Filter to valid domain: |x| < 1 (strict inequality)
mask = torch_input.float().abs() < 1.0
torch_input = torch.where(mask, torch_input, torch.zeros_like(torch_input))
```

## Performance Characteristics

### Computational Efficiency

- **Hardware optimized**: Direct SFPU execution without CPU overhead
- **Vectorized**: 32×32 tile processing in parallel
- **Memory efficient**: Minimal data movement (coefficients pre-loaded)
- **Iteration control**: Configurable ITERATIONS parameter for throughput tuning

### Accuracy Trade-offs

- **High precision**: Uses IEEE 754 decomposition for robust logarithm computation
- **Controlled approximation**: Cubic polynomial balances speed vs. accuracy
- **Template parameterization**: `APPROXIMATE` flag allows precision/performance tuning

## Operation Dependencies

### Required SFPU Instructions

The atanh kernel relies on hardware SFPU support for:
- IEEE 754 floating-point decomposition (`exexp`, `setexp`)
- Integer-to-float conversion
- Vector arithmetic operations
- Constant register access

### Hardware Compatibility

- **Wormhole B0**: Full support
- **Blackhole**: Full support (identical implementation)
- **Earlier generations**: May require different coefficient tuning or fallback methods

## Summary

The atanh SFPU kernel demonstrates sophisticated numerical computing within TT hardware constraints:

1. **Mathematical Robustness**: Uses IEEE 754 decomposition to avoid naive logarithm computation
2. **Hardware Optimization**: Leverages SFPU vector operations for maximum parallelism
3. **Precision Control**: Provides configurable approximation modes via template parameters
4. **Integration**: Seamlessly integrates with UnaryProgramFactory and Python API
5. **Testing**: Comprehensive validation with appropriate tolerance thresholds for polynomial-based computation

The implementation successfully balances computational efficiency with numerical accuracy, making it suitable for ML workloads requiring inverse hyperbolic tangent operations.
