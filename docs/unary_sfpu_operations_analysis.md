# SFPU Operations Analysis and Classification

## What is SFPU?

**SFPU (Special Function Processing Unit)** is a dedicated hardware unit in Tenstorrent's AI accelerators (Blackhole, Wormhole, Quasar) that executes element-wise mathematical operations. These operations are fundamental to neural network computations, including activations, mathematical transformations, and data manipulation.

---

## Complete Operation List (112 Operations)

The operations are defined in `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`:

```cpp
enum class UnaryOpType {
    EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, LOG1P, TANH, LOG2, LOG10,
    SIN, COS, COSH, SINH, ABS, ABS_INT32, SIGN, SQUARE, EQZ, NEZ, GTZ, LTZ,
    GEZ, LEZ, RELU_MAX, RELU_MIN, POWER, LEAKY_RELU, ELU, EXP2, HEAVISIDE,
    EXPM1, SIGNBIT, ASIN, ACOS, ACOSH, RSQRT, RELU6, ATAN, ASINH, ATANH,
    ERF, ERFC, ISINF, ISPOSINF, ISNEGINF, ISNAN, LOGICAL_NOT_UNARY, ISFINITE,
    ERFINV, I0, I1, TAN, RSUB, RDIV, SILU, SOFTPLUS, IDENTITY, NEG,
    ADD_UNARY_SFPU, SUB_UNARY_SFPU, MUL_UNARY_SFPU, DIV_UNARY_SFPU,
    UNARY_NE, UNARY_EQ, UNARY_GT, UNARY_LT, UNARY_GE, UNARY_LE, TILED_PROD,
    TYPECAST, BITCAST, BITWISE_XOR, BITWISE_NOT, BITWISE_AND, BITWISE_OR,
    RIGHT_SHIFT, FLOOR, CEIL, TRUNC, FRAC, ROUND, LEFT_SHIFT, REMAINDER, FMOD,
    DROPOUT, FILL, PRELU_SFPU, ZERO_POINT, ALT_COMPLEX_ROTATE90, MISH, HARDMISH,
    MAXIMUM, MINIMUM, TANHSHRINK, THRESHOLD, SOFTSHRINK, HARDSHRINK, HARDTANH,
    HARDSIGMOID, HARDSWISH, WHERE_TSS, SOFTSIGN, CELU, CLAMP_TSS, SELU, RPOW,
    CBRT, LOGSIGMOID, LOGIT
};
```

---

## Classification by Complexity

### EASY (~25 Operations) - Single-Step, Direct Value Manipulation

These operations perform simple transformations using built-in SFPU instructions or trivial logic.

#### Basic Value Operations (4)
Direct manipulation of value representation.

| Operation | Description | Implementation Pattern |
|-----------|-------------|------------------------|
| `abs` | Absolute value | `sfpi::abs(v)` - single instruction |
| `neg` | Negation | Sign bit flip |
| `sign` | Sign of value (-1, 0, 1) | Conditional assignment |
| `identity` | Pass-through (no change) | Direct copy |

#### Simple Activations (3)
Basic threshold-based activations with minimal logic.

| Operation | Description | Implementation Pattern |
|-----------|-------------|------------------------|
| `relu` | max(0, x) | Single conditional zero |
| `relu6` | min(max(0, x), 6) | Two conditionals |
| `heaviside` | Step function (0 or 1) | Single comparison |

#### Rounding Operations (5)
Integer/fractional value manipulation.

| Operation | Description | Implementation Pattern |
|-----------|-------------|------------------------|
| `floor` | Round down | Truncation instruction |
| `ceil` | Round up | Truncation + adjustment |
| `trunc` | Truncate to integer | Direct truncation |
| `frac` | Fractional part | `x - trunc(x)` |
| `round` | Round to nearest | Standard rounding |

#### Bitwise Operations (6)
Bit-level manipulation using hardware instructions.

| Operation | Description | Implementation Pattern |
|-----------|-------------|------------------------|
| `bitwise_and` | AND operation | Single instruction |
| `bitwise_or` | OR operation | Single instruction |
| `bitwise_xor` | XOR operation | Single instruction |
| `bitwise_not` | NOT operation | Single instruction |
| `left_shift` | Shift bits left | Shift instruction |
| `right_shift` | Shift bits right | Shift instruction |

#### Value Classification (6)
Checking floating-point special values and properties.

| Operation | Description | Implementation Pattern |
|-----------|-------------|------------------------|
| `isnan` | Check for NaN | Bit pattern check |
| `isinf` | Check for infinity | Bit pattern check |
| `isposinf` | Check for +infinity | Bit pattern + sign check |
| `isneginf` | Check for -infinity | Bit pattern + sign check |
| `isfinite` | Check if finite | Bit pattern check |
| `logical_not` | Boolean NOT | Bit manipulation |
| `signbit` | Sign bit extraction | Bit extraction |

#### Simple Arithmetic (1)
Trivial arithmetic operations.

| Operation | Description | Implementation Pattern |
|-----------|-------------|------------------------|
| `square` | x² | Single multiply |


### MEDIUM (~40 Operations) - Standard Mathematical Functions

These require polynomial approximations, lookup tables, or multi-step computation.

#### Exponential Functions (3)
Functions based on e^x computation.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `exp` | e^x | Polynomial approximation (range reduction) |
| `exp2` | 2^x | Similar to exp with base conversion |
| `expm1` | e^x - 1 | Specialized for numerical stability near 0 |

#### Logarithmic Functions (4)
Inverse of exponential functions.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `log` | Natural logarithm | Minimax approximation + normalization |
| `log1p` | log(1 + x) | Specialized for small x values |
| `log2` | Base-2 logarithm | log with base conversion |
| `log10` | Base-10 logarithm | log with base conversion |

#### Roots and Reciprocals (3)
Iterative convergence algorithms.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `sqrt` | Square root | Newton-Raphson iteration |
| `rsqrt` | 1/sqrt(x) | Fast inverse square root algorithm |
| `recip` | 1/x | Newton-Raphson iteration |

#### Forward Trigonometric (5)
Standard trigonometric functions.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `sin` | Sine | Range reduction + polynomial |
| `cos` | Cosine | Range reduction + polynomial |
| `tan` | Tangent | sin/cos or direct polynomial |
| `sinh` | Hyperbolic sine | (e^x - e^-x) / 2 |
| `cosh` | Hyperbolic cosine | (e^x + e^-x) / 2 |

#### Standard Activations (6)
Common neural network activation functions.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `sigmoid` | 1/(1+exp(-x)) | exp + reciprocal composition |
| `tanh` | Hyperbolic tangent | Polynomial approximation |
| `tanh_derivative` | Derivative of tanh | 1 - tanh²(x) |
| `gelu` | Gaussian Error Linear Unit | 15th-degree Chebyshev polynomial |
| `silu` | x * sigmoid(x) | sigmoid composition |
| `mish` | x * tanh(softplus(x)) | tanh + softplus composition |

#### ELU Family Activations (3)
Exponential Linear Unit variants.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `elu` | alpha * (e^x - 1) for x < 0 | exp + conditional |
| `celu` | Continuously differentiable ELU | exp + conditional with alpha |
| `selu` | Scaled ELU | elu with fixed scale parameters |

#### Soft Activations (5)
Smooth approximations of hard functions.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `softplus` | log(1 + e^x) | log + exp composition |
| `softsign` | x / (1 + abs(x)) | abs + division |
| `softshrink` | Soft shrinkage | Conditional subtraction |
| `tanhshrink` | x - tanh(x) | tanh composition |
| `logsigmoid` | log(sigmoid(x)) | Numerically stable computation |

#### Hard Activations (5)
Piecewise linear approximations (faster, less accurate).

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `hardtanh` | Clamped linear | min(max(x, -1), 1) |
| `hardsigmoid` | Linear approximation of sigmoid | Piecewise linear |
| `hardswish` | x * hardsigmoid(x) | Piecewise polynomial |
| `hardmish` | Hard approximation of mish | Piecewise linear |
| `hardshrink` | Hard shrinkage | Conditional zeroing |

#### Leaky/Parametric Activations (2)
ReLU variants with non-zero negative slope.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `leaky_relu` | max(x, alpha*x) | Conditional with slope |
| `prelu` | Parametric ReLU | Learnable slope parameter |

#### Error Functions (2)
Statistical/probability functions.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `erf` | Error function | Polynomial approximation |
| `erfc` | Complementary error function (1 - erf) | Polynomial approximation |

#### Data/Type Operations (4)
Value transformation and manipulation.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `typecast` | Type conversion | Format conversion logic |
| `bitcast` | Reinterpret bits as different type | No computation, just reinterpret |
| `clamp` | Clamp to [min, max] | Two conditionals |
| `threshold` | Replace below threshold | Conditional replacement |

#### Stochastic Operations (1)
Operations involving randomness.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `dropout` | Random zeroing | PRNG + conditional |

#### Scalar Arithmetic (5)
Operations with scalar parameters.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `add_unary_sfpu` | x + scalar | Direct addition |
| `sub_unary_sfpu` | x - scalar | Direct subtraction |
| `mul_unary_sfpu` | x * scalar | Direct multiplication |
| `div_unary_sfpu` | x / scalar | Multiply by reciprocal |
| `pow` | x^n (integer n) | Repeated multiplication or exp(n*log(x)) |

#### Modulo Operations (2)
Remainder calculations.

| Operation | Description | Implementation Technique |
|-----------|-------------|-------------------------|
| `remainder` | IEEE remainder | Division + rounding |
| `fmod` | Floating-point modulo | Truncated division remainder |

**Example Implementation - SIGMOID** (`ckernel_sfpu_sigmoid.h`):
```cpp
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    sfpi::vFloat exp_neg_x;

    // Choose accuracy level based on destination format
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x);  // High precision exp
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);       // Lower precision, faster
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x;

    // Compute reciprocal with appropriate iterations
    if constexpr (is_fp32_acc_to_dest_mode) {
        return _sfpu_reciprocal_<2>(denominator);   // 2 Newton iterations
    } else {
        return _sfpu_reciprocal_<1>(denominator);   // 1 Newton iteration
    }
}
```

### HARD (~25 Operations) - Complex Multi-Step Algorithms

These involve iterative algorithms, series expansions, multiple dependent operations, or complex special cases.

#### Inverse Trigonometric (3)
Require careful domain handling and polynomial approximations.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `asin` | Inverse sine | Domain [-1,1], polynomial with sqrt composition |
| `acos` | Inverse cosine | Domain [-1,1], derived from asin |
| `atan` | Inverse tangent | Wide domain, careful range reduction |

#### Inverse Hyperbolic (3)
Compositions of log and sqrt with edge cases.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `asinh` | Inverse hyperbolic sine | log(x + sqrt(x² + 1)) |
| `acosh` | Inverse hyperbolic cosine | log(x + sqrt(x² - 1)), domain x >= 1 |
| `atanh` | Inverse hyperbolic tangent | 0.5 * log((1+x)/(1-x)), domain (-1,1) |

#### Advanced Root Operations (1)
Non-square roots requiring different convergence.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `cbrt` | Cube root | Newton-Raphson with cubic convergence, handles negative inputs |

#### Special Mathematical Functions (3)
Functions requiring series expansions or complex approximations.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `erfinv` | Inverse error function | Winitzki approximation: nested sqrt + log |
| `i0` | Modified Bessel function (1st kind, order 0) | 10th-degree polynomial series |
| `i1` | Modified Bessel function (1st kind, order 1) | 10th-degree polynomial series |

#### Reduction Operations (4)
Operations with dependencies across multiple elements.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `cumsum` | Cumulative sum | Sequential dependency between elements |
| `reduce` | General reduction | Multi-element aggregation |
| `tiled_prod` | Tiled product | Multi-tile coordination |
| `topk` | Top-K selection | Sorting/comparison across elements |

#### Quantization Operations (3)
Integer/fixed-point conversion with scaling.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `quant` | Float to quantized | Scale + round + clamp |
| `requant` | Requantize with new scale | Multiple scale operations |
| `dequant` | Quantized to float | Scale + offset restoration |

#### Numerical Stability Operations (2)
Algorithms designed to avoid overflow/underflow.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `logaddexp` | log(exp(a) + exp(b)) | Max trick for stability |
| `hypot` | sqrt(x² + y²) | Overflow-safe scaling algorithm |

#### Number Theory Operations (2)
Integer algorithms adapted for SFPU.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `gcd` | Greatest common divisor | Euclidean algorithm iterations |
| `lcm` | Least common multiple | gcd-based computation |

#### Complex Number Operations (1)
Operations on complex representations.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `alt_complex_rotate90` | 90° complex rotation | Real/imaginary swap and sign |

#### Power Operations (2)
General exponentiation with various bases/exponents.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `rpow` | scalar^x (reverse power) | exp(x * log(scalar)) |
| `power_iterative` | General x^y | Multiple iterations for convergence |

#### Stochastic Operations (1)
Random number generation on SFPU.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `rand` | Random number generation | PRNG state management |

#### Pooling Operations (1)
Operations tracking indices alongside values.

| Operation | Description | Why Hard |
|-----------|-------------|----------|
| `max_pool_indices` | Max pooling with index tracking | Comparison + index bookkeeping |

**Example Implementation - ERFINV** (`ckernel_sfpu_erfinv.h`):
```cpp
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) {
    // Winitzki approximation (2008):
    // erfinv(x) = sqrt(-2/(pi*a) - log(1-x²)/2 + sqrt((2/(pi*a) + log(1-x²))² - (1/a)*log(1-x²)))

    // Step 1: Compute log(1 - x²)
    sfpi::vFloat log_value = in * in;
    log_value = 1 - log_value;
    log_value = calculate_log_body<false, false, true>(log_value, 0);

    sfpi::vFloat temp = log_value * 0.5;

    // Constants from paper (a = 0.147)
    constexpr float TwoPiA = 4.330746750799873f;   // 2 / (pi * a)
    constexpr float OneDivA = 6.802721088435375f;  // 1/a

    // Step 2: Compute intermediate values
    temp = TwoPiA + temp;
    temp = -temp;

    // Step 3: Inner sqrt
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * OneDivA);
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    calculated_value = temp + intermediate_result;

    // Step 4: Outer sqrt
    return calculate_sqrt_custom<false>(calculated_value);
}
```

**Example Implementation - I0 (Bessel Function)** (`ckernel_sfpu_i0.h`):
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input = dst_reg[0];
        vFloat x = input * input;

        // 10th-degree polynomial approximation (Taylor series truncation)
        vFloat result = 1.0f + POLYVAL10(
            1.50E-22f, 7.24E-20f, 2.90E-17f, 9.39E-15f, 2.40E-12f,
            4.71E-10f, 6.78E-08f, 0.000006781684028f, 0.0004340277778f,
            0.015625f, 0.25f, x);

        dst_reg[0] = result;
        dst_reg++;
    }
}
```

---

## Classification by Functional Category

### Activation Functions (22)
```
relu, relu6, relu_max, relu_min, leaky_relu, prelu, elu, celu, selu,
gelu, silu, mish, hardmish, sigmoid, hardsigmoid, hardswish, hardtanh,
tanh, softplus, softsign, softshrink, hardshrink
```

### Trigonometric (12)
```
sin, cos, tan, sinh, cosh, asin, acos, atan, asinh, acosh, atanh
```

### Exponential/Logarithmic (11)
```
exp, exp2, expm1, log, log1p, log2, log10, logsigmoid, logit, logaddexp
```

### Arithmetic (14)
```
add, sub, mul, div, neg, recip, sqrt, rsqrt, cbrt, square, pow, rpow, remainder, fmod
```

### Comparison/Classification (17)
```
eqz, nez, gtz, ltz, gez, lez, isnan, isinf, isposinf, isneginf, isfinite,
unary_eq, unary_ne, unary_gt, unary_lt, unary_ge, unary_le
```

### Bitwise (8)
```
bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift, signbit, bitcast
```

### Rounding (5)
```
floor, ceil, trunc, frac, round
```

### Special Functions (7)
```
erf, erfc, erfinv, i0, i1, heaviside, sign
```

### Data Manipulation (8)
```
identity, typecast, fill, dropout, clamp, threshold, where, abs
```

### Reduction/Aggregation (5)
```
cumsum, reduce, tiled_prod, topk, max_pool_indices
```

---

## Parametrized vs Non-Parametrized Operations

### Parametrized (45) - Take additional scalar arguments
```
relu_max(limit), relu_min(limit), power(exponent), leaky_relu(slope),
elu(alpha), softplus(beta, threshold), rsqrt(epsilon), heaviside(value),
typecast(dtype), clamp(min, max), threshold(value, replacement), etc.
```

### Non-Parametrized (67) - No additional arguments
```
abs, sign, neg, relu, relu6, sigmoid, tanh, sin, cos, exp, log, sqrt, etc.
```

---

## Architecture Support

| Architecture | # Operations | Notes |
|--------------|-------------|-------|
| **Blackhole** | 86 | Full feature set |
| **Wormhole B0** | ~80 | Slightly fewer than Blackhole |
| **Quasar** | 9 | Legacy subset: `add, exp, lrelu, recip, relu, sqrt, tanh, typecast_*` |

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **Total SFPU Operations** | 112 |
| **Easy Complexity** | ~25 (22%) |
| **Medium Complexity** | ~45 (40%) |
| **Hard Complexity** | ~26 (23%) |
| **Parametrized** | 45 (40%) |
| **Non-Parametrized** | 67 (60%) |

### Subcategory Breakdown

| Complexity | Subcategory | Count |
|------------|-------------|-------|
| **Easy** | Basic Value Operations | 4 |
| | Simple Activations | 3 |
| | Rounding Operations | 5 |
| | Bitwise Operations | 6 |
| | Value Classification | 7 |
| | Simple Arithmetic | 1 |
| **Medium** | Exponential Functions | 3 |
| | Logarithmic Functions | 4 |
| | Roots and Reciprocals | 3 |
| | Forward Trigonometric | 5 |
| | Standard Activations | 6 |
| | ELU Family Activations | 3 |
| | Soft Activations | 5 |
| | Hard Activations | 5 |
| | Leaky/Parametric Activations | 2 |
| | Error Functions | 2 |
| | Data/Type Operations | 4 |
| | Stochastic Operations | 1 |
| | Scalar Arithmetic | 5 |
| | Modulo Operations | 2 |
| **Hard** | Inverse Trigonometric | 3 |
| | Inverse Hyperbolic | 3 |
| | Advanced Root Operations | 1 |
| | Special Mathematical Functions | 3 |
| | Reduction Operations | 4 |
| | Quantization Operations | 3 |
| | Numerical Stability Operations | 2 |
| | Number Theory Operations | 2 |
| | Complex Number Operations | 1 |
| | Power Operations | 2 |
| | Stochastic Operations | 1 |
| | Pooling Operations | 1 |

---

## Key Implementation Patterns

1. **Iteration Loop**: All operations iterate over `ITERATIONS` (typically 8) elements
2. **Approximation Mode**: Many ops support `APPROXIMATION_MODE` template parameter for speed vs accuracy tradeoff
3. **FP32/BF16 Variants**: Some ops have different code paths for different precision levels
4. **Polynomial Approximation**: Complex functions use Chebyshev/minimax polynomial fitting
5. **Newton-Raphson**: Used for `sqrt`, `rsqrt`, `recip`, and derived operations
6. **Composability**: Many complex ops compose simpler ones (e.g., sigmoid = exp + recip)

---

## Implementation Locations

- **Low-level kernels (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/`
- **Low-level kernels (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/`
- **Compute Kernel API**: `tt_metal/include/compute_kernel_api/eltwise_unary/`
- **TTNN Operations**: `ttnn/cpp/ttnn/operations/eltwise/unary/`
- **Type definitions**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
