# SFPU Kernel Analysis: `atanh`

## 1. Operation Overview

**Math definition**: `atanh(x) = 0.5 * ln((1+x)/(1-x))`, valid for |x| < 1.

**Equivalent form used in kernel**: `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))`, which avoids an explicit division by computing two independent logarithms and subtracting.

**UnaryOpType enum value**: `ATANH`

**Supported data types**: BFLOAT16, BFLOAT8_B, FLOAT32

**Python API**: `ttnn.atanh(input_tensor)`

**Golden function**: `torch.atanh(input_tensor)`

---

## 2. File Inventory (All Abstraction Layers)

### Layer 1: Python Nanobind
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — `bind_unary_operation<"atanh", &ttnn::atanh>(...)`

### Layer 2: C++ Operation Registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — `REGISTER_UNARY_OPERATION(atanh, ATANH)`

### Layer 3: UnaryOpType Enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — `ATANH` entry in `UnaryOpType` enum

### Layer 4: SFPU Op Dispatch (op_utils)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
  - `get_sfpu_op_name()`: returns `"SFPU_OP_ATANH_INCLUDE"` for `UnaryOpType::ATANH`
  - `get_op_init_and_func_parameterized()`: returns `{"atanh_tile_init();", "atanh_tile(idst);"}`

### Layer 5: Compute API Header
- `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
  - `atanh_tile(uint32_t idst)` — calls `llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst)`
  - `atanh_tile_init()` — calls `llk_math_eltwise_unary_sfpu_atanh_init<APPROX>()`

### Layer 6: Conditional Include Guard
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` — `#if SFPU_OP_ATANH_INCLUDE` guards `#include "api/compute/eltwise_unary/atanh.h"`

### Layer 7: LLK Math Wrappers (per-architecture, identical)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
  - `llk_math_eltwise_unary_sfpu_atanh_init<APPROXIMATE>()` — calls `llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>` with `sfpu::atanh_init<APPROXIMATE>` as the init functor
  - `llk_math_eltwise_unary_sfpu_atanh<APPROXIMATE, ITERATIONS>()` — calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` with `ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>` as the compute functor

### Layer 8: SfpuType Enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` — `atanh` entry in `SfpuType` enum
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` — same

### Layer 9: SFPU Kernel Implementation (per-architecture, identical)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`

### Layer 10: Golden Function
- `ttnn/ttnn/experimental_loader/golden_functions.py` — `_atanh_golden_function` → `torch.atanh(input_tensor)`

### Layer 11: Backward Operation
- `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` — `atanh_bw()` computes `grad / (1 - x^2)` using op chain: square → sub(1) → neg → recip → mul(grad), with NaN/Inf edge handling

---

## 3. SFPU Kernel Deep Analysis

### 3.1 Algorithm

The kernel computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` by evaluating two natural logarithms and subtracting them. Each `ln(y)` is computed via IEEE 754 floating-point decomposition:

1. **Decompose**: `y = 2^e * m` where `m ∈ [1, 2)` (i.e., extract exponent `e` and normalize mantissa `m`)
2. **Polynomial**: `ln(y) = e * ln(2) + P(m)` where `P(m)` is a cubic minimax polynomial approximation for `ln(m)` on `[1, 2)`

The polynomial is evaluated in Horner form:
```
P(m) = c0 + m * (c1 + m * (c2 + m * c3))
```

### 3.2 Polynomial Coefficients

Set during `atanh_init()` using programmable constant registers:

| Register | Hex Literal | Approximate Value | Role |
|----------|-------------|-------------------|------|
| `vConstFloatPrgm0` | `-0x1.952992p+0f` | -1.5828 | c0 (constant term) |
| `vConstFloatPrgm1` | `0x2.4f5388p+0f` | +2.3110 | c1 (linear coeff) |
| `vConstFloatPrgm2` | `-0xd.e712ap-4f` | -0.8691 | c2 (quadratic coeff) |
| (local `constexpr`) | `0x2.44734p-4f` | +0.1416 | c3 (cubic coeff) |

Note: `c3` is a compile-time constant, not stored in a programmable register. This is because all 3 programmable float registers (`vConstFloatPrgm0/1/2`) are consumed by `c0`, `c1`, `c2`. The kernel places `c3` as a `constexpr float` inside `calculate_atanh()` itself, relying on instruction-immediate encoding.

### 3.3 SFPI Intrinsics Used

| Intrinsic | Count per iteration | Purpose |
|-----------|-------------------|---------|
| `dst_reg[0]` (read) | 1 | Load input tile element `x` |
| `vConst1` | 2 | Hardware constant `1.0f` for computing `1+x` and `1-x` |
| `exexp()` | 2 | Extract biased exponent from IEEE 754 float (for `a` and `b`) |
| `setexp(val, 127)` | 2 | Set exponent to 127 (normalize mantissa to [1,2)) |
| `int32_to_float(val, 0)` | 2 | Convert integer exponent to float for `e * ln(2)` computation |
| `vConstFloatPrgm0/1/2` | 6 (3 per ln) | Programmable constant loads for polynomial coefficients |
| `dst_reg[0] =` (write) | 1 | Store result back |
| `dst_reg++` | 1 | Advance to next tile element |

### 3.4 Instruction Count Estimate (per iteration)

Counting SFPI vector operations per loop iteration:

```
x = dst_reg[0]                          // 1 load
a = x + vConst1                         // 1 vFloat add
b = -x + vConst1                        // 1 negate + 1 vFloat add (2)
ea = exexp(a)                           // 1 exexp
ma = setexp(a, 127)                     // 1 setexp
pa = ma * c3 + vConstFloatPrgm2         // 1 mul + 1 add (FMA)
pa = pa * ma + vConstFloatPrgm1         // 1 mul + 1 add (FMA)
pa = pa * ma + vConstFloatPrgm0         // 1 mul + 1 add (FMA)
ln_a = int32_to_float(ea,0) * ln2 + pa // 1 i2f + 1 mul + 1 add
eb = exexp(b)                           // 1 exexp
mb = setexp(b, 127)                     // 1 setexp
pb = mb * c3 + vConstFloatPrgm2         // 1 mul + 1 add (FMA)
pb = pb * mb + vConstFloatPrgm1         // 1 mul + 1 add (FMA)
pb = pb * mb + vConstFloatPrgm0         // 1 mul + 1 add (FMA)
ln_b = int32_to_float(eb,0) * ln2 + pb // 1 i2f + 1 mul + 1 add
result = (ln_a - ln_b) * 0.5f          // 1 sub + 1 mul
dst_reg[0] = result                     // 1 store
dst_reg++                               // 1 advance
```

**Approximate total: ~30 SFPI vector instructions per element** (actual count depends on compiler FMA fusion).

### 3.5 Loop Structure

- **Template parameter `ITERATIONS`**: default 8, meaning 8 elements (tile rows) per kernel invocation
- **Unroll pragma**: `#pragma GCC unroll 8` — full unroll of the 8-iteration loop, eliminating branch overhead at the cost of increased code size
- **No conditional branches**: The kernel contains no `v_if`/`v_endif` blocks, so all SIMD lanes execute identically. This means:
  - No lane divergence penalty
  - Uniform execution time regardless of input values
  - Out-of-domain values (|x| >= 1) produce undefined results rather than being caught

### 3.6 Init Function

`atanh_init<APPROXIMATE>()` loads three polynomial coefficients into programmable constant registers. This is called once before the tile loop begins, so the cost is amortized across all tiles.

- **No conditional behavior**: The `APPROXIMATE` template parameter is present but unused — the same coefficients are loaded regardless of approximation mode
- **No SFPU configuration**: No special SFPU modes or state need to be configured beyond the constants

---

## 4. Numerical Analysis

### 4.1 Precision Characteristics

The kernel uses a **cubic minimax polynomial** to approximate `ln(m)` on [1, 2). A degree-3 polynomial provides roughly **10-12 bits of mantissa accuracy** for ln. Since atanh requires computing *two* such logarithms and subtracting, errors accumulate:

1. **Large |x| regime** (|x| > 0.25): `1+x` and `1-x` are well-separated, so `ln(1+x)` and `ln(1-x)` have different magnitudes. The subtraction is well-conditioned. Accuracy is dominated by the polynomial approximation quality (~10-bit effective precision).

2. **Small |x| regime** (|x| < 0.25): **Catastrophic cancellation** occurs. Both `ln(1+x)` and `ln(1-x)` are close to zero, and their difference loses significant bits. For very small x, `atanh(x) ≈ x`, but the kernel computes a difference of two nearly-equal logarithms, losing precision.

### 4.2 Test Tolerances

From `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py`:

| Check | Regime | Threshold |
|-------|--------|-----------|
| `assert_allclose` (bfloat16) | Full domain | `rtol=1.6e-2, atol=1e-2` |
| `assert_allclose` (float32) | Full domain | `rtol=1.6e-2, atol=2e-3` |
| `assert_with_ulp` (bfloat16 only) | \|expected\| > 0.25 | `ulp_threshold=4` |

The ULP check is restricted to the large-magnitude regime (>0.25) to avoid failures from the catastrophic cancellation region. The 4-ULP threshold for bfloat16 is reasonable for a cubic polynomial approximation.

Float32 ULP checking is explicitly skipped because the cubic polynomial has only ~10-bit effective precision, far coarser than float32's 23-bit mantissa.

### 4.3 Domain Handling

- **Valid domain**: |x| < 1 (strict inequality)
- **No clamping in kernel**: The kernel does not validate inputs. Values at x = ±1 will cause `b = 1-x = 0` or `a = 1+x = 0`, leading to `ln(0)` which produces `-inf`, resulting in `±inf` output. Values |x| > 1 make one of `a` or `b` negative, and `ln(negative)` is undefined — the IEEE decomposition `exexp`/`setexp` will produce garbage.
- **Test-side filtering**: The test replaces out-of-domain values with 0.0 before comparison.

---

## 5. Comparison with Related Operations

### vs. `sinh` (same-repo)
- `sinh` uses the `exp_21f` helper for exponential computation (Moroz et al. 2022 algorithm), which is significantly more complex per invocation (~15 SFPI instructions) but provides good precision
- `sinh` includes a **conditional small-value path** (`v_if(abs_x < 0.5)` with Taylor approximation `x + x³/6`) to avoid catastrophic cancellation — `atanh` does NOT have this mitigation
- `sinh` explicitly clamps inputs to avoid overflow (`z < -127.0f`)
- `sinh` has an explicit bfloat16 rounding step (`float_to_fp16b`) — `atanh` does not

### vs. typical log-based operations
- `atanh` is unique in computing *two* logarithms per element, making it roughly 2x the instruction cost of a single-logarithm operation
- The shared polynomial coefficients (loaded once in init) keep the two ln evaluations efficient — no re-loading needed

---

## 6. Architecture Consistency

The Blackhole and Wormhole B0 implementations are **identical** at all layers:
- `ckernel_sfpu_atanh.h` — same source
- `llk_math_eltwise_unary_sfpu_atanh.h` — same source
- Both use the same polynomial coefficients, same SFPI intrinsics, same loop structure

This is expected since both architectures share the SFPU instruction set for these operations.

---

## 7. Key Observations for Implementation Reference

1. **Programmable constants**: Uses all 3 `vConstFloatPrgm` registers. A new operation that needs to coexist or call atanh must be aware that these are consumed.

2. **No approximation mode differentiation**: The `APPROXIMATION_MODE` template parameter is threaded through but never branched on. The same cubic polynomial is always used.

3. **No edge-case handling**: Unlike `sinh` which has small-value and overflow guards, `atanh` trusts the caller to provide in-domain inputs.

4. **Fully unrolled loop**: `#pragma GCC unroll 8` eliminates loop overhead but increases code size. This is a tradeoff chosen for throughput.

5. **Catastrophic cancellation weakness**: The small-|x| regime loses precision. A potential improvement would be a conditional path using the Taylor series `atanh(x) ≈ x + x³/3 + x⁵/5` for small |x|, similar to how `sinh` handles its small-value case.

6. **Clean structure**: The operation follows the standard unary SFPU pattern perfectly — init loads constants, calculate processes elements, no special CB or memory configuration needed.
