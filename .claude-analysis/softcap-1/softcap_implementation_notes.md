# Softcap Implementation Notes

## Operation Definition
softcap(x, cap) = cap * tanh(x / cap)

Where cap is a positive float scalar (default 50.0).

## Algorithm

Two-regime tanh approximation for computing tanh(u) where u = x/cap:

### Regime 1: Small |u| (|u| < 1.0) - Degree-9 Taylor Polynomial
```
tanh(u) = u * (1 + u^2*(-1/3 + u^2*(2/15 + u^2*(-17/315 + u^2*(62/2835)))))
```
Evaluated in Horner form on u^2. Converges within the Taylor series radius of convergence (pi/2 ~ 1.57). At the boundary |u|=1.0, the error is ~0.8% relative, which is within ~2 ULP for bfloat16.

### Regime 2: Large |u| (|u| >= 1.0) - Exp-based Formula
Uses the identity tanh(|u|) = (1-f)/(1+f) where f = exp(-2|u|).

The reciprocal 1/(1+f) is approximated via 5-term geometric series:
```
tanh(|u|) = 1 - 2f + 2f^2 - 2f^3 + 2f^4
```
Evaluated in Horner form: `1 + 2*f*(-1 + f*(1 + f*(-1 + f)))`

exp(-2|u|) is computed via the Moroz et al. 2022 `exp_21f` algorithm (2^z computation). This is a self-contained helper copied from the sinh kernel.

### Parameter Passing
- `cap` is passed as a compile-time constant embedded in the init/func strings
- `softcap_init(cap)` stores `1/cap` in `vConstFloatPrgm0` and `cap` in `vConstFloatPrgm1`
- The compute loop reads these programmable constant registers

## Reference Operations Used
1. **sinh** (most useful): Provided the `exp_21f` helper function and the two-regime pattern (Taylor for small inputs, exp-based for large inputs). The overall kernel structure closely follows sinh.
2. **swish**: Provided patterns for `v_if`/`v_endif` conditional execution and the piecewise SFPU computation approach.
3. **atanh**: Provided patterns for programmable constant register usage in init functions.
4. **hardshrink**: Provided patterns for how parameterized ops pass scalar values through the dispatch chain.

## Deviations from Standard Patterns
1. **Parameterized init via compile-time embedding**: The cap value is formatted into the init string (e.g., `softcap_tile_init(50.0f)`) rather than passed as a runtime arg. This is simpler than the runtime arg approach used by hardshrink (which uses a custom compute kernel), and is enabled by the standard `eltwise_sfpu.cpp` dispatch path.
2. **Self-contained exp_21f helper**: Copied from sinh rather than imported, to avoid cross-kernel header dependencies. Named `exp_21f_softcap` to avoid ODR conflicts.

## Known Limitations
- The Taylor polynomial at the boundary (|u|=1.0) has ~0.8% relative error, translating to ~2 ULP in bfloat16. For fp32, the relative error is small but ULP count is higher due to fp32's finer granularity.
- The geometric series approximation of 1/(1+f) is accurate to O(f^5). At |u|=1.0 (f~0.135), the error from this term is ~5e-5, negligible.
- For very large |x/cap| (> ~64), the exp computation clamps to -127 in log2 space, producing tanh(u) ≈ 1.0 exactly.
- `#pragma GCC unroll 0` is used (no unrolling) to manage instruction cache pressure, following the sinh pattern.

## Implementation Details

### Kernel Entry Points

**Public API: `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`**

```cpp
// Performs element-wise softcap operation: softcap(x, cap) = cap * tanh(x / cap).
ALWI void softcap_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst))); }
ALWI void softcap_tile_init(float cap) { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>(cap))); }
```

### SFPU Kernel Implementation

**Core SFPU kernel: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`**

Two-regime dispatch with flat control flow:

```cpp
// exp_21f_softcap: Compute 2^z using Moroz et al. 2022 algorithm
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f_softcap(sfpi::vFloat z) {
    z = sfpi::addexp(z, 23);
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = float_to_int32_pos_simple_(z + bias);
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);
    d2 = d1 * d2;
    sfpi::vInt frac_int = float_to_int32_pos_simple_(d2 * d3);
    sfpi::vInt result_int = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));
    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// Flat control flow: compute both regimes unconditionally, select via v_if
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap() {
    constexpr float tc1 = -0.33333333f;  // -1/3
    constexpr float tc2 = 0.13333333f;   //  2/15
    constexpr float tc3 = -0.05396825f;  // -17/315
    constexpr float tc4 = 0.02186949f;   //  62/2835
    constexpr float neg2_log2e = -2.8853900817779268f;  // -2 * log2(e)

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // u = x / cap = x * (1/cap)
        sfpi::vFloat u = x * sfpi::vConstFloatPrgm0;
        sfpi::vFloat abs_u = sfpi::setsgn(u, 0);

        // Regime 1: degree-9 Taylor polynomial (computed unconditionally)
        sfpi::vFloat u_sq = u * u;
        sfpi::vFloat tanh_u = u * (1.0f + u_sq * (tc1 + u_sq * (tc2 + u_sq * (tc3 + u_sq * tc4))));

        // Regime 2: exp-based (also computed unconditionally, with clamping for safety)
        sfpi::vFloat z_neg = abs_u * neg2_log2e;
        v_if(z_neg < -127.0f) { z_neg = -127.0f; }
        v_endif;

        sfpi::vFloat f = exp_21f_softcap<APPROXIMATION_MODE>(z_neg);

        // tanh(|u|) = 1 + 2f*(-1 + f*(1 + f*(-1 + f)))
        sfpi::vFloat h = f - 1.0f;
        h = f * h + 1.0f;
        h = f * h - 1.0f;
        sfpi::vFloat tanh_exp = 1.0f + 2.0f * f * h;

        // Apply original sign: tanh is odd
        v_if(u < 0.0f) { tanh_exp = -tanh_exp; }
        v_endif;

        // Select regime: use Taylor for |u| < 1.0, exp-based for |u| >= 1.0
        v_if(abs_u >= 1.0f) { tanh_u = tanh_exp; }
        v_endif;

        // result = cap * tanh(u)
        sfpi::dst_reg[0] = sfpi::vConstFloatPrgm1 * tanh_u;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void softcap_init(float cap) {
    sfpi::vConstFloatPrgm0 = 1.0f / cap;
    sfpi::vConstFloatPrgm1 = cap;
}
```

### LLK Math Wrapper

**Wrapper: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`**

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softcap_init(float cap) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softcap, APPROXIMATE>(
        ckernel::sfpu::softcap_init<APPROXIMATE>, cap);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softcap(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_softcap<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```

### Type System Registration

**Added to enum: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`**

```cpp
enum class UnaryOpType {
    // ... other operations ...
    SOFTCAP,  // softcap(x, cap) = cap * tanh(x / cap)
};
```

**Registered in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`:**

```cpp
enum class SfpuType {
    // ... other types ...
    softcap,
};
```

### Operation Dispatch

**Init and function strings in `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`:**

```cpp
case UnaryOpType::SOFTCAP:
    return {fmt::format("softcap_tile_init({});", param0), fmt::format("softcap_tile({});", idst)};
```

Parametrized type check in `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`:

```cpp
template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::SOFTCAP: return true;
        // ... other parametrized types ...
        default: return false;
    }
}
```

### Python Bindings

**Golden function in `ttnn/ttnn/operations/unary.py`:**

```python
def _golden_function_softcap(input_tensor_a, *args, **kwargs):
    import torch
    cap = kwargs.get("cap", 50.0)
    return cap * torch.tanh(input_tensor_a / cap)

ttnn.attach_golden_function(ttnn.softcap, golden_function=_golden_function_softcap)
```

### Test Coverage

**Exhaustive test in `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py`:**

```python
@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0], ids=["cap1", "cap10", "cap50"])
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_softcap(device, cap, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference: softcap(x, cap) = cap * tanh(x / cap)
    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    torch_output = cap * torch.tanh(golden_input / cap)
    expected = flush_subnormal_values_to_zero(torch_output)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softcap(tt_input, cap=cap)
    actual = ttnn.to_torch(tt_output)

    # Assertions: ULP threshold of 2 for bfloat16, allclose for fp32
    assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
```

Tests all 3 cap values with both bfloat16 and fp32 inputs, covering regime transitions and sign handling.

## New Files

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` — SFPU kernel implementation
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` — Blackhole variant
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` — LLK math wrapper
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` — Blackhole variant
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h` — Public API
- `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py` — Test suite

## Modified Files

- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` — Added `softcap` enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` — Added `softcap` enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` — Added `#if SFPU_OP_SOFTCAP_INCLUDE` guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — Added `SOFTCAP` enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` — Added `SOFTCAP` to `is_parametrized_type()`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — Added dispatch for `SOFTCAP`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — Added `softcap` via macro
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — Added nanobind for `softcap`
- `ttnn/ttnn/operations/unary.py` — Added golden function and Python API
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` — Dispatch definitions
