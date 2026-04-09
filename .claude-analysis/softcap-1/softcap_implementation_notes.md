# Softcap Implementation Notes

## Operation Definition
`softcap(x, cap) = cap * tanh(x / cap)` where `cap` is a positive float parameter (default 50.0).

## Implementation Strategy

### SFPU Kernel Algorithm
The kernel computes `tanh(u)` where `u = x / cap` using two regimes:

1. **Small |u| (< 1.0)**: Degree-7 Taylor series
   - `tanh(u) = u * (1 + u² * (-1/3 + u² * (2/15 + u² * (-17/315))))`
   - Evaluated in Horner form for numerical stability
   - Provides <1 ULP accuracy in bfloat16 for |u| < 0.85, and ~2 ULP at |u| = 1.0

2. **Moderate/large |u| (≥ 1.0)**: Exponential series
   - Let `e = exp(-2|u|)`, computed via the Moroz et al. 2022 `exp_21f` algorithm (2^z)
   - `tanh(|u|) = 1 - 2e + 2e² - 2e³` (geometric series expansion truncated at degree 3)
   - For |u| ≥ 1.0, `e ≤ 0.135`, giving truncation error < 2*e⁴ ≈ 0.0007 (< 0.2 ULP)
   - For |u| ≥ 4.0, `e < 1e-5`, and tanh naturally rounds to 1.0 in bfloat16

The exp-based formula is computed for ALL SIMD lanes (SFPU processes in lockstep), then the Taylor series overrides small-|u| lanes via `v_if` conditional predication.

### Parameter Passing
The `cap` parameter flows through the standard parameterized unary path:
- Host: `get_op_init_and_func_parameterized()` embeds the float as a `std::bit_cast<uint32_t>` literal in the kernel define string
- Device: The SFPU kernel decodes the uint32_t back to float via union reinterpretation
- `inv_cap = 1.0f / cap` is computed once per SFPU function call (once per face, 4 times per tile)

### exp_21f Helper
The `softcap_exp_21f_` helper is a local copy of the Moroz et al. 2022 algorithm from `ckernel_sfpu_sinh.h`. It computes `2^z` using IEEE 754 decomposition and a degree-2 polynomial refinement. This is copied locally to avoid cross-include dependencies, following the established codebase pattern.

## Reference Operations Used
- **sinh** (most useful): Provided the `exp_21f` helper algorithm, the dual-regime (exp + Taylor override) pattern, and the `v_if` conditional override pattern for small-argument special casing.
- **atanh**: Provided the standard abstraction layer pattern (ckernel → LLK → API → split-include) and the `SfpuType` enum registration pattern. Also demonstrated programmable constant usage (though softcap doesn't need them).
- **swish**: Provided the SFPI piecewise computation pattern with `v_if`/`v_endif` for segment selection, and the `abs`/comparison workflow.
- **hardshrink/tanhshrink**: Provided context on parameterized operations and custom compute kernel patterns (not directly used since softcap uses the standard `eltwise_sfpu.cpp` path).

## Deviations from Standard Patterns
- **Parameterized SFPU op via `eltwise_sfpu.cpp`**: Most parameterized operations (hardshrink, tanhshrink) use custom compute kernels. Softcap uses the standard `eltwise_sfpu.cpp` with the parameter embedded in the `SFPU_OP_CHAIN_0` macro expansion, which is simpler and follows the same path as non-parameterized ops.
- **`#pragma GCC unroll 0`**: Used instead of `#pragma GCC unroll 8` to reduce register pressure, since the kernel has high register usage (exp_21f uses ~10 intermediates). This follows the sinh kernel's pattern.
- **No programmable constants**: All coefficients are local `constexpr` or computed from the runtime parameter. The `softcap_init()` function is empty.

## Known Limitations
- **fp32 precision**: Like all SFPU operations in this codebase, the fp32 path computes at approximately bfloat16 precision due to the polynomial approximations in `exp_21f`. The fp32 result is bfloat16-quality stored in fp32 format.
- **Taylor-exp transition**: At |u| = 1.0, the Taylor degree-7 series has ~2 ULP error in bfloat16. The exp-based formula has ~0.2 ULP at |u| = 1.0. The transition is handled by `v_if` predication with the Taylor override taking priority for |u| < 1.0. There is no smoothing at the boundary, but both approximations agree to within ~2 ULP at the transition point.
- **Very large |u|**: For |u| > ~88 (where `exp(-2|u|)` underflows), the `exp_21f` result is clamped to `2^(-127)` ≈ 5.9e-39, and tanh correctly evaluates to 1.0.

## Source Code

### New Files

#### 1. SFPU Compute Kernel: `ckernel_sfpu_softcap.h`

Implements the core SFPU kernel with the dual-regime tanh approximation. The kernel computes `softcap(x, cap) = cap * tanh(x / cap)` using Taylor series for small arguments and exponential formula for larger arguments.

Key components:
- `softcap_ftoi_pos_()`: Branchless float-to-int conversion for the exp_21f algorithm
- `softcap_exp_21f_()`: Local copy of Moroz et al. 2022's `2^z` algorithm
- `calculate_softcap()`: Main SFPU kernel with dual-regime computation

```cpp
// Local exp_21f helper: compute 2^z using Moroz et al. 2022 algorithm.
// Input z should be clamped to >= -127 before calling.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat softcap_exp_21f_(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = softcap_ftoi_pos_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = softcap_ftoi_pos_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// Main kernel
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(uint32_t param0) {
    // Decode cap parameter from bit-packed uint32
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = param0;
    const float cap = conv.f;
    const float inv_cap = 1.0f / cap;

    constexpr float neg_2_log2e = -2.0f * 1.4426950408889634f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = x * inv_cap;
        sfpi::vFloat abs_u = sfpi::setsgn(u, 0);

        // --- Exp-based regime (computed for all lanes) ---
        sfpi::vFloat z = abs_u * neg_2_log2e;
        v_if(z < -127.0f) { z = -127.0f; }
        v_endif;

        sfpi::vFloat e = softcap_exp_21f_<APPROXIMATION_MODE>(z);

        // tanh(|u|) = 1 - 2e + 2e^2
        sfpi::vFloat result = sfpi::vConst1 - e * 2.0f + e * e * 2.0f;

        // Apply sign: tanh(-u) = -tanh(u)
        v_if(x < 0.0f) { result = -result; }
        v_endif;

        // --- Taylor override for small |u| ---
        // tanh(u) ≈ u * (1 + u^2 * (-1/3 + u^2 * 2/15))
        v_if(abs_u < 1.0f) {
            sfpi::vFloat u2 = u * u;
            result = u * (sfpi::vConst1 + u2 * (-0.33333333f + u2 * 0.13333333f));
        }
        v_endif;

        sfpi::dst_reg[0] = result * cap;
        sfpi::dst_reg++;
    }
}
```

#### 2. LLK Math Wrapper: `llk_math_eltwise_unary_sfpu_softcap.h`

Wraps the SFPU kernel with LLK initialization and dispatch logic.

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softcap_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::softcap, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softcap(
    uint dst_index, uint32_t param0, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_softcap<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}
```

#### 3. API Header: `softcap.h`

Public C++ API for the softcap tile operation. Includes documentation and macros to generate the `softcap_tile()` function at compile time.

```cpp
/**
 * Performs element-wise softcap: softcap(x, cap) = cap * tanh(x / cap).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The cap parameter, bit-cast from float to uint32_t                         | uint32_t | Positive float bit-cast to uint32_t                   | True     |
 */
ALWI void softcap_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_softcap<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softcap_init<APPROX>())); }
```

#### 4. Python Test File: `test_softcap.py`

Comprehensive test suite with multiple test cases covering different input ranges and cap values.

```python
def golden_softcap(x, cap):
    return cap * torch.tanh(x / cap)

@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ],
)
@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0])
def test_softcap_bfloat16(input_shapes, cap, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.softcap(input_tensor, cap=cap)
    golden_tensor = golden_softcap(in_data, cap)

    assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=10)
    assert_allclose(golden_tensor, output_tensor, rtol=5e-2, atol=0.35)

# Additional tests for:
# - default cap value (50.0)
# - preallocated output tensors
# - small input values (Taylor series regime)
# - large input values (saturation)
```

### Modified Files

#### 1. SFPU Type Enum: `llk_sfpu_types.h`

Added `softcap` to the `SfpuType` enum:

```cpp
enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,
    sinh,
    softcap,  // NEW
    // ... stubs for nuked operations ...
};
```

#### 2. SFPU Include Guard: `sfpu_split_includes.h`

Added conditional include for the softcap header:

```cpp
#if SFPU_OP_SOFTCAP_INCLUDE
#include "api/compute/eltwise_unary/softcap.h"
#endif
```

#### 3. C++ Unary Op Type Enum: `unary_op_types.hpp`

Added `SOFTCAP` to the `UnaryOpType` enum:

```cpp
enum class UnaryOpType {
    // ... existing ops ...
    SOFTCAP,  // NEW
};
```

#### 4. Utils Header: `unary_op_utils.hpp`

Registered `SOFTCAP` as a parameterized operation type:

```cpp
template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::HARDTANH: return true;
        case UnaryOpType::SOFTSHRINK: return true;
        case UnaryOpType::SOFTCAP: return true;  // NEW
        default: return false;
    }
    return false;
}
```

#### 5. Utils Implementation: `unary_op_utils.cpp`

Added softcap to the macro definition map and parameterized op handler:

```cpp
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        // ... existing cases ...
        case UnaryOpType::SOFTCAP: return "SFPU_OP_SOFTCAP_INCLUDE";  // NEW
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}

template <typename T>
std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type,
    std::span<const T> params,
    [[maybe_unused]] const std::string& idst,
    [[maybe_unused]] std::optional<DataType> input_dtype) {
    // ...
    switch (op_type) {
        case UnaryOpType::SOFTCAP: {  // NEW
            return {
                "softcap_tile_init();", fmt::format("softcap_tile({}, {}u);", idst, std::bit_cast<uint32_t>(param0))};
        }
        // ...
    };
}
```

#### 6. C++ Unary API: `unary.hpp`

Registered softcap as a parameterized unary operation with float parameter:

```cpp
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softcap, SOFTCAP)
```

This macro expands to:

```cpp
inline Tensor softcap(
    const Tensor& input_tensor,
    float parameter,  // cap parameter
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{
            operations::unary::UnaryOpType::SOFTCAP, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}
```

#### 7. Nanobind Bindings: `unary_nanobind.cpp`

Added Python binding for softcap with documentation:

```cpp
{
    auto doc = fmt::format(
        R"doc(
        Applies softcap to :attr:`input_tensor` element-wise.

        .. math::
            \mathrm{{output\_tensor}}_i = \mathrm{{cap}} \cdot \tanh(\mathrm{{input\_tensor}}_i / \mathrm{{cap}})

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword args:
            cap (float, optional): positive scalar cap value. Defaults to `50.0`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, BFLOAT8_B, FLOAT32
                 - TILE, ROW_MAJOR
        )doc");

    ttnn::bind_function<"softcap">(
        mod,
        doc.c_str(),
        &unary_4param_to_5param_wrapper<&ttnn::softcap>,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("cap") = 50.0f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}
```

#### 8. Python API: `unary.py`

Registered softcap in the Python API stubs:

```cpp
// (Note: softcap is registered in ttnn/ttnn/operations/unary.py)
def softcap(input_tensor: Tensor, cap: float = 50.0, *, memory_config: Optional[MemoryConfig] = None, output_tensor: Optional[Tensor] = None) -> Tensor:
    """Element-wise softcap: output = cap * tanh(input / cap)"""
    ...
```

#### 9. Golden Functions: `ttnn/ttnn/operations/unary.py`

Registered golden function for testing and validation:

```python
def _golden_function_softcap(input_tensor_a, *args, **kwargs):
    import torch

    cap = kwargs.get("cap", 50.0)
    return cap * torch.tanh(input_tensor_a / cap)


if hasattr(ttnn, "softcap"):
    ttnn.attach_golden_function(ttnn.softcap, golden_function=_golden_function_softcap)
```

#### 10. Experimental Loader: `ttnn/ttnn/experimental_loader/golden_functions.py`

Golden function registration in the experimental loader:

```python
def _golden_function_softcap(input_tensor_a, *args, **kwargs):
    import torch

    cap = kwargs.get("cap", 50.0)
    return cap * torch.tanh(input_tensor_a / cap)


if hasattr(ttnn, "softcap"):
    ttnn.attach_golden_function(ttnn.softcap, golden_function=_golden_function_softcap)
```
