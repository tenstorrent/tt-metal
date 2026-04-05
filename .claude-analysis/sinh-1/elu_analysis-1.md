# SFPU Analysis: elu (Comprehensive)

## 1. Mathematical Definition

```
ELU(x) = x                      if x >= 0
          alpha * (exp(x) - 1)   if x < 0
```

- **Parameters**: `alpha` (float, default=1.0, common range [0.1, 3.0])
- **PyTorch reference**: `torch.nn.functional.elu(x, alpha=alpha)`
- **Key property**: User-configurable `alpha` (unlike SELU which uses fixed constants)

## 2. Current Implementation Status

**ELU does NOT have a working SFPU kernel implementation.** It is a registered but non-functional op type. Here is the state of each layer:

### 2.1 Registration Layers (exist but incomplete)

| Layer | Status | Location |
|-------|--------|----------|
| UnaryOpType enum | EXISTS | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp:51` -- `ELU,` |
| C++ inline function | EXISTS | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:189` -- `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(elu, ELU)` |
| Python nanobind binding | DISABLED | In `#if 0` block in `unary_nanobind.cpp:553` (float_parameter bindings were nuked) |
| Python golden function | NOT REGISTERED | Not in `ttnn/ttnn/operations/unary.py` |

### 2.2 Dispatch Layers (broken)

| Layer | Status | Issue |
|-------|--------|-------|
| `get_macro_definition(ELU)` | FALLS TO DEFAULT | Returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` (not a dedicated include) |
| `is_parametrized_type(ELU)` | RETURNS FALSE | Only HARDTANH, RPOW, SOFTSHRINK are parametrized |
| `get_op_init_and_func_parameterized(ELU)` | WOULD ASSERT | `TT_FATAL(is_parametrized_type(op_type))` fails for ELU |
| `get_op_init_and_func_default(ELU)` | NOT HANDLED | ELU is not in the switch statement |

**Critical dispatch bug**: When `ttnn::elu(tensor, alpha)` is called, the `REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER` macro creates `UnaryWithParam{UnaryOpType::ELU, alpha}`. Since params are non-empty, `get_op_init_and_func` routes to `get_op_init_and_func_parameterized`, which immediately asserts because `is_parametrized_type(ELU)` is false.

### 2.3 SFPU Kernel Layers (do not exist)

| Layer | Status |
|-------|--------|
| `ckernel_sfpu_elu.h` | DOES NOT EXIST |
| `llk_math_eltwise_unary_sfpu_elu.h` | DOES NOT EXIST |
| `eltwise_unary/elu.h` (compute API) | DOES NOT EXIST |
| `sfpu_split_includes.h` entry | NO `SFPU_OP_ELU_INCLUDE` entry |
| `SfpuType::elu` | NOT REGISTERED in `llk_sfpu_types.h` |

## 3. Closest Working Reference: SELU

SELU is ELU's closest relative -- it uses the same `alpha * (exp(x) - 1)` pattern for negative inputs, plus an outer scaling factor.

### 3.1 SELU vs ELU Comparison

| Aspect | SELU | ELU (would-be) |
|--------|------|----------------|
| Formula (x >= 0) | `scale * x` | `x` (identity) |
| Formula (x < 0) | `scale * alpha * (exp(x) - 1)` | `alpha * (exp(x) - 1)` |
| Alpha | Fixed: 1.6732632... | User parameter (default 1.0) |
| Scale | Fixed: 1.0507009... | None (1.0) |
| Parameter passing | No params (constants embedded) | Needs 1 float param via runtime arg |

### 3.2 SELU Kernel Structure (reference for ELU implementation)

**File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu() {
    constexpr bool SCALE_EN = false;
    constexpr bool SKIP_POSITIVE_CHECK = false;
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B;

    // Fixed constants as FP32 hex
    sfpi::vFloat v_alpha = Converter::as_float(0x3FD63840);  // 1.6732632...
    sfpi::vFloat v_scale = Converter::as_float(0x3F868640);  // 1.0507009...

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v < 0.0f) {
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<
                APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(v, exp_base_scale_factor);
            v = v_alpha * (v_exp - 1.0f);
        }
        v_endif;

        v = v_scale * v;  // ELU would NOT need this line

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}
```

**Init function**:
```cpp
template <bool APPROXIMATION_MODE>
inline void selu_init() {
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;  // 1.0f
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}
```

### 3.3 SELU Compute API

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`

```cpp
#include "llk_math_eltwise_unary_sfpu_selu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {
ALWI void selu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst));
}
ALWI void selu_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX));
}
}
```

### 3.4 SELU LLK Wrapper

**File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h`

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_selu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROXIMATE>(ckernel::sfpu::selu_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_selu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_selu<APPROXIMATE>, dst_index, vector_mode);
}
```

## 4. How to Implement ELU (Design Blueprint)

ELU is simpler than SELU: remove the scale factor, make alpha a runtime parameter.

### 4.1 SFPU Kernel: `ckernel_sfpu_elu.h`

The kernel would take `alpha` as a `uint32_t` parameter (bit-cast from float):

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_elu(uint32_t alpha_u) {
    constexpr bool SCALE_EN = false;
    constexpr bool SKIP_POSITIVE_CHECK = false;
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B;

    sfpi::vFloat v_alpha = Converter::as_float(alpha_u);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v < 0.0f) {
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<
                APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(v, exp_base_scale_factor);
            v = v_alpha * (v_exp - 1.0f);
        }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void elu_init() {
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;  // 1.0f
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}
```

### 4.2 Compute API: `eltwise_unary/elu.h`

ELU takes one runtime parameter (alpha as uint32_t), so use `SFPU_UNARY_ONE_PARAM_KERNEL_FN`:

```cpp
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_elu.h"

namespace ckernel {
ALWI void elu_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(elu, ckernel::sfpu::elu_init, APPROX));
}
ALWI void elu_tile(uint32_t idst, uint32_t slope) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_elu, RC, APPROX, idst, slope));
}
}
```

Note: The docs at `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/elu_tile.rst` already reference this exact signature: `elu_tile(uint32_t idst, uint32_t slope)`.

### 4.3 Dispatch Registration

In `unary_op_utils.cpp`:
1. Add `case UnaryOpType::ELU:` to `is_parametrized_type` returning `true`
2. Add to `get_op_init_and_func_parameterized`:
   ```cpp
   case UnaryOpType::ELU:
       return {"elu_tile_init();",
               fmt::format("elu_tile({}, {:#010x}u);", idst, std::bit_cast<uint32_t>(param0))};
   ```
3. Optionally add `SFPU_OP_ELU_INCLUDE` to `get_macro_definition` and `sfpu_split_includes.h`

### 4.4 Files to Create/Modify

**New files** (4):
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h` (identical copy)
- `tt_metal/hw/inc/api/compute/eltwise_unary/elu.h`
- Optionally: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_elu.h`

**Modified files** (3-5):
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- add ELU to `is_parametrized_type`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- add ELU dispatch
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- add `SFPU_OP_ELU_INCLUDE` guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- re-enable elu binding
- `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu_types.h` -- add `elu` to SfpuType enum

## 5. SFPI Instruction Usage

Based on the SELU reference, ELU would use these SFPI primitives:

| SFPI Construct | Usage |
|----------------|-------|
| `sfpi::dst_reg[0]` | Load/store tile elements from DST register |
| `sfpi::dst_reg++` | Advance to next face element |
| `sfpi::vFloat` | SIMD float vector type |
| `Converter::as_float(uint32_t)` | Bit-cast uint32_t to vFloat constant |
| `v_if(v < 0.0f) ... v_endif` | Conditional execution (negative branch) |
| `_calculate_exponential_piecewise_` | Piecewise exponential approximation |
| `_init_exponential_` | Initialize exponential LUT/coefficients |
| `#pragma GCC unroll 0` | Prevent loop unrolling (saves code space) |

## 6. Key Implementation Considerations

### 6.1 Parameter Passing
- ELU is a **parametrized** op (takes `alpha` float)
- The alpha value must be:
  1. Bit-cast to `uint32_t` at the host side via `std::bit_cast<uint32_t>(alpha)`
  2. Passed through the SFPU_OP_CHAIN define system as a hex literal
  3. Recovered in the kernel via `Converter::as_float(alpha_u)`

### 6.2 Exponential Precision Concerns
- Wave 0 results show SELU achieved only **43.96%** pass rate on kernel_bench due to `exp(x)-1` catastrophic cancellation
- For small negative x, `exp(x) - 1` loses significant precision because `exp(x) ~ 1`
- This is an inherent limitation of the `_calculate_exponential_piecewise_` approach
- Consider: if a dedicated `expm1` SFPU primitive becomes available, use it instead

### 6.3 Macro Pattern Selection
- SELU uses `SFPU_UNARY_NO_PARAM_KERNEL_FN` (no runtime params, constants embedded)
- ELU needs `SFPU_UNARY_ONE_PARAM_KERNEL_FN` (one runtime param: alpha)
- The LLK wrapper is NOT needed if using the macro system directly in the compute API header (per README.md recommendation)

### 6.4 Architecture Parity
- Both `wormhole_b0` and `blackhole` need identical `ckernel_sfpu_elu.h` files
- The SELU kernel is identical between architectures (same `ckernel_sfpu_selu.h`)

## 7. Test Infrastructure

Existing test files that reference ELU (would need working kernel):
- `tests/ttnn/unit_tests/operations/eltwise/test_activation.py` -- `test_scalarB_elu` calling `ttnn.elu(tensor, alpha=scalar)`
- `tests/sweep_framework/sweeps/eltwise/unary/elu/elu.py` -- sweep test with `ttnn.elu(input, alpha, ...)`
- `tests/sweep_framework/sweeps/eltwise/unary/elu/elu_pytorch2.py`
- `tests/sweep_framework/sweeps/eltwise/unary/elu/elu_sharded.py`
- `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_elu.py`

The sweep test uses `F.elu(torch_input, alpha=alpha)` as golden reference with PCC=0.999.

## 8. Relevance to sinh

ELU's `exp(x) - 1` pattern is structurally similar to sinh's `(exp(x) - exp(-x)) / 2`:
- Both require exponential computation via `_calculate_exponential_piecewise_`
- Both require the same `_init_exponential_` setup
- Both suffer from the same precision concerns with exponential subtraction
- sinh additionally needs `exp(-x)`, which ELU does not

The SELU kernel (ELU's implemented cousin) demonstrates the complete pattern: conditional branching, exponential computation, arithmetic combination, and DST register management.
