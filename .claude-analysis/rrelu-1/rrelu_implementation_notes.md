# RReLU Implementation Notes

## Operation Summary
RReLU (Randomized Leaky ReLU) in evaluation mode:
- `rrelu(x) = x` if `x >= 0`
- `rrelu(x) = slope * x` if `x < 0`
- where `slope = (lower + upper) / 2`

Parameters: `lower` (default 0.125), `upper` (default 1/3)

## Architecture
Standard parametrized unary SFPU operation using the SFPU_OP_CHAIN dispatch path through `eltwise_sfpu.cpp`. The slope is pre-computed on the host from `(lower + upper) / 2`, packed as FP16_B (bfloat16 bits in uint32_t), and embedded as a hex literal in the SFPU_OP_CHAIN_0 macro string. No runtime args needed.

## Reference Operations Used
- **frac**: Primary template for the standard unary SFPU wiring pattern (SFPU kernel, LLK wrapper, compute API, split include, host dispatch). Frac is the cleanest no-parameter example.
- **hardtanh**: Primary template for parametrized ops (2 float params, `is_parametrized_type`, `get_op_init_and_func_parameterized`, `s2vFloat16b` parameter loading in kernel, C++ API with explicit params, `unary_two_float_5param_to_6param_wrapper` nanobind).
- **swish**: Referenced for golden function pattern and nanobind registration.

## Deviations from Standard Patterns
- None. Followed standard parametrized unary SFPU patterns exactly.

## Known Limitations
- Only evaluation mode is implemented (deterministic slope). Training mode (stochastic per-element slope from Uniform(lower, upper)) is not supported because the SFPU does not have a suitable random number generator for this use case.
- The slope is quantized to bfloat16 precision when packed for the SFPU kernel.

---

## Layer 1: SFPU Compute Kernel

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`

The core SFPU kernel implements the RReLU conditional logic using SFPI macros:

```cpp
// RReLU (Randomized Leaky ReLU) in evaluation mode:
//   rrelu(x) = x          if x >= 0
//   rrelu(x) = slope * x  if x < 0
//
// where slope = (lower + upper) / 2, pre-computed on the host.
// param0: slope packed as FP16_B (bfloat16 bits in a uint32_t).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint32_t param0) {
    sfpi::vFloat slope = sfpi::s2vFloat16b(param0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) { result = x * slope; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

**Key techniques**:
- `sfpi::s2vFloat16b()`: Unpacks bfloat16 (upper 16 bits of uint32_t) to vector float
- `v_if`/`v_endif`: SFPI conditional macros for data-parallel execution
- `sfpi::dst_reg`: Destination register iterator (8 elements per iteration across ITERATIONS loops)

**Note**: Blackhole variant at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` is identical (architecture-independent compute logic).

---

## Layer 2: LLK Math Abstraction

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`

LLK layer wraps the SFPI kernel in the standard parameter-passing template:

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, int vector_mode = (int)VectorMode::RC, uint32_t param0 = 0) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
}
```

**Key design**:
- `_llk_math_eltwise_unary_sfpu_params_`: Generic macro that dispatches to SFPI kernel with parameter management
- Separates init and function calls for DST register acquisition/release semantics
- `APPROXIMATE` template parameter for fast vs. accurate mode (unused for rrelu, always false)

**Note**: Blackhole variant at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` is identical.

---

## Layer 3: Compute API (Host-visible tile operations)

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

Public C++ API for calling rrelu from compute kernel code:

```cpp
// clang-format off
 /**
 * Performs element-wise RReLU operation (eval mode): max(0,x) + slope*min(0,x).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Slope value packed as FP16_B (bfloat16 bits)                               | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, static_cast<int>(VectorMode::RC), param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }
```

**Key design**:
- `rrelu_tile_init()`: Initializes DST register state for rrelu computation
- `rrelu_tile(idst, param0)`: Applies rrelu to tile at DST index with packed slope parameter
- `ALWI` / `MATH()`: TT-Metalium macros for always-inline host functions that generate device code

---

## Layer 4: Type System Registrations

### 4a. SFPU Type Enum
**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

```cpp
enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,
    sinh,
    rrelu,  // <-- Added here
    // ... other types
};
```

### 4b. Split Include Guard
**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

```cpp
#if SFPU_OP_RRELU_INCLUDE
#include "api/compute/eltwise_unary/rrelu.h"
#endif
```

**Purpose**: Compile-time guard allows host code to conditionally include rrelu API based on `SFPU_OP_RRELU_INCLUDE` define set during dispatch.

---

## Layer 5: C++ Operation Types and Utilities

### 5a. UnaryOpType Enum
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`

```cpp
enum class UnaryOpType {
    // ... many operations
    SWISH,
    RRELU,  // <-- Added here
};
```

### 5b. Parametrized Type Detection
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`

```cpp
template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::HARDTANH: return true;
        case UnaryOpType::SOFTSHRINK: return true;
        case UnaryOpType::RRELU: return true;  // <-- Added here
        default: return false;
    }
    return false;
}
```

### 5c. Parameter Processing and Macro Generation
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Host-side slope computation and packing:

```cpp
template <typename T>
std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type,
    std::span<const T> params,
    [[maybe_unused]] const std::string& idst,
    [[maybe_unused]] std::optional<DataType> input_dtype) {
    TT_FATAL(
        is_parametrized_type(op_type),
        "operator should support at least one parameter but op_type {} does not",
        op_type);
    // TODO don't cast T to float when precision needs to be preserved
    [[maybe_unused]] const T param0_raw = params[0];
    [[maybe_unused]] float param0 = static_cast<float>(params[0]);
    switch (op_type) {
        case UnaryOpType::RRELU: {
            float lower = param0;
            float upper = params.size() > 1 ? static_cast<float>(params[1]) : (1.0f / 3.0f);
            float slope = (lower + upper) / 2.0f;
            uint32_t packed = std::bit_cast<uint32_t>(slope) >> 16;  // Extract upper 16 bits (bfloat16)
            return {"rrelu_tile_init();", fmt::format("rrelu_tile({}, 0x{:x});", idst, packed)};
        }
        default: TT_THROW("unexpected parameterized op type {}", op_type);
    };
}
```

Macro definition mapping:

```cpp
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::FRAC: return "SFPU_OP_FRAC_INCLUDE";
        case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";
        case UnaryOpType::ATANH: return "SFPU_OP_ATANH_INCLUDE";
        case UnaryOpType::SINH: return "SFPU_OP_SINH_INCLUDE";
        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";  // <-- Added here
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    };
}
```

**Key algorithm**:
1. Extract `lower` and `upper` float parameters
2. Compute `slope = (lower + upper) / 2.0f`
3. Bit-cast slope to uint32_t, then right-shift by 16 to extract bfloat16 (upper 16 bits)
4. Format as hex literal embedded in tile call: `rrelu_tile(idst, 0xPACKED_VALUE)`

### 5d. unary_ng utilities (deprecated/alternative path)
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`

Also includes rrelu support for alternative dispatch path:

```cpp
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::FRAC: return "SFPU_OP_FRAC_INCLUDE";
        case UnaryOpType::SWISH: return "SFPU_OP_SWISH_INCLUDE";
        case UnaryOpType::SINH: return "SFPU_OP_SINH_INCLUDE";
        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";  // <-- Added here
        default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
    }
}
```

---

## Layer 6: C++ API Registration

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

Public C++ function signature:

```cpp
// rrelu: two float parameters (lower, upper) for evaluation mode slope
inline Tensor rrelu(
    const Tensor& input_tensor,
    float lower = 0.125f,
    float upper = 1.0f / 3.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RRELU, lower, upper}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}
```

**Design**:
- Two float parameters: `lower` (default 0.125) and `upper` (default 1/3)
- Creates `UnaryWithParam` variant with both float parameters
- Delegates to `unary_impl` which dispatches through the parameter-aware codegen path

---

## Layer 7: Python Nanobind Bindings

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

Helper wrapper for two-float-parameter operations:

```cpp
template <auto Func>
Tensor unary_two_float_5param_to_6param_wrapper(
    const Tensor& input_tensor,
    float parameter_a,
    float parameter_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    return Func(input_tensor, parameter_a, parameter_b, memory_config, output_tensor, std::nullopt);
}
```

Nanobind binding registration:

```cpp
{
    auto doc = R"doc(
        Applies the RReLU (Randomized Leaky ReLU) function element-wise in evaluation mode.

        .. math::
            \mathrm{{output\_tensor}}_i = \max(0, x) + \frac{{\mathrm{{lower}} + \mathrm{{upper}}}}{{2}} \cdot \min(0, x)

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            lower (float, optional): lower bound of the uniform distribution. Defaults to `0.125`.
            upper (float, optional): upper bound of the uniform distribution. Defaults to `0.3333`.
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
        )doc";

    ttnn::bind_function<"rrelu">(
        mod,
        doc.c_str(),
        &unary_two_float_5param_to_6param_wrapper<&ttnn::rrelu>,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("lower") = 0.125f,
        nb::arg("upper") = 1.0f / 3.0f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("output_tensor") = nb::none());
}
```

**Design**:
- Wrapper function adapts 5-parameter C++ API (lower, upper, memory_config, output_tensor, sub_core_grids) to 4-parameter Python API (lower, upper, memory_config, output_tensor)
- Documentation includes LaTeX math formula and parameter descriptions
- Keyword-only arguments via `nb::kw_only()`
- Default parameter values: `lower=0.125`, `upper=1/3`

---

## Layer 8: Python Golden Function

**File**: `ttnn/ttnn/operations/unary.py`

Testing integration with torch reference:

```python
def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=1.0 / 3.0, **kwargs):
    import torch

    return torch.nn.functional.rrelu(input_tensor_a, lower=lower, upper=upper, training=False)


ttnn.attach_golden_function(ttnn.rrelu, golden_function=_golden_function_rrelu)
```

**Purpose**:
- Provides PyTorch ground truth for testing and validation
- `training=False` ensures evaluation mode (deterministic slope) matches TTNN implementation
- Attached at module load time for automatic test harness integration

---

## Implementation Summary by Abstraction Layer

| Layer | File(s) | Key Concept |
|-------|---------|-------------|
| **SFPI Kernel** | `ckernel_sfpu_rrelu.h` (Wormhole, Blackhole) | SFPU compute logic: conditional multiply |
| **LLK Math** | `llk_math_eltwise_unary_sfpu_rrelu.h` | Parameter passing template wrapper |
| **Compute API** | `rrelu.h` | Public tile-level function signatures |
| **Type Registry** | `llk_sfpu_types.h`, `sfpu_split_includes.h` | Enum registration, compile guards |
| **Op Types** | `unary_op_types.hpp`, `unary_op_utils.hpp/cpp` | Parametrized type flag, slope computation |
| **C++ API** | `unary.hpp` | High-level Tensor → Tensor operation |
| **Python Bindings** | `unary_nanobind.cpp` | 2-parameter wrapper, Python signature |
| **Golden Function** | `unary.py` | PyTorch reference for testing |

---

## Data Flow: Host to Device

```
Python: ttnn.rrelu(tensor, lower=0.2, upper=0.4)
  ↓ [Nanobind wrapper → C++ wrapper]
C++: ttnn::rrelu(tensor, 0.2, 0.4, ...)
  ↓ [unary_impl dispatch]
Codegen: get_op_init_and_func_parameterized(RRELU, [0.2, 0.4])
  ↓ [Slope computation]
slope = (0.2 + 0.4) / 2 = 0.3
packed = bit_cast<uint32_t>(0.3) >> 16 = 0x...
  ↓ [Macro generation]
SFPU_OP_RRELU_INCLUDE = 1
"rrelu_tile_init(); rrelu_tile(0, 0x...);"
  ↓ [Kernel compilation at runtime]
Device: rrelu_tile_init() → rrelu_tile(0, param0)
  ↓ [SFPU execution]
slope = s2vFloat16b(param0)
for each x: result = (x < 0) ? x * slope : x
```

---

## Testing Notes

The implementation supports golden function testing via:

```python
import torch
import ttnn

# Create test tensor
x = torch.randn(32, 32, dtype=torch.bfloat16)
x_ttnn = ttnn.from_torch(x, ...)

# Run operation
output_ttnn = ttnn.rrelu(x_ttnn, lower=0.2, upper=0.4)
output_torch = torch.nn.functional.rrelu(x, lower=0.2, upper=0.4, training=False)

# Compare
assert torch.allclose(ttnn.to_torch(output_ttnn), output_torch, atol=1e-2)
```

**Known precision considerations**:
- Slope is quantized to bfloat16 (16-bit precision)
- SFPU operations may accumulate rounding error compared to float32 torch
- Test tolerances should account for ≤2-3 ULP (unit in the last place) differences

---

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` — Added `rrelu` enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` — Added `rrelu` enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` — Added conditional include guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` — Added `RRELU` to enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` — Added parametrized dispatch case
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` — Added `RRELU` to `is_parametrized_type`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` — Added C++ function signature
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` — Added nanobind binding
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` — Added alternative dispatch path support
- `ttnn/ttnn/operations/unary.py` — Added golden function attachment
