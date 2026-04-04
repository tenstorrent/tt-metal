# SFPU Kernel Analysis: hardtanh

## 1. Operation Overview

**Operation**: `hardtanh` (element-wise clamping activation)

**Math Definition**:
```
hardtanh(x, min_val, max_val) =
    min_val   if x < min_val
    x         if min_val <= x <= max_val
    max_val   if x > max_val
```

Default parameters: `min_val = -1.0`, `max_val = 1.0`.

**PyTorch equivalent**: `torch.nn.functional.hardtanh(x, min_val, max_val)`

**Classification**: Parametrized unary element-wise SFPU operation with two float parameters. This is a pure clamping operation — no transcendentals, no LUT usage, no approximation mode dependency.

---

## 2. SFPU Kernel Implementation

### 2.1 Core SFPU Kernel Function

**File (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
**File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`

Both architectures share identical implementation:

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    // param0 = min_val as IEEE 754 float bits (bitcast uint32_t)
    // param1 = max_val as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if(val < min_val) { val = min_val; }
        v_endif;

        v_if(val > max_val) { val = max_val; }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
```

### 2.2 SFPI Instructions Used

| SFPI Construct | Purpose | Notes |
|---|---|---|
| `Converter::as_float(uint32_t)` | Bitcast IEEE 754 `uint32_t` → `vFloat` | Reinterprets parameter bits as SFPU float vector |
| `sfpi::dst_reg[0]` (read) | Load one face-row from DST register | Standard SFPU data input |
| `v_if(val < min_val)` | Predicated comparison (less-than) | SFPU conditional — sets per-lane predicate mask |
| `v_if(val > max_val)` | Predicated comparison (greater-than) | SFPU conditional — sets per-lane predicate mask |
| `sfpi::dst_reg[0] = val` (write) | Store result back to DST register | Standard SFPU data output |
| `sfpi::dst_reg++` | Advance DST register pointer to next face-row | Iterates through the 8 face-rows per tile face |

### 2.3 Key Implementation Characteristics

- **No transcendental math**: Pure comparison and assignment — no `exp`, `log`, `recip`, `lut` usage.
- **No LUT dependency**: Does not use `lut_mode_set()` or any lookup table.
- **APPROXIMATION_MODE unused**: The `APPROXIMATION_MODE` template parameter is declared but never referenced in the body. The operation is exact for all floating-point values (modulo hardware float representation).
- **Two conditional blocks**: Two sequential `v_if/v_endif` blocks for lower and upper clamping. These compile to SFPU predicated instructions.
- **Parameter passing via bitcast**: `min_val` and `max_val` are passed as `uint32_t` (IEEE 754 bit representation) and reinterpreted on-device using `Converter::as_float()`.
- **Loop unroll**: `#pragma GCC unroll 8` — full unroll of the 8-iteration loop for maximum throughput.
- **Wormhole/Blackhole identical**: No architecture-specific divergence; same code for both targets.

---

## 3. Abstraction Layer Stack

### Layer 1: `SfpuType` Enum
**File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h`
```cpp
enum class SfpuType {
    unused = 0,
    cosh,
    cbrt,
    hardsigmoid,
    selu,
    hardtanh,   // <-- registered here
};
```

### Layer 2: LLK Math Wrapper
**File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardtanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardtanh(
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>,
        dst_index, vector_mode, param0, param1);
}
```

**Dispatch mechanism**: Uses `_llk_math_eltwise_unary_sfpu_params_` with two runtime parameters (`param0`, `param1`). The function pointer to `calculate_hardtanh` is passed as the compute functor.

### Layer 3: Compute API (Tile-level)
**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
```cpp
ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)));
}

ALWI void hardtanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()));
}
```

**Split include guard**: Conditionally included via `SFPU_OP_HARDTANH_INCLUDE` in `sfpu_split_includes.h`.

### Layer 4: `UnaryOpType` Enum
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
```cpp
HARDTANH,  // line 120
```

### Layer 5: Parametrized Type Registration
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
```cpp
template <typename T>
bool is_parametrized_type(T val) {
    switch (val) {
        case UnaryOpType::HARDTANH: return true;
        default: return false;
    }
}
```
HARDTANH is registered as a **parametrized type** — its init/func strings are generated via `get_op_init_and_func_parameterized()` rather than the default path.

### Layer 6: Compute Kernel Code Generation
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

**Macro define**: `get_macro_definition(UnaryOpType::HARDTANH)` → `"SFPU_OP_HARDTANH_INCLUDE"`

**Init/func string generation** (parametrized path):
```cpp
case UnaryOpType::HARDTANH: {
    float min_val = params.size() > 0 ? param0 : -1.0f;
    float max_val = params.size() > 1 ? static_cast<float>(params[1]) : 1.0f;
    return {
        "hardtanh_tile_init();",
        fmt::format("hardtanh_tile({}, {:#010x}u, {:#010x}u);",
            idst,
            std::bit_cast<uint32_t>(min_val),
            std::bit_cast<uint32_t>(max_val))};
}
```

**Parameter encoding**: `min_val` and `max_val` are bitcast from `float` to `uint32_t` using `std::bit_cast`, then emitted as hex literals (e.g., `0xbf800000u` for -1.0f) into the kernel source string. This is how float parameters reach the SFPU kernel without runtime argument passing — they are baked into the generated compute kernel source as compile-time constants.

### Layer 7: C++ API
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
```cpp
inline Tensor hardtanh(
    const Tensor& input_tensor,
    float min_val = -1.0f,
    float max_val = 1.0f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, min_val, max_val}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}
```

Note: Constructs `UnaryWithParam` with **two float parameters** (`min_val`, `max_val`).

### Layer 8: Python Nanobinding
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
```cpp
ttnn::bind_function<"hardtanh">(
    mod,
    doc.c_str(),
    &unary_two_float_5param_to_6param_wrapper<&ttnn::hardtanh>,
    nb::arg("input_tensor"),
    nb::kw_only(),
    nb::arg("min_val") = -1.0f,
    nb::arg("max_val") = 1.0f,
    nb::arg("memory_config") = nb::none(),
    nb::arg("output_tensor") = nb::none());
```

Uses `unary_two_float_5param_to_6param_wrapper` — a template wrapper that adapts the 6-parameter C++ signature to a 5-parameter Python signature (omitting `sub_core_grids`).

### Layer 9: Python Golden Function
**File**: `ttnn/ttnn/operations/unary.py`
```python
def _golden_function_hardtanh(input_tensor_a, *args, min_val=-1.0, max_val=1.0, **kwargs):
    import torch
    return torch.nn.functional.hardtanh(input_tensor_a, min_val=min_val, max_val=max_val)

ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)
```

---

## 4. Parameter Flow Summary

```
Python: ttnn.hardtanh(tensor, min_val=-1.0, max_val=1.0)
  │
  ▼ nanobind wrapper (unary_two_float_5param_to_6param_wrapper)
C++ API: ttnn::hardtanh(tensor, min_val, max_val, ...)
  │
  ▼ UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}
  │
  ▼ get_op_init_and_func_parameterized()
  │   float → uint32_t via std::bit_cast
  │   Emitted as hex literal in kernel source string
  │
  ▼ Generated compute kernel defines:
  │   SFPU_OP_CHAIN_0_INIT_0 = "hardtanh_tile_init();"
  │   SFPU_OP_CHAIN_0_FUNC_0 = "hardtanh_tile(0, 0xbf800000u, 0x3f800000u);"
  │
  ▼ hardtanh_tile(idst, param0, param1)                    [compute API]
  ▼ llk_math_eltwise_unary_sfpu_hardtanh(dst_index, p0, p1) [LLK wrapper]
  ▼ _llk_math_eltwise_unary_sfpu_params_()                  [generic dispatch]
  ▼ calculate_hardtanh(param0, param1)                       [SFPU kernel]
      Converter::as_float(param0) → vFloat min_val
      Converter::as_float(param1) → vFloat max_val
      Per-lane clamp via v_if comparisons
```

---

## 5. Testing

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py`

Test strategy:
- **Exhaustive bfloat16 bitpattern coverage**: Uses `generate_all_bfloat16_bitpatterns()` (256x256 tensor covering all 65536 bfloat16 values).
- **Parameter variants**: 4 `(min_val, max_val)` pairs: `(-1,1)`, `(-0.5,0.5)`, `(0,6)` (relu6-like), `(-2,2)`.
- **Dual precision**: Tests both bfloat16 and float32 input dtypes.
- **Subnormal handling**: Flushes subnormals to zero to match hardware behavior.
- **Tolerances**:
  - bfloat16: ULP threshold = 2, rtol = 1.6e-2, atol = 1e-2
  - float32: ULP threshold = 3, rtol = 1e-3, atol = 1e-4

---

## 6. Key Design Patterns for Reuse

### Pattern: Two-Parameter Clamping SFPU Kernel
This operation demonstrates the cleanest possible parametrized SFPU kernel pattern:

1. **Parameter passing**: Two `uint32_t` params bitcast from floats at code-generation time → `Converter::as_float()` on-device.
2. **Predicated assignment**: Two `v_if/v_endif` blocks for lower/upper bounds — the canonical SFPU branching pattern.
3. **No math dependencies**: No LUT, no exp/log, no approximation — just comparisons and assignments.
4. **Parametrized type registration**: Requires `is_parametrized_type() → true` and a case in `get_op_init_and_func_parameterized()`.
5. **Split include**: Uses `SFPU_OP_HARDTANH_INCLUDE` guard to avoid pulling the header into unrelated compute kernels.
6. **Python API with defaults**: `unary_two_float_5param_to_6param_wrapper` adapts the C++ 6-param signature to Python's 5-param signature.

### Applicable To
Any operation that clamps, clips, or applies piecewise-linear activation based on scalar float thresholds (e.g., `clamp`, `relu6`, `clip`, custom activation ranges).

---

## 7. File Manifest

| Layer | File Path |
|---|---|
| SFPU kernel (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` |
| SFPU kernel (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` |
| LLK wrapper (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` |
| LLK wrapper (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` |
| Compute API | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| Split includes | `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` |
| SfpuType enum | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h` |
| UnaryOpType enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` |
| Op utils (dispatch) | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` |
| Op utils (header) | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` |
| C++ API | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` |
| Nanobind | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` |
| Python golden | `ttnn/ttnn/operations/unary.py` |
| Unit test | `tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py` |
