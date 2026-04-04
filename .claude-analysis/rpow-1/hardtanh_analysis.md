# Reference Analysis: hardtanh (HARDTANH)

## Overview
The `hardtanh` operation clamps input to [min_val, max_val]. It demonstrates the parameterized operation pattern.

## Parameterized Operation Pattern

### unary_op_utils.hpp - is_parametrized_type
```cpp
case UnaryOpType::HARDTANH: return true;
```

### unary_op_utils.cpp - get_op_init_and_func_parameterized
```cpp
case UnaryOpType::HARDTANH: {
    float min_val = params.size() > 0 ? param0 : -1.0f;
    float max_val = params.size() > 1 ? static_cast<float>(params[1]) : 1.0f;
    return {
        "hardtanh_tile_init();",
        fmt::format("hardtanh_tile({}, {:#010x}u, {:#010x}u);", idst,
            std::bit_cast<uint32_t>(min_val), std::bit_cast<uint32_t>(max_val))};
}
```

### unary_op_utils.cpp - get_macro_definition
```cpp
case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
```

### Compute API (hardtanh.h)
```cpp
ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)));
}
ALWI void hardtanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>())); }
```

### LLK Dispatch (llk_math_eltwise_unary_sfpu_hardtanh.h)
```cpp
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardtanh(
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
}
```

### SFPU Kernel (ckernel_sfpu_hardtanh.h)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);
    // ...loop over ITERATIONS...
}
```

## Relevance to rpow
rpow has ONE float parameter (base), so the pattern is similar but with a single parameter instead of two.
- `rpow_tile(idst, base_val)` - takes 1 param
- LLK passes 1 uint32_t to the SFPU kernel
- SFPU kernel decodes with Converter::as_float
