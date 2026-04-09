# Reference Analysis: hardtanh

## Operation Overview
- **Name**: hardtanh
- **Math**: max(min_val, min(max_val, x))
- **Parameters**: min_val (float, default -1.0), max_val (float, default 1.0)
- **Type**: Parameterized clamping operation

## Full Stack Analysis

### Layer 1: SFPU Kernel
**Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0);  // -(neg_threshold)
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1);  // -(pos_threshold - neg_threshold)
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2);  // -(pos_threshold)
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val += p0;
        v_if (val < 0.0f) { val = 0.0f; } v_endif;
        val += p1;
        v_if (val >= 0.0f) { val = 0.0f; } v_endif;
        val += p2;
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
```

### Key Patterns
1. **Parameter conversion**: `sfpi::s2vFloat16b(param)` converts uint32_t to vFloat in FP16_B format
2. **Pre-computed parameters**: Parameters are pre-computed on host (negated thresholds) to reduce SFPU work
3. **Multi-parameter passing**: 3 uint32_t params through function signature
4. **Iteration with `iterations` parameter**: Uses `const int iterations` runtime param + ITERATIONS template
5. **v_if/v_endif conditional masking**: Standard conditional execution pattern

### Layer 5: C++ API (unary.hpp)
```cpp
inline Tensor hardtanh(const Tensor& input_tensor, float min_val = -1.0f, float max_val = 1.0f, ...) {
    return ttnn::detail::unary_impl(input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, min_val, max_val}},
        ...);
}
```

### Layer 6: Python Binding (unary_nanobind.cpp)
```cpp
ttnn::bind_function<"hardtanh">(mod, doc.c_str(),
    &unary_two_float_5param_to_6param_wrapper<&ttnn::hardtanh>,
    nb::arg("input_tensor"), nb::kw_only(),
    nb::arg("min_val") = -1.0f, nb::arg("max_val") = 1.0f,
    nb::arg("memory_config") = nb::none(), nb::arg("output_tensor") = nb::none());
```

### Relevance to rrelu
- **HIGH**: Shows how to pass 2+ float parameters through UnaryWithParam
- **HIGH**: Shows the Python binding pattern for two-float-parameter operations
- **HIGH**: Shows s2vFloat16b() for parameter conversion in SFPU kernel
- **MEDIUM**: Different math (clamping vs. conditional scaling), but same infrastructure
