# Reference Analysis: fill

## Operation Overview
- **Name**: fill
- **Math**: output[i] = fill_value for all i
- **Parameters**: value (float)
- **Type**: Constant fill

## SFPU Kernel Analysis

### Source: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`

### Key Patterns
1. **Simplest SFPU kernel**: Shows minimal iteration structure
2. **Direct float parameter**: `const float value` - no format conversion needed
3. **No conditional logic**: Pure assignment
4. **Includes**: `ckernel_ops.h`, `ckernel_sfpu_converter.h`, `ckernel_sfpu_load_config.h`

### Full Stack (for reference)

#### Compute API Pattern
```
void xxx_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_xxx<APPROX>(idst))); }
void xxx_tile_init() { MATH((llk_math_eltwise_unary_sfpu_xxx_init<APPROX>())); }
```

#### LLK Dispatch Pattern
```
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_xxx_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::xxx, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_xxx(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_xxx<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}
```

#### SFPU Kernel Pattern
```
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_xxx() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // ... computation ...
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

### Converter Utilities
- `Converter::as_float(uint32_t)` - interprets uint32_t bit pattern as float (bit_cast)
- `sfpi::s2vFloat16a(uint32_t)` - converts uint32_t to vFloat in FP16_A format
- `sfpi::s2vFloat16b(uint32_t)` - converts uint32_t to vFloat in FP16_B format

### Relevance to rrelu
- **MEDIUM**: Provides baseline template for SFPU kernel structure
- **MEDIUM**: Shows include requirements and namespace patterns
- **LOW**: Too simple (no conditional logic, no parameters)
