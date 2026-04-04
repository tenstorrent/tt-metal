# SFPU Kernel Analysis: heaviside

## Operation
- **Name**: heaviside
- **Definition**: 0 if x < 0, 1 if x > 0, value if x == 0
- **Parameters**: value (float for x=0 case)

## Architecture Layers

### Layer 1: SFPU Kernel (`ckernel_sfpu_heaviside.h`)
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(uint value) {
    vFloat s = Converter::as_float(value);
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(v < 0.0f) { v = 0.0f; }
        v_elseif(v > 0.0f) { v = 1.0f; }
        v_else { v = s; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}
```

**Key patterns**:
- Uses `v_if`/`v_elseif`/`v_else`/`v_endif` for three-way branching
- Assigns literal float constants (0.0f, 1.0f) directly
- `#pragma GCC unroll 0` disables unrolling (unusual, most ops use `unroll 8`)
- Simple parameter handling with `Converter::as_float`

### Layer 2: LLK Dispatch
Standard pattern with `_llk_math_eltwise_unary_sfpu_params_` and `SfpuType::heaviside`.

### Layer 3: Compute API
Declared in `compute_kernel_api.h`:
```cpp
ALWI void heaviside_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_heaviside<APPROX>(idst, param0)));
}
ALWI void heaviside_tile_init() { MATH((llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>())); }
```

## Relevance to hardsigmoid
- **Conditional constant assignment**: Hardsigmoid outputs exactly 0 when x <= -3, exactly 1 when x >= 3
- **v_if/v_elseif/v_else pattern**: Three-region piecewise function, same structure as hardsigmoid
- **Simple arithmetic**: Only assigns constants, no complex math
