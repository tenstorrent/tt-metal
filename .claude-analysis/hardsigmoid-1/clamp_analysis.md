# SFPU Kernel Analysis: clamp

## Operation
- **Name**: clamp
- **Definition**: min(max(x, min_val), max_val)
- **Parameters**: min_val, max_val (both float, passed as uint)

## Architecture Layers

### Layer 1: SFPU Kernel (`ckernel_sfpu_clamp.h`)
```cpp
#include "ckernel_sfpu_unary_max_min.h"

enum { Max = true, Min = false };

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint min_val, uint max_val) {
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(min_val);
        calculate_unary_max_min_float_body<Max>();
        load_value_param_float(max_val);
        calculate_unary_max_min_float_body<Min>();
        sfpi::dst_reg++;
    }
}
```

**Key patterns**:
- Delegates to shared `unary_max_min` primitives
- Uses `load_value_param_float` + `calculate_unary_max_min_float_body` for each clamp direction
- Sequential max then min application
- Parameters reloaded each iteration (less efficient than hardtanh's pre-load)

### Layer 2: LLK Dispatch (`llk_math_eltwise_unary_sfpu_clamp.h`)
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_clamp_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::clamp, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_clamp(
    uint dst_index, uint min_val, uint max_val, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_clamp<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, min_val, max_val);
}
```

### Layer 3: Compute API (`api/compute/eltwise_unary/clamp.h`)
```cpp
ALWI void clamp_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_clamp<APPROX>(idst, param0, param1)));
}
ALWI void clamp_tile_init() { MATH((llk_math_eltwise_unary_sfpu_clamp_init<APPROX>())); }
```

## Relevance to hardsigmoid
- **Two-parameter clamp**: Same structural pattern of applying min then max
- **Simple init**: No custom init callback needed
- **Alternative approach**: Could implement hardsigmoid as linear transform + clamp
