# SFPU Kernel Analysis: silu

## Operation
- **Name**: silu (Sigmoid Linear Unit)
- **Definition**: silu(x) = x * sigmoid(x)
- **Parameters**: none

## Architecture Layers

### Layer 1: SFPU Kernel (`ckernel_sfpu_silu.h`)
```cpp
#include "ckernel_sfpu_sigmoid.h"

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();
    } else {
        _init_sfpu_reciprocal_<true>();
    }
}
```

**Key patterns**:
- `#pragma GCC unroll 8` for full unrolling
- fp32/bf16 rounding via `float_to_fp16b` when not in fp32 accumulation mode
- Custom init function (`silu_init`) that initializes reciprocal tables
- Template parameters for fp32 accumulation mode

### Layer 2: LLK Dispatch (`llk_math_eltwise_unary_sfpu_silu.h`)
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_silu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu, APPROXIMATE>(sfpu::silu_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
}
```

### Layer 3: Compute API (in `compute_kernel_api.h`)
```cpp
ALWI void silu_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst))); }
ALWI void silu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_silu_init<APPROX>())); }
```

### SfpuType enum entry
```cpp
enum class SfpuType {
    ...
    silu,
    ...
};
```

## Relevance to hardsigmoid
- **Full stack template**: Shows complete wiring from ckernel -> LLK -> compute API
- **No-parameter operation**: Hardsigmoid also takes no runtime parameters
- **DST_ACCUM_MODE pattern**: Shows how to handle fp32 vs bf16 accumulation mode
- **Init function pattern**: silu needs a custom init; hardsigmoid does not (simpler)
