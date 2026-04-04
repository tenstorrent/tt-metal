# SFPU Kernel Analysis: hardtanh

## Operation
- **Name**: hardtanh
- **Definition**: clamp(x, min_val, max_val) = max(min_val, min(max_val, x))
- **Parameters**: min_val (float), max_val (float)

## Architecture Layers

### Layer 1: SFPU Kernel (`ckernel_sfpu_hardtanh.h`)
**Location**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1) {
    // Load both params outside the loop for better performance
    // param0 = min_val -> LREG2, param1 = max_val -> LREG3
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, param0 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, param0 >> 16);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, param1 & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, param1 >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // x = max(x, min_val) using LREG2
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);  // smaller to LREG0
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);  // store max

        // x = min(x, max_val) using LREG3
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);  // smaller to LREG0
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);  // store min

        sfpi::dst_reg++;
    }
}
```

**Key patterns**:
- Uses low-level TTI instructions for min/max (SFPSWAP)
- Loads parameters into LREG2/LREG3 before the loop for efficiency
- SFPSWAP with flag=1 puts the smaller value in LREG0

### Layer 2: LLK Dispatch (`llk_math_eltwise_unary_sfpu_hardtanh.h`)
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardtanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardtanh(
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
}
```

### Layer 3: Compute API (`api/compute/eltwise_unary/hardtanh.h`)
```cpp
ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)));
}
ALWI void hardtanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>())); }
```

## Relevance to hardsigmoid
- **Direct reuse**: The TTI_SFPSWAP min/max pattern can clamp the final result of hardsigmoid
- **Parameter loading**: TT_SFPLOADI pattern for loading constants into LREG registers
- **No init function needed**: Simple init with no custom callback (SFPU_UNARY_KERNEL_INIT)
