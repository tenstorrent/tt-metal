# Reference Analysis: dropout

## Operation Overview
- **Name**: dropout
- **Math**: output = 0 if random < probability; output = x * scale otherwise
- **Parameters**: probability (uint32_t), scale (uint32_t), seed (uint32_t for init)
- **Type**: Stochastic operation with RNG

## SFPU Kernel Analysis

### Source: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h`

### Key Implementation Patterns

1. **Template signature**: `template <bool APPROXIMATION_MODE, int ITERATIONS>`
2. **RNG generation**: `TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8)` - generates random uint32_t when instr_mod1=8 and lreg_c=9
3. **PRNG init**: `init_prng_seed(seed)` called from `_init_dropout_`
4. **Raw TTI instructions**: Uses `TT_SFPLOADI`, `TTI_SFPLOAD`, `TTI_SFPMUL`, `TTI_SFPMOV`, etc. instead of SFPI C++ API
5. **Iteration pattern**: `#pragma GCC unroll 0` (no unrolling for RNG)
6. **Sign clearing**: `TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1)` - clears sign bit for unsigned comparison

### RNG Pattern (Critical for rrelu training mode)
```
// Generate random number
TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);  // Random uint32 in LREG3
TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // Clear sign bit

// Compare random vs threshold
TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);  // Integer comparison
TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Conditional set to zero
TTI_SFPENCC(0, 0, 0, 0);  // End conditional
```

### Init Function
```cpp
inline void _init_dropout_(const std::uint32_t seed) {
    init_prng_seed(seed);
}
```

### Relevance to rrelu
- **CRITICAL**: For training mode, rrelu needs per-element random slope from Uniform(lower, upper)
- **HIGH**: Shows RNG instruction pattern (SFPMOV with instr_mod1=8, lreg_c=9)
- **HIGH**: Shows init_prng_seed() for RNG initialization
- **MEDIUM**: dropout uses raw TTI instructions; rrelu can use higher-level SFPI API with v_if
- **NOTE**: For eval mode, rrelu doesn't need RNG at all (slope is deterministic)
