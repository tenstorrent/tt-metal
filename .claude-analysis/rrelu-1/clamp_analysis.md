# Reference Analysis: clamp

## Operation Overview
- **Name**: clamp
- **Math**: min(max(x, min_val), max_val) with offset
- **Parameters**: min (uint32_t, FP16_A), max (uint32_t, FP16_A), offset (uint32_t, FP16_B)
- **Type**: Parameterized conditional clamping

## SFPU Kernel Analysis

### Source: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`

### Key Patterns
1. **Mixed format conversion**: Uses both `s2vFloat16a(param)` and `s2vFloat16b(param)` for different precision
2. **v_if/v_elseif/v_endif branching**: Shows multi-branch conditional
3. **Parameter pre-loading**: Parameters converted to vFloat before loop for efficiency
4. **Unroll pragma**: `#pragma GCC unroll 0` (no unrolling for parameterized ops)

### SFPI Instructions Used
- `sfpi::s2vFloat16a()` - FP16_A format conversion (for min/max)
- `sfpi::s2vFloat16b()` - FP16_B format conversion (for offset)
- `v_if` / `v_elseif` / `v_endif` - multi-branch conditional
- Direct comparison operators: `val < min`, `val >= max`

### Code Pattern
```cpp
sfpi::vFloat min = sfpi::s2vFloat16a(param0);
sfpi::vFloat max = sfpi::s2vFloat16a(param1);
sfpi::vFloat offset = sfpi::s2vFloat16b(param2);
for (int d = 0; d < iterations; d++) {
    sfpi::vFloat val = sfpi::dst_reg[0];
    v_if (val < min) { val = min; }
    v_elseif (val >= max) { val = max; }
    v_endif;
    sfpi::dst_reg[0] = val + offset;
    sfpi::dst_reg++;
}
```

### Relevance to rrelu
- **HIGH**: Shows v_if/v_elseif/v_endif multi-branch pattern (rrelu has 2 branches)
- **HIGH**: Shows FP16_A and FP16_B conversion functions
- **MEDIUM**: Clamping vs. scaling, but similar conditional structure
