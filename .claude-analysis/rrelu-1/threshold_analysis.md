# Reference Analysis: threshold

## Operation Overview
- **Name**: threshold
- **Math**: output = x if x > threshold; output = value if x <= threshold
- **Parameters**: threshold (float), value (float)
- **Type**: Conditional piecewise with 2 parameters

## SFPU Kernel Analysis

### Source: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`

### Key Implementation Patterns

1. **Template signature**: `template <bool APPROXIMATION_MODE, int ITERATIONS, typename T>`
2. **Parameter conversion**: Uses `Converter::as_float(threshold)` for uint32_t params, direct assignment for float
3. **Conditional branching**: Uses `v_if (in <= v_threshold) { dst_reg[0] = v_value; } v_endif;`
4. **Iteration pattern**: `#pragma GCC unroll 8` + `for (int d = 0; d < ITERATIONS; d++)`
5. **Register access**: `sfpi::dst_reg[0]` for read/write, `sfpi::dst_reg++` to advance

### Parameter Passing
- Parameters come in as `T threshold, T value` (templated on float or uint32_t)
- Type-based dispatch: `if constexpr (std::is_same_v<T, float>)` vs `if constexpr (std::is_same_v<T, std::uint32_t>)`
- Float params assigned directly to `sfpi::vFloat`
- uint32_t params converted via `Converter::as_float(param)`

### SFPI Instructions Used
- `sfpi::vFloat` - vector float type
- `sfpi::dst_reg[0]` - destination register access
- `v_if` / `v_endif` - conditional execution (lane masking)
- `Converter::as_float()` - bit-level float conversion

### Relevance to rrelu
- **HIGH**: Shows the exact conditional check pattern (x <= 0 comparison)
- **HIGH**: Shows parameter conversion pattern (uint32_t -> vFloat)
- **MEDIUM**: Only 1 branch (rrelu needs 2 branches: x >= 0 passthrough, x < 0 scale)
