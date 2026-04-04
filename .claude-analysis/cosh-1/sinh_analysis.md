# sinh -- SFPU Kernel Analysis

## SFPU Kernel Implementation

### Unary Dispatch Summary
- **UnaryOpType**: `SINH`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `sinh_tile(0)`
- **Macro group**: `SFPU_OP_TRIG_FAMILY_INCLUDE`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | false | `get_op_approx_mode(SINH)` falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (default) | `get_op_init_and_func()` returns `{"sinh_tile_init();", "sinh_tile(0);"}` -- no param |
| Effective SFPU path | Non-approximate exponential via `_sfpu_exp_21f_bf16_` | Both controls resolve to non-approximate mode |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (via `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` and `SFPU_INIT_KERNEL_CALL` macros) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` |
| **Parameters Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` (init), macros file (params dispatch) |

### Call Chain

1. `sinh_tile(idst)` in `trigonometry.h` calls `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_sinh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)`
2. The macro expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_sinh<APPROX, DST_ACCUM_MODE, 8>, idst, (int)VectorMode::RC)`
3. The params dispatch function sets up DEST addressing and calls `calculate_sinh<false, is_fp32_dest_acc_en, 8>()` once per face (4 faces per tile)
4. `sinh_tile_init()` calls `SFPU_INIT_KERNEL_CALL(sinh, ckernel::sfpu::init_hyperbolic_trig, APPROX)` which calls `init_hyperbolic_trig<false>()` -> `_init_exponential_<false, false, p_sfpu::kCONST_1_FP16B>()`

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces of the tile
- **Operation invocation**: The dispatch function calls `calculate_sinh<APPROX, is_fp32_dest_acc_en, 8>()` once per face. ITERATIONS=8 processes all 8 sfpi rows per face (256 elements per face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces)

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h

// sinh = (exp(x) - exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS> // APPROXIMATION_MODE=false
inline void calculate_sinh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];  // Load input element from DEST
        sfpi::vFloat result =
            (_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v) - _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)) * 0.5f;
        sfpi::dst_reg[0] = result;  // Write result back to DEST
        sfpi::dst_reg++;  // Advance to next sfpi row (32 elements)
    }
}

// Init function for both cosh and sinh
template <bool APPROXIMATION_MODE>
void init_hyperbolic_trig() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
}
```

### SFPU Instructions Used
- `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)`: Computes e^v using a 21-term bfloat16 exponential approximation. Called twice: once for v and once for -v.
- `SFPMAD` (implicit via multiply `* 0.5f`): Multiplies result by 0.5
- `SFPLOAD` / `SFPSTORE` (implicit via `dst_reg[0]` read/write): Loads/stores from DEST register
- Subtraction is implemented as `SFPMAD` with appropriate negation

### SFPU Register Usage
- `dst_reg[0]`: Used for both input read and output write at the current sfpi address
- `dst_reg++`: Advances the sfpi address pointer by 1 (= 2 physical DEST rows = 32 elements)
- Internal to `_sfpu_exp_21f_bf16_`: uses LREGs for polynomial evaluation and intermediate results

### Address Mode Configuration
Standard unary SFPU address mode. ADDR_MOD_2 on Wormhole. The init function `_init_exponential_` sets up the address mode configuration for the exponential computation.
