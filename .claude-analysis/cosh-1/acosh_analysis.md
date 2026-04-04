# acosh -- SFPU Kernel Analysis

## SFPU Kernel Implementation

### Unary Dispatch Summary
- **UnaryOpType**: `ACOSH`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `acosh_tile(0)`
- **Macro group**: `SFPU_OP_TRIG_FAMILY_INCLUDE`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | false | `get_op_approx_mode(ACOSH)` falls through to `default: return false` |
| Template parameter | `8` (ITERATIONS via `SFPU_TWO_PARAM_KERNEL`) | `get_op_init_and_func()` returns `{"acosh_tile_init();", "acosh_tile(0);"}` |
| Effective SFPU path | Non-approximate path using log, sqrt, and multiply | Single code path |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (via `SFPU_TWO_PARAM_KERNEL`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` |

### Call Chain

1. `acosh_tile(idst)` calls `SFPU_TWO_PARAM_KERNEL(_calculate_acosh_, APPROX, 8, idst, (int)VectorMode::RC)`
2. Expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_acosh_<APPROX, 8>, idst, (int)VectorMode::RC)`
3. `acosh_tile_init()` calls `SFPU_INIT_KERNEL_CALL(acosh, ckernel::sfpu::_init_inverse_hyperbolic_, APPROX)` which initializes log-related constants

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces
- **Operation invocation**: `_calculate_acosh_<APPROX, 8>()` called per face
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face)

### Key Observations for cosh Implementation
- acosh is in the same trig family (`SFPU_OP_TRIG_FAMILY_INCLUDE`) and same `trigonometry.h` files
- acosh uses `SFPU_TWO_PARAM_KERNEL` (2 template params: APPROX, ITERATIONS) while cosh uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` (3 template params: APPROX, DST_ACCUM_MODE, ITERATIONS)
- acosh has a different init function (`_init_inverse_hyperbolic_`) than cosh (`init_hyperbolic_trig`)
- acosh is more complex (uses log, sqrt, multiply) while cosh is simpler (just two exp calls and addition)
- This demonstrates that even within the same trig family, different operations can use different macro variants and init functions

### SFPU Instructions Used
- Log approximation primitives (from shared SFPU library)
- Square root via Newton-Raphson iteration
- `SFPMAD`: Float multiply-add for polynomial evaluation and arithmetic
- `SFPLOAD` / `SFPSTORE`: DEST register access
- `v_if`/`v_endif`: Conditional branches

### SFPU Register Usage
- `dst_reg[0]`: Input read and output write
- `dst_reg++`: Standard advancement
- `vConstFloatPrgm0/1`: Constants for log computation (set by `_init_inverse_hyperbolic_`)
- LREGs: Intermediate polynomial terms

### Address Mode Configuration
Standard unary SFPU address mode (ADDR_MOD_2 on Wormhole).
