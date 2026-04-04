# cos -- SFPU Kernel Analysis

## SFPU Kernel Implementation

### Unary Dispatch Summary
- **UnaryOpType**: `COS`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cos_tile(0)`
- **Macro group**: `SFPU_OP_TRIG_FAMILY_INCLUDE`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | false | `get_op_approx_mode(COS)` falls through to `default: return false` |
| Template parameter | none (default) | `get_op_init_and_func()` returns `{"cos_tile_init();", "cos_tile(0);"}` -- no param |
| Effective SFPU path | Polynomial approximation with Cody-Waite range reduction | Single path, APPROXIMATION_MODE only affects coefficient precision |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (via `SFPU_THREE_PARAM_KERNEL_FP32_FIRST`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` |

### Call Chain

1. `cos_tile(idst)` calls `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosine, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)`
2. Expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_cosine<APPROX, DST_ACCUM_MODE, 8>, idst, (int)VectorMode::RC)`
3. `cos_tile_init()` calls `SFPU_INIT_KERNEL_CALL(cosine, ckernel::sfpu::cosine_init, APPROX)` -> `cosine_init<false>()` which sets up Cody-Waite reduction constants

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces
- **Operation invocation**: `calculate_cosine<APPROX, is_fp32_dest_acc_en, 8>()` called per face, ITERATIONS=8
- **DEST address progression**: Standard DEST progression

### Key Observations for cosh Implementation
- cos is a sibling in the same trig family (`SFPU_OP_TRIG_FAMILY_INCLUDE`), so it shares the same include macro
- cos and cosh both live in `ckernel_sfpu_trigonometry.h` and `trigonometry.h`
- cos uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` -- the same macro that cosh and sinh use
- cos uses `cosine_init` while cosh uses `init_hyperbolic_trig` -- different init functions because cos needs Cody-Waite constants while cosh needs exponential constants
- The compute API registration pattern (`{op}_tile_init()` / `{op}_tile(idst)`) is identical

### SFPU Instructions Used
- Polynomial evaluation via `sfpu/ckernel_sfpu_polyval.h` for minimax polynomial approximation of cosine
- `SFPMAD`: Used extensively in Horner's method polynomial evaluation
- `SFPLOAD` / `SFPSTORE`: DEST register access
- Range reduction using Cody-Waite method with PI/2 constants
- `v_if`/`v_endif`: Conditional branches for quadrant selection

### SFPU Register Usage
- `dst_reg[0]`: Input read and output write
- `vConstFloatPrgm0/1/2`: Cody-Waite reduction constants for PI (set by `cosine_init`)
- LREGs: Polynomial coefficients and intermediate values

### Address Mode Configuration
Standard unary SFPU address mode (ADDR_MOD_2 on Wormhole).
