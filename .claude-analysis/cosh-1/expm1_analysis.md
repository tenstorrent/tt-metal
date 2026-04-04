# expm1 -- SFPU Kernel Analysis

## SFPU Kernel Implementation

### Unary Dispatch Summary
- **UnaryOpType**: `EXPM1`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `expm1_tile(0)`
- **Macro group**: `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | false | `get_op_approx_mode(EXPM1)` falls through to `default: return false` |
| Template parameter | none (default) | `get_op_init_and_func()` returns `{"expm1_tile_init();", "expm1_tile(0);"}` -- no param |
| Effective SFPU path | Non-approximate exponential via `_sfpu_exp_21f_bf16_` with -1 subtraction | Single code path |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/expm1.h` |
| **LLK Dispatch** | Standard LLK params dispatch |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h` |

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_expm1.h

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_expm1() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v) - 1.0f;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

### Key Observations for cosh Implementation
- expm1 follows the exact same pattern as cosh/sinh: load from dst_reg, call `_sfpu_exp_21f_bf16_`, apply simple arithmetic, store back
- The init function for expm1 also calls `_init_exponential_` since it depends on the same exp primitive
- This confirms that any operation composing `_sfpu_exp_21f_bf16_` follows the same boilerplate pattern

### SFPU Instructions Used
- `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)`: 21-term bfloat16 exponential
- `SFPMAD`: Subtraction of 1.0f (a * 1.0 + (-1.0))
- `SFPLOAD` / `SFPSTORE`: DEST register access

### SFPU Register Usage
- Standard: `dst_reg[0]` for input/output, `dst_reg++` for advancement
- Same programmable constants as exp (set by `_init_exponential_`)

### Address Mode Configuration
Standard unary SFPU address mode (ADDR_MOD_2 on Wormhole).
