# exp -- SFPU Kernel Analysis

## SFPU Kernel Implementation

### Unary Dispatch Summary
- **UnaryOpType**: `EXP`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `exp_tile(0)` (non-parameterized path)
- **Macro group**: `SFPU_OP_EXP_INCLUDE`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | false | `get_op_approx_mode(EXP)` falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | Parameterized: `(uint32_t)param0` controls fast_and_approximate_mode. Non-parameterized: uses default template args (false) | `get_op_init_and_func()`: parameterized returns `exp_tile_init<{param0}u>()` / `exp_tile<{param0}u>(0)`, non-parameterized returns `exp_tile_init()` / `exp_tile(0)` |
| Effective SFPU path | Two paths: (1) APPROXIMATION_MODE=true uses `_calculate_exponential_` with fast lookup; (2) APPROXIMATION_MODE=false uses `_sfpu_exp_accurate_` per-element | `if constexpr (APPROXIMATION_MODE)` branch in `calculate_exponential` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h` (or equivalent -- exp_tile declared as compute API) |
| **LLK Dispatch** | Standard LLK params dispatch via `_llk_math_eltwise_unary_sfpu_params_` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` (metal layer) + `sfpu/ckernel_sfpu_exp.h` (tt_llk shared primitives) |

### Call Chain

1. `exp_tile(idst)` calls the LLK params dispatch with `calculate_exponential<APPROX, FAST_APPROX, is_fp32_dest_acc_en, ...>`
2. The dispatch iterates over faces (VectorMode::RC), calling the calculate function 4 times
3. In the non-approximate path (`APPROXIMATION_MODE=false`): loops over ITERATIONS, calling `_sfpu_exp_accurate_<is_fp32_dest_acc_en>(val)` per element
4. `exp_tile_init()` calls `exp_init<APPROX, FAST_APPROX>()` -> `_init_exponential_<APPROX, FAST_APPROX, scale, CLAMP_NEGATIVE>()`

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces
- **Operation invocation**: Dispatch calls `calculate_exponential<APPROX, FAST_APPROX, DST_ACCUM_MODE, SCALE_EN, 8, ...>()` per face
- **DEST address progression**: Standard DEST progression (ITERATIONS=8, dst_reg++, SETRWC between faces)

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // Fast approximate path -- uses _calculate_exponential_ from tt_llk shared primitives
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Accurate path -- per-element exponential
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);
            }
            sfpi::vFloat result = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, uint32_t scale = 0x3F800000, bool CLAMP_NEGATIVE = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale, CLAMP_NEGATIVE>();
}
```

### Key Shared Primitives (from tt_llk `sfpu/ckernel_sfpu_exp.h`)

The following functions are defined in the tt_llk submodule and are used by many operations:

- `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)`: 21-term bfloat16 exponential approximation. Used by cosh, sinh, expm1, and other operations that need exp as a building block.
- `_sfpu_exp_accurate_<is_fp32_dest_acc_en>(v)`: Higher-accuracy exponential for the non-approximate path.
- `_init_exponential_<APPROX, FAST_APPROX, scale, CLAMP_NEGATIVE>()`: Initializes SFPU state for exponential computation (programmable constants, address modes).
- `_calculate_exponential_<APPROX, SCALE_EN, ITERATIONS, FAST_APPROX, ...>()`: Fast approximate exponential using SFPU lookup/replay.

### SFPU Instructions Used
- `_sfpu_exp_accurate_` / `_sfpu_exp_21f_bf16_`: Internal exp primitives (use SFPMAD chains for polynomial evaluation, SFPEXEXP for exponent extraction)
- `SFPMAD`: Float multiply-add (used in polynomial evaluation and scaling)
- `SFPLOAD` / `SFPSTORE`: DEST register access
- `SFPEXEXP`: Exponent extraction (used internally by exp primitives)
- `SFPLOADI`: Load immediate float values into LREGs

### SFPU Register Usage
- `dst_reg[0]`: Input read and output write
- `dst_reg++`: Advances sfpi address by 1
- LREGs: Used internally by exp primitives for intermediate polynomial terms
- `vConstFloatPrgm0/1/2`: Programmable constants set by `_init_exponential_` (ln2 reciprocal, etc.)

### Address Mode Configuration
Standard unary SFPU address mode (ADDR_MOD_2 on Wormhole). `_init_exponential_` configures the address mode for the exponential computation path.
