## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `CELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `celu_tile(0, {alpha_bits}u, {alpha_recip_bits}u)` where `alpha_bits = std::bit_cast<uint32_t>(param0)` and `alpha_recip_bits = std::bit_cast<uint32_t>(1.0f / param0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` (the switch has only a `default` case) |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized approximation) | `get_op_init_and_func()` returns `celu_tile_init()` / `celu_tile(idst, alpha, alpha_recip)` -- no template approximation parameter in the SFPU_OP_CHAIN |
| Effective SFPU path | `APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en=false` (default from `DST_ACCUM_MODE`). The kernel always takes the `v < 0` branch for negative values and calls `_sfpu_exp_21f_bf16_<true>` with hardcoded `true` for the template parameter (avoiding intermediate rounding). | `ckernel_sfpu_celu.h` line 28 -- the `<true>` is hardcoded in the call to `_sfpu_exp_21f_bf16_`, not derived from `APPROXIMATION_MODE` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_activations.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_celu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`celu_tile(idst, alpha, alpha_recip)`** (API header `activations.h`, line 71) wraps the call in the `MATH()` macro, invoking `llk_math_eltwise_unary_sfpu_celu<APPROX, DST_ACCUM_MODE>(idst, alpha, alpha_recip)`.
2. **`llk_math_eltwise_unary_sfpu_celu`** (LLK dispatch `llk_math_eltwise_unary_sfpu_activations.h`, line 48) forwards to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` with a lambda that calls `calculate_celu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>(alpha, alpha_recip)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (params dispatch, line 14) handles DEST addressing, face iteration (VectorMode::RC = all 4 faces), and SETRWC between faces, then invokes the lambda for each face.
4. **`calculate_celu`** (core SFPU implementation `ckernel_sfpu_celu.h`, line 14) executes the per-face SFPU microcode: loads elements from DEST, conditionally computes `alpha * (exp(x/alpha) - 1)` for negative values, and writes results back.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (32x32 = 4 faces of 16x16).
- **Operation invocation**: For each of the 4 faces, the lambda `[](uint32_t alpha, uint32_t alpha_recip) { calculate_celu<...>(alpha, alpha_recip); }` is called once. Each invocation runs `ITERATIONS=8` loop iterations within the face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The init configures `ADDR_MOD_7` with `dest.incr = 0` (the SFPI `dst_reg++` handles row advancement internally via stride-2 addressing). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (advancing by 16 physical DEST rows = one face).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`, `Converter::as_float`) -- Style A applies.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_celu.h
// (Wormhole B0 and Blackhole implementations are identical)

namespace ckernel::sfpu {

// CELU: alpha * (exp(x / alpha) - 1)
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_celu(uint32_t param0, uint32_t param1) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    // All params are in FP16_B format
    // param0 = alpha
    // param1 = alpha_recip
    sfpi::vFloat alpha = Converter::as_float(param0);       // SFPLOADI: load alpha constant into LREG
    sfpi::vFloat alpha_recip = Converter::as_float(param1); // SFPLOADI: load 1/alpha constant into LREG
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        v_if(v < sfpi::vConst0) { // SFPSETCC: set CC based on v < 0, guards all operations inside
            // Compute exp(x / alpha)
            sfpi::vFloat exp_val =
                _sfpu_exp_21f_bf16_<true>(v * alpha_recip); // SFPMAD: v * alpha_recip; then exp_21f subroutine (is_fp32_dest_acc_en=true to avoid intermediate rounding)

            sfpi::vFloat result = alpha * (exp_val - sfpi::vConst1); // SFPMAD: exp_val - 1.0; SFPMAD: alpha * (exp_val - 1.0)
            if constexpr (!is_fp32_dest_acc_en) { // true for default path (is_fp32_dest_acc_en=false)
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFPSTOCHRND: round FP32 to BF16 (round-to-nearest-even, mode 0)
            }
            sfpi::dst_reg[0] = result; // SFPSTORE: write result back to DEST (only for lanes where v < 0)
        }
        v_endif; // restore CC state -- positive values pass through unchanged
        sfpi::dst_reg++; // advance to next sfpi row (2 physical DEST rows, 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (via `dst_reg[0]` read) | Load 32 elements (2 physical DEST rows) from the current DEST address into an LREG for SFPU processing |
| `SFPLOADI` (via `Converter::as_float`) | Load an immediate constant (alpha, alpha_recip) into an LREG from a 32-bit parameter encoded as FP16_B |
| `SFPMAD` (via `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat`) | Fused multiply-add: used for `v * alpha_recip`, `exp_val - vConst1`, and `alpha * (exp_val - 1)`. All vFloat arithmetic maps to SFPMAD (a * 1.0 + b for addition, a * b + 0 for multiplication) |
| `SFPSETCC` (via `v_if(v < sfpi::vConst0)`) | Set condition codes based on comparison `v < 0`; enables predicated execution for negative-value branch |
| `SFPSTORE` (via `dst_reg[0]` write) | Store 32 elements from LREG back to the current DEST address, respecting CC predication |
| `SFPSTOCHRND` (via `float_to_fp16b`) | Convert FP32 result to BF16 format using round-to-nearest-even; only used when `is_fp32_dest_acc_en=false` |
| `_sfpu_exp_21f_bf16_` (subroutine) | Computes exp(x) using the exp_21f algorithm from Moroz et al. 2022. Internally generates multiple SFPMAD, SFPEXEXP, SFPSETEXP, SFPSETMAN, SFPDIVP2, and SFPIADD instructions. **Note**: The source definition of this function was not found in the current checkout -- it is called by multiple SFPU kernels (sigmoid, elu, celu, selu, cosh, sinh, softplus, exp2, expm1) but its definition does not exist in the source tree as checked out. Based on usage context and the reference in ckernel_sfpu_binary_pow.h, it implements the exp_21f algorithm adapted to BF16 precision. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `dst_reg[0]` (DEST rows via SFPI stride-2) | Source and destination for tile element data. Each access reads/writes 32 elements (2 physical DEST rows x 16 elements/row) |
| `LREG` (local registers, implicit via vFloat variables) | `alpha`: holds the alpha parameter constant; `alpha_recip`: holds 1/alpha; `v`: holds current input element; `exp_val`: holds exp(x/alpha) result; `result`: holds final CELU output. The SFPU has 4 general-purpose LREGs (LREG0-3); the compiler allocates these automatically for SFPI code |
| `vConst0` (SFPU constant register) | Provides the value 0.0f for the comparison `v < 0` |
| `vConst1` (SFPU constant register) | Provides the value 1.0f for the subtraction `exp_val - 1` |
| Condition Code (CC) register | Set by `v_if(v < vConst0)` to predicate the negative-value computation. Elements where `v >= 0` are left unchanged (identity pass-through) |

### Address Mode Configuration

The CELU init function (`llk_math_eltwise_unary_sfpu_celu_init`) calls the generic `_llk_math_eltwise_unary_sfpu_init_<SfpuType::celu>()` which configures:

- **`ADDR_MOD_7`**: `{ srca.incr = 0, srcb.incr = 0, dest.incr = 0 }` -- configured in `eltwise_unary_sfpu_configure_addrmod()` in `llk_math_eltwise_unary_sfpu.h` (line 28-33). This is the same for all hardware generations (Wormhole B0, Blackhole).
- **Base selection**: `set_addr_mod_base()` sets the base to 1, selecting ADDR_MOD range 4-7 (so ADDR_MOD_7 maps to physical ADDR_MOD slot 7). This avoids conflicts with A2D operations which use ADDR_MOD_0 and ADDR_MOD_2.
- **Why `dest.incr = 0`**: The SFPI `dst_reg++` instruction handles DEST row advancement internally through the stride-2 addressing mechanism, so the hardware addr_mod auto-increment is not needed. Between faces, explicit `SETRWC` instructions advance the DEST pointer by 16 physical rows (one face).
- The CELU-specific init (`celu_tile_init()`) does not configure any additional programmable constants or address modes beyond the generic unary SFPU init. No custom `_init_*` function is called (unlike exp which configures ln2_recip constants).

## External Knowledge Sources
### DeepWiki Queries
No DeepWiki queries were needed for this analysis. The CELU kernel is straightforward SFPI code with well-documented abstractions.

### Confluence References
No Confluence references were needed. The kernel does not use complex CC manipulation or raw TTI instructions that would require ISA-level documentation.

### Glean References
No Glean references were needed.

### Notes on Missing Source Definition
The function `_sfpu_exp_21f_bf16_<bool is_fp32_dest_acc_en>(vFloat x)` is called by `calculate_celu` and at least 9 other SFPU kernel files (sigmoid, elu, selu, cosh, sinh, softplus, exp2, expm1, trigonometry) but its definition was **not found** anywhere in the source tree of this checkout (commit `57b5fc5fa6`). All files that call it include `sfpu/ckernel_sfpu_exp.h` (from the `tt_llk` third-party submodule), and `ckernel.h`, but neither file nor any file in their transitive include chains defines this function. The function likely existed in a prior version or is expected to be provided by a header not present in this checkout. Similarly, `_sfpu_exp_accurate_<bool is_fp32_dest_acc_en>(vFloat x)` and `_sfpu_exp_fp32_accurate_(vFloat x)` are referenced without definitions. Based on usage context and the algorithm description in `ckernel_sfpu_binary_pow.h`, `_sfpu_exp_21f_bf16_` implements the `exp_21f` algorithm from Moroz et al. 2022 ("Simple Multiple Precision Algorithms for Exponential Functions"), adapted for BF16 precision on the SFPU.
