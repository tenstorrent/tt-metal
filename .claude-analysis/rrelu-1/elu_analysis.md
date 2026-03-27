## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the ELU (Exponential Linear Unit) operation.

### Unary Dispatch Summary
- **UnaryOpType**: `ELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `elu_tile_init(); elu_tile({idst}, {alpha_as_hex}u);`

The `alpha` parameter (the ELU slope) is passed from the Python/C++ API as a `float`, bit-cast to `uint32_t` via `std::bit_cast<uint32_t>(param0)`, and injected into the SFPU_OP_CHAIN_0 as a hex literal.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(ELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` (the switch has only a `default` case) |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized approximation control) | `get_op_init_and_func()` returns `elu_tile_init()` and `elu_tile({idst}, {alpha_hex}u)` -- no template parameter for approximation mode. The `elu_tile()` function uses the compile-time `APPROX` constant directly. |
| Effective SFPU path | `APPROXIMATION_MODE=false` in the hw/ckernels version. The kernel calls `_sfpu_exp_21f_bf16_<true>(v)` with a hardcoded `true` template argument (to avoid premature bf16 rounding of intermediate results), so the exp sub-function always uses the "21f" algorithm regardless of APPROX. | In the hw/ckernels version, `calculate_elu` does not branch on `APPROXIMATION_MODE` at all -- it always uses `_sfpu_exp_21f_bf16_`. In the tt_llk version, `_calculate_elu_` calls `_calculate_exponential_piecewise_<APPROXIMATION_MODE, false, false>()`. With `APPROXIMATION_MODE=false`, this takes the non-approximate path: `_sfpu_exp_(setsgn(in, 0))` followed by conditional reciprocal for negative inputs. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/elu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the API header directly invokes the macro `SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM` which calls `_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h` (hw/ckernels version) and `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h` (tt_llk version) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

### Call Chain
1. The compute kernel (`eltwise_sfpu.cpp`) expands `SFPU_OP_CHAIN_0` to `elu_tile(0, {alpha_hex}u)`.
2. `elu_tile()` (in `elu.h`) expands via the `MATH()` macro (active only on the TRISC_MATH thread) to `SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(calculate_elu, RC, APPROX, DST_ACCUM_MODE, idst, param0)`.
3. The macro expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_elu<APPROX, DST_ACCUM_MODE>, idst, (int)VectorMode::RC, param0)`.
4. `_llk_math_eltwise_unary_sfpu_params_` (in the LLK params dispatch) sets the DEST write address, stalls for SFPU readiness, then loops over 4 faces (for `VectorMode::RC`), calling `calculate_elu<APPROX, DST_ACCUM_MODE>(param0)` once per face. Between faces, `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` x2 (Blackhole) advances the DEST pointer by one face (16 physical rows).
5. Inside `calculate_elu`, the function loads each element from `dst_reg[0]`, conditionally computes `alpha * (exp(x) - 1)` for negative inputs using `_sfpu_exp_21f_bf16_`, writes the result back, and advances `dst_reg++` per iteration.
6. `elu_tile_init()` expands to `SFPU_UNARY_KERNEL_INIT(elu, APPROX)` which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::elu, APPROX>()`. This configures the SFPU config register, sets `ADDR_MOD_7` (dest increment = 0), and resets counters.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (the full 32x32 = 1024 elements).
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_elu<APPROX, DST_ACCUM_MODE>(slope)` once per face. Each invocation of `calculate_elu` internally loops 8 iterations (ITERATIONS=8 default), processing one face's worth of data (8 iterations x 32 elements = 256 elements per face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, `ADDR_MOD_7` is the active address mode (base offset applies: `set_addr_mod_base()` shifts to use ADDR_MOD 4-7), configured with `.dest = {.incr = 0}`. On Blackhole, `ADDR_MOD_7` is also used (configured identically with `.dest = {.incr = 0}`), but Blackhole does not use `set_addr_mod_base()`; instead face advancement uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice. In both cases, within-face progression is handled by the SFPI `dst_reg++` instruction (advancing 1 sfpi row = 2 physical DEST rows = 32 elements per iteration).

### Annotated SFPU Kernel Source

There are two versions of the ELU SFPU kernel: the `hw/ckernels` version (used by the TTNN compute kernel at compile time) and the `tt_llk` version (the upstream LLK implementation). Both are documented below as they differ in their exponential computation strategy.

#### hw/ckernels Version (used by TTNN)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_elu.h
// (Blackhole version is identical)

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_elu(uint slope) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat s = Converter::as_float(slope); // Reinterpret uint32_t bit pattern as float, then broadcast to vFloat
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position
        v_if(v < 0.0f) { // SFPSETCC + PUSHC: predicate on sign bit; only negative lanes proceed
            sfpi::vFloat v_exp =
                _sfpu_exp_21f_bf16_<true>(v) - sfpi::vConst1; // exp(x) - 1.0; template arg 'true' avoids premature bf16 rounding
                                                               // in the exp sub-function (rounding deferred to end of operation)
            sfpi::vFloat result = s * v_exp; // SFPMAD: alpha * (exp(x) - 1)
            if constexpr (!is_fp32_dest_acc_en) { // When DEST is bfloat16:
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFP_STOCH_RND: explicit round-to-nearest-even to bf16
            }
            sfpi::dst_reg[0] = result; // SFPSTORE: write result back (only for negative-input lanes)
        }
        v_endif; // POPC: restore condition code state; positive inputs pass through unchanged
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

#### tt_llk Version (upstream LLK)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h
// (Blackhole version is identical)

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_elu_(std::uint32_t slope) // APPROXIMATION_MODE=false, ITERATIONS=8
{
    const bool SCALE_EN                       = false; // Elu does not use scale.
    const bool SKIP_POSITIVE_CHECK            = false; // Elu does not skip positive check.
    const std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B; // 1.0 in bf16 = no scaling

    sfpi::vFloat s = Converter::as_float(slope); // Reinterpret slope bits as float -> vFloat
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load from DEST

        v_if (v < 0.0f) // SFPSETCC + PUSHC: predicate on negative values
        {
            // With APPROXIMATION_MODE=false, this calls _sfpu_exp_(|x|) then conditional reciprocal
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(v, exp_base_scale_factor);
            v                  = s * (v_exp - 1.0f); // alpha * (exp(x) - 1)
        }
        v_endif; // POPC: restore CC state

        sfpi::dst_reg[0] = v; // SFPSTORE: write result (always, even for positive inputs -- v is unchanged for those)

        sfpi::dst_reg++; // Advance 1 sfpi row
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_elu_()
{
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000; // 1.0f in IEEE 754
    const bool FAST_APPROX                    = false; // Elu does not use fast approximation.
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
}

} // namespace ckernel::sfpu
```

#### Key Helper: `_sfpu_exp_21f_bf16_` (called by hw/ckernels version)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

// Helper: convert float to integer using properties of IEEE 754 encoding
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);   // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);  // SFPEXMAN: extract mantissa with implicit bit (8-bit precision)
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp)); // SFPSHFT: shift mantissa by exponent
    return man;
}

// exp_21f algorithm: based on Moroz et al. 2022
// "Simple Multiple Precision Algorithms for Exponential Functions"
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) // is_fp32_dest_acc_en=true (hardcoded in ELU call)
{
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f); // SFPMAD: x/ln2 + 127 (bias for IEEE exponent)

    // Clamp to [0, 255] to avoid overflow/underflow in intermediate values
    sfpi::vFloat threshold_low  = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);  // SFPSETCC + conditional swap: ensures xlog2 >= 0
    sfpi::vec_min_max(xlog2, threshold_high);  // SFPSETCC + conditional swap: ensures xlog2 <= 255

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2); // Convert to fixed-point integer

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP: extract exponent (= 2^(integer part))
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN: extract mantissa (= fractional part, [0,1])

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 -> float

    // 2nd degree polynomial refinement of 2^(fractional_part) via Horner's method -> chain of SFPMAD
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: 2^(integer) * 2^(fractional) by setting exponent field
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: inject exponent into mantissa

    if constexpr (!is_fp32_dest_acc_en) // is_fp32_dest_acc_en=true for ELU, so this branch is SKIPPED
    {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}
```

### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `SFPLOAD` (`dst_reg[0]` read) | Load 32 elements from the current DEST position into an LREG for SFPU processing |
| `SFPSTORE` (`dst_reg[0] = val` write) | Store 32 elements from an LREG back to the current DEST position |
| `SFPMAD` (vFloat arithmetic: `*`, `+`, `-`) | Fused multiply-add; used for: `val * ONE_LN2 + 127.f`, `s * v_exp`, `v_exp - vConst1`, and the Horner polynomial evaluation chain |
| `SFPLOADI` (constant loading) | Load immediate values into LREGs for constants like `0.0f`, `255.f`, `1.0f`, polynomial coefficients |
| `SFPSETCC` (`v_if(v < 0.0f)`) | Set condition codes based on comparison; used to predicate ELU computation on negative inputs only |
| `PUSHC` / `POPC` (`v_if` / `v_endif`) | Push/pop condition code state for nested predication |
| `SFPEXEXP` (`exexp`, `exexp_nodebias`) | Extract the biased exponent field from a floating-point value; used in the exp_21f algorithm to separate integer and fractional parts |
| `SFPEXMAN` (`exman8`, `exman9`) | Extract the mantissa field from a floating-point value (with 8-bit or 9-bit precision); used to isolate the fractional part in exp_21f |
| `SFPSHFT` (`shft`) | Bitwise shift; used to align mantissa by exponent amount in the float-to-int conversion |
| `SFPSETEXP` (`setexp`) | Set the exponent field of a floating-point value; used to recombine 2^(integer) * 2^(fractional) in exp_21f |
| `SFPCAST` (`int32_to_float`) | Convert integer to floating-point; used to convert the fractional part back to float for polynomial evaluation |
| `SFP_STOCH_RND` (`float_to_fp16b`) | Convert float32 to bfloat16 using round-to-nearest-even; used for explicit bf16 rounding when DEST is not fp32 accumulation mode |
| `vec_min_max` | Conditional min/max via SFPSETCC + conditional swap; used for clamping `xlog2` to [0, 255] |
| `SFPENCC` (implicit in `v_if` / `v_endif`) | Enable/disable writes based on condition codes; ensures only negative-input lanes are modified |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` (dst_reg[0]) | Primary working register: loads input from DEST, stores result back to DEST |
| `LREG1-LREG3` | Intermediate computation registers used by the compiler for vFloat temporaries (`v`, `s`, `v_exp`, `result`, `xlog2`, `frac`, etc.) |
| `vConst1` (LREG constant) | The constant `1.0f`, subtracted from exp(x) to compute `exp(x) - 1` |
| `vConstFloatPrgm0-2` | Programmable constants used by `_calculate_exponential_approx_` in the tt_llk path (only relevant when `APPROXIMATION_MODE=true`): `Prgm0` = 1/ln2, `Prgm1` = conversion constant, `Prgm2` = exponent adjustment |
| DEST registers | Input tile data resides in DEST. The SFPU reads from and writes to DEST rows in-place. Each `dst_reg[0]` access touches 32 elements (2 physical rows x 16 elements/row). |

### Address Mode Configuration

The address mode for ELU is configured during `elu_tile_init()` -> `_llk_math_eltwise_unary_sfpu_init_<SfpuType::elu>()` -> `eltwise_unary_sfpu_configure_addrmod<SfpuType::elu>()`.

**Wormhole B0**:
- `ADDR_MOD_7` is configured with: `.srca = {.incr = 0}`, `.srcb = {.incr = 0}`, `.dest = {.incr = 0}` (all zero increments).
- `set_addr_mod_base()` is called before SFPU execution, shifting the active address mode set to 4-7. The SFPI compiler emits SFPLOAD/SFPSTORE with `ADDR_MOD_3` encoding, which maps to `ADDR_MOD_7` (3 + base offset 4) at runtime.
- The zero dest increment means hardware does not auto-increment DEST addresses between SFPLOAD/SFPSTORE pairs. Instead, address progression within a face is handled entirely by the SFPI `dst_reg++` instruction, and between faces by `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice (advancing 16 physical rows = 1 face).

**Blackhole**:
- `ADDR_MOD_7` is configured identically: `.srca = {.incr = 0}`, `.srcb = {.incr = 0}`, `.dest = {.incr = 0}`.
- Blackhole does NOT call `set_addr_mod_base()` or `clear_addr_mod_base()`.
- Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice (advancing 16 physical rows = 1 face).
- Within a face, `dst_reg++` handles per-iteration address advancement, same as Wormhole.

The ELU operation does not fall into any special address mode case (those are reserved for `topk_local_sort`, `typecast`, and `unary_max/min` variants).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPU v_if / v_endif conditional execution work in SFPI? What instructions does it generate and how does it manage condition codes?"
   **Reason**: Needed to understand the predicated execution model used in the ELU kernel for the `v_if(v < 0.0f)` conditional.
   **Key Findings**: `v_if`/`v_endif` implement predicated vector conditional execution using PUSHC/POPC for CC stack management and SFPSETCC/SFPENCC for per-lane write predication. Both branches execute as straight-line code but only the active branch's vector writes take effect.

2. **Query**: "What is the Converter::as_float utility in SFPU kernels? How does it convert a uint32_t to a vFloat?"
   **Reason**: Needed to understand how the `slope` parameter (passed as uint32_t) is converted to a vFloat for SFPU computation.
   **Key Findings**: `Converter::as_float` performs a bitwise reinterpretation (not numeric conversion) of a uint32_t to float, treating the bit pattern as IEEE 754 binary representation. The resulting float is then implicitly broadcast to a vFloat vector.

3. **Query**: "In the SFPI programming model, what SFPU instructions do the following SFPI intrinsics map to?"
   **Reason**: Attempted to map SFPI intrinsics to specific TTI instructions for the Instructions Used table.
   **Key Findings**: DeepWiki could not provide this information as the SFPI-to-TTI mapping is handled by the external SFPI compiler toolchain, not exposed in the tt-metal repository. Instruction mappings were determined from the hardware model reference and instruction naming conventions.

### Confluence References
No Confluence references were consulted for this analysis.

### Glean References
No Glean references were consulted for this analysis.
