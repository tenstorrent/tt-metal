<<<<<<< HEAD
## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `COSH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `cosh_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(COSH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized version) | `get_op_init_and_func_default()` returns `"cosh_tile_init();"` and `"cosh_tile({idst});"` -- no template arguments in the macro expansion |
| Effective SFPU path | `APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en=DST_ACCUM_MODE`, `ITERATIONS=8`. Since `cosh_init` calls `_init_exponential_<APPROX, false, kCONST_1_FP16B>()` with `FAST_APPROX=false`, the `if constexpr (FAST_APPROX && APPROXIMATION_MODE && ...)` branch is NOT taken -- no LOADMACRO fast-path is configured. The `_sfpu_exp_21f_bf16_` polynomial-based algorithm is always used. | `ckernel_sfpu_cosh.h` calls `_sfpu_exp_21f_bf16_` unconditionally; `_init_exponential_` FAST_APPROX branch is `false` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (function `_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h` (BH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain
1. **`cosh_tile(idst)`** (API Header, `cosh.h`) -- expands via `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` macro to call `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_cosh<APPROX, DST_ACCUM_MODE, 8>, idst, (int)VectorMode::RC)`.
2. **`_llk_math_eltwise_unary_sfpu_params_<APPROX>(...)`** (LLK Dispatch, `llk_math_eltwise_unary_sfpu_params.h`) -- sets DST write address, stalls until SFPU is ready, then invokes the functor `calculate_cosh<...>()` once per face (4 faces for `VectorMode::RC`), advancing the DEST face address via `TTI_SETRWC` (WH) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (BH) between faces.
3. **`calculate_cosh<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()`** (Core SFPU, `ckernel_sfpu_cosh.h`) -- iterates 8 times (one per sfpi row in a face), loading from `dst_reg[0]`, computing `(exp(v) + exp(-v)) * 0.5`, storing back, and advancing `dst_reg++`.
4. **`_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)`** (Helper, `ckernel_sfpu_exp.h`) -- called twice per iteration (once for `v`, once for `-v`) to compute the exponential using the Moroz et al. 2022 polynomial approximation algorithm (`exp_21f`).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (Face0 through Face3).
- **Operation invocation**: The params dispatch calls `calculate_cosh()` 4 times in an unrolled loop (`for (int face = 0; face < 4; face++)`). Each invocation processes one face (ITERATIONS=8 sfpi rows). Between faces, DEST address is advanced by one face stride (16 physical DEST rows = 8 sfpi rows), using `TTI_SETRWC` on Wormhole or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (which calls `math::inc_dst_addr<8>()` twice) on Blackhole.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). Address mode is `ADDR_MOD_7` on both Wormhole and Blackhole (configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::cosh>()` which hits the default path setting `ADDR_MOD_7` with `.srca.incr=0, .srcb.incr=0, .dest.incr=0`).

### Annotated SFPU Kernel Source

The cosh kernel uses SFPI abstractions (`vFloat`, `dst_reg`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h
// (Identical on Blackhole: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h)

// cosh(x) = (exp(x) + exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_cosh() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=DST_ACCUM_MODE, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position
        sfpi::vFloat result =
            (_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v) + _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)) * 0.5f;
            // Two exp evaluations (see helper below), then SFPMAD for addition, SFPMAD for *0.5
        sfpi::dst_reg[0] = result; // SFPSTORE: write result back to current DEST position
        sfpi::dst_reg++; // advance by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
void cosh_init() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
    // FAST_APPROX=false, scale=0x3F80 (1.0f in FP16B)
    // With FAST_APPROX=false, the if-constexpr branch in _init_exponential_ is NOT taken,
    // so no LOADMACRO constants or macro instructions are programmed.
    // This is effectively a no-op init for the exp_21f path.
}
```

The core exponential helper function `_sfpu_exp_21f_bf16_` and its dependency `_float_to_int32_for_exp_21f_` are defined in the shared LLK exp library:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val); // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val); // SFPEXMAN(PAD8): extract mantissa with implicit bit
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));
        // SFPSHFT: logical left-shift mantissa by exponent value; converts float to scaled int32
    return man;
}

// Implementation notes, see the original file for more details
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val)
{
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f); // SFPMAD: val * 1/ln2, then SFPMAD: + 127.0

    // Clamp xlog2 to [0, 255] to prevent overflow in intermediate values
    sfpi::vFloat threshold_low  = 0.f; // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f); // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2); // SFPSWAP(VEC_MIN_MAX): clamp lower bound
    sfpi::vec_min_max(xlog2, threshold_high); // SFPSWAP(VEC_MIN_MAX): clamp upper bound

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2); // See helper above

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));
        // SFPEXEXP(NODEBIAS): extract exponent without bias subtraction (= integer part of val/ln2)
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));
        // SFPEXMAN(PAD9): extract mantissa (= fractional part, in [0; 1])

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);
        // SFPCAST(INT32_TO_FP32_RNE): convert integer mantissa bits to float

    // 2nd degree polynomial adjustment: 2^(x_f) approximation on fractional part via Horner's method
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
        // Expands to: 1.0017248 + frac * (7.84e-08 + frac * 4.79e-15)
        // Chain of SFPMAD instructions (Horner's method)

    // Recombine exponent and mantissa: 2^(x_i) * 2^(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: set exponent field

    if constexpr (!is_fp32_dest_acc_en)
    {
        // Round to bfloat16 to avoid truncation artifacts when SFPSTORE writes to bf16 DEST
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
            // SFP_STOCH_RND(FP32_TO_FP16B, RNE): round-to-nearest-even conversion
    }

    return y;
}
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the SFPI abstractions used in `calculate_cosh` and its `_sfpu_exp_21f_bf16_` helper:

| Instruction | SFPI Abstraction | Purpose |
|-------------|-----------------|---------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from current DEST position into LREG |
| **SFPSTORE** | `dst_reg[0] = result` (write) | Store result from LREG back to current DEST position |
| **SFPLOADI** | `vFloat(0.f)`, `vFloat(255.f)`, float constants | Load 16-bit immediate values into LREGs for constants |
| **SFPMAD** | `val * ONE_LN2`, `+ 127.f`, `+ exp(-v)`, `* 0.5f`, Horner polynomial | Fused multiply-add: all float arithmetic (multiply, add, subtract) |
| **SFPSWAP** | `vec_min_max(a, b)` | Vector min/max operation for clamping xlog2 to [0, 255] |
| **SFPEXEXP** | `exexp(val)`, `exexp_nodebias(z)` | Extract exponent field from float (with or without bias removal) |
| **SFPEXMAN** | `exman8(val)`, `exman9(z)` | Extract mantissa field with implicit bit (8-bit or 9-bit padding) |
| **SFPSHFT** | `shft(man, exp)` | Logical left-shift: converts float mantissa to scaled integer |
| **SFPSETEXP** | `setexp(frac, exponential_part)` | Set exponent field: recombines integer and fractional parts of 2^x |
| **SFPCAST** | `int32_to_float(fractional_part, 0)` | Integer-to-float conversion (INT32 to FP32, round-to-nearest-even) |
| **SFP_STOCH_RND** | `float_to_fp16b(y, 0)` | FP32 to BF16 rounding (round-to-nearest-even); only when `!is_fp32_dest_acc_en` |

Note: Each `_sfpu_exp_21f_bf16_` call emits the full instruction sequence above. Since `calculate_cosh` calls it twice per iteration (once for `v`, once for `-v`), the total instruction count per iteration is roughly 2x the above, plus the final addition and multiplication by 0.5.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST (via dst_reg)** | Source and destination for tile data. Each iteration reads 32 elements (2 physical DEST rows), computes cosh, and writes back. |
| **LREG0-3** | General-purpose registers used by the SFPI compiler for intermediate values: input `v`, `xlog2`, `threshold_low`, `threshold_high`, `z`, `exponential_part`, `fractional_part`, `frac`, `y`, and the final `result`. The compiler allocates these automatically. |
| **LREG4-7** | May be used by the compiler for additional temporaries when register pressure is high (e.g., during the two `_sfpu_exp_21f_bf16_` evaluations that must be summed). |
| **Programmable Constants** | No programmable constants are configured by `cosh_init` (since `FAST_APPROX=false` in `_init_exponential_`, the LOADMACRO constant-loading path is skipped). All constants (ONE_LN2, 127.0, 0.0, 255.0, polynomial coefficients, 0.5) are loaded via SFPLOADI instructions at runtime. |

### Address Mode Configuration

The address mode for this SFPU operation is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::cosh>()` in `llk_math_eltwise_unary_sfpu.h`.

Since `SfpuType::cosh` does not match any of the `if constexpr` specializations (topk_local_sort, typecast, reciprocal, unary_max/min, etc.), only the default `ADDR_MOD_7` is configured:

| Hardware Generation | Address Mode | srca.incr | srcb.incr | dest.incr |
|---------------------|-------------|-----------|-----------|-----------|
| **Wormhole B0** | `ADDR_MOD_7` | 0 | 0 | 0 |
| **Blackhole** | `ADDR_MOD_7` | 0 | 0 | 0 |

Both generations configure identical address modes. The zero-increment ADDR_MOD_7 means that DEST auto-increment is not used at the hardware level for this operation. Instead, DEST address advancement is handled explicitly:
- **Within a face**: `dst_reg++` in the SFPI code advances the SFPU's internal DEST pointer by 1 sfpi row (2 physical DEST rows) per iteration.
- **Between faces**: The params dispatch issues `TTI_SETRWC(CR_D, 8, SET_D)` twice on Wormhole (or `math::inc_dst_addr<8>()` twice on Blackhole) to advance past the face boundary (16 physical DEST rows total).

On Wormhole, `math::set_addr_mod_base()` and `math::clear_addr_mod_base()` are called around the SFPU execution to manage the address mode context. On Blackhole, these calls are absent from the params dispatch (the `_llk_math_eltwise_unary_sfpu_start_`/`_llk_math_eltwise_unary_sfpu_done_` functions handle DEST addressing without explicit addr_mod_base management).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, init/func strings, approx mode, and macro defines for COSH
   **Key Findings**: `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default); `get_op_approx_mode()` returns `false` (default); `get_op_init_and_func_default()` returns `cosh_tile_init()` / `cosh_tile({idst})`; `get_macro_definition()` returns `"SFPU_OP_COSH_INCLUDE"`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`
   **Reason**: API header defining `cosh_tile()` and `cosh_tile_init()` -- entry point for the SFPU dispatch
   **Key Findings**: `cosh_tile()` uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)`; `cosh_tile_init()` uses `SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::cosh_init, APPROX)`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Macro definitions for SFPU kernel dispatch patterns
   **Key Findings**: `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, FP32, ITER>, DST_IDX, VECTOR_MODE)`; `SFPU_INIT_KERNEL_CALL` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>)`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`
   **Reason**: Core SFPU implementation for cosh (Wormhole B0)
   **Key Findings**: `calculate_cosh()` computes `(exp(v) + exp(-v)) * 0.5` using `_sfpu_exp_21f_bf16_` helper; `cosh_init()` calls `_init_exponential_<APPROX, false, kCONST_1_FP16B>()`

5. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`
   **Reason**: Core SFPU implementation for cosh (Blackhole)
   **Key Findings**: Identical implementation to Wormhole B0

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Shared helper functions `_sfpu_exp_21f_bf16_`, `_float_to_int32_for_exp_21f_`, and `_init_exponential_`
   **Key Findings**: `_sfpu_exp_21f_bf16_` implements the Moroz et al. 2022 `exp_21f` algorithm: converts x to base-2 via `x/ln2`, clamps, extracts integer/fractional parts, evaluates 2nd-degree polynomial for fractional part, recombines via `setexp`; `_init_exponential_` with `FAST_APPROX=false` is effectively a no-op (no LOADMACRO setup)

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h`
   **Reason**: `PolynomialEvaluator::eval` used for the fractional-part polynomial refinement
   **Key Findings**: Implements Horner's method via recursive variadic template; `eval(x, c0, c1, c2)` = `c0 + x * (c1 + x * c2)`, which compiles to a chain of SFPMAD instructions

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function `_llk_math_eltwise_unary_sfpu_params_` (Wormhole)
   **Key Findings**: For `VectorMode::RC`, iterates over 4 faces calling the SFPU functor once per face, with `TTI_SETRWC` face-stride advancement between faces

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function (Blackhole variant)
   **Key Findings**: Same face iteration pattern as Wormhole but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` and `_llk_math_eltwise_unary_sfpu_start_`/`_done_` abstractions instead of direct `TTI_SETRWC`

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h`
    **Reason**: Init dispatcher that calls `_llk_math_eltwise_unary_sfpu_init_` and then the user-provided init callback
    **Key Findings**: Two-step init: first configures addr_mod and resets counters, then calls `cosh_init<APPROX>()`

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Address mode configuration (`eltwise_unary_sfpu_configure_addrmod`) and start/done functions
    **Key Findings**: Default `ADDR_MOD_7` with all increments = 0; `SfpuType::cosh` does not trigger any special addr_mod configuration

12. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: SFPI intrinsic definitions mapping C++ abstractions to hardware builtins
    **Key Findings**: `vec_min_max` -> `SFPSWAP`; `exexp_nodebias` -> `SFPEXEXP(NODEBIAS)`; `exman8`/`exman9` -> `SFPEXMAN`; `int32_to_float` -> `SFPCAST`; `float_to_fp16b` -> `SFP_STOCH_RND`; `shft` -> `SFPSHFT`

13. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware model reference for tile geometry, DEST layout, stride-2 addressing, instruction semantics
    **Key Findings**: ITERATIONS=8 per face, dst_reg++ = 2 physical DEST rows = 32 elements, SFPMAD is used for all float add/multiply, SFPSWAP for vec_min_max
=======
# SFPU Analysis: cosh

## Overview
cosh is the hyperbolic cosine operation: cosh(x) = (exp(x) + exp(-x)) / 2.

## SFPU Kernel
- **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`
- **Function**: `calculate_cosh<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()`
- **Init**: `cosh_init()` calls `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()`

### Key Implementation Pattern
```cpp
for (int d = 0; d < ITERATIONS; d++) {
    sfpi::vFloat v = sfpi::dst_reg[0];
    sfpi::vFloat result =
        (_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v) + _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)) * 0.5f;
    sfpi::dst_reg[0] = result;
    sfpi::dst_reg++;
}
```

Uses `_sfpu_exp_21f_bf16_` for computing exp(x) and exp(-x), then averages.

## Compute API
- **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`
- **Init**: `cosh_tile_init()` -> `SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::cosh_init, APPROX)`
- **Compute**: `cosh_tile(idst)` -> `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)`

## Split Includes
In `sfpu_split_includes.h`:
```cpp
#if SFPU_OP_COSH_INCLUDE
#include "api/compute/eltwise_unary/cosh.h"
#endif
```

## Op Registration
- **Enum**: `UnaryOpType::COSH` in `unary_op_types.hpp`
- **Block defines**: `case UnaryOpType::COSH: return "SFPU_OP_COSH_INCLUDE";`
- **Init/func**: `case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};`
- **String parse**: `if (name == "cosh") { return UnaryWithParam(UnaryOpType::COSH); }`

## Python Binding
- `unary.hpp`: `REGISTER_UNARY_OPERATION(cosh, COSH)`
- `unary_nanobind.cpp`: `bind_unary_operation<"cosh", &ttnn::cosh>(...)`
- Golden: `ttnn.attach_golden_function(ttnn.cosh, golden_function=_golden_function_cosh)` using `torch.cosh`

## Relevance to sinh
Extremely high - identical structure, just change `+` to `-` in the formula. Same exp-based approach, same init.
>>>>>>> gen-sinh-v2
