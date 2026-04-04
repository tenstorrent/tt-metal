## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `COSH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cosh_tile_init(); cosh_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(COSH)` in `unary_op_utils.cpp` -- returns `false` (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (= `math_approx_mode` = `false`), `DST_ACCUM_MODE`, `8` | `get_op_init_and_func_default()` -- non-parameterized: `cosh_tile_init()` / `cosh_tile(idst)`. Inside `cosh.h`, init uses `SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::cosh_init, APPROX)` and tile uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)` |
| Effective SFPU path | `APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en=DST_ACCUM_MODE` (typically false). The `calculate_cosh` function does not branch on `APPROXIMATION_MODE` -- it always uses `_sfpu_exp_21f_bf16_`. The init function `_init_exponential_<false, false, kCONST_1_FP16B>()` takes the final `else` branch, calling `_init_sfpu_reciprocal_<false>()`. | `ckernel_sfpu_cosh.h` line 16-25 (no `if constexpr` on APPROXIMATION_MODE); `ckernel_sfpu_exp.h` line 923-927 (else branch of `_init_exponential_`) |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the API header calls the macro `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` which expands directly to `_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h` (WH and BH are identical) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `cosh_tile_init(); cosh_tile(0);` per tile.
2. **API header** (`cosh.h`): `cosh_tile(idst)` calls `MATH(SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC))`.
3. **Macro expansion** (`llk_math_eltwise_unary_sfpu_macros.h`): The macro expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_cosh<APPROX, DST_ACCUM_MODE, 8>, idst, (int)VectorMode::RC)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, stalls for SFPU readiness, then loops over 4 faces (VectorMode::RC), calling `calculate_cosh()` once per face with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_cosh.h`): `calculate_cosh<APPROX, DST_ACCUM_MODE, 8>()` iterates 8 times per face, computing `cosh(x) = (exp(x) + exp(-x)) / 2` using `_sfpu_exp_21f_bf16_` from `ckernel_sfpu_exp.h`.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (face 0 through face 3).
- **Operation invocation**: The dispatch function calls `calculate_cosh()` once per face in a `for (int face = 0; face < 4; face++)` loop. Each call processes 8 iterations (ITERATIONS=8), covering one full face (8 iterations x 32 elements = 256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `SETRWC` with `CR_D, 8, 0, 0, SET_D` is called twice between faces (advancing 16 physical DEST rows = 1 face). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice. Address mode `ADDR_MOD_7` is configured with all-zero increments (srca=0, srcb=0, dest=0) on both WH and BH.

### Annotated SFPU Kernel Source

The cosh kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, etc.), so Style A is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h

// cosh(x) = (exp(x) + exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_cosh() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false (typical), ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];       // SFPLOAD: load 32 elements from current DEST position
        sfpi::vFloat result =
            (_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)   // exp(x) via 21-float algorithm
             + _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)) // exp(-x); negation is SFPMAD with sign inversion
            * 0.5f;                                         // SFPMAD: multiply sum by 0.5
        sfpi::dst_reg[0] = result;                // SFPSTORE: write 32 elements back to current DEST position
        sfpi::dst_reg++;                          // advance by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
void cosh_init() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>(); // APPROXIMATION_MODE=false, FAST_APPROX=false, scale=0x3F80 (1.0 in FP16B)
    // With APPROXIMATION_MODE=false and FAST_APPROX=false, this takes the else branch:
    // calls _init_sfpu_reciprocal_<false>() -- sets up reciprocal tables
    // Note: calculate_cosh uses _sfpu_exp_21f_bf16_ which does NOT need reciprocal;
    // the init is inherited from the generic _init_exponential_ structure
}
```

The core computation is delegated to `_sfpu_exp_21f_bf16_` which implements the `exp_21f` algorithm from Moroz et al. 2022. This function is called twice per iteration: once for `exp(x)` and once for `exp(-x)`. Below is the annotated source:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

// Implementation notes, see the original file for more details
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);     // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);    // SFPEXMAN: extract mantissa with implicit bit (8-bit pad)
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp)); // SFPSHFT: shift mantissa left by exponent
    return man;
}

// Implementation notes, see the original file for more details
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) // is_fp32_dest_acc_en=false (typical)
{
    // exp(x) = 2^(x/ln2) = 2^(z_i) * 2^(z_f)
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f); // SFPMAD: val * ONE_LN2 + 127.0

    // Clamp xlog2 to [0, 255] to avoid overflow in intermediate values
    sfpi::vFloat threshold_low  = 0.f;                  // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);   // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2);             // SFPSWAP: min/max sort -> threshold_low=min, xlog2=max (clamp low)
    sfpi::vec_min_max(xlog2, threshold_high);            // SFPSWAP: min/max sort -> xlog2=min, threshold_high=max (clamp high)

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2); // SFPEXEXP + SFPEXMAN + SFPSHFT

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP (no debias): extract 2^(integer part)
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN (9-bit pad): extract fractional part

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 -> fp32

    // 2nd degree polynomial approximation of 2^x on [0, 2^23]
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
    // Horner's method: frac = 1.0017248 + frac * (7.8396e-08 + frac * 4.7918e-15)
    // This generates a chain of SFPMAD instructions (2 multiplies + 2 adds = 2 SFPMADs)

    // Recombine: 2^(z_i) * 2^(z_f) by setting the exponent
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: set exponent field

    if constexpr (!is_fp32_dest_acc_en)
    {
        // Implementation notes, see the original file for more details
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: fp32 -> fp16b (round to nearest even)
    }

    return y;
}
```

### SFPU Instructions Used

| Instruction | SFPU Opcode | Description | Usage in cosh |
|------------|-------------|-------------|---------------|
| `SFPLOAD` | 0x70 | Load data from DEST row into LREG | Load input value `x` from `dst_reg[0]` each iteration |
| `SFPSTORE` | 0x72 | Store LREG value back to DEST row | Write `cosh(x)` result back to `dst_reg[0]` each iteration |
| `SFPMAD` | 0x84 | Fused multiply-add: `VD = VA * VB + VC` | (1) `val * ONE_LN2 + 127.0`, (2) negation of input (`-v`), (3) Horner's polynomial chain (2 SFPMADs for degree-2 poly), (4) `(exp(x) + exp(-x)) * 0.5`. Called many times per iteration since both `exp(x)` and `exp(-x)` are computed. |
| `SFPSWAP` | 0x92 | Conditional swap / vec_min_max | Clamp `xlog2` to [0, 255] range (2 swaps per `_sfpu_exp_21f_bf16_` call, so 4 total per cosh iteration) |
| `SFPEXEXP` | 0x77 | Extract exponent field from float | Extract exponent in `_float_to_int32_for_exp_21f_` (with debias) and `exexp_nodebias` (without debias) for integer/fractional part separation |
| `SFPEXMAN` | 0x75 | Extract mantissa field from float | Extract mantissa with implicit bit in `exman8` and `exman9` for integer/fractional part separation |
| `SFPSHFT` | 0x7A | Bit shift operation | Shift mantissa left by exponent value in `_float_to_int32_for_exp_21f_` |
| `SFPSETEXP` | 0x7C | Set exponent field of float | Recombine integer exponent with fractional polynomial result to form final `2^x` value |
| `SFPCAST` | 0x90 | Format conversion (int32 to fp32) | Convert fractional part from integer to float before polynomial evaluation |
| `SFP_STOCH_RND` | 0x91 | Stochastic/deterministic rounding | Convert fp32 result to fp16b (round-to-nearest-even) when `!is_fp32_dest_acc_en` |
| `SFPLOADI` | 0x71 | Load 16-bit immediate to LREG | Load constant values (0.0, 255.0, ONE_LN2, polynomial coefficients, 0.5, etc.) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST** (via `dst_reg[0]`) | Input source and output destination. Each iteration loads 32 elements (2 physical rows x 16 elements) from the current DEST position, computes cosh, and stores back. |
| **LREG0-LREG3** | General-purpose computation registers. Used to hold intermediate values: input `v`, `xlog2`, `z`, `exponential_part`, `fractional_part`, `frac`, polynomial intermediates, and the final result. The compiler allocates these dynamically across the many operations in `_sfpu_exp_21f_bf16_`. |
| **Programmable constants** | Not explicitly programmed by `calculate_cosh`. The init function `_init_exponential_<false, false, ...>()` calls `_init_sfpu_reciprocal_<false>()` which may configure programmable constants for reciprocal, but these are not used by the `_sfpu_exp_21f_bf16_` path. |

### Address Mode Configuration

The address mode is configured identically on both Wormhole and Blackhole:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode with no auto-increment. DEST addressing is managed manually by `dst_reg++` (SFPI abstraction) within the kernel loop, and by `SETRWC` (WH) or `inc_dst_addr<8>()` (BH) between faces in the parameters dispatch. |

The `eltwise_unary_sfpu_configure_addrmod<SfpuType::cosh>()` function only sets `ADDR_MOD_7` with all-zero increments, since `SfpuType::cosh` does not match any of the special-case conditions (topk_local_sort, typecast, unary_min/max). No additional address modes (e.g., `ADDR_MOD_6`) are configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 define, and approximation mode for COSH
   **Key Findings**: COSH uses `eltwise_sfpu.cpp`, macro define `SFPU_OP_COSH_INCLUDE`, init `cosh_tile_init()`, func `cosh_tile(idst)`, `get_op_approx_mode` returns `false` for all ops

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`
   **Reason**: API header that exposes `cosh_tile()` and `cosh_tile_init()` to the compute kernel
   **Key Findings**: Uses `SFPU_INIT_KERNEL_CALL` for init and `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` for tile function with `APPROX, DST_ACCUM_MODE, 8` template args and `VectorMode::RC`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`
   **Reason**: Core SFPU implementation of `calculate_cosh`
   **Key Findings**: WH and BH implementations are identical. Computes `cosh(x) = (exp(x) + exp(-x)) / 2` using `_sfpu_exp_21f_bf16_` helper. No branching on `APPROXIMATION_MODE`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_sfpu_exp_21f_bf16_`, `_float_to_int32_for_exp_21f_`, and `_init_exponential_` functions
   **Key Findings**: `_sfpu_exp_21f_bf16_` implements the exp_21f algorithm from Moroz et al. 2022 using range reduction to base-2, integer/fractional decomposition, and 2nd degree polynomial approximation. Init with `APPROX=false, FAST_APPROX=false` calls `_init_sfpu_reciprocal_<false>()`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h`
   **Reason**: Contains `PolynomialEvaluator::eval` used for polynomial approximation in exp
   **Key Findings**: Implements Horner's method via recursive variadic template. For 3 coefficients, produces 2 SFPMADs.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that handles VectorMode and face iteration
   **Key Findings**: For VectorMode::RC, loops 4 faces, calling sfpu_func once per face with SETRWC x2 between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: LLK init and address mode configuration
   **Key Findings**: `ADDR_MOD_7` configured with all-zero increments for cosh (no special-case match). Same on BH.

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-instruction mappings for `vec_min_max`, `exexp_nodebias`, `exman8/9`, `int32_to_float`, `float_to_fp16b`
   **Key Findings**: `vec_min_max` -> `SFPSWAP`, `exexp` -> `SFPEXEXP`, `exman8/9` -> `SFPEXMAN`, `int32_to_float` -> `SFPCAST`, `float_to_fp16b` -> `SFP_STOCH_RND`

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Macro definitions for `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` and `SFPU_INIT_KERNEL_CALL`
   **Key Findings**: `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(FN, APPROXIMATE, FP32, ITER, DST_IDX, VECTOR_MODE)` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, FP32, ITER>, DST_IDX, VECTOR_MODE)`
