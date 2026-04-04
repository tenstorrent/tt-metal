## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the CELU activation operation.

**Important Note**: In the current codebase, CELU has a fully implemented SFPU kernel at the ckernel level (`ckernel_sfpu_activations.h`), and `UnaryOpType::CELU` exists in the enum. However, the operation is **not wired up** in the host-side dispatch layers — there is no `celu_tile()`/`celu_tile_init()` compute API function, no LLK dispatch function, and no `get_op_init_and_func()` case for `UnaryOpType::CELU`. The analysis below documents the SFPU kernel as it exists, and describes the intended dispatch path based on the generic `_calculate_activation_` / `_llk_math_eltwise_unary_sfpu_params_` pattern used by other activations.

### Unary Dispatch Summary
- **UnaryOpType**: `CELU` (defined in `unary_op_types.hpp:126`)
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` — CELU has no explicit case, falls through to `default`)
- **SFPU_OP_CHAIN_0 expansion**: Not currently wired — would expand to something like `celu_tile(0, alpha_packed, alpha_recip_packed)` once compute API is implemented
- **Intended ckernel call**: `_calculate_activation_<APPROXIMATION_MODE, ActivationType::Celu, ITERATIONS>(param0, param1)` where `param0` = alpha (as uint32-packed float) and `param1` = 1/alpha (as uint32-packed float)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CELU)` in `unary_op_utils.cpp` — the switch has only `default: return false`, so all ops return `false` unless explicitly overridden |
| Template parameter (SFPU_OP_CHAIN) | None (not wired) | CELU has no `get_op_init_and_func()` case. If it were wired, the `APPROXIMATION_MODE` template parameter in `_calculate_activation_` would be controlled by the `math_approx_mode` compile-time define passed via `ComputeConfig` |
| Effective SFPU path | `APPROXIMATION_MODE=false` → `_calculate_exponential_body_<false>` → non-approximate branch: `_sfpu_exp_(setsgn(in, 0))` + conditional `_sfpu_reciprocal_<2>()` for negative inputs | `ckernel_sfpu_exp.h:313-322` — the `else` branch of `if constexpr (APPROXIMATION_MODE)` |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist — no `celu_tile()` / `celu_tile_init()` compute API function has been implemented |
| **LLK Dispatch** | This level of abstraction doesn't exist — no `_llk_math_eltwise_unary_sfpu_celu_()` LLK function. The intended dispatch pattern is `_llk_math_eltwise_unary_sfpu_params_()` in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic parametric dispatch) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_activations.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_activations.h` (Blackhole) — both files are identical |
| **Parameters Dispatch** | Intended: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` — the generic `_llk_math_eltwise_unary_sfpu_params_()` template function, which handles VectorMode dispatch and SETRWC face advancement |

### Call Chain
The intended call chain (once wired) would be:

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` macro expands to `celu_tile(0, alpha, alpha_recip)` (tile-level API call)
2. **Compute API** (not yet implemented): `celu_tile(idst, alpha, alpha_recip)` → calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(_calculate_activation_<APPROXIMATE, ActivationType::Celu, 8>, idst, VectorMode::RC, alpha, alpha_recip)`
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_()` → sets DEST write address, sets addr_mod base, stalls for SFPU, then loops over 4 faces calling `sfpu_func(args...)` with SETRWC between faces
4. **Core SFPU** (`ckernel_sfpu_activations.h`): `_calculate_activation_<APPROX, ActivationType::Celu, 8>(param0, param1)` → loops 8 iterations per face, loading `dst_reg[0]`, calling `ActivationImpl<APPROX, Celu>::apply(v, param0, param1)`, writing back to `dst_reg[0]`, advancing `dst_reg++`
5. **CELU Apply** (`ckernel_sfpu_activations.h`): Converts packed params to floats via `Converter::as_float()`, applies conditional `v_if (v < 0.0f)` → computes `alpha * (exp(v * alpha_recip) - 1)` using `_calculate_exponential_body_<APPROX>()`

**Current state**: Only steps 4-5 exist. Steps 1-3 are not implemented.

### Parameters Dispatch Summary

- **Vector mode**: The intended dispatch would use `VectorMode::RC`, processing all 4 faces of the tile (32×32 = 1024 elements total). This is the standard mode for element-wise unary operations.
- **Operation invocation**: The generic `_llk_math_eltwise_unary_sfpu_params_()` function calls the SFPU function once per face (4 calls total for RC mode). Each call processes 8 SFPU iterations within `_calculate_activation_<..., 8>()`, covering one full 16×16 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, `set_addr_mod_base()` activates `ADDR_MOD_4..7` range; `ADDR_MOD_7` is set with `dest.incr=0` (SFPU uses software `dst_reg++` not hardware auto-increment). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` × 2 advances by 16 physical DEST rows = 1 face. On Blackhole, the same pattern is used via `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The CELU kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_activations.h

// General template structure to implement activations
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
struct ActivationImpl;

// Specialization for CELU activation
template <bool APPROXIMATION_MODE>
struct ActivationImpl<APPROXIMATION_MODE, ActivationType::Celu>
{
    static inline void apply(sfpi::vFloat& v, std::uint32_t param0, std::uint32_t param1)
    {
        // All params are in FP16_B format
        // param0 = alpha
        // param1 = alpha_recip
        sfpi::vFloat alpha       = Converter::as_float(param0);  // Reinterpret uint32 bits as float
        sfpi::vFloat alpha_recip = Converter::as_float(param1);  // Reinterpret uint32 bits as float (1/alpha)

        v_if (v < 0.0f)  // SFPU conditional: only modify negative elements
        {
            // Compute exp(x / alpha): multiply by reciprocal, then exponentiate
            sfpi::vFloat exp_val = _calculate_exponential_body_<APPROXIMATION_MODE>(v * alpha_recip);

            // Compute CELU: alpha * (exp(x / alpha) - 1)
            v = alpha * (exp_val - 1.0f);  // For x >= 0, v is unchanged (identity)
        }
        v_endif;
    }
};

// Dispatch loop: processes ITERATIONS sfpi rows (default 8 = one face)
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE, int ITERATIONS = 8>
inline void _calculate_activation_(std::uint32_t param0, std::uint32_t param1)
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];  // Load 32 elements (2 physical DEST rows) into vFloat
        apply_activation<APPROXIMATION_MODE, ACTIVATION_TYPE>(v, param0, param1);
        sfpi::dst_reg[0] = v;  // Store result back to same DEST location
        sfpi::dst_reg++;       // Advance 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}
```

#### Helper: `Converter::as_float`

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h

class Converter
{
public:
    static float as_float(std::uint32_t value)  // Reinterpret uint32 bit pattern as float
    {
        union
        {
            std::uint32_t u;
            float f;
        } converter {value};

        return converter.f;
    }
};
```

#### Helper: `_calculate_exponential_body_<false>` (non-approximate path)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE>  // APPROXIMATION_MODE=false for CELU default
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in)
{
    sfpi::vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        // Implementation notes, see the original file for more details
        constexpr int FRAC_BITS         = 3;
        constexpr std::uint32_t SP_BIAS = 127 << FRAC_BITS;

        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
        sfpi::vFloat conv           = in * vConstLn2Recip;

        sfpi::vInt c23_73 = p_exp::C23_73;
        sfpi::vInt tmp    = sfpi::reinterpret<sfpi::vInt>(conv) - c23_73;

        tmp += SP_BIAS;

        out = sfpi::reinterpret<sfpi::vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Force sign to 0 (make number positive), compute exp(|in|)
        out = _sfpu_exp_(sfpi::setsgn(in, 0));

        v_if (in < 0)  // For negative inputs: exp(x) = 1/exp(|x|)
        {
            out = _sfpu_reciprocal_<2>(out);  // 2 Newton-Raphson iterations
        }
        v_endif;
    }

    return out;
}
```

#### Helper: `_sfpu_exp_` (Horner-form exponential)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126);  // Clamp to [-1, 0) range for series convergence
    }
    v_endif;

    // Run series in Horner form: polynomial approximation of exp(x) for x in [-1, 0]
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val              = val * tmp + sfpi::vConst1;

    v_if (exp >= 0)
    {
        val = val * val;  // Square to recover the extracted exponent bits
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            v_and(exp >= 0);  // Narrow predication: only continue while exp >= 0
            val = val * val;  // Repeated squaring: exp(x) = exp(frac)^(2^int_part)
        }
    }
    v_endif;

    return val;
}
```

#### Helper: `_sfpu_reciprocal_<2>` (Newton-Raphson reciprocal, 2 iterations)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>  // max_iter=2 when called from _calculate_exponential_body_<false>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Scale input to [1.0, 2.0) range and negate
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Compute scale factor: 2^(255-in.Exp) via bitwise NOT of exponent
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Clear mantissa from scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // First Newton-Raphson iteration: t = 1.0 - x*y; y = y + y*t
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    scale *= 0.5f;  // Adjust scale: 2^(254-in.Exp) = correct bias

    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Second Newton-Raphson iteration
        t = sfpi::vConst1 + negative_x * y;
        y = y + y * t;
    }

    // Apply scaling and restore sign
    y = y * scale;
    y = sfpi::setsgn(y, in);

    return y;
}
```

### SFPU Instructions Used

The following SFPU instructions/intrinsics are used by the CELU kernel and its helpers:

| Instruction/Intrinsic | Description | Used In |
|---|---|---|
| `SFPMAD` (via `vFloat * vFloat`, `vFloat + vFloat`, `vFloat * vFloat + vFloat`) | Fused multiply-add: `a * b + c`. All float adds are implemented as `a * 1.0 + b` via SFPMAD. | `ActivationImpl::apply`, `_sfpu_exp_`, `_sfpu_reciprocal_` |
| `SFPLOAD` (via `dst_reg[0]` read) | Load 32 elements from DEST into LREG | `_calculate_activation_` loop |
| `SFPSTORE` (via `dst_reg[0] = v` write) | Store 32 elements from LREG back to DEST | `_calculate_activation_` loop |
| `SFPLOADI` (via `vFloat literal`, `sfpi::vConst*`) | Load immediate constant into LREG | Various constants (0.0f, 1.0f, 0.5f, etc.) |
| `SFPSETCC` (via `v_if (v < 0.0f)`, `v_if (in < 0)`, `v_if (exp >= 0)`) | Set condition codes based on comparison | `ActivationImpl::apply`, `_calculate_exponential_body_`, `_sfpu_exp_` |
| `SFPENCC` / `SFPCOMPC` (via `v_endif`) | End conditional / complement condition codes | Matching every `v_if` |
| `SFPAND` (via `v_and(exp >= 0)`) | Narrow (AND) predication within conditional block | `_sfpu_exp_` repeated squaring loop |
| `SFPEXEXP` (via `sfpi::exexp(val)`) | Extract exponent field from float | `_sfpu_exp_` |
| `SFPSETEXP` (via `sfpi::setexp(val, 126)`) | Set exponent field of float | `_sfpu_exp_` range clamping |
| `SFPSETSGN` (via `sfpi::setsgn(in, 0)`, `sfpi::setsgn(y, in)`) | Set/clear sign bit of float | `_calculate_exponential_body_`, `_sfpu_reciprocal_` |
| `SFPSETMAN` (via `sfpi::setman(...)`) | Set mantissa field of float | `_sfpu_reciprocal_` (scale factor) |
| `SFPNOT` (via `~sfpi::reinterpret<sfpi::vUInt>(in)`) | Bitwise NOT (used to compute 255-exponent for reciprocal scale) | `_sfpu_reciprocal_` |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **DEST registers** (via `dst_reg[0]`) | Source and destination for tile data. Each `dst_reg[0]` access reads/writes 32 elements (2 physical DEST rows × 16 elements/row). Advanced via `dst_reg++` per iteration. |
| **LREGs (L0-L3)** | Implicit working registers used by SFPI abstractions. `vFloat` variables map to LREGs for intermediate computation (e.g., `v`, `alpha`, `alpha_recip`, `exp_val`, `out`, `y`, `t`, `scale`). |
| **vConstFloatPrgm0** (LREG12) | Used by `_sfpu_reciprocal_` (quadratic coefficient k0 = 0.3232325...) and by `_calculate_exponential_body_<true>` approximate path (ln2 reciprocal). For the non-approximate path used by default CELU, this register holds the reciprocal init coefficient. |
| **vConstFloatPrgm1** (LREG13) | Used by `_sfpu_reciprocal_` (quadratic coefficient k1 = 1.4545459...) |
| **vConstFloatPrgm2** (LREG14) | Used by `_sfpu_reciprocal_` (quadratic coefficient k2 = 2.1212124...) |
| **vConst1** | Float constant 1.0 — used in CELU formula (`exp_val - 1.0f`), reciprocal Newton-Raphson (`1 + neg_x * y`), and exp Horner form |
| **vConstNeg1** | Float constant -1.0 — used in reciprocal to negate scaled input |
| **vConst0p8373** | Float constant ~0.8373 — Horner coefficient in `_sfpu_exp_` |
| **Condition Code (CC) register** | Managed by `v_if`/`v_endif`/`v_and` — lanes where condition is false are masked from writes. CELU uses nested conditionals: outer `v_if (v < 0)` in apply, inner `v_if (in < 0)` in exp body, and `v_if (exp >= 0)` with `v_and` narrowing in `_sfpu_exp_`. |

### Address Mode Configuration

The CELU operation would use the default unary SFPU address mode configuration set by `eltwise_unary_sfpu_configure_addrmod<SfpuType>()`.

Since CELU does not have its own `SfpuType` enum value (it uses the generic `_calculate_activation_` template via `_llk_math_eltwise_unary_sfpu_params_`), only `ADDR_MOD_7` is configured:

**Wormhole B0** (`llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: srca.incr=0, srcb.incr=0, dest.incr=0
```
No special `ADDR_MOD_6` configuration is needed for CELU — it doesn't match any of the special-cased `SfpuType` values (topk_local_sort, typecast, unary_max/min). DEST address progression is managed entirely by software: `dst_reg++` within the SFPU kernel (advancing 1 sfpi row = 2 physical DEST rows = 32 elements per iteration) and `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` × 2 between faces (advancing 16 physical DEST rows = 1 face of 256 elements).

**Blackhole** (`llk_math_eltwise_unary_sfpu.h`):
Same as Wormhole — identical `ADDR_MOD_7` configuration with `dest.incr=0`. The Blackhole variant has `SfpuType::reciprocal` as an additional special case for `ADDR_MOD_6`, but this doesn't affect CELU since reciprocal is called as an inline helper function, not as the top-level SFPU type.

**Note on `set_addr_mod_base()`**: The LLK dispatch (`_llk_math_eltwise_unary_sfpu_start_`) calls `math::set_addr_mod_base()` which sets `ADDR_MOD_SET_Base = 1`, meaning the SFPU hardware uses addr mods 4..7 (instead of 0..3) to avoid conflicts with the A2D unpacker which uses `ADDR_MOD_0` and `ADDR_MOD_2`. After processing completes, `math::clear_addr_mod_base()` restores the base to 0..3.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_activations.h`
   **Reason**: Core SFPU implementation for CELU activation
   **Key Findings**: CELU is implemented as `ActivationImpl<APPROXIMATION_MODE, ActivationType::Celu>` template specialization. Takes two uint32 params (alpha, alpha_recip), applies conditional path for negative inputs using `_calculate_exponential_body_`. Identical between Wormhole and Blackhole.

2. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_calculate_exponential_body_<APPROXIMATION_MODE>` and `_sfpu_exp_` helper functions used by CELU
   **Key Findings**: Non-approximate path (`APPROXIMATION_MODE=false`) computes exp via `_sfpu_exp_(setsgn(in, 0))` + conditional reciprocal for negative inputs. `_sfpu_exp_` uses Horner-form polynomial with repeated squaring.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Contains `_sfpu_reciprocal_<N>` used by the non-approximate exponential path
   **Key Findings**: Quadratic initial estimate + Newton-Raphson iterations. Called with `max_iter=2` from `_calculate_exponential_body_<false>`. Uses programmable constants `vConstFloatPrgm0/1/2`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Contains `Converter::as_float()` used to convert uint32 packed params to float
   **Key Findings**: Simple union-based type punning from uint32 to float.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Generic parametric LLK dispatch function that would be used for CELU
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode dispatch (RC/R/C), face iteration with SETRWC advancement, and DEST address setup. This is the intended LLK-level dispatcher for parametric activations.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains addr_mod configuration and start/done functions for unary SFPU operations
   **Key Findings**: `ADDR_MOD_7` set to all-zero increments; `set_addr_mod_base()` uses addr mods 4..7 to avoid A2D conflicts. Standard init/done sequence with STALLWAIT.

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Host-side dispatch utilities for unary operations
   **Key Findings**: `get_op_approx_mode()` returns `false` for all ops (default case only). `get_op_init_and_func_parameterized()` does NOT have a case for CELU — it would throw `TT_THROW("unexpected parameterized op type")`. CELU is not wired to the compute kernel dispatch.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h`
   **Reason**: Defines `ActivationType` enum
   **Key Findings**: `ActivationType::Celu = 0` — first entry in the enum. Other types: Elu=1, Gelu=2, Hardtanh=3, Hardsigmoid=4.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h`
   **Reason**: Defines `p_exp` constants used by approximate exp path
   **Key Findings**: `p_exp::FRAC_BITS = 3`, `p_exp::C23_73 = 0x4340` — fixed-point conversion constants for the approximate exponential.

10. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/sfpu_operations.h`
    **Reason**: LLK test helper showing how CELU is invoked in tests
    **Key Findings**: Test invocation: `_calculate_activation_<APPROX_MODE, ActivationType::Celu, ITERATIONS>(10, 1.0f / 10.0f)` — uses alpha=10 (as uint32-reinterpreted float) and alpha_recip=0.1f as test parameters.

11. **File**: `docs/sfpu_operations/key_notes/celu_key_notes.md`
    **Reason**: Formula reference for CELU operation
    **Key Findings**: CELU formula: `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`. Default alpha=1.0. Deterministic, mode-independent.
