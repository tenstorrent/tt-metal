## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the LOGIT compute kernel dispatches to. LOGIT is a **composite operation** — unlike simple unary ops that use a single `SFPU_OP_CHAIN_0` dispatch, LOGIT has its own dedicated compute kernel that chains multiple SFPU tile-level API calls.

### Unary Dispatch Summary
- **UnaryOpType**: `LOGIT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/logit_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A — LOGIT does **not** use the standard `SFPU_OP_CHAIN_0` dispatch mechanism. Instead, `get_op_init_and_func_parameterized()` returns `{}` (empty pair), and the kernel directly calls multiple tile-level SFPU APIs in sequence.

#### Mathematical Definition
`logit(x, eps) = log(x / (1 - x))`, where `x` is optionally clamped to `[eps, 1-eps]`.

When `eps > 0.5`, a conditional negation is applied via the WHERE operation to ensure correct sign handling.

#### Compute Kernel Execution Flow
The logit kernel performs the following sequence per tile:

| Step | Condition | API Call | Operation |
|------|-----------|----------|-----------|
| 1 | `#ifdef CLAMP` (eps >= 0.0) | `clamp_tile(0, packed_scalar1, packed_scalar2)` | Clamp x to [eps, 1-eps] |
| 2 | Always | Pack to `cb_tmp0`, wait, re-load into DEST slots 0 and 1 | Duplicate clamped tile |
| 3 | Always | `rsub_tile(0, 0x3F800000u)` | Compute `1.0 - x` in DEST[0] |
| 4 | Always | `div_binary_tile(1, 0, 0)` | Compute `x / (1-x)` — DEST[1] / DEST[0] → DEST[0] |
| 5 | Always | `log_tile(0)` | Compute `log(x / (1-x))` |
| 6 | `#ifdef WHERE` (eps > 0.5) | `copy_dest_values(0, 2)` | Copy log result to DEST[2] |
| 7 | `#ifdef WHERE` | `copy_tile(cb_input, 0, 1)` + `unary_lt_tile(1, packed_scalar1)` | Load original x, compute `x < eps` mask in DEST[1] |
| 8 | `#ifdef WHERE` | `mul_unary_tile(0, 0xBF800000)` | Negate log result: `-logit(x)` in DEST[0] |
| 9 | `#ifdef WHERE` | `WHERE(1, 0, 2, 0)` | Select: if `x < eps` then `-logit(x)` else `logit(x)` |

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LOGIT)` in `unary_op_utils.cpp` — falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_parameterized()` returns `{}` — no SFPU_OP_CHAIN is used |
| Effective SFPU path | The `APPROX` macro (compile-time arg from `math_approx_mode`) is `false`, passed through to all sub-operations (clamp, div, log, where). For `div_binary_tile`, `APPROX=false` selects the 2-iteration Newton-Raphson reciprocal path (`_sfpu_reciprocal_<2>`). For `log_tile`, `APPROX=false` uses the Chebyshev polynomial path. | See individual sub-operation analyses below |

#### DEFINE Conditioning
The program factory sets preprocessor defines based on the `eps` parameter:

| eps range | `CLAMP` defined? | `WHERE` defined? |
|-----------|-------------------|-------------------|
| eps < 0 or not set | No | No |
| 0 <= eps <= 0.5 | Yes (`"clamp_tile"`) | No |
| eps > 0.5 | Yes (`"clamp_tile"`) | Yes (`"where_tile<DataFormat::...>"`) |

Source: `unary_program_factory.cpp` lines 137–148

#### Runtime Arguments
- `packed_scalar1`: `eps` value packed as the input data type
- `packed_scalar2`: `1.0 - eps` packed as the input data type

Source: `unary_program_factory.cpp` lines 138–141

### SFPU Abstraction Layers
Since LOGIT is a composite kernel, there is no single abstraction layer chain. Instead, each sub-operation has its own chain. The key paths are:

#### Sub-operation: `clamp_tile`

| Layer | File Path |
|-------|-----------|
| **API Header** | `api/compute/eltwise_unary/clamp.h` (generated at build time, not on disk) |
| **LLK Dispatch** | `llk_math_eltwise_unary_sfpu_params.h` (via generated LLK wrapper) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

#### Sub-operation: `div_binary_tile`

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h` (DIV path) + `ckernel_sfpu_recip.h` (reciprocal) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

#### Sub-operation: `log_tile`

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 92–116) |
| **LLK Dispatch** | `llk_math_eltwise_unary_sfpu_log.h` (generated at build time, not on disk) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

#### Sub-operation: `where_tile` (conditional, only when eps > 0.5)

| Layer | File Path |
|-------|-----------|
| **API Header** | `api/compute/eltwise_unary/where.h` (generated at build time, not on disk) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h` (ternary uses the binary params dispatch) |

### Call Chain
The logit compute kernel directly invokes tile-level API functions without the `SFPU_OP_CHAIN_0` indirection:

1. **`clamp_tile(0, packed_scalar1, packed_scalar2)`**: API header (generated `clamp.h`) → `llk_math_eltwise_unary_sfpu_*` dispatch with `_calculate_clamp_<APPROX, ITERATIONS>` as the SFPU functor → `_llk_math_eltwise_unary_sfpu_params_` iterates over 4 faces (VectorMode::RC), each invoking `_calculate_clamp_` with 8 iterations.

2. **`rsub_tile(0, 0x3F800000u)`**: API header (generated `rsub.h`) → dispatches an SFPU operation that computes `scalar - dst_reg[i]` for each element. The scalar `0x3F800000` is the IEEE 754 representation of `1.0f`. This produces `1.0 - x`.

3. **`div_binary_tile(1, 0, 0)`**: `eltwise_binary_sfpu.h` → `llk_math_eltwise_binary_sfpu_binop_div<APPROX, BinaryOp::DIV, DST_ACCUM_MODE>` → `_llk_math_eltwise_binary_sfpu_params_` iterates over 4 faces, each calling `_calculate_sfpu_binary_<..., BinaryOp::DIV, 8, ...>` which computes `in0 * _sfpu_reciprocal_<2>(in1)` (i.e., DEST[1] / DEST[0]).

4. **`log_tile(0)`**: `compute_kernel_api.h` → `llk_math_eltwise_unary_sfpu_log<APPROX, false, DST_ACCUM_MODE>` → dispatches `_calculate_log_<APPROX, false, ITERATIONS>` which uses a 3rd-order Chebyshev polynomial approximation of `ln(x)`.

5. **`WHERE(1, 0, 2, 0)`** (conditional): API header (generated `where.h`) → `_calculate_where_<APPROX, DataFormat, ITERATIONS>` which uses SFPLOADMACRO-based superscalar execution to perform per-element conditional selection.

### Parameters Dispatch Summary

#### Unary sub-operations (clamp, log, comp)
- **Vector mode**: `VectorMode::RC` — all 4 faces of the tile are processed
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` template iterates over 4 faces, invoking the SFPU functor once per face. Each functor invocation processes 8 sfpi rows (ITERATIONS=8), covering one 16×16 face.
- **DEST address progression**: Standard DEST progression. On Blackhole, `ADDR_MOD_7` is configured with `dest.incr = 0` (no auto-increment). Face advancement is handled by `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (16 physical rows = 1 face stride). Within a face, `dst_reg++` in the SFPU kernel advances 1 sfpi row per iteration.

#### Binary sub-operations (div)
- **Vector mode**: `VectorMode::RC` — all 4 faces
- **Operation invocation**: The `_llk_math_eltwise_binary_sfpu_params_` template iterates over 4 faces, invoking the SFPU functor once per face. The functor receives `dst_index_in0`, `dst_index_in1`, `dst_index_out` which are tile indices, and the functor uses `dst_tile_size_sfpi = 32` to compute offsets.
- **DEST address progression**: On Blackhole, `ADDR_MOD_7` with `dest.incr = 0`. Face advancement via `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice per face (8+8 = 16 physical rows).

#### Ternary sub-operations (where)
- Uses the binary sfpu params dispatch pattern. The WHERE kernel uses SFPLOADMACRO superscalar execution with raw DEST offsets computed from tile indices. Face advancement is via the same SETRWC mechanism.

### Annotated SFPU Kernel Source

#### 1. Clamp (Style A: SFPI-based)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_clamp_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{
    // All params are in FP16 format
    // param0 = min, param1 = max

    sfpi::s2vFloat16::Format format = sfpi::s2vFloat16::fp16a;

    sfpi::vFloat min    = sfpi::s2vFloat16(param0, format);
    sfpi::vFloat max    = sfpi::s2vFloat16(param1, format);
    sfpi::vFloat offset = sfpi::s2vFloat16b(param2); // 12 bits
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if (val < min)  // CC-guarded: lanes where val < min
        {
            val = min;
        }
        v_elseif (val >= max)  // CC-guarded: lanes where val >= max
        {
            val = max;
        }
        v_endif;

        sfpi::dst_reg[0] = val + offset;  // SFPMAD: val * 1.0 + offset

        sfpi::dst_reg++;
    }
}
```

#### 2. Log — Chebyshev Polynomial Approximation (Style A: SFPI-based)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = 0.692871f; // ln2
    sfpi::vConstFloatPrgm1 = 0.1058f;   // Chebyshev coeff A'
    sfpi::vConstFloatPrgm2 = -0.7166f;  // Chebyshev coeff B'
}

template <bool HAS_BASE_SCALING>  // HAS_BASE_SCALING=false for natural log
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0)
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    // Load from DEST and normalize to [1, 2) range
    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x  = setexp(in, 127); // SFPSETEXP: set exponent to bias (127), puts value in [1,2)

    // 3rd-order Chebyshev polynomial in Horner form on (x-1):
    // ln(x) ≈ x*(x*(A'*x + B') + C') + D'
    // A'=0.1058, B'=-0.7116, C'=2.0871, D'=-1.4753
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;  // A' = 0.1058
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;  // B' = -0.7166
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;
    // Each addition is SFPMAD (a*1.0+b); each multiply is SFPMAD (a*b+0.0)
    // This chain: 4 SFPMAD instructions for the polynomial

    // Extract exponent as integer and convert to float
    sfpi::vInt exp = sfpi::exexp(in);  // SFPEXEXP: extract biased exponent
    v_if (exp < 0)  // Handle negative exponent (sub-1.0 inputs)
    {
        exp = sfpi::setsgn(~exp + 1, 1);  // SFPNOT + SFPIADD + SFPSETSGN: negate
    }
    v_endif;

    sfpi::vFloat expf      = int32_to_float(exp, 0);  // SFPCAST or int-to-float conversion
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;   // ln(2) = 0.692871
    sfpi::vFloat result    = expf * vConstLn2 + series_result; // SFPMAD: exp_correction + polynomial
    // Final: ln(x) = exponent * ln(2) + polynomial_approximation

    // if constexpr (HAS_BASE_SCALING) { ... } -- not taken for natural log

    // Special case: ln(0) = -inf
    v_if (in == 0.0F)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;  // SFPSTORE
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        sfpi::dst_reg++;
    }
}
```

#### 3. Binary Division — Reciprocal + Multiply (Style A: SFPI-based top-level; reciprocal uses TTI internally)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>  // BINOP=DIV, APPROXIMATION_MODE=false
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // x (from DEST tile 1)
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // (1-x) (from DEST tile 0)
        sfpi::vFloat result = 0.0f;

        // For DIV: result = in0 * reciprocal(in1)
        // if constexpr (BINOP == BinaryOp::DIV)
        result = in0 * _sfpu_reciprocal_<2>(in1);  // 2 Newton-Raphson iterations

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;  // Write to DEST tile 0
        sfpi::dst_reg++;
    }
}
```

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>  // max_iter=2 (called from binary DIV)
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x)
{
    // SFPARECIP: hardware approximate reciprocal, ~7-bit precision
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Newton-Raphson refinement (2 iterations for ~16-bit precision):
    // t = x * y - 2.0  (negated for NaN detection)
    sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;  // vConstFloatPrgm0 = 2.0 (set by _init_sfpu_reciprocal_)

    // First NR iteration
    sfpi::vFloat y1 = y * -t - sfpi::vConst0;  // SFPMAD: y * (-t) + 0 = y * (2 - x*y)
    // NaN check: if t >= 0, input was 0 or inf; skip further refinement
    v_if (t < 0)
    {
        // Second NR iteration
        t = x * y1 - sfpi::vConstFloatPrgm0;
        y = y1 * -t - sfpi::vConst0;
    }
    v_endif;

    return y;
}
```

#### 4. Where — Conditional Select (Style B: TTI-based with CC manipulation)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2, const std::uint32_t dst_index_out)
{
    int offset0 = (dst_index_in0 * 32) << 1;  // condition mask tile
    int offset1 = (dst_index_in1 * 32) << 1;  // true-value tile
    int offset2 = (dst_index_in2 * 32) << 1;  // false-value tile

    constexpr std::uint32_t mod0 = data_format == DataFormat::Float16_b
        ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

    // SFPLOADMACRO variant for dst_index_out == dst_index_in0 (3 cycles/row):
    // Uses macros 0 and 2 for superscalar scheduling:
    //
    //   Load Unit                | Simple Unit                     | Store Unit
    //   SFPLOAD L0=Dst[offset0]  |                                 |
    //   SFPLOAD L0=Dst[offset1]  | SFPSETCC LaneEnabled=(L0 EQ 0)  |
    //   SFPLOAD L0=Dst[offset2]  | SFPENCC (LaneEnabled=true)       |
    //   (next SFPLOAD)           |                                 | SFPSTORE Dst[offset0]=L0

    // Otherwise, 4 cycles/row variant with separate output offset.
    // The core logic is:
    //   1. Load condition (L0)
    //   2. Load true-value (L0), simultaneously test: CC <- (previous L0 == 0)
    //   3. Load false-value (L0, CC-guarded: overwrites only where condition was 0)
    //   4. SFPENCC to re-enable all lanes, then store result

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        lltt::replay(0, 3);  // or lltt::replay(0, 4) for non-aliased case
    }
}
```

#### CC State Machine -- `_calculate_where_`

The WHERE kernel has a straightforward CC pattern using SFPSETCC and SFPENCC within a SFPLOADMACRO replay sequence:

```
_calculate_where_ — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPLOAD L0 = Dst[offset0]       (no CC effect) -- load condition mask
       |
       v
  +-------------------------------------+
  | SFPSETCC  mod1=6 (LREG_EQ0)        |
  |   src: LREG0 (condition mask)       |
  |                                     |
  | CC <- (LREG0 == 0)                  |
  |    = (condition is false/zero)      |
  +----------------+--------------------+
                   |
                   v
  CC State: ENABLED where condition == 0 (false lanes)
       |
       |  SFPLOAD L0 = Dst[offset1]      (CC-guarded: loads true-value, but only
       |                                   visible to false-lanes — however this is
       |                                   the "previous" L0 being tested; the new
       |                                   load is unconditional on the Load Unit)
       |
       |  SFPLOAD L0 = Dst[offset2]      (CC-guarded: false-value overwrites L0
       |                                   only where condition was 0; true-value
       |                                   lanes retain the value from offset1)
       |
       v
  +-------------------------------------+
  | SFPENCC  mod1=0 (EU_R1)            |
  |                                     |
  | CC <- ALL_ENABLED                   |
  +----------------+--------------------+
                   |
                   v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE L0 → Dst[offset_out]  (no CC effect) -- store selected value
       |
       v  (iteration complete, advance DEST via ADDR_MOD_6)
```

**Key CC observations:**
- `SFPSETCC` with `LREG_EQ0` mode tests whether the condition mask is zero (i.e., "false"). Lanes where the condition is zero get CC.Res=1 (enabled), allowing the false-value to overwrite.
- The true-value is loaded first (unconditionally by the Load Unit), then the false-value is loaded CC-guarded (only overwrites where condition was 0).
- `SFPENCC` with mode `EU_R1` resets CC to all-enabled before the store.
- The CC state persists across the SFPLOADMACRO replay iterations — each iteration resets CC via SFPENCC at the end.

#### 5. Comparison — Less-Than for float (Style A: SFPI-based)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h

// Used by unary_lt_tile(1, packed_scalar1) via _calculate_comp_unary_ with SfpuType::unary_lt
template <>
inline void apply_unary_float_comp<SfpuType::unary_lt>(sfpi::vFloat v, sfpi::vFloat s, sfpi::vFloat& out_val)
{
    v_if (v < s)     // CC-guarded: lanes where input < scalar
    {
        out_val = ONE;   // 1.0f
    }
    v_else
    {
        out_val = ZERO;  // 0.0f
    }
    v_endif;
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_comp_unary_(std::uint32_t value)  // COMP_MODE=unary_lt
{
    const sfpi::vFloat s = value;  // Scalar loaded from uint32 (reinterpreted as float)

#pragma GCC unroll 8
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];   // Load element from DEST
        sfpi::vFloat val = ZERO;                // Default output = 0.0

        apply_unary_float_comp<COMP_MODE>(v, s, val);  // Set val=1.0 if v < s

        sfpi::dst_reg[0] = val;                 // Store result
        sfpi::dst_reg++;
    }
}
```

### SFPU Instructions Used

| Instruction | Used By | Description |
|-------------|---------|-------------|
| `SFPLOAD` | All sub-ops | Load from DEST register to LREG with format conversion |
| `SFPSTORE` | All sub-ops | Store from LREG to DEST register |
| `SFPMAD` | clamp (via `+`), log (Chebyshev polynomial), recip (Newton-Raphson), binary div (`*`) | Fused multiply-add: `VD = VA × VB + VC`. Used for float addition (a*1.0+b) and multiplication (a*b+0.0) |
| `SFPARECIP` | reciprocal (for div) | Hardware approximate reciprocal, ~7-bit initial estimate |
| `SFPSETEXP` | log | Set exponent field of a float value; used to normalize input to [1,2) range |
| `SFPEXEXP` | log | Extract exponent field from float; used to get the power-of-2 component |
| `SFPNOT` | log (negative exponent path) | Bitwise NOT; used in `~exp` for 2's complement negation |
| `SFPIADD` | log (negative exponent path) | Integer add; used for `~exp + 1` (2's complement negation) |
| `SFPSETSGN` | log (negative exponent path) | Set sign bit; forces negative sign on the negated exponent |
| `SFPCAST` / `int32_to_float` | log | Convert integer exponent to float for the `exp * ln(2)` correction |
| `SFPSETCC` | where, comp (via v_if) | Set condition code for per-lane predicated execution |
| `SFPENCC` | where, comp (via v_endif) | Enable/disable condition code; resets CC to all-enabled |
| `SFPCOMPC` | clamp (via v_elseif), comp (via v_else) | Complement CC for else-branch |
| `SFPPUSHC` | clamp (via v_elseif — nested CC) | Push CC state onto stack for nested conditionals |
| `SFPPOPC` | clamp (via v_endif after elseif) | Pop CC state from stack |
| `SFPLOADI` | reciprocal (init), where (init) | Load 16-bit immediate to LREG; used for constant setup |
| `SFPLOADMACRO` | reciprocal (fast paths), where | Superscalar execution — combines load + simple + mad + store in single issue |
| `SFPCONFIG` | reciprocal (init), where (init) | Configure SFPU macro templates and control registers |
| `SFPOR` | reciprocal (8b path) | Bitwise OR; used for bit manipulation in reciprocal |
| `SFPSHFT` | reciprocal (8b path) | Bit shift; used for sign-bit extraction in reciprocal |
| `SFPSWAP` | reciprocal (24b/FP32 path) | Conditional swap; used for NaN→1.0 clamping via `min(t2, 1.0)` |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[tile 0]** | Primary working tile: holds clamped x, then (1-x), then x/(1-x), then log result |
| **DEST[tile 1]** | Secondary tile: holds copy of clamped x for division numerator; later holds comparison mask (if WHERE active) |
| **DEST[tile 2]** | Backup tile (if WHERE active): holds copy of log result for the false-branch of WHERE |
| **LREG0** | General purpose: loaded values, intermediate results |
| **LREG1** | Used in reciprocal for Newton-Raphson intermediate (`t` or `x` depending on path) |
| **LREG2** | Used in reciprocal (8b path) for sign mask; in FP32 path for error accumulation |
| **LREG3** | Used in reciprocal (FP32 path) for initial estimate `y` |
| **LREG7** | Used in reciprocal (8b path) as indirect register for SFPMAD |
| **vConstFloatPrgm0** | Log: `0.692871f` (ln2); Reciprocal: `2.0f` (Newton-Raphson constant) — **these conflict**, so init order matters: `div_binary_tile_init()` sets it to `2.0f`, then `log_tile_init()` resets it to `0.692871f` |
| **vConstFloatPrgm1** | Log: `0.1058f` (Chebyshev coefficient A') |
| **vConstFloatPrgm2** | Log: `-0.7166f` (Chebyshev coefficient B') |

**Important**: The `vConstFloatPrgm0` register is shared between the reciprocal init (sets to `2.0f`) and the log init (sets to `0.692871f`). In the logit kernel, `div_binary_tile_init()` is called before `log_tile_init()`, so the reciprocal's `_init_sfpu_reciprocal_` sets `vConstFloatPrgm0 = 2.0f`, and then `_init_log_` overwrites it to `0.692871f`. This is safe because the division operation completes before log begins, and the reciprocal's fast paths (8b_3c, 24b_5c) don't use `vConstFloatPrgm0` at runtime — only the legacy `_sfpu_reciprocal_` path does.

### Address Mode Configuration

#### Blackhole Configuration

**For unary sub-operations (clamp, log, comp):**

The `eltwise_unary_sfpu_configure_addrmod` function (in `llk_math_eltwise_unary_sfpu.h`) configures:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | No auto-increment; SFPU kernel manages DEST addressing via `dst_reg++` |

Face advancement between the 4 faces is handled by `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice per face transition (advancing by 16 physical DEST rows = 1 face).

**For binary sub-operations (div):**

The `eltwise_binary_sfpu_configure_addrmod` function (in `llk_math_eltwise_binary_sfpu.h`) configures:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | No auto-increment; SFPU kernel manages DEST addressing |

For BinaryOp::DIV, the `SfpuType::unused` template parameter means no additional ADDR_MOD_6 configuration is done (ADDR_MOD_6 is only configured for mul_int32, typecast, max/min operations). Face advancement is via `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)`.

**For reciprocal (inside div):**

The reciprocal fast paths configure additional ADDR_MODs via their init functions:

| Path | ADDR_MOD_7 | ADDR_MOD_6 | Notes |
|------|------------|------------|-------|
| `_init_reciprocal_fast_7b_` | Used for load (dest.incr=0) | Used for store (dest.incr=2) | SFPLOADMACRO scheduling |
| `_init_reciprocal_fast_8b_3c_` | Used for load (dest.incr=0) | Used for store (dest.incr=2) | SFPLOADMACRO scheduling |
| `_init_reciprocal_fast_24b_5c_` | Used for load (dest.incr=0) | Used for store (dest.incr=2) | SFPLOADMACRO scheduling |

The `ADDR_MOD_6` with `dest.incr = 2` auto-advances the DEST row pointer by 2 physical rows (= 1 sfpi row) after each SFPSTORE, enabling the SFPLOADMACRO pipeline to process consecutive rows without explicit `dst_reg++`.

**For where:**

The WHERE kernel uses the same ADDR_MOD_7 (no-incr for loads) and ADDR_MOD_6 (incr=2 for stores) pattern as the reciprocal, configured by its SFPLOADMACRO init.

#### Wormhole B0 Configuration

The Wormhole B0 configuration is identical in structure but differs in the `_llk_math_eltwise_unary_sfpu_done_` function, which additionally calls `math::set_addr_mod_base()` / `math::clear_addr_mod_base()` and adds a `TTI_STALLWAIT` for SFPU completion. On Blackhole, `ADDR_MOD_6` is also configured with `dest.incr = 2` for `SfpuType::reciprocal` specifically (added vs WH), reflecting the reciprocal fast-path improvements.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determined compute kernel path, approximation mode, and parameterized dispatch for LOGIT
   **Key Findings**: LOGIT uses dedicated `logit_kernel.cpp`, approx_mode=false (default), parameterized type returning empty init/func pair

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Understood how eps parameter controls CLAMP/WHERE defines and runtime arg packing
   **Key Findings**: eps packed as scalar1, (1-eps) as scalar2; CLAMP defined when eps>=0; WHERE defined when eps>0.5

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/logit_kernel.cpp`
   **Reason**: Primary compute kernel — understand full execution flow
   **Key Findings**: Chains clamp→rsub→div→log with optional WHERE; uses cb_tmp0 for intermediate storage

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h`
   **Reason**: Core SFPU implementation for clamp operation
   **Key Findings**: SFPI-based, uses v_if/v_elseif for min/max comparison, adds offset

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h`
   **Reason**: Core SFPU implementation for natural log
   **Key Findings**: 3rd-order Chebyshev polynomial on [1,2) range, exponent extracted separately, special case for ln(0)=-inf

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Core SFPU implementation for binary division
   **Key Findings**: DIV implemented as multiply-by-reciprocal: `in0 * _sfpu_reciprocal_<2>(in1)`

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Reciprocal implementation used by binary DIV
   **Key Findings**: Three precision tiers: 7b (SFPLOADMACRO), 8b (SFPLOADMACRO+NR), 24b (SFPLOADMACRO+multi-NR); selected by APPROX and fp32_dest_acc_en

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h`
   **Reason**: SFPU comparison for unary_lt_tile
   **Key Findings**: SFPI-based v_if comparisons producing 0.0/1.0 output mask

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h`
   **Reason**: SFPU WHERE (conditional select) implementation
   **Key Findings**: Uses SFPLOADMACRO for superscalar execution; CC-guarded selection via SFPSETCC(EQ0)+SFPENCC

10. **File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
    **Reason**: API header for div_binary_tile
    **Key Findings**: Maps to llk_math_eltwise_binary_sfpu_binop_div with BinaryOp::DIV template param

11. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
    **Reason**: API header for log_tile
    **Key Findings**: Maps to llk_math_eltwise_unary_sfpu_log with APPROX and fast_and_approx=false defaults

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Unary SFPU dispatch infrastructure — ADDR_MOD configuration and face iteration
    **Key Findings**: ADDR_MOD_7 set to dest.incr=0; face advancement via inc_dst_addr<8>() twice

13. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Unary SFPU params dispatch — VectorMode loop structure
    **Key Findings**: VectorMode::RC processes all 4 faces; callable-based functor invocation

14. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
    **Reason**: Binary SFPU params dispatch — face iteration with SETRWC
    **Key Findings**: Uses TTI_SETRWC for face advancement instead of inc_dst_addr

15. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
    **Reason**: LLK dispatch for binary SFPU binop (div)
    **Key Findings**: Specialized div template calls calculate_sfpu_binary_div with is_fp32_dest_acc_en param

16. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative hardware model for SFPU instruction semantics and register layout
    **Key Findings**: Stride-2 addressing, SFPMAD for all float arithmetic, CC mechanism details
