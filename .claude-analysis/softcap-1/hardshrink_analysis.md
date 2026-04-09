## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp` (dedicated kernel, does NOT use `eltwise_sfpu.cpp` + `SFPU_OP_CHAIN_0`)
- **SFPU_OP_CHAIN_0 expansion**: Not applicable. Hardshrink uses a **dedicated compute kernel** that directly calls individual SFPU and FPU tile-level APIs (`fill_tile`, `ltz_tile`, `gtz_tile`, `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) instead of being dispatched through the standard `SFPU_OP_CHAIN_0` macro mechanism.
- **Alternative kernel**: `hardshrink_kernel_sfpu.cpp` exists as a variant using `copy_tile`/`add_binary_tile` instead of `binary_dest_reuse_tiles` (likely for different hardware targets or compatibility modes).

#### Architectural Note: Hybrid FPU+SFPU Kernel

Hardshrink is unusual among unary operations because it does **not** have a single SFPU `_calculate_*` function. Instead, the compute kernel implements the operation by composing multiple primitive tile-level operations:

**Formula**: `hardshrink(x, lambda) = x * 1(x + lambda < 0) + x * 1(x - lambda > 0)`

This decomposes into:
1. **SFPU operations**: `fill_tile` (fill DEST tile with lambda), `ltz_tile` (less-than-zero comparison), `gtz_tile` (greater-than-zero comparison)
2. **FPU operations**: `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile` (element-wise arithmetic on full tiles)

The SFPU portion handles the comparison-with-zero logic, while the FPU handles the arithmetic composition.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | Not applicable -- hardshrink uses a dedicated compute kernel, not `SFPU_OP_CHAIN_0`. The SFPU comparison functions (`ltz_tile`, `gtz_tile`) use `APPROX` which resolves to `false` from ComputeConfig. |
| Effective SFPU path | The `_calculate_zero_comp_` and `_calculate_fill_` functions are instantiated with `APPROXIMATION_MODE=false`. Neither function has an `if constexpr (APPROXIMATION_MODE)` branch, so the approximation mode does not affect behavior. | `ckernel_sfpu_comp.h` and `ckernel_sfpu_fill.h` |

### SFPU Abstraction Layers

Hardshrink's SFPU operations (`ltz_tile`, `gtz_tile`, `fill_tile`) each have their own abstraction chain. Since hardshrink uses a dedicated compute kernel (not `eltwise_sfpu.cpp`), the standard single-chain model does not apply. Instead, three SFPU sub-operations are composed:

#### ltz_tile / gtz_tile (comparison with zero)

| Layer | File Path |
|-------|-----------|
| **API Header** | Nuked from current codebase (was `api/compute/eltwise_unary/comp.h`). The API functions `ltz_tile(idst)` and `gtz_tile(idst)` expand via `MATH(...)` to LLK calls. |
| **LLK Dispatch** | Nuked from current codebase. The LLK dispatch used the `SFPU_ZERO_KERNEL` macro from `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (line 205-207), which calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::OP>, DST_IDX, (int)VectorMode::MODE, 8)`. |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

#### fill_tile

| Layer | File Path |
|-------|-----------|
| **API Header** | Nuked from current codebase (was `api/compute/eltwise_unary/fill.h`). The API function `fill_tile(idst, param0)` expands to the LLK fill call. |
| **LLK Dispatch** | Nuked from current codebase. Uses `_llk_math_eltwise_unary_sfpu_params_` with `ckernel::sfpu::_calculate_fill_<APPROXIMATE, 8>` as the callable. |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

The hardshrink compute kernel implements the formula `a * 1(a + lambda < 0) + a * 1(a - lambda > 0)` using a two-pass approach with an intermediate circular buffer (`cb_tmp0`).

**Pass 1** (computes `a * 1(a + lambda < 0)`, stores to `cb_tmp0`):
1. `fill_tile(0, lambda)` -- fills DEST tile 0 with the lambda scalar constant
2. `binary_dest_reuse_tiles<ELWADD>(cb_input, 0, 0)` -- adds input tile to DEST tile 0 (FPU), producing `a + lambda`
3. `ltz_tile(0)` -- SFPU: replaces each element in DEST tile 0 with `1.0` if `(a + lambda) < 0`, else `0.0`
4. `binary_dest_reuse_tiles<ELWMUL>(cb_input, 0, 0)` -- multiplies DEST tile 0 by input tile (FPU), producing `a * 1(a + lambda < 0)`
5. Result packed to `cb_tmp0`

**Pass 2** (computes `a * 1(a - lambda > 0)`, adds Pass 1 result):
1. `fill_tile(0, lambda)` -- fills DEST tile 0 with the lambda scalar constant
2. `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>(cb_input, 0, 0)` -- subtracts DEST tile 0 from input tile (FPU), producing `a - lambda`
3. `gtz_tile(0)` -- SFPU: replaces each element in DEST tile 0 with `1.0` if `(a - lambda) > 0`, else `0.0`
4. `binary_dest_reuse_tiles<ELWMUL>(cb_input, 0, 0)` -- multiplies DEST tile 0 by input tile (FPU), producing `a * 1(a - lambda > 0)`
5. `binary_dest_reuse_tiles<ELWADD>(cb_tmp0, 0, 0)` -- adds the Pass 1 result from `cb_tmp0` (FPU), producing the final `hardshrink(a, lambda)`
6. Result packed to `cb_output`

**SFPU dispatch path for `ltz_tile(0)`**:
`ltz_tile(0)` -> `MATH(llk_math_eltwise_unary_sfpu_ltz<APPROX>(0))` -> `SFPU_ZERO_KERNEL(less_than_zero, RC, APPROX, 0)` -> `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_comp<false, SfpuType::less_than_zero>, 0, VectorMode::RC, 8)` -> iterates over 4 faces calling `_calculate_zero_comp_<false, SfpuType::less_than_zero>(8)` which dispatches to `apply_zero_comp<SfpuType::less_than_zero>`.

**SFPU dispatch path for `gtz_tile(0)`**: Same chain but with `SfpuType::greater_than_zero`.

**SFPU dispatch path for `fill_tile(0, lambda)`**:
`fill_tile(0, lambda)` -> `MATH(llk_math_eltwise_unary_sfpu_fill<APPROX>(0, lambda))` -> `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::_calculate_fill_<false, 8>, 0, VectorMode::RC, lambda)` -> iterates over 4 faces calling `_calculate_fill_<false, 8>(lambda)`.

### Parameters Dispatch Summary

The parameters dispatch for all three SFPU sub-operations uses `_llk_math_eltwise_unary_sfpu_params_` from `llk_math_eltwise_unary_sfpu_params.h`:

- **Vector mode**: `VectorMode::RC` for all SFPU sub-operations in hardshrink. This processes all 4 faces of the tile (full 32x32 = 1024 elements).
- **Operation invocation**: The params dispatch iterates over 4 faces in a `for (int face = 0; face < 4; face++)` loop, calling the SFPU function once per face (with `ITERATIONS=8` inside the function). Between faces, `TTI_SETRWC` advances the DEST write pointer by 2 face-strides (2 x `SETRWC(CR_D, 8)`).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the SFPU init configures `ADDR_MOD_7` with all-zero increments (`{.srca={.incr=0}, .srcb={.incr=0}, .dest={.incr=0}}`), and `set_addr_mod_base()` shifts to use address mods 4-7 (so the default SFPU ADDR_MOD maps to physical ADDR_MOD_7). The `dst_reg++` in the SFPI loop handles per-iteration advancement, while `SETRWC` handles per-face advancement.

### Annotated SFPU Kernel Source

#### fill_tile SFPU Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h

template <bool APPROXIMATION_MODE, int ITERATIONS> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void _calculate_fill_(const float value)
{
    // SFPU microcode
    sfpi::vFloat fill_val = value; // SFPLOADI: loads float constant into LREG as vFloat

    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val; // SFPSTORE: writes fill_val to current DEST row pair (32 elements)
        sfpi::dst_reg++;             // advances DEST pointer by 1 sfpi row (2 physical rows)
    }
}
```

#### ltz_tile / gtz_tile SFPU Implementation (comparison with zero)

The `_calculate_zero_comp_` template function dispatches to `apply_zero_comp<SfpuType::OP>` specializations. For hardshrink, two specializations are used: `less_than_zero` (for `ltz_tile`) and `greater_than_zero` (for `gtz_tile`).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void _calculate_zero_comp_(std::uint32_t exponent_size_8) // APPROXIMATION_MODE=false, ITERATIONS=8
{
    for (int d = ZERO; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: read current DEST row pair into vFloat
        apply_zero_comp<COMP_MODE>(v, exponent_size_8); // apply comparison, sets v to 0.0 or 1.0
        sfpi::dst_reg[0] = v; // SFPSTORE: write result back to DEST
        sfpi::dst_reg++;      // advance DEST pointer
    }
}

// Specialization for ltz_tile: less_than_zero
// Sets each element to 1.0 if element < 0, else 0.0
template <>
inline void apply_zero_comp<SfpuType::less_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v >= ZERO) // SFPENCC + SFPSETCC(GTE0): enable CC, test v >= 0
    {
        v = ZERO;    // SFPLOADI + SFPSTORE (guarded): set to 0.0 for non-negative elements
    }
    v_else           // SFPCOMPC: complement CC for else branch
    {
        v = ONE;     // SFPLOADI + SFPSTORE (guarded): set to 1.0 for negative elements
    }
    v_endif;         // SFPENCC: disable CC masking
}

// Specialization for gtz_tile: greater_than_zero
// Sets each element to 1.0 if element > 0, else 0.0
template <>
inline void apply_zero_comp<SfpuType::greater_than_zero>(sfpi::vFloat& v, std::uint32_t /*unused*/)
{
    v_if (v > ZERO)  // SFPENCC + SFPSETCC: enable CC, test v > 0 (sign bit check + zero check)
    {
        v = ONE;     // SFPLOADI + SFPSTORE (guarded): set to 1.0 for positive elements
    }
    v_else           // SFPCOMPC: complement CC for else branch
    {
        v = ZERO;    // SFPLOADI + SFPSTORE (guarded): set to 0.0 for non-positive elements
    }
    v_endif;         // SFPENCC: disable CC masking
}
```

**Note on `v > ZERO` vs `v >= ZERO`**: The `less_than_zero` specialization uses `v >= ZERO` (then assigns 0 for the true branch), which is logically equivalent to "if not less than zero, set 0; else set 1". The `greater_than_zero` specialization uses `v > ZERO` directly. The `>` comparison on vFloat requires checking both the sign bit AND that the value is non-zero, which compiles to `SFPSETCC(GTE0)` followed by a zero check using `SFPSETCC(NE0)` combined via `SFPPUSHC/SFPPOPC` AND logic, or uses the SFPI compiler's optimized CC pipeline.

### SFPU Instructions Used

| Instruction | Description | Used By |
|-------------|-------------|---------|
| `SFPLOAD` | Load DEST row pair into LREG (vFloat). Implicit via `sfpi::dst_reg[0]` read. | `_calculate_zero_comp_` (comp), reads tile data for comparison |
| `SFPSTORE` | Store LREG value back to DEST row pair. Implicit via `sfpi::dst_reg[0] = ...` write. | `_calculate_zero_comp_` (comp), `_calculate_fill_` (fill) |
| `SFPLOADI` | Load 16-bit immediate into LREG. Implicit via `vFloat fill_val = value` and `v = ZERO` / `v = ONE` constant assignments. | `_calculate_fill_`, `apply_zero_comp` specializations |
| `SFPMAD` | Fused multiply-add. Implicit via `vFloat = float_constant` when the constant requires FP32 conversion (e.g., `1.0f` loaded as `SFPMAD(1.0, 1.0, 0.0)`). | Constant loading in both comp and fill |
| `SFPSETCC` | Set condition code based on comparison. Emitted by `v_if (v >= ZERO)` (mode `LREG_GTE0`, sign bit test) and `v_if (v > ZERO)` (combined sign+zero test). | `apply_zero_comp<less_than_zero>`, `apply_zero_comp<greater_than_zero>` |
| `SFPENCC` | Enable/disable condition code masking. Emitted at `v_if` entry (enable CC) and `v_endif` (disable CC). | All `apply_zero_comp` specializations |
| `SFPCOMPC` | Complement CC for else-branch. Emitted by `v_else`. | All `apply_zero_comp` specializations |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST tile 0** | Primary working tile. Holds the intermediate computation results at each stage (lambda, a+lambda, indicator, a*indicator, etc.). SFPU operations read from and write to this tile. |
| **DEST tile 1** | In `hardshrink_kernel_sfpu.cpp` variant, used to hold a copy of the input tile for multiplication after the comparison step. In `hardshrink_kernel.cpp`, the `binary_dest_reuse_tiles` mechanism avoids needing a second DEST tile. |
| **LREG0-LREG3** | General purpose LREGs used implicitly by SFPI abstractions. `vFloat v` and `vFloat fill_val` map to LREGs for intermediate computation. The SFPI compiler allocates these automatically. |
| **cb_tmp0 (c_1)** | Circular buffer used as intermediate storage between Pass 1 and Pass 2. Holds the `a * 1(a + lambda < 0)` result tile, which is later added to the Pass 2 result. |

### Address Mode Configuration

The SFPU init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType>` (from `llk_math_eltwise_unary_sfpu.h`) configures `ADDR_MOD_7` for all standard unary SFPU operations:

**Wormhole B0**:
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

**Blackhole**: Same configuration in `_llk_math_eltwise_unary_sfpu_init_` (identical `ADDR_MOD_7` with all-zero increments).

The `ADDR_MOD_7` with zero increments means the hardware does not auto-increment DEST addresses between SFPU instructions. Instead, address progression is managed explicitly:
- **Within a face**: `dst_reg++` in the SFPI loop advances by 1 sfpi row (2 physical DEST rows, 32 elements) per iteration.
- **Between faces**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice between faces in the `_llk_math_eltwise_unary_sfpu_params_` dispatch, advancing by 16 physical rows (1 face height).

The `set_addr_mod_base()` call at the start of `_llk_math_eltwise_unary_sfpu_params_` shifts the address mode base to use mods 4-7, so the SFPI code's default address mode (logical 3) maps to physical `ADDR_MOD_7`.

Note: For `ltz_tile` and `gtz_tile` specifically, the comparison SfpuType values (`less_than_zero`, `greater_than_zero`) do not trigger any special address mode configuration in `eltwise_unary_sfpu_configure_addrmod<SfpuType>` -- they fall through to the default `ADDR_MOD_7` with zero increments.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path and approximation mode for HARDSHRINK
   **Key Findings**: `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` by default (HARDSHRINK's dedicated kernel path was nuked). `get_op_approx_mode()` returns `false` for all ops (default case only).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Understand how HARDSHRINK is dispatched, including parameter passing and circular buffer setup
   **Key Findings**: HARDSHRINK gets a `cb_tmp0` (c_1) circular buffer for intermediate results. The `packed_scalar1` is set from `pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype())` to pass the lambda parameter.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel.cpp`
   **Reason**: Read the primary dedicated compute kernel for hardshrink
   **Key Findings**: Implements `a * 1(a+lambda<0) + a * 1(a-lambda>0)` using `binary_dest_reuse_tiles` for FPU operations and `ltz_tile`/`gtz_tile` for SFPU comparisons. Two-pass approach with `cb_tmp0` as intermediate storage.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardshrink_kernel_sfpu.cpp`
   **Reason**: Read the alternative compute kernel variant
   **Key Findings**: Same algorithm but uses `copy_tile`/`add_binary_tile`/`sub_binary_tile`/`mul_binary_tile` instead of `binary_dest_reuse_tiles`. Likely for compatibility or different hardware targets.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_comp.h`
   **Reason**: Core SFPU implementation for comparison-with-zero operations (ltz, gtz)
   **Key Findings**: `_calculate_zero_comp_<APPROX, SfpuType::OP>` iterates 8 times per face, loading from dst_reg, applying `apply_zero_comp<SfpuType>`, and writing back. The `less_than_zero` specialization checks `v >= 0` and sets 0/1 accordingly. The `greater_than_zero` specialization checks `v > 0`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_fill.h`
   **Reason**: Core SFPU implementation for fill_tile
   **Key Findings**: `_calculate_fill_<APPROX, ITERATIONS>` loads a float constant into a vFloat register and writes it to all 8 dst_reg positions per face. Simple and efficient.

7. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Find the SFPU dispatch macros for comparison-with-zero operations
   **Key Findings**: `SFPU_ZERO_KERNEL(OP, MODE, APPROXIMATE, DST_IDX)` dispatches to `calculate_comp<APPROXIMATE, SfpuType::OP>` via `_llk_math_eltwise_unary_sfpu_params_`. This is the macro used by `ltz_tile` and `gtz_tile`.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch layer that manages face iteration and DEST addressing
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode dispatch (RC/R/C), iterates over faces calling the SFPU function, uses `TTI_SETRWC` for inter-face DEST advancement, and brackets with `STALL_SFPU`/`WAIT_SFPU` synchronization.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand SFPU init and address mode configuration
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_init_` configures `ADDR_MOD_7` with all-zero increments for standard unary SFPU ops. No special address mode for comparison types.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and addressing
    **Key Findings**: SFPU stride-2 model (each dst_reg access = 32 elements), ITERATIONS=8 per face, 4 faces per tile = 32 total iterations = 1024 elements. SFPSETCC modes for sign/zero tests. CC mechanism for predicated execution.

11. **File**: `docs/sfpu_operations/key_notes/hardshrink_key_notes.md`
    **Reason**: Understand the mathematical definition and parameters
    **Key Findings**: `hardshrink(x) = x if |x| > lambda, 0 otherwise`. Default lambda=0.5.
