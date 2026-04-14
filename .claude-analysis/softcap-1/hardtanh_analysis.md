## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` -- however, HARDTANH dispatch is **incomplete** in this codebase; see note below)
- **SFPU_OP_CHAIN_0 expansion**: **Not wired** -- `get_op_init_and_func_parameterized()` throws `TT_THROW("unexpected parameterized op type {}", op_type)` for HARDTANH because there is no case for it in the switch statement.

**Dispatch Chain Status**: The HARDTANH operation has a `UnaryOpType::HARDTANH` enum value, a Python-level binding (`ttnn.hardtanh`), and a ckernel SFPU implementation (`_calculate_hardtanh_`), but the TTNN-to-compute-kernel dispatch chain is incomplete. Specifically:
- The API header (`hardtanh_tile`, `hardtanh_tile_init`) does **not exist** in `tt_metal/hw/inc/api/compute/eltwise_unary/`.
- The LLK dispatch layer (`llk_math_eltwise_unary_sfpu_hardtanh.h`) does **not exist** in either Wormhole or Blackhole `llk_lib/`.
- The `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp` has no case for `UnaryOpType::HARDTANH`.
- Calling `ttnn.hardtanh()` at runtime would fail at program compilation time when `get_block_defines()` attempts to resolve the SFPU_OP_CHAIN_0 macro.

The analysis below documents the ckernel SFPU implementation that **would** be invoked if the dispatch chain were completed.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (dispatch not wired) | `get_op_init_and_func()` has no case for HARDTANH; if wired, the `_calculate_hardtanh_` template takes `APPROXIMATION_MODE` but does not use it (no `if constexpr (APPROXIMATION_MODE)` branch) |
| Effective SFPU path | Single code path regardless of approximation mode | The `_calculate_hardtanh_` function template accepts `APPROXIMATION_MODE` but has no conditional branch on it -- the same clamping logic executes regardless |

### SFPU Abstraction Layers
List the file path for each abstraction layer. If a layer does not exist for this operation, write "This level of abstraction doesn't exist" instead of a path.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (no `hardtanh_tile` / `hardtanh_tile_init` in `tt_metal/hw/inc/api/compute/eltwise_unary/`) |
| **LLK Dispatch** | This level of abstraction doesn't exist (no `llk_math_eltwise_unary_sfpu_hardtanh.h` in `llk_lib/`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical on Blackhole: `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic dispatch -- would be used if HARDTANH were wired through the standard unary pipeline) |

### Call Chain
The HARDTANH dispatch chain is incomplete. If it were fully wired, the expected call chain would be:

1. `hardtanh_tile(idst, param0, param1)` (API header, does not exist) would call `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)`.
2. The LLK dispatch function (does not exist) would call `_llk_math_eltwise_unary_sfpu_params_<APPROX>(_calculate_hardtanh_<APPROX, 8>, dst_index, VectorMode::RC, 8, param0, param1, param2)`.
3. The generic params dispatch (`llk_math_eltwise_unary_sfpu_params.h`) would set up DEST addressing, stall for SFPU readiness, then loop over 4 faces calling `_calculate_hardtanh_<APPROX, 8>(8, param0, param1, param2)` per face.
4. The core SFPU function (`ckernel_sfpu_hardtanh.h`) would execute the clamping logic on each face's 8 SFPU iterations.

### Parameters Dispatch Summary

- **Vector mode**: Would use `VectorMode::RC` (all 4 faces processed), consistent with standard unary operations that operate on every element of the tile.
- **Operation invocation**: The generic `_llk_math_eltwise_unary_sfpu_params_` template calls the SFPU function once per face (4 calls total for RC mode), with `ITERATIONS=8` per call, covering all 32 sfpi rows of the tile.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces). ADDR_MOD_7 is configured with `dest.incr=0` on both Wormhole and Blackhole -- the `dst_reg++` within the SFPI kernel handles per-iteration advancement internally, and the params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (which calls `math::inc_dst_addr<8>()` twice, advancing by 16 physical DEST rows = 1 face stride) between faces.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (Style A). The Wormhole and Blackhole implementations are byte-for-byte identical.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{   // APPROXIMATION_MODE is unused (no conditional branch), ITERATIONS=8
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load -(neg_threshold) as FP16_B immediate -> LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load -(pos_threshold - neg_threshold) as FP16_B immediate -> LREG
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load -(pos_threshold) as FP16_B immediate -> LREG
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        val += p0; // SFPMAD: val = val * 1.0 + p0 (i.e., val = x + (-(neg_threshold)) = x - neg_threshold)
        v_if (val < 0.0f) // SFPENCC + SFPSETCC(LT0): enable CC, test if (x - neg_threshold) < 0, i.e., x < neg_threshold
        {
            val = 0.0f; // SFPLOADI: masked store of 0.0 to val (only on lanes where x < neg_threshold)
        }
        v_endif; // SFPPOPC + SFPENCC: restore CC state, all lanes active again

        val += p1; // SFPMAD: val = val * 1.0 + p1 (add -(pos_threshold - neg_threshold))
        v_if (val >= 0.0f) // SFPENCC + SFPSETCC(GTE0): test if result >= 0 (i.e., x > pos_threshold after offset)
        {
            val = 0.0f; // SFPLOADI: masked store of 0.0 (only on lanes where x > pos_threshold)
        }
        v_endif; // SFPPOPC + SFPENCC: restore CC state

        val += p2; // SFPMAD: val = val * 1.0 + p2 (add -(pos_threshold) to reconstruct clamped value)

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 elements back to DEST row pair

        sfpi::dst_reg++; // advance to next sfpi row (2 physical DEST rows, 32 elements)
    }
}
```

**Algorithm explanation**: The hardtanh function clamps values to the range `[neg_threshold, pos_threshold]`. Instead of using comparisons against the thresholds directly, this implementation uses an arithmetic trick with three negated offset parameters:

1. `val = x - neg_threshold`: Shift so that the lower threshold maps to 0. If `val < 0`, then `x < neg_threshold`, so clamp to 0.
2. `val = val - (pos_threshold - neg_threshold)`: Shift so that the upper threshold maps to 0. If `val >= 0`, then `x > pos_threshold`, so clamp to 0.
3. `val = val - pos_threshold`: Undo the shift to reconstruct the original clamped value.

For the default parameters (`min_val=-1.0`, `max_val=1.0`): `param0 = -(-1.0) = 1.0`, `param1 = -(1.0 - (-1.0)) = -2.0`, `param2 = -(1.0) = -1.0`.

### SFPU Instructions Used

| Instruction | Count per iteration | Description |
|-------------|-------------------|-------------|
| `SFPLOADI` | 3 (before loop) + 2 (conditional `val=0.0f`) | Load 16-bit immediate into LREG. Used for parameter loading (FP16_B format) and for loading the constant `0.0f` inside conditional blocks. |
| `SFPLOAD` | 1 | Load 32 elements from DEST row pair into LREG. Used for `dst_reg[0]` read. |
| `SFPMAD` | 3 | Fused multiply-add: `VD = VA * VB + VC`. Used for float addition (`val += p`) as `val * 1.0 + p`. |
| `SFPSETCC` | 2 | Set CC.Res based on LREG comparison. First use: `SFPSETCC_MOD1_LREG_LT0` for `val < 0.0f`. Second use: `SFPSETCC_MOD1_LREG_GTE0` for `val >= 0.0f`. |
| `SFPENCC` | 4 | Enable/disable CC masking. Two pairs: one for each `v_if`/`v_endif` block (enable CC at `v_if`, disable at `v_endif`). |
| `SFPPUSHC` | 2 | Push CC state onto stack at each `v_if` entry. |
| `SFPPOPC` | 2 | Pop CC state from stack at each `v_endif`. |
| `SFPSTORE` | 1 | Store 32 elements from LREG back to DEST row pair. Used for `dst_reg[0] = val`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (p0)** | Holds `-(neg_threshold)` as vFloat, loaded once before the loop via `SFPLOADI` in FP16_B format. Reused across all iterations. |
| **LREG (p1)** | Holds `-(pos_threshold - neg_threshold)` as vFloat, loaded once before the loop via `SFPLOADI` in FP16_B format. Reused across all iterations. |
| **LREG (p2)** | Holds `-(pos_threshold)` as vFloat, loaded once before the loop via `SFPLOADI` in FP16_B format. Reused across all iterations. |
| **LREG (val)** | Working register for the current element values. Loaded from DEST at loop start, modified through additions and conditional zeroing, stored back to DEST at loop end. |
| **DEST rows** | Source and destination for tile data. Each iteration processes 1 sfpi row = 2 physical DEST rows = 32 elements. The loop advances through 8 sfpi rows per face call (ITERATIONS=8). |
| **CC register** | Per-lane condition code used for the two `v_if` blocks. Each `v_if` enables CC, tests the condition, masks lanes, then `v_endif` restores the CC state. |
| **CC stack** | 1 entry deep (one `v_if` nesting level at a time). Each `v_if` pushes, each `v_endif` pops. The two `v_if` blocks are sequential, not nested. |

### Address Mode Configuration

The standard unary SFPU address mode would be used, configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType>()` in `llk_math_eltwise_unary_sfpu.h`. Since HARDTANH is not one of the special-cased `SfpuType` values, it would use the default configuration:

**ADDR_MOD_7** (both Wormhole and Blackhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

All address increments are zero because the SFPU kernel manages its own DEST addressing internally via `dst_reg++` (which increments the SFPU's internal DEST pointer by 1 sfpi row = 2 physical DEST rows per iteration). The `ADDR_MOD_7` setting is a no-op for address progression -- it avoids conflicting with ADDR_MOD_0 and ADDR_MOD_2 used by the A2D (Accumulate-to-DEST) pipeline that runs concurrently.

The Wormhole and Blackhole configurations are identical for this operation. The only difference is that Wormhole's `_llk_math_eltwise_unary_sfpu_start_` additionally calls `math::set_addr_mod_base()`, while Blackhole omits this (it is handled differently on Blackhole). Similarly, `_llk_math_eltwise_unary_sfpu_done_` on Wormhole calls `math::clear_addr_mod_base()` which Blackhole omits.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Traced HARDTANH dispatch through `get_op_approx_mode()`, `get_op_init_and_func_parameterized()`, `get_compute_kernel_path()`, and `get_block_defines()`.
   **Key Findings**: HARDTANH is marked as parametrized (`is_parametrized_type` returns true) but has no case in `get_op_init_and_func_parameterized` -- the dispatch would throw at runtime. `get_op_approx_mode` returns false (default case). `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default case).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Confirmed `is_parametrized_type(UnaryOpType::HARDTANH)` returns true, and checked API signatures.
   **Key Findings**: HARDTANH is listed alongside SOFTSHRINK as a parametrized type.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Examined the Python-facing `hardtanh()` function to understand parameter passing.
   **Key Findings**: Takes `min_val=-1.0f` and `max_val=1.0f` defaults, constructs `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation for HARDTANH. Primary analysis target.
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Implements clamping via 3-step arithmetic offset pattern with negated threshold parameters. Template parameter APPROXIMATION_MODE is accepted but unused.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Blackhole variant of the SFPU implementation.
   **Key Findings**: Byte-for-byte identical to the Wormhole version.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Generic LLK dispatch layer for unary SFPU operations. Contains `eltwise_unary_sfpu_configure_addrmod`, `_llk_math_eltwise_unary_sfpu_init_`, and related setup functions.
   **Key Findings**: ADDR_MOD_7 is configured with all zero increments (default for most SFPU operations). Special-cased operations (topk_local_sort, typecast, etc.) configure ADDR_MOD_6 with custom increments.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Generic params dispatch layer that handles VectorMode and face iteration.
   **Key Findings**: VectorMode::RC processes all 4 faces; each face call invokes the SFPU function once, followed by SETRWC to advance DEST address by one face stride.

8. **File**: `runtime/sfpi/include/sfpi_fp16.h`
   **Reason**: Defines `s2vFloat16b` class for scalar-to-vector FP16_B conversion.
   **Key Findings**: `s2vFloat16b(uint32_t)` stores the raw uint32 value as-is and tags it as FP16_B format. When assigned to a `vFloat`, this emits an `SFPLOADI` instruction with `SFPLOADI_MOD0_FLOATB` format.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Defines `v_if`/`v_endif` macros for SFPI condition code management.
   **Key Findings**: `v_if(x)` expands to `cc_push().cc_if().cc_cond(x)` (SFPPUSHC + SFPENCC + SFPSETCC), and `v_endif` triggers `__vCCCtrl` destructor which pops the CC stack (SFPPOPC + SFPENCC).

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU instruction semantics, CC mechanism, and addressing model.
    **Key Findings**: SFPMAD is used for float addition (no dedicated add instruction), SFPSETCC modes LT0 and GTE0 for sign-based comparisons, CC stack for nested/sequential conditional blocks.

11. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
    **Reason**: Confirmed how `get_block_defines()` is called and how packed scalars are passed.
    **Key Findings**: HARDTANH is not special-cased for packed scalar handling (only HARDSHRINK and WHERE_TSS are). The runtime args `packed_scalar1` and `packed_scalar2` default to 0 for HARDTANH.
