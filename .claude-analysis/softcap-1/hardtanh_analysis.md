## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile(0, param0, param1, param2)` (expected when fully wired -- see note below)

**Integration Status Note**: The SFPU kernel implementation (`_calculate_hardtanh_`) exists in both Wormhole and Blackhole LLK variants, and both `UnaryOpType::HARDTANH` and `SfpuType::Hardtanh` enums are defined. However, the following integration layers are not yet wired in this codebase snapshot:
- `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp` has no HARDTANH case (will TT_THROW at runtime)
- No compute API header (`hardtanh.h`) exists under `tt_metal/hw/inc/api/compute/eltwise_unary/`
- No metal LLK API header (`llk_math_eltwise_unary_sfpu_hardtanh.h`) exists under `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/`
- No entry in `sfpu_split_includes.h` for `SFPU_OP_HARDTANH_INCLUDE`

The analysis below focuses on the **core SFPU kernel function** which is complete and ready for integration.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (not yet wired) | `get_op_init_and_func()` -- no HARDTANH case exists yet. When wired, the `_calculate_hardtanh_` template takes `APPROXIMATION_MODE` but does not use it (no conditional branches depend on it) |
| Effective SFPU path | Same code path regardless of approximation mode | `_calculate_hardtanh_` has `APPROXIMATION_MODE` as a template parameter but contains no `if constexpr (APPROXIMATION_MODE)` branches -- the algorithm is identical in both modes |

### SFPU Abstraction Layers
The table below lists the file path for each abstraction layer. Since hardtanh is not yet fully wired into the metal compute API, the API Header and LLK Dispatch layers do not exist yet.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (expected: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`) |
| **LLK Dispatch** | This level of abstraction doesn't exist (expected: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical on Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
When fully wired, the expected call chain would be:

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile_init(); hardtanh_tile(0, param0, param1, param2);`
2. **API Header** (`hardtanh.h` -- not yet created): `hardtanh_tile(idst, p0, p1, p2)` would call `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, p0, p1, p2)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardtanh.h` -- not yet created): Would call `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>, dst_index, (int)VectorMode::RC, p0, p1, p2)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Iterates over 4 faces in `VectorMode::RC`, calling `_calculate_hardtanh_<APPROX, 8>(8, p0, p1, p2)` for each face and advancing the DEST address between faces via `SETRWC`.
5. **Core SFPU** (`ckernel_sfpu_hardtanh.h`): `_calculate_hardtanh_` executes the algebraic clamping algorithm over 8 iterations per face.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (expected) -- all 4 faces of the tile are processed (standard for eltwise unary operations).
- **Operation invocation**: The parameters dispatch function (`_llk_math_eltwise_unary_sfpu_params_`) loops over 4 faces. For each face, it calls the SFPU function once with `iterations=8`, then advances the DEST address to the next face. The 3 runtime parameters (`param0`, `param1`, `param2`) are forwarded as `Args&&...` to the core function on each invocation.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole: `TTI_SETRWC` with `CR_D, 8` called twice per face (advancing by 16 physical rows = 1 face). On Blackhole: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice. The address mode configured is `ADDR_MOD_7` with `dest.incr = 0` (DEST auto-increment is handled explicitly by the SFPI `dst_reg++` within the kernel loop and by the inter-face SETRWC instructions, not by the hardware address mode auto-increment).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
// (Identical on Blackhole: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{ // APPROXIMATION_MODE is unused (no conditional branches), ITERATIONS=8
    // All params are in FP16_B format (pre-computed by host)
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)
    // NOTE: see "Algorithm Explanation" below -- the comment for param2 appears
    // to be a documentation bug; algebraic analysis shows param2 must equal
    // pos_threshold (not negated) for correct clamping behavior.

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load param0 as FP16_B scalar, broadcast to all lanes
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load param1 as FP16_B scalar, broadcast to all lanes
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load param2 as FP16_B scalar, broadcast to all lanes
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face, 32 elements per iteration
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        val += p0; // SFPMAD (val * 1.0 + p0): shifts value by -(neg_threshold), so below-threshold values become negative
        v_if (val < 0.0f) // SFPSETCC(LT0) + SFPPUSHC: CC lanes where (val - neg_threshold) < 0, i.e., input below neg_threshold
        {
            val = 0.0f; // SFPLOADI: zero lanes below neg_threshold (bottom clamp)
        }
        v_endif; // SFPPOPC: restore CC state, all lanes active again

        val += p1; // SFPMAD (val * 1.0 + p1): shifts by -(pos_threshold - neg_threshold), so above-threshold values become non-negative
        v_if (val >= 0.0f) // SFPSETCC(GTE0) + SFPPUSHC: CC lanes where shifted value >= 0, i.e., input above pos_threshold
        {
            val = 0.0f; // SFPLOADI: zero lanes above pos_threshold (top clamp)
        }
        v_endif; // SFPPOPC: restore CC state, all lanes active again

        val += p2; // SFPMAD (val * 1.0 + p2): final offset correction to produce the clamped output value

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 clamped elements back to current DEST row pair

        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

**Algorithm Explanation -- Algebraic Clamping via Three Additions**:

The hardtanh function clamps values to `[neg_threshold, pos_threshold]`: `hardtanh(x) = min(pos, max(neg, x))`. Instead of using direct comparisons (as `_calculate_clamp_` does), this kernel uses an algebraic trick with three additions and two conditional zeroing operations.

Let `neg` = `neg_threshold`, `pos` = `pos_threshold`. The required parameter values for correctness are:
- `p0 = -neg` (shifts input so values below neg become negative)
- `p1 = neg - pos` (further shifts so values above pos become non-negative)
- `p2 = pos` (offsets back to produce the correct output value)

Verification (these sum to zero: `p0 + p1 + p2 = -neg + neg - pos + pos = 0`):

| Input Region | After `val += p0` | After 1st v_if | After `val += p1` | After 2nd v_if | After `val += p2` | Result |
|---|---|---|---|---|---|---|
| `x < neg` | `x - neg < 0` | `val = 0` | `0 + (neg-pos) < 0` | stays | `neg - pos + pos` | `neg` |
| `neg <= x <= pos` | `x - neg >= 0` | stays | `x - neg + neg - pos = x - pos < 0` | stays | `x - pos + pos` | `x` |
| `x > pos` | `x - neg > 0` | stays | `x - neg + neg - pos = x - pos >= 0` | `val = 0` | `0 + pos` | `pos` |

**Source Code Comment Discrepancy**: The source code comments state `param2 = -(pos_threshold)`, but algebraic analysis proves that `param2` must equal `pos_threshold` (positive) for the algorithm to produce correct results. The host-side parameter preprocessing (when fully wired) must set `param2 = pos_threshold`, not `-(pos_threshold)`. The comments for `param0` and `param1` are correct.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOADI` | `sfpi::s2vFloat16b(param)` | Load a 16-bit FP16_B immediate value into an LREG, broadcast to all SFPU lanes. Used 3 times for parameter loading (p0, p1, p2) and 2 times for `val = 0.0f` conditional zeroing |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements (2 physical DEST rows) from current DEST address into an LREG |
| `SFPMAD` | `val += pN` (vFloat + vFloat) | Fused multiply-add: `val = val * 1.0 + pN`. Used 3 times for the three addition steps. There is no dedicated float add instruction; addition is performed via `SFPMAD` with an implicit multiply by 1.0 |
| `SFPSETCC` | `v_if (val < 0.0f)`, `v_if (val >= 0.0f)` | Set per-lane condition code bits based on comparison. LT0 mode for `< 0` check, GTE0 mode for `>= 0` check |
| `SFPPUSHC` | `v_if` (implicit) | Push current CC state onto the per-lane CC stack before entering conditional block |
| `SFPPOPC` | `v_endif` (implicit) | Pop CC state from stack, restoring pre-conditional state (all lanes active) |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` (write) | Store 32 elements from LREG back to current DEST row pair |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. Accessed via `dst_reg[0]` which maps to the current DEST row pair (2 physical rows, 32 elements). Each iteration processes one row pair, advancing by `dst_reg++`. |
| **LREG (val)** | Holds the working value being clamped. Loaded from DEST at iteration start, modified through 3 additions and 2 conditional zeroing steps, then stored back to DEST. |
| **LREG (p0)** | Holds `-(neg_threshold)` as a broadcast scalar. Loaded once before the iteration loop via `SFPLOADI` and reused across all 8 iterations per face. |
| **LREG (p1)** | Holds `-(pos_threshold - neg_threshold)` as a broadcast scalar. Same lifecycle as p0. |
| **LREG (p2)** | Holds `pos_threshold` as a broadcast scalar. Same lifecycle as p0. Note: see comment discrepancy noted above. |
| **CC bits** | Per-lane condition code (CC.En + CC.Res). Used within each `v_if`/`v_endif` block to mask which lanes execute the conditional zeroing. Two independent CC regions per iteration: one for the bottom clamp (LT0) and one for the top clamp (GTE0). |
| **CC stack** | Per-lane 8-entry CC stack. One push/pop pair per `v_if`/`v_endif` block (2 push/pop pairs per iteration). Stack depth never exceeds 1 since the two conditionals are sequential, not nested. |

### Address Mode Configuration

The address mode for hardtanh is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::Hardtanh>()` during SFPU initialization. Since `SfpuType::Hardtanh` does not appear in any `if constexpr` special case within `eltwise_unary_sfpu_configure_addrmod`, only the default `ADDR_MOD_7` is set:

**Both Wormhole and Blackhole** (identical configuration):
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This means the hardware address mode does NOT auto-increment the DEST address. Instead, DEST address progression is managed explicitly by:
1. **Within a face**: `sfpi::dst_reg++` in the kernel loop body advances the SFPI DEST pointer by 1 sfpi row (= 2 physical DEST rows) per iteration.
2. **Between faces**: The parameters dispatch function (`_llk_math_eltwise_unary_sfpu_params_`) issues `SETRWC` instructions (Wormhole) or `math::inc_dst_addr<8>()` calls (Blackhole) to advance by one full face (16 physical rows) between face invocations.

This is the standard pattern for most unary SFPU operations that do not require special DEST addressing (unlike typecast/signbit which use `ADDR_MOD_6` with `dest.incr = 2`).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and init/func dispatch for HARDTANH
   **Key Findings**: `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default case). `get_op_approx_mode` returns `false` (default case). No HARDTANH case in `get_op_init_and_func_parameterized` -- not yet wired.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check if HARDTANH is parametrized
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns `true`.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel source for Wormhole
   **Key Findings**: SFPI-based kernel using algebraic clamping with 3 params (all FP16_B), 2 conditional zeroing blocks, and `dst_reg` iteration. Identical to Blackhole variant.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel source for Blackhole
   **Key Findings**: Byte-for-byte identical to Wormhole variant.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: LLK init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod` configures `ADDR_MOD_7` with `dest.incr=0`. No special case for Hardtanh.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function -- how the SFPU kernel is called per face
   **Key Findings**: `VectorMode::RC` processes 4 faces, calling SFPU function once per face with forwarded args, SETRWC between faces.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h`
   **Reason**: SfpuType enum definition
   **Key Findings**: `ActivationType::Hardtanh = 3`.

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h`
   **Reason**: LLK SFPU initialization template
   **Key Findings**: `llk_math_eltwise_unary_sfpu_init<sfpu_op, APPROXIMATE>()` calls `_llk_math_eltwise_unary_sfpu_init_<sfpu_op>()`.

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: TTNN-level hardtanh function signature and parameter passing
   **Key Findings**: `hardtanh(input, min_val=-1.0f, max_val=1.0f)` creates `UnaryWithParam{HARDTANH, min_val, max_val}`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for instruction semantics, addressing, and register layout
    **Key Findings**: Confirmed SFPMAD for float addition, SFPLOADI for scalar broadcast, SFPSETCC modes for LT0/GTE0 comparisons, stride-2 DEST addressing model, per-lane CC mechanism.

11. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understand `s2vFloat16b` abstraction
    **Key Findings**: `s2vFloat16b(uint32_t)` converts a uint32_t value to FP16_B format and broadcasts it as a scalar to all SFPU lanes via `SFPLOADI`.

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
    **Reason**: Compare with related clamp kernel for architectural context
    **Key Findings**: Clamp uses direct `v_if (val < min)` / `v_elseif (val >= max)` comparisons, while hardtanh uses the algebraic addition-and-zeroing approach. Different trade-off: clamp preserves exact min/max values via assignment but needs more CC complexity (v_elseif); hardtanh avoids branches but relies on precise parameter pre-computation.
