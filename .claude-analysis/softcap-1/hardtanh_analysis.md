## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**Important Note: Incomplete Dispatch Chain**

The `hardtanh` operation has a core SFPU kernel implementation (`_calculate_hardtanh_`) in the LLK layer, but its dispatch chain through the Metal compute API is **not fully wired up**. Specifically:

1. `UnaryOpType::HARDTANH` exists in the enum (`unary_op_types.hpp:115`)
2. `is_parametrized_type(HARDTANH)` returns `true` (`unary_op_utils.hpp:46`)
3. The host-side `ttnn::hardtanh()` constructs a `UnaryWithParam` with `min_val` and `max_val` parameters (`unary.hpp:291`)
4. **However**, `get_op_init_and_func_parameterized()` has no `HARDTANH` case -- it falls through to `default: TT_THROW` (`unary_op_utils.cpp:42`)
5. No API header (`hardtanh.h`) exists in `tt_metal/hw/inc/api/compute/eltwise_unary/`
6. No LLK dispatch file (`llk_math_eltwise_unary_sfpu_hardtanh.h`) exists in `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/`
7. `SfpuType::hardtanh` does NOT exist in the production Metal `llk_sfpu_types.h` (only in the test helpers enum)
8. The program factory does not pack `HARDTANH` parameters as runtime args

This means calling `ttnn::hardtanh()` at runtime will throw an error in `get_op_init_and_func_parameterized`. The analysis below documents the **existing core SFPU kernel** that would be used once the dispatch chain is completed.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default for all ops via `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: Not wired -- would be `hardtanh_tile_init() hardtanh_tile(0, param0, param1)` once dispatch chain is completed (based on the documented API in `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/hardtanh_tile.rst`)

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (not wired) | `get_op_init_and_func()` -- HARDTANH has no case, would throw at runtime |
| Effective SFPU path | `APPROXIMATION_MODE` template parameter not resolved at runtime | The `_calculate_hardtanh_` kernel does not use `APPROXIMATION_MODE` in any `if constexpr` branch -- the code path is identical regardless of its value |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist -- no `hardtanh.h` in `tt_metal/hw/inc/api/compute/eltwise_unary/` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- no `llk_math_eltwise_unary_sfpu_hardtanh.h` in `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical on Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) |
| **Parameters Dispatch** | This level of abstraction doesn't exist -- would use `llk_math_eltwise_unary_sfpu_params.h` once wired (located at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

The full call chain does not exist in the current codebase. Once wired, the expected call chain (based on analogous operations like `swish`) would be:

1. `SFPU_OP_CHAIN_0` macro in the compute kernel expands to `hardtanh_tile(0, param0, param1)`
2. `hardtanh_tile(idst, param0, param1)` (API header) would call `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)` via the `MATH((...))` macro
3. `llk_math_eltwise_unary_sfpu_hardtanh(dst_index, ...)` (LLK dispatch) would call `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>, dst_index, VectorMode::RC, param0, param1, param2)` where param2 is derived from param0 and param1
4. `_llk_math_eltwise_unary_sfpu_params_` loops over 4 faces calling `_calculate_hardtanh_` with 8 iterations per face, advancing the DEST address between faces via `SETRWC`

Currently, only step 4 and the core SFPU function (step 4's callee) exist in the codebase.

### Parameters Dispatch Summary

The parameters dispatch layer (`llk_math_eltwise_unary_sfpu_params.h`) exists as a generic template and would be used for hardtanh once wired. Based on the existing infrastructure:

- **Vector mode**: `VectorMode::RC` (all 4 faces processed) -- this is the standard default for unary SFPU operations
- **Operation invocation**: The params dispatch template calls the SFPU function via `std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...)` once per face (4 times total for RC mode). Each invocation processes 8 iterations (ITERATIONS=8) covering one 16x16 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, the params dispatch uses `ADDR_MOD_7` with `dest.incr=0` (no auto-increment; the kernel manages address advancement via `dst_reg++` internally). Between faces, `TTI_SETRWC` advances by 8+8=16 physical rows. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same 16-row face stride.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{   // APPROXIMATION_MODE is unused (no if constexpr branches depend on it), ITERATIONS is unused (iterations param used instead)
    // All params are in FP16_B format
    // param0 = -(neg_threshold)       i.e., -min_val
    // param1 = -(pos_threshold - neg_threshold)  i.e., -(max_val - min_val)
    // param2 = -(pos_threshold)       i.e., -max_val

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load -min_val as FP16_B immediate into LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load -(max_val - min_val) as FP16_B immediate into LREG
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load -max_val as FP16_B immediate into LREG
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        val += p0; // SFPMAD: val = val * 1.0 + p0, i.e., val = x + (-min_val) = x - min_val
        v_if (val < 0.0f) // SFPSETCC(LT0): sets CC.Res=1 for lanes where (x - min_val) < 0, i.e., x < min_val
        {
            val = 0.0f; // SFPLOADI: conditional zero assignment -- only lanes where x < min_val are zeroed
        }
        v_endif; // Restores CC state (SFPPOPC/SFPENCC)

        val += p1; // SFPMAD: val = val + (-(max_val - min_val))
        // If x < min_val: val = 0 + (-(max_val - min_val)) = min_val - max_val (negative, since min < max)
        // If x in range:  val = (x - min_val) + (-(max_val - min_val)) = x - max_val
        v_if (val >= 0.0f) // SFPSETCC(GTE0): sets CC.Res=1 for lanes where val >= 0, i.e., x >= max_val
        {
            val = 0.0f; // SFPLOADI: conditional zero assignment -- only lanes where x >= max_val are zeroed
        }
        v_endif; // Restores CC state (SFPPOPC/SFPENCC)

        val += p2; // SFPMAD: val = val + (-max_val)
        // If x < min_val: val = (min_val - max_val) + (-max_val) = min_val - 2*max_val ... (see algorithm note below)
        // If x in range:  val = (x - max_val) + (-max_val) = x - 2*max_val ... (see algorithm note below)
        // If x >= max_val: val = 0 + (-max_val) = -max_val ... Wait, this gives -max_val not max_val.

        sfpi::dst_reg[0] = val; // SFPSTORE: write result back to current DEST row pair

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

**Algorithm Correctness Note**: The parameter encoding and algorithm as written in the current kernel raise questions about correctness. Tracing through the three cases with the documented parameter semantics (`param0 = -min_val`, `param1 = -(max_val - min_val)`, `param2 = -max_val`):

- **Case x < min_val**: `val = 0` (clamped), then `val = -(max_val - min_val)` (negative, passes GTE0 check), then `val = -(max_val - min_val) + (-max_val) = min_val - 2*max_val`. This does NOT equal `min_val`.
- **Case min_val <= x < max_val**: `val = x - min_val` (positive, passes LT0 check), then `val = x - max_val` (negative, passes GTE0 check), then `val = (x - max_val) + (-max_val) = x - 2*max_val`. This does NOT equal `x`.
- **Case x >= max_val**: `val = x - min_val` (positive, passes LT0 check), then `val = x - max_val` (non-negative, zeroed), then `val = 0 + (-max_val) = -max_val`. This does NOT equal `max_val`.

This suggests the parameter comments in the kernel may not accurately describe the values actually passed to the function at runtime. The algorithm would work correctly if the parameters were instead:
- `param0 = -min_val`
- `param1 = -(max_val - min_val)`
- `param2 = max_val` (positive, not negated)

With `param2 = max_val`:
- **x < min_val**: `val = 0 + (min_val - max_val) + max_val = min_val` -- correct
- **min_val <= x < max_val**: `val = (x - min_val) + (min_val - max_val) + max_val = x` -- correct
- **x >= max_val**: `val = 0 + max_val = max_val` -- correct

However, since the dispatch chain is incomplete and no host-side code actually computes and passes these parameters, the correct parameter semantics cannot be verified from the current codebase alone. The kernel code and comments represent an unfinished implementation.

### SFPU Instructions Used

The kernel uses SFPI abstractions that compile to the following SFPU instructions:

| Instruction | SFPI Abstraction | Description | Count per iteration |
|-------------|-----------------|-------------|---------------------|
| `SFPLOADI` | `s2vFloat16b(param)`, `val = 0.0f` | Load 16-bit immediate (FP16_B format) into LREG | 3 (params, outside loop) + 2 (conditional zeros, inside loop) |
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into LREG | 1 |
| `SFPMAD` | `val += p0`, `val += p1`, `val += p2` | Fused multiply-add: `val = val * 1.0 + pN` (addition via FMA with multiplier = 1.0) | 3 |
| `SFPSETCC` | `val < 0.0f`, `val >= 0.0f` | Set per-lane CC.Res based on sign comparison (LT0 or GTE0 mode) | 2 |
| `SFPENCC` | `v_if` entry, `v_endif` exit | Enable/disable condition code masking | 2 (enable) + 2 (disable) |
| `SFPPUSHC` | `v_if` entry | Push CC state onto per-lane CC stack | 2 |
| `SFPPOPC` | `v_endif` exit | Pop CC state from per-lane CC stack | 2 |
| `SFPSTORE` | `dst_reg[0] = val` (write) | Store LREG value back to current DEST row pair | 1 |

Note: `dst_reg++` does not emit an instruction -- it is a compile-time address offset tracked by the SFPI compiler, affecting subsequent SFPLOAD/SFPSTORE addresses.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** (estimated) | Used by the SFPI compiler to hold intermediate `vFloat` values (`val`, `p0`, `p1`, `p2`). Exact register assignment is determined by the SFPI register allocator. |
| **DEST rows** | Input tile data is read from DEST via `SFPLOAD` (stride-2 addressing: each access reads 2 physical rows = 32 elements). Results are written back to the same DEST location via `SFPSTORE`. |
| **CC registers** | Per-lane condition code bits (CC.En, CC.Res) are used for the two `v_if` blocks. CC stack is used to save/restore CC state across conditional blocks (depth 1 for each `v_if`). |

The kernel requires at most 4 LREGs simultaneously (one for each of `val`, `p0`, `p1`, `p2`), well within the 8-LREG budget.

### Address Mode Configuration

The address mode configuration for hardtanh would use `ADDR_MOD_7` (the default for generic SFPU operations), configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType>()`:

```
ADDR_MOD_7: {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0}
}
```

This configuration is identical on both Wormhole and Blackhole. The `dest.incr = 0` means no automatic DEST address increment between instructions -- the kernel handles address progression internally via `dst_reg++` (which adjusts the SFPLOAD/SFPSTORE address offsets at compile time).

Hardtanh does NOT fall into any of the special-case `SfpuType` checks (`topk_local_sort`, `typecast`, `unary_max/min`, `signbit`, `reciprocal`) that would configure `ADDR_MOD_6` with a non-zero `dest.incr`.

Note: Since `SfpuType::hardtanh` does not exist in the production Metal `llk_sfpu_types.h`, the init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::hardtanh>()` cannot currently be instantiated. Once the `SfpuType` enum is extended, the default `ADDR_MOD_7` configuration would apply.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and SFPU_OP_CHAIN_0 expansion for HARDTANH
   **Key Findings**: `get_compute_kernel_path()` defaults to `eltwise_sfpu.cpp`; `get_op_approx_mode()` defaults to `false`; `get_op_init_and_func_parameterized()` has no HARDTANH case (throws at runtime)

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check parametrized type status and dispatch infrastructure
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns `true`; HARDTANH is declared but not fully wired through dispatch

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel source code for hardtanh
   **Key Findings**: SFPI-based kernel using `s2vFloat16b` parameter loading, 3 additions via SFPMAD, 2 conditional zeroing blocks via v_if/v_endif, standard dst_reg iteration pattern

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Verify Blackhole implementation matches Wormhole
   **Key Findings**: Identical implementation to Wormhole

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch template that would be used for hardtanh
   **Key Findings**: Generic callable-based dispatch with VectorMode::RC loop over 4 faces, SETRWC between faces, STALL_SFPU/WAIT_SFPU synchronization

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand ADDR_MOD configuration and init function for unary SFPU operations
   **Key Findings**: Default ADDR_MOD_7 with dest.incr=0; no special cases apply to hardtanh

7. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Verify SfpuType enum for production Metal
   **Key Findings**: Only contains `unused`, `frac`, `swish`, `atanh`, `sinh` -- no `hardtanh` entry

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Understand host-side hardtanh function signature and parameter passing
   **Key Findings**: `hardtanh(input, min_val=-1.0f, max_val=1.0f)` constructs `UnaryWithParam{HARDTANH, min_val, max_val}`

9. **File**: `runtime/sfpi/include/sfpi_fp16.h`
   **Reason**: Understand `s2vFloat16b` parameter conversion semantics
   **Key Findings**: `s2vFloat16b(uint32_t)` passes the raw uint32_t directly (assumes it already contains FP16_B value); `s2vFloat16b(float)` converts FP32 to FP16_B by right-shifting 16 bits

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU instruction semantics, register layout, and CC mechanism
    **Key Findings**: vFloat + vFloat emits SFPMAD; SFPSETCC has LT0 and GTE0 modes; standard v_if/v_endif maps to SFPENCC/SFPSETCC/SFPPUSHC/SFPPOPC sequence

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
    **Reason**: Compare with functionally similar clamp operation for pattern understanding
    **Key Findings**: Clamp uses same 3-param pattern with s2vFloat16a/s2vFloat16b but applies direct min/max comparisons via v_if/v_elseif/v_endif rather than the addition-and-zero technique used by hardtanh

12. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
    **Reason**: Verify runtime parameter passing for HARDTANH in the program factory
    **Key Findings**: Only HARDSHRINK and WHERE_TSS have packed_scalar runtime args; HARDTANH is not handled
