## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `CLAMP_TSS`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()` which returns `"eltwise_sfpu.cpp"` for all ops not listed in its switch statement)
- **SFPU_OP_CHAIN_0 expansion**: **INCOMPLETE DISPATCH** -- `CLAMP_TSS` is not handled by `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()` in `unary_op_utils.cpp`. The operation would `TT_FATAL` at line 34-37 of `get_op_init_and_func_parameterized()` because `is_parametrized_type(UnaryOpType::CLAMP_TSS)` returns `false` (only `HARDTANH` and `SOFTSHRINK` are listed). Even if that check were relaxed, the switch at line 41-43 has only a `default: TT_THROW` case. **There is no tile-level API (`clamp_tile`), no LLK dispatch (`llk_math_eltwise_unary_sfpu_clamp`), no `SfpuType::clamp` entry in the Metal ckernels `llk_sfpu_types.h`, and no compute API header (`api/compute/eltwise_unary/clamp.h`)**. The core SFPU kernel function `_calculate_clamp_` exists only in the tt_llk submodule and is not referenced by any code in the Metal ckernel or TTNN layers.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CLAMP_TSS)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | N/A (dispatch incomplete) | `get_op_init_and_func()` does not handle `CLAMP_TSS`; no tile-level init/func is generated |
| Effective SFPU path | Would use `APPROXIMATION_MODE=false` if wired up | The `_calculate_clamp_` template parameter `APPROXIMATION_MODE` is unused in the function body (no `if constexpr (APPROXIMATION_MODE)` branch exists) |

### SFPU Abstraction Layers

The dispatch chain for `CLAMP_TSS` is **incomplete**. The core SFPU kernel exists in the tt_llk submodule, but the intermediate layers required to wire it into Metal's compute pipeline are absent. Below lists what exists and what is missing.

| Layer | File Path |
|-------|-----------|
| **API Header** | **MISSING** -- `tt_metal/hw/inc/api/compute/eltwise_unary/clamp.h` does not exist. No `clamp_tile()` or `clamp_tile_init()` function is defined anywhere in the codebase. |
| **LLK Dispatch** | **MISSING** -- `llk_math_eltwise_unary_sfpu_clamp.h` does not exist in `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/`. There is no `SfpuType::clamp` entry in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (the Metal-level `SfpuType` enum only contains: `unused`, `frac`, `swish`, `atanh`, `sinh`). |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h` (WH) and `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h` (BH) -- both files are **identical**. |
| **Parameters Dispatch** | Would use `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) -- these generic dispatch files exist and would be used if the LLK dispatch layer were wired up. |

### Call Chain

**The call chain is incomplete and cannot be invoked end-to-end in the current codebase.** Below is the intended call chain based on the pattern used by fully-wired operations (e.g., `frac_tile`):

1. `clamp_tile(idst, param0, param1)` -- **[MISSING]** would be defined in `api/compute/eltwise_unary/clamp.h`, calling `llk_math_eltwise_unary_sfpu_clamp<APPROX>(idst, param0, param1)` on the MATH thread.
2. `llk_math_eltwise_unary_sfpu_clamp<APPROX>(dst_index, ...)` -- **[MISSING]** would be defined in `llk_math_eltwise_unary_sfpu_clamp.h`, calling `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_clamp_<APPROX, 8>, dst_index, VectorMode::RC, param0, param1, param2)`.
3. `_llk_math_eltwise_unary_sfpu_params_<APPROX>(sfpu_func, dst_index, vector_mode, ...)` -- **EXISTS** in `llk_math_eltwise_unary_sfpu_params.h`, sets DEST address, stalls for SFPU, then invokes `sfpu_func` once per face (4 faces for `VectorMode::RC`), with `SETRWC` between faces.
4. `_calculate_clamp_<APPROXIMATION_MODE, ITERATIONS>(iterations, param0, param1, param2)` -- **EXISTS** in `ckernel_sfpu_clamp.h`, the core SFPU kernel function.

### Parameters Dispatch Summary

Since the dispatch is incomplete, the following describes the **intended** behavior based on the generic `_llk_math_eltwise_unary_sfpu_params_` dispatch used by similar operations:

- **Vector mode**: `VectorMode::RC` (all 4 faces processed). This is the default for standard unary SFPU operations.
- **Operation invocation**: The params dispatch calls `_calculate_clamp_<APPROX, 8>(8, param0, param1, param2)` once per face (4 times for RC mode). Each invocation processes 8 iterations (ITERATIONS=8 per face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `ADDR_MOD_7` is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` (the default all-zeros configuration for generic SFPU ops that manage their own DEST addressing via `dst_reg++` in the SFPI abstraction). On Blackhole, the same `ADDR_MOD_7` with identical configuration is used.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_elseif`/`v_endif`), so Style A (inline-commented source code) applies.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h
// (Blackhole version at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_clamp_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{   // APPROXIMATION_MODE is unused (no conditional branches depend on it), ITERATIONS=8 (per face)
    // param0 = min (uint32_t holding FP16A-encoded value)
    // param1 = max (uint32_t holding FP16A-encoded value)
    // param2 = offset (uint32_t holding FP16B-encoded value)

    sfpi::vFloat min    = sfpi::s2vFloat16a(param0); // SFPLOADI: load min as FP16A immediate into LREG
    sfpi::vFloat max    = sfpi::s2vFloat16a(param1); // SFPLOADI: load max as FP16A immediate into LREG
    sfpi::vFloat offset = sfpi::s2vFloat16b(param2); // SFPLOADI: load offset as FP16B immediate into LREG (12 bits)
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row into LREG

        v_if (val < min)  // SFPXFCMPS + SFPENCC/SFPPUSHC: compare val < min, enable CC for lanes where true
        {
            val = min;    // SFPMOV (CC-guarded): for lanes where val < min, set val = min
        }
        v_elseif (val >= max) // SFPCOMPC + SFPXFCMPS + SFPPUSHC: complement previous CC, then compare val >= max
        {
            val = max;    // SFPMOV (CC-guarded): for lanes where val >= max, set val = max
        }
        v_endif;          // SFPPOPC/SFPENCC: restore CC state, disable CC masking

        sfpi::dst_reg[0] = val + offset; // SFPMAD (val * 1.0 + offset) + SFPSTORE: write result back to DEST

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (2 physical DEST rows = 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | Emitted By | Description |
|-------------|-----------|-------------|
| `SFPLOADI` | `s2vFloat16a(param0)`, `s2vFloat16a(param1)`, `s2vFloat16b(param2)` | Load 16-bit immediate (FP16A or FP16B format) into an LREG. Three SFPLOADI instructions load the `min`, `max`, and `offset` constants. |
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from the current DEST row pair into an LREG, with format conversion from DEST format to FP32. |
| `SFPXFCMPS` | `val < min`, `val >= max` | Scalar-vector floating-point comparison. Compares each lane's value against the FP16A scalar and sets per-lane CC results. Used by the SFPI `__vCond` operator overloads for `s2vFloat16` comparisons. |
| `SFPENCC` | `v_if`, `v_endif` | Enable/disable condition code masking. `v_if` enables CC; `v_endif` disables it (returns all lanes to active). |
| `SFPPUSHC` | `v_if`, `v_elseif` | Push current CC state onto the CC stack, enabling nested conditional regions. |
| `SFPPOPC` | `v_endif` | Pop CC state from the stack, restoring the previous CC context. |
| `SFPCOMPC` | `v_elseif` | Complement CC.Res for the else-if branch -- activates lanes that did NOT pass the previous condition. |
| `SFPMOV` | `val = min`, `val = max` | CC-guarded register-to-register copy. Only active lanes receive the new value. |
| `SFPMAD` | `val + offset` | Fused multiply-add: computes `val * 1.0 + offset`. There is no dedicated float add instruction; addition is always via SFPMAD. |
| `SFPSTORE` | `dst_reg[0] = ...` (write) | Store the result from an LREG back to the current DEST row pair, with format conversion from FP32 to DEST format. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (min)** | Holds the `min` clamp bound, loaded once via `SFPLOADI` (FP16A format) before the loop. Persistent across all 8 iterations. |
| **LREG (max)** | Holds the `max` clamp bound, loaded once via `SFPLOADI` (FP16A format) before the loop. Persistent across all 8 iterations. |
| **LREG (offset)** | Holds the `offset` value, loaded once via `SFPLOADI` (FP16B format) before the loop. Persistent across all 8 iterations. |
| **LREG (val)** | Temporary register holding the current element value. Loaded from DEST each iteration via `SFPLOAD`, conditionally overwritten (CC-guarded `SFPMOV`), then used in `SFPMAD` for the add, and stored back to DEST via `SFPSTORE`. |
| **DEST rows** | Input/output storage. Each iteration reads from and writes back to `dst_reg[0]` (the current DEST row pair = 32 elements). The pointer advances by 1 sfpi row per iteration via `dst_reg++`. |
| **CC Stack** | Used by the `v_if`/`v_elseif`/`v_endif` control flow. `SFPPUSHC` saves the comparison result; `SFPCOMPC` inverts for the else-if path; `SFPPOPC` restores. Stack depth: 1 entry (no nested conditionals). |

Note: The exact LREG indices (0-7) are assigned by the SFPI compiler and are not explicitly specified in the source code. The compiler allocates registers based on liveness analysis. Typically: LREG0 for `val` (loaded from DEST), LREG1-3 for `min`, `max`, `offset`, and LREG3 may be reused for intermediate comparison results. The SFPI abstraction hides these details.

### Address Mode Configuration

The `clamp` operation would use the default `ADDR_MOD_7` configuration set by `eltwise_unary_sfpu_configure_addrmod<SfpuType::clamp>()` in the LLK init. Since `SfpuType::clamp` does not match any of the `if constexpr` specializations in the addr_mod configuration function, it falls through to only the default `ADDR_MOD_7` setup:

**Wormhole B0** (`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

**Blackhole** (`tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`):
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

Both hardware generations use identical address mode configuration. The `dest.incr = 0` means that hardware auto-increment of the DEST address is disabled. Instead, DEST address advancement is handled entirely by the SFPI abstraction's `dst_reg++` mechanism (which emits `SETRWC` instructions to manually advance the DEST read/write counter by the stride-2 amount).

Between faces, the params dispatch (`_llk_math_eltwise_unary_sfpu_params_`) uses `TTI_SETRWC` to advance by 8+8=16 physical DEST rows (equivalent to one face stride) on Wormhole, or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice on Blackhole.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine how CLAMP_TSS maps to compute kernel path, init/func defines, and approximation mode
   **Key Findings**: CLAMP_TSS is not handled in `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()` -- the dispatch would TT_FATAL/TT_THROW. `get_op_approx_mode()` returns false (default case). `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default case).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check `is_parametrized_type()` for CLAMP_TSS
   **Key Findings**: CLAMP_TSS is not listed -- only HARDTANH and SOFTSHRINK return true.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
   **Reason**: Understand how clamp_tss() constructs the UnaryWithParam
   **Key Findings**: Constructs `UnaryWithParam{UnaryOpType::CLAMP_TSS, {min_val, max_val}}` and passes to `unary_impl`.

4. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
   **Reason**: Verify the program factory dispatch and runtime arg packing
   **Key Findings**: CLAMP_TSS is not in the `packed_scalar` switch (only HARDSHRINK and WHERE_TSS are). The dispatch goes through `get_block_defines()` which would fail for CLAMP_TSS.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
   **Reason**: Core SFPU kernel source for clamp operation
   **Key Findings**: Implements `_calculate_clamp_` using SFPI abstractions. Takes 3 params (min as FP16A, max as FP16A, offset as FP16B). Uses `v_if`/`v_elseif`/`v_endif` for conditional clamping, then adds offset.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_clamp.h`
   **Reason**: Check if BH implementation differs from WH
   **Key Findings**: Identical to the WH version -- same function signature and body.

7. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Check if SfpuType::clamp exists in Metal's ckernel layer
   **Key Findings**: Only contains: unused, frac, swish, atanh, sinh. No clamp entry.

8. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/` (directory listing)
   **Reason**: Check if clamp.h API header exists
   **Key Findings**: Directory contains: eltwise_unary.h, frac.h, swish.h, atanh.h, sinh.h, activations.h, sfpu_split_includes.h, README.md. No clamp.h.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand addr_mod configuration for SFPU unary ops
   **Key Findings**: Default ADDR_MOD_7 is `{srca.incr=0, srcb.incr=0, dest.incr=0}`. SfpuType::clamp would not match any `if constexpr` specialization, so only the default applies.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Understand the params dispatch pattern for unary SFPU ops
    **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` dispatches the sfpu_func 4 times for VectorMode::RC (once per face), with SETRWC between faces.

11. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understand s2vFloat16a/s2vFloat16b conversion classes
    **Key Findings**: `s2vFloat16a` constructs with `Format::fp16a` (IEEE binary16), `s2vFloat16b` with `Format::fp16b` (bfloat16). When constructed from uint32_t, the raw value is used directly (no conversion). When from float, fp32_to_fp16a/fp32_to_fp16b conversion is applied.

12. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Understand SFPI vFloat operations, comparison operators, and instruction emission
    **Key Findings**: `vFloat(s2vFloat16)` emits `__builtin_rvtt_sfpxloadi`. Comparisons against `s2vFloat16` use `__builtin_rvtt_sfpxfcmps`. `vFloat + vFloat` compiles to `SFPMAD`. `dst_reg[0]` read emits `SFPLOAD`, write emits `SFPSTORE`.

13. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU architecture reference for instruction semantics, addressing model, CC mechanism
    **Key Findings**: Used for stride-2 addressing model, SFPMAD semantics (float add = MAD a*1.0+b), CC stack operations, SFPXFCMPS for scalar-vector comparison.

14. **File**: `docs/sfpu_operations/key_notes/clamp_tss_key_notes.md`
    **Reason**: Background on the clamp_tss operation semantics
    **Key Findings**: Formula is `clamp(x, min, max)`, deterministic and mode-independent.
