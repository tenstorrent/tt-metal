## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile_init<>(); hardtanh_tile(0, param0, param1, param2);` (reconstructed -- the API header, LLK dispatch, and `get_op_init_and_func_parameterized` case for HARDTANH are absent from this repository; only the core SFPU implementation remains)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | `get_op_init_and_func_parameterized()` case for HARDTANH is absent (nuked). In the core SFPU function, `APPROXIMATION_MODE` is a template parameter but is never referenced in the function body, so its value has no effect on the code path taken. |
| Effective SFPU path | The single code path in `_calculate_hardtanh_` is always taken regardless of `APPROXIMATION_MODE`, since the template parameter is unused in the function body. | `ckernel_sfpu_hardtanh.h` -- no `if constexpr (APPROXIMATION_MODE)` branch exists |

### SFPU Abstraction Layers
The API header and LLK dispatch files for hardtanh have been removed from this repository. The table below shows the expected structure based on the pattern used by other operations (e.g., swish, frac) and the surviving core SFPU implementation.

| Layer | File Path |
|-------|-----------|
| **API Header** | Missing (nuked). Expected: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` |
| **LLK Dispatch** | Missing (nuked). Expected: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical for Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole variant: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
1. **SFPU_OP_CHAIN_0** in the compute kernel expands to `hardtanh_tile_init<APPROX>(); hardtanh_tile(0, param0, param1, param2);` (the API header call).
2. **`hardtanh_tile(idst, p0, p1, p2)`** (API header) calls `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, p0, p1, p2)))` on the MATH RISC-V thread.
3. **`llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(dst_index, p0, p1, p2)`** (LLK dispatch) calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>, dst_index, VectorMode::RC, 8, p0, p1, p2)`.
4. **`_llk_math_eltwise_unary_sfpu_params_`** (params dispatch) sets up the DEST write address, stalls until SFPU is ready, then iterates over 4 faces in `VectorMode::RC`, calling the SFPU function once per face and advancing the DEST address by one face stride (16 sfpi rows = `2 * inc_dst_addr<8>()`) between faces.
5. **`_calculate_hardtanh_<APPROX, 8>(8, param0, param1, param2)`** (core SFPU) executes 8 iterations per face, processing 32 elements per iteration via the SFPI vector abstractions.

Note: Steps 1-3 are reconstructed from the pattern used by surviving operations (swish, frac, atanh, sinh). The core SFPU function (step 5) and the params dispatch (step 4) are present and verified.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (standard for unary elementwise operations).
- **Operation invocation**: The params dispatch calls the core SFPU function once per face (4 times total for RC mode). Each invocation processes 8 SFPU iterations (ITERATIONS=8), covering one 16x16 face (256 elements). The core function receives `iterations=8` along with the three pre-computed threshold parameters (`param0`, `param1`, `param2`).
- **DEST address progression**: Standard DEST progression. On Wormhole, the params dispatch uses `TTI_SETRWC` with `p_setrwc::CR_D, 8` twice between faces (advancing 16 sfpi rows = one face). On Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice. Within each face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering 8 iterations x 32 elements = 256 elements per face.

### Annotated SFPU Kernel Source

The kernel uses **Style A: SFPI-based kernel**. It uses `sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif` SFPI abstractions.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{   // APPROXIMATION_MODE is unused (no conditional branches depend on it)
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load -(neg_threshold) as FP16_B immediate into LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load -(pos_threshold - neg_threshold) as FP16_B immediate
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load -(pos_threshold) as FP16_B immediate
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load current DEST row into LREG

        val += p0; // SFPMAD: val = val * 1.0 + p0  (shift by -neg_threshold; result < 0 means val < neg_threshold)
        v_if (val < 0.0f) // SFPSETCC with CC_LT0: set CC.Res for lanes where val < 0 (i.e., original val < neg_threshold)
        {
            val = 0.0f; // SFPLOADI: clamp to 0.0 (shifted domain); in original domain this means val = neg_threshold after adding back
        }
        v_endif; // SFPENCC: restore unconditional execution

        val += p1; // SFPMAD: val = val * 1.0 + p1  (further shift; after this, val >= 0 means original val >= pos_threshold)
        v_if (val >= 0.0f) // SFPSETCC with CC_GTE0: set CC.Res for lanes where val >= 0 (i.e., original val >= pos_threshold)
        {
            val = 0.0f; // SFPLOADI: clamp to 0.0 (shifted domain); in original domain this means val = pos_threshold after adding back
        }
        v_endif; // SFPENCC: restore unconditional execution

        val += p2; // SFPMAD: val = val * 1.0 + p2  (final shift back to original domain by adding -pos_threshold)

        sfpi::dst_reg[0] = val; // SFPSTORE: write result back to DEST row

        sfpi::dst_reg++; // advance to next sfpi row (next 32 elements)
    }
}
```

**Algorithm explanation**: The hardtanh function clamps values to `[neg_threshold, pos_threshold]`. Rather than using direct comparisons against the thresholds (which would require FP16_A format for full-precision comparison), the kernel uses an algebraic trick with shifted arithmetic:

1. **Step 1** (`val += p0` where `p0 = -neg_threshold`): shifts the value so that `neg_threshold` maps to 0. If the shifted result is negative, the original value was below `neg_threshold`, so it is clamped (set to 0 in the shifted domain).

2. **Step 2** (`val += p1` where `p1 = -(pos_threshold - neg_threshold)`): shifts the (possibly clamped) value further. After this shift, values at or above 0 correspond to original values at or above `pos_threshold`, so they are clamped (set to 0 in the shifted domain).

3. **Step 3** (`val += p2` where `p2 = -pos_threshold`): shifts back to the original domain. Combined with the clamping in steps 1 and 2, unclamped values are restored to their original value, while clamped values resolve to `neg_threshold` (from step 1 clamping) or `pos_threshold` (from step 2 clamping).

This approach uses only additions and sign-bit comparisons, avoiding the need for explicit min/max or full-precision threshold comparisons. All comparisons are against 0 (sign-bit test), which is the cheapest comparison on the SFPU.

### SFPU Instructions Used

| Instruction | Count per iteration | Description |
|-------------|-------------------|-------------|
| `SFPLOAD` | 1 | Load 32 elements from current DEST row pair into an LREG |
| `SFPLOADI` | 3 (params) + 2 (clamp zeros) = 5 total (3 are hoisted out of loop) | Load 16-bit immediate value (FP16_B format) into an LREG. Used for threshold parameters and the 0.0f clamp value. |
| `SFPMAD` | 3 | Fused multiply-add used as addition: `val * 1.0 + param`. Each `val += pN` compiles to one SFPMAD. |
| `SFPSETCC` | 2 | Set per-lane condition code: first with `CC_LT0` (val < 0), second with `CC_GTE0` (val >= 0). |
| `SFPENCC` | 2 (per v_if/v_endif pair) + 2 (CC enable/disable surrounding each v_if) | Enable/disable conditional execution. Used to enter and exit predicated regions. |
| `SFPPUSHC` | 2 | Push CC state onto the per-lane CC stack when entering `v_if` blocks. |
| `SFPPOPC` | 2 | Pop CC state from the per-lane CC stack when exiting `v_if` blocks via `v_endif`. |
| `SFPSTORE` | 1 | Store 32 elements from LREG back to current DEST row pair. |

Note: The exact CC instruction sequence depends on the SFPI compiler's lowering of `v_if`/`v_endif`. A typical `v_if (val < 0.0f) { ... } v_endif;` pattern generates: `SFPENCC` (enable CC) -> `SFPSETCC` (test condition) -> guarded instructions -> `SFPENCC` (disable CC, restore all lanes active). If the compiler uses the CC stack for proper nesting, `SFPPUSHC`/`SFPPOPC` may also be emitted.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Primary working register: holds `val` loaded from DEST, intermediate results, and final result stored back to DEST. |
| **LREG1** | Holds `p0` = `s2vFloat16b(param0)` = `-(neg_threshold)`. Loaded once before the loop via `SFPLOADI`. |
| **LREG2** | Holds `p1` = `s2vFloat16b(param1)` = `-(pos_threshold - neg_threshold)`. Loaded once before the loop via `SFPLOADI`. |
| **LREG3** | Holds `p2` = `s2vFloat16b(param2)` = `-(pos_threshold)`. Loaded once before the loop via `SFPLOADI`. |
| **LREG4-7** | May be used as temporaries by the compiler for the `0.0f` immediate loads inside the conditional branches. The exact allocation depends on register allocation by the SFPI compiler. |
| **DEST rows** | Source and destination: `dst_reg[0]` reads/writes 2 physical rows (32 elements) at the current SFPU address. The address auto-increments by 1 sfpi row per iteration via `dst_reg++`. |
| **CC bits** | Per-lane condition code (`CC.En`, `CC.Res`) used for the two `v_if` conditional branches. The CC stack is used if the compiler lowers `v_if`/`v_endif` with push/pop semantics. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` during init. Since `SfpuType::hardtanh` does not match any special-cased `if constexpr` branch in that function, only the default `ADDR_MOD_7` is set:

**Wormhole B0:**
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

**Blackhole:**
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

Both architectures use the same configuration. The `dest.incr = 0` setting means DEST address auto-increment is disabled at the ADDR_MOD level. Instead, DEST address advancement is handled explicitly:
- **Within a face**: `dst_reg++` in the SFPI code (compiled to DEST address increment by the SFPI compiler).
- **Between faces**: `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` (Blackhole) in the params dispatch layer, advancing by 16 physical DEST rows (= 8 sfpi rows) per call, called twice between faces.

Note: In this nuked repository, `SfpuType::hardtanh` is not present in the metal API's `SfpuType` enum (`tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h`). The address mode analysis is based on the surviving `eltwise_unary_sfpu_configure_addrmod` function, which would use the default branch for any `SfpuType` value that does not match the explicitly listed special cases (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Determine if HARDTANH is a parametrized type and find dispatch configuration functions
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns `true`. The `get_op_init_and_func_parameterized` case for HARDTANH is missing (nuked).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Find approximation mode, compute kernel path, and SFPU_OP_CHAIN_0 expansion
   **Key Findings**: `get_op_approx_mode` returns `false` for all ops (default case only). `get_compute_kernel_path` returns `"eltwise_sfpu.cpp"` for all ops (default case only). The parameterized init/func case for HARDTANH is absent.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation for hardtanh
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Takes 3 pre-computed FP16_B parameters representing negated thresholds. Implements clamping via shifted arithmetic and sign-bit comparisons. APPROXIMATION_MODE template parameter is declared but unused.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Verify Blackhole variant is identical
   **Key Findings**: Identical to Wormhole implementation.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch layer (face iteration, DEST address progression, vector mode handling)
   **Key Findings**: VectorMode::RC iterates over 4 faces, calling the SFPU function once per face. DEST address advances by `TTI_SETRWC(CR_D, 8, ...)` twice between faces (16 physical rows = 1 face).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Verify Blackhole params dispatch pattern
   **Key Findings**: Same structure as Wormhole but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper instead of raw `TTI_SETRWC`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Find the SFPU init function and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod` sets `ADDR_MOD_7` with `dest.incr=0` for all ops except topk_local_sort, typecast, and unary_max/min variants. `_llk_math_eltwise_unary_sfpu_init_` calls `sfpu::_init_sfpu_config_reg()`, configures addrmod, and resets counters.

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h`
   **Reason**: Understand the init abstraction layer
   **Key Findings**: `llk_math_eltwise_unary_sfpu_init<SfpuType, APPROX>()` calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType>()`.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: Reference pattern for how a surviving LLK dispatch file connects API to ckernel
   **Key Findings**: Follows the standard pattern: init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROX>()`, tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_func, dst_index, VectorMode::RC)`.

10. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understand `s2vFloat16b` class used for parameter loading
    **Key Findings**: `s2vFloat16b(uint32_t)` wraps a raw uint32_t as an FP16_B value (no conversion). `s2vFloat16b(float)` converts FP32 to FP16_B by right-shifting the raw bits by 16. The result is used to emit an `SFPLOADI` instruction.

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and addressing
    **Key Findings**: SFPMAD is used for float addition (a * 1.0 + b). SFPSETCC modes CC_LT0 (sign bit test) and CC_GTE0 (inverted sign bit). SFPLOADI loads 16-bit immediate. Per-face iteration count is 8 (FACE_HEIGHT / SFP_DESTREG_STRIDE). Standard tile = 4 faces x 8 iterations = 32 iterations x 32 elements = 1024 elements.

12. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
    **Reason**: Verify the compute kernel dispatch structure
    **Key Findings**: Standard unary compute kernel with `init_sfpu`, tile loop, `copy_tile`, `SFPU_OP_CHAIN_0` macro expansion, `pack_tile` pattern.
