## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the hardtanh (clamp) operation.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile_init(); hardtanh_tile(0, param0, param1);` (nuked -- the API header, LLK dispatch file, and `get_op_init_and_func_parameterized` case for HARDTANH have all been removed from this codebase; the expansion is inferred from the parametrized-type registration pattern and surviving reference operations like swish and frac)

**Note on nuked layers**: The `DEEP_NUKE_MANIFEST.md` confirms that hardtanh's dispatch code, compute API header, metal ckernel (LLK dispatch file), and the `get_op_init_and_func_parameterized` switch-case were all deleted. However, the core SFPU implementation (`ckernel_sfpu_hardtanh.h`) survives intact in the `tt_llk` third-party library for both Wormhole B0 and Blackhole targets. The `UnaryOpType::HARDTANH` enum value remains registered in `unary_op_types.hpp`, and `is_parametrized_type(HARDTANH)` returns `true` in `unary_op_utils.hpp`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit case for HARDTANH) |
| Template parameter (SFPU_OP_CHAIN) | Not directly observable (dispatch nuked) | Based on the surviving function signature `_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>`, the API header would pass `APPROX` (the compile-time define derived from `math_approx_mode`). Since `math_approx_mode=false`, `APPROXIMATION_MODE=false`. |
| Effective SFPU path | `APPROXIMATION_MODE` is accepted as a template parameter but **not used** in any conditional logic within the kernel body. The same code path executes regardless of approximation mode. This is expected: hardtanh is a simple piecewise-linear clamp that requires no mathematical approximation. | `ckernel_sfpu_hardtanh.h` lines 16-53 -- no `if constexpr (APPROXIMATION_MODE)` branches |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | Deleted (was `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` -- confirmed removed in `DEEP_NUKE_MANIFEST.md`). Pattern: would expose `hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)` and `hardtanh_tile_init()`, calling the LLK dispatch on the MATH thread. |
| **LLK Dispatch** | Deleted (was `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`). Pattern: would call `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `_calculate_hardtanh_<APPROXIMATE, ITERATIONS>` as the callable and the three pre-packed uint32 threshold parameters. |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical file exists at `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared generic dispatch for all parameterized unary SFPU ops; Blackhole equivalent at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
1. **SFPU_OP_CHAIN_0** in `eltwise_sfpu.cpp` expands to the tile-level API calls: `hardtanh_tile_init()` followed by `hardtanh_tile(0, param0, param1)` (both nuked).
2. The **API header** (`hardtanh.h`, nuked) wraps these as `MATH((llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()))` and `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)))`, ensuring execution only on the MATH RISC-V thread.
3. The **LLK dispatch init** (`llk_math_eltwise_unary_sfpu_hardtanh_init`, nuked) calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>()`, which configures the SFPU config register, sets up `ADDR_MOD_7`, and resets RWC counters.
4. The **LLK dispatch function** (`llk_math_eltwise_unary_sfpu_hardtanh`, nuked) calls the generic `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `_calculate_hardtanh_<APPROXIMATE, ITERATIONS>` as the callable and the three pre-computed uint32 parameters (param0, param1, param2 -- three threshold values packed as FP16_B).
5. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing via `set_dst_write_addr`, stalls until SFPU is ready, then iterates over 4 faces (for `VectorMode::RC`), calling the SFPU function once per face and issuing `SETRWC` / `inc_dst_addr` to advance the DEST face pointer between faces.
6. **`_calculate_hardtanh_`** (in `ckernel_sfpu_hardtanh.h`) executes 8 iterations per face invocation (default `ITERATIONS=8`), processing 32 elements per iteration via the stride-2 SFPU addressing model.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the 32x32 tile are processed. The dispatch iterates over 4 faces, calling `_calculate_hardtanh_` once per face.
- **Operation invocation**: For each face, the callable `_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>` is invoked with `iterations=ITERATIONS` (default 8) plus the three pre-packed uint32 threshold parameters (`param0`, `param1`, `param2`). The inner loop runs 8 iterations per face (8 sfpi rows x 32 elements = 256 elements = 1 face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses inline `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls (two per face transition = 16 physical DEST rows = 1 face stride). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper calls `math::inc_dst_addr<8>()` twice. Both achieve the same effect: advancing the DEST pointer by one face (16 physical rows) between face invocations.

### Parameter Encoding

The hardtanh operation receives two float parameters from the host: `min_val` (default -1.0) and `max_val` (default 1.0). These are pre-transformed into three FP16_B-packed uint32 values before reaching the SFPU kernel. The source comments describe:

```
param0 = -(neg_threshold)                  // i.e., -min_val
param1 = -(pos_threshold - neg_threshold)  // i.e., -(max_val - min_val) = min_val - max_val
param2 = -(pos_threshold)                  // i.e., -max_val  [see note below]
```

**Algorithm derivation and param2 sign note**: The kernel implements `clamp(x, min_val, max_val)` through three additions and two conditional zeroing operations. For the algorithm to produce correct results, the parameter values must satisfy:

| Constraint | Derivation |
|-----------|------------|
| `p0 + p1 + p2 = 0` | Unclamped path: x passes through unchanged |
| `p1 + p2 = min_val` | Low-clamped path: zeroed value reconstructs to min_val |
| `p2 = max_val` | High-clamped path: zeroed value reconstructs to max_val |

Solving: `p0 = -min_val`, `p1 = min_val - max_val`, `p2 = max_val`. The derivation for `p0` and `p1` matches the source comments. However, the source comment states `param2 = -(pos_threshold)` (i.e., `-max_val`), whereas the mathematical derivation requires `p2 = max_val` (positive). The most likely explanation is that the source comment for `param2` has a sign error, and the host code (nuked) actually packs `+max_val` into `param2`. The algorithm is correct as shipped -- this discrepancy is only in the comment.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{   // APPROXIMATION_MODE unused (no conditional branches), ITERATIONS=8 (default)
    // All params are in FP16_B format
    // param0 = -(neg_threshold)                   -> p0 = -min_val
    // param1 = -(pos_threshold - neg_threshold)   -> p1 = min_val - max_val
    // param2 = -(pos_threshold)                   -> p2 = max_val (see Parameter Encoding note above)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // Broadcast scalar -min_val to all 32 SFPU lanes -> SFPLOADI
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // Broadcast scalar (min_val - max_val) to all lanes -> SFPLOADI
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // Broadcast scalar max_val to all lanes -> SFPLOADI
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face, 32 elements per iteration
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        val += p0; // SFPMAD (val * 1.0 + p0): computes val - min_val
        v_if (val < 0.0f) // SFPSETCC with LT0 mode: enable CC for lanes where (val - min_val) < 0
        {
            val = 0.0f; // SFPLOADI 0.0: zero out lanes where input < min_val
        }
        v_endif; // SFPENCC: restore all lanes to unconditional execution

        val += p1; // SFPMAD (val * 1.0 + p1): adds (min_val - max_val), effectively computing val - max_val for unclamped lanes
        v_if (val >= 0.0f) // SFPSETCC with GTE0 mode: enable CC for lanes where result >= 0 (input >= max_val)
        {
            val = 0.0f; // SFPLOADI 0.0: zero out lanes where input >= max_val
        }
        v_endif; // SFPENCC: restore all lanes

        val += p2; // SFPMAD (val * 1.0 + p2): adds max_val, reconstructing the final clamped value

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 clamped elements back to current DEST row pair

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Algorithm walkthrough**: The kernel implements `clamp(x, min_val, max_val)` without explicit min/max instructions by decomposing it into range-check additions and conditional zeroing:

1. **Shift by -min_val** (`val += p0`): After this, `val = x - min_val`. If negative, `x < min_val`.
2. **Clamp low** (`v_if val < 0: val = 0`): Lanes below `min_val` are zeroed.
3. **Shift by (min_val - max_val)** (`val += p1`): For unclamped lanes, `val = x - max_val`. For low-clamped lanes, `val = 0 + (min_val - max_val) = min_val - max_val` (negative, since min < max).
4. **Clamp high** (`v_if val >= 0: val = 0`): Lanes above `max_val` (where `x - max_val >= 0`) are zeroed.
5. **Shift by max_val** (`val += p2`): Reconstructs the final value:
   - **Unclamped**: `(x - max_val) + max_val = x` (original value preserved)
   - **Low-clamped**: `(min_val - max_val) + max_val = min_val` (clamped to lower bound)
   - **High-clamped**: `0 + max_val = max_val` (clamped to upper bound)

This approach uses exactly 3 additions (SFPMAD) and 2 conditional assignments (v_if/v_endif pairs), processing all 32 lanes in parallel per iteration.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Count per Iteration | Description |
|-------------|-----------------|---------------------|-------------|
| **SFPLOAD** | `dst_reg[0]` (read) | 1 | Load 32 elements from current DEST row pair into an LREG |
| **SFPLOADI** | `s2vFloat16b(param)`, `val = 0.0f` | 3 (params) + 2 (zeroing) = 5 | Load immediate scalar value into LREG. The 3 parameter loads (`p0`, `p1`, `p2`) occur before the loop and are hoisted. The 2 zeroing loads occur conditionally within the loop. |
| **SFPMAD** | `val += pN` | 3 | Fused multiply-add implementing float addition: `val = val * 1.0 + pN`. There is no dedicated SFPU float-add instruction; all float additions are emitted as SFPMAD with multiplicand = 1.0. |
| **SFPSETCC** | `v_if (val < 0.0f)`, `v_if (val >= 0.0f)` | 2 | Set per-lane condition code. First use: LT0 mode (CC.Res = 1 if val < 0). Second use: GTE0 mode (CC.Res = 1 if val >= 0). |
| **SFPENCC** | `v_if` (entry), `v_endif` | 4 | Enable/disable per-lane condition code masking. Each `v_if` enables CC (so subsequent instructions are masked), and each `v_endif` disables CC (all lanes active again). |
| **SFPSTORE** | `dst_reg[0] = val` (write) | 1 | Store 32 elements from LREG back to current DEST row pair |

**Note**: `SFPPUSHC` and `SFPCOMPC` are NOT used because there are no nested conditionals or else-branches. Each `v_if`/`v_endif` pair is independent and non-nested.

### SFPU Register Usage

| Register | Usage | Notes |
|----------|-------|-------|
| **DEST row pair** (via `dst_reg[0]`) | Input/output data (32 elements per access) | Read at loop start, written at loop end. Stride-2 addressing: each sfpi address spans 2 physical DEST rows (16 elements each). |
| **LREG (val)** | Working register for the clamping computation | Holds the intermediate and final clamped value. Compiler-assigned LREG index. |
| **LREG (p0)** | Holds `-min_val` (FP16_B scalar broadcast to all lanes) | Loaded once before the loop via `s2vFloat16b(param0)`. Reused across all 8 iterations. |
| **LREG (p1)** | Holds `min_val - max_val` (FP16_B scalar broadcast) | Loaded once before the loop via `s2vFloat16b(param1)`. Reused across all 8 iterations. |
| **LREG (p2)** | Holds `max_val` (FP16_B scalar broadcast) | Loaded once before the loop via `s2vFloat16b(param2)`. Reused across all 8 iterations. |
| **CC (Condition Code)** | Per-lane predication for conditional zeroing | Used by `v_if`/`v_endif` to selectively zero lanes that fall outside [min_val, max_val]. Two independent CC regions per iteration (no nesting). |

The kernel uses up to 4 LREGs simultaneously (val + 3 parameter registers). Since the SFPU provides 8 LREGs per lane, this is well within budget.

### Address Mode Configuration

The standard unary SFPU address mode `ADDR_MOD_7` is configured during init with:
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

This configuration is **identical on both Wormhole B0 and Blackhole** (verified by comparing `eltwise_unary_sfpu_configure_addrmod<>()` in both platforms' `llk_math_eltwise_unary_sfpu.h`).

HARDTANH is **not** in the special-case lists that configure `ADDR_MOD_6` with a non-zero dest increment. The `ADDR_MOD_6` special cases are limited to `topk_local_sort` (dest.incr=32), `typecast`, `unary_max/min` variants, `reciprocal` (Blackhole only), and `signbit` (dest.incr=2). Since hardtanh uses `dst_reg++` in its inner loop for DEST advancement rather than hardware auto-increment, `ADDR_MOD_7` with all-zero increments is correct.

The inter-face DEST advancement is handled by the params dispatch layer:
- **Wormhole B0**: Two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls per face transition (each advances by 8 physical rows, totaling 16 = one face).
- **Blackhole**: Two `math::inc_dst_addr<8>()` calls per face transition (same net effect).

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation of the hardtanh kernel
   **Key Findings**: Implements clamp(x, min, max) using 3 FP16_B parameters, 3 SFPMAD additions, and 2 conditional zeroing operations with v_if/v_endif. APPROXIMATION_MODE template parameter is accepted but unused.

2. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Verify Blackhole variant is identical to Wormhole
   **Key Findings**: Byte-identical to Wormhole B0 version.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the generic parameterized dispatch pattern for unary SFPU ops
   **Key Findings**: Iterates over 4 faces for VectorMode::RC, calls the SFPU function once per face, advances DEST via TTI_SETRWC between faces.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Verify Blackhole params dispatch differences
   **Key Findings**: Functionally identical to Wormhole but refactored to use helper functions (_llk_math_eltwise_unary_sfpu_start_, _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_, _llk_math_eltwise_unary_sfpu_done_) instead of inline TTI calls.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand ADDR_MOD configuration and init/done patterns for unary SFPU ops
   **Key Findings**: ADDR_MOD_7 configured with all-zero increments. HARDTANH is not in any special-case ADDR_MOD_6 list. Init configures SFPU config reg, address modes, and resets RWC counters.

6. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path and approximation mode for HARDTANH
   **Key Findings**: get_compute_kernel_path returns "eltwise_sfpu.cpp" (default). get_op_approx_mode returns false (default). HARDTANH case missing from get_op_init_and_func_parameterized (nuked).

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Verify HARDTANH is registered as parametrized type
   **Key Findings**: is_parametrized_type(HARDTANH) returns true.

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Understand host-side parameter passing for hardtanh
   **Key Findings**: hardtanh(input, min_val=-1.0f, max_val=1.0f) creates UnaryWithParam{HARDTANH, min_val, max_val}.

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
   **Reason**: Verify compute kernel structure and SFPU_OP_CHAIN_0 dispatch pattern
   **Key Findings**: Standard pattern: init_sfpu, for each block/tile: acquire->copy->SFPU_OP_CHAIN_0->commit->wait->pack->pop->release.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
    **Reason**: Compare with structurally similar clamp operation for algorithm understanding
    **Key Findings**: Clamp uses explicit min/max comparisons with v_if/v_elseif/v_endif. Hardtanh uses a fundamentally different algebraic approach with three additions and two conditional zeroings.

11. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understand s2vFloat16b scalar-to-vector conversion
    **Key Findings**: s2vFloat16b takes a uint32 and interprets it as a bfloat16 (FP16_B) bit pattern, broadcasting the scalar value to all SFPU lanes.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU instruction semantics, addressing model, and register file
    **Key Findings**: SFPMAD is used for all float additions (no dedicated add instruction). Stride-2 addressing model: each sfpi row = 2 physical DEST rows = 32 elements. 8 iterations per face, 4 faces per tile.

13. **File**: `DEEP_NUKE_MANIFEST.md`
    **Reason**: Confirm which hardtanh layers were deleted
    **Key Findings**: Hardtanh dispatch removed, compute API deleted, metal ckernel deleted (wh+bh), metal LLK deleted (wh+bh). Core tt_llk implementation preserved.

14. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
    **Reason**: Reference for intact LLK dispatch pattern (frac is a surviving non-nuked op)
    **Key Findings**: Pattern: llk_math_eltwise_unary_sfpu_frac calls _llk_math_eltwise_unary_sfpu_params_ with calculate_frac as callable. Init calls llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROXIMATE>().

15. **File**: `docs/sfpu_operations/key_notes/hardtanh_key_notes.md`
    **Reason**: Understand mathematical definition and parameters
    **Key Findings**: hardtanh = clamp(x, min_val, max_val), default min=-1.0, max=1.0. Deterministic, mode-independent.
