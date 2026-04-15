## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: **NOT WIRED** -- the TTNN dispatch layer (`get_op_init_and_func`) has no case for `HARDTANH`. See "Integration Gap" below.

**Integration Gap**: The `HARDTANH` operation has a complete core SFPU kernel implementation in `ckernel_sfpu_hardtanh.h` (both Wormhole B0 and Blackhole), and the `SfpuType::hardtanh` enum value exists in `llk_sfpu_types.h`. However, the following layers are missing and prevent end-to-end execution:
1. **TTNN dispatch** (`unary_op_utils.cpp`): `get_op_init_and_func_parameterized()` has no case for `HARDTANH` (falls through to `default: TT_THROW`). This means calling `ttnn::hardtanh()` at runtime would throw before any kernel is compiled.
2. **Compute API header**: No `hardtanh_tile()` / `hardtanh_tile_init()` functions exist in `tt_metal/hw/inc/api/compute/`.
3. **LLK dispatch**: No `llk_math_eltwise_unary_sfpu_hardtanh.h` exists in `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/`.
4. **Include guard**: No `SFPU_OP_HARDTANH_INCLUDE` is defined in `get_macro_definition()` (HARDTANH falls to `default: "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"`).

The analysis below documents the core SFPU kernel that exists and is ready to be wired into the dispatch pipeline.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported here.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (not wired) | `get_op_init_and_func()` has no case for HARDTANH; the core kernel template `APPROXIMATION_MODE` has no effect on the implementation (no `if constexpr (APPROXIMATION_MODE)` branches) |
| Effective SFPU path | Single code path regardless of approximation mode | The `_calculate_hardtanh_` function template accepts `APPROXIMATION_MODE` but never branches on it -- the implementation is identical for both `true` and `false` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist -- no `hardtanh_tile()` in `tt_metal/hw/inc/api/compute/` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- no `llk_math_eltwise_unary_sfpu_hardtanh.h` in `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (BH) -- both identical |
| **Parameters Dispatch** | This level of abstraction doesn't exist -- no LLK dispatch function calls `_calculate_hardtanh_` |

### Call Chain
The call chain is **incomplete** due to the missing dispatch layers. The intended chain (based on analogous operations like `sinh` and `clamp`) would be:

1. `hardtanh_tile(idst, param0, param1, param2)` (API header) -- does not exist yet
2. `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1, param2)` (LLK dispatch) -- does not exist yet
3. `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>, idst, VectorMode::RC, param0, param1, param2)` (params dispatch via variadic template) -- would use existing `llk_math_eltwise_unary_sfpu_params.h`
4. `ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>(iterations, param0, param1, param2)` (core SFPU kernel) -- **exists and is complete**

The core SFPU kernel takes 3 `uint32_t` parameters (pre-encoded in FP16_B format), plus an iteration count. The `_llk_math_eltwise_unary_sfpu_params_` function provides the iteration count implicitly via the standard per-face call pattern.

Note: The `_calculate_hardtanh_` function signature takes an explicit `iterations` parameter (unlike many other SFPU kernels that use the `ITERATIONS` template parameter). This suggests the LLK dispatch layer would need to pass a runtime iteration count. However, examining analogous parameterized ops like `clamp`, the `iterations` parameter is not passed by the params dispatch -- instead, `_calculate_clamp_` also has this parameter and it gets its value from the standard per-face iteration count. The params dispatch calls `sfpu_func(args...)` where `args` are the forwarded extra arguments. The `iterations` parameter is typically embedded in the kernel's loop and defaults to the template `ITERATIONS=8`.

### Parameters Dispatch Summary
Since no LLK dispatch function exists, this section describes the **expected** dispatch behavior based on the standard `_llk_math_eltwise_unary_sfpu_params_` template used by similar operations.

- **Vector mode**: Would use `VectorMode::RC` (all 4 faces), consistent with standard unary operations.
- **Operation invocation**: The params dispatch function calls the SFPU function once per face with forwarded arguments. For `VectorMode::RC`, it loops 4 times, calling `sfpu_func(args...)` followed by `SETRWC` (on WH: `TTI_SETRWC` with `CR_D, 8`) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (on BH) to advance to the next face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The init function would configure `ADDR_MOD_7` with `dest.incr = 0` (the default for most unary SFPU ops on both WH and BH). The `dst_reg++` in the kernel loop advances the SFPU read/write pointer by 1 sfpi row (2 physical DEST rows, 32 elements) per iteration.
- **Parameter encoding**: The kernel takes 3 parameters (`param0`, `param1`, `param2`), all pre-encoded in FP16_B format. The source code comments state: `param0 = -(neg_threshold)`, `param1 = -(pos_threshold - neg_threshold)`, `param2 = -(pos_threshold)`. However, mathematical analysis reveals that the correct encoding for the algorithm to produce correct results is: `param0 = -min_val`, `param1 = -(max_val - min_val)`, `param2 = +max_val`. See the Mathematical Analysis section for the detailed derivation.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
// (Blackhole version at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{ // APPROXIMATION_MODE unused (no branches on it), ITERATIONS unused (runtime 'iterations' controls loop)
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI (FP16_B mode): broadcast param0 to all SFPU lanes via LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI (FP16_B mode): broadcast param1 to all SFPU lanes via LREG
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI (FP16_B mode): broadcast param2 to all SFPU lanes via LREG
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face (standard)
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        val += p0; // SFPMAD (val * 1.0 + p0): shift value so min_val maps to 0
        v_if (val < 0.0f) // SFPPUSHC + SFPSETCC(LREG_LT0): set CC for lanes where shifted val < 0 (original < min_val)
        {
            val = 0.0f; // SFPLOADI (CC-guarded): zero out lanes below min_val threshold
        }
        v_endif; // SFPPOPC: pop CC stack, restoring unconditional execution

        val += p1; // SFPMAD (val * 1.0 + p1): shift so max_val maps to 0
        v_if (val >= 0.0f) // SFPPUSHC + SFPSETCC(LREG_GTE0): set CC for lanes where shifted val >= 0 (original >= max_val)
        {
            val = 0.0f; // SFPLOADI (CC-guarded): zero out lanes above max_val threshold
        }
        v_endif; // SFPPOPC: pop CC stack

        val += p2; // SFPMAD (val * 1.0 + p2): reverse the shifts, restoring output to correct scale

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 result elements back to DEST row pair

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

### Mathematical Analysis

The hardtanh function is defined as:
```
hardtanh(x, min_val, max_val) = max(min_val, min(max_val, x))
```

The kernel implements this using an additive-shift-and-clamp strategy. The algorithm avoids direct comparisons against arbitrary threshold values (which would require loading the thresholds separately for comparison). Instead, it shifts values so the threshold maps to zero, uses the efficient sign-bit check (`< 0` or `>= 0`) to identify out-of-range lanes, clamps those lanes to zero, and then reverses the shifts.

**Required parameter encoding** (derived from the algorithm, see derivation below):
- `p0 = -min_val`
- `p1 = -(max_val - min_val)`
- `p2 = max_val`

**Note on source code comments**: The source code comments state `param2 = -(pos_threshold)`, but mathematical analysis (below) shows the algorithm only produces correct results when `param2 = +pos_threshold = +max_val`. The comment appears to be erroneous. Since the TTNN dispatch layer is not wired (no host-side parameter encoding exists), this discrepancy has not been exposed at runtime.

**Algorithm derivation (3 cases):**

For default values min_val = -1, max_val = 1: p0 = 1, p1 = -2, p2 = 1.

**Case 1: x in range (min_val <= x <= max_val), e.g., x = 0.5:**
1. val = 0.5 + 1 = 1.5 (not < 0, keep)
2. val = 1.5 + (-2) = -0.5 (not >= 0, keep)
3. val = -0.5 + 1 = 0.5. Output = 0.5. Correct.

**Case 2: x below min (x < min_val), e.g., x = -2:**
1. val = -2 + 1 = -1 (< 0, clamp to 0)
2. val = 0 + (-2) = -2 (not >= 0, keep)
3. val = -2 + 1 = -1. Output = -1 = min_val. Correct.

**Case 3: x above max (x > max_val), e.g., x = 3:**
1. val = 3 + 1 = 4 (not < 0, keep)
2. val = 4 + (-2) = 2 (>= 0, clamp to 0)
3. val = 0 + 1 = 1. Output = 1 = max_val. Correct.

**General proof:**
- If x < min_val: After step 1, val = x - min_val < 0, clamped to 0. After step 2, val = 0 - (max_val - min_val) = -(max_val - min_val) < 0 (since max > min), not clamped. After step 3, val = -(max_val - min_val) + max_val = min_val. Correct.
- If x > max_val: After step 1, val = x - min_val > 0, not clamped. After step 2, val = (x - min_val) - (max_val - min_val) = x - max_val > 0, clamped to 0. After step 3, val = 0 + max_val = max_val. Correct.
- If min_val <= x <= max_val: After step 1, val = x - min_val >= 0, not clamped. After step 2, val = (x - min_val) - (max_val - min_val) = x - max_val <= 0, not clamped. After step 3, val = (x - max_val) + max_val = x. Correct.

### SFPU Instructions Used

| Instruction | Count per iteration | Description |
|-------------|-------------------|-------------|
| `SFPLOAD` | 1 | Load 32 elements from current DEST row pair into LREG for processing |
| `SFPLOADI` | 3 (setup) + 2 (per iteration, CC-guarded) | Load FP16_B immediate: 3x for parameter broadcast before loop, 2x for zero-constant loads inside v_if blocks |
| `SFPMAD` | 3 | Fused multiply-add used for float addition: `val * 1.0 + offset`. Used for all three additive shift operations |
| `SFPSETCC` | 2 | Set condition code based on LREG value: 1x with `LREG_LT0` mode (sign bit test for `< 0`), 1x with `LREG_GTE0` mode (inverted sign bit test for `>= 0`) |
| `SFPPUSHC` | 2 | Push current CC state onto CC stack before each v_if block |
| `SFPPOPC` | 2 | Pop CC state from stack at each v_endif, restoring unconditional execution |
| `SFPSTORE` | 1 | Store 32 result elements from LREG back to current DEST row pair |

Note: The `SFPENCC` instruction is emitted by the SFPI compiler as part of the `v_if`/`v_endif` CC management (to enable CC at the start of `__vCCCtrl` construction and disable it at destruction), but the exact emission depends on compiler optimization. The `SFPCOMPC` instruction is not used because there are no `v_else`/`v_elseif` blocks.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | Used by the compiler for intermediate values. `p0`, `p1`, `p2` are loaded into LREGs before the loop. `val` occupies an LREG during computation. The exact LREG allocation is determined by the SFPI compiler's register allocator. |
| **DEST rows** | Input tile data read via `SFPLOAD` from `dst_reg[0]` (current DEST row pair). Results written back via `SFPSTORE` to the same position. Each iteration processes 2 physical DEST rows (32 elements). |
| **CC register** | Used for conditional execution. Two CC blocks per iteration: one for below-min clamping (`LREG_LT0`), one for above-max clamping (`LREG_GTE0`). CC stack depth reaches 1 (one `SFPPUSHC` per `v_if`). |

### Address Mode Configuration

The address mode for `HARDTANH` would be configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` in the init function. Since `SfpuType::hardtanh` does not match any of the special-cased `if constexpr` branches in the addrmod configuration function (only `topk_local_sort`, `typecast`, `signbit`, and a few others have custom configs), it uses the default configuration:

**Both Wormhole B0 and Blackhole:**
```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

This means the hardware auto-increment for DEST addressing is 0 -- all DEST address advancement is handled explicitly by the `dst_reg++` instruction in the kernel loop (which emits `SETRWC` to advance the DEST read/write counter) and by the per-face `SETRWC`/`_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls in the params dispatch.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Traced dispatch path for HARDTANH -- checked `get_op_approx_mode()`, `get_op_init_and_func_parameterized()`, `get_op_init_and_func_default()`, `get_compute_kernel_path()`, and `get_macro_definition()`
   **Key Findings**: HARDTANH has no case in `get_op_init_and_func_parameterized()` (throws at runtime). `get_op_approx_mode()` returns false (default). `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default). `get_macro_definition()` returns `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Verified `is_parametrized_type(HARDTANH)` returns true, confirming HARDTANH is expected to take parameters
   **Key Findings**: HARDTANH is marked as parametrized but the parameterized dispatch has no implementation.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (lines 281-295)
   **Reason**: Found the `ttnn::hardtanh()` frontend function and its parameter passing
   **Key Findings**: Constructs `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}` with defaults min_val=-1.0, max_val=1.0.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation (Wormhole B0)
   **Key Findings**: Complete `_calculate_hardtanh_` function using SFPI abstractions. Takes 3 FP16_B-encoded parameters. Uses additive-shift-and-clamp algorithm with v_if/v_endif CC blocks.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation (Blackhole)
   **Key Findings**: Identical to Wormhole B0 version.

6. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h`
   **Reason**: Verified `SfpuType::hardtanh` exists in the enum
   **Key Findings**: `SfpuType::hardtanh` is at index 1 in the enum.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Checked address mode configuration for hardtanh
   **Key Findings**: `SfpuType::hardtanh` is not special-cased in `eltwise_unary_sfpu_configure_addrmod()`, so it uses the default ADDR_MOD_7 with all increments = 0.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understood the variadic params dispatch template
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` forwards extra arguments to the SFPU function via perfect forwarding, calls it once per face in VectorMode::RC, advances DEST address between faces.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
   **Reason**: Compared with analogous parametrized operation (clamp) for algorithm understanding
   **Key Findings**: Clamp uses direct min/max comparison with v_if/v_elseif. Hardtanh uses a different additive-shift approach. Both take 3 parameters.

10. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Traced SFPI abstraction layer for v_if/v_endif CC instruction emission and comparison operators
    **Key Findings**: `v_if(val < 0.0f)` emits SFPPUSHC + SFPSETCC(LREG_LT0). `v_if(val >= 0.0f)` emits SFPPUSHC + SFPSETCC(LREG_GTE0). v_endif triggers SFPPOPC via __vCCCtrl destructor.

11. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understood `s2vFloat16b` constructor for uint32_t inputs
    **Key Findings**: `s2vFloat16b(uint32_t)` passes the value directly as a pre-encoded FP16_B immediate, used with SFPLOADI instruction.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU architecture, instruction semantics, CC mechanism, and addressing model
    **Key Findings**: SFPSETCC modes (LREG_LT0=0, LREG_GTE0=4), SFPMAD semantics for float addition, stride-2 addressing, CC stack mechanism.

13. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
    **Reason**: Verified standard UnaryProgramFactory dispatch pattern
    **Key Findings**: Confirmed HARDTANH has no special CB allocation (unlike HARDSHRINK). Confirmed the program factory calls `get_block_defines()` and `get_compute_kernel_path()` which would fail for HARDTANH.
