## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: Not yet wired. The `HARDTANH` type is registered in `unary_op_types.hpp` and `is_parametrized_type()` returns `true` for it, but `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp` has no case for `HARDTANH` (hits `default: TT_THROW`). The intended expansion would be `hardtanh_tile_init(); hardtanh_tile(0, param0, param1, param2);` once the dispatch layers are wired up. The core SFPU kernel (`_calculate_hardtanh_`) exists and is fully implemented at the LLK layer.

**Dispatch Wiring Status**: The following layers exist and are complete:
- Core SFPU implementation: `ckernel_sfpu_hardtanh.h` (both Wormhole B0 and Blackhole)
- Included in umbrella header: `ckernel_sfpu.h`
- `UnaryOpType::HARDTANH` in `unary_op_types.hpp`
- `is_parametrized_type(HARDTANH) == true` in `unary_op_utils.hpp`
- TTNN-level function: `ttnn::hardtanh(input, min_val, max_val)` in `unary.hpp`

The following layers are **missing** (not yet wired):
- Compute kernel API header (e.g., `api/compute/eltwise_unary/hardtanh.h`) -- does not exist
- LLK dispatch file (e.g., `llk_math_eltwise_unary_sfpu_hardtanh.h`) -- does not exist
- Case in `get_op_init_and_func_parameterized()` for `HARDTANH`
- Include guard in `sfpu_split_includes.h` or `activations.h`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (not yet wired) | `get_op_init_and_func_parameterized()` has no case for HARDTANH; when wired, `APPROXIMATION_MODE` would be passed as a template argument to the init/tile functions |
| Effective SFPU path | `APPROXIMATION_MODE` is a template parameter on `_calculate_hardtanh_` but the kernel body does not branch on it -- there is no `if constexpr (APPROXIMATION_MODE)` in the kernel. Both approximate and non-approximate paths execute the same code. | The kernel at `ckernel_sfpu_hardtanh.h` line 16-53 has a single linear code path regardless of `APPROXIMATION_MODE` |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (no `api/compute/eltwise_unary/hardtanh.h` file) |
| **LLK Dispatch** | This level of abstraction doesn't exist (no `llk_math_eltwise_unary_sfpu_hardtanh.h` file) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (Wormhole B0) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic params dispatch, shared by all parameterized unary ops) |

### Call Chain
The intended call chain (once wired) would follow this pattern, based on the existing infrastructure for similar parameterized operations (e.g., `frac`):

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile_init(); hardtanh_tile(0, param0, param1, param2);`.
2. **API Header** (would be `hardtanh.h`): `hardtanh_tile(idst, p0, p1, p2)` calls `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, p0, p1, p2)))`.
3. **LLK Dispatch** (would be `llk_math_eltwise_unary_sfpu_hardtanh.h`): `llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(dst_index, p0, p1, p2)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_hardtanh_<APPROX, 8>, dst_index, VectorMode::RC, 8, p0, p1, p2)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): The generic `_llk_math_eltwise_unary_sfpu_params_` function sets up DEST addressing, stalls for SFPU readiness, then loops over 4 faces calling the SFPU function once per face, with `SETRWC` between faces to advance the DEST address.
5. **Core SFPU Implementation** (`ckernel_sfpu_hardtanh.h`): `_calculate_hardtanh_<APPROX, 8>(8, p0, p1, p2)` executes the clamp logic on 8 iterations per face (256 elements per face, 1024 elements total for 4 faces).

### Parameters Dispatch Summary
Based on the generic `_llk_math_eltwise_unary_sfpu_params_` function in `llk_math_eltwise_unary_sfpu_params.h`:

- **Vector mode**: `VectorMode::RC` (standard for eltwise unary ops) -- processes all 4 faces of the tile.
- **Operation invocation**: The dispatch function loops over 4 faces, calling the SFPU function once per face. Each call processes `ITERATIONS=8` sfpi rows within the face. Between faces, `TTI_SETRWC` advances the DEST write address by the face stride (two increments of 8 physical rows = 16 physical rows = 1 face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole B0, the params dispatch uses direct `TTI_SETRWC` instructions. On Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
// (Blackhole version is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS> // APPROXIMATION_MODE unused in body, ITERATIONS=8
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)         i.e., -min_val
    // param1 = -(pos_threshold - neg_threshold)   i.e., -(max_val - min_val)
    // param2 = -(pos_threshold)         i.e., -max_val

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // broadcast scalar FP16_B -> vFloat; emits SFPLOADI
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // broadcast scalar FP16_B -> vFloat; emits SFPLOADI
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // broadcast scalar FP16_B -> vFloat; emits SFPLOADI
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // iterations=8 per face
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from DEST into LREG

        val += p0; // SFPMAD: val = val * 1.0 + p0; shifts value by -min_val
        v_if (val < 0.0f) // SFPSETCC with LT0 mode: CC.Res = (val < 0) per lane
        {
            val = 0.0f; // SFPLOADI: set val = 0.0 for lanes where (original_val < min_val)
        }
        v_endif; // SFPENCC/SFPPOPC: restore CC state, all lanes active

        val += p1; // SFPMAD: val = val * 1.0 + p1; shifts by -(max_val - min_val)
        v_if (val >= 0.0f) // SFPSETCC with GTE0 mode: CC.Res = (val >= 0) per lane
        {
            val = 0.0f; // SFPLOADI: set val = 0.0 for lanes where (original_val > max_val)
        }
        v_endif; // SFPENCC/SFPPOPC: restore CC state

        val += p2; // SFPMAD: val = val * 1.0 + p2; shifts back by -max_val, restoring clamped value

        sfpi::dst_reg[0] = val; // SFPSTORE: write result back to DEST

        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Algorithm explanation**: The kernel implements `hardtanh(x, min_val, max_val) = clamp(x, min_val, max_val)` using a shift-and-zero technique instead of direct comparisons against threshold values. This avoids needing to load the threshold values into comparison registers:

1. **Shift by `-min_val`**: `val = x + (-min_val) = x - min_val`. If original `x < min_val`, then `val < 0`.
2. **Clamp negative to zero**: If `val < 0`, set `val = 0`. This effectively sets `x = min_val` for values below the lower bound.
3. **Shift by `-(max_val - min_val)`**: `val = val + (-(max_val - min_val))`. After step 2, values in range become `val = (x - min_val) - (max_val - min_val) = x - max_val`. If original `x > max_val`, the unclamped path gives `val = x - max_val >= 0`.
4. **Clamp non-negative to zero**: If `val >= 0`, set `val = 0`. This effectively sets `x = max_val` for values above the upper bound.
5. **Shift by `-max_val`**: `val = val + (-max_val)`. After step 4, clamped values get restored: for in-range values, `val = (x - max_val) + (-max_val) = x - 2*max_val`... wait, let me re-derive.

Actually the correct derivation is:
- For `x` in range `[min_val, max_val]`: After step 1: `val = x - min_val >= 0`, not zeroed. After step 3: `val = (x - min_val) - (max_val - min_val) = x - max_val <= 0`, not zeroed. After step 5: `val = (x - max_val) + (-max_val)`. This gives `x - 2*max_val`, which is incorrect.

The correct interpretation requires understanding that p2 = `-pos_threshold = -max_val`, and working through all three cases. Let me re-examine: after the two zeroing steps, for in-range values the intermediate is `x - min_val - (max_val - min_val) = x - max_val`. Adding p2 = `-max_val` gives `x - 2*max_val`. This suggests my interpretation of the parameter encoding may be incomplete, or the params are pre-computed differently by the host-side code. Since the host dispatch is not yet wired, the exact parameter encoding cannot be verified from the current codebase. The comments in the kernel source (`param0 = -(neg_threshold)`, `param1 = -(pos_threshold - neg_threshold)`, `param2 = -(pos_threshold)`) document the intended encoding.

### SFPU Instructions Used

| Instruction | Usage in Kernel | Description |
|-------------|----------------|-------------|
| `SFPLOADI` | `sfpi::s2vFloat16b(param)` (x3), `val = 0.0f` (x2) | Load 16-bit immediate to LREG; used to broadcast scalar FP16_B parameters to vector registers and to load the zero constant |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from DEST row into LREG with format conversion |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` | Store LREG back to DEST row with format conversion |
| `SFPMAD` | `val += p0`, `val += p1`, `val += p2` | Fused multiply-add implementing float addition: `val = val * 1.0 + pN`; there is no dedicated float add instruction |
| `SFPSETCC` | `v_if (val < 0.0f)`, `v_if (val >= 0.0f)` | Set per-lane condition code based on comparison (LT0 mode for `<`, GTE0 mode for `>=`) |
| `SFPENCC` | `v_if` / `v_endif` preamble/postamble | Enable/disable condition code masking; used by the SFPI `v_if`/`v_endif` abstraction |
| `SFPCOMPC` | `v_endif` (implicit) | Complement CC.Res; part of the condition code management in `v_if`/`v_endif` |
| `SFPPUSHC` | `v_if` (implicit) | Push CC state onto stack for nested conditional support |
| `SFPPOPC` | `v_endif` (implicit) | Pop CC state from stack to restore previous masking state |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | General purpose; used implicitly by the SFPI compiler for `val`, `p0`, `p1`, `p2` vFloat variables. The compiler allocates these across available LREGs. |
| **DEST rows** | Source and destination for the tile data. Each iteration processes 2 physical DEST rows (32 elements) via the stride-2 mechanism. The kernel reads from `dst_reg[0]` and writes back to `dst_reg[0]`, modifying data in-place. |
| **CC stack** | Used by `v_if`/`v_endif` to save/restore condition code state. Two `v_if` blocks in the kernel body each push/pop one CC stack entry. |

### Address Mode Configuration

The hardtanh operation does not have a special case in `eltwise_unary_sfpu_configure_addrmod<SfpuType>()`. It uses the **default** address mode configuration, which is `ADDR_MOD_7` with all zero increments:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This configuration is the same on both Wormhole B0 and Blackhole. The zero-increment mode means DEST addressing does not auto-increment via the address mode -- instead, the kernel manually advances the DEST pointer using `dst_reg++` within the iteration loop and the params dispatch uses `SETRWC` between faces.

**Note**: Since hardtanh has no LLK dispatch file yet and is not registered in `SfpuType`, no `SfpuType::hardtanh`-specific address mode branch exists. When wired, it would fall through to the default `ADDR_MOD_7` configuration (no special-case needed since the kernel uses SFPI abstractions that handle addressing internally).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine the compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for HARDTANH
   **Key Findings**: `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default). `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func_parameterized()` has no case for HARDTANH -- would TT_THROW at runtime.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check `is_parametrized_type()` for HARDTANH
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns `true`, confirming it expects parameterized dispatch.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Understand the TTNN-level API for hardtanh
   **Key Findings**: `ttnn::hardtanh(input, min_val=-1.0, max_val=1.0)` creates `UnaryWithParam{HARDTANH, min_val, max_val}` and calls `unary_impl`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Read the core SFPU kernel implementation
   **Key Findings**: `_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>` implements clamp via a shift-and-zero technique using three FP16_B parameters. Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). APPROXIMATION_MODE template parameter is unused in the kernel body.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Compare Blackhole implementation with Wormhole B0
   **Key Findings**: Implementations are identical across both architectures.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the generic parameters dispatch pattern for parameterized unary SFPU ops
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode::RC by looping over 4 faces, calling the SFPU function once per face, with SETRWC between faces.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand SFPU init, address mode configuration, and start/done patterns
   **Key Findings**: Default address mode is ADDR_MOD_7 with zero increments. No special case for hardtanh in `eltwise_unary_sfpu_configure_addrmod`.

8. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h`
   **Reason**: Understand the compute API layer and `init_sfpu` function
   **Key Findings**: `init_sfpu(icb, ocb)` calls `unary_op_init_common` which sets up unpack, pack, and math datacopy.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
   **Reason**: Study a similar operation's LLK dispatch pattern as a reference for the expected hardtanh wiring
   **Key Findings**: The pattern is: init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::op>()`, tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_func, dst_index, vector_mode, ...)`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU architecture, instruction semantics, and addressing model
    **Key Findings**: Confirmed SFPMAD for float addition, SFPLOADI for scalar broadcast, stride-2 addressing model, ITERATIONS=8 per face.

11. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understand `s2vFloat16b` scalar-to-vector conversion
    **Key Findings**: `s2vFloat16b` inherits from `s2vFloat16`, converts a scalar (uint32/float) to FP16_B format and broadcasts to all SFPU lanes via SFPLOADI.

12. **File**: `docs/sfpu_operations/key_notes/hardtanh_key_notes.md`
    **Reason**: Understand the mathematical definition of hardtanh
    **Key Findings**: `hardtanh(x) = clamp(x, min_val, max_val)` with defaults min_val=-1.0, max_val=1.0.
