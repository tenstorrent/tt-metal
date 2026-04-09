## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `THRESHOLD`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `threshold_tile_init(); threshold_tile(0, param0, param1);`

Note: The `threshold_tile_init()` and `threshold_tile()` API header, the `llk_math_eltwise_unary_sfpu_threshold.h` LLK dispatch file, and the corresponding case in `get_op_init_and_func_parameterized()` have been removed from this codebase (nuked for evaluation). The analysis below is reconstructed from the surviving core SFPU kernel (`ckernel_sfpu_threshold.h`), the generic LLK params dispatch infrastructure, existing analogous operations (e.g., `sinh`), and the LLK test helper `sfpu_operations.h` which directly calls `_calculate_threshold_`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(THRESHOLD)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (default) | The API header (nuked) would pass `APPROX` which resolves to the compute config's `math_approx_mode`. The test helper `sfpu_operations.h` passes `APPROX_MODE` as template argument |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but this has no effect | The `_calculate_threshold_` function body does not branch on `APPROXIMATION_MODE` at all -- the same code path executes regardless of its value |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | Nuked from this codebase. In the full repo: `tt_metal/hw/inc/api/compute/eltwise_unary/threshold.h` (would contain `threshold_tile()` and `threshold_tile_init()`) |
| **LLK Dispatch** | Nuked from this codebase. In the full repo: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_threshold.h` (would contain `llk_math_eltwise_unary_sfpu_threshold()` calling `_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h` (identical on Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_threshold.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic parameterized dispatch, shared across all unary SFPU ops) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `threshold_tile_init(); threshold_tile(0, param0, param1);`, calling the API header functions.
2. **API Header** (`threshold.h`, nuked): `threshold_tile_init()` calls `llk_math_eltwise_unary_sfpu_threshold_init<APPROX>()` which invokes the generic `llk_math_eltwise_unary_sfpu_init<SfpuType::threshold, APPROX>()`. `threshold_tile(idst, param0, param1)` calls `llk_math_eltwise_unary_sfpu_threshold<APPROX>(idst, param0, param1)`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_threshold.h`, nuked): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>()` with a lambda that wraps `_calculate_threshold_<APPROX, 8, uint32_t>(param0, param1)`, passing `dst_index` and `VectorMode::RC`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls for SFPU availability, then loops over 4 faces calling the SFPU function once per face with `ITERATIONS=8`, advancing the DEST write pointer by one face stride (via `SETRWC` on WH or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on BH) between faces.
5. **Core SFPU** (`ckernel_sfpu_threshold.h`): `_calculate_threshold_<false, 8, uint32_t>(threshold, value)` reinterprets the uint32_t params as floats via `Converter::as_float()`, then for each of 8 iterations loads the current DEST element, conditionally replaces it with `value` if `in <= threshold`, and advances `dst_reg`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (standard for unary ops).
- **Operation invocation**: The SFPU function `_calculate_threshold_<APPROX, 8, uint32_t>(param0, param1)` is called once per face (4 times total). Each call processes 8 SFPU iterations (ITERATIONS=8), covering one 16x16 face (8 iterations x 32 elements/iteration = 256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces on Wormhole / `inc_dst_addr<8>` x2 between faces on Blackhole).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h

namespace ckernel::sfpu
{

template <typename T>
constexpr bool is_supported_threshold_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

template <bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _calculate_threshold_(T threshold, T value) // APPROXIMATION_MODE=false, ITERATIONS=8, T=uint32_t (from TTNN dispatch)
{
    static_assert(is_supported_threshold_type_v<T>, "Type T must be either float or uint32_t");

    sfpi::vFloat v_threshold;
    sfpi::vFloat v_value;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold; // SFPLOADI: load float scalar into LREG
        v_value     = value;     // SFPLOADI: load float scalar into LREG
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>) // This branch taken when called from TTNN with uint32_t params
    {
        v_threshold = Converter::as_float(threshold); // Bitcast uint32_t to float, then SFPLOADI into LREG
        v_value     = Converter::as_float(value);     // Bitcast uint32_t to float, then SFPLOADI into LREG
    }
#pragma GCC unroll 8 // Compiler hint to fully unroll the 8 iterations
    for (int d = 0; d < ITERATIONS; d++) // 8 iterations per face
    {
        sfpi::vFloat in = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        v_if (in <= v_threshold) // SFPPUSHC + SFPENCC + SFPMAD(in - threshold) + SFPSETCC(LTE) -- CC guarded region
        {
            sfpi::dst_reg[0] = v_value; // SFPSTORE: write replacement value to DEST (CC-guarded, only lanes where in <= threshold)
        }
        v_endif; // SFPPOPC: restore CC state

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

} // namespace ckernel::sfpu
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler for this kernel. Since the kernel uses SFPI abstractions, exact instruction selection is compiler-dependent, but the semantic mapping is well-defined.

| Instruction | Source Expression | Description |
|-------------|-------------------|-------------|
| `SFPLOADI` | `v_threshold = scalar; v_value = scalar;` | Load 16-bit immediate (float constant) into LREG. Two SFPLOADI pairs needed per float constant (hi16 + lo16) for full 32-bit precision |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG |
| `SFPMAD` | `in <= v_threshold` (comparison) | The `<=` comparison is implemented by the SFPI compiler as a subtraction `in - v_threshold` via SFPMAD (a * 1.0 + b), followed by sign/zero testing |
| `SFPSETCC` | `in <= v_threshold` (comparison) | Set per-lane CC based on the subtraction result (LTE: result <= 0 means `in <= threshold`) |
| `SFPPUSHC` | `v_if` | Push current CC state onto the CC stack before entering the conditional block |
| `SFPENCC` | `v_if` | Enable CC masking (activates per-lane conditional execution) |
| `SFPSTORE` | `sfpi::dst_reg[0] = v_value` (write) | Store LREG value (replacement value) to current DEST row pair. CC-guarded: only executes on lanes where `in <= threshold` |
| `SFPPOPC` | `v_endif` | Pop CC state from stack, restoring CC to pre-`v_if` state |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (v_threshold)** | Holds the threshold parameter as a vFloat. Loaded once before the iteration loop via SFPLOADI. Remains constant across all iterations. |
| **LREG (v_value)** | Holds the replacement value parameter as a vFloat. Loaded once before the iteration loop via SFPLOADI. Remains constant across all iterations. |
| **LREG (in)** | Temporary register holding the current tile element loaded from DEST via SFPLOAD. Overwritten each iteration. Also used as the source for the comparison subtraction. |
| **DEST (dst_reg)** | Source and destination for tile data. Read via SFPLOAD at each iteration, conditionally written via SFPSTORE with the replacement value. The DEST pointer auto-advances by 1 sfpi row (2 physical rows) per iteration via `dst_reg++`. |
| **CC stack** | Used by `v_if`/`v_endif` to save/restore condition code state. One level of CC stack used (single `v_if` block, no nesting). |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod()` (in `llk_math_eltwise_unary_sfpu.h`), called during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::threshold>()`.

| Hardware | Address Mode | Configuration |
|----------|-------------|---------------|
| **Wormhole B0** | `ADDR_MOD_7` | `srca.incr = 0, srcb.incr = 0, dest.incr = 0` |
| **Blackhole** | `ADDR_MOD_7` | `srca.incr = 0, srcb.incr = 0, dest.incr = 0` |

The `ADDR_MOD_7` with all-zero increments is the standard configuration for SFPU unary operations. The DEST address advancement is handled explicitly by the SFPI `dst_reg++` instruction within the kernel (which maps to DEST RWC increment) and by the `SETRWC` (Wormhole) or `inc_dst_addr<8>` (Blackhole) calls between faces in the parameters dispatch layer.

`SfpuType::threshold` does not appear in any of the special-case `if constexpr` blocks in `eltwise_unary_sfpu_configure_addrmod()`, so it uses only the default `ADDR_MOD_7` configuration. No additional address modes (e.g., `ADDR_MOD_6`) are configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: Core SFPU implementation of the threshold operation (Wormhole B0)
   **Key Findings**: Uses SFPI abstractions (vFloat, v_if/v_endif, dst_reg). Simple conditional: loads element, if <= threshold replaces with value. Template params: APPROXIMATION_MODE (unused in body), ITERATIONS (8 per face), T (float or uint32_t). Identical on Blackhole.

2. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: Core SFPU implementation of the threshold operation (Blackhole)
   **Key Findings**: Identical to Wormhole B0 version.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Converter utility used by threshold kernel to bitcast uint32_t to float
   **Key Findings**: `Converter::as_float(uint32_t)` uses union-based reinterpret cast.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Generic parameterized SFPU dispatch layer (handles VectorMode, face iteration, DEST addressing)
   **Key Findings**: For VectorMode::RC, loops over 4 faces, calls SFPU function once per face, advances DEST by face stride between faces using SETRWC.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base LLK infrastructure for unary SFPU operations (init, addr_mod config, start/done)
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::threshold>()` sets ADDR_MOD_7 with all-zero increments. No special-case addr_mod for threshold.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Blackhole version of base LLK infrastructure
   **Key Findings**: Same ADDR_MOD_7 configuration. Blackhole uses `_llk_math_eltwise_unary_sfpu_start_` / `_llk_math_eltwise_unary_sfpu_done_` instead of explicit `set_addr_mod_base()` / `clear_addr_mod_base()`.

7. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/sfpu_operations.h`
   **Reason**: Test helper that directly calls `_calculate_threshold_` to verify dispatch pattern
   **Key Findings**: `case SfpuType::threshold: _calculate_threshold_<APPROX_MODE, ITERATIONS>(5.0f, 10.0f);` -- confirms no init function needed, threshold called with float params directly in test context.

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine approximation mode and compute kernel path for UnaryOpType::THRESHOLD
   **Key Findings**: `get_op_approx_mode()` returns `false` (default). `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` (default). The `get_op_init_and_func_parameterized()` case for THRESHOLD has been nuked from this codebase.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: SFPI C++ abstraction layer defining vFloat, v_if/v_endif, dst_reg, comparison operators
   **Key Findings**: `v_if` expands to SFPPUSHC + SFPENCC + comparison condition. `v_endif` triggers SFPPOPC via `~__vCCCtrl` destructor. `vFloat <= vFloat` comparison uses `__builtin_rvtt_sfpxfcmpv` which compiles to SFPMAD (subtraction) + SFPSETCC (sign test).

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU architecture, instruction semantics, register layout
    **Key Findings**: SFPMAD is used for both addition and subtraction (no dedicated add instruction). SFPSETCC sets per-lane CC. SFPPUSHC/SFPPOPC manage CC stack for nested conditionals. SFPLOADI loads 16-bit immediates.

11. **File**: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/threshold_tile.rst`
    **Reason**: API documentation confirming the threshold tile function signature
    **Key Findings**: `threshold_tile(uint32_t idst, uint32_t param0, uint32_t param1)` -- two uint32_t parameters (threshold value and replacement value, both bitcast from float).
