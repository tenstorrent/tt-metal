## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**IMPORTANT: Disconnected Operation Notice**

The `threshold` operation has a working core SFPU kernel (`_calculate_threshold_` in `ckernel_sfpu_threshold.h`) but is **not connected** through the full TTNN unary dispatch chain. Specifically:

- `UnaryOpType::THRESHOLD` exists in the enum (`unary_op_types.hpp:112`) but is **never referenced** anywhere in the codebase via `UnaryOpType::THRESHOLD`.
- No `threshold_tile()` or `threshold_tile_init()` API functions are defined in the compute kernel API headers.
- No LLK dispatch function (e.g., `llk_math_eltwise_unary_sfpu_threshold`) exists.
- The `SfpuType::threshold` enum value exists only in the LLK test helper (`tests/helpers/include/llk_sfpu_types.h`), not in the metal ckernel layer's `llk_sfpu_types.h`.
- `ttnn.threshold` does not exist as a Python attribute at runtime (confirmed: `AttributeError: module 'ttnn' has no attribute 'threshold'`).
- The `bind_unary_threshold` nanobind template in `unary_nanobind.cpp` is defined but never instantiated.
- `get_op_init_and_func_default()` and `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp` have no case for `THRESHOLD`.

The analysis below documents the core SFPU kernel that **would** be used if the operation were fully wired up.

### Unary Dispatch Summary
- **UnaryOpType**: `THRESHOLD` (defined but unused)
- **Compute kernel**: `eltwise_sfpu.cpp` (default for all ops via `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: Would be `threshold_tile(0, param0, param1)` if connected (currently no such function exists)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_op_utils.cpp` -- the switch has only a `default: return false` case, so THRESHOLD returns `false` |
| Template parameter (SFPU_OP_CHAIN) | none (not connected) | No case for THRESHOLD in `get_op_init_and_func()` -- operation is not wired |
| Effective SFPU path | N/A -- operation is disconnected | If connected, `APPROXIMATION_MODE` would be passed as the first template parameter to `_calculate_threshold_`. The kernel does not branch on `APPROXIMATION_MODE` (it is unused in the implementation). |

### SFPU Abstraction Layers
The threshold operation is missing the top two abstraction layers. Only the core SFPU implementation exists.

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist (no `threshold_tile()` function is defined anywhere) |
| **LLK Dispatch** | This level of abstraction doesn't exist (no `llk_math_eltwise_unary_sfpu_threshold` function) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h` (identical on Blackhole: `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_threshold.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic dispatch template, would be used if threshold were connected) |

### Call Chain
The threshold SFPU kernel is currently only callable from the LLK test infrastructure, not from the TTNN unary dispatch chain.

**LLK test path** (the only working path):
1. `sfpu_operations.h` test dispatch function calls `_calculate_threshold_<APPROX_MODE, ITERATIONS>(5.0f, 10.0f)` directly with hardcoded test parameters.

**Intended TTNN path** (not connected -- would require implementation):
1. `threshold_tile(idst, param0, param1)` (API header -- does not exist) would call
2. `llk_math_eltwise_unary_sfpu_threshold<APPROX>()` (LLK dispatch -- does not exist) which would call
3. `_llk_math_eltwise_unary_sfpu_params_()` with a lambda wrapping `_calculate_threshold_<APPROX, 8>(threshold, value)` (parameters dispatch), which would call
4. `_calculate_threshold_<APPROXIMATION_MODE, ITERATIONS>(threshold, value)` (core SFPU function) for each face of the tile.

### Parameters Dispatch Summary
Since threshold is not connected, the parameters dispatch behavior is described for the **generic** `_llk_math_eltwise_unary_sfpu_params_` template that would be used if the operation were integrated.

- **Vector mode**: Standard unary operations use `VectorMode::RC`, which processes all 4 faces of the tile. The params dispatch iterates over 4 faces, calling the SFPU function once per face.
- **Operation invocation**: The SFPU function (`_calculate_threshold_`) would be called 4 times (once per face), each invocation running `ITERATIONS=8` loop iterations to cover 8 sfpi rows per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces on Wormhole / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h

namespace ckernel::sfpu
{

template <typename T>
constexpr bool is_supported_threshold_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

template <bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _calculate_threshold_(T threshold, T value) // APPROXIMATION_MODE unused, ITERATIONS=8
{
    static_assert(is_supported_threshold_type_v<T>, "Type T must be either float or uint32_t");

    sfpi::vFloat v_threshold; // Broadcast scalar threshold to all SFPU lanes
    sfpi::vFloat v_value;     // Broadcast scalar replacement value to all SFPU lanes
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold; // Direct float assignment -> SFPLOADI (2 instructions for 32-bit)
        v_value     = value;     // Direct float assignment -> SFPLOADI (2 instructions for 32-bit)
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        v_threshold = Converter::as_float(threshold); // Reinterpret uint32 bits as float
        v_value     = Converter::as_float(value);     // Reinterpret uint32 bits as float
    }
#pragma GCC unroll 8                          // Compiler hint to fully unroll the loop
    for (int d = 0; d < ITERATIONS; d++)      // 8 iterations per face
    {
        sfpi::vFloat in = sfpi::dst_reg[0];   // SFPLOAD: load 32 elements from current DEST row pair

        v_if (in <= v_threshold)              // SFPMAD (subtract) + SFPSETCC + SFPENCC/SFPPUSHC (CC setup)
        {
            sfpi::dst_reg[0] = v_value;       // SFPSTORE (predicated): replace with value where condition holds
        }
        v_endif;                              // SFPPOPC/SFPENCC (CC teardown): restore unconditional execution

        sfpi::dst_reg++;                      // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

} // namespace ckernel::sfpu
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the abstractions in this kernel:

| Instruction | Source Construct | Description |
|-------------|-----------------|-------------|
| `SFPLOADI` | `v_threshold = threshold; v_value = value;` | Load 16-bit immediate to LREG (two instructions per 32-bit float: hi16 + lo16) |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG |
| `SFPMAD` | `in <= v_threshold` (comparison) | Compute `in - v_threshold` via fused multiply-add (`in * 1.0 + (-v_threshold)`), used to determine comparison sign |
| `SFPSETCC` | `in <= v_threshold` (comparison) | Set per-lane CC.Res based on sign of the subtraction result (LTE test) |
| `SFPENCC` | `v_if` / `v_endif` | Enable/disable condition code masking for predicated execution |
| `SFPPUSHC` | `v_if` | Push current CC state onto CC stack to establish a conditional scope |
| `SFPSTORE` | `sfpi::dst_reg[0] = v_value` | Store `v_value` back to current DEST row pair (predicated -- only executes for lanes where `in <= threshold`) |
| `SFPPOPC` | `v_endif` | Pop CC state from stack, restoring unconditional execution |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input elements are read from and written back to the current DEST row pair. The SFPU processes 32 elements (2 physical rows x 16 columns) per iteration via stride-2 addressing. |
| **LREG (v_threshold)** | Holds the broadcast threshold value. Loaded once before the loop via `SFPLOADI` (2 instructions for full 32-bit float). Persists across all 8 iterations. |
| **LREG (v_value)** | Holds the broadcast replacement value. Loaded once before the loop via `SFPLOADI`. Persists across all 8 iterations. |
| **LREG (in)** | Temporary register holding the loaded DEST element for each iteration. Loaded via `SFPLOAD`, used in comparison, then discarded (overwritten next iteration). |
| **LREG (comparison result)** | Implicit temporary used by `SFPMAD` during the `<=` comparison (holds `in - v_threshold`). Used by `SFPSETCC` to set CC. |
| **CC (Condition Code)** | Per-lane condition code bits used for predicated execution. `CC.En` is toggled by `SFPENCC`; `CC.Res` is set by `SFPSETCC` based on the comparison. The CC stack is used (push/pop) to scope the conditional block. |

### Address Mode Configuration

The threshold operation, if connected through the standard unary SFPU dispatch, would use the default address mode configuration from `eltwise_unary_sfpu_configure_addrmod()`:

**ADDR_MOD_7** (set for all unary SFPU operations):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

The threshold operation does NOT fall into any of the special-case branches (`topk_local_sort`, `typecast`, `signbit`, etc.) that configure `ADDR_MOD_6`. The DEST address increment is zero because the SFPI `dst_reg++` construct handles DEST pointer advancement directly via software (not hardware auto-increment). This is the same for both Wormhole and Blackhole.

Note on Blackhole: The Blackhole version of `eltwise_unary_sfpu_configure_addrmod()` additionally includes `SfpuType::reciprocal` in the `ADDR_MOD_6` special cases, but this does not affect the threshold operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Verify that `UnaryOpType::THRESHOLD` exists in the enum
   **Key Findings**: THRESHOLD is at line 112 in the UnaryOpType enum

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Check dispatch path for THRESHOLD (get_op_approx_mode, get_op_init_and_func, get_compute_kernel_path)
   **Key Findings**: No case for THRESHOLD in any dispatch function. get_op_approx_mode returns false (default), get_compute_kernel_path returns "eltwise_sfpu.cpp" (default). THRESHOLD is completely unhandled.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: Read the core SFPU kernel implementation
   **Key Findings**: Simple conditional replacement kernel using SFPI abstractions. If input <= threshold, replace with value. Template takes APPROXIMATION_MODE (unused) and ITERATIONS parameters. Supports float and uint32_t parameter types via Converter::as_float.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_threshold.h`
   **Reason**: Check if Blackhole implementation differs from Wormhole
   **Key Findings**: Identical implementation to Wormhole B0 version

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand the Converter::as_float utility used in the uint32_t branch
   **Key Findings**: Simple union-based type punning from uint32_t to float (bit reinterpretation)

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand the LLK-level SFPU unary infrastructure (addr_mod config, start/done/init)
   **Key Findings**: Default addr_mod sets ADDR_MOD_7 with all increments = 0. Threshold not special-cased.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the generic parameters dispatch that would be used for threshold
   **Key Findings**: VectorMode::RC iterates 4 faces with SETRWC between faces. Each face calls the SFPU function once.

8. **File**: `tt_metal/third_party/tt_llk/tests/helpers/include/sfpu_operations.h`
   **Reason**: Find how the LLK test infrastructure calls the threshold kernel
   **Key Findings**: Test dispatch calls `_calculate_threshold_<APPROX_MODE, ITERATIONS>(5.0f, 10.0f)` with hardcoded test parameters.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Check if SfpuType::threshold exists in the metal compute kernel API
   **Key Findings**: Only has 4 entries (unused, frac, swish, atanh, sinh). No threshold entry.

10. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
    **Reason**: Check Python binding status for threshold
    **Key Findings**: `bind_unary_threshold` template is defined (lines 1599-1655) but never instantiated. Uses `unary_composite_3param_to_4param_wrapper` pattern.

11. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: Understand SFPI vFloat comparison operators and v_if/v_endif macros
    **Key Findings**: `vFloat <= vFloat` maps to `__builtin_rvtt_sfpxfcmpv` with `SFPXCMP_MOD1_CC_LTE` mode. `v_if` pushes CC state, `v_endif` pops it.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU instruction semantics, CC mechanism, and addressing model
    **Key Findings**: Stride-2 addressing, 8 iterations per face, v_if/v_endif maps to SFPENCC/SFPPUSHC/SFPSETCC/SFPPOPC/SFPENCC sequence.
