## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**IMPORTANT: Incomplete Dispatch Chain.** The `HARDTANH` operation has an incomplete dispatch chain in the current codebase. The `UnaryOpType::HARDTANH` enum value exists and `is_parametrized_type()` returns `true`, but no case exists in `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()` -- calling `get_op_init_and_func()` with `HARDTANH` at runtime would hit the `default: TT_THROW("unexpected parameterized op type {}", op_type)` branch. The core SFPU kernel `_calculate_hardtanh_` exists in `tt_llk` but the upper dispatch layers (compute API, LLK dispatch) are missing.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default from `get_compute_kernel_path()` which falls through to `default: return "eltwise_sfpu.cpp"`)
- **SFPU_OP_CHAIN_0 expansion**: **Not wired** -- no `hardtanh_tile(idst, ...)` function exists. The intended expansion would be `hardtanh_tile(0, param0, param1, param2)` based on the pattern used by similar parametrized operations.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (dispatch chain incomplete) | No `get_op_init_and_func` case exists for `HARDTANH`. The core SFPU kernel template declares `APPROXIMATION_MODE` but does not use it -- there is no `if constexpr` branch or any reference to the parameter in the function body. |
| Effective SFPU path | Single code path regardless of `APPROXIMATION_MODE` value | The `_calculate_hardtanh_` function body has no conditional branching based on `APPROXIMATION_MODE`. The same subtraction-clamp-add logic executes unconditionally. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | This level of abstraction doesn't exist -- no `hardtanh.h` in `tt_metal/hw/inc/api/compute/eltwise_unary/` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- no `llk_math_eltwise_unary_sfpu_hardtanh.h` in `tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h` -- identical) |
| **Parameters Dispatch** | This level of abstraction doesn't exist for hardtanh specifically. The generic template is at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) and `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole). |

### Call Chain
The full call chain is **incomplete** in the current codebase. The intended chain (based on the pattern used by working operations like `frac`) would be:

1. **`hardtanh_tile(idst, param0, param1, param2)`** -- compute API header function (does not exist). Would wrap the LLK call with `MATH((...))`.
2. **`llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1, param2)`** -- LLK dispatch function (does not exist). Would call `_llk_math_eltwise_unary_sfpu_params_` with the core SFPU function as a callable.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROX>(...)`** -- generic parameters dispatch template at `llk_math_eltwise_unary_sfpu_params.h`. Handles DEST addressing, face iteration, and stall synchronization. Calls the SFPU function once per face in `VectorMode::RC` mode.
4. **`_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>(iterations, param0, param1, param2)`** -- core SFPU kernel at `ckernel_sfpu_hardtanh.h`. Performs the actual element-wise clamping computation.

Only layers 3 (generic template) and 4 (core kernel) exist. Layers 1 and 2 need to be implemented for end-to-end functionality.

### Parameters Dispatch Summary
Since the hardtanh-specific LLK dispatch does not exist, this section documents the **expected** behavior based on the generic `_llk_math_eltwise_unary_sfpu_params_` template used by similar operations:

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (faces 0-3, each 16x16 = 256 elements).
- **Operation invocation**: The parameters dispatch calls the SFPU function once per face (4 times total for RC mode). Each invocation processes `ITERATIONS=8` sfpi rows within one face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC/inc_dst_addr between faces). On Wormhole, `TTI_SETRWC` advances by 8+8=16 physical DEST rows between faces. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
// (Blackhole version at tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS> // APPROXIMATION_MODE is unused in this kernel, ITERATIONS=8
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{
    // All params are in FP16_B format (bfloat16 encoded as uint32_t in lower 16 bits)
    // param0 = -low           (negation of lower clamp bound)
    // param1 = -(high - low)  (negation of the clamping range width)
    // param2 = high           (upper clamp bound; see Algorithm Note below)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: broadcast -low into all SFPU lanes as FP16_B
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: broadcast -(high-low) into all SFPU lanes
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: broadcast high into all SFPU lanes
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face, processing 32 elements each
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements (2 physical DEST rows) into LREG

        val += p0; // SFPADD: val = x + (-low) = x - low; shifts value so lower bound maps to 0
        v_if (val < 0.0f) // SFPPUSHC + SFPENCC + SFPXFCMPS: enable CC, test if (x - low) < 0
        {
            val = 0.0f; // SFPLOADI: for lanes where x < low, zero out the shifted value
        }
        v_endif; // SFPPOPC: pop CC stack, restore all-lanes-active state

        val += p1; // SFPADD: val = val + (low - high); shifts so upper bound maps to 0
        v_if (val >= 0.0f) // SFPPUSHC + SFPENCC + SFPXFCMPS: enable CC, test if val >= 0 (x >= high)
        {
            val = 0.0f; // SFPLOADI: for lanes where x >= high, zero out the shifted value
        }
        v_endif; // SFPPOPC: pop CC stack, restore all-lanes-active state

        val += p2; // SFPADD: val = val + high; final offset restores correct output values

        sfpi::dst_reg[0] = val; // SFPSTORE: write 32 result elements back to current DEST row pair

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}
```

**Algorithm: Subtraction-Clamp-Add Trick**

The `hardtanh(x, low, high) = clamp(x, low, high)` function is implemented using three additions and two conditional zeroing operations, avoiding direct comparison against non-zero threshold values. The key insight is that comparing against 0 is cheaper (uses the sign bit) than comparing against arbitrary thresholds.

Given parameters `p0 = -low`, `p1 = low - high`, `p2 = high` (with `p0 + p1 + p2 = 0`):

| Input range | After `+p0` | Clamp 1 | After `+p1` | Clamp 2 | After `+p2` | Output |
|---|---|---|---|---|---|---|
| `x < low` | `x-low < 0` | `val = 0` | `low-high < 0` | no change | `low-high+high = low` | `low` |
| `low <= x <= high` | `x-low >= 0` | no change | `x-high < 0` | no change | `x-high+high = x` | `x` |
| `x > high` | `x-low > 0` | no change | `x-high >= 0` | `val = 0` | `0+high = high` | `high` |

**Note on parameter comments**: The source code comments state `param2 = -(pos_threshold)`. Verified analysis shows that for correct operation, `param2` must equal `high` (the positive upper bound), NOT `-high`. The constraint `p0 + p1 + p2 = 0` requires `p2 = -(p0 + p1) = -(-low + low - high) = high`. The comment may use a sign convention where `pos_threshold = -high`, making `-(pos_threshold) = high`, or the comment may simply be inaccurate. The caller must ensure `p0 + p1 + p2 = 0` for correct results.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Count per Iteration | Description |
|---|---|---|---|
| `SFPLOADI` | `s2vFloat16b(param)` | 3 (before loop) + 0-2 (conditional per iteration) | Load 16-bit FP16_B immediate value into LREG, broadcast to all SFPU lanes |
| `SFPLOAD` | `dst_reg[0]` (read) | 1 | Load 32 elements from current DEST row pair into LREG |
| `SFPADD` | `val += pN` (vFloat + vFloat) | 3 | Floating-point addition via `__builtin_rvtt_sfpadd`. Opcode 0x82, 2-cycle latency. |
| `SFPXFCMPS` | `val < 0.0f`, `val >= 0.0f` | 2 | Scalar float comparison: compares each SFPU lane value against a float immediate, sets CC result per lane |
| `SFPPUSHC` | `v_if(...)` | 2 | Push current CC state onto the per-lane 8-entry CC stack (via macro expansion of `v_if`) |
| `SFPENCC` | `v_if(...)` | 2 | Enable condition code masking (called during `v_if` setup before comparison) |
| `SFPPOPC` | `v_endif` | 2 | Pop CC state from stack, restoring previous CC enable/result bits (via destructor of `__vCCCtrl`) |
| `SFPSTORE` | `dst_reg[0] = val` | 1 | Store LREG value back to current DEST row pair (32 elements), FP16_B/SRCB format |

Total per iteration: approximately 16 instructions (3 SFPADD + 2 SFPXFCMPS + 2 SFPPUSHC + 2 SFPENCC + 2 SFPPOPC + 2 SFPLOADI for conditional zeroing + 1 SFPLOAD + 1 SFPSTORE + setup).

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG (via vFloat p0)** | Holds `-low` parameter, broadcast to all lanes. Allocated by compiler, persists across all iterations (loop-invariant). |
| **LREG (via vFloat p1)** | Holds `-(high-low)` parameter, broadcast to all lanes. Loop-invariant. |
| **LREG (via vFloat p2)** | Holds `high` parameter, broadcast to all lanes. Loop-invariant. |
| **LREG (via vFloat val)** | Working register for the current element value. Loaded from DEST, modified through additions and conditional zeroing, stored back to DEST each iteration. |
| **DEST register** | Source and destination for tile data. Accessed via `dst_reg[0]` with stride-2 addressing. Each iteration reads/writes 32 elements (2 physical rows x 16 elements/row). Pointer advances by 1 sfpi row per iteration via `dst_reg++`. |
| **CC register** | Per-lane condition code bits. Used by the two `v_if` blocks to selectively zero lanes. CC stack depth reaches 1 during each `v_if` block (push on entry, pop on `v_endif`). |

### Address Mode Configuration

The hardtanh operation would use `SfpuType::hardtanh` for address mode configuration via `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()`. Since `hardtanh` does not match any `if constexpr` special case in that function, only the default `ADDR_MOD_7` is configured:

| Address Mode | srca.incr | srcb.incr | dest.incr | Hardware | Notes |
|---|---|---|---|---|---|
| `ADDR_MOD_7` | 0 | 0 | 0 | Wormhole B0 | Default configuration. DEST addressing is managed entirely through `dst_reg++` in the SFPI code (software-controlled), not through hardware auto-increment. |
| `ADDR_MOD_7` | 0 | 0 | 0 | Blackhole | Identical default configuration. |

The `dest.incr = 0` setting means the hardware does not auto-increment the DEST address between SFPU instructions. Instead, DEST row advancement is handled by:
1. **Within a face**: The SFPI `dst_reg++` abstraction (compiled to appropriate address manipulation) advances 1 sfpi row = 2 physical DEST rows per loop iteration.
2. **Between faces**: The `_llk_math_eltwise_unary_sfpu_params_` dispatch function issues `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) to advance the DEST write pointer by 16 physical rows (1 face height).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine `get_op_approx_mode()`, `get_op_init_and_func()`, and `get_compute_kernel_path()` for HARDTANH
   **Key Findings**: HARDTANH has no case in `get_op_init_and_func_parameterized()` or `get_op_init_and_func_default()`. Falls through to `default: return false` for approx mode and `default: return "eltwise_sfpu.cpp"` for compute kernel path.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check `is_parametrized_type()` for HARDTANH
   **Key Findings**: Returns `true` for HARDTANH, confirming it expects parameters.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
   **Reason**: Understand the Python/C++ API surface for hardtanh
   **Key Findings**: `hardtanh()` takes `min_val=-1.0f` and `max_val=1.0f` defaults, creates `UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU kernel implementation for hardtanh
   **Key Findings**: SFPI-based kernel using subtraction-clamp-add trick. Takes 3 pre-computed FP16_B parameters. Identical on Wormhole B0 and Blackhole.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the generic parameters dispatch layer for SFPU unary operations
   **Key Findings**: Handles VectorMode dispatch (RC/R/C), DEST address progression via TTI_SETRWC, and stall synchronization. Would be used by the hardtanh LLK dispatch if it existed.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand SFPU init and address mode configuration
   **Key Findings**: Default `ADDR_MOD_7` with all-zero increments. No special case for hardtanh in `eltwise_unary_sfpu_configure_addrmod`.

7. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Map SFPI C++ abstractions to underlying SFPU instructions
   **Key Findings**: `vFloat + vFloat` maps to `SFPADD` (via `__builtin_rvtt_sfpadd`), `val < 0.0f` maps to `SFPXFCMPS`, `dst_reg[0]` read/write maps to `SFPLOAD`/`SFPSTORE`, `v_if`/`v_endif` maps to `SFPPUSHC`/`SFPENCC`/`SFPPOPC`.

8. **File**: `runtime/sfpi/include/sfpi_fp16.h`
   **Reason**: Understand `s2vFloat16b` scalar-to-vector conversion
   **Key Findings**: `s2vFloat16b(uint32_t)` stores the raw uint32 value as a bfloat16-format scalar. When constructed from `uint32_t`, no conversion is performed -- the value is used as-is as a pre-encoded FP16_B bit pattern.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
   **Reason**: Reference for how a complete LLK dispatch layer looks (frac as working example)
   **Key Findings**: Pattern: include `llk_math_eltwise_unary_sfpu_init.h` + `llk_math_eltwise_unary_sfpu_params.h`, define init and dispatch functions calling `_llk_math_eltwise_unary_sfpu_params_` with the core SFPU function.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and addressing
    **Key Findings**: SFPADD (opcode 0x82) is a real float add instruction (2-cycle latency). SFPXFCMPS is the scalar float comparison. Stride-2 addressing model confirmed.
