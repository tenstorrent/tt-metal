## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `swish_tile_init(); swish_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default) | `get_op_init_and_func_default()` returns `"swish_tile_init();"` and `"swish_tile(0);"` -- no template arguments specified, so `APPROX` (resolved to `false` via `genfiles.cpp`) is used as the default |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_swish<false, 8>()` | The template parameter `APPROXIMATION_MODE` is `false`; however, the kernel does not contain any `if constexpr (APPROXIMATION_MODE)` branches, so the same code path is taken regardless |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. The compute kernel's `SFPU_OP_CHAIN_0` macro expands to `swish_tile_init(); swish_tile(0);`.
2. `swish_tile_init()` (in `swish.h`) calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()` to configure SFPU address modes and reset counters.
3. `swish_tile(0)` (in `swish.h`) calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(0)`, which calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, 8>, 0, VectorMode::RC)`.
4. The params dispatch function sets the DEST write address for tile 0, stalls until SFPU is ready, then invokes `calculate_swish<false, 8>()` once per face (4 times total for `VectorMode::RC`), advancing the DEST address between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (all 4 faces of the tile are processed). Each face is 16x16 = 256 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_swish<false, 8>()` once per face. Each invocation processes 8 iterations (ITERATIONS=8), covering all 8 sfpi rows of that face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` / `inc_dst_addr<8>` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` with `p_setrwc::CR_D, 8` twice per face (advancing by 16 physical DEST rows = 8 sfpi rows). On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source
The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::abs`, `v_if`/`v_endif`, `sfpi::vConst1`). This is **Style A**.

The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h

namespace ckernel {
namespace sfpu {

// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Implementation notes, see the original file for more details

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;
    constexpr float c2 = -0.01479f;
    constexpr float c3 = -0.00747f;

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;
    constexpr float lin_offset = 0.855f;

    // Breakpoints
    constexpr float bp1 = 2.5f;
    constexpr float bp2 = 5.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row

        // Compute sigmoid(|x|) using polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS: absolute value (float mode)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3)); // Horner's form: chain of SFPMUL/SFPADD/SFPMAD

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { sig_pos = ax * lin_slope + lin_offset; } // CC: SFPSETCC on (ax - bp1), SFPMUL + SFPADD guarded
        v_endif; // CC reset via SFPENCC/SFPPOPC

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { sig_pos = sfpi::vConst1; } // CC: SFPSETCC on (ax - bp2), load CREG_IDX_1 guarded
        v_endif;

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { sig_pos = sfpi::vConst1 - sig_pos; } // CC: SFPSETCC on x sign bit, SFPADD(1.0 - sig_pos) guarded
        v_endif;

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMUL + SFPSTORE: multiply then store back to DEST
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Source (SFPI Abstraction) | Description |
|-------------|--------------------------|-------------|
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Loads 32 elements from the current DEST row pair into an LREG with format conversion |
| `SFPSTORE` | `sfpi::dst_reg[0] = ...` (write) | Stores an LREG value back to the current DEST row pair with format conversion |
| `SFPABS` | `sfpi::abs(x)` | Computes element-wise absolute value in float mode (`SFPABS_MOD1_FLOAT`) |
| `SFPLOADI` | Float literal constants (`0.5f`, `0.2533f`, etc.) | Loads 16-bit immediate values into LREGs to materialize float constants. Two `SFPLOADI` instructions are needed for a full 32-bit float (high 16 bits and low 16 bits) |
| `SFPMUL` | `vFloat * vFloat` (e.g., `ax * c1`, `x * sig_pos`) | Floating-point multiply (2-cycle latency, fully pipelined). Emitted by `__builtin_rvtt_sfpmul` |
| `SFPADD` | `vFloat + vFloat` (e.g., `0.5f + ...`, `vConst1 - sig_pos`) | Floating-point add (2-cycle latency). Subtraction is add with sign inversion. Emitted by `__builtin_rvtt_sfpadd` |
| `SFPMAD` | Fused multiply-add chains | The compiler may fuse multiply+add sequences (e.g., `ax * lin_slope + lin_offset`) into a single `SFPMAD` instruction: `VD = VA * VB + VC` |
| `SFPSETCC` | `v_if(ax > bp1)`, `v_if(x < 0.0f)` | Sets per-lane condition code based on comparison result. For `>` comparisons, the compiler computes `a - b` via `SFPMAD`/`SFPADD` then tests the sign. For `x < 0.0f`, it directly tests the sign bit of `x` |
| `SFPENCC` | `v_if` / `v_endif` | Enables/disables condition code masking. `v_if` enables CC; `v_endif` disables CC to restore all-lanes-active |
| `SFPPUSHC` | Nested `v_if` (if compiler emits nesting) | Pushes current CC state onto the CC stack for nested conditionals |
| `SFPPOPC` | `v_endif` | Pops CC state from the CC stack to restore the previous conditional context |
| `SFPCOMPC` | (implicit in `v_if`/`v_endif` patterns) | Complements CC.Res for else-branch handling (may be emitted by the compiler for certain conditional patterns) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (via `dst_reg[0]`) | Input: the tile data loaded from CB c_0. Output: swish(x) result written back. Each iteration reads/writes 32 elements (2 physical DEST rows) |
| **LREGs (LREG0-LREG7)** | Temporary storage for intermediate values. The SFPI compiler allocates LREGs automatically for: `x` (input), `ax` (absolute value), `sig_pos` (sigmoid approximation), polynomial intermediates, and the final `x * sig_pos` product. At least 4-5 LREGs are active simultaneously during the polynomial evaluation |
| **Constant registers** | `CREG_IDX_1` (Fixed Const 2, value 1.0) is used via `sfpi::vConst1` for the saturation branch (`sig_pos = 1.0`) and the sign correction (`1.0 - sig_pos`). Float literal constants (0.2533, -0.01479, -0.00747, 0.0276, 0.855, 2.5, 5.0, 0.5, 0.0) are materialized via `SFPLOADI` pairs into LREGs |
| **CC bits** (per-lane) | Used by four `v_if` blocks: (1) `ax > 2.5` for linear segment override, (2) `ax > 5.0` for saturation, (3) `x < 0.0` for sign correction. Each `v_if` enables CC, sets condition, executes guarded code, then disables CC |
| **CC stack** | Used by the compiler to manage `v_if`/`v_endif` scoping. Since the four `v_if` blocks are sequential (not nested), the stack depth is at most 1 |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()` in `llk_math_eltwise_unary_sfpu.h`.

Since `SfpuType::swish` does not match any of the special-case `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

| Address Mode | Field | Value | Purpose |
|-------------|-------|-------|---------|
| `ADDR_MOD_7` | `srca.incr` | 0 | No auto-increment for source A |
| `ADDR_MOD_7` | `srcb.incr` | 0 | No auto-increment for source B |
| `ADDR_MOD_7` | `dest.incr` | 0 | No auto-increment for DEST |

This configuration is **identical across Wormhole and Blackhole**. The `ADDR_MOD_7` with all-zero increments means SFPU DEST addressing does not auto-increment between instructions -- instead, the `dst_reg++` in the SFPI kernel explicitly advances the DEST pointer by 1 sfpi row (2 physical rows) per iteration, and the params dispatch function advances between faces via `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` (Blackhole).

Note: The init function also configures `ADDR_MOD_7` with all-zero increments for general use. No additional address modes (`ADDR_MOD_6`, etc.) are configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SWISH
   **Key Findings**: SWISH uses `eltwise_sfpu.cpp`, init/func are `swish_tile_init()/swish_tile(idst)`, macro define is `SFPU_OP_SWISH_INCLUDE`, `get_op_approx_mode` returns `false` (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header exposing `swish_tile()` and `swish_tile_init()`
   **Key Findings**: `swish_tile(idst)` calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` calls `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU implementation
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)` with default `ITERATIONS=8` and `VectorMode::RC`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Implements swish as `x * sigmoid(x)` using a 3-segment piecewise sigmoid approximation: degree-3 polynomial for |x| <= 2.5, linear segment for 2.5 < |x| <= 5.0, saturation to 1.0 for |x| > 5.0. Uses SFPI abstractions throughout. Wormhole and Blackhole implementations are byte-identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that invokes the SFPU kernel per-face
   **Key Findings**: For VectorMode::RC, loops over 4 faces calling the SFPU function once per face, advancing DEST address between faces via `TTI_SETRWC(CR_D, 8)` twice

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: `SfpuType::swish` only gets `ADDR_MOD_7` with all-zero increments (no special address mode cases). Init calls `_init_sfpu_config_reg()`, configures addrmod, and resets counters.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Determine how `APPROX` constexpr is generated during JIT compilation
   **Key Findings**: `emit_math_scalar_descriptors` generates `constexpr bool APPROX = {math_approx_mode};` into `chlkc_descriptors.h`. For SWISH, this resolves to `false`.

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU instruction semantics, register layout, and addressing model
   **Key Findings**: Confirmed SFPMUL/SFPADD are 2-cycle latency MAD-type instructions, SFPABS is 1-cycle, SFPLOAD/SFPSTORE latencies, CC mechanism details

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: SFPI abstraction mappings to hardware instructions
   **Key Findings**: `vFloat + vFloat` maps to `__builtin_rvtt_sfpadd` (SFPADD), `vFloat * vFloat` maps to `__builtin_rvtt_sfpmul` (SFPMUL), `operator>` creates `__vCond` via `__builtin_rvtt_sfpxfcmps` (SFPSETCC-based comparison), `dst_reg[0]` read maps to `__builtin_rvtt_sfpload`, `vConst1` is CREG index 10 (fixed constant 1.0)

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: SFPI library function implementations
    **Key Findings**: `sfpi::abs(vFloat)` maps to `__builtin_rvtt_sfpabs(v, SFPABS_MOD1_FLOAT)`, confirming the SFPABS instruction with float mode
