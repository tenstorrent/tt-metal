## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `softshrink_tile_init(); softshrink_tile(0, <param0_hex>u);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SOFTSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` (the switch has only a default case) |
| Template parameter (SFPU_OP_CHAIN) | none (no template parameter in init/func calls) | `get_op_init_and_func_parameterized()` -- emits `softshrink_tile_init()` with no template argument and `softshrink_tile(idst, param0)` |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout the entire SFPU call chain | The API header `softshrink.h` passes `APPROX` (the compute kernel's `math_approx_mode`, i.e., `false`) to `llk_math_eltwise_unary_sfpu_softshrink<APPROX>`, which in turn passes it to `calculate_softshrink<APPROXIMATE>`. However, the `calculate_softshrink` function does not use `APPROXIMATION_MODE` in any conditional branch -- it has no `if constexpr` on `APPROXIMATION_MODE`. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `softshrink_tile_init(); softshrink_tile(0, <param0_hex>u);` inside the per-tile loop.
2. **API Header** (`softshrink.h`): `softshrink_tile(idst, param0)` calls `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)` on the MATH thread. `softshrink_tile_init()` calls `llk_math_eltwise_unary_sfpu_softshrink_init<APPROX>()`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_softshrink.h`): The init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROXIMATE>()`, which configures address modes and resets counters. The tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS=8>, dst_index, VectorMode::RC, param0)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets DEST write address, stalls for SFPU readiness, then loops over 4 faces in `VectorMode::RC`, calling `calculate_softshrink(param0)` once per face, with `SETRWC` between faces to advance the DEST pointer.
5. **Core SFPU** (`ckernel_sfpu_softshrink.h`): `calculate_softshrink<APPROXIMATE, ITERATIONS=8>(param0)` performs the softshrink element-wise operation using SFPI abstractions over 8 iterations per face.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed.
- **Operation invocation**: The core SFPU function `calculate_softshrink(param0)` is called once per face, 4 times total. Each invocation processes 8 iterations (ITERATIONS=8), covering all 256 elements of one face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` to advance DEST by 8+8=16 physical rows between faces. On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which issues `math::inc_dst_addr<8>()` twice. Both are equivalent: they advance DEST by 16 physical rows (= 1 face of 256 elements). The init function configures `ADDR_MOD_7` with `dest.incr = 0` (SFPU manages its own addressing via `dst_reg++`).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h
// (Wormhole and Blackhole implementations are identical)

namespace ckernel {
namespace sfpu {

// softshrink(x, lambda) =
//   x - lambda   if x > lambda
//   x + lambda   if x < -lambda
//   0            otherwise
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(std::uint32_t param0) { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    // Bitcast uint32_t param to float via union -- lambda threshold value
    sfpi::vFloat lambda_val = Converter::as_float(param0); // SFPLOADI (load immediate float)
    sfpi::vFloat neg_lambda = -lambda_val; // SFPMOV with COMPSIGN (complement sign bit)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD from current DEST row
        sfpi::vFloat result = 0.0f; // SFPLOADI (load float immediate 0.0)

        v_if(val > lambda_val) { // SFPXCMP (float vector compare GT) + SFPPUSHC + SFPENCC
            result = val - lambda_val; // SFPADD (val + (-lambda_val)) via flt_add(-lambda_val)
        } // SFPPOPC (restore CC state)
        v_endif;

        v_if(val < neg_lambda) { // SFPXCMP (float vector compare LT) + SFPPUSHC + SFPENCC
            result = val + lambda_val; // SFPADD (val + lambda_val)
        } // SFPPOPC (restore CC state)
        v_endif;

        sfpi::dst_reg[0] = result; // SFPSTORE to current DEST row
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**Semantic explanation**: The softshrink function applies a thresholding operation:
- If the input `x` exceeds `lambda`, the output is `x - lambda` (shifted toward zero).
- If the input `x` is below `-lambda`, the output is `x + lambda` (shifted toward zero).
- Otherwise (when `|x| <= lambda`), the output is `0`.

The `result` variable is initialized to `0.0f` before the two `v_if` blocks. Each `v_if` block is independent and non-overlapping (a value cannot be both `> lambda` and `< -lambda`), so the two conditional assignments are mutually exclusive. The final `result` is either `x - lambda`, `x + lambda`, or `0.0f` (default).

### SFPU Instructions Used

| Instruction | Emitted via | Description |
|-------------|-------------|-------------|
| `SFPLOADI` | `vFloat lambda_val = Converter::as_float(param0)` and `vFloat result = 0.0f` | Load 16-bit immediate to LREG. Used to load the lambda constant and the zero default. For 32-bit floats, two SFPLOADI instructions are needed (high 16 bits + low 16 bits). |
| `SFPMOV` | `-lambda_val` (unary negate) | Register copy with sign-complement mode (`SFPMOV_MOD1_COMPSIGN`). Negates lambda to produce `-lambda`. |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load from DEST register row into LREG with format conversion. Reads the current input element. |
| `SFPSTORE` | `sfpi::dst_reg[0] = result` (write) | Store LREG to DEST register row with format conversion. Writes the computed result back. |
| `SFPXCMP` | `val > lambda_val` and `val < neg_lambda` | Float vector-to-vector comparison. Sets per-lane CC bits based on the comparison result (GT or LT). Emitted via `__builtin_rvtt_sfpxfcmpv`. |
| `SFPADD` | `val - lambda_val` and `val + lambda_val` | Float addition (the subtraction `val - lambda_val` is implemented as `val + (-lambda_val)` where the negation is performed by `SFPMOV` on lambda prior to the add). Emitted via `__builtin_rvtt_sfpadd`. |
| `SFPPUSHC` | `v_if(...)` | Push current CC state onto the CC stack to enable nested/sequential conditional regions. |
| `SFPPOPC` | `v_endif` | Pop CC state from stack, restoring previous CC enable/result. |
| `SFPENCC` | `v_if(...)` / `v_endif` | Enable CC masking (set CC.En=1) so subsequent instructions are predicated, or disable it after the conditional block. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG (lambda_val)** | Holds the `lambda` threshold value, loaded from `param0` via `Converter::as_float`. Persists across all 8 iterations of the inner loop. |
| **LREG (neg_lambda)** | Holds `-lambda`, computed once via `SFPMOV` with sign complement. Also persists across all iterations. |
| **LREG (val)** | Temporary: loaded from `dst_reg[0]` each iteration via `SFPLOAD`. Holds the current input value. |
| **LREG (result)** | Temporary: initialized to `0.0f` via `SFPLOADI` each iteration. Conditionally overwritten to `val - lambda` or `val + lambda` depending on comparisons. Written back to DEST via `SFPSTORE`. |
| **DEST** | Source and destination for tile data. Read via `SFPLOAD` from `dst_reg[0]`, written via `SFPSTORE` to `dst_reg[0]`, then advanced by `dst_reg++` (1 sfpi row = 2 physical DEST rows = 32 elements). |
| **CC Stack** | Used by `v_if`/`v_endif` blocks. Each `v_if` pushes CC state (`SFPPUSHC`), and each `v_endif` pops it (`SFPPOPC`). The two `v_if` blocks are sequential (not nested), so the CC stack depth never exceeds 1. |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::softshrink>()` called from `_llk_math_eltwise_unary_sfpu_init_()`.

**Wormhole B0 and Blackhole**: Both use the same default configuration since `SfpuType::softshrink` does not match any specialized `if constexpr` branches:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode. The SFPU manages its own DEST addressing internally via `dst_reg++` (which uses `SFP_DESTREG_STRIDE=2`), so the hardware address mode auto-increment is set to zero. |

The address mode is identical on both Wormhole B0 and Blackhole. No `ADDR_MOD_6` is configured for this operation (it is only configured for `topk_local_sort`, `typecast`, and `unary_max`/`unary_min` variants).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and SFPU_OP_CHAIN_0 expansion for SOFTSHRINK
   **Key Findings**: Compute kernel is `eltwise_sfpu.cpp` (default). `get_op_approx_mode()` returns `false` (default). Init/func: `softshrink_tile_init()` / `softshrink_tile(idst, param0_hex)`. Macro: `SFPU_OP_SOFTSHRINK_INCLUDE`. SOFTSHRINK is a parameterized type (requires lambda parameter).

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Confirm `is_parametrized_type()` for SOFTSHRINK and default API signatures
   **Key Findings**: `SOFTSHRINK` returns `true` from `is_parametrized_type()`. Default lambda is 0.5f if no parameter provided.

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h`
   **Reason**: API header layer - trace from tile-level API to LLK dispatch
   **Key Findings**: `softshrink_tile(idst, param0)` calls `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)`. Init calls `llk_math_eltwise_unary_sfpu_softshrink_init<APPROX>()`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`
   **Reason**: LLK dispatch layer - trace from LLK to core SFPU and params dispatch
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROXIMATE>()`. Tile function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with `calculate_softshrink<APPROXIMATE, ITERATIONS=8>` and `VectorMode::RC`.

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`
   **Reason**: Core SFPU kernel implementation (Wormhole B0)
   **Key Findings**: Pure SFPI-based kernel. Uses `v_if` with float comparisons, `SFPADD` for arithmetic, `SFPMOV` for sign negation, `SFPLOADI` for constants. APPROXIMATION_MODE is unused. Identical to Blackhole version.

6. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`
   **Reason**: Core SFPU kernel implementation (Blackhole) - verify identical to Wormhole
   **Key Findings**: Byte-for-byte identical to Wormhole B0 version.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function - understand face iteration and DEST progression
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_()` calls the SFPU function once per face, 4 times for VectorMode::RC. Uses `TTI_SETRWC` with stride 8+8=16 between faces. Stalls for SFPU readiness before and after.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::softshrink>()` sets `ADDR_MOD_7` with all increments = 0. No specialized configuration for softshrink.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Blackhole init/address mode - verify same as Wormhole
   **Key Findings**: Same `ADDR_MOD_7` configuration. Minor differences in `_llk_math_eltwise_unary_sfpu_start_` (no `set_addr_mod_base` call) and `_llk_math_eltwise_unary_sfpu_done_` (no `clear_addr_mod_base` call), but these are general infrastructure differences, not softshrink-specific.

10. **File**: `runtime/sfpi/include/sfpi.h`
    **Reason**: SFPI abstraction mappings - understand what hardware instructions the C++ abstractions emit
    **Key Findings**: `vFloat + vFloat` -> `__builtin_rvtt_sfpadd` (SFPADD). `vFloat - vFloat` -> `flt_add(-b)` (SFPMOV for negate + SFPADD). `-vFloat` -> `__builtin_rvtt_sfpmov(SFPMOV_MOD1_COMPSIGN)` (SFPMOV). `vFloat > vFloat` -> `__builtin_rvtt_sfpxfcmpv(CC_GT)` (SFPXCMP). `vFloat(float)` -> `__builtin_rvtt_sfpxloadi` (SFPLOADI).

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware model reference
    **Key Findings**: SFP_DESTREG_STRIDE=2, ITERATIONS=8 per face, dst_tile_size_sfpi=32. CC stack operations (SFPPUSHC/SFPPOPC) and CC masking (SFPENCC/SFPCOMPC) mechanisms.

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
    **Reason**: Understand `Converter::as_float(uint32_t)` utility
    **Key Findings**: Simple union-based bitcast from uint32_t to float. No computation involved.
