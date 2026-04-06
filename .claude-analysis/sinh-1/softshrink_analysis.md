## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `softshrink_tile(0, {lambda_hex}u)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SOFTSHRINK)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none (no template arg in chain) | `get_op_init_and_func_parameterized()` -- `softshrink_tile_init()` has no template parameter; `softshrink_tile(idst, param0)` has no template parameter. Both use the `APPROX` JIT constant directly in the API header. |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout; however, the `calculate_softshrink` kernel does NOT branch on `APPROXIMATION_MODE` -- it is declared as a template parameter but never referenced in the kernel body, so approximation mode has no behavioral effect. | `ckernel_sfpu_softshrink.h` -- no `if constexpr(APPROXIMATION_MODE)` or equivalent branch exists |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `softshrink_tile_init(); softshrink_tile(0, {lambda_hex}u);`.
2. **API header** (`softshrink.h`): `softshrink_tile(idst, param0)` calls `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)` inside the `MATH((...))` macro, restricting execution to the math thread.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_softshrink.h`): `llk_math_eltwise_unary_sfpu_softshrink<APPROXIMATE>(dst_index, param0)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_softshrink<APPROXIMATE, 8>, dst_index, VectorMode::RC, param0)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up the DEST write address, stalls for SFPU readiness, then iterates over all 4 faces in `VectorMode::RC` mode, calling `calculate_softshrink(param0)` once per face and advancing the DEST face address after each call.
5. **Core SFPU** (`ckernel_sfpu_softshrink.h`): `calculate_softshrink<APPROXIMATION_MODE, 8>(param0)` executes the softshrink algorithm using SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (Face 0 through Face 3), covering all 1024 elements of the 32x32 tile.
- **Operation invocation**: The `calculate_softshrink` function is called once per face (4 times total). Each invocation processes one face via 8 iterations of its internal loop (`ITERATIONS=8`), processing 32 elements per iteration (2 physical DEST rows x 16 elements/row).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, the params dispatch uses explicit `TTI_SETRWC` to advance by 8 sfpi rows (= 16 physical DEST rows) twice per face transition. On Blackhole, the params dispatch calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice, achieving the same effect. The address mode is `ADDR_MOD_7` with all-zero increments (srca=0, srcb=0, dest=0), since DEST addressing within the SFPU kernel is managed by `dst_reg++` rather than hardware auto-increment.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so this follows **Style A**.

The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h

namespace ckernel {
namespace sfpu {

// softshrink(x, lambda) =
//   x - lambda   if x > lambda
//   x + lambda   if x < -lambda
//   0            otherwise
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(std::uint32_t param0) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // param0 = lambda as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat lambda_val = Converter::as_float(param0); // SFPLOADI: load lambda into an LREG
    sfpi::vFloat neg_lambda = -lambda_val;                  // SFPMOV with COMPSIGN: negate lambda

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];    // SFPLOAD: load current element from DEST
        sfpi::vFloat result = 0.0f;              // SFPLOADI: load 0.0 into result LREG

        v_if(val > lambda_val) {                 // SFPXFCMPV (val, lambda): computes (val - lambda), tests sign
                                                 // Internally: SFPMAD/SFPADD to subtract, SFPSETCC to test
                                                 // SFPENCC to enable CC masking
            result = val - lambda_val;           // SFPMOV (negate lambda) + SFPADD, or SFPMAD(val, 1.0, -lambda)
                                                 // CC-guarded: only lanes where val > lambda execute this
        }
        v_endif;                                 // SFPENCC: disable CC (all lanes active again)

        v_if(val < neg_lambda) {                 // SFPXFCMPV (val, neg_lambda): computes (val - neg_lambda), tests sign
                                                 // SFPENCC to enable CC masking
            result = val + lambda_val;           // SFPADD(val, lambda), or SFPMAD(val, 1.0, lambda)
                                                 // CC-guarded: only lanes where val < -lambda execute this
        }
        v_endif;                                 // SFPENCC: disable CC (all lanes active again)

        sfpi::dst_reg[0] = result;               // SFPSTORE: write result back to DEST
        sfpi::dst_reg++;                          // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler for this kernel. Since the kernel uses SFPI abstractions, the exact instruction sequence depends on the SFPI compiler's optimization choices, but the logical mapping is:

| SFPI Abstraction | Emitted SFPU Instruction(s) | Description |
|---|---|---|
| `Converter::as_float(param0)` | `SFPLOADI` | Load 32-bit float immediate from a uint32_t bit pattern into an LREG |
| `-lambda_val` | `SFPMOV` (with `COMPSIGN` modifier) | Negate a float value by complementing its sign bit |
| `sfpi::dst_reg[0]` (read) | `SFPLOAD` | Load 32 elements (2 physical DEST rows) from the current DEST position into an LREG |
| `vFloat result = 0.0f` | `SFPLOADI` | Load float immediate 0.0 into an LREG |
| `val > lambda_val` | `SFPMAD` or `SFPADD` + `SFPSETCC` | Float comparison: compute `val - lambda`, then set CC based on sign of result. This is part of the `__builtin_rvtt_sfpxfcmpv` intrinsic. |
| `v_if(...)` | `SFPENCC` | Enable condition code masking so subsequent instructions only execute on lanes where the condition is true |
| `val - lambda_val` | `SFPADD` (or `SFPMAD`) | Float subtraction: `val + (-lambda)`. Since there is no dedicated float subtract, this is `SFPADD` with the negated operand, or `SFPMAD(val, 1.0, -lambda)`. CC-guarded. |
| `val + lambda_val` | `SFPADD` (or `SFPMAD`) | Float addition: `val + lambda`. Emits `SFPADD` or `SFPMAD(val, 1.0, lambda)`. CC-guarded. |
| `v_endif` | `SFPENCC` | Disable condition code masking, restoring all lanes to active |
| `sfpi::dst_reg[0] = result` (write) | `SFPSTORE` | Store 32 elements from an LREG back to the current DEST position |
| `sfpi::dst_reg++` | (address increment) | Advance the SFPU DEST read/write pointer by 1 sfpi row (= 2 physical DEST rows). This is a software counter increment, not a hardware instruction. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG (lambda_val)** | Holds the lambda threshold value, loaded once before the loop via `Converter::as_float(param0)`. Persists across all 8 iterations per face. |
| **LREG (neg_lambda)** | Holds `-lambda`, computed once before the loop via SFPMOV with sign complement. Persists across all 8 iterations. |
| **LREG (val)** | Temporary: holds the current input element loaded from DEST via `SFPLOAD`. Overwritten each iteration. |
| **LREG (result)** | Temporary: initialized to 0.0 each iteration, conditionally updated to `val - lambda` or `val + lambda`. Written back to DEST via SFPSTORE. |
| **DEST register** | Source and destination of tile data. Each iteration reads 32 elements (2 physical rows x 16 columns) from the current DEST position, processes them, and writes back. The pointer advances by 1 sfpi row per iteration. |
| **CC (Condition Code)** | Used twice per iteration: first for the `val > lambda` test, then for the `val < -lambda` test. Each `v_if`/`v_endif` block enables CC, sets the condition, executes guarded instructions, and disables CC. CC state does not persist between the two `v_if` blocks. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::softshrink>()` during init. Since `SfpuType::softshrink` does not match any special-case `if constexpr` branch, only the default `ADDR_MOD_7` is set:

**Wormhole B0 and Blackhole** (identical for this operation):

```
ADDR_MOD_7:
  .srca = { .incr = 0 }
  .srcb = { .incr = 0 }
  .dest = { .incr = 0 }
```

All increments are zero because the SFPU kernel manages DEST addressing through the SFPI `dst_reg++` abstraction (which emits software address counter updates) rather than relying on hardware auto-increment from the address mode. The face-to-face transitions are handled by explicit `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` (Blackhole) calls in the parameters dispatch layer.

No additional `ADDR_MOD_6` is configured for this operation (that is reserved for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, and related ops).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SOFTSHRINK
   **Key Findings**: Compute kernel is `eltwise_sfpu.cpp` (default). Init: `softshrink_tile_init()`. Func: `softshrink_tile({idst}, {lambda_hex}u)`. Include guard: `SFPU_OP_SOFTSHRINK_INCLUDE`. Lambda parameter defaults to 0.5f if not specified. `get_op_approx_mode()` returns `false` for all ops.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Verify `is_parametrized_type` for SOFTSHRINK
   **Key Findings**: SOFTSHRINK is a parameterized type (returns `true`), meaning it always goes through `get_op_init_and_func_parameterized()`.

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h`
   **Reason**: API header -- trace the tile-level API call
   **Key Findings**: `softshrink_tile(idst, param0)` calls `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)`. `softshrink_tile_init()` calls `llk_math_eltwise_unary_sfpu_softshrink_init<APPROX>()`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`
   **Reason**: LLK dispatch layer -- trace from API to core SFPU
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROXIMATE>()`. Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_softshrink<APPROXIMATE, 8>, dst_index, VectorMode::RC, param0)`. Identical on Blackhole.

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`
   **Reason**: Core SFPU implementation -- the main analysis target
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Compares input with lambda and neg_lambda, conditionally subtracts or adds lambda, defaults to 0.0. Identical on Blackhole. APPROXIMATION_MODE template parameter is declared but never used in the kernel body.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- understand face iteration and DEST address management
   **Key Findings**: VectorMode::RC processes all 4 faces. Each face: call sfpu_func(args...), then TTI_SETRWC x2 to advance by 16 physical DEST rows. Stalls SFPU before and after.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_init_` calls `_init_sfpu_config_reg()`, `eltwise_unary_sfpu_configure_addrmod<sfpu_op>()`, and `math::reset_counters()`. For softshrink, only ADDR_MOD_7 (all-zero increments) is configured.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand `Converter::as_float()` utility
   **Key Findings**: Simple union-based type pun from uint32_t to float. Used to reinterpret the lambda parameter from its bit-cast uint32_t form.

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand SFPI abstraction mappings to SFPU instructions
   **Key Findings**: `vFloat > vFloat` maps to `__builtin_rvtt_sfpxfcmpv` (float compare vector). `vFloat + vFloat` maps to `__builtin_rvtt_sfpadd` (SFPADD). `vFloat(float)` maps to `__builtin_rvtt_sfpxloadi` (SFPLOADI). `-vFloat` maps to `__builtin_rvtt_sfpmov` with COMPSIGN.

10. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Trace how the `APPROX` constant is generated at JIT compile time
    **Key Findings**: `APPROX` is emitted as `constexpr bool APPROX = {math_approx_mode};` in the JIT-generated `chlkc_descriptors.h`. For SOFTSHRINK, this resolves to `false`.

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware model reference
    **Key Findings**: Tile geometry (32x32, 4 faces of 16x16), stride-2 addressing model, ITERATIONS=8 per face, SFPU instruction semantics.
