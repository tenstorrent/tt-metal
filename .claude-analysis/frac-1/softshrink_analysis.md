## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `eltwise_sfpu.cpp` (default for all unary ops in this codebase)
- **SFPU_OP_CHAIN_0 expansion**: `softshrink_tile_init(); softshrink_tile(idst, param0);` where `param0` is the lambda threshold as IEEE 754 bits (bitcast `uint32_t`)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SOFTSHRINK)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (hardcoded `APPROX`) | `get_op_init_and_func_parameterized()` -- generates `softshrink_tile_init()` and `softshrink_tile({idst}, {param0})` with no explicit template argument; the API header uses `APPROX` which is the `math_approx_mode` compile-time define |
| Effective SFPU path | `APPROXIMATION_MODE=false`; however the kernel has no `if constexpr (APPROXIMATION_MODE)` branches, so the path is the same regardless | `calculate_softshrink` in `ckernel_sfpu_softshrink.h` has no approximation-dependent branches |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `softshrink_tile(0, param0)`.
2. **API Header** (`softshrink.h`): `softshrink_tile(idst, param0)` calls `MATH((llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_softshrink.h`): `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(dst_index, param0)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, stalls for SFPU availability, then iterates over faces (4 faces for `VectorMode::RC`), calling the SFPU function once per face with `param0` forwarded as the argument.
5. **Core SFPU Implementation** (`ckernel_sfpu_softshrink.h`): `calculate_softshrink<false, 8>(param0)` performs the per-face computation over 8 iterations (one per sfpi row within the face).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch loops over 4 faces. For each face, it calls `calculate_softshrink(param0)` once, which internally loops 8 iterations (ITERATIONS=8). After each face, `SETRWC` (Wormhole) or `inc_dst_addr<8>` x2 (Blackhole) advances the DEST write pointer to the next face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` with `CR_D, 8` twice per face; on Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source
The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h
// NOTE: WH and BH implementations are identical.

namespace ckernel {
namespace sfpu {

// softshrink(x, lambda) =
//   x - lambda   if x > lambda
//   x + lambda   if x < -lambda
//   0            otherwise
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softshrink(std::uint32_t param0) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Reinterpret param0 (IEEE 754 bits) as float -> load into vFloat LREG
    sfpi::vFloat lambda_val = Converter::as_float(param0); // SFPLOADI (16-bit hi + 16-bit lo to build full float)
    sfpi::vFloat neg_lambda = -lambda_val; // SFPMOV with COMPSIGN (negate sign bit)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD from current DEST row
        sfpi::vFloat result = 0.0f; // SFPLOADI with 0.0 immediate

        // First conditional block: if val > lambda, result = val - lambda
        v_if(val > lambda_val) { // SFPENCC(enable) + SFPPUSHC + SFPXCMP(GT) -> SFPMAD(val-lambda)+SFPSETCC(GTE0)
            result = val - lambda_val; // SFPMAD: val * 1.0 + (-lambda_val), CC-guarded
        }
        v_endif; // SFPPOPC (restore CC state from stack)

        // Second conditional block: if val < -lambda, result = val + lambda
        v_if(val < neg_lambda) { // SFPENCC(enable) + SFPPUSHC + SFPXCMP(LT) -> SFPMAD(val-neg_lambda)+SFPSETCC(LT0)
            result = val + lambda_val; // SFPMAD: val * 1.0 + lambda_val, CC-guarded
        }
        v_endif; // SFPPOPC (restore CC state from stack)

        sfpi::dst_reg[0] = result; // SFPSTORE to current DEST row
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Purpose in this kernel |
|-------------|----------------------|
| **SFPLOAD** | Load current element vector from DEST register into an LREG (`val = dst_reg[0]`) |
| **SFPLOADI** | Load immediate values into LREGs: the lambda constant (from `param0` via `Converter::as_float`), the zero constant (`0.0f`), and intermediate 16-bit halves for building 32-bit floats |
| **SFPMOV** | Negate sign bit to compute `neg_lambda = -lambda_val` (via `SFPMOV_MOD1_COMPSIGN`) |
| **SFPMAD** | Fused multiply-add used for: (1) subtraction `val - lambda_val` (= val * 1.0 + (-lambda)), (2) addition `val + lambda_val` (= val * 1.0 + lambda), (3) comparison difference computation `val - lambda` for SFPXCMP lowering |
| **SFPSETCC** | Set condition code based on comparison result sign (part of SFPXCMP lowering for `>` and `<` comparisons). For `val > lambda`: tests if `val - lambda >= 0` (GTE0 mode). For `val < neg_lambda`: tests if `val - neg_lambda < 0` (LT0 mode) |
| **SFPENCC** | Enable/disable condition code masking. Used at the start of each `v_if` block to activate CC-guarded execution, and at the end to restore all-lanes-enabled state |
| **SFPPUSHC** | Push current CC state onto the CC stack at the start of each `v_if` block, preserving the outer CC context |
| **SFPPOPC** | Pop CC state from the stack at `v_endif`, restoring the previous CC context |
| **SFPSTORE** | Store result vector from LREG back to DEST register (`dst_reg[0] = result`) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST (via `dst_reg`)** | Source and destination for per-iteration element data. Read at start of each iteration (`val = dst_reg[0]`), written at end (`dst_reg[0] = result`). Pointer advances by 1 sfpi row per iteration. |
| **LREG (compiler-assigned)** | The SFPI compiler allocates LREGs from the pool of 8 (LREG0-LREG7) for: `lambda_val`, `neg_lambda`, `val`, `result`, and temporary comparison results. Since these are SFPI abstractions, the exact LREG assignments depend on the compiler's register allocator, but the kernel uses at most 4-5 live `vFloat` values simultaneously, well within the 8-LREG budget. |
| **CC bits (per-lane)** | Two CC bits per lane (CC.En, CC.Res) are manipulated by `v_if`/`v_endif` blocks. The CC stack (8-deep) is used to save/restore CC state across the two independent `v_if` blocks. Each `v_if` pushes one entry; `v_endif` pops it. Maximum stack depth is 1 (no nesting). |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::softshrink>()` during `_llk_math_eltwise_unary_sfpu_init_`. Since `softshrink` does not match any of the special-cased `SfpuType` values (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default address mode is set:

| Hardware | ADDR_MOD | srca.incr | srcb.incr | dest.incr |
|----------|----------|-----------|-----------|-----------|
| **Wormhole B0** | `ADDR_MOD_7` | 0 | 0 | 0 |
| **Blackhole** | `ADDR_MOD_7` | 0 | 0 | 0 |

Both Wormhole and Blackhole configure `ADDR_MOD_7` identically with all zero increments. This means DEST address auto-increment is disabled at the hardware level -- the SFPU kernel manages DEST addressing explicitly via `dst_reg++` (which the SFPI compiler translates to appropriate DEST pointer manipulation). The between-face advancement is handled by the params dispatch layer using `SETRWC` (WH) or `inc_dst_addr` (BH), not by `ADDR_MOD`.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SOFTSHRINK.
   **Key Findings**: `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` (default). `get_op_approx_mode()` returns `false` (default). `get_op_init_and_func_parameterized()` generates `softshrink_tile_init()` and `softshrink_tile({idst}, {param0})` with `lambda_val` defaulting to 0.5f. The macro define is `SFPU_OP_SOFTSHRINK_INCLUDE`.

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softshrink.h`
   **Reason**: Trace the tile-level API call to understand the bridge between compute kernel and LLK layer.
   **Key Findings**: `softshrink_tile(idst, param0)` calls `llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)`. `softshrink_tile_init()` calls `llk_math_eltwise_unary_sfpu_softshrink_init<APPROX>()`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softshrink.h`
   **Reason**: Understand LLK dispatch layer for softshrink.
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::softshrink, APPROXIMATE>()`. Compute calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_softshrink<APPROXIMATE, ITERATIONS>` and forwards `param0`. WH and BH LLK dispatch files are identical.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`
   **Reason**: Read the core SFPU implementation for softshrink.
   **Key Findings**: Uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`). Implements piecewise function: x-lambda if x>lambda, x+lambda if x<-lambda, 0 otherwise. WH and BH implementations are identical. No approximation-mode-dependent branching.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch layer (face iteration, DEST addressing, stall management).
   **Key Findings**: For `VectorMode::RC`, loops 4 faces, calls SFPU function once per face, advances DEST with `SETRWC` (WH) or `inc_dst_addr` (BH). Stalls SFPU before computation, stalls CFG after.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration during init.
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::softshrink>()` sets only `ADDR_MOD_7` with all zero increments (softshrink does not match any special-cased SfpuType). Init also calls `_init_sfpu_config_reg()` and `reset_counters`.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Understand the `Converter::as_float()` helper used to reinterpret `uint32_t` param as float.
   **Key Findings**: Simple union-based bitcast from `uint32_t` to `float`.

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Understand how SFPI C++ abstractions (`vFloat` comparisons, `v_if`/`v_endif`, arithmetic operators) map to SFPU instructions.
   **Key Findings**: `vFloat > vFloat` creates `__vCond(__vCondGT, ...)` using `SFPXCMP_MOD1_CC_GT`. `v_if` expands to `cc_push().cc_if().cc_cond(x)` which generates SFPENCC + SFPPUSHC + comparison. `v_endif` generates SFPPOPC. Subtraction `-=` on WH uses SFPMAD with `vConstNeg1`. Addition `+` uses SFPMAD (a * 1.0 + b).

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Reference for SFPU architecture, instruction semantics, register layout, and CC mechanism.
   **Key Findings**: Standard tile/face geometry (4 faces x 8 iterations = 32 sfpi rows), stride-2 addressing, 8 LREGs per lane, CC stack 8-deep, instruction latencies.

10. **File**: `.claude/references/diagram-templates.md`
    **Reason**: Reference for CC State Machine diagram format (not needed since this kernel uses Style A with SFPI abstractions where CC flow is explicit via `v_if`/`v_endif`).
    **Key Findings**: Template available but not used -- softshrink uses SFPI style (Style A), not raw TTI instructions.
