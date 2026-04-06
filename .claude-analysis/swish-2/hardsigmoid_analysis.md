## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSIGMOID`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardsigmoid_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSIGMOID)` in `unary_op_utils.cpp` — falls through to `default: return false` (no explicit case for HARDSIGMOID) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` returns `hardsigmoid_tile_init()` and `hardsigmoid_tile(idst)` with no template arguments; the API header `hardsigmoid.h` passes `APPROX` (the compile-time `math_approx_mode` value) as the template argument |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel ignores it entirely | The `calculate_hardsigmoid` function is templated on `APPROXIMATION_MODE` but does not use `if constexpr` or any branch that depends on it — the same piecewise-linear code executes regardless |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h` (identical for Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h` (identical for Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): Macro `SFPU_OP_CHAIN_0` expands to `hardsigmoid_tile(0);`.
2. **API Header** (`hardsigmoid.h`): `hardsigmoid_tile(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardsigmoid.h`): `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(dst_index)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` sets up DEST addressing, stalls until SFPU is ready, then loops over 4 faces calling the SFPU function once per face (with `VectorMode::RC`), advancing the DEST face address between faces.
5. **Core SFPU** (`ckernel_sfpu_hardsigmoid.h`): `calculate_hardsigmoid<APPROXIMATION_MODE, ITERATIONS=8>()` executes 8 iterations per face, performing the piecewise-linear hardsigmoid computation on each 32-element SFPU vector.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` — all 4 faces of the tile are processed (face 0–3), covering all 1024 elements.
- **Operation invocation**: The params dispatch function loops `for (int face = 0; face < 4; face++)`, calling `calculate_hardsigmoid<false, 8>()` once per face. Each invocation processes 8 SFPU iterations × 32 elements = 256 elements (one full face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr<8>` × 2 between faces). On Wormhole, the params dispatch uses `math::set_addr_mod_base()` at start and `math::clear_addr_mod_base()` at end. On Blackhole, the params dispatch uses `_llk_math_eltwise_unary_sfpu_start_` (which calls `math::set_dst_write_addr` + stall) and `_llk_math_eltwise_unary_sfpu_done_` (which clears dst reg addr). The `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>` sets only `ADDR_MOD_7` with `dest.incr=0` (no special hardsigmoid case in either architecture).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A is used. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h

namespace ckernel::sfpu {

// hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
// Piecewise linear:
//   x <= -3  =>  0
//   x >= 3   =>  1
//   else     =>  x * (1/6) + 0.5
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];           // SFPLOAD: load 32 elements from current DEST row pair
        sfpi::vFloat result = x * one_sixth + 0.5f;  // SFPMAD: result = x * (1/6) + 0.5 (fused multiply-add)

        // Clamp to [0, 1]
        v_if(result < 0.0f) { result = 0.0f; }       // SFPSETCC(CC_LT0) + conditional SFPLOADI(0.0) + SFPENCC
        v_endif;
        v_if(result > sfpi::vConst1) { result = sfpi::vConst1; } // SFPSETCC(CC_GTE0 on result-1.0) + conditional SFPLOADI(1.0) + SFPENCC
        v_endif;

        sfpi::dst_reg[0] = result;                    // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++;                              // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Description | Usage in Kernel |
|-------------|-------------|-----------------|
| `SFPLOAD` | Load 32 elements from a DEST row pair into an SFPU LREG | `sfpi::dst_reg[0]` reads the current DEST position into `x` |
| `SFPMAD` | Fused multiply-add: `a * b + c` | `x * one_sixth + 0.5f` computes the linear portion; also used for `result < 0.0f` (subtraction via MAD) and `result > vConst1` (comparison via MAD with negated 1.0) |
| `SFPLOADI` | Load an immediate constant into an SFPU LREG | Loading `one_sixth`, `0.5f`, `0.0f`, and `1.0f` constants; `sfpi::vConst1` may use a hardware constant register |
| `SFPSETCC` | Set condition code based on LREG sign/exponent bits | Used by `v_if(result < 0.0f)` and `v_if(result > sfpi::vConst1)` to enable conditional execution |
| `SFPENCC` | Enable (restore) condition code — end conditional block | Used by `v_endif` to restore the condition code state after each `v_if` block |
| `SFPSTORE` | Store 32 elements from an SFPU LREG back to a DEST row pair | `sfpi::dst_reg[0] = result` writes the clamped result back |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **DEST row pairs** (via `dst_reg`) | Source and destination for tile data. Each iteration processes one sfpi row = 2 physical DEST rows = 32 elements. The base address is set by `math::set_dst_write_addr` and advances via `dst_reg++`. |
| **LREG 0–3** (implicit) | SFPU local registers used by the compiler to hold intermediate values: `x`, `result`, `one_sixth`, `0.5f`, `0.0f`, `1.0f`. The SFPI compiler manages register allocation automatically. |
| **Condition Code (CC)** | Set by `SFPSETCC` during `v_if` comparisons. The first `v_if(result < 0.0f)` sets CC based on sign of `result`. The second `v_if(result > vConst1)` computes `result - 1.0` and sets CC based on sign. `SFPENCC` restores CC at each `v_endif`. |
| **`sfpi::vConst1`** | Hardware constant register holding `1.0f`, used as the upper clamp bound |

### Address Mode Configuration

The `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>()` function configures address modes during init. Since `SfpuType::hardsigmoid` does not match any special-case `if constexpr` branch in either architecture, only the default `ADDR_MOD_7` is configured:

| Address Mode | Field | Value | Description |
|-------------|-------|-------|-------------|
| `ADDR_MOD_7` | `srca.incr` | 0 | No auto-increment for source A |
| `ADDR_MOD_7` | `srcb.incr` | 0 | No auto-increment for source B |
| `ADDR_MOD_7` | `dest.incr` | 0 | No auto-increment for DEST |

This configuration is **identical for Wormhole and Blackhole**. The `dest.incr=0` means DEST addressing does not auto-increment via the address mode — instead, the SFPU kernel manages DEST progression explicitly via `dst_reg++` (which translates to SFPU register pointer advancement within the iteration loop) and `SETRWC`/`inc_dst_addr` between faces (handled by the params dispatch layer).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, approximation mode, and include macro for HARDSIGMOID
   **Key Findings**: `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default). `get_op_init_and_func_default` returns `hardsigmoid_tile_init()` / `hardsigmoid_tile(idst)`. `get_op_approx_mode` returns `false` (default). `get_macro_definition` returns `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (default — hardsigmoid is NOT in the split-include special cases).

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
   **Reason**: Trace API-level dispatch from `hardsigmoid_tile()` to LLK layer
   **Key Findings**: Calls `llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)` and `llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()`. Template parameter `APPROX` comes from the compile-time define.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
   **Reason**: Trace LLK dispatch to the core SFPU function
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardsigmoid<APPROXIMATE, 8>, dst_index, VectorMode::RC)`. Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>()`.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Read the core SFPU kernel source code
   **Key Findings**: Simple piecewise-linear function using SFPI abstractions. Computes `x * (1/6) + 0.5`, then clamps to [0, 1] using two `v_if` blocks. APPROXIMATION_MODE template parameter is unused.

5. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
   **Reason**: Verify Blackhole implementation matches Wormhole
   **Key Findings**: Identical to Wormhole implementation.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch layer (face loop, DEST progression)
   **Key Findings**: VectorMode::RC loops over 4 faces, calling the SFPU function once per face, with `TTI_SETRWC` advancing DEST address by face stride between faces. Uses `TTI_STALLWAIT` to synchronize SFPU.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Verify Blackhole params dispatch matches Wormhole pattern
   **Key Findings**: Same 4-face loop structure but uses `_llk_math_eltwise_unary_sfpu_start_/_done_/_inc_dst_face_addr_` helper functions instead of inline TTI instructions.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardsigmoid>` only sets `ADDR_MOD_7` with all increments = 0. No special case for hardsigmoid.

9. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
   **Reason**: Understand how hardsigmoid.h is included by the compute kernel
   **Key Findings**: Aggregation header that includes `hardsigmoid.h`. However, the primary include path for hardsigmoid is via `llk_math_unary_sfpu_api.h` which directly includes the LLK hardsigmoid header.

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
    **Reason**: Verify hardsigmoid LLK header is always included (not gated by split-include macro)
    **Key Findings**: Directly includes `llk_math_eltwise_unary_sfpu_hardsigmoid.h` — no conditional compilation guard needed.
