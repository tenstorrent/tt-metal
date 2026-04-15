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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SWISH)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none (uses default `APPROX` compile-time define, which resolves to `false` from `math_approx_mode`) | `get_op_init_and_func_default()` returns `"swish_tile_init();"` and `"swish_tile({idst});"` -- no parameterized template argument |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel does not branch on `APPROXIMATION_MODE` at all -- the same piecewise polynomial+linear code executes unconditionally | `calculate_swish<APPROXIMATION_MODE, ITERATIONS>()` has no `if constexpr (APPROXIMATION_MODE)` branch |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`swish_tile(idst)`** (API header `swish.h`): Wraps in `MATH(...)` macro, calls `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_swish<APPROX>(dst_index, VectorMode::RC)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_swish.h`): Passes the core function `ckernel::sfpu::calculate_swish<APPROXIMATE, ITERATIONS=8>` as a callable to the params dispatch.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (params dispatch `llk_math_eltwise_unary_sfpu_params.h`): Sets DEST write address, stalls for SFPU readiness, then loops over 4 faces calling `calculate_swish<false, 8>()` for each face, advancing DEST face address between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
4. **`calculate_swish<false, 8>()`** (core SFPU `ckernel_sfpu_swish.h`): Executes 8 iterations per face, each processing 32 elements (2 physical DEST rows) via the piecewise sigmoid approximation.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch calls `calculate_swish<false, 8>()` once per face in a `for (int face = 0; face < 4; face++)` loop. Each invocation runs 8 iterations (ITERATIONS=8), processing the full face of 256 elements.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice between faces to advance by 16 physical DEST rows (one face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice, achieving the same effect.

### Annotated SFPU Kernel Source

This kernel uses **Style A** (SFPI-based abstractions). The `APPROXIMATION_MODE` template parameter is accepted but never branched on -- the same piecewise approximation code runs regardless of its value.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
// NOTE: Wormhole and Blackhole implementations are identical.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_swish() { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    // Polynomial coefficients for sigmoid(t) over [0, 2.5]
    // Fitted to minimize max error at t = 0, 0.5, 1.0, 1.5, 2.0, 2.5
    constexpr float c1 = 0.2533f;   // loaded via SFPLOADI into LREG
    constexpr float c2 = -0.01479f; // loaded via SFPLOADI into LREG
    constexpr float c3 = -0.00747f; // loaded via SFPLOADI into LREG

    // Linear segment coefficients for [2.5, 5.0]
    constexpr float lin_slope = 0.0276f;  // loaded via SFPLOADI into LREG
    constexpr float lin_offset = 0.855f;  // loaded via SFPLOADI into LREG

    // Breakpoints
    constexpr float bp1 = 2.5f;  // loaded via SFPLOADI for comparison
    constexpr float bp2 = 5.0f;  // loaded via SFPLOADI for comparison

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair

        // Compute sigmoid(|x|) using degree-3 polynomial for |x| <= 2.5
        sfpi::vFloat ax = sfpi::abs(x); // SFPABS (MOD1_FLOAT): clear sign bit
        // Horner-form polynomial: 0.5 + ax * (c1 + ax * (c2 + ax * c3))
        // Each * emits SFPMAD (a*b+0), each + emits SFPMAD (a*1+b)
        sfpi::vFloat sig_pos = 0.5f + ax * (c1 + ax * (c2 + ax * c3));

        // Override with linear segment for 2.5 < |x| <= 5.0
        v_if(ax > bp1) { // SFPPUSHC + SFPXFCMPS(GT) + SFPSETCC: push CC, compare ax > 2.5, set CC
            sig_pos = ax * lin_slope + lin_offset; // SFPMAD (mul) + SFPMAD (add), CC-guarded
        }
        v_endif; // SFPPOPC: pop CC stack, restore previous predication

        // Saturate to 1.0 for |x| > 5.0
        v_if(ax > bp2) { // SFPPUSHC + SFPXFCMPS(GT) + SFPSETCC: compare ax > 5.0
            sig_pos = sfpi::vConst1; // SFPREADLREG(CREG_IDX_1): load constant 1.0, CC-guarded
        }
        v_endif; // SFPPOPC

        // For x < 0: sigmoid(x) = 1 - sigmoid(|x|)
        v_if(x < 0.0f) { // SFPPUSHC + SFPXFCMPS(LT) + SFPSETCC: compare x < 0
            sig_pos = sfpi::vConst1 - sig_pos; // SFPREADLREG(CREG_IDX_1) + SFPMAD (sub via add of negation), CC-guarded
        }
        v_endif; // SFPPOPC

        // swish(x) = x * sigmoid(x)
        sfpi::dst_reg[0] = x * sig_pos; // SFPMAD (mul) + SFPSTORE: multiply and write back to DEST
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[0]` (read) | Loads 32 elements from current DEST row pair into an LREG |
| **SFPABS** | `sfpi::abs(x)` | Clears sign bit of each element (floating-point absolute value), `MOD1_FLOAT` |
| **SFPLOADI** | `vFloat(float_literal)` | Loads immediate float constants (0.5, 0.2533, -0.01479, -0.00747, 0.0276, 0.855, 2.5, 5.0, 0.0) into LREGs |
| **SFPMAD** | `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat` | Fused multiply-add: multiplication emits `a * b + 0`, addition emits `a * 1.0 + b`, subtraction emits `a * 1.0 + (-b)` via `__builtin_rvtt_sfpmul` / `__builtin_rvtt_sfpadd` |
| **SFPPUSHC** | `v_if(...)` | Pushes current condition code state onto the CC stack before entering a predicated block |
| **SFPXFCMPS** | `ax > bp1`, `ax > bp2`, `x < 0.0f` | Scalar float comparison: compares vFloat against immediate float, sets CC per lane (GT, LT modes) |
| **SFPSETCC** | (implicit in `v_if` via `cc_cond`) | Sets condition code lanes based on comparison result, enabling predicated execution |
| **SFPPOPC** | `v_endif` | Pops CC stack, restoring previous predication state |
| **SFPREADLREG** | `sfpi::vConst1` | Reads constant register CREG_IDX_1 (value 1.0) into an LREG |
| **SFPSTORE** | `sfpi::dst_reg[0] = ...` (write) | Stores 32 elements from LREG back to current DEST row pair |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST row pairs** | Input/output: each iteration reads from and writes to `dst_reg[0]` (current DEST row pair = 32 elements), then advances `dst_reg++` |
| **LREGs (L0-L7)** | Temporary storage for intermediate values. The compiler allocates LREGs for: `x` (input), `ax` (absolute value), `sig_pos` (sigmoid result), float constants (c1, c2, c3, lin_slope, lin_offset, bp1, bp2, 0.5, 0.0), and intermediate products. Up to 8 LREGs available (SFP_LREG_COUNT=8) |
| **CREG_IDX_1** | Constant register holding 1.0, accessed via `sfpi::vConst1` for saturation and sign-flip operations |
| **CC stack** | Condition code stack used by 3 nested `v_if`/`v_endif` blocks (max depth 1 at any point since they are sequential, not nested). Each `v_if` pushes CC, each `v_endif` pops |

### Address Mode Configuration

The `SfpuType::swish` operation uses the **default address mode configuration** in `eltwise_unary_sfpu_configure_addrmod<SfpuType::swish>()`. Since `swish` does not match any of the special-cased `SfpuType` values (topk_local_sort, typecast, unary_max, unary_min, etc.), only `ADDR_MOD_7` is configured:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| **ADDR_MOD_7** | 0 | 0 | 0 | Default for SFPU operations; no auto-increment since DEST addressing is managed by `dst_reg++` in the SFPI kernel and `SETRWC`/`inc_dst_addr` between faces in the params dispatch |

This configuration is **identical on both Wormhole and Blackhole**. The `swish` kernel does not set or use `ADDR_MOD_6` or any other address modes. DEST progression within a face is handled by the SFPI `dst_reg++` operator (which advances the SFPU's internal row pointer by stride 2), not by hardware auto-increment.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, and approximation mode for SWISH
   **Key Findings**: SWISH uses `eltwise_sfpu.cpp`, `swish_tile_init()/swish_tile(idst)`, approx mode returns `false` (default case), include guard is `SFPU_OP_SWISH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
   **Reason**: API header exposing `swish_tile()` and `swish_tile_init()` to the compute kernel
   **Key Findings**: `swish_tile(idst)` dispatches to `llk_math_eltwise_unary_sfpu_swish<APPROX>(idst)`, `swish_tile_init()` dispatches to `llk_math_eltwise_unary_sfpu_swish_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::swish, APPROXIMATE>()`, tile function passes `calculate_swish<APPROXIMATE, ITERATIONS=8>` to params dispatch with `VectorMode::RC`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
   **Reason**: Core SFPU implementation -- the actual computation kernel
   **Key Findings**: Implements swish via piecewise sigmoid approximation: degree-3 polynomial for |x| <= 2.5, linear segment for 2.5 < |x| <= 5.0, saturation to 1.0 for |x| > 5.0, with sign flip for negative inputs. WH and BH implementations are identical. APPROXIMATION_MODE is accepted but unused (no branching on it).

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- controls face iteration and DEST address progression
   **Key Findings**: VectorMode::RC processes all 4 faces, calling the SFPU function once per face. Wormhole uses `TTI_SETRWC` for inter-face DEST advancement; Blackhole uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (two `inc_dst_addr<8>()` calls).

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `SfpuType::swish` uses only the default `ADDR_MOD_7` (all zero increments). Does not match any special-cased SfpuType for ADDR_MOD_6.

7. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: SFPI C++ wrapper mapping to SFPU hardware instructions
   **Key Findings**: `vFloat + vFloat` maps to `__builtin_rvtt_sfpadd` (SFPMAD), `vFloat * vFloat` maps to `__builtin_rvtt_sfpmul` (SFPMAD), `v_if`/`v_endif` maps to SFPPUSHC/SFPXFCMPS/SFPSETCC/SFPPOPC, `dst_reg[0]` read maps to SFPLOAD, write maps to SFPSTORE.

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library functions including `abs()`
   **Key Findings**: `sfpi::abs(vFloat)` maps to `__builtin_rvtt_sfpabs` with `SFPABS_MOD1_FLOAT`.

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative hardware model for SFPU architecture, addressing, and tile geometry
   **Key Findings**: Stride-2 addressing model, 8 iterations per face, 32 elements per iteration, DEST row width of 16 elements.
