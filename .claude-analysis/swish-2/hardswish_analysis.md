## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDSWISH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `hardswish_tile_init(); hardswish_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDSWISH)` in `unary_op_utils.cpp` — falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` — non-parameterized: `hardswish_tile_init()` / `hardswish_tile({idst})` with no template arguments passed. The API header's `APPROX` constant (set to `false` by genfiles.cpp from `math_approx_mode`) is forwarded as the template argument. |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout | `calculate_hardswish<false, 8>()` — the kernel does not branch on `APPROXIMATION_MODE`, so the value has no effect on execution |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` expands to `hardswish_tile_init(); hardswish_tile(0);`
2. **API Header** (`hardswish.h`): `hardswish_tile(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst)))`, and `hardswish_tile_init()` calls `MATH((llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>()))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardswish.h`): `llk_math_eltwise_unary_sfpu_hardswish<APPROXIMATE>(dst_index)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardswish<APPROXIMATE, 8>, dst_index, (int)VectorMode::RC)`. The init variant calls `llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>()`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, stalls until SFPU is ready, then loops over 4 faces calling `calculate_hardswish<false, 8>()` once per face, with `SETRWC`/`inc_dst_addr<8>` between faces.
5. **Core SFPU** (`ckernel_sfpu_hardswish.h`): `calculate_hardswish()` performs 8 iterations per face, each processing 32 elements via SFPI abstractions.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (hardcoded default in `llk_math_eltwise_unary_sfpu_hardswish`). Processes all 4 faces of the tile (face 0–3), covering all 1024 elements.
- **Operation invocation**: The params dispatch calls `calculate_hardswish<false, 8>()` once per face (4 times total). Each invocation runs the inner loop for `ITERATIONS=8`, processing 8 sfpi rows × 32 elements = 256 elements = 1 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr<8>` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 physical DEST rows = 1 face). On Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (same effect). Only `ADDR_MOD_7` is configured for this operation, with all increments set to 0 — the auto-increment comes from `dst_reg++` within the SFPI kernel, not from address mode registers.

### Annotated SFPU Kernel Source

The kernel uses **Style A: SFPI-based kernel** — all logic is expressed through SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`).

The Wormhole B0 and Blackhole implementations are **identical**.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h

namespace ckernel::sfpu {

// hardswish(x) = x * min(max(x + 3, 0), 6) / 6
//              = x * hardsigmoid(x)
//              = x * clamp(x/6 + 0.5, 0, 1)
// Piecewise:
//   x <= -3  =>  0
//   x >= 3   =>  x
//   else     =>  x * (x/6 + 0.5)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardswish() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float one_sixth = 1.0f / 6.0f; // ~0.16667f, loaded via SFPLOADI pair

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD from current DEST row pair
        sfpi::vFloat hsigmoid = x * one_sixth + 0.5f; // SFPMAD(x, 1/6, 0) then SFPMAD(result, 1.0, 0.5)

        // Clamp hardsigmoid to [0, 1]
        v_if(hsigmoid < 0.0f) { hsigmoid = 0.0f; } // SFPSETCC(LT0) on hsigmoid; guarded SFPLOADI 0.0 -> hsigmoid
        v_endif; // SFPENCC to restore unconditional execution
        v_if(hsigmoid > sfpi::vConst1) { hsigmoid = sfpi::vConst1; } // negate hsigmoid, SFPSETCC(LT0) on (vConst1 - hsigmoid); guarded MOV vConst1 -> hsigmoid
        v_endif; // SFPENCC to restore unconditional execution

        sfpi::dst_reg[0] = x * hsigmoid; // SFPMAD(x, hsigmoid, 0.0); SFPSTORE to current DEST row pair
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Description | Usage in Kernel |
|-------------|-------------|-----------------|
| `SFPLOAD` | Load from DEST row pair into LREG with format conversion | Loads `x` from `dst_reg[0]` at start of each iteration |
| `SFPLOADI` | Load 16-bit immediate into LREG | Loads constants `one_sixth` (~0.16667f), `0.5f`, and `0.0f` — each FP32 constant requires two SFPLOADI instructions (upper 16 bits + lower 16 bits) |
| `SFPMAD` | Fused multiply-add: VD = VA × VB + VC | Computes `x * one_sixth + 0.5f` (the hardsigmoid linear part), and `x * hsigmoid` (the final product). Float addition `a + b` is also emitted as `SFPMAD(a, 1.0, b)` |
| `SFPSTORE` | Store LREG to DEST row pair with format conversion | Writes final result `x * hsigmoid` back to `dst_reg[0]` |
| `SFPSETCC` | Set CC.Res based on comparison (predicated) | Used by `v_if(hsigmoid < 0.0f)` and `v_if(hsigmoid > vConst1)` to set per-lane condition codes |
| `SFPENCC` | Enable/disable CC masking | Used by `v_if` to enable CC masking before the comparison, and by `v_endif` to disable CC masking (restore unconditional execution) |
| `SFPCOMPC` | Complement CC.Res | May be emitted by the SFPI compiler as part of the `>` comparison lowering (negating the `<=` test) |
| `SFPMOV` | Register copy | May be used for intermediate value shuffling between LREGs |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows (via `dst_reg`)** | Source of input `x` (SFPLOAD) and destination for output `x * hsigmoid` (SFPSTORE). Accessed through stride-2 addressing: each `dst_reg[0]` access reads/writes 2 physical DEST rows (32 elements). |
| **LREG0–LREG3** | Used as working registers for intermediate values: `x`, `hsigmoid`, temporary products. The SFPI compiler allocates LREGs automatically. |
| **Constant: `sfpi::vConst1`** | Hardware fixed constant register = `1.0f` (Fixed Const 2, hex `0x3F80_0000`). Used for the upper clamp `hsigmoid > 1.0f` and as the clamp target value. |
| **Constant: `sfpi::vConst0`** | Hardware fixed constant register = `0.0f` (Fixed Const 1, hex `0x0000_0000`). Implicitly used when the SFPI compiler lowers `0.0f` literals, though the kernel may also use SFPLOADI for zero. |
| **Scalar constants** | `one_sixth` (1/6 ≈ 0.16667f) and `0.5f` are loaded via SFPLOADI pairs into LREGs. These are compile-time constants embedded in the instruction stream. |

### Address Mode Configuration

For `SfpuType::hardswish`, the `eltwise_unary_sfpu_configure_addrmod()` function (in `llk_math_eltwise_unary_sfpu.h`) configures only one address mode:

**ADDR_MOD_7** (configured identically on both Wormhole B0 and Blackhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
.set(ADDR_MOD_7);
```

No additional address modes are configured because `SfpuType::hardswish` does not match any of the special-case `if constexpr` branches (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.) that set up `ADDR_MOD_6`.

The DEST address auto-increment is handled entirely within the SFPI kernel via `dst_reg++` (which advances the SFPU's internal DEST pointer by 1 sfpi row = 2 physical DEST rows). The address mode register's zero increment means the hardware does not apply any additional auto-increment on top of what the SFPI runtime manages.

Between faces, the params dispatch advances the DEST write address:
- **Wormhole B0**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` called twice (advancing 16 physical DEST rows per pair = 1 face)
- **Blackhole**: `math::inc_dst_addr<8>()` called twice via `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`

Both achieve the same result: advancing the DEST base address by 16 physical rows (1 face) between invocations of `calculate_hardswish()`.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, and approximation mode for HARDSWISH
   **Key Findings**: HARDSWISH uses `eltwise_sfpu.cpp`, define `SFPU_OP_HARDSWISH_INCLUDE`, init `hardswish_tile_init()`, func `hardswish_tile({idst})`, approx_mode=false (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`
   **Reason**: API header that exposes `hardswish_tile()` and `hardswish_tile_init()` to compute kernels
   **Key Findings**: Forwards to `llk_math_eltwise_unary_sfpu_hardswish<APPROX>()` and `llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>()`

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
   **Reason**: Conditional include mechanism that includes `hardswish.h` when `SFPU_OP_HARDSWISH_INCLUDE` is defined
   **Key Findings**: `#if SFPU_OP_HARDSWISH_INCLUDE` → `#include "api/compute/eltwise_unary/hardswish.h"`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU implementation
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_hardswish<APPROXIMATE, 8>, dst_index, VectorMode::RC)` with default ITERATIONS=8

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
   **Reason**: Core SFPU kernel implementation (primary analysis target)
   **Key Findings**: Pure SFPI kernel — computes `x * clamp(x/6 + 0.5, 0, 1)` using vFloat arithmetic and v_if clamps. 8 iterations per face, 32 elements per iteration. Does not branch on APPROXIMATION_MODE.

6. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
   **Reason**: Verify Blackhole variant matches Wormhole
   **Key Findings**: Identical implementation to Wormhole B0

7. **File**: `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that orchestrates per-face SFPU invocation
   **Key Findings**: VectorMode::RC loops over 4 faces, calling sfpu_func per face, with SETRWC(CR_D, 8) × 2 between faces

8. **File**: `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and ADDR_MOD configuration for SfpuType::hardswish
   **Key Findings**: Only ADDR_MOD_7 (all zeros) is configured; hardswish does not match any special-case if constexpr branches

9. **File**: `/localdev/vignjatijevic/tt-metal/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Verify Blackhole params dispatch
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_start_`/`_done_`/`_inc_dst_face_addr_` instead of direct TTI_SETRWC, but same logical behavior

10. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Determine how `APPROX` compile-time constant is generated
    **Key Findings**: `constexpr bool APPROX = {math_approx_mode};` is emitted into the generated scalar descriptors file

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU architecture, instruction semantics, and addressing model
    **Key Findings**: Confirmed stride-2 model, SFPMAD for float add/mul, ITERATIONS=8 per face, CC mechanism for v_if lowering
