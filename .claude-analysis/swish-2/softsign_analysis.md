## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**IMPORTANT NOTE**: The softsign SFPU kernel is currently **stubbed out** on both Wormhole and Blackhole. The `calculate_softsign()` and `softsign_init()` functions have empty bodies with the comment: _"Implementation removed -- depends on recip primitive (Family 3)"_. All dispatch wiring (API header, LLK dispatch, ckernel header) is in place, but the actual SFPU computation is a no-op. This analysis documents the complete dispatch chain and wiring that exists, which would be used once the kernel is implemented.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSIGN`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `softsign_tile(0)` (with init: `softsign_tile_init()`)
- **Math definition**: `softsign(x) = x / (1 + |x|)`
- **Dispatch path**: The operation is registered via `unary_ng` dispatch (`unary_ng_op_utils.cpp` line 90), not the legacy `unary_op_utils.cpp` dispatch (which does not have a SOFTSIGN case).

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode()` in `unary_ng_op_utils.cpp` line 150 -- returns `false` unconditionally (no switch cases, just `return false`) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` in `unary_ng_op_utils.cpp` line 90 -- non-parameterized: `softsign_tile_init()` / `softsign_tile(idst)` with no explicit template arguments |
| Effective SFPU path | `APPROX=false` passed through all layers | The API header `softsign.h` uses `<APPROX>` which resolves to `false`. This propagates to `llk_math_eltwise_unary_sfpu_softsign<false>()` and ultimately to `calculate_softsign<false, 8>()`. However, since the kernel body is empty, neither approximation path is executed. |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (the `_llk_math_eltwise_unary_sfpu_params_` function) |

### Call Chain
1. **Compute kernel** calls `softsign_tile(idst)` (expanded from `SFPU_OP_CHAIN_0`).
2. **API Header** (`softsign.h` line 27): `softsign_tile(idst)` calls `MATH((llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_softsign.h` line 19): `llk_math_eltwise_unary_sfpu_softsign<APPROXIMATE>(dst_index)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_softsign<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls for SFPU, iterates over faces calling the SFPU function once per face (4 faces for `VectorMode::RC`), then advances the DEST write address between faces.
5. **Core SFPU** (`ckernel_sfpu_softsign.h` line 14): `calculate_softsign<false, 8>()` is called -- but the function body is empty (stubbed out).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default, processes all 4 faces of the tile).
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function loops over 4 faces, calling `calculate_softsign<false, 8>()` once per face. Each call is expected to process 8 SFPU iterations (one full face of 256 elements) using the default `ITERATIONS=8` template parameter.
- **DEST address progression**: Standard DEST progression. On Wormhole, the params function uses `TTI_SETRWC` to advance the DEST write counter by 8 rows between calls (equivalent to one face stride = 16 physical DEST rows). On Blackhole, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice. Within a face, the kernel (once implemented) would use `dst_reg++` to advance 1 sfpi row per iteration.

### Annotated SFPU Kernel Source

The kernel is **stubbed out** on both Wormhole and Blackhole. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h
// (Blackhole version at tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h is identical)

namespace ckernel::sfpu {

// Implementation removed -- depends on recip primitive (Family 3)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>  // APPROXIMATION_MODE=false, ITERATIONS=8
inline void calculate_softsign() {}

template <bool APPROXIMATION_MODE>  // APPROXIMATION_MODE=false
inline void softsign_init() {}

}  // namespace ckernel::sfpu
```

**Why it is stubbed**: The comment states the implementation "depends on recip primitive (Family 3)". The softsign formula `x / (1 + |x|)` requires computing a reciprocal (`1 / (1 + |x|)`). The reciprocal primitive is classified as "Family 3" in the operation taxonomy, and the softsign kernel was removed because it depends on that primitive being available. A full implementation would need to:
1. Load the element from DEST (`dst_reg[0]`)
2. Compute `|x|` (absolute value)
3. Compute `1 + |x|`
4. Compute the reciprocal `1 / (1 + |x|)` (using `SFPNONLINEAR` with reciprocal mode, or a software reciprocal approximation)
5. Multiply by `x` to get `x / (1 + |x|)`
6. Store the result back to DEST

### SFPU Instructions Used
No SFPU instructions are currently used -- the kernel body is empty.

When implemented, the following instructions would likely be needed:

| Instruction | Purpose |
|-------------|---------|
| `SFPLOAD` | Load element from DEST to LREG |
| `SFPABS` | Compute absolute value `\|x\|` |
| `SFPMAD` or `SFPADD` | Compute `1 + \|x\|` (add 1.0 to absolute value) |
| `SFPNONLINEAR` (mode 0) | Hardware-accelerated reciprocal `1 / (1 + \|x\|)` |
| `SFPMUL` or `SFPMAD` | Multiply `x * reciprocal` to get final result |
| `SFPSTORE` | Store result back to DEST |

### SFPU Register Usage
No registers are currently used -- the kernel body is empty.

When implemented, the kernel would use:
- **DEST register**: Input tile data (read via `SFPLOAD`) and output tile data (written via `SFPSTORE`)
- **LREGs**: Temporary storage for intermediate values (`x`, `|x|`, `1 + |x|`, reciprocal, final result)

### Address Mode Configuration
The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::softsign>()` during init. Since `SfpuType::softsign` does not match any of the special-cased `if constexpr` branches, only the default `ADDR_MOD_7` is set:

**Wormhole and Blackhole** (identical):
```
ADDR_MOD_7: {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0}
}
```

This means no auto-increment on DEST addressing -- the SFPU kernel (once implemented) would need to manage DEST address progression manually via `dst_reg++` in its iteration loop. This is the standard configuration for most unary SFPU operations.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Confirm SOFTSIGN is a registered UnaryOpType
   **Key Findings**: `SOFTSIGN` is at line 124 in the enum

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
   **Reason**: Find the SFPU_OP_CHAIN_0 expansion and dispatch configuration
   **Key Findings**: Line 90 maps SOFTSIGN to `softsign_tile_init()` / `softsign_tile(idst)`. Line 150 returns `false` for approx mode. Line 114 returns `eltwise_sfpu.cpp` as compute kernel path.

3. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Check if SOFTSIGN has a legacy dispatch path
   **Key Findings**: SOFTSIGN is NOT in the legacy `get_op_init_and_func_default` or `get_op_init_and_func_parameterized` switch statements. The operation uses the `unary_ng` path only.

4. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/softsign.h`
   **Reason**: API header that exposes `softsign_tile()` and `softsign_tile_init()`
   **Key Findings**: Calls `llk_math_eltwise_unary_sfpu_softsign<APPROX>(idst)` and `llk_math_eltwise_unary_sfpu_softsign_init<APPROX>()`

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softsign.h`
   **Reason**: LLK dispatch layer for Wormhole
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` with `calculate_softsign<APPROXIMATE, ITERATIONS>` functor and `VectorMode::RC` default

6. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
   **Reason**: Core SFPU implementation for Wormhole
   **Key Findings**: Empty stub -- `calculate_softsign()` and `softsign_init()` have empty bodies with comment about recip primitive dependency

7. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
   **Reason**: Core SFPU implementation for Blackhole
   **Key Findings**: Identical to Wormhole -- empty stub

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that iterates over faces and calls the SFPU functor
   **Key Findings**: For VectorMode::RC, loops 4 faces, calling the SFPU function once per face, advancing DEST by 16 physical rows (2x SETRWC of 8) between faces

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and address mode configuration
   **Key Findings**: `_llk_math_eltwise_unary_sfpu_init_<SfpuType::softsign>()` configures `ADDR_MOD_7` with all-zero increments (default path, no special cases for softsign)

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware model reference for instruction semantics and addressing
    **Key Findings**: Used for documenting expected instruction usage and DEST addressing patterns
