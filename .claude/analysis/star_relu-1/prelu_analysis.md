## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path)
- **SFPU_OP_CHAIN_0 expansion**: `prelu_tile(0, {param0_hex}u)` where `param0` is the bit-cast `uint32_t` representation of the PReLU slope parameter

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default `APPROX`) | `get_op_init_and_func()` returns `prelu_tile_init()` and `prelu_tile({idst}, {param0}u)` -- the API header defines `APPROX` which resolves to the `math_approx_mode` define (false) |
| Effective SFPU path | `APPROXIMATION_MODE=false` but this has no effect -- `calculate_prelu` does not branch on `APPROXIMATION_MODE` | The template parameter is declared but unused in `ckernel_sfpu_prelu.h` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`prelu_tile(idst, param0)`** (API header `prelu.h`) expands via `MATH(...)` to call `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_prelu, RC, APPROX, idst, param0)`.
2. **`SFPU_UNARY_ONE_PARAM_KERNEL_FN`** (macro in `llk_math_eltwise_unary_sfpu_macros.h`) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_prelu<APPROXIMATE>, idst, (int)VectorMode::RC, param0)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets up the DEST write address, stalls until SFPU is ready, then iterates over all 4 faces (since `VectorMode::RC`), calling `calculate_prelu<false>(param0)` once per face, with `SETRWC`/`inc_dst_face_addr` between faces.
4. **`calculate_prelu<false>`** (in `ckernel_sfpu_prelu.h`) executes the core SFPU microcode: 8 iterations per face, each loading 32 elements from DEST, conditionally multiplying negative values by the slope, and writing back.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (Face 0, 1, 2, 3).
- **Operation invocation**: `calculate_prelu<APPROXIMATE>` is called once per face with 8 iterations (default `ITERATIONS=8`), for a total of 4 calls x 8 iterations = 32 sfpi iterations covering 1024 elements.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` with `CR_D, 8` twice between faces (advancing by 16 sfpi rows = 1 face). On Blackhole, equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

**Wormhole B0 variant:**

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    vFloat init = Converter::as_float(value); // Reinterpret uint32 param as float (the PReLU slope)

#pragma GCC unroll 8 // Compiler hint to fully unroll the loop
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position into LREG
        v_if(a < 0.0f) { a = a * init; } // SFPSETCC on (a < 0.0), then SFPMAD: a = a * init + 0.0 (conditional)
        v_endif;
        dst_reg[0] = a; // SFPSTORE: write 32 elements back to DEST
        dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

**Blackhole variant:**

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    vFloat init = Converter::as_float(value); // Reinterpret uint32 param as float (the PReLU slope)

#pragma GCC unroll 0 // Compiler hint to NOT unroll the loop (Blackhole differs from Wormhole here)
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position into LREG
        v_if(a < 0.0f) { a = a * init; } // SFPSETCC on (a < 0.0), then SFPMAD: a = a * init + 0.0 (conditional)
        v_endif;
        dst_reg[0] = a; // SFPSTORE: write 32 elements back to DEST
        dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

The only difference between the Wormhole and Blackhole variants is the `#pragma GCC unroll` directive: Wormhole uses `unroll 8` (full unroll) while Blackhole uses `unroll 0` (no unroll). The `const` qualifier on the `value` parameter in Blackhole is also a minor difference. The core SFPU logic is identical.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements (2 physical DEST rows) from the current DEST address into an SFPU local register (LREG) |
| **SFPSTORE** | `dst_reg[0] = a` (write) | Store 32 elements from an SFPU local register back to the current DEST address |
| **SFPSETCC** | `v_if(a < 0.0f)` | Set condition codes based on comparing `a` against 0.0 (sign check). Elements where `a < 0.0` have their CC bit set, enabling conditional execution for the subsequent SFPMAD |
| **SFPMAD** | `a * init` | Fused multiply-add: computes `a * init + 0.0`. This is conditionally executed only for lanes where CC is set (negative elements). Positive/zero elements are passed through unchanged |
| **SFPENCC** | `v_endif` | End conditional execution block, restoring all lanes to active |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST** (tile rows) | Source and destination for tile data. Accessed via `dst_reg[0]` at stride-2 addressing. Each iteration processes 2 physical DEST rows (32 elements). |
| **LREG (vFloat a)** | Temporary local register holding the loaded tile data for comparison and conditional multiply |
| **LREG (vFloat init)** | Holds the PReLU slope parameter, reinterpreted from the `uint32_t` bit pattern via `Converter::as_float()`. This value persists across all 8 iterations within a face call. |

### Address Mode Configuration

The PReLU operation uses `SfpuType::prelu` which does NOT match any special-case `if constexpr` branch in `eltwise_unary_sfpu_configure_addrmod()`. Therefore, only the default `ADDR_MOD_7` is configured:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default no-increment mode for SFPU operations. DEST address progression is handled explicitly by `dst_reg++` (within-face) and `SETRWC`/`inc_dst_face_addr` (between faces), not by hardware auto-increment. |

This configuration is identical on both Wormhole B0 and Blackhole.

## External Knowledge Sources
### DeepWiki Queries
No DeepWiki queries were needed for this analysis. The PReLU SFPU kernel is straightforward (conditional multiply using SFPI abstractions) and all necessary information was obtained directly from the source code.

### Confluence References
No Confluence page sections were consulted. The SFPU instructions used (SFPLOAD, SFPSTORE, SFPMAD, SFPSETCC, SFPENCC) are standard and well-understood from the SFPI abstraction layer.

### Glean References
No confidential hardware specifications were retrieved. The operation does not use any hardware-specific features beyond standard SFPU conditional execution.
