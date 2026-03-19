## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SOFTSHRINK`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `softshrink_tile(0, <param0_as_uint32>u)` where `param0` is the lambda parameter bit-cast from float to uint32

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SOFTSHRINK)` in `unary_op_utils.cpp` -- switch has only a `default: return false` case |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` returns `softshrink_tile_init()` / `softshrink_tile({idst}, {param0}u)` -- no template parameter exposed in the chain macro; the API header `softshrink_tile()` passes `APPROX` (resolved from `math_approx_mode`) |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the kernel does not branch on it | The `calculate_softshrink` function template accepts `APPROXIMATION_MODE` but does not use it in any `if constexpr` branch -- the same code path executes regardless |

### SFPU Abstraction Layers
List the file path for each abstraction layer. If a layer does not exist for this operation, write "This level of abstraction doesn't exist" instead of a path.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_activations.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`softshrink_tile(idst, param0)`** (API header `activations.h`) calls `MATH((llk_math_eltwise_unary_sfpu_softshrink<APPROX>(idst, param0)))`.
2. **`llk_math_eltwise_unary_sfpu_softshrink<APPROX>(dst_index, param0)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_activations.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_softshrink<APPROX, 8>, dst_index, VectorMode::RC, param0)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROX>(...)`** (Parameters dispatch `llk_math_eltwise_unary_sfpu_params.h`) sets DEST write address, stalls for SFPU readiness, then loops over 4 faces calling `calculate_softshrink<false, 8>(param0)` per face with `SETRWC` between faces.
4. **`calculate_softshrink<false, 8>(param0)`** (Core SFPU `ckernel_sfpu_softshrink.h`) executes 8 SFPI iterations per face, performing the softshrink piecewise function on 32 elements per iteration.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch loops over 4 faces, calling `calculate_softshrink<false, 8>(param0)` once per face. Each call internally loops 8 iterations (`ITERATIONS=8`), processing 32 elements per iteration (2 physical DEST rows x 16 elements/row due to stride-2).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch explicitly calls `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 physical rows = 1 face). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` helper calls `inc_dst_addr<8>()` twice. The base address mode is `ADDR_MOD_7` (all increments = 0), set via `set_addr_mod_base()` which selects the upper addr_mod bank (4..7).

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_elseif`/`v_endif`), so Style A applies.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h
// (Blackhole implementation is identical)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(uint32_t param0) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Softshrink(x) = x - lambda if x > lambda, x + lambda if x < -lambda, else 0
    sfpi::vFloat lambda = Converter::as_float(param0); // reinterpret uint32 bits as float -> vFloat
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];       // SFPLOAD: load 32 elements from current DEST row into LREG
        sfpi::dst_reg[0] = sfpi::vConst0;         // SFPSTORE: write 0.0 to current DEST row (default output)
        v_if(v > lambda) {                         // SFPXFCMPV(v, lambda, CC_GT) + SFPPUSHC + SFPXCONDB: enable lanes where v > lambda
            sfpi::dst_reg[0] = v - lambda;         // CC-guarded: SFPMAD(v, 1.0, -lambda) -> SFPSTORE to DEST
        }
        v_elseif(v < (-lambda)) {                  // SFPCOMPC + SFPPUSHC + SFPXFCMPV(v, -lambda, CC_LT) + SFPXCONDB: enable lanes where v < -lambda
            sfpi::dst_reg[0] = v + lambda;         // CC-guarded: SFPMAD(v, 1.0, lambda) -> SFPSTORE to DEST
        }
        v_endif;                                   // SFPPOPC (pop inner) + SFPPOPC (pop outer): restore CC state
        sfpi::dst_reg++;                           // advance to next sfpi row (next 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Load 32 elements from DEST register row into an SFPU LREG. Emitted by `sfpi::dst_reg[0]` read. |
| `SFPSTORE` | Store 32 elements from an SFPU LREG back to DEST register row. Emitted by `sfpi::dst_reg[0] = ...` write. |
| `SFPMAD` | Fused multiply-add (a * b + c). Emitted by `v - lambda` as `v * 1.0 + (-lambda)` and by `v + lambda` as `v * 1.0 + lambda`. There is no dedicated float add instruction; addition/subtraction is always via SFPMAD. |
| `SFPXFCMPV` | Vector floating-point compare. Emitted by `v > lambda` (with CC_GT mode) and `v < (-lambda)` (with CC_LT mode). Sets condition code per-lane based on the comparison result. |
| `SFPPUSHC` | Push current condition code state onto the CC stack. Part of the `v_if`/`v_elseif` mechanism to enable nested conditionals. |
| `SFPXCONDB` | Conditional branch/enable based on comparison result. Applies the comparison result to the condition code to gate subsequent stores. |
| `SFPCOMPC` | Complement condition code. Used by `v_elseif` to flip the condition (from "matched first branch" to "did not match first branch") before evaluating the next condition. |
| `SFPPOPC` | Pop condition code state from the CC stack. Used by `v_endif` to restore the previous CC state. Two pops occur: one for the inner `v_elseif` push and one for the outer `v_if` push. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST row (via `dst_reg[0]`)** | Source and destination for tile data. Each iteration reads 32 elements from the current DEST row, conditionally modifies them, and writes back. |
| **LREG (implicit)** | The SFPI compiler allocates local registers (LREGs) for intermediate values. `v` (the loaded input value) and `lambda` (the parameter) each occupy an LREG. The SFPMAD result is stored in an LREG before being written back to DEST. |
| **CREG_IDX_0 (`vConst0`)** | Constant register holding 0.0. Used as the default output value written to DEST before conditional branches. |
| **CC stack** | The condition code stack is used by `SFPPUSHC`/`SFPPOPC` to save and restore CC state for the `v_if`/`v_elseif`/`v_endif` construct. |

### Address Mode Configuration

The `softshrink` operation uses `SfpuType::softshrink`, which does not match any special-case `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod()`. Therefore, only the default address mode is configured:

| Address Mode | Field Values | Purpose |
|-------------|-------------|---------|
| `ADDR_MOD_7` | `.srca.incr=0, .srcb.incr=0, .dest.incr=0` | Default SFPU address mode. DEST auto-increment is 0 because the SFPI `dst_reg++` instruction handles address advancement explicitly via software. |

This configuration is identical on both Wormhole and Blackhole. The `set_addr_mod_base()` call in the params dispatch switches to the upper address mode bank (4..7), so `ADDR_MOD_7` is active during SFPU execution. After completion, `clear_addr_mod_base()` restores the lower bank (0..3).

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "How do SFPI conditional execution constructs v_if, v_elseif, v_endif work in terms of SFPU instructions? What instructions do they emit (e.g., SFPSETCC, SFPENCC, SFPCOMPC)?"
   **Reason**: Needed to understand what low-level SFPU instructions the `v_if`/`v_elseif`/`v_endif` constructs in the softshrink kernel translate to.
   **Key Findings**: These constructs use `SFPSETCC`, `SFPCOMPC`, and `SFPENCC` for CC management. Both branches are executed but only enabled lanes write results (predicated execution). However, the newer SFPI framework uses `SFPPUSHC`/`SFPPOPC`/`SFPXCONDB` for structured conditional execution via a CC stack.

2. [SFPU] **Query**: "In the SFPI framework, what SFPU instruction does vFloat subtraction (a - b) and vFloat addition (a + b) emit?"
   **Reason**: Needed to confirm that `v - lambda` and `v + lambda` in the kernel emit SFPMAD instructions (since there is no dedicated float add/sub instruction).
   **Key Findings**: Confirmed that vFloat addition and subtraction are compiled into SFPMAD instructions. `a + b` becomes `SFPMAD(a, 1.0, b)` and `a - b` becomes `SFPMAD(a, 1.0, -b)`.

3. [SFPU] **Query**: "What SFPU instruction does vFloat comparison v_if(a > b) emit for floating point values?"
   **Reason**: Needed to identify the comparison instruction emitted for `v > lambda` and `v < (-lambda)`.
   **Key Findings**: DeepWiki was not fully conclusive for vFloat-vs-vFloat comparisons. Direct source inspection of `sfpi.h` revealed that `vFloat > vFloat` constructs a `__vCond` using `__builtin_rvtt_sfpxfcmpv` (SFPXFCMPV instruction) with the appropriate CC mode (CC_GT or CC_LT).

### Confluence References
No Confluence pages were consulted for this analysis.

### Glean References
No Glean searches were performed for this analysis.
