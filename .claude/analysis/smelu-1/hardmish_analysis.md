## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `HARDMISH`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `hardmish_tile(0)` (non-parameterized) or `hardmish_tile<1u>(0)` (parameterized, when `param0 = static_cast<float>(true)` i.e. `1u`)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDMISH)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit case for HARDMISH) |
| Template parameter (SFPU_OP_CHAIN) | Parameterized: `1u` (from `param0`); Non-parameterized: none (default template args) | `get_op_init_and_func()` -- parameterized case: `hardmish_tile_init<1u>()` / `hardmish_tile<1u>(0)`; non-parameterized case: `hardmish_tile_init()` / `hardmish_tile(0)` |
| Effective SFPU path | `APPROXIMATION_MODE` is controlled by `APPROX` macro (derived from `math_approx_mode` = `false`). The template parameter (`1u` vs default) is passed through the API but the hardmish SFPU kernel does NOT use `APPROXIMATION_MODE` in any conditional -- the same code path executes regardless. | The `hardmish()` function in `ckernel_sfpu_hardmish.h` has no `if constexpr (APPROXIMATION_MODE)` branches |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardmish.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardmish.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardmish.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`hardmish_tile(idst)`** (API header `hardmish.h`) calls `llk_math_eltwise_unary_sfpu_hardmish<APPROX>(idst)` inside a `MATH(...)` guard.
2. **`llk_math_eltwise_unary_sfpu_hardmish<APPROXIMATE>(dst_index)`** (LLK dispatch) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu::hardmish<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (params dispatch in `llk_math_eltwise_unary_sfpu_params.h`) sets DEST write address, stalls for SFPU, then loops over 4 faces (VectorMode::RC), calling the SFPU functor once per face with `SETRWC` between faces.
4. **`sfpu::hardmish<APPROXIMATE, 8>()`** (core SFPU in `ckernel_sfpu_hardmish.h`) executes 8 iterations per face, processing 32 elements per iteration via SFPI `vFloat` operations.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (all 4 faces processed). The params dispatch loops `for (int face = 0; face < 4; face++)`, calling the SFPU functor once per face.
- **Operation invocation**: The core function `hardmish<APPROXIMATE, 8>()` is called 4 times (once per face). Each invocation runs an internal loop of `ITERATIONS=8`, processing 8 sfpi rows per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The `SETRWC` advances by 16 physical DEST rows (two calls of `TTI_SETRWC(..., 8, ...)`) between faces. Within a face, `dst_reg++` advances 1 sfpi row = 2 physical DEST rows = 32 elements per iteration.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `sfpi::vec_min_max`), so Style A applies.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_hardmish.h

// hardmish(x) = x * clamp(x + 2.8, 0.0, 5.0) / 5
//             = x * clamp(x + 2.8, 0.0, 5.0) * 0.2
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];     // SFPLOAD: load 32 elements from current DEST row into LREG
        sfpi::vFloat a = x + 2.8f;             // SFPADDI or SFPMAD: a = x + 2.8

        // sfpi::vec_min_max(a, b) puts min in a, max in b
        sfpi::vFloat low_bound = 0.0f;         // SFPLOADI: load 0.0 into LREG
        sfpi::vFloat high_bound = 5.0f;        // SFPLOADI: load 5.0 into LREG
        sfpi::vec_min_max(low_bound, a);        // SFPSWAP(VEC_MIN_MAX): low_bound=min(0,a), a=max(0,a) -- clamp lower bound
        sfpi::vec_min_max(a, high_bound);       // SFPSWAP(VEC_MIN_MAX): a=min(a,5), high_bound=max(a,5) -- clamp upper bound

        sfpi::dst_reg[0] = x * a * 0.2f;       // SFPMAD + SFPMULI or chain: x*a*0.2, then SFPSTORE to DEST
        sfpi::dst_reg++;                        // advance DEST pointer by 1 sfpi row (2 physical rows)
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** | Loads 32 elements from current DEST row pair into an LREG. Emitted by `sfpi::dst_reg[0]` read. |
| **SFPLOADI** | Loads an immediate float constant into an LREG. Emitted by `vFloat low_bound = 0.0f` and `vFloat high_bound = 5.0f`. |
| **SFPADDI / SFPMAD** | Adds float constant 2.8 to the loaded value. The SFPI compiler may emit `SFPADDI` (add-immediate) or fold into `SFPMAD` (multiply-add with multiply-by-1.0). Emitted by `x + 2.8f`. |
| **SFPSWAP** | With `MOD1_VEC_MIN_MAX` mode: compares two LREG vectors element-wise and places min in one, max in the other. Emitted by `sfpi::vec_min_max()`. Two SFPSWAP instructions implement the `clamp(a, 0.0, 5.0)` operation. |
| **SFPMAD** | Multiply-accumulate: computes `a * b + c`. Used for the `x * a * 0.2f` expression, likely as a chain of two SFPMAD instructions (one for `x * a`, one for `result * 0.2f`, or fused). |
| **SFPMULI** | Multiply-immediate: may be used for `* 0.2f` if the compiler optimizes the constant multiply separately. |
| **SFPSTORE** | Stores 32 elements from an LREG back to the current DEST row pair. Emitted by `sfpi::dst_reg[0] = ...` write. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. Each `dst_reg[0]` access reads/writes a pair of physical DEST rows (32 elements). The DEST pointer auto-advances via `dst_reg++`. |
| **LREG (temporary)** | The SFPI compiler allocates LREGs automatically for `vFloat` variables. At minimum: one LREG for `x`, one for `a`, one for `low_bound`, one for `high_bound`. The compiler may reuse LREGs as variables go out of scope. Up to 4 LREGs may be live simultaneously. |
| **LREG (constants)** | `0.0f`, `2.8f`, `5.0f`, and `0.2f` are loaded into LREGs as immediate constants via `SFPLOADI`. |

### Address Mode Configuration

The init function `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardmish>()` configures:

- **ADDR_MOD_7**: `srca.incr=0, srcb.incr=0, dest.incr=0` (all-zero increments)
  - This is the only address mode configured for hardmish on both Wormhole B0 and Blackhole.
  - `SfpuType::hardmish` does not match any of the special-case `if constexpr` branches (topk_local_sort, typecast, unary_max/min), so no ADDR_MOD_6 is set.

The DEST address progression between iterations is handled explicitly by the SFPI `dst_reg++` mechanism (which advances the SFPU's internal DEST pointer by `SFP_DESTREG_STRIDE=2` physical rows per iteration), not by hardware auto-increment via ADDR_MOD. Between faces, the params dispatch issues two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls, advancing the DEST read/write counter by 16 physical rows (one face stride).

This configuration is identical on Wormhole B0 and Blackhole.

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "How does the SFPSWAP instruction work in the SFPU? Specifically, what does SFPSWAP_MOD1_VEC_MIN_MAX mode do?"
   **Reason**: The hardmish kernel uses `sfpi::vec_min_max()` which compiles to SFPSWAP. Needed to understand the element-wise min/max semantics and register placement.
   **Key Findings**: SFPSWAP with VEC_MIN_MAX mode compares two LREG vectors element-wise, placing the minimum in the first register and maximum in the second. This is consistent across Wormhole B0 and Blackhole architectures.

2. [SFPU] **Query**: "In the SFPI programming model, what SFPU instructions are emitted for vFloat arithmetic operations like addition, multiplication, and loading constants?"
   **Reason**: The hardmish kernel uses vFloat addition (`x + 2.8f`), multiplication (`x * a * 0.2f`), and constant loading. Needed to map these to concrete SFPU instructions.
   **Key Findings**: vFloat addition with a constant can emit SFPADDI or SFPMAD (a*1.0+b). Multiplication can emit SFPMAD or SFPMULI. Constants are loaded via SFPLOADI. The SFPI compiler optimizer handles MAD generation and immediate-operand optimizations.

### Confluence References
No Confluence pages were consulted for this analysis.

### Glean References
No Glean searches were performed for this analysis.
