## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(0, <slope_as_u32>)` where `<slope_as_u32>` is `std::bit_cast<uint32_t>(param0)` -- the slope parameter reinterpreted as an unsigned 32-bit integer

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LEAKY_RELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` -- `leaky_relu_tile_init()` and `leaky_relu_tile(idst, slope)` have no template parameter for approximation; `APPROX` is passed via the macro but is always `false` from ComputeConfig |
| Effective SFPU path | Single code path; `APPROXIMATION_MODE` template parameter is `false` but the `_calculate_lrelu_` implementation does not branch on it -- the same instruction sequence is emitted regardless | `_calculate_lrelu_` in `ckernel_sfpu_relu.h` has no `if constexpr(APPROXIMATION_MODE)` branches |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (Blackhole) |
| **Parameters Dispatch** | Same as LLK Dispatch -- `_llk_math_eltwise_unary_sfpu_params_` is the unified dispatch function in the params header |

Note: There is also a metal-level wrapper at `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` that defines `calculate_lrelu<APPROXIMATION_MODE>()` which delegates to `_calculate_lrelu_<APPROXIMATION_MODE>()` from the tt_llk layer.

### Call Chain
1. **`leaky_relu_tile(idst, slope)`** (API header `relu.h`) expands `MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope))`.
2. **`SFPU_UNARY_ONE_PARAM_KERNEL_FN`** macro (in `llk_math_eltwise_unary_sfpu_macros.h`) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_lrelu<APPROXIMATE>, DST_IDX, (int)VectorMode::RC, slope)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU readiness, then calls `sfpu_func(args...)` once per face (4 times for `VectorMode::RC`), advancing DEST address between faces via `SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
4. **`calculate_lrelu<APPROX>(slope)`** (in metal `ckernel_sfpu_relu.h`) calls `_calculate_lrelu_<APPROX>(ITERATIONS=8, slope)`.
5. **`_calculate_lrelu_<APPROX>(iterations, slope)`** (in tt_llk `ckernel_sfpu_relu.h`) executes the core SFPU instruction sequence: loads the slope constant into LREG2, then loops 8 times per face, loading each element from DEST, conditionally multiplying by slope if negative, and storing back.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (the full 32x32 tile).
- **Operation invocation**: The core SFPU function `calculate_lrelu<false>` is called 4 times (once per face). Each invocation runs an internal loop of 8 iterations (the default `ITERATIONS=8`), processing all 8 sfpi rows of one face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces on Wormhole / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole). The SFPLOAD/SFPSTORE instructions use `ADDR_MOD_3` on Wormhole and `ADDR_MOD_7` on Blackhole, both configured with `dest.incr=0` (no auto-increment from ADDR_MOD; the `dst_reg++` SFPI statement handles per-iteration advancement).

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with CC manipulation (`SFPSETCC` / `SFPENCC`). However, the CC logic is straightforward -- a single SETCC/ENCC pair per iteration -- so Style A (inline annotation) is appropriate.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h
// (Blackhole version is identical except ADDR_MOD_7 replaces ADDR_MOD_3)

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope) // APPROXIMATION_MODE=false, iterations=8
{
    // Load slope as IEEE-754 float into LREG2 in two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);  // Load low 16 bits into LREG2 (mod1=10: low half)
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);       // Load high 16 bits into LREG2 (mod1=8: high half, preserves low)
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // Load element from DEST into LREG0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // Set CC if LREG0 is negative (sign bit = 1)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // CC-guarded: LREG0 = LREG0 * LREG2 (x * slope), only for negative elements
        TTI_SFPENCC(0, 0, 0, 0);                                                      // Clear CC -- all lanes enabled again
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // Store LREG0 back to DEST
        sfpi::dst_reg++;  // Advance to next sfpi row (2 physical DEST rows = 32 elements)
    }
}
```

**Semantic summary**: Leaky ReLU computes `f(x) = x if x >= 0, else slope * x`. The kernel achieves this by:
1. Loading the slope constant into LREG2 (done once before the loop).
2. For each vector of 32 elements: loading from DEST, setting the condition code based on the sign bit (negative = CC enabled), performing a CC-guarded multiply by slope (only negative lanes are modified), clearing CC, and storing back. Positive elements pass through unchanged because the multiply is only executed for CC-enabled (negative) lanes.

### SFPU Instructions Used

| Instruction | Mnemonic | Description |
|-------------|----------|-------------|
| `TT_SFPLOADI` | SFPLOADI | Loads a 16-bit immediate value into a specified LREG. Used twice to construct the full 32-bit IEEE-754 slope value in LREG2 (low half then high half). |
| `TTI_SFPLOAD` | SFPLOAD | Loads a 32-element vector from the current DEST row(s) into LREG0. Uses `InstrModLoadStore::DEFAULT` (float format). |
| `TTI_SFPSETCC` | SFPSETCC | Sets the condition code register based on the sign of LREG0. After this instruction, CC is enabled for lanes where the value is negative (sign bit = 1). mod1=0 means `CC <- (sign_bit(LREG_src) == 1)`. |
| `TTI_SFPMUL` | SFPMUL | CC-guarded floating-point multiply: `LREG0 = LREG0 * LREG2 + LCONST_0`. Only lanes with CC enabled (negative values) are modified. LCONST_0 = 0.0, so this is effectively `LREG0 = LREG0 * LREG2`. |
| `TTI_SFPENCC` | SFPENCC | Clears (resets) the condition code register so all lanes are enabled again. |
| `TTI_SFPSTORE` | SFPSTORE | Stores LREG0 back to the current DEST row(s). Uses `InstrModLoadStore::DEFAULT` (float format). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register -- holds the current element loaded from DEST, modified by CC-guarded multiply, then stored back to DEST. |
| **LREG2** | Holds the slope constant (IEEE-754 float) for the duration of the entire tile computation. Loaded once before the iteration loop via two `SFPLOADI` instructions. |
| **LCONST_0** | Hardware constant register = 0.0f. Used as the addend in SFPMUL to make it a pure multiply (MAD with +0.0). |
| **CC register** | Condition code register. Set by SFPSETCC (enabled for negative lanes), guarding SFPMUL, then cleared by SFPENCC each iteration. |
| **DEST** | Source and destination for tile data. Each iteration reads and writes one sfpi row (32 elements = 2 physical rows x 16 elements). |

### Address Mode Configuration

The address mode used by SFPLOAD/SFPSTORE in `_calculate_lrelu_` differs by architecture:

**Wormhole B0**: Uses `ADDR_MOD_3`
- This address mode is NOT explicitly configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` -- the init function only configures `ADDR_MOD_7` (with `dest.incr=0`).
- `ADDR_MOD_3` retains whatever configuration was set by a prior kernel or defaults to `dest.incr=0`. The per-iteration DEST advancement is handled entirely by the `sfpi::dst_reg++` statement (which emits an internal pointer increment), not by ADDR_MOD auto-increment.

**Blackhole**: Uses `ADDR_MOD_7`
- Configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` with:
  - `srca.incr = 0`
  - `srcb.incr = 0`
  - `dest.incr = 0`
- As with Wormhole, per-iteration DEST advancement is via `sfpi::dst_reg++`, not ADDR_MOD.

Both architectures: `dest.incr = 0` means SFPLOAD/SFPSTORE do not auto-increment the DEST address. The `sfpi::dst_reg++` after each loop iteration advances by 1 sfpi row = 2 physical DEST rows = 32 elements. Between faces, `SETRWC` (Wormhole) or `inc_dst_addr<8>` x2 (Blackhole) advances to the next face.

## External Knowledge Sources
### DeepWiki Queries
No DeepWiki queries were needed for this analysis. The SFPU kernel implementation is straightforward raw-instruction code (SFPLOADI, SFPLOAD, SFPSETCC, SFPMUL, SFPENCC, SFPSTORE) with clear inline comments in the source. All information was derived directly from the source code.

### Confluence References
No Confluence page consultation was needed. The SFPU instructions used (SFPSETCC for sign-based CC, SFPMUL as CC-guarded multiply, SFPENCC to clear CC) are well-documented in the source code comments.

### Glean References
No Glean searches were performed for this analysis.
