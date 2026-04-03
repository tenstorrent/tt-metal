## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path)
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(0, <slope_as_uint32>u)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LEAKY_RELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized variant) | `get_op_init_and_func()` returns `leaky_relu_tile_init()` and `leaky_relu_tile(idst, <param0>u)` -- no template parameter in the tile call |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed through to `_calculate_lrelu_<false>(...)`, but the kernel does not contain any `if constexpr (APPROXIMATION_MODE)` branches -- the template parameter is unused | `_calculate_lrelu_` in `ckernel_sfpu_relu.h` has no conditional paths based on `APPROXIMATION_MODE` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the API header macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN` calls `_llk_math_eltwise_unary_sfpu_params_` directly) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (BH) |

### Call Chain
1. The compute kernel invokes `leaky_relu_tile(idst, slope)` (defined in `relu.h`).
2. `leaky_relu_tile` expands via the `MATH()` macro to `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope)`, which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, idst, (int)VectorMode::RC, slope)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls for SFPU availability, then loops over 4 faces (VectorMode::RC), calling `calculate_lrelu<false>(slope)` once per face.
4. `calculate_lrelu<false>(slope)` (in `ckernel_sfpu_relu.h` metal wrapper) delegates to `_calculate_lrelu_<false>(ITERATIONS=8, slope)`.
5. `_calculate_lrelu_<false>(8, slope)` (in `ckernel_sfpu_relu.h` tt_llk) executes the raw TTI instruction sequence that implements leaky ReLU.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the 32x32 tile are processed (face 0 through face 3).
- **Operation invocation**: The params function calls `calculate_lrelu<false>(slope)` once per face in a `for (int face = 0; face < 4; face++)` loop. Each invocation processes one face (8 iterations x 32 elements = 256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, the params function uses `TTI_SETRWC` to advance the DEST pointer by 16 physical rows (2 SETRWC of stride 8) between faces. On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which performs `math::inc_dst_addr<8>()` twice. Within each face, the kernel uses `sfpi::dst_reg++` to advance by 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration.

### Annotated SFPU Kernel Source

The kernel uses raw `TT_`/`TTI_` instructions with CC manipulation. Since there is only a single SFPSETCC/SFPENCC pair (a simple LT0 guard), this qualifies as Style A (simple CC usage in a short kernel).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h
// (Blackhole version is identical except ADDR_MOD_7 replaces ADDR_MOD_3)

template <bool APPROXIMATION_MODE>  // APPROXIMATION_MODE is unused -- no conditional paths
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    // Load the 32-bit FP32 slope into LREG2 using two 16-bit immediates
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);  // instr_mod0=10: write bits [15:0] of LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);       // instr_mod0=8: write bits [31:16] of LREG2
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)  // iterations=8 per face; processes 32 elements per iteration
    {
        // Load current element from DEST into LREG0 (FP32 default format)
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);

        // Set CC: CC.En=1, CC.Res=1 where LREG0 < 0 (sign bit test, mod1=0 = SFPSETCC_MOD1_LREG_LT0)
        // After this, only lanes with negative values are active
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

        // CC-guarded multiply: LREG0 = LREG0 * LREG2 + LCONST_0 = x * slope + 0.0
        // Only negative-value lanes execute this; positive lanes keep their original value
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Reset CC: mod1=0 = SFPENCC_MOD1_EU_R1 (Enable unchanged, Result=1)
        // All lanes become active again (En=1, Res=1)
        TTI_SFPENCC(0, 0, 0, 0);

        // Store LREG0 back to DEST (unconditional, all lanes active)
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);

        // Advance to next sfpi row (2 physical DEST rows = 32 elements)
        sfpi::dst_reg++;
    }
}
```

**Note on Blackhole vs Wormhole**: The only difference is `ADDR_MOD_7` (Blackhole) vs `ADDR_MOD_3` (Wormhole). On Wormhole, `set_addr_mod_base()` remaps the 0-3 range to physical slots 4-7, so `ADDR_MOD_3` resolves to physical slot 7 at runtime. Both architectures configure physical `ADDR_MOD_7` with `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}` (no auto-increment). Blackhole does not use address mode base remapping.

### SFPU Instructions Used

| Instruction | Opcode | Count per Iteration | Description |
|-------------|--------|---------------------|-------------|
| `SFPLOADI` | 0x71 | 2 (once before loop) | Loads a 16-bit immediate into an LREG. Used twice to construct the full 32-bit FP32 slope in LREG2 (low 16 bits then high 16 bits). |
| `SFPLOAD` | 0x70 | 1 | Loads a value from DEST into LREG0 using the DEFAULT format mode. |
| `SFPSETCC` | 0x7B | 1 | Sets per-lane condition code based on LREG0 sign (mod1=0: `LREG_LT0`). Enables CC masking -- lanes where the value is negative get CC.Res=1 (active), lanes where the value is non-negative get CC.Res=0 (masked). |
| `SFPMUL` | 0x86 | 1 | CC-guarded multiply: `LREG0 = LREG0 * LREG2 + LCONST_0 (0.0)`. Only executes on lanes where CC.En=1 and CC.Res=1 (negative values). This computes `slope * x` for negative elements. |
| `SFPENCC` | 0x8A | 1 | Resets condition code to make all lanes active. Mode EU_R1 (mod1=0): Enable unchanged, Result=1 -- since CC.En was 1 from SFPSETCC, this makes all lanes active (En=1, Res=1). |
| `SFPSTORE` | 0x72 | 1 | Stores LREG0 back to DEST. Executes unconditionally (all lanes active after SFPENCC). For positive values, LREG0 still holds the original loaded value (passthrough); for negative values, it holds `slope * x`. |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **LREG0** | Working register. Loaded with the input element from DEST, conditionally multiplied by slope, then stored back. |
| **LREG2** | Holds the slope parameter as a 32-bit FP32 value. Loaded once before the iteration loop via two `SFPLOADI` instructions (low 16 bits then high 16 bits). Persists across all iterations and faces. |
| **LCONST_0** (index 9) | Fixed Const 1 = 0.0. Used as the addend in `SFPMUL` to make it a pure multiply (`x * slope + 0.0`). |
| **DEST** | Source and destination for tile data. Each iteration loads one sfpi row (32 elements) and stores the result back. |

### Address Mode Configuration

The address mode used by this SFPU operation has zero auto-increment, since the kernel manages DEST address progression manually via `sfpi::dst_reg++` and SETRWC between faces.

**Wormhole B0:**
- `ADDR_MOD_7` is configured during init with: `{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}`
- The kernel references `ADDR_MOD_3`, but `set_addr_mod_base()` sets the base to 1, remapping slots 0-3 to physical slots 4-7. So `ADDR_MOD_3` -> physical `ADDR_MOD_7`.
- Source: `eltwise_unary_sfpu_configure_addrmod()` in `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` lines 28-33, and `set_addr_mod_base()` in `cmath_common.h` line 173-176.

**Blackhole:**
- `ADDR_MOD_7` is configured during init with: `{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}`
- The kernel directly references `ADDR_MOD_7` (no base remapping on Blackhole).
- Source: `eltwise_unary_sfpu_configure_addrmod()` in `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` lines 28-33.

Both architectures use the same effective configuration: zero auto-increment on all register files. This is the standard no-auto-increment mode for SFPU operations that manage address progression via `dst_reg++` and SETRWC.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPU condition code (CC) mechanism work for predicated execution in SFPU kernels?"
   **Reason**: Needed to understand the SFPSETCC/SFPENCC interaction for the leaky_relu CC guard pattern.
   **Key Findings**: DeepWiki was unavailable (repository not indexed). Relied on the SFPU Hardware Model Reference document (`.claude/references/sfpu-hardware-model.md`) which provides authoritative CC semantics: SFPSETCC mod1=0 tests LREG sign bit (LT0), sets CC.En=1; SFPENCC mod1=0 (EU_R1) sets CC.Res=1 for all lanes without changing CC.En, effectively re-enabling all lanes.

### Confluence References
No Confluence pages were consulted for this analysis. The SFPU Hardware Model Reference document contained sufficient instruction semantics for all instructions used (SFPLOADI, SFPLOAD, SFPSETCC, SFPMUL, SFPENCC, SFPSTORE).

### Glean References
No Glean queries were made for this analysis.
