## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(0, <slope_as_u32>)` where `<slope_as_u32>` is `std::bit_cast<uint32_t>(param0)` -- the slope float reinterpreted as a raw uint32

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LEAKY_RELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (not parameterized for approx) | `get_op_init_and_func_parameterized()` -- LEAKY_RELU case returns `leaky_relu_tile_init()` / `leaky_relu_tile(idst, slope_u32)` with no approximation template parameter |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed to `_calculate_lrelu_<false>(...)` | The `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope)` macro resolves `APPROX` to the value from `math_approx_mode`, which is `false`. The `_calculate_lrelu_` function does not branch on `APPROXIMATION_MODE` -- the template parameter is accepted but unused in the function body. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (same as LLK Dispatch -- the `_llk_math_eltwise_unary_sfpu_params_` function handles both face iteration and SFPU function invocation) |

### Call Chain
1. The compute kernel invokes `leaky_relu_tile(idst, slope)` (defined in `relu.h`).
2. `leaky_relu_tile` expands via `MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope))` which calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, idst, (int)VectorMode::RC, slope)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is ready, then iterates over 4 faces in `VectorMode::RC`, calling `calculate_lrelu<false>(slope)` for each face and advancing the DEST address between faces via `SETRWC`.
4. `calculate_lrelu<false>(slope)` (in the metal-layer `ckernel_sfpu_relu.h`) delegates to `_calculate_lrelu_<false>(ITERATIONS=8, slope)` in the tt_llk layer.
5. `_calculate_lrelu_` (in `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`) executes the raw SFPU instruction sequence: load slope into LREG2 via `TT_SFPLOADI`, then for each of 8 iterations: load from DEST, conditionally multiply by slope if negative, store back.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (Face 0 through Face 3), covering all 1024 elements.
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function calls `calculate_lrelu<APPROX>(slope)` once per face (4 times total). Each call internally runs 8 iterations, processing one face (256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `ADDR_MOD_3` is used within the SFPU kernel (for `SFPLOAD`/`SFPSTORE`), and the face stride is advanced by two `TTI_SETRWC(... CR_D, 8 ...)` calls between faces. On Blackhole, the kernel uses `ADDR_MOD_7` instead of `ADDR_MOD_3`, and face advancement uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with CC manipulation via `SFPSETCC`/`SFPENCC`. The CC flow is straightforward (single set/clear pair per iteration), so Style A (inline-commented) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope) // APPROXIMATION_MODE=false, iterations=8
{
    // Load slope (FP32 as raw uint32) into LREG2 in two halves:
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);  // InstrMod=10 (LO16_ONLY): write low 16 bits of slope into LREG2[15:0], preserve LREG2[31:16]
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);       // InstrMod=8 (HI16_ONLY): write high 16 bits of slope into LREG2[31:16], preserve LREG2[15:0]
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // Load current DEST element into LREG0 (FP32 implied format)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // InstrMod=0: set CC.Res = sign bit of LREG0 (CC.Res=1 if LREG0 is negative)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // CC-guarded: LREG0 = (LREG0 * LREG2) + LCONST_0; i.e. x * slope + 0.0 (only for negative lanes)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // InstrMod=0: CC.Res=1 (all lanes), InstrMod[1:0]=0 (keep CC.En); effectively re-enables all lanes
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // Store LREG0 back to DEST (FP32 implied format)
        sfpi::dst_reg++;                                                                // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

**Note on Blackhole variant**: The Blackhole implementation (`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h`) is identical except it uses `ADDR_MOD_7` instead of `ADDR_MOD_3` for `SFPLOAD`/`SFPSTORE` instructions. The algorithmic behavior is the same.

**Key CC observations:**
- `SFPSETCC` (InstrMod=0) tests the sign bit of LREG0. For negative inputs, CC.Res is set to 1, enabling the subsequent `SFPMUL` to execute. For non-negative inputs, CC.Res=0 and `SFPMUL` is skipped (the original value is preserved in LREG0).
- `SFPMUL` is CC-guarded: it only executes on lanes where CC.Res=1 (negative inputs). The multiplication computes `x * slope`, implementing the leaky ReLU formula for the negative region.
- `SFPENCC` (all zeros) resets CC.Res to 1 for all lanes, restoring uniform execution for the `SFPSTORE` that follows.
- The CC set/clear is contained within a single iteration -- there is no CC state carried across iterations.

### SFPU Instructions Used

| Instruction | Opcode | Description |
|-------------|--------|-------------|
| `SFPLOADI` | 0x71 | Loads a 16-bit immediate value into a specified LREG. Used twice to construct the full 32-bit slope value in LREG2 (first low 16 bits with InstrMod=10/LO16_ONLY, then high 16 bits with InstrMod=8/HI16_ONLY). |
| `SFPLOAD` | 0x70 | Loads a value from the DEST register file into an LREG. Here loads the current tile element from DEST into LREG0 using the implied FP32 format (InstrMod=DEFAULT). |
| `SFPSETCC` | 0x7B | Sets the CC.Res condition code based on a register value. With InstrMod=0, sets CC.Res to the sign bit of LREG0 (CC.Res=1 if negative). This predicts subsequent instructions. |
| `SFPMUL` (alias of `SFPMAD`) | 0x86 (alias of 0x84) | Performs fused multiply-add: `(LREG0 * LREG2) + LCONST_0` = `x * slope + 0.0`. CC-guarded -- only executes on lanes where CC.Res=1 (negative values). LCONST_0 is the hardware constant 0.0. |
| `SFPENCC` | 0x8A | Directly sets CC state. With all-zero arguments: sets CC.Res=1 (InstrMod[3]=0 means "set CC.Res to 1"), keeps CC.En unchanged. Effectively re-enables all lanes after the conditional multiply. |
| `SFPSTORE` | -- | Stores the value from an LREG back to the DEST register file. Writes the result (either original value for non-negative, or `x*slope` for negative) back to DEST. |

### SFPU Register Usage

| Register | Role | Details |
|----------|------|---------|
| **LREG0** | Working register | Holds the current element loaded from DEST. After conditional multiply, holds the result (original or `x*slope`). Written back to DEST. |
| **LREG2** | Slope constant | Loaded once before the iteration loop with the full 32-bit float representation of the slope parameter. Persists across all 8 iterations within a face call. |
| **LCONST_0** | Hardware constant 0.0 | Used as the addend in `SFPMUL` (which is an alias for `SFPMAD`), making the operation a pure multiply: `(x * slope) + 0.0`. Register index 9. |
| **DEST** | Source/destination tile data | Each iteration reads one sfpi row (32 elements) via `SFPLOAD` and writes it back via `SFPSTORE`. The DEST pointer auto-advances via `dst_reg++`. |

### Address Mode Configuration

The `_calculate_lrelu_` function uses `ADDR_MOD_3` (Wormhole) or `ADDR_MOD_7` (Blackhole) for its `SFPLOAD` and `SFPSTORE` instructions.

The address mode is configured during the `_llk_math_eltwise_unary_sfpu_init_<SfpuType::lrelu>()` call, which invokes `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()`. For `SfpuType::lrelu`, none of the special-case `if constexpr` branches match, so only `ADDR_MOD_7` is configured:

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

This sets all increments to 0 for ADDR_MOD_7. The DEST address progression within the kernel is handled explicitly by `sfpi::dst_reg++` (which advances the SFPI DEST pointer by 1 sfpi row = 2 physical rows), not by the address mode auto-increment. The address mode specifier in `SFPLOAD`/`SFPSTORE` (`ADDR_MOD_3` on Wormhole, `ADDR_MOD_7` on Blackhole) selects a mode with zero auto-increment, ensuring the load/store target the exact address set by the software-managed pointer.

**Wormhole vs Blackhole difference**: The Wormhole metal-layer wrapper (`hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h`) calls into the same `_calculate_lrelu_` from the tt_llk layer which uses `ADDR_MOD_3`. The Blackhole tt_llk implementation uses `ADDR_MOD_7`. Both modes are configured identically with zero increments by `eltwise_unary_sfpu_configure_addrmod`.

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "What are the SFPU instructions SFPSETCC, SFPMUL, SFPENCC, SFPLOAD, SFPSTORE, and SFPLOADI? How does SFPSETCC set the condition code register? What does SFPENCC do to enable/clear the condition code?"
   **Reason**: Needed to understand the semantics of each raw SFPU instruction used in the leaky_relu kernel.
   **Key Findings**: SFPSETCC sets CC.Res based on register value with configurable comparison mode (InstrMod selects sign test, zero test, etc.). SFPENCC with all-zero args clears CC state making all lanes active. SFPMUL is an alias for SFPMAD with addend expected to be 0. TTI_ prefix is for direct instruction emission, TT_ for wrapped calls.

### Confluence References
1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**: SFPSETCC, SFPMAD, SFPMUL, SFPENCC, SFPLOADI, SFPLOAD
   **Key Findings**:
   - SFPSETCC (0x7B): InstrMod=0 sets CC.Res to RG[VC].Sgn (sign bit). Sets CC.Res only, not CC.En.
   - SFPMAD (0x84): FMA operation `(A * B) + C`, CC-guarded via LaneEnabled. SFPMUL (0x86) is its alias declaring C=0.
   - SFPENCC (0x8A): InstrMod[3]=0 sets CC.Res=1; InstrMod[1:0]=0 keeps CC.En unchanged. Executes on all lanes regardless of LaneEnabled.
   - SFPLOADI (0x71): InstrMod=0xA (LO16_ONLY) writes low 16 bits preserving high; InstrMod=0x8 (HI16_ONLY) writes high 16 bits preserving low. Used to construct a full 32-bit immediate in two steps.
   - SFPLOAD (0x70): Loads from DEST register file to LREG; InstrMod=DEFAULT uses implied format.

### Glean References
[No Glean queries were necessary for this analysis.]
