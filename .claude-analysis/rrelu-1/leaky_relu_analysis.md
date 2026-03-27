## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(0, {slope_as_hex}u)` where `{slope_as_hex}` is `std::bit_cast<uint32_t>(param0)` -- the slope float reinterpreted as a uint32

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LEAKY_RELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` returns `leaky_relu_tile_init()` / `leaky_relu_tile(0, {slope_hex}u)` -- no template approximation parameter |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but `_calculate_lrelu_` does not branch on `APPROXIMATION_MODE` -- the same code path executes regardless | The `_calculate_lrelu_` function is not templated on `APPROXIMATION_MODE` in the tt_llk implementations; the template parameter is only on the `calculate_lrelu` wrapper in the metal ckernel layer |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header calls directly into the macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN` (defined in `llk_math_eltwise_unary_sfpu_macros.h`) which invokes `_llk_math_eltwise_unary_sfpu_params_` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (BH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain

1. **`leaky_relu_tile(idst, slope)`** (API header, `relu.h` line 112) expands via the `MATH(...)` wrapper to call `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope)`.

2. **`SFPU_UNARY_ONE_PARAM_KERNEL_FN`** (macro in `llk_math_eltwise_unary_sfpu_macros.h` line 130) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, idst, (int)VectorMode::RC, slope)`.

3. **`_llk_math_eltwise_unary_sfpu_params_<false>`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets up the DEST write address, stalls until SFPU is ready, then iterates over 4 faces (VectorMode::RC), calling `calculate_lrelu<false>(slope)` once per face with `SETRWC` between faces.

4. **`calculate_lrelu<false>(slope)`** (in `ckernel_sfpu_relu.h` in the metal ckernel layer) delegates to `_calculate_lrelu_<false>(ITERATIONS=8, slope)`.

5. **`_calculate_lrelu_<false>(8, slope)`** (in `tt_llk/.../sfpu/ckernel_sfpu_relu.h`) is the core SFPU implementation that loads the slope into LREG2, then loops 8 iterations performing the conditional multiply.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (face 0, 1, 2, 3).
- **Operation invocation**: The core SFPU function `calculate_lrelu<APPROX>(slope)` is called once per face inside a `for (int face = 0; face < 4; face++)` loop. Each call processes one face (8 SFPU iterations x 32 elements = 256 elements).
- **DEST address progression**: On Wormhole, the SFPLOAD/SFPSTORE instructions reference `ADDR_MOD_3`; on Blackhole, they reference `ADDR_MOD_7`. Both are configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` to `{.dest = {.incr = 0}}` (zero auto-increment). The DEST pointer advances via `sfpi::dst_reg++` at the end of each iteration (1 sfpi row = 2 physical DEST rows = 32 elements). Between faces, `SETRWC` (Wormhole: `TTI_SETRWC(..., 8, ..., p_setrwc::SET_D)` x2 = advance by 16 physical rows) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole: `math::inc_dst_addr<8>()` x2) advances the DEST write pointer to the next face.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with CC manipulation via SFPSETCC and SFPENCC. The CC flow is straightforward (one SFPSETCC/SFPENCC pair per iteration), so Style A (inline-commented source) is used.

#### Wormhole B0 Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope) // APPROXIMATION_MODE not used in body
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);   // Load lower 16 bits of slope into LREG2[15:0] (insmod=0xA: LO16_ONLY, preserves upper bits)
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);        // Load upper 16 bits of slope into LREG2[31:16] (insmod=0x8: HI16_ONLY, preserves lower bits)
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // Load current DEST row into LREG0 (float format)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // Set CC.Res = LREG0.Sgn (1 if negative, 0 if positive); mod1=0 tests sign bit
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // CC-guarded: LREG0 = LREG0 * LREG2 (x * slope) only for negative lanes
        TTI_SFPENCC(0, 0, 0, 0);                                                      // Reset CC: CC.Res=1, CC.En unchanged (mod1=0x0: keep CC.En, set CC.Res=1)
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // Store LREG0 back to DEST row
        sfpi::dst_reg++;                                                                // Advance DEST pointer by 1 sfpi row (= 2 physical rows)
    }
}
```

#### Blackhole Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope) // APPROXIMATION_MODE not used in body
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);   // Load lower 16 bits of slope into LREG2[15:0] (insmod=0xA: LO16_ONLY)
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);        // Load upper 16 bits of slope into LREG2[31:16] (insmod=0x8: HI16_ONLY)
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);        // Load current DEST row into LREG0 (float format)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // Set CC.Res = LREG0.Sgn (1 if negative, 0 if positive); mod1=0 tests sign bit
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // CC-guarded: LREG0 = LREG0 * LREG2 (x * slope) only for negative lanes
        TTI_SFPENCC(0, 0, 0, 0);                                                      // Reset CC: CC.Res=1, CC.En unchanged
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);       // Store LREG0 back to DEST row
        sfpi::dst_reg++;                                                                // Advance DEST pointer by 1 sfpi row (= 2 physical rows)
    }
}
```

**Key CC observations:**
- SFPSETCC with `mod1=0` tests the sign bit of the value in the source register. For each SFPU lane, `CC.Res` is set to 1 if the value is negative, 0 if positive or zero.
- SFPMUL is CC-guarded: it only executes on lanes where `CC.Res=1` (i.e., where the input was negative). For positive lanes, LREG0 retains its original value.
- SFPENCC with `(0, 0, 0, 0)` maps to `InstrMod=0`: `CC.Res` is unconditionally set to 1 (all lanes enabled for subsequent non-predicated execution), and `CC.En` is kept at its previous value (mod1[1:0]=0 means "keep previous CC Enable").
- The net effect is: negative values get multiplied by the slope, positive values pass through unchanged -- exactly the leaky ReLU definition: `f(x) = x if x >= 0, slope * x if x < 0`.

### SFPU Instructions Used

| Instruction | Opcode | Description |
|-------------|--------|-------------|
| **SFPLOADI** | 0x71 | Loads a 16-bit immediate value into an LREG. Used twice with `insmod=0xA` (LO16_ONLY) and `insmod=0x8` (HI16_ONLY) to construct the full 32-bit FP32 slope value in LREG2. Uses `TT_SFPLOADI` (non-immediate variant that takes runtime arguments). |
| **SFPLOAD** | 0x70 | Loads data from a DEST register row into an LREG. Used with `InstrModLoadStore::DEFAULT` (FP32 format), zero-increment address mode, and DEST offset 0. The actual DEST row is determined by the current DEST write pointer. |
| **SFPSETCC** | 0x7B | Sets the CC Result register based on a comparison of the source LREG. With `InstrMod=0`, sets `CC.Res = RG[VC].Sgn` (the sign bit), effectively enabling predicated execution only for lanes with negative values. IPC=1, Latency=1. |
| **SFPMUL** | 0x86 | Floating-point multiply: `RG[VD] = RG[VA] * RG[VB]`. This instruction is an alias of SFPMAD with `RG[VC]` expected to be 0.0 (LCONST_0). CC-guarded: only executes on lanes where `CC.Res=1` (negative values). IPC=1, Latency=2. |
| **SFPENCC** | 0x8A | Directly sets CC.En and CC.Res. With `InstrMod=0x0`: `CC.Res` is set to 1 (all lanes result-enabled), `CC.En` is kept at its previous value. Executes on all lanes regardless of current LaneEnabled state. IPC=1, Latency=1. |
| **SFPSTORE** | 0x78 | Stores data from an LREG back to a DEST register row. Used with `InstrModLoadStore::DEFAULT` (FP32 format) and zero-increment address mode. |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **LREG0** (`p_sfpu::LREG0 = 0`) | Working register: loaded with the input value from DEST, conditionally multiplied by the slope, then stored back to DEST. |
| **LREG2** (`p_sfpu::LREG2 = 2`) | Holds the slope parameter (FP32). Loaded once before the iteration loop via two SFPLOADI instructions (low 16 bits then high 16 bits). Remains constant across all iterations and faces. |
| **LCONST_0** (`p_sfpu::LCONST_0 = 9`) | Hardware constant register holding 0.0f. Passed as the addend (src_c) to SFPMUL, which is an alias of SFPMAD that expects src_c=0. |
| **CC.Res** | Condition code result register. Set per-lane by SFPSETCC to indicate negative values. Cleared by SFPENCC after each conditional multiply. |

### Address Mode Configuration

The address mode used for SFPLOAD/SFPSTORE in `_calculate_lrelu_` differs between hardware generations, but both are configured to zero-increment:

**Wormhole B0**: Uses `ADDR_MOD_3` in SFPLOAD/SFPSTORE. The init function `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` configures `ADDR_MOD_7` (not ADDR_MOD_3) to `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}`. ADDR_MOD_3 is not explicitly configured by the lrelu init, so it retains whatever default the infrastructure sets. The actual DEST address progression is handled by `sfpi::dst_reg++` at the end of each iteration, not by the address mode auto-increment. Between faces, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice (advancing by 2x8=16 physical rows = one face stride).

**Blackhole**: Uses `ADDR_MOD_7` in SFPLOAD/SFPSTORE, which is explicitly configured to `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}` by the init function. DEST progression is identical: `sfpi::dst_reg++` per iteration, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (which calls `math::inc_dst_addr<8>()` twice) between faces.

Both configurations result in Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC/inc_dst_addr between faces).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do SFPSETCC and SFPENCC instructions work in the SFPU? Specifically, what does SFPSETCC do with regard to condition code (CC) register, what condition does mod1=0 on SFPSETCC test (negative values?), and how does SFPENCC clear/reset the CC state?"
   **Reason**: Needed to understand the CC predicated execution mechanism used in the leaky relu kernel.
   **Key Findings**: SFPSETCC sets CC.Res per-lane based on a comparison; SFPENCC clears/resets CC state and re-enables all lanes. The specific mod1 values for SFPSETCC were not fully detailed by DeepWiki -- required Confluence ISA page for precise semantics.

2. **Query**: "What is SFPMUL instruction in the SFPU? How does it differ from SFPMAD? Does SFPMUL exist as a separate hardware instruction, or is it always encoded as SFPMAD with the addend being LCONST_0?"
   **Reason**: The lrelu kernel uses TTI_SFPMUL which has a separate opcode from TTI_SFPMAD. Needed to confirm whether it is a true distinct instruction.
   **Key Findings**: DeepWiki could not definitively answer. Source code analysis confirmed SFPMUL (opcode 0x86) and SFPMAD (opcode 0x84) are separate opcodes. Confluence ISA page confirmed SFPMUL is "an alias of SFPMAD, provided to declare intent that RG[VC] should be 0.0. It still performs the full FMA operation."

### Confluence References

1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**:
   - **SFPSETCC** (at position ~94875): Confirmed `InstrMod=0` sets `CC.Res = RG[VC].Sgn` (sign bit test). Opcode 0x7B, IPC=1, Latency=1. Sets CC Result but not CC Enable.
   - **SFPENCC** (at position ~87224): Confirmed with `InstrMod=0x0` it sets `CC.Res=1` unconditionally and keeps `CC.En` at its previous value. Opcode 0x8A. Executes on all lanes regardless of current LaneEnabled state.
   - **SFPMUL** (at position ~146294): Confirmed it is "an alias of SFPMAD, provided to declare intent that RG[VC] should be 0.0. It still performs the full FMA operation." Opcode 0x86, IPC=1, Latency=2.
   - **SFPLOADI** (at position ~79003): Confirmed `InstrMod=0x8` is HI16_ONLY (writes upper 16 bits, preserves lower) and `InstrMod=0xA` is LO16_ONLY (writes lower 16 bits, preserves upper). Used to construct full 32-bit slope value.
   - **Predicated Execution overview**: Confirmed the CC.En/CC.Res interaction model: a lane is active when `CC.En=0` (predication disabled) or when `CC.En=1 AND CC.Res=1`.

### Glean References

No Glean queries were needed for this analysis. The Confluence ISA page and source code provided sufficient detail.
