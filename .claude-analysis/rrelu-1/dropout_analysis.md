## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `DROPOUT` (listed in `UnaryOpType` enum but NOT dispatched via `UnaryProgramFactory`)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- dropout uses a dedicated compute kernel that calls `dropout_tile(0, int_probability, int_scale_factor)` directly, not via the `SFPU_OP_CHAIN_0` macro mechanism.

**Note on dispatch path**: Dropout is an experimental operation with its own program factory at `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`. Although `DROPOUT` appears in the `UnaryOpType` enum, it is NOT handled by `unary_op_utils.cpp`'s `get_op_init_and_func()` or `get_block_defines()`. The compute kernel directly includes the dropout API header and calls `dropout_tile()` / `dropout_kernel_init()`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are documented below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | Hardcoded at `dropout_program_factory.cpp:240` as `bool math_approx_mode = false;` |
| Template parameter (SFPU kernel) | `APPROXIMATION_MODE=false` (resolved from `APPROX`) | `APPROX` is a `constexpr bool` generated at JIT build time from `math_approx_mode` (`tt_metal/jit_build/genfiles.cpp:394`). Since `math_approx_mode=false`, `APPROX=false`. |
| Effective SFPU path | `APPROXIMATION_MODE` is not used by the dropout kernel -- there are no `if constexpr (APPROXIMATION_MODE)` branches in `_calculate_dropout_`. The template parameter is accepted but ignored. | `ckernel_sfpu_dropout.h` -- no conditional branches on `APPROXIMATION_MODE` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header directly invokes the `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` and `SFPU_ONE_PARAM_KERNEL_INIT` macros (defined in `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`) which expand to calls to `_llk_math_eltwise_unary_sfpu_params_` and `llk_math_eltwise_unary_sfpu_init` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h` (BH) -- implementations are **identical** |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain
1. **Compute kernel** (`dropout_kernel.cpp`) calls `dropout_tile(0, int_probability, int_scale_factor)`.
2. **API header** (`dropout.h`) wraps this via the `MATH()` macro into `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, idst, probability, scale_factor)`.
3. **Macro** (`llk_math_eltwise_unary_sfpu_macros.h`) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_dropout<APPROX>, idst, (int)VectorMode::RC, probability, scale_factor)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU, then iterates over 4 faces (RC mode), calling `calculate_dropout<false>(probability, scale_factor)` per face with `SETRWC`/`_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` between faces.
5. **Wrapper** (`ckernel_sfpu_dropout.h` in `hw/ckernels/`) calls `_calculate_dropout_<false, 8>(8, probability, scale)`.
6. **Core SFPU function** (`ckernel_sfpu_dropout.h` in `tt_llk/`) executes the raw TTI instruction sequence.

For initialization, a similar chain exists: `dropout_kernel_init(seed)` -> `SFPU_ONE_PARAM_KERNEL_INIT(dropout, sfpu::dropout_init, APPROX, seed)` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, false>(dropout_init<false>, seed)` -> `_init_dropout_(seed)` -> `init_prng_seed(seed)`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (4 faces x 8 iterations = 32 iterations = 1024 elements).
- **Operation invocation**: The params dispatch calls `calculate_dropout<false>(probability, scale)` once per face. Each invocation internally loops for `ITERATIONS=8` sfpi rows (via the `for (int d = 0; d < iterations; d++)` loop). Between faces, the dispatch advances DEST addressing.
- **DEST address progression**: Standard DEST progression. On **Wormhole**, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 physical rows = 1 face). On **Blackhole**, it uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice (same net effect). Within a face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration.
- **Address mode**: `ADDR_MOD_7` is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` for all standard unary SFPU ops including dropout (no special case for dropout in `eltwise_unary_sfpu_configure_addrmod`). The kernel uses `ADDR_MOD_3` (value 3) in its SFPLOAD/SFPSTORE instructions -- this is the address mode argument passed directly to the SFPLOAD/SFPSTORE instructions which controls auto-increment behavior. Note: the kernel manually increments DEST addressing via `dst_reg++` rather than relying on ADDR_MOD auto-increment.

### Annotated SFPU Kernel Source

The dropout kernel uses raw `TT_`/`TTI_` instructions with CC manipulation via SFPIADD and SFPENCC. The CC flow involves an SFPIADD that updates CC.Res to gate a subsequent SFPMOV, followed by SFPENCC to reset CC state. This pattern is moderately complex -- the SFPIADD uses a non-obvious InstrMod encoding (value 10 = `SFPIADD_MOD1_CC_GTE0 | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`). A CC State Machine diagram is provided after the source code (Style B).

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h
// (Blackhole implementation is identical)

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, std::uint32_t probability, std::uint32_t scale)
{
    // SFPU microcode

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        ////////////////////////
        // Scale samples
        // dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);
        ///////////////////////
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        ////////////////////////
        // Instruction SFPMOV generates a uint32_t pseudorandom number
        // when instr_mod1 = 8 and lreg_c =  9.
        // Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        // Unset sign-bit for easy comparison with probability
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   dst_reg[0] = vConst0;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(0, 0, 3, 0);

        sfpi::dst_reg++;
    }
}

inline void _init_dropout_(const std::uint32_t seed)
{
    init_prng_seed(seed);
}
```

#### CC State Machine -- `_calculate_dropout_`

The kernel uses SFPIADD to set CC.Res for conditional zeroing, followed by SFPENCC to reset CC state. The SFPIADD with InstrMod=10 (`SFPIADD_MOD1_CC_GTE0 | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`) performs integer subtraction and sets CC.Res with inverted sense. Per-lane CC masking relies on SFPIADD's CC.Res update to gate the subsequent SFPMOV instruction.

**Important note on CC.En**: The ISA specification (Tensix SFPU ISA, Confluence page 1170505767) states that SFPIADD "Sets CC Enable? N" and only SFPENCC can directly set CC.En. The ISA's LaneEnabled formula is `(~CC.En | CC.Res) & ~RowDisable`. Under a strict reading where CC.En=0 (the reset default), SFPIADD would only update CC.Res while all lanes remain unconditionally active. However, this dropout kernel (and other production TTI kernels like leaky relu) demonstrably use SFPIADD/SFPSETCC CC.Res updates to gate subsequent instructions without a prior SFPENCC enable. This indicates that either (a) SFPIADD implicitly sets CC.En=1 when it updates CC.Res (an undocumented side effect not reflected in the ISA table's "Sets CC Enable?" column), or (b) the actual hardware CC model differs from the ISA pseudocode for these specific instructions. The diagram below documents the **intended** CC behavior based on the code comments and operational semantics.

```
_calculate_dropout_ — CC State Transitions (per iteration)
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED (CC.En effectively 0, all lanes active)
       |
       |  SFPLOAD  LREG0 <- DEST[current_row]    (no CC effect) -- load tile value
       |  SFPMUL   LREG0 = LREG0 * LREG1 + 0.0  (no CC effect) -- value * scale_factor
       |
       |  SFPMOV   LREG3 <- RS[9] (mod1=8)       (no CC effect) -- read PRNG, advance PRNG
       |  SFPSETSGN LREG3.sign = 0 (mod1=1)      (no CC effect) -- clear sign bit for unsigned comparison
       |
       v
  +---------------------------------------------+
  | SFPIADD  mod1=10 (CC_GTE0 | ARG_2SCOMP)    |
  |   operation: LREG3 = LREG2 - LREG3          |
  |            = probability - rand              |
  |                                              |
  | CC.Res <- !((probability - rand) < 0)        |
  |         = (probability >= rand)              |
  |         = (rand <= probability)              |
  |   i.e., CC.Res=1 on lanes to be DROPPED     |
  +---------------------+-----------------------+
                        |
                        v
  CC State: ENABLED where rand <= probability (lanes to drop)
       |
       |  SFPMOV  LREG0 = LCONST_0 (= 0.0)   (CC-guarded: zeros LREG0 only on drop lanes)
       |
       v
  +---------------------------------------------+
  | SFPENCC  mod1=0, imm12=0                    |
  |   InstrMod[3]=0: CC.Res = 1                 |
  |   InstrMod[1:0]=0: CC.En unchanged          |
  |                                              |
  | CC <- ALL_ENABLED (all lanes pass)           |
  +---------------------+-----------------------+
                        |
                        v
  CC State: ALL_ENABLED (all lanes active again)
       |
       |  SFPSTORE  LREG0 -> DEST[current_row]   (unconditional) -- store result
       |                      (= scaled value if kept, 0.0 if dropped)
       |
       |  dst_reg++                                -- advance to next sfpi row
       |
       v  (loop continues for next iteration)
```

**Key CC observations:**
- SFPIADD with InstrMod=10 sets CC.Res = (probability >= rand), marking lanes where the element should be dropped (zeroed).
- The subsequent SFPMOV writes 0.0 to LREG0 only on lanes where CC.Res=1 (drop lanes). On other lanes, LREG0 retains the scaled value from SFPMUL.
- SFPENCC(0,0,0,0) resets CC.Res=1 for all lanes, restoring unconditional execution so SFPSTORE writes to all DEST rows.
- The comparison is done as unsigned integers: the sign bit of the PRNG output is cleared (SFPSETSGN with Imm12[0]=0), and probability is passed as a non-negative integer (0 to INT_MAX). The SFPIADD subtraction `probability - rand` is a signed 32-bit operation; if the result is negative, `rand > probability` and the lane should NOT be dropped.
- CC state is fully reset between iterations by SFPENCC, so there is no CC state leakage across loop iterations.

### SFPU Instructions Used

| Instruction | Opcode | Count per Iteration | Description |
|-------------|--------|--------------------:|-------------|
| `SFPLOADI` (`TT_SFPLOADI`) | 0x71 | 0 (4 total, before loop) | Load 16-bit immediate to LREG. Used to construct 32-bit scale factor (LREG1) and probability (LREG2) from two 16-bit halves each. `InstrMod=10` loads low 16 bits; `InstrMod=8` loads high 16 bits. |
| `SFPLOAD` (`TTI_SFPLOAD`) | 0x70 | 1 | Load value from DEST register into LREG0. `InstrMod=0` (DEFAULT format), `ADDR_MOD=3`. |
| `SFPMUL` (`TTI_SFPMUL`) | 0x86 | 1 | Multiply LREG0 by LREG1 (scale factor), add LCONST_0 (0.0). Result: `LREG0 = value * scale`. `InstrMod=0`. |
| `SFPMOV` (`TTI_SFPMOV`) | 0x7C | 2 | (1) With `InstrMod=8, VC=9`: reads from RS[9] (PRNG Counter) into LREG3, advancing the PRNG by one step. (2) With `InstrMod=0, VC=LCONST_0`: CC-guarded copy of 0.0 into LREG0 (zeros dropped elements). |
| `SFPSETSGN` (`TTI_SFPSETSGN`) | 0x89 | 1 | Set sign bit of LREG3 to `Imm12[0]=0` (clear sign bit). `InstrMod=1` selects immediate as sign source. Makes the PRNG value non-negative for unsigned comparison. |
| `SFPIADD` (`TTI_SFPIADD`) | 0x79 | 1 | Integer subtraction: `LREG3 = LREG2 - LREG3` (probability - rand). `InstrMod=10` = `CC_GTE0 | ARG_2SCOMP_LREG_DST`: updates CC.Res with inverted sense, so CC.Res=1 when result >= 0 (rand <= probability = drop this lane). |
| `SFPENCC` (`TTI_SFPENCC`) | 0x8A | 1 | Reset CC state. `InstrMod=0`: CC.Res=1, CC.En unchanged. Effectively makes all lanes active again. |
| `SFPSTORE` (`TTI_SFPSTORE`) | 0x72 | 1 | Store LREG0 back to DEST register. `InstrMod=0` (DEFAULT format), `ADDR_MOD=3`. Runs unconditionally after SFPENCC resets CC. |
| `SFPNOP` (`TTI_SFPNOP`) | 0x8E | 600 (in init only) | No operation. Used in `init_prng_seed()` as a delay loop (600 cycles) after writing the PRNG seed to allow the PRNG to stabilize. |

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| **LREG0** | Holds the tile element value loaded from DEST. Multiplied by scale factor, then conditionally zeroed (on drop lanes). Finally stored back to DEST. | Per-iteration: loaded, modified, stored |
| **LREG1** | Holds the 32-bit scale factor (FP32 bit pattern). Loaded once before the loop via two `SFPLOADI` calls (low 16 + high 16 bits). | Entire function lifetime (read-only in loop) |
| **LREG2** | Holds the 32-bit probability threshold (integer, 0 to INT_MAX). Loaded once before the loop via two `SFPLOADI` calls. | Entire function lifetime (read-only in loop) |
| **LREG3** | Holds the PRNG random number. Generated each iteration by `SFPMOV(mod1=8, VC=9)`, then sign-cleared by `SFPSETSGN`. Used in `SFPIADD` subtraction with probability, then overwritten (result stored back to LREG3 by SFPIADD). | Per-iteration: generated, compared, discarded |
| **LCONST_0** (RG[9]) | Fixed Constant 1 = `0x00000000` (0.0 in FP32, 0 in INT32). Used as the zero value for conditional zeroing in SFPMOV and as the addend in SFPMUL (multiply-add with 0.0 addend). | Hardware constant (read-only) |
| **RS[9]** (PRNG Counter) | Special Register: the per-lane PRNG. Reading from this register via `SFPMOV(mod1=8, VC=9)` returns the current 32-bit random value and advances the PRNG to the next state. | Hardware state (read with side-effect) |
| **DEST** | Source/destination register file. Tile elements are loaded from DEST, processed, and stored back. Addressing progresses via `dst_reg++` (1 sfpi row = 2 physical rows = 32 elements per iteration). | Managed by params dispatch and kernel loop |

### Address Mode Configuration

The `eltwise_unary_sfpu_configure_addrmod` function (called during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()`) configures the following address mode:

**ADDR_MOD_7** (same on both Wormhole and Blackhole):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
.set(ADDR_MOD_7);
```

This is the default address mode for all standard unary SFPU operations. Dropout does not match any of the special-case `if constexpr` branches in `eltwise_unary_sfpu_configure_addrmod` (which only apply to `topk_local_sort`, `typecast`, `unary_max/min`, `signbit`, and on Blackhole additionally `reciprocal`).

The kernel's SFPLOAD and SFPSTORE instructions use `ADDR_MOD_3` as their address mode argument (the 3rd parameter in the TTI calls). This value (`3`) is passed directly to the instruction encoding. DEST addressing within the kernel is managed explicitly by `sfpi::dst_reg++` at the end of each loop iteration, which advances the SFPU DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements). Between faces, the params dispatch layer handles the face-stride advance via `SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` (Blackhole).

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "What does the SFPMOV instruction do when instr_mod1=8 and lreg_c=9? Is this the PRNG mode?"
   **Reason**: Needed to understand the PRNG generation mechanism used in the dropout kernel
   **Key Findings**: DeepWiki was not available for the tenstorrent/tt-metal repository. Information was obtained from the Confluence SFPU ISA page instead.

### Confluence References
1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: `1170505767`, Cloud ID: `b9d94484-5dbd-4ae2-b670-6f414aefb4cd`)
   - **SFPMOV section**: Confirmed that InstrMod=8 reads from RS (Special Register) view. When VC=9 (PRNG Counter), the instruction reads the current PRNG value and advances the PRNG by one step. Algorithmic implementation: `if (InstrMod == 4'h8) { if (VC == 9) { AdvancePrng(); } RG[VD] = RS[VC]; }`.
   - **RS View table**: Confirmed RS[9] = PRNG Counter (read-only, with side effect of advancing PRNG).
   - **PRNG section**: Each SFPU lane has an independent 32-bit LFSR PRNG with XNOR taps at positions 31, 30, 10, and 0 (polynomial `x^32 + x^31 + x^11 + x^1 + 1`). Period >= 2^32 - 1. Re-seedable via PRNG_SEED config register.
   - **SFPSETSGN section**: With InstrMod[0]=1, sets the sign bit of RG[VD] to Imm12[0]. Used in dropout to clear the sign bit (Imm12[0]=0) of the PRNG output for unsigned comparison.
   - **SFPIADD section**: InstrMod[3]=1 inverts CC result sense; InstrMod[2]=0 enables CC update; InstrMod[1:0]=2 selects reg/reg subtraction. Confirmed the instruction computes `Tmp = RG[VC] - RG[VD]` and sets `CC.Res = !(Tmp < 0)` when InstrMod[3]=1.
   - **SFPENCC section**: InstrMod[1:0]=0 keeps CC.En unchanged; InstrMod[3]=0 sets CC.Res=1. Executes on all lanes regardless of LaneEnabled.
   - **Predicated Execution section**: Documented CC.En/CC.Res truth table and LaneEnabled formula: `(~CC.En | CC.Res) & ~RowDisable`. The ISA states SFPIADD "Sets CC Enable? N", but production code (including this dropout kernel) relies on SFPIADD CC.Res updates to gate subsequent instructions, indicating an implicit CC enable mechanism not fully described in the ISA documentation.
   - **SFPLOADI section**: InstrMod=10 loads low 16 bits; InstrMod=8 loads high 16 bits to construct 32-bit values in LREGs.

### Glean References
No Glean queries were made for this analysis.
