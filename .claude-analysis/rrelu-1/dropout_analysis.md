## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the DROPOUT operation.

### Unary Dispatch Summary
- **UnaryOpType**: `DROPOUT` (defined in `unary_op_types.hpp` but **not** wired into the standard `UnaryProgramFactory`; uses a dedicated `DropoutProgramFactory` at `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- DROPOUT does not use the `SFPU_OP_CHAIN_0` macro. The compute kernel calls `dropout_tile(0, int_probability, int_scale_factor)` directly.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `dropout_program_factory.cpp` line 240: `bool math_approx_mode = false;` -- hardcoded in the factory |
| Template parameter (SFPU kernel) | `APPROXIMATION_MODE=false` (from `APPROX`) | `dropout.h` passes `APPROX` (a `constexpr bool` generated at JIT compile time from `ComputeConfig.math_approx_mode` via `genfiles.cpp`) to the macros `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, ...)` and `SFPU_ONE_PARAM_KERNEL_INIT(dropout, sfpu::dropout_init, APPROX, seed)` |
| Effective SFPU path | `APPROXIMATION_MODE=false`, but the `_calculate_dropout_` kernel has **no** `if constexpr` branches on `APPROXIMATION_MODE` -- the template parameter is accepted but entirely unused. The kernel executes the same instruction sequence regardless. | `ckernel_sfpu_dropout.h` -- no conditional paths based on `APPROXIMATION_MODE` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- `dropout.h` invokes the macro `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` which directly expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_dropout<APPROX>, ...)` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` (identical on Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain
1. **Compute kernel** (`dropout_kernel.cpp`) calls `dropout_tile(0, int_probability, int_scale_factor)`.
2. **API header** (`dropout.h`) wraps this as `MATH(SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, idst, probability, scale_factor))`, which expands to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_dropout<false>, 0, (int)VectorMode::RC, probability, scale_factor)`.
3. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls until SFPU is ready, then iterates over 4 faces (`VectorMode::RC`), calling `ckernel::sfpu::calculate_dropout<false>(probability, scale_factor)` per face, with `SETRWC` advancing the DEST pointer between faces.
4. **LLK wrapper** (`ckernel_sfpu_dropout.h` in `hw/ckernels/`) calls `_calculate_dropout_<false, 8>(8, probability, scale)`.
5. **Core SFPU implementation** (`ckernel_sfpu_dropout.h` in `tt_llk`) executes the low-level TTI instruction sequence for 8 iterations per face.

Additionally, `dropout_kernel_init(seed)` is called once at kernel startup. This expands via `SFPU_ONE_PARAM_KERNEL_INIT(dropout, sfpu::dropout_init, APPROX, seed)` to:
1. `llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, false>(sfpu::dropout_init<false>, seed)` which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` (configures SFPU config register, address modes, and resets counters), then calls `dropout_init<false>(seed)`.
2. `dropout_init<false>(seed)` calls `_init_dropout_(seed)` which writes the seed to the PRNG seed configuration register and waits 600 NOP cycles for the PRNG to initialize.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (faces 0, 1, 2, 3), covering all 1024 elements.
- **Operation invocation**: The dispatch loop calls `calculate_dropout<false>(probability, scale_factor)` once per face. Within each invocation, the core function loops for `ITERATIONS=8` sfpi rows per face (the default template parameter value).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On both Wormhole and Blackhole, `ADDR_MOD_7` is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` (the default for all non-special-cased `SfpuType` values in `eltwise_unary_sfpu_configure_addrmod`). The actual DEST row advancement is done explicitly by `dst_reg++` in the SFPU kernel (which advances 1 sfpi row = 2 physical DEST rows) and by `TTI_SETRWC(..., CR_D, 8, ..., SET_D)` pairs between faces in the parameters dispatch.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with condition code manipulation (`SFPIADD` sets CC.Res per lane, `SFPMOV` is CC-guarded, `SFPENCC` resets CC.Res). Style B analysis follows.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h
// (Wormhole B0 and Blackhole implementations are identical)

namespace ckernel
{
namespace sfpu
{

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
        // sfpi::dst_reg[0] = sfpi::dst_reg[0] * s2vFloat16b(scale);
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
        //   sfpi::dst_reg[0] = vConst0;
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

} // namespace sfpu
} // namespace ckernel
```

#### CC State Machine -- `_calculate_dropout_`

The kernel has one CC block per iteration: `SFPIADD` sets `CC.Res` based on a comparison between a random number and the dropout probability, `SFPMOV` is CC-guarded to conditionally zero out elements that should be dropped, and `SFPENCC` resets `CC.Res=1` so the subsequent `SFPSTORE` executes unconditionally on all lanes.

**Prerequisite**: For the predicated execution to function correctly, `CC.En` must be 1 (enabled) when the kernel begins execution. The `SFPENCC(0,0,0,0)` at the end of each iteration uses `InstrMod[1:0]=0` ("keep previous CC.En"), so once CC.En=1, it is preserved across all iterations. The initial CC.En=1 state is established by prior SFPU context or hardware default.

```
_calculate_dropout_ -- CC State Transitions (per iteration d)
================================================================

  CC State: CC.En=1, CC.Res=1 (all lanes active)
       |
       |  SFPLOAD  LREG0 <- DEST[current_row]       (no CC effect) -- load input element
       |  SFPMUL   LREG0 = LREG0 * LREG1 + LCONST_0 (no CC effect) -- scale by scale_factor
       |           (SFPMUL is alias for SFPMAD with VC=LCONST_0=0.0, so: LREG0 * LREG1 + 0)
       |  SFPMOV   LREG3 = RS[9] (PRNG), mod1=8     (no CC effect) -- generate random uint32
       |           (RS[9] = PRNG Counter; reading it advances PRNG by 1 step)
       |  SFPSETSGN LREG3, mod1=1 (Imm12-based)     (no CC effect) -- set sign bit to Imm12[0]=0
       |           (clears sign bit, making rand non-negative for integer comparison)
       |
       v
  +---------------------------------------------------------+
  | SFPIADD  imm12=0, VC=LREG2, VD=LREG3, mod1=10         |
  |   mod1=10 = binary 1010:                                |
  |     [1:0]=2: Reg/Reg subtraction (Tmp = LREG2 - LREG3) |
  |     [2]=0:   CC.Res IS updated (based on sign of Tmp)   |
  |     [3]=1:   CC.Res IS inverted after update             |
  |                                                          |
  | Operation: Tmp = probability - rand                      |
  | CC.Res <- (Tmp < 0), then inverted                       |
  | CC.Res <- !(probability - rand < 0)                      |
  |        =  (probability >= rand)                          |
  |                                                          |
  | Result (Tmp) stored in LREG3 (VD), but value is unused.  |
  | CC.En is NOT modified (SFPIADD does not set CC Enable).  |
  +---------------------------+-----------------------------+
                              |
                              v
  CC State: CC.En=1, CC.Res per lane = (probability >= rand)
       |
       |  Lanes where rand < probability:  CC.Res=1 (active)  -> element DROPPED
       |  Lanes where rand >= probability: CC.Res=0 (inactive) -> element KEPT
       |
       |  SFPMOV  LREG0 = RG[LCONST_0], mod1=0
       |    (CC-guarded: only active lanes where rand < probability)
       |    (In active lanes: LREG0 overwritten with 0.0, destroying scaled value)
       |    (In inactive lanes: LREG0 retains the scaled value from SFPMUL)
       |
       v
  +-----------------------------------------------------------+
  | SFPENCC  imm12=0, lreg_c=0, lreg_dest=0, mod1=0          |
  |   mod1=0 = SFPENCC_MOD1_EU_R1:                            |
  |     InstrMod[3]=0: CC.Res = 1                             |
  |     InstrMod[1:0]=0: CC.En unchanged (stays 1)            |
  |                                                            |
  | CC <- CC.En=1, CC.Res=1 (all lanes active again)          |
  | NOTE: SFPENCC executes on ALL lanes regardless of          |
  |       current LaneEnabled state.                           |
  +---------------------------+-------------------------------+
                              |
                              v
  CC State: CC.En=1, CC.Res=1 (all lanes active)
       |
       |  SFPSTORE LREG0 -> DEST[current_row], mod1=0
       |    (unconditional: all lanes active, stores either 0.0 or scaled value)
       |
       v
  dst_reg++ (advance to next sfpi row)
  (loop continues for next iteration with CC.En=1, CC.Res=1)
```

**Key CC observations:**
- `SFPIADD` with `mod1=10` performs integer subtraction `probability - rand` and sets `CC.Res = !(result < 0)`, which is equivalent to `CC.Res = (probability >= rand)`. This means lanes where `rand < probability` have `CC.Res=1` (active) and will be zeroed.
- `SFPMOV` with `mod1=0` is CC-guarded: it only executes on lanes where `LaneEnabled = (~CC.En | CC.Res) & ~RowDisable`. With `CC.En=1`, only lanes with `CC.Res=1` are active. This selectively zeroes elements that should be dropped.
- `SFPENCC` with `mod1=0` (`SFPENCC_MOD1_EU_R1`) resets `CC.Res=1` across ALL lanes (this instruction is not itself CC-guarded) while keeping `CC.En=1`. This ensures the subsequent `SFPSTORE` executes unconditionally on all lanes, writing back either the zeroed value (dropped) or the scaled value (kept).
- The `SFPIADD` side effect of writing the subtraction result to `LREG3` (VD) is benign -- `LREG3` is not used afterward in this iteration and is overwritten by the PRNG `SFPMOV` at the start of the next iteration.
- `CC.En` is NEVER explicitly set in this kernel. It relies on `CC.En=1` being established before the kernel runs (either by hardware default or prior SFPU operations). The `SFPENCC(0,0,0,0)` at the end of each iteration preserves this state.

### SFPU Instructions Used

| Instruction | Opcode | Count per Iteration | Description |
|-------------|--------|---------------------|-------------|
| `TT_SFPLOADI` | 0x71 | 4 (before loop) | Load 16-bit immediate into LREG. Used with `InstrMod=10` (LO16_ONLY, preserves upper bits) and `InstrMod=8` (HI16_ONLY, preserves lower bits) to build full 32-bit values in LREG1 (scale) and LREG2 (probability). Uses `TT_` form (writes to instruction buffer) rather than `TTI_` (immediate execution). |
| `TTI_SFPLOAD` | 0x70 | 1 | Load from DEST register file into LREG0. `InstrMod=0` (IMPLIED format, auto-detects FP16_B/FP32), `AddrMod=3`, `Addr=0`. Reads the current tile element from DEST at the address set by the DEST write pointer. |
| `TTI_SFPMUL` | 0x86 | 1 | Alias of SFPMAD with `VC=LCONST_0` (0.0 addend). Computes `LREG0 = LREG0 * LREG1 + 0.0`, effectively multiplying the input by the scale factor. `InstrMod=0` (no sign inversions, result to VD). Latency: 2 cycles. |
| `TTI_SFPMOV` | 0x7C | 2 | (1) With `InstrMod=8, VC=9`: reads from `RS[9]` (PRNG Counter), advancing the PRNG by one step. Generates a random uint32 in LREG3. (2) With `InstrMod=0, VC=LCONST_0`: conditionally copies 0.0 to LREG0 (CC-guarded, only on lanes where `rand < probability`). |
| `TTI_SFPSETSGN` | 0x89 | 1 | Sets the sign of LREG3 to `Imm12[0]=0` (clear sign bit). `InstrMod=1` (source from Imm12). Makes the random number non-negative so it can be compared with `probability` (which is also non-negative) via unsigned-like integer subtraction. |
| `TTI_SFPIADD` | 0x79 | 1 | Integer subtraction with CC update. `InstrMod=10` (binary 1010): `[1:0]=2` (Reg/Reg subtraction, `Tmp = LREG2 - LREG3` = `probability - rand`), `[2]=0` (update CC.Res based on sign), `[3]=1` (invert CC.Res). Sets `CC.Res = (probability >= rand)` per lane. IPC: 1, Latency: 1. |
| `TTI_SFPENCC` | 0x8A | 1 | Directly sets CC state. `InstrMod=0` (`SFPENCC_MOD1_EU_R1`): keeps CC.En unchanged, sets CC.Res=1 on all lanes. Executes unconditionally regardless of LaneEnabled state. Used to "exit" the predicated block so SFPSTORE runs on all lanes. |
| `TTI_SFPSTORE` | 0x72 | 1 | Store from LREG0 back to DEST register file. `InstrMod=0` (IMPLIED format), `AddrMod=3`, `Addr=0`. Writes the result (either 0.0 for dropped elements or scaled value for kept elements) back to DEST. Latency: 3 cycles (DEST target). |

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| **LREG0** | Working register for the tile element. Loaded from DEST, scaled by SFPMUL, conditionally zeroed by SFPMOV, stored back to DEST. | Per-iteration |
| **LREG1** | Holds the `scale` parameter as a 32-bit float. Loaded once before the loop via two SFPLOADI calls (lower 16 bits then upper 16 bits). Used as multiplicand in SFPMUL. | Entire function (read-only after setup) |
| **LREG2** | Holds the `probability` parameter as a 32-bit unsigned integer. Loaded once before the loop via two SFPLOADI calls. Used as VC operand in SFPIADD for the comparison. | Entire function (read-only after setup) |
| **LREG3** | Holds the random number generated by the PRNG. Written by SFPMOV (PRNG read), sign-cleared by SFPSETSGN, then used as VD operand in SFPIADD (also receives the subtraction result, which is discarded). | Per-iteration |
| **LCONST_0** (`RG[9]`) | Hardware constant 0.0. Used as the addend in SFPMUL (making it a pure multiplication) and as the source for the conditional zero-out SFPMOV. | Hardware constant (read-only) |
| **RS[9]** (PRNG Counter) | Special status register. Reading it via SFPMOV with `InstrMod=8` advances the PRNG and returns the current pseudorandom value. | Hardware PRNG state |
| **DEST** | Source and destination for tile data. SFPLOAD reads from current DEST row; SFPSTORE writes back. Address advancement is via `dst_reg++` (1 sfpi row = 2 physical DEST rows). | Per-tile face |

### Address Mode Configuration

The address mode for DROPOUT is `ADDR_MOD_7`, configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` via `eltwise_unary_sfpu_configure_addrmod<SfpuType::dropout>()`.

Since `SfpuType::dropout` does not match any special-cased type in the `if constexpr` chain (which handles `topk_local_sort`, `typecast`, `unary_max/min`, etc.), the default `ADDR_MOD_7` is used:

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

| Field | Value | Meaning |
|-------|-------|---------|
| `srca.incr` | 0 | No auto-increment for source A |
| `srcb.incr` | 0 | No auto-increment for source B |
| `dest.incr` | 0 | No auto-increment for DEST |

This is the same on both Wormhole and Blackhole. The zero-increment address mode means the SFPU kernel explicitly controls DEST addressing via `dst_reg++` (per iteration) and `SETRWC` (between faces) rather than relying on hardware auto-increment.

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "How does the SFPMOV instruction generate pseudorandom numbers on the SFPU? Specifically, what does instr_mod1=8 and lreg_c=9 mean for SFPMOV in the context of the PRNG?"
   **Reason**: Needed to understand the PRNG generation mechanism used in the dropout kernel.
   **Key Findings**: SFPMOV with InstrMod=8 reads from the RS (Status) register view. RS[9] is the PRNG Counter. Reading from it advances the PRNG by one step and returns a random 32-bit value to the destination LREG.

2. [SFPU] **Query**: "How does SFPIADD affect the condition code (CC) flags on the SFPU? Specifically, what happens when instr_mod1=10 is used with SFPIADD?"
   **Reason**: Needed to understand the conditional execution flow in the dropout kernel's drop-or-keep logic.
   **Key Findings**: DeepWiki could not provide sufficient detail. Escalated to Confluence ISA page.

### Confluence References
1. [SFPU] **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**:
   - **SFPIADD**: Confirmed InstrMod[1:0]=2 is Reg/Reg subtraction, [2]=0 updates CC.Res based on sign, [3]=1 inverts CC.Res. Sets CC.Res but NOT CC.En.
   - **SFPMOV**: Confirmed InstrMod=8 reads from RS register view, RS[9]=PRNG Counter with side-effect of advancing PRNG. InstrMod=0 is conditional on LaneEnabled.
   - **SFPSETSGN**: Confirmed InstrMod=1 sets sign bit from Imm12[0].
   - **SFPENCC**: Confirmed InstrMod[1:0]=0 keeps CC.En unchanged, InstrMod[3]=0 sets CC.Res=1. Executes on ALL lanes regardless of LaneEnabled.
   - **SFPMUL/SFPMAD**: Confirmed SFPMUL is alias of SFPMAD with VC=0.0 addend. FP32 FMA with subnormal flush.
   - **SFPLOAD/SFPSTORE**: Confirmed IMPLIED format mode (InstrMod=0) auto-detects format from register file.
   - **SFPLOADI**: Confirmed InstrMod=10 (LO16_ONLY) preserves upper bits, InstrMod=8 (HI16_ONLY) preserves lower bits.
   - **SFPU Status (RS) View**: Confirmed RS[9] = PRNG Counter with side-effect of advancing PRNG on read.
   - **Predicated Execution**: Confirmed LaneEnabled = (~CC.En | CC.Res) & ~RowDisable. CC.En=0 means all lanes active; CC.En=1 means CC.Res controls per-lane activity.
   - **SFPU Control Register**: Confirmed fields including ROW_DISABLE[15:12], STORE_DISABLE, LOAD_DISABLE.
   - **SFPSETCC**: Noted that with CC.En=0, SFPSETCC sets CC.Res=0 regardless of test, unlike SFPIADD which always updates CC.Res based on the operation result.

### Glean References
None required -- the Confluence SFPU ISA page provided sufficient detail for all instructions used in the dropout kernel.
