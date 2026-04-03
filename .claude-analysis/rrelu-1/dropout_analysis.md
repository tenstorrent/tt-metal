## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `DROPOUT` (exists in `unary_op_types.hpp` but is NOT dispatched through the standard `UnaryProgramFactory`)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- dropout uses a custom program factory (`DropoutProgramFactory`) with its own compute kernel that directly calls `dropout_tile(0, int_probability, int_scale_factor)` instead of using `SFPU_OP_CHAIN_0`

**Non-standard dispatch note**: Unlike most unary operations that route through `UnaryProgramFactory` and `unary_op_utils.cpp`, dropout has its own dedicated program factory at `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`. The `DROPOUT` enum value exists in `unary_op_types.hpp` but has no corresponding case in `get_op_init_and_func()` or `get_op_approx_mode()`. All dispatch happens through the experimental dropout path.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | Hardcoded in `dropout_program_factory.cpp` line 241: `bool math_approx_mode = false;` |
| Template parameter (SFPU function) | `APPROXIMATION_MODE` (resolved from `APPROX` compile-time define) | The `dropout_tile()` API uses `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, ...)` where `APPROX` is the compile-time `math_approx_mode` define -- resolves to `false` |
| Effective SFPU path | `APPROXIMATION_MODE` is a template parameter of `_calculate_dropout_` but the function body does NOT branch on it -- the same raw TTI instruction sequence executes regardless | The `_calculate_dropout_` function has `APPROXIMATION_MODE` in its template signature but never uses `if constexpr (APPROXIMATION_MODE)` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_dropout.h` (thin wrapper that forwards to core implementation) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_dropout.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (generic `_llk_math_eltwise_unary_sfpu_params_` function) |

### Call Chain

1. **Compute kernel** (`dropout_kernel.cpp`) calls `dropout_tile(0, int_probability, int_scale_factor)`.
2. **API header** (`dropout.h`) expands this via macro `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, idst, probability, scale_factor)` which calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_dropout<APPROX>, idst, (int)VectorMode::RC, probability, scale_factor)`.
3. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls until SFPU is ready, then calls `calculate_dropout(probability, scale_factor)` once per face (4 faces for VectorMode::RC), with `TTI_SETRWC` between faces.
4. **LLK wrapper** (`ckernel_sfpu_dropout.h` in the llk_api layer) defines `calculate_dropout<APPROX, ITERATIONS>()` which directly calls `_calculate_dropout_<APPROX, ITERATIONS>(ITERATIONS, probability, scale)`.
5. **Core SFPU implementation** (`ckernel_sfpu_dropout.h` in tt_llk) executes the raw TTI instruction sequence.

Additionally, `dropout_kernel_init(seed)` is called once at kernel startup:
1. **API header** (`dropout.h`) expands via `SFPU_ONE_PARAM_KERNEL_INIT(dropout, sfpu::dropout_init, APPROX, seed)` which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, APPROX>(dropout_init<APPROX>, seed)`.
2. **Init dispatch** (`llk_math_eltwise_unary_sfpu_init.h`) calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` (which configures SFPU config reg, address modes, resets counters) and then calls `dropout_init<APPROX>(seed)`.
3. **Init function** (`ckernel_sfpu_dropout.h`) calls `_init_dropout_(seed)` which calls `init_prng_seed(seed)` from `ckernel.h` -- writes the seed to the PRNG config register and executes 600 SFPNOP cycles to let the PRNG state settle.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full tile coverage of 1024 elements).
- **Operation invocation**: For RC mode, `_llk_math_eltwise_unary_sfpu_params_` calls `calculate_dropout(probability, scale_factor)` in a loop of 4 face iterations. Within each face, the core SFPU function runs 8 sfpi-row iterations (the `ITERATIONS` template default). Between faces, two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions advance the DEST write counter by 16 physical rows (one face stride).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration within the core SFPU function, SETRWC between faces). Address mode is `ADDR_MOD_7` on both Wormhole B0 and Blackhole with all increment fields set to 0 (`srca.incr=0, srcb.incr=0, dest.incr=0`). The SFPLOAD/SFPSTORE instructions in the kernel use `addr_mode=3` (ADDR_MOD_7 after base offset 4) which provides a zero-increment mode -- the DEST address auto-increment is instead handled by `dst_reg++` (the SFPI `dst_reg` abstraction) within the loop.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with CC manipulation via SFPIADD. Style B is applied for the CC State Machine diagram.

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
```

#### CC State Machine -- `_calculate_dropout_`

The CC state machine controls which lanes get their value zeroed out (dropped) vs which lanes keep the scaled value. The SFPIADD instruction compares the per-lane PRNG output against the dropout probability threshold. On Wormhole B0 and Blackhole, CC.En is assumed to be 1 (enabled) at kernel entry -- this is consistent with all raw-TTI SFPU kernels in the codebase which never explicitly enable CC before using SFPSETCC or SFPIADD for conditional execution.

```
_calculate_dropout_ -- CC State Transitions (per iteration)
================================================================

  CC State: ENABLED (CC.En=1, CC.Res=1)       <-- assumed at entry
       |
       |  SFPLOADI LREG1 = lo16(scale)         (no CC effect) -- load scale factor lower 16 bits
       |  SFPLOADI LREG1 = hi16(scale)         (no CC effect) -- load scale factor upper 16 bits
       |  SFPLOADI LREG2 = lo16(probability)   (no CC effect) -- load probability lower 16 bits
       |  SFPLOADI LREG2 = hi16(probability)   (no CC effect) -- load probability upper 16 bits
       |
       |  == Begin per-sfpi-row loop (8 iterations per face) ==
       |
       |  SFPLOAD  LREG0 = DEST[current_row]   (no CC effect) -- load input element from DEST
       |  SFPMUL   LREG0 = LREG0 * LREG1       (no CC effect) -- scale input by scale_factor
       |  SFPMOV   LREG3 = RS[9] (PRNG)         (no CC effect) -- read 32-bit random value from PRNG
       |  SFPSETSGN LREG3.sign = imm12[0]=0     (no CC effect) -- clear sign bit to make unsigned
       |
       v
  +----------------------------------------------+
  | SFPIADD  mod1=10 (0b1010)                   |
  |   operation: Tmp = LREG2.INT32 - LREG3.INT32|
  |   = probability - rand                       |
  |   result stored: LREG3 = Tmp[31:0]           |
  |                                              |
  |   InstrMod[2]=0: CC.Res IS updated           |
  |   InstrMod[3]=1: CC.Res is INVERTED          |
  |                                              |
  | CC.Res <- !(Tmp < 0)                        |
  |        = !(probability - rand < 0)           |
  |        = (probability >= rand)               |
  |        = (rand <= probability)               |
  +-----------------------+----------------------+
                          |
                          v
  CC State: ENABLED where rand <= probability (element should be DROPPED)
       |
       |  SFPMOV   LREG0 = LCONST_0 (0.0)   (CC-guarded: only lanes where rand <= probability
       |                                       get LREG0 overwritten to 0.0; other lanes keep
       |                                       scaled value from SFPMUL)
       |
       v
  +----------------------------------------------+
  | SFPENCC  mod1=0 (0b0000)                    |
  |   InstrMod[3]=0: CC.Res = 1                 |
  |   InstrMod[1:0]=0: CC.En unchanged (stays 1)|
  |                                              |
  | CC.Res <- 1 (all lanes active again)         |
  +-----------------------+----------------------+
                          |
                          v
  CC State: ENABLED (CC.En=1, CC.Res=1) -- all lanes active
       |
       |  SFPSTORE LREG0 -> DEST[current_row]  (no CC effect) -- store result to DEST
       |  dst_reg++                             -- advance to next sfpi row
       |
       v  (loop continues or returns)

  == End of per-sfpi-row loop ==
```

**Key CC observations:**
- SFPIADD with `InstrMod=10` (binary `1010`) performs integer subtraction (`probability - rand`) and sets CC.Res to the *inverted* sign of the result, effectively testing `rand <= probability`
- The SFPMOV immediately after is CC-guarded: lanes where `rand <= probability` (should be dropped) get LREG0 overwritten to 0.0, while lanes where `rand > probability` (should be kept) retain the scaled value from SFPMUL
- SFPENCC(0,0,0,0) resets CC.Res=1 for all lanes (unconditional execution), keeping CC.En=1 for the next iteration
- The SFPSTORE after SFPENCC executes unconditionally on all lanes, writing either the scaled value or 0.0 depending on what SFPMOV did
- CC.En is assumed to be 1 at kernel entry -- this is NOT explicitly set by the dropout kernel or the dispatch layer, but is consistent with all WH B0/BH raw-TTI SFPU kernels in the codebase (none of them call SFPENCC to enable CC before first use)

### SFPU Instructions Used

| Instruction | Count | Description |
|-------------|-------|-------------|
| `SFPLOADI` | 4 (setup) | Load 16-bit immediate to LREG. Used to load 32-bit `scale` and `probability` values as two 16-bit halves (low bits with `InstrMod=10`, high bits with `InstrMod=8`) |
| `SFPLOAD` | 1 (per iter) | Load from DEST to LREG0. Uses `InstrMod=0` (implied format) and `addr_mode=3` (ADDR_MOD_7). Reads the current tile element from DEST into LREG0 |
| `SFPMUL` | 1 (per iter) | Multiply LREG0 by LREG1 (scale factor), add LCONST_0 (0.0). Result: `LREG0 = input * scale`. This is the inverted-dropout scaling |
| `SFPMOV` | 1 (per iter, PRNG) | With `InstrMod=8, VC=9`: reads from the SFPU Status register view RS[9] (PRNG Counter) into LREG3. This advances the per-lane PRNG by one step and loads the 32-bit random value |
| `SFPMOV` | 1 (per iter, zero) | With `InstrMod=0, VC=LCONST_0, VD=LREG0`: CC-guarded copy of constant 0.0 to LREG0. Only executes on lanes where `rand <= probability` |
| `SFPSETSGN` | 1 (per iter) | With `InstrMod=1, Imm12=0`: sets the sign bit of LREG3 to `Imm12[0]=0`, effectively clearing the sign bit to make the random value non-negative for unsigned comparison with probability |
| `SFPIADD` | 1 (per iter) | Integer subtraction with CC update: `LREG3 = LREG2 - LREG3` (probability - rand). `InstrMod=10` (`0b1010`): `[3]=1` inverts CC, `[2]=0` updates CC, `[1:0]=2` reg/reg subtract. Sets `CC.Res = !(result < 0) = (probability >= rand)` |
| `SFPENCC` | 1 (per iter) | With `InstrMod=0, Imm12=0`: sets CC.Res=1 for all lanes, keeping CC.En unchanged. Resets predication so all lanes are active for the subsequent SFPSTORE |
| `SFPSTORE` | 1 (per iter) | Store LREG0 to DEST. Uses `InstrMod=0` (implied format) and `addr_mode=3` (ADDR_MOD_7). Writes the result (scaled value or 0.0) back to DEST |

**Init-only instructions** (in `_init_dropout_` -> `init_prng_seed`):

| Instruction | Description |
|-------------|-------------|
| `SFPNOP` | 600 no-ops executed after seeding the PRNG via config register write, allowing the PRNG LFSR state to stabilize |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register: loaded from DEST (input element), then multiplied by scale factor. May be overwritten to 0.0 by CC-guarded SFPMOV if the element should be dropped. Final value is stored back to DEST |
| **LREG1** | Holds the 32-bit `scale` factor (loaded once before the loop via two SFPLOADI instructions). Persists across all iterations within a face |
| **LREG2** | Holds the 32-bit `probability` threshold as a signed integer (loaded once before the loop via two SFPLOADI instructions). Persists across all iterations within a face |
| **LREG3** | Temporary: receives the 32-bit PRNG value from RS[9] via SFPMOV, then has its sign bit cleared by SFPSETSGN. Also receives the subtraction result from SFPIADD (probability - rand), though this result is not used further |
| **LCONST_0** (index 9) | Fixed constant 0.0. Read by SFPMOV as the "zero" value to store when dropping an element, and by SFPMUL as the addend (making it a pure multiply) |
| **DEST** | Source and destination for tile data. Current sfpi row loaded via SFPLOAD, result written back via SFPSTORE. `dst_reg++` advances by 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration |
| **RS[9] (PRNG Counter)** | Read-only status register containing the PRNG output. Accessed via SFPMOV with `InstrMod=8, VC=9`. Reading advances the PRNG by one step (32-bit LFSR with polynomial `x^32 + x^31 + x^11 + x^1 + 1`) |
| **PRNG_SEED config register** | Written once during init via `init_prng_seed(seed)` through the Tensix front-end config path (`cfg[PRNG_SEED_Seed_Val_ADDR32] = seed`). Seeds the per-lane LFSR |

### Address Mode Configuration

**Wormhole B0 and Blackhole** (identical):

The dropout operation uses `ADDR_MOD_7` (after the addr mod base offset of 4, so physical ADDR_MOD index 7 maps to the "ADDR_MOD_7" configured during init):

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);
```

All increment fields are zero. The SFPLOAD and SFPSTORE instructions in the kernel use `addr_mode=3` which, with the addr mod base set to 1 (by `math::set_addr_mod_base()`), maps to ADDR_MOD_7. This zero-increment mode means the DEST RWC is not auto-incremented by SFPLOAD/SFPSTORE -- instead, DEST address progression is managed by `sfpi::dst_reg++` at the end of each loop iteration, which increments the SFPU's internal pointer by 1 sfpi row (= 2 physical DEST rows, covering 32 elements).

Dropout does NOT have a special `ADDR_MOD_6` case (unlike typecast, signbit, or topk operations). Both hardware generations use the same address mode configuration.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPMOV instruction generate pseudorandom numbers when used with instr_mod1=8?"
   **Reason**: Needed to understand the PRNG access mechanism in the dropout kernel
   **Key Findings**: DeepWiki was unavailable (repository not indexed). Fell back to Confluence ISA page.

### Confluence References
1. **Section**: SFPMOV instruction description (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: SFPMOV with `InstrMod=8` copies from the SFPU Status register view (RS). When `VC=9` (PRNG Counter), the instruction first calls `AdvancePrng()` to step the LFSR, then reads the new PRNG value into `RG[VD]`.

2. **Section**: PRNG specification (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: Each SFPU lane has a 32-bit LFSR-based PRNG with XNOR taps at positions 31, 30, 10, and 0 (polynomial `x^32 + x^31 + x^11 + x^1 + 1`). Period is at least `2^32 - 1` cycles. Can be re-seeded via the Tensix front-end (software provided 32-bit value).

3. **Section**: SFPU Status (RS) View table (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: RS[9] maps to the PRNG Counter register (read-only, with side effect of advancing the PRNG by 1 step on read).

4. **Section**: SFPIADD instruction description (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: `InstrMod[3]=1` inverts CC.Res, `InstrMod[2]=0` enables CC update, `InstrMod[1:0]=2` selects reg/reg subtraction. Does NOT set CC.En (per ISA table). The conditional execution in the dropout kernel relies on CC.En being 1 at entry.

5. **Section**: SFPSETSGN instruction description (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: With `InstrMod[0]=1`, sets the sign bit of VD to `Imm12[0]`. Copies exponent and mantissa from VC. Used in dropout to clear the sign bit of the PRNG output (`Imm12=0` means sign bit = 0, making the value non-negative).

6. **Section**: SFPENCC instruction description (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: With `InstrMod=0` (`InstrMod[3]=0, InstrMod[1:0]=0`): sets CC.Res=1, keeps CC.En unchanged. Executes on ALL lanes regardless of current LaneEnabled state.

7. **Section**: Predicated Execution and CC Registers (Tensix SFPU ISA page ID 1170505767)
   **Key Findings**: CC.En=0 means unconditional execution (lane always active). CC.En=1 enables predicated execution where CC.Res determines lane activity. The dropout kernel (and all WH B0/BH raw-TTI kernels) assumes CC.En=1 at entry without explicitly setting it.

### Glean References
No Glean queries were made for this analysis.
