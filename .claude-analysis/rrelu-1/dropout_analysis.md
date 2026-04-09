## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `DROPOUT` (listed in `unary_op_types.hpp:104`, but dropout uses its own experimental program factory rather than the standard `UnaryProgramFactory`)
- **Program Factory**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp` (`DropoutProgramFactory::create`)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: Not applicable -- dropout does not use the standard `SFPU_OP_CHAIN_0` macro mechanism. Instead, the compute kernel directly calls `dropout_tile(0, int_probability, int_scale_factor)` and `dropout_kernel_init(seed)`.

**Note on non-standard dispatch**: Dropout is an experimental operation with its own program factory. It does not route through `UnaryProgramFactory` or `get_block_defines()`. The compute kernel (`dropout_kernel.cpp`) includes `"api/compute/eltwise_unary/dropout.h"` which defines `dropout_tile()` and `dropout_kernel_init()`. These API-level functions (nuked from this codebase) would call through to the LLK dispatch layer and ultimately to `_calculate_dropout_()` and `_init_dropout_()` in `ckernel_sfpu_dropout.h`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `dropout_program_factory.cpp:240` -- hardcoded `bool math_approx_mode = false;` |
| Template parameter (SFPU kernel) | `APPROXIMATION_MODE` (unused in kernel body) | `_calculate_dropout_` is templated on `APPROXIMATION_MODE` but does not use it -- no `if constexpr` branches check it |
| Effective SFPU path | Single code path regardless of approximation mode | The kernel body contains no approximation-dependent branches; `APPROXIMATION_MODE` template parameter is present for API consistency but has no effect on execution |

### SFPU Abstraction Layers
The API header (`dropout.h`) and LLK dispatch file (`llk_math_eltwise_unary_sfpu_dropout.h`) have been removed from this codebase (deep-nuke). The core SFPU implementation files remain. The expected layer structure (based on analogous ops like `frac`) is documented below.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` [REMOVED in this codebase -- would define `dropout_tile()` and `dropout_kernel_init()`] |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_dropout.h` [REMOVED -- would call `_llk_math_eltwise_unary_sfpu_params_` with `_calculate_dropout_`] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_dropout.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`dropout_kernel.cpp`): Calls `dropout_tile(0, int_probability, int_scale_factor)` for each tile.
2. **API Header** (`dropout.h`, removed): Would wrap the call as `MATH((llk_math_eltwise_unary_sfpu_dropout<APPROX>(idst, probability, scale_factor)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_dropout.h`, removed): Would call `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_dropout_<APPROXIMATE, ITERATIONS>, dst_index, VectorMode::RC, ITERATIONS, probability, scale)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls SFPU, iterates over 4 faces calling the SFPU function, and advances DEST addresses between faces via `SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
5. **Core SFPU** (`ckernel_sfpu_dropout.h`): `_calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>()` executes the per-face SFPU microcode.
6. **Init path**: `dropout_kernel_init(seed)` -> `_init_dropout_(seed)` -> `init_prng_seed(seed)` which writes the seed to `PRNG_SEED_Seed_Val_ADDR32` and waits 600 NOPs for PRNG initialization.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces processed). Dropout operates on every element in the tile -- it must scale all elements and conditionally zero out a fraction based on probability.
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` template dispatches `_calculate_dropout_` once per face (4 times total for VectorMode::RC). The function receives `iterations` (8 per face), `probability`, and `scale` as arguments. Each invocation processes one full face (8 iterations x 32 elements = 256 elements).
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is configured with `{.srca={.incr=0}, .srcb={.incr=0}, .dest={.incr=0}}` (no auto-increment), and the `set_addr_mod_base()` call selects ADDR_MOD 4-7 range. Within each face, `dst_reg++` manually advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration. Between faces, `SETRWC` with `CR_D, 8` advances by face stride (2 x `inc_dst_addr<8>()` calls on Blackhole). The manual `dst_reg++` in the loop is what actually advances the DEST pointer within each face.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with condition code manipulation (SFPIADD sets CC, SFPMOV is CC-guarded, SFPENCC resets CC). The CC logic is relatively straightforward (single CC region per iteration), so **Style A** (inline-commented source) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h
// (Blackhole version is identical)

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, std::uint32_t probability, std::uint32_t scale)
{   // APPROXIMATION_MODE is unused (no branches depend on it), ITERATIONS is unused (iterations param used instead)
    // SFPU microcode

    // Load scale factor into LREG1 as a 32-bit float (two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);       // LREG1 lower 16 bits = scale[15:0], instr_mod0=10 means load to lo16
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);            // LREG1 upper 16 bits = scale[31:16], instr_mod0=8 means load to hi16
    // Load probability threshold into LREG2 as a 32-bit integer (two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);  // LREG2 lower 16 bits = probability[15:0]
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);      // LREG2 upper 16 bits = probability[31:16]
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        // === Step 1: Scale the input sample ===
        // Load current DEST row into LREG0 (instr_mod0=0: implied format, sfpu_addr_mode=3: RWC mode, addr=0)
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);
        // LREG0 = LREG0 * LREG1 + LCONST_0 (= 0.0), i.e., scaled_value = input * scale
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // === Step 2: Generate random number and clear sign bit ===
        // SFPMOV with instr_mod1=8 and lreg_c=9 triggers PRNG: generates a uint32 random number into LREG3
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
        // Clear sign bit of random number for unsigned comparison with probability
        // instr_mod1=1 means set sign from imm12_math (which is 0), effectively clearing the sign bit
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        // === Step 3: Compare random vs probability and conditionally zero ===
        // Integer subtract: LREG3 = LREG2 - LREG3 (probability - rand)
        // instr_mod1=10 = 0b1010: bit[1:0]=2 means subtract mode (VC - VD), bit[3]=1 inverts CC sense
        // CC.Res is set to 1 when result >= 0 (rand <= probability), i.e., lane should be dropped
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);
        // CC-guarded: Move LCONST_0 (=0.0) into LREG0 -- only executes on lanes where CC.Res=1
        // This zeros out the scaled value for lanes that should be "dropped"
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // Disable CC (all lanes active again): SFPENCC(0, 0, 0, 0) = mode EU_R1 with imm12=0
        // Sets CC.En=0 (unconditional), CC.Res=1
        TTI_SFPENCC(0, 0, 0, 0);
        // Store LREG0 back to DEST (instr_mod0=0: implied format, sfpu_addr_mode=3: RWC mode, addr=0)
        TTI_SFPSTORE(0, 0, 3, 0);

        sfpi::dst_reg++;  // Advance to next sfpi row (2 physical DEST rows = 32 elements)
    }
}

inline void _init_dropout_(const std::uint32_t seed)
{
    init_prng_seed(seed);  // Writes seed to PRNG_SEED register and waits 600 SFPNOPs for initialization
}
```

#### CC State Machine -- `_calculate_dropout_`

The CC logic per iteration is a single enable-compare-guard-reset cycle. SFPIADD sets CC based on the comparison result, one SFPMOV is CC-guarded, then SFPENCC resets CC for the unconditional SFPSTORE.

```
_calculate_dropout_ -- CC State Transitions (per iteration)
================================================================

  CC State: ALL_ENABLED (CC.En=0)                   <-- initial state (start of iteration)
       |
       |  TTI_SFPLOAD  LREG0 from DEST[0]           (no CC effect) -- load input value
       |  TTI_SFPMUL   LREG0 = LREG0 * LREG1       (no CC effect) -- scale input by scale factor
       |  TTI_SFPMOV   LREG3 = PRNG()               (no CC effect) -- generate random number
       |  TTI_SFPSETSGN LREG3 sign = 0               (no CC effect) -- clear sign bit for unsigned compare
       |
       v
  +---------------------------------------------+
  | TTI_SFPIADD  mod1=10 (subtract, invert CC)  |
  |   LREG3 = LREG2 - LREG3                     |
  |   (probability - rand)                       |
  |                                              |
  | CC.En <- 1                                   |
  | CC.Res <- (result >= 0) [inverted sense]     |
  |        = (rand <= probability)               |
  |        = (this lane should be DROPPED)       |
  +---------------------+------------------------+
                        |
                        v
  CC State: ENABLED where rand <= probability (lane should drop)
       |
       |  TTI_SFPMOV  LREG0 = LCONST_0 (0.0)   (CC-guarded: zeros out only dropped lanes)
       |
       v
  +---------------------------------------------+
  | TTI_SFPENCC  (0, 0, 0, 0)                   |
  |   mode = EU_R1: Enable unchanged, Res = 1   |
  |   imm12 = 0                                  |
  |                                              |
  | CC.En <- 0  (disable CC, all lanes active)   |
  | CC.Res <- 1                                  |
  +---------------------+------------------------+
                        |
                        v
  CC State: ALL_ENABLED (CC.En=0)
       |
       |  TTI_SFPSTORE LREG0 to DEST[0]         (no CC effect) -- store result unconditionally
       |  dst_reg++                              -- advance DEST pointer
       |
       v  (next iteration or function returns)
```

**Key CC observations:**
- SFPIADD with `instr_mod1=10` (`0b1010`) performs integer subtraction (`LREG2 - LREG3`) and sets CC with inverted sense (`bit[3]=1`). This means CC.Res=1 when the subtraction result is non-negative, i.e., `probability >= rand`, meaning the lane should be dropped (zeroed).
- Only one instruction (SFPMOV) is CC-guarded per iteration: it conditionally writes 0.0 to LREG0 for dropped lanes.
- SFPENCC immediately resets CC state so the subsequent SFPSTORE writes unconditionally to all lanes.
- CC state is fully reset at the end of each iteration, so there is no CC state carry-over between iterations.

### SFPU Instructions Used

| Instruction | Count per iteration | Description |
|-------------|-------------------|-------------|
| `TT_SFPLOADI` | 4 (once, before loop) | Load 16-bit immediate to LREG. Used to construct 32-bit scale and probability values in LREG1 and LREG2 by loading lower and upper 16-bit halves separately. `instr_mod0=10` loads to lo16, `instr_mod0=8` loads to hi16. |
| `TTI_SFPLOAD` | 1 | Load from DEST row to LREG0 with implied format conversion. `sfpu_addr_mode=3` selects RWC addressing, `dest_reg_addr=0` reads from the current DEST row. |
| `TTI_SFPMUL` | 1 | Multiply LREG0 by LREG1 with LCONST_0 addend (effectively `LREG0 = LREG0 * LREG1 + 0.0`). Scales the input value by the dropout scale factor. |
| `TTI_SFPMOV` | 2 | Register-to-register move. First instance (`instr_mod1=8, lreg_c=9`): special mode that triggers PRNG to generate a random number into LREG3. Second instance (`instr_mod1=0`): CC-guarded move of LCONST_0 (0.0) into LREG0 to zero dropped lanes. |
| `TTI_SFPSETSGN` | 1 | Set sign bit of LREG3. With `imm12_math=0` and `instr_mod1=1`, clears the sign bit of the random number, making it non-negative for unsigned comparison. |
| `TTI_SFPIADD` | 1 | Integer add/subtract. With `instr_mod1=10` (subtract mode with inverted CC), computes `LREG2 - LREG3` and sets CC.Res=1 where result >= 0 (i.e., `probability >= rand`). This is the core dropout decision. |
| `TTI_SFPENCC` | 1 | Enable/disable condition code. With all args=0 (mode EU_R1), disables CC masking so subsequent instructions execute on all lanes unconditionally. |
| `TTI_SFPSTORE` | 1 | Store LREG0 back to DEST row. `sfpu_addr_mode=3` selects RWC addressing, `dest_reg_addr=0` writes to the current DEST row. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register: loaded with input value from DEST, multiplied by scale, conditionally zeroed, then stored back to DEST. |
| **LREG1** | Scale factor: loaded once before the loop with the 32-bit float representation of the dropout scale (`1 / (1 - prob)`). Persists across all iterations. |
| **LREG2** | Probability threshold: loaded once before the loop with the 32-bit integer representation of `INT_MAX * prob`. Used as the comparison threshold for dropout decisions. Persists across all iterations. |
| **LREG3** | Random number: generated each iteration via the PRNG special mode of SFPMOV, then has its sign bit cleared. Used as the comparison operand against LREG2. Overwritten each iteration. |
| **LCONST_0** | Hardware constant register (index 9): always 0.0. Used as the zero value for dropped lanes (SFPMOV conditional zero) and as the addend in SFPMUL. |
| **DEST** | Source and destination for tile data. Each iteration loads one sfpi row (32 elements across 2 physical rows), processes it, and stores back. Advanced via `dst_reg++`. |

### Address Mode Configuration

Dropout uses the standard SFPU address mode configuration inherited from `_llk_math_eltwise_unary_sfpu_init_` / `eltwise_unary_sfpu_configure_addrmod`:

**Wormhole B0:**
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```
The `set_addr_mod_base()` call in `_llk_math_eltwise_unary_sfpu_start_` sets the addr mod base to 1, which maps logical ADDR_MOD references to the 4-7 range. Since `.dest.incr = 0`, there is no automatic DEST address increment -- the kernel manually advances via `sfpi::dst_reg++` each iteration.

**Blackhole:**
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```
Blackhole does NOT call `set_addr_mod_base()` in its `_llk_math_eltwise_unary_sfpu_start_`. The ADDR_MOD_7 configuration is the same. Manual DEST advancement via `sfpi::dst_reg++` is identical.

Both architectures: No special ADDR_MOD overrides for dropout (it does not match any of the `if constexpr` conditions for `topk_local_sort`, `typecast`, or min/max operations that would set ADDR_MOD_6).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checked for dropout dispatch configuration (approx mode, compute kernel path, block defines)
   **Key Findings**: DROPOUT enum exists in types but is not present in `get_op_init_and_func_default` or `get_op_init_and_func_parameterized` -- dropout uses its own experimental program factory

2. **File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`
   **Reason**: Understand how dropout configures its compute kernel and passes parameters
   **Key Findings**: `math_approx_mode = false`, probability converted to `INT_MAX * prob`, scale passed as bit-cast uint32, seed passed as runtime arg

3. **File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
   **Reason**: Identify the compute kernel dispatch calls
   **Key Findings**: Calls `dropout_kernel_init(seed)` and `dropout_tile(0, int_probability, int_scale_factor)` with standard tile-regs-acquire/commit/wait/release pattern

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h`
   **Reason**: Core SFPU implementation for Wormhole B0
   **Key Findings**: Uses raw TTI instructions with PRNG generation via special SFPMOV mode, integer comparison for dropout decision, CC-guarded zeroing

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h`
   **Reason**: Core SFPU implementation for Blackhole
   **Key Findings**: Identical to Wormhole B0 version

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the per-face dispatch and DEST address management for Wormhole
   **Key Findings**: VectorMode::RC iterates over 4 faces, calling SFPU function once per face with SETRWC between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the per-face dispatch and DEST address management for Blackhole
   **Key Findings**: Same structure as Wormhole but uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` and `_llk_math_eltwise_unary_sfpu_done_()` helpers

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and SFPU init/start/done sequences
   **Key Findings**: ADDR_MOD_7 set to all zeros, `set_addr_mod_base()` selects ADDR_MOD 4-7 range on Wormhole

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h`
   **Reason**: `init_prng_seed` implementation
   **Key Findings**: Writes seed to PRNG_SEED config register and waits 600 SFPNOPs for PRNG initialization

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU architecture reference for instruction semantics, CC mechanism, register layout
    **Key Findings**: SFPIADD always sets CC.Res, SFPENCC mode EU_R1 disables CC, SFPMOV opcode 0x7B, stride-2 addressing model

11. **File**: `.claude/references/diagram-templates.md`
    **Reason**: CC State Machine diagram format template
    **Key Findings**: Used generalized template for CC state transition diagram

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h`
    **Reason**: p_sfpu constant definitions for LREG indices and LCONST values
    **Key Findings**: LREG0=0, LREG1=1, LREG2=2, LREG3=3, LCONST_0=9

13. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
    **Reason**: Analogous LLK dispatch file (for frac) to understand the standard dispatch pattern
    **Key Findings**: Confirmed pattern: init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::X, APPROXIMATE>()`, tile call uses `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_func, dst_index, vector_mode)`
