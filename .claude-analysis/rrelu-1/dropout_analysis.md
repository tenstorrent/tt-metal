## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `DROPOUT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- Dropout does **not** use the standard `UnaryProgramFactory` or `SFPU_OP_CHAIN_0` dispatch. It has its own dedicated program factory (`DropoutProgramFactory` at `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`) and a standalone compute kernel that directly calls `dropout_tile(0, int_probability, int_scale_factor)` and `dropout_kernel_init(seed)`.

**Important note**: While `UnaryOpType::DROPOUT` is defined in `unary_op_types.hpp`, it is **not** handled by `get_op_init_and_func_default()` or `get_op_init_and_func_parameterized()` in `unary_op_utils.cpp`. Dropout is implemented as an experimental operation with its own end-to-end pipeline. The `DROPOUT` enum value exists for type identification but the SFPU dispatch bypasses the standard unary chain entirely.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | Hardcoded in `DropoutProgramFactory::create()` at line 240: `bool math_approx_mode = false;` |
| Template parameter (SFPU kernel) | `APPROXIMATION_MODE` (unused in kernel body) | `_calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>()` -- the template parameter is declared but never read within the function body |
| Effective SFPU path | The single code path is always taken regardless of `APPROXIMATION_MODE` | The kernel body has no `if constexpr (APPROXIMATION_MODE)` branches; the template parameter is vestigial |

### SFPU Abstraction Layers
Dropout uses a non-standard abstraction path. The API header (`api/compute/eltwise_unary/dropout.h`) has been removed from this repo (deep-nuked), but the core SFPU implementation and compute kernel remain.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` [REMOVED in deep-nuke -- file does not exist in this repo] |
| **LLK Dispatch** | This level of abstraction doesn't exist -- dropout bypasses `llk_math_eltwise_unary_sfpu.h` and has no dedicated `llk_math_*_dropout.h` file |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` (identical for Blackhole: `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h`) |
| **Parameters Dispatch** | This level of abstraction doesn't exist -- the removed API header would have directly called `_calculate_dropout_` with `VectorMode::RC`, iterating over 4 faces via `_llk_math_eltwise_unary_sfpu_params_` |

### Call Chain
The intended call chain (reconstructed from existing code and the compute kernel) is:

1. **Compute kernel** (`dropout_kernel.cpp`): calls `dropout_kernel_init(seed)` and `dropout_tile(0, int_probability, int_scale_factor)`.
2. **API header** (`dropout.h`, removed): `dropout_kernel_init(seed)` would call `_init_dropout_(seed)` on the MATH thread. `dropout_tile(idst, prob, scale)` would call `_llk_math_eltwise_unary_sfpu_start_<>()`, then invoke `_llk_math_eltwise_unary_sfpu_params_<false>()` with `_calculate_dropout_<APPROXIMATION_MODE, 8>(8, probability, scale)` as the callable, then `_llk_math_eltwise_unary_sfpu_done_()`.
3. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): iterates over 4 faces in `VectorMode::RC`, calling the SFPU function once per face (8 iterations each), with `SETRWC` between faces.
4. **Core SFPU implementation** (`ckernel_sfpu_dropout.h`): `_calculate_dropout_<APPROXIMATION_MODE, 8>(8, probability, scale)` executes the dropout SFPU microcode for 8 iterations per face.
5. **PRNG initialization** (`ckernel.h`): `init_prng_seed(seed)` writes to the `PRNG_SEED_Seed_Val` config register and waits 600 NOPs for the PRNG to stabilize.

### Parameters Dispatch Summary
Since the LLK dispatch layer does not exist for dropout, the following is reconstructed from the compute kernel pattern and the standard `_llk_math_eltwise_unary_sfpu_params_` template:

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (standard for element-wise unary operations).
- **Operation invocation**: The core SFPU function `_calculate_dropout_` is called once per face with `iterations=8`. The function loops 8 times per face, processing 32 elements per iteration (2 physical DEST rows x 16 elements/row).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration within the loop, `SETRWC` between faces). The kernel uses `sfpi::dst_reg++` at the end of each iteration. Raw SFPLOAD/SFPSTORE use `sfpu_addr_mode=3` (RWC auto-increment mode) with `dest_reg_addr=0`, which reads/writes at the current DEST RWC pointer.
- **PRNG initialization**: `_init_dropout_` calls `init_prng_seed(seed)` which writes to the PRNG hardware seed register (`PRNG_SEED_Seed_Val_ADDR32`) and executes 600 `SFPNOP` instructions to allow the PRNG state to settle.

### Annotated SFPU Kernel Source

The kernel uses raw `TT_`/`TTI_` instructions with CC manipulation via SFPIADD. Since the CC pattern is relatively simple (single SFPIADD sets CC, one guarded SFPMOV, then SFPENCC reset), Style B is used for maximum clarity about the CC flow.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h

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

The CC flow uses a single SFPIADD to set per-lane condition codes, one guarded SFPMOV to conditionally zero elements, then SFPENCC to restore all-lane execution before the store.

```
_calculate_dropout_ -- CC State Transitions (per iteration)
================================================================

  CC State: ALL_ENABLED (CC.En=0)              <-- initial state

       |  TT_SFPLOADI LREG1, lo16, scale[15:0]     (no CC effect, before loop) -- load scale lower 16 bits
       |  TT_SFPLOADI LREG1, hi16, scale[31:16]    (no CC effect, before loop) -- load scale upper 16 bits
       |  TT_SFPLOADI LREG2, lo16, prob[15:0]      (no CC effect, before loop) -- load probability lower 16 bits
       |  TT_SFPLOADI LREG2, hi16, prob[31:16]     (no CC effect, before loop) -- load probability upper 16 bits

  == Per-iteration loop (8 iterations per face, 4 faces per tile) ==

       |  SFPLOAD LREG0 <- DEST[current_row]        (no CC effect) -- load tile data
       |  SFPMUL  LREG0 = LREG0 * LREG1 + 0.0      (no CC effect) -- scale the input value
       |
       |  SFPMOV  LREG3 = PRNG()                     (no CC effect) -- generate pseudorandom uint32
       |                                              instr_mod1=8, lreg_c=9 triggers PRNG mode
       |  SFPSETSGN LREG3, sign=0                    (no CC effect) -- clear sign bit to make unsigned
       |                                              instr_mod1=1 sets sign from imm12[0]=0
       v
  +---------------------------------------------+
  | SFPIADD  LREG3 = LREG2 - LREG3             |
  |   instr_mod1 = 10 = 0b1010                  |
  |     bit[1:0] = 2: subtract (VC - VD)        |
  |     bit[3]   = 1: invert CC result           |
  |                                              |
  |   Computation: LREG3 = probability - rand    |
  |   CC.En <- 1 (always enabled by SFPIADD)     |
  |   CC.Res <- inverted(result < 0)             |
  |          = (result >= 0)                     |
  |          = (probability >= rand)             |
  +-------------------+--------------------------+
                      |
                      v
  CC State: ENABLED where probability >= rand (element should be dropped)
       |
       |  SFPMOV  LREG0 = LCONST_0 (0.0)    (CC-guarded: zero out elements where probability >= rand)
       |
       v
  +---------------------------------------------+
  | SFPENCC  instr_mod1=0 (EU_R1)               |
  |   imm12=0                                   |
  |                                              |
  |   CC.En unchanged (still 1), CC.Res <- 1    |
  |   All lanes now active (En=1, Res=1)         |
  +-------------------+--------------------------+
                      |
                      v
  CC State: ALL_ACTIVE (CC.En=1, CC.Res=1 for all lanes)
       |
       |  SFPSTORE  DEST[current_row] <- LREG0  (all lanes) -- store result
       |  dst_reg++                              -- advance to next SFPI row
       |
       v  (next iteration or function returns)

  == End of per-iteration loop ==
```

**Key CC observations:**
- SFPIADD is the sole CC-modifying instruction. It computes `probability - rand` as a signed integer subtraction and sets CC based on the result sign, inverted by `InstrMod[3]=1`.
- The inversion means CC.Res=1 when `probability >= rand` -- these are the lanes to be dropped (zeroed).
- Only SFPMOV is CC-guarded. For lanes where `probability >= rand`, it overwrites the scaled value in LREG0 with 0.0 (from the fixed constant register `LCONST_0 = 9`).
- SFPENCC with mode `EU_R1` sets CC.Res=1 for all lanes while keeping CC.En=1, making all lanes active for the unconditional SFPSTORE that follows.
- The CC pattern repeats identically for each of the 8 iterations per face. There is no cross-iteration CC state dependency.

### SFPU Instructions Used

| Instruction | Opcode | Count per iter | Description |
|-------------|--------|---------------|-------------|
| `SFPLOADI` (via `TT_SFPLOADI`) | 0x71 | 0 (4 total, before loop) | Load 16-bit immediate into LREG. Used to construct 32-bit `scale` and `probability` values in LREG1 and LREG2 via lo16/hi16 pairs. |
| `SFPLOAD` (via `TTI_SFPLOAD`) | 0x70 | 1 | Load from DEST row into LREG0. Format: IMPLIED (instr_mod0=0), RWC address mode (sfpu_addr_mode=3). |
| `SFPMUL` (via `TTI_SFPMUL`) | 0x86 | 1 | Multiply LREG0 by LREG1 (scale factor), add LCONST_0 (0.0). Result: `input * scale`. |
| `SFPMOV` (via `TTI_SFPMOV`, PRNG mode) | 0x7C | 1 | With `instr_mod1=8` and `lreg_c=9`: generates a pseudorandom uint32 into LREG3 using the hardware PRNG. |
| `SFPSETSGN` (via `TTI_SFPSETSGN`) | 0x8C | 1 | Set sign bit of LREG3 to 0 (clear sign). `instr_mod1=1` means set sign from `imm12[0]`, which is 0. Makes the random number unsigned for comparison. |
| `SFPIADD` (via `TTI_SFPIADD`) | 0x79 | 1 | Integer subtract: `LREG3 = LREG2 - LREG3` (probability - rand). Sets CC with inverted sign (`InstrMod[3]=1`): CC.Res=1 when probability >= rand. |
| `SFPMOV` (via `TTI_SFPMOV`, conditional) | 0x7C | 1 | CC-guarded move: copies LCONST_0 (0.0) into LREG0. Only executes for lanes where probability >= rand (element is dropped). |
| `SFPENCC` (via `TTI_SFPENCC`) | 0x8A | 1 | Reset CC state: mode `EU_R1` (instr_mod1=0) keeps CC.En=1, sets CC.Res=1 for all lanes. All lanes become active. |
| `SFPSTORE` (via `TTI_SFPSTORE`) | 0x72 | 1 | Store LREG0 to DEST row. Format: IMPLIED (instr_mod0=0), RWC address mode (sfpu_addr_mode=3). Unconditional (all lanes active after SFPENCC). |
| `SFPNOP` (via `TTI_SFPNOP`, in `init_prng_seed`) | 0x8E | 600 (init only) | No-operation. Used in `init_prng_seed()` to wait for the PRNG seed to propagate through the hardware. |

### SFPU Register Usage

| Register | Name | Purpose |
|----------|------|---------|
| LREG0 | `p_sfpu::LREG0` (0) | Working register: holds the tile element loaded from DEST, then the scaled value (`input * scale`), and finally either the scaled value or 0.0 (after conditional zeroing). Stored back to DEST. |
| LREG1 | `p_sfpu::LREG1` (1) | Scale factor: holds the 32-bit float `scale = 1 / (1 - probability)`. Loaded once before the loop via two SFPLOADI instructions (lo16 + hi16). Constant across all iterations. |
| LREG2 | `p_sfpu::LREG2` (2) | Probability threshold: holds the 32-bit integer `probability` (range 0 to INT_MAX). Loaded once before the loop. Used as the comparison value in SFPIADD. Constant across all iterations. |
| LREG3 | `p_sfpu::LREG3` (3) | Random number: receives the PRNG output from SFPMOV(PRNG mode), then has its sign bit cleared by SFPSETSGN. Used as the subtrahend in the SFPIADD comparison. Overwritten each iteration. |
| LCONST_0 | `p_sfpu::LCONST_0` (9) | Hardware fixed constant: 0.0. Used as the addend in SFPMUL (to make it a pure multiply) and as the zero value for dropped elements in the conditional SFPMOV. |
| DEST[row] | Current DEST RWC row | Source and destination for tile data. SFPLOAD reads from it; SFPSTORE writes to it. Advanced by `dst_reg++` each iteration. |
| `PRNG_SEED_Seed_Val` | Config register | Hardware PRNG seed register. Written by `init_prng_seed()` during initialization. Not directly accessed during the main kernel loop. |

### Address Mode Configuration

The dropout SFPU kernel (`_calculate_dropout_`) does **not** configure any `addr_mod_t` or `ADDR_MOD_N` registers itself. The SFPLOAD and SFPSTORE instructions use `sfpu_addr_mode=3` (RWC mode), which uses the DEST Read/Write Counter for addressing. The `dst_reg++` at the end of each iteration advances the SFPI DEST pointer by 1 sfpi row (= 2 physical DEST rows), handling the stride-2 addressing.

The address mode configuration is expected to be set by the calling layer (the removed API header, which would call `_llk_math_eltwise_unary_sfpu_start_` and `_llk_math_eltwise_unary_sfpu_done_` which set `math::set_addr_mod_base()` and `math::clear_addr_mod_base()`). The standard `eltwise_unary_sfpu_configure_addrmod` function (in `llk_math_eltwise_unary_sfpu.h`) sets `ADDR_MOD_7` with all increments = 0, which is the default for operations that handle their own DEST pointer advancement (as dropout does via `dst_reg++`).

Since dropout does not appear in the `SfpuType` enum checks within `eltwise_unary_sfpu_configure_addrmod`, it would use only `ADDR_MOD_7` with `{srca.incr=0, srcb.incr=0, dest.incr=0}`. This is identical across Wormhole and Blackhole.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checked whether DROPOUT is handled by standard unary dispatch
   **Key Findings**: DROPOUT is not present in `get_op_init_and_func_default()`, `get_op_init_and_func_parameterized()`, or `get_op_approx_mode()`. It is not dispatched through the standard UnaryProgramFactory pipeline.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Verified UnaryOpType::DROPOUT enum exists
   **Key Findings**: `DROPOUT` is defined at line 104 in the UnaryOpType enum.

3. **File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`
   **Reason**: Understanding the dropout-specific program factory
   **Key Findings**: Dropout has its own dedicated program factory. `math_approx_mode = false` is hardcoded. Probability is converted to int via `(double)INT_MAX * prob`. Scale is bit-cast from float to uint32.

4. **File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
   **Reason**: Understanding the compute kernel that dispatches SFPU work
   **Key Findings**: Calls `dropout_kernel_init(seed)` for PRNG initialization, then `dropout_tile(0, int_probability, int_scale_factor)` per tile. Includes `api/compute/eltwise_unary/dropout.h` (removed file).

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Identical between WH and BH. Uses raw TTI instructions. APPROXIMATION_MODE template parameter is declared but unused. Algorithm: scale input, generate PRNG, compare with threshold, conditionally zero.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h`
   **Reason**: Understanding `init_prng_seed()` function
   **Key Findings**: Writes seed to `PRNG_SEED_Seed_Val_ADDR32` config register, then waits 600 SFPNOP cycles for PRNG hardware to settle.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h`
   **Reason**: Understanding p_sfpu register constants
   **Key Findings**: LREG0-3 = 0-3, LCONST_0 = 9 (hardware fixed constant 0.0), LCONST_1 = 10.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_load_config.h`
   **Reason**: Understanding SFPLOADI instr_mod0 values
   **Key Findings**: `instr_mod0=10` writes lower 16 bits, `instr_mod0=8` writes upper 16 bits. Used to construct 32-bit values via two SFPLOADI instructions.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understanding the standard SFPU unary dispatch framework
   **Key Findings**: Provides `_llk_math_eltwise_unary_sfpu_start_`, `_llk_math_eltwise_unary_sfpu_done_`, and `eltwise_unary_sfpu_configure_addrmod`. ADDR_MOD_7 is set with all increments = 0.

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
    **Reason**: Understanding the parameters dispatch pattern used for unary SFPU operations
    **Key Findings**: `_llk_math_eltwise_unary_sfpu_params_` iterates over faces with SETRWC between them, calling the SFPU function once per face. VectorMode::RC processes all 4 faces.

11. **File**: `tests/tt_metal/tt_metal/llk/test_dropout_sfpu_compute.cpp`
    **Reason**: Understanding the test harness and how dropout parameters are prepared
    **Key Findings**: Uses `SFPU_OP_DROPOUT_INCLUDE` define. Probability converted via `probability * (double)INT_MAX`. Scale = `1.0f / (1.0f - probability)`. Verifies both dropout rate and correct scaling of non-dropped elements.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative SFPU hardware reference
    **Key Findings**: SFPIADD always sets CC.Res (unless CC update disabled). InstrMod[3] inverts CC result. SFPENCC mode 0 (EU_R1) keeps Enable unchanged, sets Result=1. SFPMOV opcode 0x7C.

13. **File**: `.claude/references/diagram-templates.md`
    **Reason**: Template for CC state machine diagrams
    **Key Findings**: Used the generalized CC state machine template for the dropout CC flow diagram.
