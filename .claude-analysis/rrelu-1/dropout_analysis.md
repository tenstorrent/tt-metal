## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `DROPOUT` (defined in `unary_op_types.hpp` but NOT dispatched through `UnaryProgramFactory`)
- **Compute kernel**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- dropout uses a custom compute kernel that calls `dropout_tile(0, int_probability, int_scale_factor)` directly rather than using the `SFPU_OP_CHAIN_0` macro mechanism

**Note**: Dropout is an experimental operation with its own `DropoutProgramFactory` at `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`. It does NOT go through `UnaryProgramFactory` or `unary_op_utils`. The compute kernel explicitly calls `dropout_tile()` and `dropout_kernel_init()` from `api/compute/eltwise_unary/dropout.h`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | Hardcoded at line 240 of `dropout_program_factory.cpp`: `bool math_approx_mode = false;` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (which resolves to `false`) | The API header `dropout.h` passes `APPROX` to `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, ...)`. `APPROX` is a JIT-generated `constexpr bool` set from `ComputeConfig.math_approx_mode` (see `genfiles.cpp:394`). Since `math_approx_mode = false`, `APPROX = false`. |
| Effective SFPU path | `APPROXIMATION_MODE = false` has no effect on behavior -- the `_calculate_dropout_` function is templated on `APPROXIMATION_MODE` but does not branch on it. The kernel uses the same raw TTI instruction sequence regardless of the approximation mode value. | See `ckernel_sfpu_dropout.h` -- no `if constexpr (APPROXIMATION_MODE)` branch exists. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header uses the `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro which directly calls `_llk_math_eltwise_unary_sfpu_params_` from `llk_math_eltwise_unary_sfpu_params.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` (identical on Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (function `_llk_math_eltwise_unary_sfpu_params_`) |
| **Init Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` (function `llk_math_eltwise_unary_sfpu_init`) which calls `_init_dropout_` from the core SFPU file |

### Call Chain

1. **Compute kernel** (`dropout_kernel.cpp`): calls `dropout_tile(0, int_probability, int_scale_factor)` for each tile.
2. **API header** (`dropout.h`): `dropout_tile()` expands via `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, idst, probability, scale_factor)` which becomes `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_dropout<APPROX>, idst, (int)VectorMode::RC, probability, scale_factor)`.
3. **LLK params dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, stalls for SFPU, then loops over 4 faces calling `calculate_dropout<false>(probability, scale_factor)` per face with `SETRWC` between faces.
4. **Metal-level wrapper** (`ckernel_sfpu_dropout.h` in `hw/ckernels/`): `calculate_dropout<APPROX, 8>()` forwards to `_calculate_dropout_<false, 8>(8, probability, scale)`.
5. **Core SFPU implementation** (`ckernel_sfpu_dropout.h` in `tt_llk/`): `_calculate_dropout_` executes the raw SFPU instruction sequence that performs scaling, random number generation, comparison, and conditional zeroing.

For initialization:
1. **Compute kernel**: calls `dropout_kernel_init(seed)` once before the tile loop.
2. **API header**: `dropout_kernel_init()` expands via `SFPU_ONE_PARAM_KERNEL_INIT(dropout, sfpu::dropout_init, APPROX, seed)` which becomes `llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, APPROX>(sfpu::dropout_init<APPROX>, seed)`.
3. **LLK init** (`llk_math_eltwise_unary_sfpu_init.h`): calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` to configure SFPU registers and address modes, then calls `dropout_init<false>(seed)`.
4. **Metal-level init**: `dropout_init()` calls `_init_dropout_(seed)`.
5. **Core init** (`ckernel_sfpu_dropout.h`): `_init_dropout_` writes the seed to the PRNG configuration register and waits 600 SFPNOP cycles for the PRNG to stabilize.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full tile coverage of 1024 elements).
- **Operation invocation**: The params dispatch function `_llk_math_eltwise_unary_sfpu_params_` loops `for (int face = 0; face < 4; face++)`, calling `calculate_dropout<false>(probability, scale_factor)` once per face. Each invocation internally loops 8 iterations (ITERATIONS=8), processing 8 sfpi rows per face. Between faces, `TTI_SETRWC` is called twice (advancing by 8+8=16 physical DEST rows = 1 face stride).
- **DEST address progression**: Standard DEST progression. Within each face, `sfpi::dst_reg++` advances 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration, covering 8 iterations x 32 elements = 256 elements per face. Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice to advance the DEST write pointer by 16 physical rows. The configured address mode is `ADDR_MOD_7` with all increments = 0 (both Wormhole and Blackhole), since DEST advancement is handled explicitly by `dst_reg++` in the kernel.

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with condition code manipulation (SFPIADD with CC_GTE0, SFPMOV under CC guard, SFPENCC to reset). This qualifies as **Style B**.

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

The kernel has one CC region per iteration: SFPIADD sets CC based on a comparison, SFPMOV is CC-guarded to conditionally zero the output, and SFPENCC resets CC before storing.

```
_calculate_dropout_ -- CC State Transitions (per iteration)
================================================================

  CC State: ALL_ENABLED                   <-- initial state (or reset from previous iteration)
       |
       |  SFPLOAD  LREG0 <- DEST[current]    (no CC effect) -- load input element
       |  SFPMUL   LREG0 = LREG0 * LREG1 + 0 (no CC effect) -- scale input by scale_factor
       |
       |  SFPMOV   LREG3 <- PRNG             (no CC effect) -- generate random uint32
       |            (instr_mod1=8, lreg_c=9 triggers PRNG)
       |  SFPSETSGN LREG3 sign bit <- 0      (no CC effect) -- clear sign bit to make non-negative
       |            (instr_mod1=1: set sign from imm12_math=0)
       |
       v
  +---------------------------------------------+
  | SFPIADD  imm=0, lreg_c=LREG2,              |
  |          lreg_dest=LREG3, instr_mod1=10     |
  |                                              |
  | instr_mod1=10 = ARG_2SCOMP_LREG_DST (2)    |
  |               | CC_GTE0 (8)                  |
  |                                              |
  | Operation: LREG3 = LREG2 + (-LREG3)         |
  |          = probability - random              |
  |                                              |
  | CC <- (result >= 0)                          |
  |    = (probability >= random)                 |
  |    = (element should be DROPPED/zeroed)      |
  +---------------------+------------------------+
                        |
                        v
  CC State: ENABLED where (probability >= random)
       |
       |  SFPMOV  LREG0 <- LCONST_0, mod1=0  (CC-guarded: zero LREG0 only for lanes to drop)
       |
       v
  +---------------------------------------------+
  | SFPENCC  (0, 0, 0, 0)                      |
  |                                              |
  | CC <- ALL_ENABLED                            |
  +---------------------+------------------------+
                        |
                        v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE LREG0 -> DEST[current]    (no CC effect) -- write result back to DEST
       |  dst_reg++                           -- advance to next sfpi row
       |
       v  (next iteration or return)
```

**Key CC observations:**
- SFPIADD with `instr_mod1=10` (`ARG_2SCOMP_LREG_DST | CC_GTE0`) performs integer subtraction `probability - random` and sets CC for lanes where the result is non-negative (i.e., the random number is less than or equal to the probability threshold). These are the lanes that should be dropped (zeroed).
- SFPMOV with `instr_mod1=0` is CC-guarded: it only writes `LCONST_0` (zero) into `LREG0` for lanes where CC is enabled (lanes to drop). For lanes where CC is not enabled (random > probability), `LREG0` retains the scaled input value.
- SFPENCC resets CC to ALL_ENABLED before the SFPSTORE, ensuring the store writes to all lanes unconditionally. Lanes that were dropped now contain zero; lanes that survived contain the scaled value.
- CC state does not persist across iterations -- SFPENCC resets it at the end of each iteration.

### SFPU Instructions Used

| Instruction | Count | Description |
|-------------|-------|-------------|
| `TT_SFPLOADI` | 4 (before loop) | Load 16-bit immediate into an LREG. Used to construct full 32-bit values for `scale` (LREG1) and `probability` (LREG2) by loading lower then upper 16 bits. `instr_mod0=10` loads lower 16 bits; `instr_mod0=8` loads upper 16 bits. |
| `TTI_SFPLOAD` | 1 (per iteration) | Load data from DEST register into LREG0. `sfpu_addr_mode=3` uses the current DEST write address; `dest_reg_addr=0` is the offset. Loads 32 elements (one sfpi row = 2 physical DEST rows). |
| `TTI_SFPMUL` | 1 (per iteration) | Fused multiply-add: `LREG0 = LREG0 * LREG1 + LCONST_0`. Since `LCONST_0 = 0.0`, this is effectively `input * scale_factor`. |
| `TTI_SFPMOV` | 2 (per iteration) | Move/copy between registers. First use (`instr_mod1=8, lreg_c=9`): special PRNG mode that generates a pseudorandom uint32 into LREG3. Second use (`instr_mod1=0`): CC-guarded copy of `LCONST_0` (zero) into LREG0 for dropped lanes. |
| `TTI_SFPSETSGN` | 1 (per iteration) | Set the sign bit of LREG3. With `imm12_math=0` and `instr_mod1=1`, clears the sign bit of the random number to ensure it is non-negative for comparison with probability. |
| `TTI_SFPIADD` | 1 (per iteration) | Integer add with 2's complement and CC update. `instr_mod1=10 = ARG_2SCOMP_LREG_DST(2) | CC_GTE0(8)`: computes `LREG3 = LREG2 + (-LREG3)` = `probability - random`, and sets CC for lanes where result >= 0. |
| `TTI_SFPENCC` | 1 (per iteration) | Enable Condition Code reset. Clears CC back to ALL_ENABLED so subsequent SFPSTORE writes unconditionally. |
| `TTI_SFPSTORE` | 1 (per iteration) | Store LREG0 back to DEST at the current address. `sfpu_addr_mode=3` uses the current DEST address; `dest_reg_addr=0` is the offset. |

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| **LREG0** | Working register: holds input data loaded from DEST, then the scaled value, then either the scaled value (kept) or zero (dropped). Written back to DEST at end of iteration. | Per-iteration |
| **LREG1** | Scale factor: loaded as a 32-bit float via two `SFPLOADI` calls before the loop. Holds `scale_factor` for the entire tile face. | Entire face (loaded once before loop) |
| **LREG2** | Probability threshold: loaded as a 32-bit unsigned integer via two `SFPLOADI` calls before the loop. Holds `probability` for the entire tile face. | Entire face (loaded once before loop) |
| **LREG3** | Random number: receives PRNG output from `SFPMOV` (PRNG mode), then has sign bit cleared by `SFPSETSGN`, then is consumed by `SFPIADD` for comparison. Destructively overwritten by the subtraction result. | Per-iteration |
| **LCONST_0** | Hardware constant `0.0`. Used as the addend in `SFPMUL` (making it a pure multiply) and as the zero value for dropping elements via CC-guarded `SFPMOV`. | Hardware constant (always available) |
| **DEST** | Source and destination for tile data. `SFPLOAD` reads from DEST into LREG0; `SFPSTORE` writes LREG0 back to DEST. Address auto-advanced by `dst_reg++` (1 sfpi row per iteration). | Tile lifetime |

### Address Mode Configuration

Dropout configures only `ADDR_MOD_7` during initialization (via `eltwise_unary_sfpu_configure_addrmod<SfpuType::dropout>()`). The configuration is identical on both Wormhole and Blackhole:

```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

Dropout does NOT set `ADDR_MOD_6` or any other address mode -- it is not in any of the special-case branches in `eltwise_unary_sfpu_configure_addrmod`.

The DEST address auto-increment is not used by this kernel. Instead, DEST advancement is handled explicitly:
- **Within a face**: `sfpi::dst_reg++` in the C++ source advances the sfpi address by 1 (= 2 physical DEST rows = 32 elements) after each iteration.
- **Between faces**: The `_llk_math_eltwise_unary_sfpu_params_` dispatch calls `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces, advancing the DEST pointer by 16 physical rows (one face).
- **SFPLOAD/SFPSTORE addr mode**: Both use `sfpu_addr_mode=3` with `dest_reg_addr=0`, reading from / writing to the current DEST address as managed by the write pointer.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "What are the SFPU instructions SFPLOADI, SFPLOAD, SFPMUL, SFPMOV, SFPSETSGN, SFPIADD, SFPENCC, and SFPSTORE? Explain their roles, operands, and what condition code manipulation SFPIADD and SFPENCC perform. Also explain the SFPMOV instruction when used with instr_mod1=8 and lreg_c=9 to generate pseudorandom numbers."
   **Reason**: Needed to understand the semantics and operand encoding of each SFPU instruction used in the dropout kernel, particularly the PRNG behavior of SFPMOV and the condition code mechanics of SFPIADD/SFPENCC.
   **Key Findings**: DeepWiki was unavailable (429 Too Many Requests). Information was obtained by direct codebase analysis: reading instruction macro definitions in `ckernel_ops.h`, register constants in `ckernel_instr_params.h`, and SFPIADD mod constants in `sfpi_constants.h`.

### Confluence References
No Confluence pages were consulted for this analysis.

### Glean References
No Glean searches were performed for this analysis.
