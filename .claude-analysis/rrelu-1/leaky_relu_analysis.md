## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(idst, param0)` where `param0` is the slope as `uint32_t` (bit-cast from `float`)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LEAKY_RELU)` in `unary_op_utils.cpp` -- switch has only a `default: return false` case |
| Template parameter (SFPU_OP_CHAIN) | none (no approx template param in dispatch) | `get_op_init_and_func()` -- `leaky_relu_tile_init()` and `leaky_relu_tile(idst, param0)` use no parameterized approx mode; the `APPROX` macro from compute config is passed through `SFPU_UNARY_ONE_PARAM_KERNEL_FN` as `calculate_lrelu<APPROX>` |
| Effective SFPU path | `APPROXIMATION_MODE=false` -- however, the `_calculate_lrelu_` function has no `if constexpr` branches that depend on `APPROXIMATION_MODE`; the template parameter is accepted but unused, so the kernel executes identically regardless of this value | The function body in `ckernel_sfpu_relu.h` contains no conditional logic based on `APPROXIMATION_MODE` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN` calls `_llk_math_eltwise_unary_sfpu_params_` directly) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (BH) |

Note: The `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` files are thin wrappers that call `_calculate_lrelu_` from the `tt_llk` implementations.

### Call Chain

1. **`leaky_relu_tile(idst, slope)`** (API header `relu.h` line 112): Wraps `MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope))`.
2. **`SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro** (macros header `llk_math_eltwise_unary_sfpu_macros.h` line 130): Expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, idst, (int)VectorMode::RC, slope)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (params header `llk_math_eltwise_unary_sfpu_params.h`): Sets DEST write address, stalls for SFPU availability, then loops over 4 faces (VectorMode::RC), calling `calculate_lrelu<APPROX>(slope)` per face and advancing DEST address between faces with `SETRWC` (WH) or `inc_dst_addr` (BH).
4. **`calculate_lrelu<APPROX>(slope)`** (wrapper in `ckernel_sfpu_relu.h` line 43 in metal/llk_api): Calls `_calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope)` with ITERATIONS=8.
5. **`_calculate_lrelu_<APPROX>(iterations, slope)`** (core SFPU implementation in `tt_llk/.../sfpu/ckernel_sfpu_relu.h`): The actual SFPU kernel that executes raw TTI instructions to compute leaky relu element-wise.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the 32x32 tile are processed. The params dispatch loops `for (int face = 0; face < 4; face++)`, calling the SFPU function once per face.
- **Operation invocation**: Per face, `calculate_lrelu<APPROX>(slope)` is called, which internally loops 8 iterations (ITERATIONS=8). Each iteration processes 32 elements (2 physical DEST rows x 16 elements/row). Total per face: 8 x 32 = 256 elements = one complete 16x16 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC/inc_dst_addr between faces). The SFPLOAD/SFPSTORE instructions use `ADDR_MOD_3` on Wormhole and `ADDR_MOD_7` on Blackhole, both configured with `dest.incr = 0` (no auto-increment via address mode). Instead, `sfpi::dst_reg++` at the end of each loop iteration explicitly advances the SFPU DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` x 2 (WH) or `math::inc_dst_addr<8>()` x 2 (BH) advances by 16 physical DEST rows to skip to the next face.

### Annotated SFPU Kernel Source

The kernel uses raw `TT_`/`TTI_` instructions with a simple CC manipulation pattern (single SFPSETCC + SFPENCC pair per iteration). This falls under **Style B** for CC documentation, but because the CC pattern is simple (a single LT0 guard), we present it as **Style A** (inline-commented source) with a CC State Machine diagram for clarity.

#### Wormhole B0 Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE is unused in this function
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);  // Load lower 16 bits of slope into LREG2 (mod0=10 = SFPLOADI_MOD0_LOWER)
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);       // Load upper 16 bits of slope into LREG2 (mod0=8 = SFPLOADI_MOD0_UPPER), completing full 32-bit float
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // Load current DEST value into LREG0 (float mode)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // CC <- (LREG0 < 0), mod1=0 = SFPSETCC_MOD1_LREG_LT0; enables lanes where input is negative
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // CC-guarded: LREG0 = LREG0 * LREG2 + 0.0 = x * slope (only for negative lanes)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // CC <- ALL_ENABLED (clear condition code)
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // Store LREG0 back to DEST (float mode)
        sfpi::dst_reg++;                                                                // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

#### Blackhole Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE is unused in this function
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);  // Load lower 16 bits of slope into LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);       // Load upper 16 bits of slope into LREG2
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);        // Load current DEST value into LREG0 (uses ADDR_MOD_7 on BH)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // CC <- (LREG0 < 0)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // CC-guarded: LREG0 = x * slope (negative lanes only)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // CC <- ALL_ENABLED
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);       // Store LREG0 back to DEST
        sfpi::dst_reg++;                                                                // Advance DEST pointer
    }
}
```

The Wormhole and Blackhole implementations are identical except for the address mode slot used (ADDR_MOD_3 on WH, ADDR_MOD_7 on BH). Both address modes are configured with `dest.incr = 0` by the init function.

#### CC State Machine -- `_calculate_lrelu_`

The function has one CC block per iteration: SFPSETCC sets CC based on sign of LREG0, SFPMUL is guarded by CC, then SFPENCC resets CC.

```
_calculate_lrelu_ -- CC State Transitions (per iteration)
================================================================

  CC State: ALL_ENABLED                   <-- initial state (or reset from previous iteration)
       |
       |  SFPLOAD LREG0 <- DEST[current]  (no CC effect) -- load input value x
       |
       v
  +-------------------------------------+
  | SFPSETCC  mod1=0 (LREG_LT0)        |
  |   src: LREG0                        |
  |                                     |
  | CC <- (LREG0 < 0)                   |
  |    = (input x is negative)          |
  +----------------+--------------------+
                   |
                   v
  CC State: ENABLED where x < 0 (negative lanes only)
       |
       |  SFPMUL LREG0 = LREG0 * LREG2 + 0.0   (CC-guarded: only negative lanes get x*slope)
       |                                          Positive lanes retain original x value
       |
       v
  +-------------------------------------+
  | SFPENCC                             |
  |                                     |
  | CC <- ALL_ENABLED                   |
  +----------------+--------------------+
                   |
                   v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE LREG0 -> DEST[current]  (no CC effect) -- write result back
       |  dst_reg++                         (no CC effect) -- advance to next 32-element group
       |
       v  (next iteration or function returns)
```

**Key CC observations:**
- SFPSETCC with `mod1=0` (`SFPSETCC_MOD1_LREG_LT0`) tests the sign of LREG0. CC is enabled (true) for SIMD lanes where the value is negative.
- SFPMUL is CC-guarded: the multiplication `x * slope` only executes on lanes where CC is true (x < 0). Lanes where CC is false (x >= 0) retain their original LREG0 value unchanged.
- SFPENCC immediately resets CC to ALL_ENABLED, ensuring the subsequent SFPSTORE writes all lanes unconditionally.
- The CC pattern repeats identically each iteration; there is no CC state carried across iterations.
- This implements the leaky ReLU formula: `f(x) = x if x >= 0, slope * x if x < 0`.

### SFPU Instructions Used

| Instruction | Count per iteration | Description |
|-------------|-------------------|-------------|
| `TT_SFPLOADI` | 2 (once before loop) | Load a 16-bit immediate value into an LREG. Used twice to load the full 32-bit float slope: first the lower 16 bits (`mod0=10`), then the upper 16 bits (`mod0=8`). |
| `TTI_SFPLOAD` | 1 | Load 32 elements from the current DEST row into an LREG. `InstrModLoadStore::DEFAULT` means float format. The address mode (`ADDR_MOD_3` on WH / `ADDR_MOD_7` on BH) specifies `dest.incr=0` (no auto-increment). |
| `TTI_SFPSETCC` | 1 | Set the condition code register based on the sign of LREG0. With `instr_mod1=0` (`SFPSETCC_MOD1_LREG_LT0`), CC is set to true for each SIMD lane where the value is negative (sign bit = 1). |
| `TTI_SFPMUL` | 1 | Fused multiply-add: `dest = src_a * src_b + src_c`. Here: `LREG0 = LREG0 * LREG2 + LCONST_0` = `x * slope + 0.0`. This instruction is CC-guarded, so only lanes where CC=true (negative x) are modified. |
| `TTI_SFPENCC` | 1 | Enable all condition codes (reset CC to ALL_ENABLED). Clears the condition set by SFPSETCC so subsequent instructions are unconditional. |
| `TTI_SFPSTORE` | 1 | Store 32 elements from an LREG back to the current DEST row. Same address mode and format as SFPLOAD. |

Total per face (8 iterations): 2 SFPLOADI + 8 x (SFPLOAD + SFPSETCC + SFPMUL + SFPENCC + SFPSTORE) = 2 + 40 = 42 instructions.
Total per tile (4 faces): 2 SFPLOADI + 4 x 40 = 162 instructions (plus inter-face SETRWC/inc_dst_addr overhead).

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| `LREG0` (index 0) | Working register: holds the input value loaded from DEST, then the result (either original x or x*slope). | Per-iteration: loaded at start, potentially modified by SFPMUL, stored at end. |
| `LREG2` (index 2) | Holds the slope parameter as a 32-bit float. Loaded once via two SFPLOADI instructions before the loop. | Entire function: loaded before the loop, read-only during iterations. |
| `LCONST_0` (index 9) | Hardware constant register holding 0.0. Used as the addend in SFPMUL (making it a pure multiply). | Permanent: hardware-provided, read-only. |
| `CC` (condition code) | Per-lane boolean flag set by SFPSETCC, cleared by SFPENCC. Guards SFPMUL execution. | Per-iteration: set, used, then cleared within each loop iteration. |
| DEST (destination register file) | Source and sink for tile data. SFPLOAD reads from it, SFPSTORE writes to it. `dst_reg++` advances the pointer. | Managed externally by the params dispatch layer; the kernel reads/writes at the current DEST offset. |

### Address Mode Configuration

The address mode used by SFPLOAD/SFPSTORE in `_calculate_lrelu_` differs between hardware generations, but both are configured identically (all zeros):

| Hardware | Address Mode Slot | Configuration |
|----------|------------------|---------------|
| **Wormhole B0** | `ADDR_MOD_3` | `srca.incr=0, srcb.incr=0, dest.incr=0` -- not explicitly programmed by the `lrelu` init (only `ADDR_MOD_7` is explicitly set); `ADDR_MOD_3` relies on default/reset state (all zeros). |
| **Blackhole** | `ADDR_MOD_7` | `srca.incr=0, srcb.incr=0, dest.incr=0` -- explicitly programmed by `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` in the init function. |

Since `dest.incr = 0`, the SFPLOAD/SFPSTORE instructions do not auto-increment the DEST address. Instead, address progression is handled explicitly by `sfpi::dst_reg++` at the end of each iteration (advancing 1 sfpi row = 2 physical DEST rows = 32 elements). Between faces, the params dispatch layer advances the DEST pointer by 16 physical rows (8 sfpi rows x 2) using `TTI_SETRWC` (WH) or `math::inc_dst_addr<8>()` x 2 (BH).

The init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::lrelu>()` calls:
1. `_init_sfpu_config_reg()` -- initializes SFPU configuration registers
2. `eltwise_unary_sfpu_configure_addrmod<SfpuType::lrelu>()` -- configures `ADDR_MOD_7` with `dest.incr=0` (default case; `lrelu` does not match any special-case `if constexpr` branches)
3. `math::reset_counters(p_setrwc::SET_ABD_F)` -- resets address counters

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "What are the SFPU instructions SFPLOADI, SFPLOAD, SFPSETCC, SFPMUL, SFPENCC, and SFPSTORE?"
   **Reason**: Needed to understand the semantics of each SFPU instruction used in the leaky relu kernel.
   **Key Findings**: DeepWiki returned 429 Too Many Requests. Information was obtained directly from source code analysis of instruction definitions in `ckernel_ops.h` and constant definitions in `sfpi_constants.h`.

2. **Query**: "How does the leaky_relu SFPU kernel work in tt-metal?"
   **Reason**: Needed to understand the CC-guarded multiplication pattern for negative value detection.
   **Key Findings**: DeepWiki returned 429 Too Many Requests. Full understanding was derived from source code analysis: SFPSETCC with mod1=0 (LREG_LT0) sets CC for negative lanes, SFPMUL is CC-guarded so only negative values get multiplied by slope, SFPENCC resets CC.

### Confluence References
No Confluence pages were consulted for this analysis. The SFPU ISA page (ID 1170505767) was not needed as all instruction semantics were determinable from source code.

### Glean References
No Glean queries were made for this analysis.
