# Wormhole → Blackhole SFPU Porting Guide

This guide provides translation patterns for porting SFPU kernels from Wormhole to Blackhole.

---

## ⚠️ MANDATORY: Verify All Claims Against tt-isa-documentation ⚠️

**Before relying on ANY information in this guide:**
1. Cross-reference with `tenstorrent/tt-isa-documentation` via deepwiki
2. Check BOTH Wormhole AND Blackhole codebases for actual patterns
3. NEVER assume a feature doesn't exist without verification

This guide is a starting point, not authoritative truth. The ISA documentation and actual code are the sources of truth.

---

## Overview

| Aspect | Wormhole | Blackhole |
|--------|----------|-----------|
| **API** | `sfpi::` C++ library | `sfpi::` C++ library (same) |
| **Style** | High-level, operator overloading | High-level, operator overloading |
| **Differences** | Fewer features | More LUT modes, macro instructions |

---

## Key Differences

Blackhole and Wormhole share the same SFPI C++ library, so most code is portable. Key differences:

1. **More SFPU instructions** - Blackhole adds `SFPSHFT2`, `SFPLUTFP32`, `SFPLE`, `SFPGT`, `SFPMUL24`, `SFPARECIP`
2. **Enhanced LUT support** - Blackhole has `lut2`, `lut2_sign` for 6-piece piecewise linear approximations
3. **Macro instructions** - Blackhole supports `SFPLOADMACRO` for complex instruction sequences
4. **Replay buffer API** - Different API: Blackhole uses `load_replay_buf()`, Wormhole uses `lltt::replay()`/`lltt::record()`

---

## Include Files

**Wormhole/Blackhole** (same):
```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

---

## Core Operations (Identical)

### Load from Dest Register

```cpp
// Both Wormhole and Blackhole
sfpi::vFloat val = sfpi::dst_reg[0];
```

### Store to Dest Register

```cpp
// Both Wormhole and Blackhole
sfpi::dst_reg[0] = result;
```

### Increment Dest Pointer

```cpp
// Both Wormhole and Blackhole
sfpi::dst_reg++;
```

### Conditionals

```cpp
// Both Wormhole and Blackhole
v_if (val < 0.0f)
{
    val = 0.0f;
}
v_endif;
```

---

## Blackhole-Specific Features

### 6-Piece LUT Approximation

Blackhole introduces `lut2` for piecewise linear approximations:

```cpp
// Initialize LUT coefficients in _init_ function
_sfpu_load_imm32_(0, 0x32F433D9);  // A0, A1
_sfpu_load_imm32_(4, 0x23C89018);  // B0, B1
_sfpu_load_imm32_(1, 0x300A318A);  // A2, A3
_sfpu_load_imm32_(5, 0x30272BAA);  // B2, B3
_sfpu_load_imm32_(2, 0x7C002A35);  // A4, A5
_sfpu_load_imm32_(6, 0x37ff34CC);  // B4, B5

// Use in calculation
sfpi::vFloat result = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode);
```

### Macro Instructions (Blackhole Only)

For high-performance operations, use macro instruction sequences:

```cpp
// Configure macro sequence
TTI_SFPLOADI(0, 0xA, 0x85EF);
TTI_SFPLOADI(0, 0x8, 0x731E);
TTI_SFPCONFIG(0, 4, 0);  // Load into Macro Sequence Register 0

// Execute macro
TTI_SFPLOADMACRO(lreg_ind, instr_mod0, addr_mode, dest_offset);
```

### Replay Buffer (Blackhole Only)

Record and replay instruction sequences:

```cpp
// Record 32 instructions into replay buffer
lltt::record(0, 32);
// ... instructions ...

// Replay recorded instructions
lltt::replay(0, 32);
```

---

## Common Patterns

### Pattern: LUT-Based Approximation

Both architectures support LUT, but Blackhole has enhanced support:

**Wormhole** (basic LUT):
```cpp
sfpi::vFloat result = lut(val, l0, l1, imm2);
```

**Blackhole** (6-piece LUT):
```cpp
sfpi::vFloat result = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode);
result = lut2_sign(val, l0, l1, l2, l4, l5, l6);  // With sign handling
```

### Pattern: Save/Restore LUT Registers

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_operation_(const int iterations)
{
    // Save LUT registers
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        // ... computation using LUT ...
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    // Restore LUT registers
    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}
```

### Pattern: Init Function for Constants

```cpp
template <bool APPROXIMATION_MODE>
inline void _init_operation_()
{
    // Set programmable constants
    sfpi::vConstFloatPrgm0 = 0.5f;

    // Load LUT coefficients
    _sfpu_load_imm32_(0, 0x37E7322B);
    _sfpu_load_imm32_(4, 0xB12286D8);
    // ... more coefficients ...
}
```

### Pattern: Exponential with Reciprocal

```cpp
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _calculate_exponential_body_(sfpi::vFloat in)
{
    sfpi::vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        // Fast approximation using bit manipulation
        constexpr int FRAC_BITS = 3;
        constexpr std::uint32_t SP_BIAS = 127 << FRAC_BITS;

        sfpi::vFloat vConstLn2Recip = sfpi::vConstFloatPrgm0;
        sfpi::vFloat conv = in * vConstLn2Recip;
        sfpi::vInt c23_73 = p_exp::C23_73;
        sfpi::vInt tmp = sfpi::reinterpret<sfpi::vInt>(conv) - c23_73;
        tmp += SP_BIAS;
        out = sfpi::reinterpret<sfpi::vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Accurate series expansion
        out = _sfpu_exp_(sfpi::setsgn(in, 0));
        v_if (in < 0)
        {
            out = _sfpu_reciprocal_<2>(out);
        }
        v_endif;
    }

    return out;
}
```

---

## Complete Translation Example: GELU

### Blackhole GELU (LUT-Based Approximation)

```cpp
#pragma once
#include "ckernel_sfpu_cdf.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
inline void _calculate_gelu_appx_()
{
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat half = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = l2;
    sfpi::l_reg[sfpi::LRegs::LReg4] = l4;
    sfpi::l_reg[sfpi::LRegs::LReg5] = l5;
    sfpi::l_reg[sfpi::LRegs::LReg6] = l6;
}

template <bool APPROXIMATION_MODE>
inline void _init_gelu_()
{
    sfpi::vConstFloatPrgm0 = 0.5f;

    // LUT coefficients for GELU approximation
    _sfpu_load_imm32_(0, 0x37E7322B);
    _sfpu_load_imm32_(4, 0xB12286D8);
    _sfpu_load_imm32_(1, 0x38E138F3);
    _sfpu_load_imm32_(5, 0xB437B479);
    _sfpu_load_imm32_(2, 0x38003852);
    _sfpu_load_imm32_(6, 0x7c00afa4);
}

} // namespace ckernel::sfpu
```

---

## Key Translation Rules

### For SFPU Kernels
1. **Most code is portable**: The SFPI library is shared between Wormhole and Blackhole.
2. **Use Blackhole enhancements when available**: Prefer `lut2` over `lut` for better accuracy.
3. **Leverage macro instructions**: For high-performance operations, use Blackhole's macro instruction support.
4. **Check instruction availability**: Some instructions (like `SFPARECIP`) are Blackhole-only.

### For ALL Kernel Types (CRITICAL)
1. **BH-first design**: Start from existing BH patterns, use WH only for understanding semantics.
2. **Template params from BH**: Derive template params from BH test harness + parent file, NOT from WH.
3. **Init/uninit symmetry**: `_uninit_` must reverse what `_init_` changes.
4. **Don't over-port**: Drop WH features not referenced in BH test/parent (e.g., `diagonal` mode, extra ADDR_MODs).
5. **Verify against existing BH kernels**: Read the closest BH kernel of the same type line-by-line.
6. **Test harness is the API contract**: The test file defines what signatures BH expects.

---

## Existing Blackhole Examples

Use these as reference implementations:

| Operation | File |
|-----------|------|
| Sigmoid | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sigmoid.h` |
| Exp | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` |
| GELU | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h` |
| Recip | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h` |
| Tanh | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_tanh.h` |
| Binary | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h` |
| Comp | `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_comp.h` |

---

## Unpack/Pack Kernel Porting (Blackhole-Specific)

**CRITICAL**: Unpack and pack kernels have significant architectural differences between Wormhole and Blackhole. Do NOT assume patterns transfer directly.

### Key Differences from Wormhole

| Aspect | Wormhole | Blackhole |
|--------|----------|-----------|
| MOP Template | `ckernel_template` AND `ckernel_unpack_template` | Same (both available) |
| Replay Functionality | Has replay via `lltt::replay()` and `lltt::record()` | Has `load_replay_buf` with lambdas + `lltt::replay_insn()` |
| Config Writes | `TTI_REG2FLOP` common | `TTI_STALLWAIT` + `TTI_WRCFG` preferred |
| Address Increment | Loop-based or manual | `TTI_CFGSHIFTMASK` in replay buffers |
| Context Handling | Similar pattern | Explicit `unp_cfg_context` register selection |

**IMPORTANT**: Both architectures have replay capabilities, but the API differs:
- **Wormhole**: Uses `lltt::replay()` and `lltt::record()` directly (see `ckernel_sfpu_reduce.h`)
- **Blackhole**: Uses `load_replay_buf()` function with lambda syntax for loading instruction sequences

**⚠️ WARNING**: NEVER assume features don't exist in an architecture without checking tt-isa-documentation AND the codebase. Always verify with:
```bash
grep -r "replay" tt_llk_wormhole_b0/
grep -r "replay" tt_llk_blackhole/
```

### Blackhole Unpack Pattern: Replay Buffer + `ckernel_unpack_template`

Many Blackhole unpack kernels use replay buffers instead of simple MOP loops:

```cpp
// 1. Load instructions into replay buffer
load_replay_buf(
    0,                    // buffer offset
    replay_buf_len,       // number of instructions
    []                    // lambda containing TTI instructions
    {
        TTI_UNPACR(SrcA, ...);
        TTI_CFGSHIFTMASK(1, 0b011, 32-1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
        TTI_UNPACR(SrcA, ...);
        TTI_CFGSHIFTMASK(1, 0b011, 32-1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
    });

// 2. Create unpack template with replay instructions
ckernel_unpack_template tmp = ckernel_unpack_template(
    false,                                        // src B enable
    false,                                        // halo enable
    lltt::replay_insn(0, replay_buf_half_len),   // context 0 instruction
    0, 0, 0,                                      // A1-A3 (not used without halo)
    lltt::replay_insn(half_len, replay_buf_half_len),  // context 1 instruction
    0, 0);                                        // B instructions
tmp.program();
```

### Context-Based Address Configuration

Blackhole requires explicit context-based register selection:

```cpp
// CORRECT Blackhole pattern
if (0 == unp_cfg_context)
{
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
}
else
{
    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
}

// WRONG: Using abstraction that hides context
_llk_unpack_configure_single_address_(address, cfg);  // May not handle context
```

### Config Write Sequence

**Blackhole requires stalls before config writes:**

```cpp
// CORRECT Blackhole pattern
TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);
TTI_NOP;  // May be needed for timing

// AVOID in Blackhole (works in Wormhole)
TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);
```

### Tile Dimension Configuration

**CRITICAL**: Blackhole tilize/untilize modes require explicit tile dimension setup:

```cpp
// Set tile X dimension
const std::uint32_t Tile_x_dim = face_r_dim * num_faces * FACE_C_DIM;
cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));

// Set tile Z dimension
const std::uint32_t Tile_z_dim = 1;  // For tilize mode
cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xffff0000>(0 | (Tile_z_dim << 16));

// Set unpacker x-end
TT_SETADCXX(p_setadc::UNP0, Tile_x_dim - 1, 0x0);
```

### CFGSHIFTMASK for Address Auto-Increment

Blackhole uses `CFGSHIFTMASK` to auto-increment addresses in replay buffers:

```cpp
// In init: Set scratch register with increment value
TTI_WRCFG(p_gpr_unpack::TMP0, 0, SCRATCH_SEC0_val_ADDR32);

// In replay buffer: Add scratch value to address register
TTI_CFGSHIFTMASK(
    1,          // use scratch register
    0b011,      // operation: add
    32 - 1,     // shift amount (full 32-bit)
    0,          // mask offset
    0b11,       // scratch select (based on thread ID)
    THCON_SEC0_REG3_Base_address_ADDR32  // target register
);
```

### Helper Functions to Use

```cpp
// Use config_unpacker_x_end template instead of direct TT_SETADCXX
config_unpacker_x_end<p_setadc::UNP_A>(face_r_dim);
config_unpacker_x_end<p_setadc::UNP_B>(1);  // Single row for tilizeA_B

// Y stride configuration for row-by-row processing
std::uint32_t unpA_ch1_y_stride = SCALE_DATUM_SIZE(unpack_dst_format, FACE_C_DIM);
cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(unpA_ch1_y_stride);
```

### Blackhole Unpack Reference Files

**ALWAYS check these files for patterns before implementing unpack kernels:**

| File | Pattern Demonstrated |
|------|---------------------|
| `llk_unpack_untilize.h` | `load_replay_buf`, `ckernel_unpack_template`, state save/restore |
| `llk_unpack_AB_matmul.h` | Replay buffer with `TTI_WRCFG` address updates |
| `llk_unpack_A.h` | Context-based addressing, `config_unpacker_x_end` |
| `llk_unpack_AB_reduce.h` | Multiple MOP configurations, pool-type handling |
| `llk_unpack_common.h` | `_llk_unpack_hw_configure_`, tile dimension setup |
| `cunpack_common.h` | Helper functions like `config_unpacker_x_end` |

### Search Commands for Finding Patterns

```bash
# Find replay buffer usage
grep -r "load_replay_buf" tt_llk_blackhole/llk_lib/

# Find ckernel_unpack_template usage
grep -r "ckernel_unpack_template" tt_llk_blackhole/

# Compare instruction usage
grep -c "TTI_WRCFG" tt_llk_blackhole/llk_lib/*.h
grep -c "TTI_REG2FLOP" tt_llk_blackhole/llk_lib/*.h

# Find CFGSHIFTMASK usage
grep -r "CFGSHIFTMASK" tt_llk_blackhole/

# Find context addressing patterns
grep -r "unp_cfg_context.*==.*0" tt_llk_blackhole/
```
