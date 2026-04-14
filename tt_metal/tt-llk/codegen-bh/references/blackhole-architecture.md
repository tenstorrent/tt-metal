# Blackhole SFPU Architecture Reference

This document provides the essential architecture knowledge needed to write Blackhole SFPU kernels.

## Overview

Blackhole's Vector Engine (SFPU) has:
- **32 SFPU lanes** organized in 4x8 grid
- **FP32 precision** internally
- **8 Local Registers** (LREG0-LREG7) per lane
- **Conditional execution** per lane via `v_if` / `v_endif`
- **SFPI C++ library** for high-level operations

## Required Includes

```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

## Namespace Structure

```cpp
namespace ckernel
{
namespace sfpu
{
    // Your SFPU kernel functions here
} // namespace sfpu
} // namespace ckernel
```

---

## Core API (SFPI Library)

### Loading from Dest Register

```cpp
// Load from dest register into a vFloat
sfpi::vFloat val = sfpi::dst_reg[0];
```

### Storing to Dest Register

```cpp
// Store result back to dest register
sfpi::dst_reg[0] = result;
```

### Incrementing Dest Pointer

```cpp
// Advance to next dest register
sfpi::dst_reg++;
```

---

## Data Types

### sfpi::vFloat
Primary floating-point vector type for SFPU operations.

```cpp
sfpi::vFloat val = sfpi::dst_reg[0];
sfpi::vFloat result = val * 2.0f;
```

### sfpi::vInt / sfpi::vUInt
Integer vector types for bitwise operations and LUT indexing.

```cpp
sfpi::vInt exp = exexp(val);  // Extract exponent
sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];  // Load LUT register
```

---

## Arithmetic Operations

### Basic Arithmetic (Operator Overloading)

```cpp
// Addition
sfpi::vFloat sum = a + b;

// Subtraction
sfpi::vFloat diff = a - b;

// Multiplication
sfpi::vFloat prod = a * b;

// Negation
sfpi::vFloat neg_x = -x;
```

### Multiply-Add

```cpp
// result = a * b + c
sfpi::vFloat result = a * b + c;
```

---

## Conditional Execution

Blackhole uses `v_if` / `v_endif` for per-lane conditional execution:

```cpp
v_if (val < 0.0f)
{
    val = 0.0f;  // Only applies to lanes where condition is true
}
v_endif;

// v_elseif and v_else also supported
v_if (exp >= 0)
{
    // ... code ...
}
v_elseif (exp < -10)
{
    // ... code ...
}
v_else
{
    // ... code ...
}
v_endif;
```

### Nested Conditions with v_and

```cpp
v_if (exp >= 0)
{
    for (int s_iter = 0; s_iter < 7; s_iter++)
    {
        exp = exp - 1;
        v_and(exp >= 0);  // Narrow predication on each loop
        val = val * val;
    }
}
v_endif;
```

---

## Constants

### Built-in Constants

```cpp
sfpi::vConst0p8373  // 0.8373
sfpi::vConst1       // 1.0
```

### Loading Constants

```cpp
// Convert scalar to vFloat16b
sfpi::vFloat c = sfpi::s2vFloat16b(0.863281f);

// Load programmable constant
sfpi::vConstFloatPrgm0 = 0.5f;  // Set in init function
sfpi::vFloat half = sfpi::vConstFloatPrgm0;  // Use in calculation
```

### Loading Immediate Values

```cpp
// Load 32-bit immediate into LREG
_sfpu_load_imm32_(0, 0x32F433D9);  // LREG0 = 0x32F433D9

// Load 16-bit immediate
_sfpu_load_imm16_(0, 0x28FF);
```

---

## Built-in Functions

### Exponential Helpers

```cpp
// Extract exponent from floating point value
sfpi::vInt exp = exexp(val);

// Set exponent of floating point value
val = setexp(val, 126);

// Set sign of floating point value
val = setsgn(val, 0);  // Make positive
```

### Nonlinear Functions

```cpp
// Exponential (custom implementation)
sfpi::vFloat result = _sfpu_exp_(val);

// Reciprocal with iterations
sfpi::vFloat recip = _sfpu_reciprocal_<2>(val);
```

---

## LUT (Look-Up Table) Operations

Blackhole extensively uses LUT-based approximations:

```cpp
// Save LUT registers at function start
sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

// Use LUT
sfpi::vFloat result = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode);

// Alternative LUT functions
result = lut(val, l0, l1, imm2);
result = lut2_sign(val, l0, l1, l2, l4, l5, l6);

// Restore LUT registers at function end
sfpi::l_reg[sfpi::LRegs::LReg0] = l0;
sfpi::l_reg[sfpi::LRegs::LReg1] = l1;
// ... etc
```

---

## Low-Level TTI Instructions

When needed, Blackhole also supports direct TTI instructions:

### TTI_SFPLOAD - Load from Dest

```cpp
TTI_SFPLOAD(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr, done);
```

### TTI_SFPLOADI - Load Immediate

```cpp
// Load 16-bit immediate into LREG
TTI_SFPLOADI(lreg_ind, mode, imm16);

// Load 32-bit value (two instructions)
TTI_SFPLOADI(0, 0xA, lo16);  // Low 16 bits
TTI_SFPLOADI(0, 0x8, hi16);  // High 16 bits
```

### TTI_SFPSTORE - Store to Dest

```cpp
TTI_SFPSTORE(lreg_ind, done, sfpu_addr_mode, dest_reg_addr, instr_mod0);
```

### TTI_SFPMAD - Multiply-Add

```cpp
TTI_SFPMAD(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1);
```

### TTI_SFPCONFIG - Configuration

```cpp
TTI_SFPCONFIG(value, dest, mode);
```

### TTI_SFPLOADMACRO - Macro Instruction

```cpp
TTI_SFPLOADMACRO(lreg_ind, instr_mod0, addr_mode, dest_offset);
```

---

## Iteration Pattern

Standard SFPU iteration pattern for Blackhole:

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_operation_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        // ... computation ...

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

---

## Complete Examples

### Example 1: Sigmoid (LUT-Based)

```cpp
#pragma once
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sigmoid_(const int iterations)
{
    constexpr int lut_mode = 0;
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
        sfpi::dst_reg[0] = lut2(val, l0, l1, l2, l4, l5, l6, lut_mode) + 0.5f;
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

template <bool APPROXIMATION_MODE>
inline void _init_sigmoid_()
{
    // 6-piece LUT for sigmoid approximation
    _sfpu_load_imm32_(0, 0x32F433D9);  // A0, A1 coefficients
    _sfpu_load_imm32_(4, 0x23C89018);  // B0, B1 coefficients
    _sfpu_load_imm32_(1, 0x300A318A);  // A2, A3 coefficients
    _sfpu_load_imm32_(5, 0x30272BAA);  // B2, B3 coefficients
    _sfpu_load_imm32_(2, 0x7C002A35);  // A4, A5 coefficients
    _sfpu_load_imm32_(6, 0x37ff34CC);  // B4, B5 coefficients
}

} // namespace sfpu
} // namespace ckernel
```

### Example 2: Exponential (Series Expansion)

```cpp
#pragma once
#include "sfpi.h"

namespace ckernel::sfpu
{

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);
    v_if (exp >= 0)
    {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);
    val = val * tmp + sfpi::vConst1;

    v_if (exp >= 0)
    {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void _calculate_exponential_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        // Force sign to 0 (make number positive)
        sfpi::vFloat out = _sfpu_exp_(sfpi::setsgn(in, 0));

        v_if (in < 0)
        {
            out = _sfpu_reciprocal_<2>(out);
        }
        v_endif;

        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
```

---

## Querying assembly.yaml

For instruction details not covered here, grep assembly.yaml:

```bash
# Get SFPLOAD details
grep -A 30 "^SFPLOAD:" tt_llk_blackhole/instructions/assembly.yaml

# Get all SFPU instructions
grep "^SFP" tt_llk_blackhole/instructions/assembly.yaml

# Get specific instruction
grep -A 20 "^SFPMAD:" tt_llk_blackhole/instructions/assembly.yaml
```

---

## Available SFPU Instructions (Blackhole)

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Load from dest to LREG |
| `SFPLOADI` | Load immediate to LREG |
| `SFPSTORE` | Store LREG to dest |
| `SFPMAD` | Multiply-add |
| `SFPADD` | Add |
| `SFPMUL` | Multiply |
| `SFPLUT` | LUT lookup |
| `SFPLUTFP32` | FP32 LUT lookup |
| `SFPSETCC` | Set condition code |
| `SFPENCC` | Enable condition code |
| `SFPCOMPC` | Compare condition |
| `SFPEXEXP` | Extract exponent |
| `SFPEXMAN` | Extract mantissa |
| `SFPSETEXP` | Set exponent |
| `SFPSETMAN` | Set mantissa |
| `SFPSETSGN` | Set sign |
| `SFPABS` | Absolute value |
| `SFPAND` | Bitwise AND |
| `SFPOR` | Bitwise OR |
| `SFPXOR` | Bitwise XOR |
| `SFPNOT` | Bitwise NOT |
| `SFPSHFT` | Shift |
| `SFPSHFT2` | Extended shift |
| `SFPCAST` | Type cast |
| `SFPSWAP` | Swap values |
| `SFPCONFIG` | Configuration |
| `SFPLOADMACRO` | Macro instruction |
| `SFPARECIP` | Approximate reciprocal |
| `SFP_STOCH_RND` | Stochastic rounding |
| `SFPNOP` | No operation |

---

## Reference Paths

| Architecture | SFPU | Math/Pack/Unpack |
|--------------|------|------------------|
| Blackhole | `tt_llk_blackhole/common/inc/sfpu/` | `tt_llk_blackhole/llk_lib/` |
| Wormhole | `tt_llk_wormhole_b0/common/inc/sfpu/` | `tt_llk_wormhole_b0/llk_lib/` |
