# LLK Architecture Reference

Low-Level Kernels (LLK) for Tenstorrent Blackhole architecture.

## Kernel Categories

| Category | Purpose | Location | Naming Pattern |
|----------|---------|----------|----------------|
| **SFPU** | Vector operations (sigmoid, exp, etc.) | `common/inc/sfpu/` | `ckernel_sfpu_{op}.h` |
| **Math** | Matrix/tensor operations (matmul, reduce) | `llk_lib/` | `llk_math_{op}.h` |
| **Pack** | Pack data from dest to L1 | `llk_lib/` | `llk_pack_{op}.h` |
| **Unpack** | Unpack data from L1 to src registers | `llk_lib/` | `llk_unpack_{op}.h` |

---

## 1. SFPU Kernels

Vector operations on the Special Function Processing Unit.

### Architecture
- 32 lanes in 4x8 grid
- FP32 precision internally
- 8 local registers (LREG0-LREG7)
- Uses SFPI C++ library for high-level operations

### Key API (SFPI Library)

```cpp
// Load from dest register
sfpi::vFloat val = sfpi::dst_reg[0];

// Store to dest register
sfpi::dst_reg[0] = result;

// Increment dest pointer
sfpi::dst_reg++;

// Conditional execution
v_if (val < 0.0f) { val = 0.0f; } v_endif;
```

### Low-Level Instructions (TTI)

```cpp
TTI_SFPLOAD(lreg, mod, addr_mode, addr, done)  // Load from dest
TTI_SFPLOADI(lreg, mode, imm16)                 // Load immediate
TTI_SFPSTORE(lreg, done, addr_mode, addr, mod)  // Store to dest
TTI_SFPMAD(a, b, c, dest, mod)                  // dest = a*b + c
TTI_SFPLUT(lreg, mode)                          // LUT lookup
TTI_SFPLUTFP32(lreg, mode)                      // FP32 LUT lookup
```

### Template Structure

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_{op}_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat val = sfpi::dst_reg[0];
        // ... operation ...
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

### Examples
- `ckernel_sfpu_sigmoid.h` - LUT-based approximation
- `ckernel_sfpu_exp.h` - Series expansion with conditionals
- `ckernel_sfpu_gelu.h` - Complex multi-step operation

---

## 2. Math Kernels

Matrix and tensor operations using FPU/SFPU.

### Key Files
| File | Operations |
|------|------------|
| `llk_math_matmul.h` | Matrix multiplication |
| `llk_math_reduce.h` | Reduction (sum, max, avg) |
| `llk_math_eltwise_binary.h` | Element-wise binary ops |
| `llk_math_eltwise_unary_datacopy.h` | Data copy |

### Common Patterns
```cpp
// Namespace usage
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// Template parameters
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void _llk_math_reduce_(...) { ... }
```

### Key Instructions (Math)
```cpp
TTI_GMPOOL(...)    // Global max pool
TTI_GAPOOL(...)    // Global average pool
TTI_ELWADD(...)    // Element-wise add
TTI_ELWMUL(...)    // Element-wise multiply
TTI_MVMUL(...)     // Matrix-vector multiply
TTI_SETRWC(...)    // Set read/write counters
TTI_ZEROSRC(...)   // Zero source registers
TTI_TRNSPSRCB(...) // Transpose source B
```

### Reduce Dimensions
```cpp
ReduceDim::REDUCE_ROW  // Reduce along rows
ReduceDim::REDUCE_COL  // Reduce along columns
```

### Pool Types
```cpp
PoolType::MAX  // Max pooling
PoolType::AVG  // Average pooling
PoolType::SUM  // Sum reduction
```

---

## 3. Pack Kernels

Pack data from destination registers to L1 memory.

### Key Files
| File | Purpose |
|------|---------|
| `llk_pack.h` | Basic pack operations |
| `llk_pack_common.h` | Common pack utilities |
| `llk_pack_rows.h` | Pack rows |
| `llk_pack_untilize.h` | Pack with untilize |

### Template Structure
```cpp
template <std::uint8_t PACK_SEL>
inline void _llk_pack_init_(const std::uint8_t buf_desc_id, const std::uint32_t num_tiles)
{
    _llk_pack_mop_config_<PACK_SEL>(buf_desc_id, num_tiles);
}

template <std::uint8_t PACK_SEL>
inline void _llk_pack_(
    const std::uint32_t start_math_dest_tile_idx,
    const std::uint32_t start_l1_tile_idx)
{ ... }
```

### Pack Resources
```cpp
p_pacr::PACK0  // First packer
p_pacr::PACK1  // Second packer
```

### Key Instructions (Pack)
```cpp
TTI_PACR(...)       // Pack operation
TT_OP_PACR0_TILE_INC(...)  // Pack with PACK0
TT_OP_PACR1_TILE_INC(...)  // Pack with PACK1
```

---

## 4. Unpack Kernels

Unpack data from L1 memory to source registers.

### Key Files
| File | Purpose |
|------|---------|
| `llk_unpack_A.h` | Unpack to SRC A |
| `llk_unpack_AB.h` | Unpack to SRC A and B |
| `llk_unpack_AB_matmul.h` | Unpack for matmul inputs |
| `llk_unpack_AB_reduce.h` | Unpack for reduction |
| `llk_unpack_tilize.h` | Unpack with tilize |

### Template Structure
```cpp
template <std::uint8_t UNPACK_SEL>
inline void _llk_unpack_init_(...)
{ ... }

template <std::uint8_t UNPACK_SEL>
inline void _llk_unpack_(...)
{ ... }
```

### Unpack Resources
```cpp
p_unpacr::SRCAB  // Unpack to both SRC A and B
p_unpacr::SRCA   // Unpack to SRC A only
p_unpacr::SRCB   // Unpack to SRC B only
```

---

## Common Includes

### SFPU Kernels
```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

### Math/Pack/Unpack Kernels
```cpp
#include "ckernel_trisc_common.h"
#include "llk_math_common.h"   // for math
#include "llk_pack_common.h"   // for pack
#include "llk_unpack_common.h" // for unpack
```

---

## Namespaces

```cpp
namespace ckernel { }           // Main namespace
namespace ckernel::sfpu { }     // SFPU operations
namespace ckernel::math { }     // Math utilities
namespace ckernel::trisc { }    // TRISC utilities
```

---

## MathFidelity

```cpp
MathFidelity::LoFi    // Low fidelity (faster)
MathFidelity::HiFi    // High fidelity (more accurate)
MathFidelity::HiFi2   // Higher fidelity
MathFidelity::HiFi3   // Highest fidelity
MathFidelity::HiFi4   // Maximum fidelity
```

---

## Reference Paths

| Architecture | SFPU | Math/Pack/Unpack |
|--------------|------|------------------|
| Blackhole | `tt_llk_blackhole/common/inc/sfpu/` | `tt_llk_blackhole/llk_lib/` |
| Wormhole | `tt_llk_wormhole_b0/common/inc/sfpu/` | `tt_llk_wormhole_b0/llk_lib/` |
