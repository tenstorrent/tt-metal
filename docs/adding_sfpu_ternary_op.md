# Adding a New SFPU Ternary Operation

This guide documents the necessary steps to add a new SFPU (Special Function Processing Unit) ternary operation to the tt-metal/ttnn codebase.

## Overview

A ternary operation takes three inputs and produces one output. In TTNN, ternary operations support different variants:
- **TTT**: Three tensor inputs
- **TTS**: Two tensor inputs + one scalar
- **TST**: Tensor + scalar + tensor
- **TSS**: Tensor + two scalars

---

## Hardware Architecture: Data Flow from DRAM to DEST

Before diving into implementation details, it's essential to understand how data flows through the hardware. This section explains the memory hierarchy and the critical differences between SFPU and FPU operations.

### Memory Hierarchy Overview

Data flows through three levels of memory:

```
DRAM (off-chip, GB)  →  L1 Circular Buffers (on-chip, ~512KB)  →  DEST Registers (16 tiles)
        ↑                         ↑                                      ↑
    via NOC                  Reader kernel                       copy_tile() or FPU ops
```

| Memory Level | Location | Size | Access Speed |
|--------------|----------|------|--------------|
| DRAM | Off-chip (global memory) | GB | Slow, via NOC |
| L1 Memory / Circular Buffers | On-chip (per core) | 256KB-512KB | Fast, direct L1 access |
| DEST Registers | On-chip (register file) | 16 tiles | Very fast, direct element/tile access |
| SFPU Registers (LREG0-LREG5) | On-chip (SFPU unit) | 8 elements | Fastest, for computations |

### Step 1: DRAM to L1 — The Reader Kernel

The **reader kernel** executes on the dataflow thread and moves data from DRAM into L1 memory via circular buffers using the Network-on-Chip (NOC).

```cpp
// Reader kernel pattern (simplified)
for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    // 1. Reserve space in circular buffer for incoming tile
    cb_reserve_back(cb_id_src0, 1);

    // 2. Get the L1 write pointer (where to write the tile)
    uint32_t l1_write_addr = get_write_ptr(cb_id_src0);

    // 3. Use NOC to read tile from DRAM asynchronously
    noc_async_read_tile(tile_id, dram_accessor, l1_write_addr);

    // 4. Wait for NOC read to complete
    noc_async_read_barrier();

    // 5. Make tile visible to compute kernel
    cb_push_back(cb_id_src0, 1);
}
```

### Step 2: Circular Buffers — L1 Producer-Consumer Interface

Circular buffers (CBs) manage FIFO queues of tiles in L1 memory, enabling producer-consumer communication between kernels.

```
Circular Buffer in L1:
┌─────────────────────────────────────┐
│ Tile 0 │ Tile 1 │ Tile 2 │ ... │ Tile N │
└─────────────────────────────────────┘
    ↑                           ↑
  read_ptr                   write_ptr
  (consumer)                 (producer)
```

**Key CB Operations:**

| Operation | Thread | Purpose |
|-----------|--------|---------|
| `cb_reserve_back(cb_id, n)` | Producer/Reader | Wait for space to write n tiles |
| `cb_push_back(cb_id, n)` | Producer/Reader | Make n tiles visible to consumer |
| `cb_wait_front(cb_id, n)` | Consumer/Compute | Wait for n tiles to be available |
| `cb_pop_front(cb_id, n)` | Consumer/Compute | Free space, advance read pointer |

**Important constraint:** CB operations must maintain synchronization — the reader kernel exclusively updates write pointers, and the compute kernel exclusively updates read pointers.

### Step 3: L1 to DEST — The `copy_tile` Mechanism

Once data is in L1 circular buffers, the compute kernel moves it to **DEST registers** for processing.

```cpp
// Initialize copy operation for a specific CB
copy_tile_to_dst_init_short(cb_in0);

// Copy tile from CB to DEST register
// - in_tile_index: which tile in the CB (relative to read pointer)
// - dst_tile_index: which DEST register to write to (0-15)
copy_tile(cb_in0, 0, 0);  // CB[0] → DEST[0]
```

**Two-stage internal process:**
1. **UNPACK thread:** Reads from L1 CB, unpacks into SRC registers
2. **MATH thread:** Moves data from SRC to DEST registers

**DEST Register Layout:**
- 16 DEST register slots available (DEST[0] through DEST[15])
- Each holds one 32×32 tile
- For SFPU: 64 elements per tile (processed in 8-element rows)

### Two Patterns for SFPU Operations

SFPU operations can be invoked in two distinct patterns, depending on whether they complement FPU operations or replace them entirely.

#### Pattern 1: SFPU Called Within FPU Compute Kernels (Pre/Post-Processing)

This pattern is primarily used for **binary operations** where FPU can handle part of the computation and SFPU handles transcendental transformations. Examples include:

- **DIV**: `a / b = a * SFPU_RECIP(b)` — SFPU computes reciprocal, FPU multiplies
- **LOGADDEXP**: `log(exp(a) + exp(b))` — SFPU computes exp (prescale), FPU adds, SFPU computes log (postscale)
- **HYPOT**: `sqrt(a² + b²)` — SFPU squares inputs (prescale), FPU adds, SFPU computes sqrt (postscale)

**Why ternary operations don't typically use Pattern 1:**
FPU operations are inherently binary (2 inputs). A ternary operation like `a * b + c` would require multiple FPU calls with intermediate storage, losing the performance benefit. Instead, ternary operations use Pattern 2 where SFPU handles all three inputs directly.

For detailed documentation on Pattern 1, see `adding_sfpu_binary_op.md`.

---

#### Pattern 2: Standalone SFPU Operations (Used for Ternary Operations)

When operations cannot use FPU at all—either because the operation doesn't exist in FPU hardware or because the **data format isn't supported by FPU**—SFPU is used exclusively with explicit `copy_tile` for all inputs.

---

##### Deep Dive: ADDCMUL Operation

ADDCMUL is the canonical example of a standalone SFPU ternary operation. It computes:
```
addcmul(a, b, c, value) = a + (value * b * c)
```

All three inputs must be copied to DEST registers, and SFPU performs the entire computation element-by-element.

###### 1. How the Operation is Called from Python

**Test function** (from `tests/ttnn/unit_tests/operations/eltwise/test_ternary.py`):

```python
import ttnn
import torch

def test_addcmul(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.full(in_data1_shape, 0.0031, dtype=torch_dtype)
    in_data2 = torch.full(in_data2_shape, 508.0, dtype=torch_dtype)
    in_data3 = torch.full(in_data3_shape, 748.0, dtype=torch_dtype)

    # Convert to TTNN tensors
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Call the ternary operation
    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    # Validate against golden function
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_with_ulp(output_tensor, golden_tensor)
```

**Golden function** (from `ttnn/ttnn/operations/ternary.py`):

```python
def _golden_function_addcmul(input_tensor_a, input_tensor_b, input_tensor_c, *args, value=1, **kwargs):
    import torch
    return torch.addcmul(input_tensor_a, input_tensor_b, input_tensor_c, value=value)

ttnn.attach_golden_function(ttnn.addcmul, golden_function=_golden_function_addcmul)
```

###### 2. Which Macros Are Defined

When `TernaryOpType::ADDCMUL` is processed, `get_compute_defines()` in `ternary_op_utils.cpp` generates:

```cpp
// From ternary_op_utils.cpp
case TernaryOpType::ADDCMUL:
    defines["TERNARY_SFPU_OP_INIT"] = "llk_math_eltwise_ternary_sfpu_addcmul_init<APPROX>";
    defines["TERNARY_SFPU_OP_FUNC"] =
        "llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DST_ACCUM_MODE, DataFormat::Float16_b>";
    break;
```

These become compiler flags:
```cpp
#define TERNARY_SFPU_OP_INIT()  llk_math_eltwise_ternary_sfpu_addcmul_init<APPROX>()
#define TERNARY_SFPU_OP_FUNC(d0, d1, d2, od, val)  \
    llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DST_ACCUM_MODE, DataFormat::Float16_b>(d0, d1, d2, od, val)
```

###### 3. The Compute Kernel

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_sfpu.cpp`

```cpp
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/addcmul.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);  // The 'value' parameter

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;  // output

    // Note: unary_op_init_common, NOT binary_op_init_common
    // This signals standalone SFPU mode (no FPU involvement)
    unary_op_init_common(cb_in0, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for all THREE input tiles
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_wait_front(cb_in2, 1);

        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // CRITICAL: Copy ALL inputs to DEST registers
        // SFPU can ONLY read from DEST, not from CBs
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0, 0);  // CB[0] → DEST[0] (input_a)

        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0, 1);  // CB[1] → DEST[1] (input_b)

        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0, 2);  // CB[2] → DEST[2] (input_c)

        // SFPU operation: reads DEST[0,1,2], writes result to DEST[0]
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);
        // Computes: DEST[0] = DEST[0] + (scalar_arg * DEST[1] * DEST[2])
        //           result  = a       + (value      * b       * c      )

        tile_regs_commit();
        tile_regs_wait();

        // Pack result from DEST[0] to output CB
        pack_tile(0, cb_out);

        tile_regs_release();

        // Release input CBs, push output
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_pop_front(cb_in2, 1);
    }
}
}
```

###### 4. The LLK Wrapper

**File:** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_addcmul.h`

```cpp
#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_addcmul.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_addcmul(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, uint value,
    int vector_mode = (int)VectorMode::RC) {

    // Delegates to the common ternary SFPU parameter handler
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_addcmul<APPROXIMATE, is_fp32_dest_acc_en, data_format, ITERATIONS>,
        dst_index0,   // DEST index for input_a
        dst_index1,   // DEST index for input_b
        dst_index2,   // DEST index for input_c
        odst,         // DEST index for output (can overlap with input)
        vector_mode,
        value);       // scalar multiplier
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_addcmul_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::addcmul>();
}

}  // namespace ckernel
```

###### 5. The SFPU Calculation (The Core Math)

**File:** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h`

```cpp
#pragma once

#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul(
    const uint dst_index_in0,  // DEST index containing input_a
    const uint dst_index_in1,  // DEST index containing input_b
    const uint dst_index_in2,  // DEST index containing input_c
    const uint dst_index_out,  // DEST index for output
    const uint value) {        // scalar value to multiply

    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT;

    // Each tile is 64 elements in DEST addressing
    constexpr uint dst_tile_size = 64;

    // Load the scalar 'value' into LREG3 (split into lower/upper 16 bits)
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);

    // Process 8 elements per iteration (one row of the tile)
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load input_b from DEST[dst_index_in1]
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        // LREG4 = value * input_b
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);

        // Load input_a and input_c from DEST
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_7, dst_index_in2 * dst_tile_size);

        // LREG5 = (value * input_b) * input_c + input_a
        //       = LREG4 * LREG2 + LREG0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0);

        TTI_SFPNOP;  // Pipeline delay

        // Stochastic rounding for non-FP32 formats
        if constexpr (!is_fp32_dest_acc_en) {
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A,
                0, p_sfpu::LREG5, p_sfpu::LREG5, InstrModLoadStore::FP16A);
        }

        // Store result to DEST[dst_index_out]
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_7, dst_index_out * dst_tile_size);

        // Advance to next 8-element group
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

###### 6. Complete Data Flow Diagram for ADDCMUL

**Operation**: `addcmul(a, b, c, value) = a + (value * b * c)`

```
                        Standalone SFPU Operation (No FPU)
                        ┌─────────────────────────────────────────────────────┐
                        │  All THREE inputs must be copied to DEST            │
                        │  SFPU performs the entire computation               │
                        │                                                     │
  Input CB (cb_in0)     │    ┌─────────────────────────────┐                  │
  ──────────────────────┼──▶ │ copy_tile(cb_in0, 0, 0)     │                  │
         (input_a)      │    │ Unpacker → DEST[0]          │                  │
                        │    └──────────────┬──────────────┘                  │
                        │                   │                                 │
  Input CB (cb_in1)     │    ┌──────────────┴──────────────┐                  │
  ──────────────────────┼──▶ │ copy_tile(cb_in1, 0, 1)     │                  │
         (input_b)      │    │ Unpacker → DEST[1]          │                  │
                        │    └──────────────┬──────────────┘                  │
                        │                   │                                 │
  Input CB (cb_in2)     │    ┌──────────────┴──────────────┐    ┌───────┐    │
  ──────────────────────┼──▶ │ copy_tile(cb_in2, 0, 2)     │    │       │    │
         (input_c)      │    │ Unpacker → DEST[2]          │    │       │    │
                        │    └──────────────┬──────────────┘    │       │    │
                        │                   │                   │ SFPU  │    │
                        │                   ▼                   │       │    │
                        │         ┌─────────────────┐           │addcmul│    │
                        │         │     DEST        │           │       │    │
                        │         │  [0] = a        │──────────▶│       │    │
                        │         │  [1] = b        │           │ a +   │    │
                        │         │  [2] = c        │◀──────────│(v*b*c)│    │
                        │         └────────┬────────┘           │       │    │
                        │                  │ DEST[0] = result   └───────┘    │
                        │                  ▼                                 │
                        │         ┌─────────────────┐                        │   Output CB
                        │         │   pack_tile     │────────────────────────┼──────────────▶
                        │         │  DEST[0] → CB   │                        │   (cb_out)
                        │         └─────────────────┘                        │
                        └─────────────────────────────────────────────────────┘
```

**Circular Buffer Roles:**
| CB | Index | Contents | Role |
|----|-------|----------|------|
| `cb_in0` | c_0 | `input_a` | First input from DRAM |
| `cb_in1` | c_1 | `input_b` | Second input from DRAM |
| `cb_in2` | c_2 | `input_c` | Third input from DRAM |
| `cb_out` | c_3 | `result` | Output to DRAM |

**Note:** Unlike Pattern 1 (FPU with pre/post-processing), there are **no intermediate CBs**. All data flows directly from input CBs → DEST → output CB.

###### Key Differences from Pattern 1 (FPU-based)

| Aspect | Pattern 1 (FPU + SFPU) | Pattern 2 (SFPU Only) |
|--------|------------------------|----------------------|
| **Init function** | `binary_op_init_common()` | `unary_op_init_common()` |
| **Inputs to DEST** | Only those needing SFPU | **ALL inputs** |
| **FPU usage** | Main computation | None |
| **Intermediate CBs** | Yes (between SFPU and FPU) | No |
| **Example** | DIV: `a * RECIP(b)` | ADDCMUL: `a + v*b*c` |
| **Macro prefix** | `SFPU_OP_INIT_PRE_IN*` | `TERNARY_SFPU_OP_*` |

---

**Case A: Operations Not Available in FPU**

The generic pattern for standalone SFPU ternary operations:

```cpp
// Standalone SFPU ternary kernel pattern
unary_op_init_common(cb_in0, cb_out);  // Note: unary_op_init, not binary_op_init

for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    cb_wait_front(cb_in2, 1);

    tile_regs_acquire();

    // ALL inputs must be copied to DEST first
    copy_tile_init(cb_in0);
    copy_tile(cb_in0, 0, 0);  // input_a → DEST[0]

    copy_tile_init(cb_in1);
    copy_tile(cb_in1, 0, 1);  // input_b → DEST[1]

    copy_tile_init(cb_in2);
    copy_tile(cb_in2, 0, 2);  // input_c → DEST[2]

    // SFPU operates on DEST register indices (not CB indices!)
    TERNARY_SFPU_OP_INIT();
    TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);  // DEST[0,1,2] → DEST[0]

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    cb_pop_front(cb_in2, 1);
    cb_push_back(cb_out, 1);
}
```

**Case B: Data Formats Not Supported by FPU**

The FPU has native support for certain floating-point formats, but **INT32, UINT16, and FP32 binary operations** require SFPU. These SFPU binary operations take **DEST register indices** (not CB indices):

**Integer Addition** (from `add_int_sfpu.h`):
```cpp
// SFPU-based integer addition - FPU doesn't support INT32
ALWI void add_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, InstrModLoadStore::INT32, false>(
        idst0, idst1, odst)));  // Operates on DEST indices!
}

ALWI void add_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, InstrModLoadStore::LO16, false>(
        idst0, idst1, odst)));
}
```

**FP32 Binary Operations** (from `eltwise_binary_sfpu.h`):
```cpp
// SFPU-based FP32 operations - for formats where FPU may not have native support
ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::ADD>(idst0, idst1, odst)));
}

ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::MUL>(idst0, idst1, odst)));
}

ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::DIV>(idst0, idst1, odst)));
}
```

**Usage pattern for format-specific SFPU:**
```cpp
// When using SFPU binary ops for unsupported formats
tile_regs_acquire();

// Must copy both inputs to DEST first
copy_tile(cb_in0, 0, 0);  // input_a → DEST[0]
copy_tile(cb_in1, 0, 1);  // input_b → DEST[1]

// SFPU binary op takes DEST indices
add_binary_tile_init();
add_binary_tile(0, 1, 0);  // DEST[0] + DEST[1] → DEST[0]

tile_regs_commit();
```

**Contrast with FPU binary ops** (from `eltwise_binary.h`):
```cpp
// FPU binary ops take CB indices and read directly from L1
ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));  // Reads from CBs
    MATH((llk_math_eltwise_binary<ELWADD, ...>(icb0, icb1, idst, true)));
}
```

---

#### Comparison: The Two SFPU Patterns

| Aspect | Pattern 1: SFPU in FPU Kernel | Pattern 2: Standalone SFPU |
|--------|-------------------------------|----------------------------|
| **When used** | Operation partially supported by FPU | Operation/format not in FPU |
| **Init function** | `binary_op_init_common()` | `unary_op_init_common()` |
| **FPU involvement** | Yes (main computation) | None |
| **SFPU role** | Pre/post-processing | Entire computation |
| **Data for SFPU** | `copy_tile` before SFPU only | `copy_tile` for ALL inputs |
| **Macro names** | `PRE_IN0_0`, `PRE_IN1_0`, `SFPU_OP_0` | `TERNARY_SFPU_OP_*` or direct calls |
| **Examples** | DIV, LOGADDEXP, SQUARED_DIFF | MULADD, WHERE, `add_int32_tile`, `add_binary_tile` |
| **CB usage** | Intermediate CBs between phases | Direct input CBs only |

**Key insight:** The fundamental difference is whether FPU participates in the computation. In Pattern 1, data flows through intermediate CBs between SFPU and FPU phases. In Pattern 2, all inputs go directly to DEST via `copy_tile`, and SFPU does everything.

### Complete Data Path: SFPU Ternary Operation

Here's the complete journey of tiles from DRAM to result for an SFPU ternary operation:

```
DRAM                    L1 (Circular Buffers)        DEST Registers          Output L1
│                       │                            │                        │
├─ Tile A ──(NOC)──→ CB0[0] ─────────────────────→ DEST[0] ────────────→ CB_out[0]
├─ Tile B ──(NOC)──→ CB1[0] ────→ SFPU Processing ─ (overwrites DEST[0])     │
└─ Tile C ──(NOC)──→ CB2[0] ─────────────────────→ DEST[2]                   │
                                                                              │
Reader Kernel      copy_tile (UNPACK/MATH)    SFPU (MATH)        pack_tile (PACK)
(Dataflow thread)                                                (to output CB)
```

**Step-by-step:**

1. **Reader Kernel** reads tiles from DRAM via NOC into L1 circular buffers
2. **Compute Kernel** waits for tiles: `cb_wait_front(cb_in0, 1)`
3. **copy_tile** moves each input from CB to DEST registers
4. **SFPU executes** element-by-element computation on DEST registers
5. **pack_tile** moves result from DEST to output circular buffer
6. **CB operations** release input tiles and push output tile

### SFPU Execution Model Detail

Inside the SFPU calculation function, processing happens element-by-element:

```cpp
#pragma GCC unroll 8  // Critical for performance
for (int i = 0; i < ITERATIONS; i++) {  // 8 iterations per tile row
    // Load elements from DEST registers
    vFloat a = dst_reg[dst_index0 * 64 + i];
    vFloat b = dst_reg[dst_index1 * 64 + i];
    vFloat c = dst_reg[dst_index2 * 64 + i];

    // Your operation (e.g., muladd: a * b + c)
    vFloat result = a * b + c;

    // Store result back to DEST
    dst_reg[odst * 64 + i] = result;
}
```

**Key points:**
- `dst_reg` is the DEST register file, indexed by `tile_index * 64 + element`
- Each tile occupies 64 elements in the DEST register addressing
- The `#pragma GCC unroll 8` is critical for performance
- SFPU uses local registers (LREG0-LREG5) for intermediate values

### Memory Constraints

- **Total L1:** ~512KB per core, shared among circular buffers, SRC registers, and DEST registers
- **DEST size:** 16 tiles maximum — limits what you can keep resident simultaneously
- **CB space:** Limited — must pop consumed tiles to free space for new ones
- **NOC bandwidth:** Shared across all cores — can become a bottleneck

---

## Understanding Boilerplate vs. Non-Boilerplate Code

Throughout this guide, code sections are marked as:
- **🔷 BOILERPLATE**: Code that follows a fixed pattern. You copy the structure and replace names/identifiers.
- **🔶 NON-BOILERPLATE**: Code that requires understanding of your operation's semantics, mathematical properties, or hardware constraints.

---

## Files to Modify/Create

### 1. Operation Type Definition

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp`

> **🔷 BOILERPLATE**
>
> **Why it's boilerplate:** This is a simple enum extension. You just add your operation's name to the list.
>
> **What to change:** Only the enum value name (e.g., `NEW_OP`). The name should match your operation's mathematical function.

Add your new operation to the `TernaryOpType` enum:

```cpp
enum class TernaryOpType {
    WHERE,
    LERP,
    ADDCMUL,
    ADDCDIV,
    MULADD,
    NEW_OP,  // Add your operation here
};
```

---

### 2. Operation Declaration

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.hpp`

> **🔷 BOILERPLATE** (structure) + **🔶 NON-BOILERPLATE** (variant selection)
>
> **Why the structure is boilerplate:** Every ternary operation follows the same struct pattern with `invoke()` methods. The method signatures are standardized.
>
> **What to change (boilerplate):**
> - Struct name (`NewOpOperation`)
> - Registration name (`"ttnn::new_op"`)
> - Variable name (`new_op`)
>
> **🔶 NON-BOILERPLATE consideration - Which variants to support:**
> You must decide which input combinations make sense for your operation:
> - Does your operation mathematically support a scalar as the third argument? (TTS)
> - Does your operation require all tensors? (TTT only)
> - Example: `LERP(a, b, weight)` supports TTS because weight can be a scalar, but `WHERE(condition, x, y)` requires TTT because the condition must be element-wise.

Declare the operation struct and register it:

```cpp
struct NewOpOperation {
    // TTT variant (3 tensors)
    // 🔷 BOILERPLATE: Standard method signature, just copy and rename
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const Tensor& input_c,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    // TTS variant (2 tensors + scalar)
    // 🔶 NON-BOILERPLATE DECISION: Only add this if your operation
    // mathematically supports a scalar as input_c
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        float scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

// 🔷 BOILERPLATE: Registration macro - just change names
constexpr auto new_op = ttnn::register_operation<"ttnn::new_op", operations::ternary::NewOpOperation>();
```

---

### 3. Operation Implementation

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp`

> **🔷 BOILERPLATE**
>
> **Why it's boilerplate:** These `invoke()` methods are thin wrappers that forward to `ttnn::prim::ternary`. The pattern is identical for all ternary operations - you're just specifying which operation type and variant to use.
>
> **What to change:**
> - Function name prefix (`NewOpOperation::invoke`)
> - `TernaryOpType::NEW_OP` enum value
> - `TernaryVariant` matching the signature (TTT, TTS, etc.)
>
> **Why no logic here:** The actual computation happens in the SFPU kernel. This layer just routes the call through the primitive system with correct metadata.

Implement the invoke methods:

```cpp
// 🔷 BOILERPLATE: Copy pattern, change enum value and function name
Tensor NewOpOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::ternary(
        input_a, input_b, input_c,
        TernaryOpType::NEW_OP,       // <- Change this to your enum
        TernaryVariant::TTT,
        memory_config);
}

// 🔷 BOILERPLATE: Same pattern for TTS variant
Tensor NewOpOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    float scalar,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::ternary(
        input_a, input_b, scalar,
        TernaryOpType::NEW_OP,       // <- Change this to your enum
        TernaryVariant::TTS,
        memory_config);
}
```

---

### 4. Kernel Configuration

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.hpp`

> **🔷 BOILERPLATE** (if reusing existing kernels) / **🔶 NON-BOILERPLATE** (if creating new kernels)
>
> **Why it can be boilerplate:** If your operation can reuse existing SFPU kernel files (which is common - the kernel files use macros that get substituted at compile time), you don't need to add new kernel names.
>
> **🔶 NON-BOILERPLATE - When you need new kernel names:**
> - Your operation requires a fundamentally different data flow pattern
> - Your operation needs custom circular buffer setup
> - Your operation processes tiles in a non-standard order
>
> Most SFPU operations can reuse existing kernel files because the actual operation is injected via `TERNARY_SFPU_OP_FUNC` macro.

Add kernel names to the `KernelName` enum if you're creating new kernel files:

```cpp
enum class KernelName {
    // ... existing kernels ...
    // 🔷 BOILERPLATE if needed: Just add enum values with your op name
    NEW_OP_SFPU_NO_BCAST,
    NEW_OP_SFPU_BCAST,
    NEW_OP_SFPU_ROW_BCAST,
};
```

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp`

> **🔷 BOILERPLATE** (config map entries) + **🔶 NON-BOILERPLATE** (compute defines)

Add kernel configuration entries to `kernel_config_map`:

```cpp
// 🔷 BOILERPLATE: Copy existing entries, change TernaryOpType enum
// The pattern is: {OpType, Variant, BroadcastType} -> {Reader, Compute, Writer}
// You can reuse existing kernel names if your operation follows standard data flow

{{TernaryOpType::NEW_OP, TernaryVariant::TTT, TernaryBroadcastType::NONE},
 {KernelName::READER_NO_BCAST, KernelName::NEW_OP_SFPU_NO_BCAST, KernelName::WRITER_NO_BCAST}},

{{TernaryOpType::NEW_OP, TernaryVariant::TTT, TernaryBroadcastType::COL_BCAST},
 {KernelName::READER_COL_BCAST, KernelName::NEW_OP_SFPU_BCAST, KernelName::WRITER_BCAST}},

// Add entries for each variant and broadcast type combination
```

Add kernel file path mapping in `get_kernel_file_path()`:

```cpp
// 🔷 BOILERPLATE: Map kernel name to file path
case KernelName::NEW_OP_SFPU_NO_BCAST:
    return "ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_new_op_sfpu.cpp";
```

Add compute defines in `get_compute_defines()`:

```cpp
// 🔶 NON-BOILERPLATE: These defines connect your operation to the LLK layer
//
// Why this is NOT boilerplate:
// - The function names must exactly match what you define in the LLK headers
// - The template parameters (APPROX, DST_ACCUM_MODE) affect numerical precision
// - If your operation doesn't need approximation modes, you might use different templates
//
// What to change:
// - `llk_math_eltwise_ternary_sfpu_new_op_init` - your init function name
// - `llk_math_eltwise_ternary_sfpu_new_op` - your operation function name

case TernaryOpType::NEW_OP:
    defines["TERNARY_SFPU_OP_INIT()"] = "llk_math_eltwise_ternary_sfpu_new_op_init<APPROX>()";
    defines["TERNARY_SFPU_OP_FUNC(d0, d1, d2, od)"] =
        "llk_math_eltwise_ternary_sfpu_new_op<APPROX, DST_ACCUM_MODE>(d0, d1, d2, od)";
    break;
```

---

### 5. Compute Kernels

Create compute kernel files in `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/`

> **🔷 BOILERPLATE** (entire kernel structure)
>
> **Why it's boilerplate:** The compute kernel is a template that:
> 1. Waits for input tiles in circular buffers
> 2. Copies tiles to DST registers
> 3. Calls the SFPU operation via macros
> 4. Packs result and pushes to output buffer
>
> This data flow is identical for ALL SFPU ternary operations. The actual computation is injected via the `TERNARY_SFPU_OP_INIT()` and `TERNARY_SFPU_OP_FUNC()` macros which are defined at compile time.
>
> **What to change:** Only the `#include` for your LLK header file.
>
> **Why macros instead of direct function calls:**
> - Compile-time injection allows the same kernel source to be reused for many operations
> - The specific operation is selected via `-D` compiler defines
> - This reduces code duplication and maintenance burden

**No Broadcast Kernel:** `ternary_new_op_sfpu.cpp`

```cpp
// 🔷 BOILERPLATE: This entire file follows a fixed pattern
// The only change needed is the #include for your specific LLK header

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "llk_math_eltwise_ternary_sfpu_new_op.h"  // <- Your LLK header

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // 🔷 BOILERPLATE: Standard circular buffer indices for ternary ops
    constexpr auto cb_in0 = tt::CBIndex::c_0;   // First input
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // Second input
    constexpr auto cb_in2 = tt::CBIndex::c_2;   // Third input
    constexpr auto cb_out = tt::CBIndex::c_3;   // Output

    unary_op_init_common(cb_in0, cb_out);

    // 🔷 BOILERPLATE: Standard tile processing loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Wait for all three input tiles
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_wait_front(cb_in2, 1);

        tile_regs_acquire();

        // Copy input tiles to DST registers 0, 1, 2
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);  // input_a -> dst[0]

        copy_tile_to_dst_init_short(cb_in1);
        copy_tile(cb_in1, 0, 1);  // input_b -> dst[1]

        copy_tile_to_dst_init_short(cb_in2);
        copy_tile(cb_in2, 0, 2);  // input_c -> dst[2]

        // 🔷 BOILERPLATE: These macros are replaced at compile time
        // with your specific operation's init and compute functions
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);  // dst[0], dst[1], dst[2] -> dst[0]

        tile_regs_commit();

        // Pack and push result
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);

        // Release input tiles
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_pop_front(cb_in2, 1);
    }
}
}
```

Create additional variants for broadcast types as needed:
- `ternary_new_op_sfpu_bcast.cpp` - For column/scalar broadcast
- `ternary_new_op_sfpu_row_bcast.cpp` - For row broadcast

---

### 6. LLK (Low Level Kernel) SFPU Headers

> **🔷 BOILERPLATE** (wrapper structure) + **🔶 NON-BOILERPLATE** (function reference)
>
> **Why the structure is boilerplate:** This header follows a fixed template pattern that wraps your SFPU calculation function with the LLK infrastructure.
>
> **What to change (boilerplate):**
> - Function names (`llk_math_eltwise_ternary_sfpu_new_op`, `llk_math_eltwise_ternary_sfpu_new_op_init`)
> - Include filename (`ckernel_sfpu_new_op.h`)
> - Calculation function name (`sfpu::calculate_new_op`)
>
> **🔶 NON-BOILERPLATE consideration - Template parameters:**
> - `APPROXIMATE`: Whether to use fast approximations (affects transcendental functions)
> - `data_format`: Input data format - some operations behave differently for different formats
> - `ITERATIONS`: Loop unrolling factor (usually 8 for standard tile processing)
>
> Most operations just copy these parameters, but if your operation has specific precision requirements, you may need to adjust.

Create headers for each supported architecture:

**Blackhole:** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_new_op.h`

**Wormhole B0:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_new_op.h`

```cpp
#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_new_op.h"  // 🔷 BOILERPLATE: Your SFPU implementation header

namespace ckernel {

// 🔷 BOILERPLATE: This wrapper function follows a fixed pattern
// It delegates to _llk_math_eltwise_ternary_sfpu_params_ which handles
// the low-level register setup and calls your calculation function

template <bool APPROXIMATE, DataFormat data_format = DataFormat::Invalid, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_new_op(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst,
    int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_new_op<APPROXIMATE, data_format, ITERATIONS>,  // <- Your function
        dst_index0, dst_index1, dst_index2, odst, vector_mode);
}

// 🔷 BOILERPLATE: Standard init function - usually unchanged
template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_new_op_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::unused>();
}

}  // namespace ckernel
```

---

### 7. SFPU Calculation Implementation

> **🔶 NON-BOILERPLATE** - This is the core of your operation!
>
> **Why this is NOT boilerplate:** This file contains the actual mathematical computation of your operation. Everything else in this guide is infrastructure to get data to and from this function.
>
> **Critical considerations:**
>
> 1. **Mathematical correctness:**
>    - Your formula must correctly implement the operation's semantics
>    - Consider edge cases: division by zero, negative inputs to sqrt, etc.
>
> 2. **Numerical precision:**
>    - `APPROXIMATE` template: Use `sfpu_reciprocal` vs `sfpu_reciprocal_approx`
>    - Some operations need higher precision intermediate calculations
>
> 3. **Data format handling:**
>    - BFloat16 has limited precision - some operations may need special handling
>    - BFloat8 has even less precision - test carefully
>
> 4. **Hardware constraints:**
>    - SFPU has limited instruction set - not all operations are directly available
>    - You may need to compose complex operations from primitives
>    - Available primitives: add, sub, mul, reciprocal, sqrt, exp, log, etc.
>
> 5. **Performance:**
>    - The loop processes 8 elements (ITERATIONS) per tile row
>    - `#pragma GCC unroll 8` is critical for performance
>    - Avoid branches inside the loop when possible
>
> 6. **DST register layout:**
>    - Each tile occupies 64 elements in DST (32x32 tile, processed in rows of 8)
>    - `dst_index * 64 + i` gives you element i of tile dst_index

Create the SFPU calculation header:

**File:** `tt_metal/hw/ckernels/[ARCH]/metal/llk_api/llk_sfpu/ckernel_sfpu_new_op.h`

```cpp
#pragma once

#include "ckernel_sfpu_common.h"

namespace ckernel {
namespace sfpu {

// 🔶 NON-BOILERPLATE: This is where your actual operation is implemented!
//
// Parameters explained:
// - dst_index0, dst_index1, dst_index2: DST register indices containing input tiles
// - odst: Output DST register index (can overlap with an input if you don't need it)
// - vector_mode: Usually ignored for standard operations
//
// Each DST register holds a tile (32x32 elements = 1024 values)
// The ITERATIONS loop processes one row of 8 elements at a time

template <bool APPROXIMATE, DataFormat data_format = DataFormat::Invalid, int ITERATIONS = 8>
inline void calculate_new_op(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = 0) {

    // 🔶 NON-BOILERPLATE: Your mathematical operation goes here
    //
    // Example operations and their implementations:
    //
    // ADDCMUL: a + b * c
    //   result = a + (b * c);
    //
    // ADDCDIV: a + b / c
    //   result = a + (b * sfpu_reciprocal<APPROXIMATE>(c));
    //
    // LERP: a + (b - a) * c
    //   result = a + (b - a) * c;
    //
    // WHERE: c ? a : b  (condition-based selection)
    //   result = (c != 0) ? a : b;  // Requires bitwise ops
    //
    // MULADD: a * b + c
    //   result = a * b + c;

    #pragma GCC unroll 8  // 🔶 CRITICAL for performance - unroll the inner loop
    for (int i = 0; i < ITERATIONS; i++) {
        // Load input values from DST registers
        // Each tile is 64 elements in DST (processing 8 elements per iteration)
        vFloat a = dst_reg[dst_index0 * 64 + i];
        vFloat b = dst_reg[dst_index1 * 64 + i];
        vFloat c = dst_reg[dst_index2 * 64 + i];

        // 🔶🔶🔶 YOUR OPERATION'S FORMULA GOES HERE 🔶🔶🔶
        //
        // This is the ONLY truly non-boilerplate code in the entire operation!
        // Everything else is infrastructure to get data here and results back.
        //
        // Available SFPU operations (check ckernel_sfpu_common.h for full list):
        // - Arithmetic: +, -, * (native vFloat operations)
        // - sfpu_reciprocal<APPROXIMATE>(x) - 1/x
        // - sfpu_sqrt<APPROXIMATE>(x) - square root
        // - sfpu_exp<APPROXIMATE>(x) - exponential
        // - sfpu_log<APPROXIMATE>(x) - natural log
        // - Bitwise: use lut operations for comparisons
        //
        // Example for MULADD (a * b + c):
        vFloat result = a * b + c;

        // Store result to output DST register
        dst_reg[odst * 64 + i] = result;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

---

### 8. Python Bindings

**File:** `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary_nanobind.cpp`

> **🔷 BOILERPLATE**
>
> **Why it's boilerplate:** The binding follows a standard pattern using `detail::bind_ternary_operation`. This helper handles all the complexity of exposing the C++ operation to Python.
>
> **What to change:**
> - Operation reference (`ttnn::new_op`)
> - Documentation string (describe what your operation does)
>
> **Note:** The helper function automatically handles all input type variants (TTT, TTS, etc.) based on the C++ overloads you defined.

Add bindings in the `bind_ternary_operations` function:

```cpp
// 🔷 BOILERPLATE: Standard binding pattern
detail::bind_ternary_operation(
    module,
    ttnn::new_op,       // <- Your operation
    R"doc(
    Performs new_op operation on input tensors.

    Args:
        input_tensor_a: First input tensor
        input_tensor_b: Second input tensor
        input_tensor_c: Third input tensor (or scalar)
        memory_config: Optional memory configuration

    Returns:
        Output tensor with result of new_op(a, b, c)
    )doc");
```

**File:** `ttnn/ttnn/operations/ternary.py`

> **🔷 BOILERPLATE** (structure) + **🔶 NON-BOILERPLATE** (golden function implementation)
>
> **Why the structure is boilerplate:** Every operation needs a golden function attached, the pattern is identical.
>
> **🔶 NON-BOILERPLATE - The actual golden function:**
> This function must compute the expected result using PyTorch. It's used for:
> - Unit test validation
> - Numerical accuracy verification
>
> You must implement the EXACT same mathematical operation in PyTorch that your SFPU kernel computes.

Add golden function for testing/validation:

```python
# 🔷 BOILERPLATE: Function signature and attachment pattern
# 🔶 NON-BOILERPLATE: The actual computation inside the function

def _golden_function_new_op(input_tensor_a, input_tensor_b, input_tensor_c, *args, **kwargs):
    import torch
    # 🔶 NON-BOILERPLATE: Must match your SFPU calculation exactly!
    #
    # Examples:
    # MULADD:  return input_tensor_a * input_tensor_b + input_tensor_c
    # ADDCMUL: return torch.addcmul(input_tensor_a, input_tensor_b, input_tensor_c)
    # LERP:    return torch.lerp(input_tensor_a, input_tensor_b, input_tensor_c)
    #
    return input_tensor_a * input_tensor_b + input_tensor_c  # Example for MULADD

# 🔷 BOILERPLATE: Standard attachment pattern
ttnn.attach_golden_function(ttnn.new_op, golden_function=_golden_function_new_op)
```

---

### 9. Tests

**File:** `tests/ttnn/unit_tests/operations/eltwise/test_ternary.py`

> **🔷 BOILERPLATE** (test structure) + **🔶 NON-BOILERPLATE** (test parameters and golden computation)
>
> **Why the structure is boilerplate:** All ternary operation tests follow the same pattern:
> 1. Create PyTorch inputs
> 2. Compute expected result
> 3. Convert to TTNN tensors
> 4. Run TTNN operation
> 5. Compare with expected
>
> **🔶 NON-BOILERPLATE considerations:**
>
> 1. **Input ranges:**
>    - Does your operation have input constraints? (e.g., no negative values for sqrt)
>    - Use `torch.rand` for [0,1], `torch.randn` for normal distribution
>    - Add input validation tests for edge cases
>
> 2. **Expected precision (PCC threshold):**
>    - Simple arithmetic: PCC > 0.9999
>    - Operations with divisions: PCC > 0.999
>    - Transcendental functions: PCC > 0.99
>
> 3. **Shape testing:**
>    - Test various shapes, not just [1, 1, 32, 32]
>    - Test broadcast scenarios if supported
>
> 4. **Data type coverage:**
>    - bfloat16 is the primary type
>    - bfloat8_b has lower precision - adjust PCC threshold

Add unit tests:

```python
# 🔷 BOILERPLATE: Test structure and parametrization pattern
# 🔶 NON-BOILERPLATE: Input generation, golden computation, PCC threshold

@pytest.mark.parametrize("input_shapes", [[[1, 1, 32, 32]] * 3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_new_op_ttt(device, input_shapes, dtype):
    # 🔶 NON-BOILERPLATE: Choose appropriate input ranges for your operation
    # Example: torch.randn for unrestricted, torch.rand for positive values
    torch_input_a = torch.randn(input_shapes[0])
    torch_input_b = torch.randn(input_shapes[1])
    torch_input_c = torch.randn(input_shapes[2])

    # 🔶 NON-BOILERPLATE: Golden computation - must match your operation
    torch_output = torch_new_op(torch_input_a, torch_input_b, torch_input_c)

    # 🔷 BOILERPLATE: Standard tensor conversion
    input_a = ttnn.from_torch(torch_input_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_input_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_c = ttnn.from_torch(torch_input_c, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # 🔷 BOILERPLATE: Run operation and convert back
    output = ttnn.new_op(input_a, input_b, input_c)
    output = ttnn.to_torch(output)

    # 🔶 NON-BOILERPLATE: PCC threshold depends on operation complexity
    # - Simple ops (mul, add): 0.9999
    # - Division involved: 0.999
    # - Transcendental functions: 0.99 or lower
    assert_with_pcc(torch_output, output, pcc=0.999)
```

**File:** `tests/ttnn/nightly/unit_tests/operations/eltwise/test_ternary_sharded.py`

Add sharded tests for production validation.

---

## Summary: Boilerplate vs Non-Boilerplate

| Component | Type | Effort | Notes |
|-----------|------|--------|-------|
| Enum in `ternary_op_types.hpp` | 🔷 Boilerplate | 1 line | Just add enum value |
| Struct in `ternary.hpp` | 🔷 Boilerplate | Copy-paste | Change names only |
| `invoke()` in `ternary.cpp` | 🔷 Boilerplate | Copy-paste | Change enum value |
| Kernel config map | 🔷 Boilerplate | Copy-paste | Change enum value |
| Compute defines | 🔷 Boilerplate | Copy-paste | Change function names |
| Compute kernel `.cpp` | 🔷 Boilerplate | Copy-paste | Same for all SFPU ops |
| LLK header wrapper | 🔷 Boilerplate | Copy-paste | Change function name |
| Python bindings | 🔷 Boilerplate | Copy-paste | Change name and docs |
| **SFPU calculation** | 🔶 **NON-BOILERPLATE** | **Your code** | **The actual math** |
| Golden function | 🔶 Non-Boilerplate | Your code | PyTorch reference |
| Test parameters | 🔶 Non-Boilerplate | Think | Input ranges, PCC |

**Key insight:** ~90% of adding a new ternary operation is boilerplate. The only truly custom code is:
1. The SFPU calculation formula (1-10 lines)
2. The PyTorch golden function (1-5 lines)
3. Test-specific considerations (input ranges, precision thresholds)

---

## Checklist Summary

| Step | File(s) | Action |
|------|---------|--------|
| 1 | `ternary_op_types.hpp` | Add enum value to `TernaryOpType` |
| 2 | `ternary.hpp` | Declare operation struct and register |
| 3 | `ternary.cpp` | Implement `invoke()` methods |
| 4 | `ternary_op_utils.hpp` | Add kernel names (if new kernels) |
| 5 | `ternary_op_utils.cpp` | Add kernel config map entries, file paths, compute defines |
| 6 | `kernels/compute/` | Create SFPU compute kernel(s) |
| 7 | `llk_sfpu/` (per arch) | Create LLK header with init and op functions |
| 8 | `ckernel_sfpu_*.h` | Implement SFPU calculation |
| 9 | `ternary_nanobind.cpp` | Add Python bindings |
| 10 | `ternary.py` | Add golden function |
| 11 | `test_ternary.py` | Add unit tests |

---

## Key Patterns

### Macro-Based SFPU Operations
Compute kernels use macros that are populated via compile-time defines:
- `TERNARY_SFPU_OP_INIT()` - Initialization macro
- `TERNARY_SFPU_OP_FUNC(dst0, dst1, dst2, odst)` - Operation macro

### Supported Data Types
- `BFLOAT16`
- `BFLOAT8_B`
- `BFLOAT4_B`
- Requires `TILE` layout

### Broadcast Types
- `NONE` - All inputs have same shape
- `COL_BCAST` - Column broadcast (height=1)
- `ROW_BCAST` - Row broadcast (width=1)
- `SCALAR_BCAST` - Scalar value broadcast
- `OUTER_BCAST` - Outer dimension broadcast

---

## Building and Testing

```bash
# Build
./build_metal.sh

# Run unit tests
pytest tests/ttnn/unit_tests/operations/eltwise/test_ternary.py -k "new_op" -v

# Run nightly/sharded tests
pytest tests/ttnn/nightly/unit_tests/operations/eltwise/test_ternary_sharded.py -k "new_op" -v
```
