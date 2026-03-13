# Adding a New SFPU Binary Operation

This document outlines the necessary steps to add a new SFPU (Special Function Processing Unit) binary operation in the tt-metal codebase.

## Overview

Adding a SFPU binary operation requires modifications across multiple layers of the stack, from low-level hardware kernel implementations to the Python API. The operation flow is:

```
Python API (ttnn.your_op)
    ↓
Nanobind Bindings (binary_nanobind.cpp)
    ↓
C++ Operation Templates (BinaryOperation<OpType>)
    ↓
Device Operation (BinaryDeviceOperation)
    ↓
Program Factory Selection (is_binary_sfpu_op)
    ↓
SFPU Path: ElementWiseMultiCoreSfpu::create()
    ├→ Circular Buffer Configuration
    ├→ Dataflow Kernels (Reader/Writer)
    └→ Compute Kernel (eltwise_binary_sfpu_kernel.cpp)
        ↓
    Compute Kernel API (eltwise_binary_sfpu.h)
        ↓
    Math Wrappers (llk_math_eltwise_binary_sfpu_*.h)
        ↓
    SFPU Implementation (ckernel_sfpu_binary.h)
        ↓
    Hardware SFPU Execution
```

---

## Hardware Data Flow: DRAM to DEST

Understanding how data flows through the hardware is essential for implementing SFPU operations correctly. This section explains the data path and the two ways SFPU binary operations are invoked.

### The Complete Data Path

```
DRAM → [NoC] → L1 (Circular Buffers) → [Unpacker] → SRC registers → [FPU] → DEST → [SFPU] → DEST → [Packer] → L1 → [NoC] → DRAM
```

Key components:
- **L1 / Circular Buffers (CBs)**: On-chip SRAM for streaming data
- **Unpacker**: Reads from CBs and populates SRC registers
- **SRC registers**: Source register files that feed the FPU
- **FPU**: Performs basic arithmetic (ADD, SUB, MUL) for BFLOAT16
- **DEST registers**: Holds FPU outputs and SFPU inputs/outputs
- **SFPU**: Performs transcendental and complex operations
- **Packer**: Reads from DEST and writes to CBs

### The Critical Constraint: SFPU Can Only Access DEST

| Unit | Can Read From | Writes To |
|------|---------------|-----------|
| **FPU** | SRC registers | DEST registers |
| **SFPU** | DEST registers only | DEST registers |

**The SFPU cannot access SRC registers directly.** Data must be in DEST before the SFPU can operate on it.

---

### Pattern 1: SFPU Called from FPU Kernel (Pre/Post-Processing)

**Kernel**: `eltwise_binary_kernel.cpp`
**Defines function**: `get_defines()`
**Used for**: Composite operations on BFLOAT16 where FPU does the main work and SFPU handles transcendentals

#### Data Flow

```
                        SFPU Pre-processing (optional)
                        ┌─────────────────────────────────────────────────────┐
                        │  For each input that needs pre-processing:          │
                        │  copy_tile() - uses Unpacker → datacopy → DEST      │
                        │  (for BFLOAT16, data goes through SRC registers)    │
                        │                                                     │
  Input CB (cb_in0/1)   │    ┌──────────┐   ┌───────┐   ┌────────┐           │
  ──────────────────────┼──▶ │ Unpacker │──▶│  SRC  │──▶│datacopy│           │
         (raw input)    │    └──────────┘   └───────┘   └────┬───┘           │
                        │                                    │               │
                        │                   ┌────────────────┘               │
                        │                   ▼                                │
                        │              ┌─────────┐              ┌───────┐    │   Intermediate CB
                        │              │  DEST   │─────────────▶│       │    ├──────────────────▶
                        │              │         │◀─────────────│ SFPU  │    │   (pre-processed)
                        │              └────┬────┘              │  op   │    │    (cb_inp0/1)
                        │                   │                   └───────┘    │
                        │                   ▼                                │
                        │              ┌─────────┐                           │
                        │              │  Pack   │ ──────────────────────────┘
                        │              └─────────┘
                        └─────────────────────────────────────────────────────┘

                        FPU Operation (required)
                        ┌─────────────────────────────────────────────────────┐
                        │                                                     │
  Intermediate CB       │    ┌───────┐      ┌──────┐      ┌───────┐          │
  (or original if no    │    │Unpack │      │ SRC  │      │       │          │
   pre-processing)      │    │  AB   │ ──▶  │ A+B  │ ──▶  │  FPU  │          │
  ──────────────────────┼──▶ │       │      │      │      │  op   │          │
   (cb_inp0 + cb_inp1)  │    └───────┘      └──────┘      │       │          │
                        │                                 │(add/  │          │
                        │                                 │sub/mul)          │
                        │                                 └───┬───┘          │
                        │                                     │              │
                        │                                     ▼              │
                        │                               ┌───────────┐        │
                        │                               │   DEST    │────────┼───▶ (to post-processing
                        │                               │  (result) │        │      or directly to pack)
                        │                               └───────────┘        │
                        └─────────────────────────────────────────────────────┘

                        SFPU Post-processing (optional)
                        ┌─────────────────────────────────────────────────────┐
                        │  Result already in DEST from FPU operation          │
                        │                                                     │
                        │                 ┌───────────┐    ┌───────┐          │   Output CB
                        │                 │   DEST    │    │       │          │
                        │                 │  (from    │ ──▶│ SFPU  │          ├──────────────────▶
                        │                 │   FPU)    │    │  op   │          │    (cb_out0)
                        │                 │           │ ◀──│       │          │
                        │                 └─────┬─────┘    └───────┘          │
                        │                       │                             │
                        │                       ▼                             │
                        │                 ┌───────────┐                       │
                        │                 │   Pack    │ ──────────────────────┘
                        │                 └───────────┘
                        └─────────────────────────────────────────────────────┘
```

**Key points:**
- **SFPU pre-processing** uses `copy_tile` which goes through SRC → datacopy → DEST for BFLOAT16 (direct unpack to DEST only works for 32-bit formats)
- **FPU operation** reads from SRC via `llk_unpack_AB`, writes result to DEST
- **SFPU post-processing** operates in-place on DEST (data already there from FPU)

**Why doesn't BFLOAT16 use direct unpack to DEST?**
The `copy_tile` function passes `UnpackToDestEn=true`, but direct unpack only activates for 32-bit formats (FLOAT32, INT32). In `llk_unpack_A.h`:
```cpp
if (unpack_to_dest && is_32bit_input(unpack_src_format, unpack_dst_format))
```
For BFLOAT16, `is_32bit_input()` returns false, so data follows the standard path: Unpacker → SRC → datacopy → DEST.

#### Concrete Example 1: HYPOT (SFPU pre + FPU + SFPU post)

**Operation**: `hypot(a, b) = sqrt(a² + b²)`

**Defines generated** (in `get_defines()`):
```cpp
case BinaryOpType::HYPOT:
    // SFPU pre-processing: square both inputs
    defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "PRE_IN0_0", "0", input_dtype));
    defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "PRE_IN1_0", "0", input_dtype));
    // FPU operation: add the squared values
    op_name = "add_tiles";
    op_binary_type = "EltwiseBinaryType::ELWADD";
    // SFPU post-processing: take square root
    defines.merge(get_defines(UnaryOpType::SQRT, std::nullopt, "0", idst, input_dtype));
    break;
```

**Kernel execution** (in `eltwise_binary_kernel.cpp`):
```cpp
// Phase 1: Pre-process input A - compute a²
#ifdef SFPU_OP_INIT_PRE_IN0_0  // Defined for HYPOT
    cb_wait_front(cb_in0, per_core_block_size);
    cb_reserve_back(cb_inp0, per_core_block_size);  // Intermediate buffer for squared values

    tile_regs_acquire();
    SFPU_OP_INIT_PRE_IN0_0   // square_tile_init()
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in0, i, i);     // Direct unpack: CB → DEST (UnpackToDestEn=true)
        SFPU_OP_FUNC_PRE_IN0_0       // square_tile(i) - computes a² in DEST
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        pack_tile(i, cb_inp0);       // DEST → intermediate CB (a² stored)
    }
    tile_regs_release();

    cb_pop_front(cb_in0, per_core_block_size);
    cb_push_back(cb_inp0, per_core_block_size);
#endif

// Phase 2: Pre-process input B - compute b² (same pattern)
#ifdef SFPU_OP_INIT_PRE_IN1_0
    // ... identical structure, operates on cb_in1 → cb_inp1
    // Result: b² stored in cb_inp1
#endif

// Phase 3: FPU add + SFPU sqrt
cb_wait_front(cb_inp0, per_core_block_size);   // a² ready
cb_wait_front(cb_inp1, per_core_block_size);   // b² ready
cb_reserve_back(cb_out0, per_core_block_size);

tile_regs_acquire();
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);  // add_tiles(): a² + b² → DEST

    #ifdef SFPU_OP_INIT_0
        SFPU_OP_INIT_0    // sqrt_tile_init()
        SFPU_OP_FUNC_0    // sqrt_tile(i) - computes sqrt(a² + b²) in DEST
    #endif
}
tile_regs_commit();

tile_regs_wait();
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    pack_tile(i, cb_out0);  // Final result → output CB
}
tile_regs_release();
```

#### Concrete Example 2: DIV (SFPU pre + FPU)

**Operation**: `div(a, b) = a * (1/b)`

**Defines generated**:
```cpp
case BinaryOpType::DIV:
    // SFPU pre-processing: compute reciprocal of input B
    defines.merge(get_defines(UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0"));
    // FPU operation: multiply
    op_name = "mul_tiles";
    op_binary_type = "EltwiseBinaryType::ELWMUL";
    break;
```

**Kernel execution**:
```cpp
// Phase 1: No pre-processing on input A (passes through unchanged)

// Phase 2: Pre-process input B - compute 1/b
#ifdef SFPU_OP_INIT_PRE_IN1_0  // Defined for DIV
    cb_wait_front(cb_in1, per_core_block_size);
    cb_reserve_back(cb_inp1, per_core_block_size);

    tile_regs_acquire();
    SFPU_OP_INIT_PRE_IN1_0   // recip_tile_init()
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in1, i, i);     // CB → DEST
        SFPU_OP_FUNC_PRE_IN1_0       // recip_tile(i) - computes 1/b in DEST
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        pack_tile(i, cb_inp1);       // DEST → intermediate CB (1/b stored)
    }
    tile_regs_release();

    cb_pop_front(cb_in1, per_core_block_size);
    cb_push_back(cb_inp1, per_core_block_size);
#endif

// Phase 3: FPU multiply (no post-processing)
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);  // mul_tiles(): a * (1/b) → DEST
    // No SFPU_OP_FUNC_0 for DIV
}
```

#### Concrete Example 3: GT (FPU + SFPU post)

**Operation**: `gt(a, b) = (a - b) > 0 ? 1.0 : 0.0`

**Defines generated**:
```cpp
case BinaryOpType::GT:
    // No pre-processing
    // FPU operation: subtract (op_name defaults to "sub_tiles")
    // SFPU post-processing: greater-than-zero check
    defines.merge(get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst, input_dtype));
    break;
```

**Kernel execution**:
```cpp
// No pre-processing phases

// Main loop: FPU subtract + SFPU comparison
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);  // sub_tiles(): a - b → DEST

    #ifdef SFPU_OP_INIT_0
        SFPU_OP_INIT_0    // gtz_tile_init()
        SFPU_OP_FUNC_0    // gtz_tile(i) - if DEST > 0 then 1.0 else 0.0
    #endif
}
```

#### Summary: FPU Kernel + SFPU Operations

| Operation | SFPU Pre (input A) | SFPU Pre (input B) | FPU Op | SFPU Post |
|-----------|-------------------|-------------------|--------|-----------|
| GT | - | - | sub_tiles | gtz_tile |
| LT | - | - | sub_tiles | ltz_tile |
| EQ | - | - | sub_tiles | eqz_tile |
| RSUB | neg_tile | - | add_tiles | - |
| DIV | - | recip_tile | mul_tiles | - |
| LOGADDEXP | exp_tile | exp_tile | add_tiles | log_tile |
| HYPOT | square_tile | square_tile | add_tiles | sqrt_tile |
| LOGICAL_OR | nez_tile | nez_tile | add_tiles | gtz_tile |

---

### Pattern 2: Pure SFPU Kernel

**Kernel**: `eltwise_binary_sfpu_kernel.cpp`
**Defines function**: `get_defines_fp32()`
**Used for**: FLOAT32/INT32/UINT32/UINT16 data types, or SFPU-only operations like POWER

When the FPU cannot handle the data type, the **arithmetic operation** must go through the SFPU.

The `copy_tile` function (in `tile_move_copy.h`) moves data from Circular Buffers to DEST:
```cpp
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    // Unpacker with UnpackToDestEn=true can write directly to DEST
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));

    // Math datacopy coordinates the operation
    MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        dst_tile_index, in_cb_id)));
}
```

The key is the `UnpackToDestEn` template parameter (defined as `true` in `llk_defs.h`). When enabled, the **Unpacker can write directly to DEST registers**, bypassing SRC. This is particularly used for 32-bit formats (FLOAT32, INT32) where the data path is:

```
CB → [Unpacker with unpack_to_dest=true] → DEST → [SFPU] → DEST
```

So "pure SFPU kernel" means both:
1. The **arithmetic** is done by SFPU
2. The **data path** can bypass SRC registers entirely (direct unpack to DEST)

#### Data Flow

```
                        Stage 1: Copy Input A tiles to DEST (even indices)
                        ┌─────────────────────────────────────────────────────┐
                        │  copy_tile() with UnpackToDestEn=true               │
                        │                                                     │
  Input CB A (cb_inp0)  │    ┌─────────────────────────────────┐             │
  ──────────────────────┼──▶ │ Unpacker (unpack_to_dest=true)  │             │
                        │    │                                 │             │
                        │    │ Reads from CB, writes directly  │             │
                        │    │ to DEST (bypasses SRC)          │             │
                        │    └────────────────┬────────────────┘             │
                        │                     │                              │
                        │                     ▼                              │
                        │                 ┌──────────────────────────────┐   │
                        │                 │           DEST               │   │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │   │
                        │                 │  │ A0  │     │ A1  │     │...│   │
                        │                 │  │idx 0│idx 1│idx 2│idx 3│   │   │
                        │                 │  └─────┴─────┴─────┴─────┘   │   │
                        │                 │   even indices populated     │   │
                        │                 └──────────────────────────────┘   │
                        └─────────────────────────────────────────────────────┘

                        Stage 2: Copy Input B tiles to DEST (odd indices)
                        ┌─────────────────────────────────────────────────────┐
                        │  copy_tile() with UnpackToDestEn=true               │
                        │                                                     │
  Input CB B (cb_inp1)  │    ┌─────────────────────────────────┐             │
  ──────────────────────┼──▶ │ Unpacker (unpack_to_dest=true)  │             │
                        │    │                                 │             │
                        │    │ Reads from CB, writes directly  │             │
                        │    │ to DEST (bypasses SRC)          │             │
                        │    └────────────────┬────────────────┘             │
                        │                     │                              │
                        │                     ▼                              │
                        │                 ┌──────────────────────────────┐   │
                        │                 │           DEST               │   │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │   │
                        │                 │  │ A0  │ B0  │ A1  │ B1  │...│   │
                        │                 │  │idx 0│idx 1│idx 2│idx 3│   │   │
                        │                 │  └─────┴─────┴─────┴─────┘   │   │
                        │                 │   all indices now populated  │   │
                        │                 └──────────────────────────────┘   │
                        └─────────────────────────────────────────────────────┘

                        Stage 3: SFPU Binary Operation
                        ┌─────────────────────────────────────────────────────┐
                        │                                                     │
                        │                 ┌──────────────────────────────┐    │
                        │                 │           DEST               │    │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │    │
                        │                 │  │ A0  │ B0  │ A1  │ B1  │...│    │
                        │                 │  │idx 0│idx 1│idx 2│idx 3│   │    │
                        │                 │  └──┬──┴──┬──┴─────┴─────┘   │    │
                        │                 │     │     │                  │    │
                        │                 │     ▼     ▼                  │    │
                        │                 │   ┌─────────┐                │    │
                        │                 │   │  SFPU   │                │    │
                        │                 │   │ binary  │ e.g., add_binary_tile(0, 1, 0)
                        │                 │   │   op    │                │    │
                        │                 │   └────┬────┘                │    │
                        │                 │        │                     │    │
                        │                 │        ▼  result overwrites A│    │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │    │
                        │                 │  │A0+B0│ B0  │ A1  │ B1  │...│    │
                        │                 │  │idx 0│idx 1│idx 2│idx 3│   │    │
                        │                 │  └─────┴─────┴─────┴─────┘   │    │
                        │                 └──────────────────────────────┘    │
                        └─────────────────────────────────────────────────────┘

                        Stage 4: Pack Result to Output CB
                        ┌─────────────────────────────────────────────────────┐
                        │                                                     │
                        │                 ┌──────────────────────────────┐    │   Output CB
                        │                 │           DEST               │    │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │    │
                        │                 │  │A0+B0│     │A1+B1│     │...│    ├──────────────▶
                        │                 │  │idx 0│     │idx 2│     │   │    │   (cb_out0)
                        │                 │  └──┬──┴─────┴──┬──┴─────┘   │    │
                        │                 │     │           │            │    │
                        │                 └─────┼───────────┼────────────┘    │
                        │                       ▼           ▼                 │
                        │                 ┌───────────────────────┐           │
                        │                 │   Pack (even indices) │───────────┘
                        │                 └───────────────────────┘
                        └─────────────────────────────────────────────────────┘
```

**Key points:**
- **Stage 1 & 2**: `copy_tile` with `UnpackToDestEn=true` enables direct CB → DEST path for 32-bit formats
- **Direct unpack to DEST**: Only for FLOAT32/INT32 - the Unpacker writes directly to DEST registers, bypassing SRC
- **How it works**: In `llk_unpack_A.h`, the condition `if (unpack_to_dest && is_32bit_input(...))` gates the direct path. `is_32bit_input()` checks that both source and destination formats are FLOAT32 or INT32.
- **Interleaved layout**: A tiles at even indices (0, 2, 4...), B tiles at odd indices (1, 3, 5...)
- **SFPU binary op**: Reads from two DEST indices, writes result back to first index (overwrites A)
- **Pack**: Only even indices contain results, those get packed to output CB

#### Concrete Example 1: ADD with FLOAT32

**Operation**: `add(a, b)` where both inputs are FLOAT32

**Why SFPU kernel?** The FPU only supports BFLOAT16 arithmetic natively. For FLOAT32, we must use the SFPU.

**Defines generated** (in `get_defines_fp32()`):
```cpp
case BinaryOpType::ADD:
    new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
    op_name = "add_binary_tile";
    break;

// At the end of get_defines_fp32():
new_defines.insert({"BINARY_SFPU_OP", fmt::format("{}({}, {}, {});", op_name, idst1, idst2, idst1)});
// Result: BINARY_SFPU_OP = "add_binary_tile(i*2, i*2+1, i*2);"
```

**Kernel execution** (in `eltwise_binary_sfpu_kernel.cpp`):
```cpp
cb_wait_front(cb_inp0, per_core_block_size);
cb_wait_front(cb_inp1, per_core_block_size);
cb_reserve_back(cb_out0, per_core_block_size);

tile_regs_acquire();
tile_regs_wait();

// Initialize data format conversion for the two input CBs
copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);

// Copy all A tiles to DEST at even indices
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    copy_tile(cb_inp0, i, i * 2);      // A[0]→DEST[0], A[1]→DEST[2], A[2]→DEST[4], ...
}

// Switch data format for input B
copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);

// Copy B tiles to DEST at odd indices, then perform SFPU operation
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    copy_tile(cb_inp1, i, i * 2 + 1);  // B[0]→DEST[1], B[1]→DEST[3], B[2]→DEST[5], ...

    #ifdef BINOP_INIT
        BINOP_INIT   // add_binary_tile_init()
    #endif

    #ifdef BINARY_SFPU_OP
        BINARY_SFPU_OP   // add_binary_tile(i*2, i*2+1, i*2)
                         // Reads DEST[0] and DEST[1], writes result to DEST[0]
    #endif

    pack_tile(i * 2, cb_out0);  // Pack result from DEST[0] to output CB
}

tile_regs_commit();
tile_regs_release();

cb_pop_front(cb_inp0, per_core_block_size);
cb_pop_front(cb_inp1, per_core_block_size);
cb_push_back(cb_out0, per_core_block_size);
```

#### Concrete Example 2: POWER

**Operation**: `power(a, b) = a^b`

**Why SFPU kernel?** POWER has no FPU equivalent—it's inherently an SFPU operation.

**Defines generated**:
```cpp
case BinaryOpType::POWER:
    new_defines.insert({"BINOP_INIT", fmt::format("power_binary_tile_init();")});
    op_name = "power_binary_tile";
    break;
// Result: BINARY_SFPU_OP = "power_binary_tile(i*2, i*2+1, i*2);"
```

**Kernel execution**: Same structure as ADD above, but calls:
```cpp
#ifdef BINARY_SFPU_OP
    BINARY_SFPU_OP   // power_binary_tile(i*2, i*2+1, i*2)
                     // Computes DEST[0]^DEST[1], writes to DEST[0]
#endif
```

#### Concrete Example 3: HYPOT with FLOAT32 (SFPU pre + SFPU binary + SFPU post)

**Operation**: `hypot(a, b) = sqrt(a² + b²)` where inputs are FLOAT32

**Defines generated**:
```cpp
case BinaryOpType::HYPOT:
    // SFPU pre-processing on both inputs
    new_defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "PRE_IN0_0", idst, input_a_dtype));
    new_defines.merge(get_defines(UnaryOpType::SQUARE, std::nullopt, "PRE_IN1_0", idst, input_b_dtype));
    // SFPU binary operation (not FPU!)
    new_defines.insert({"BINOP_INIT", fmt::format("add_binary_tile_init();")});
    op_name = "add_binary_tile";
    // SFPU post-processing
    new_defines.merge(get_defines(UnaryOpType::SQRT, std::nullopt, "0", idst1, input_a_dtype));
    break;
```

**Kernel execution**:
```cpp
// Phase 1: Pre-process input A with SFPU (square)
#ifdef SFPU_OP_INIT_PRE_IN0_0
    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in0, i, i);
        SFPU_OP_FUNC_PRE_IN0_0   // square_tile(i) on DEST
    }
    // pack a² to intermediate CB
#endif

// Phase 2: Pre-process input B with SFPU (square)
#ifdef SFPU_OP_INIT_PRE_IN1_0
    // ... same pattern for b²
#endif

// Phase 3: SFPU binary add + SFPU post sqrt
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    copy_tile(cb_inp0, i, i * 2);      // a² → DEST[even]
    copy_tile(cb_inp1, i, i * 2 + 1);  // b² → DEST[odd]

    BINOP_INIT                          // add_binary_tile_init()
    BINARY_SFPU_OP                      // add_binary_tile(i*2, i*2+1, i*2) → a² + b²

    #ifdef SFPU_OP_INIT_0
        SFPU_OP_INIT_0                  // sqrt_tile_init()
        SFPU_OP_FUNC_0                  // sqrt_tile(i*2) → sqrt(a² + b²)
    #endif

    pack_tile(i * 2, cb_out0);
}
```

#### Tile Layout in DEST for Binary SFPU

```
DEST register layout (interleaved for binary operations):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ A tile0 │ B tile0 │ A tile1 │ B tile1 │ A tile2 │ B tile2 │ ...
│ (idx 0) │ (idx 1) │ (idx 2) │ (idx 3) │ (idx 4) │ (idx 5) │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
     ↓         ↓
     └────┬────┘
          ▼
    SFPU binary op
    result → idx 0
```

- Input A tiles: even indices (0, 2, 4, ...)
- Input B tiles: odd indices (1, 3, 5, ...)
- Output: overwrites input A's position (even indices)

---

### Summary: When Each Pattern is Used

| Scenario | Pattern | Kernel | Key Characteristic |
|----------|---------|--------|-------------------|
| HYPOT with BFLOAT16 | Pattern 1 | FPU kernel | SFPU `square` pre → FPU `add_tiles` → SFPU `sqrt` post |
| DIV with BFLOAT16 | Pattern 1 | FPU kernel | SFPU `recip` pre → FPU `mul_tiles` |
| GT with BFLOAT16 | Pattern 1 | FPU kernel | FPU `sub_tiles` → SFPU `gtz` post |
| ADD with FLOAT32 | Pattern 2 | SFPU kernel | `copy_tile` → SFPU `add_binary_tile` |
| POWER (any dtype) | Pattern 2 | SFPU kernel | `copy_tile` → SFPU `power_binary_tile` |
| HYPOT with FLOAT32 | Pattern 2 | SFPU kernel | SFPU `square` pre → SFPU `add_binary_tile` → SFPU `sqrt` post |

### Why This Matters for Your Implementation

1. **Pattern determines kernel**: Your operation's data type determines which kernel path is used. FLOAT32 always uses the SFPU kernel.

2. **SFPU always reads from DEST**: Whether called as pre/post-processing in the FPU kernel or as the main operation in the SFPU kernel, SFPU operations always read from and write to DEST registers.

3. **Intermediate buffers for pre-processing**: When SFPU pre-processes inputs, results are packed to intermediate circular buffers before the main operation.

4. **Interleaved layout for SFPU kernel**: The pure SFPU kernel interleaves A and B tiles in DEST (even/odd indices), then the SFPU binary operation reads both and writes the result.

---

## Step 1: Add Operation Type Enum

### 1.1 Add to BinaryOpType (High-Level)

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_types.hpp`

Add your new operation to the `BinaryOpType` enum:

```cpp
enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    // ... existing operations ...
    YOUR_NEW_OP,  // Add your operation here
};
```

> **📋 BOILERPLATE**: This is pure boilerplate. You are simply adding a new identifier to an enum list.
>
> **Why it's boilerplate**: The enum is just a type-safe label used throughout the codebase to identify your operation. There's no logic here—just pick a unique, descriptive name.
>
> **What to change**: Replace `YOUR_NEW_OP` with your operation's name (e.g., `LOGICAL_AND`, `HYPOT`, `ATAN2`). Follow the existing naming convention (UPPER_SNAKE_CASE).

### 1.2 Add to Hardware BinaryOp Enum (If Needed)

**File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_defs.h`

If your operation is a fundamental SFPU operation (not composed of existing ones), add it to the hardware-level enum:

```cpp
enum class BinaryOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    // ... existing operations ...
    YOUR_NEW_OP = <next_value>,
};
```

> **⚠️ NOT BOILERPLATE**: This step requires architectural understanding and decision-making.
>
> **Why it's NOT boilerplate**:
> - **Decision required**: You must determine if your operation is "fundamental" or can be composed from existing primitives. For example, `a * b + c` might be composed from MUL and ADD, while `atan2(a, b)` is a fundamental operation with no simpler decomposition.
> - **Hardware implications**: Adding to this enum may affect the hardware dispatch mechanism. The enum value becomes part of the binary interface between software and hardware.
> - **Shared across architectures**: This file is in the LLK (Low-Level Kernel) submodule shared across Blackhole, Wormhole, and potentially future architectures.
>
> **Considerations**:
> - If your operation CAN be expressed as a composition of existing SFPU ops, you might not need this step at all—implement it as a composite in the higher-level kernel.
> - If adding here, ensure the numeric value doesn't conflict with existing entries.
> - Check if the operation truly needs hardware-level dispatch or can be handled purely in software.

---

## Step 2: Implement Low-Level LLK Kernels

These files must be implemented for each supported architecture (Blackhole, Wormhole B0).

### 2.1 Create SFPU Kernel Implementation

**Files** (one per architecture):
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_your_op.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_your_op.h`

Example template:

```cpp
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_your_op(const uint dst_index_in0,
                               const uint dst_index_in1,
                               const uint dst_index_out) {
    // Read input tiles from destination registers
    _llk_sfpu_calc_tile_init_<APPROXIMATION_MODE, 3, 1>(dst_index_in0);

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load values from both inputs
        vFloat in0 = dst_reg[dst_index_in0 * 64 + d];
        vFloat in1 = dst_reg[dst_index_in1 * 64 + d];

        // Perform your operation
        vFloat result = /* your computation using in0 and in1 */;

        // Store result
        dst_reg[dst_index_out * 64 + d] = result;
    }
}

}  // namespace ckernel::sfpu
```

> **🔴 NOT BOILERPLATE - THIS IS THE CORE IMPLEMENTATION**: This is the most critical non-boilerplate section. Here you implement the actual mathematical computation.
>
> **Why it's NOT boilerplate**:
>
> 1. **Mathematical correctness**: You must implement the actual algorithm. This requires understanding:
>    - The mathematical definition of your operation
>    - Numerical stability concerns (overflow, underflow, precision loss)
>    - Edge cases (division by zero, negative inputs for sqrt-like ops, etc.)
>
> 2. **SFPI intrinsics knowledge**: The SFPU uses a custom instruction set (SFPI). You need to understand:
>    - `vFloat` - vector float type representing multiple values processed in parallel
>    - `dst_reg[]` - destination register file for tile data
>    - Available SFPI operations (add, mul, reciprocal, log, exp, etc.)
>    - Which operations are native vs. need approximation
>
> 3. **Approximation modes**: The `APPROXIMATION_MODE` template parameter affects accuracy vs. speed:
>    - `true`: Use faster but less accurate approximations (e.g., polynomial approximations)
>    - `false`: Use more accurate but slower implementations
>    - You must implement BOTH paths appropriately for your operation
>
> 4. **Architecture differences**: Blackhole and Wormhole may have:
>    - Different SFPU capabilities
>    - Different register layouts
>    - Different performance characteristics
>    - You may need different implementations per architecture
>
> 5. **Iteration count**: `ITERATIONS = 8` processes 8 faces of a tile. This is tied to tile geometry (32x32 tiles = 8 faces of 64 elements each). Understand why this value exists.
>
> 6. **Register indexing**: The formula `dst_index * 64 + d` reflects the tile memory layout. Incorrect indexing corrupts data.
>
> **Common pitfalls**:
> - Forgetting to handle NaN/Inf inputs
> - Incorrect loop bounds causing memory corruption
> - Using operations not available on SFPU (must use SFPI intrinsics)
> - Poor numerical stability (e.g., computing `a/b` directly instead of `a * recip(b)`)

### 2.2 Create Math Wrapper

**Files** (one per architecture):
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_your_op.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_your_op.h`

```cpp
#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_your_op.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_your_op(
    uint dst_index_in0,
    uint dst_index_in1,
    uint dst_index_out,
    int vector_mode = VectorMode::RC) {

    llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_your_op<APPROXIMATE>,
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_your_op_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::your_op, APPROXIMATE>();
}

}  // namespace ckernel
```

> **📋 BOILERPLATE**: This is largely boilerplate—a thin wrapper connecting the SFPU implementation to the LLK math interface.
>
> **Why it's boilerplate**: The structure is identical for all binary SFPU operations:
> 1. Include the standard headers
> 2. Create a templated function that calls `llk_math_eltwise_binary_sfpu_params` with your compute function
> 3. Create an init function that calls `llk_math_eltwise_binary_sfpu_init`
>
> **What to change**:
> - Function names: `llk_math_eltwise_binary_sfpu_your_op` → your operation name
> - Include: `ckernel_sfpu_your_op.h` → your SFPU implementation header
> - Compute function reference: `calculate_your_op<APPROXIMATE>` → your function name
> - SfpuType: `SfpuType::your_op` → your operation's SfpuType enum value
>
> **Minor consideration**: The `vector_mode` parameter controls how the operation processes the tile (row-wise, column-wise, or both). For most binary ops, `VectorMode::RC` (row and column) is correct, but special operations might need different modes.

---

## Step 3: Add Compute Kernel API

**File**: `tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h`

Add API functions for use in compute kernels:

```cpp
/**
 * Performs your_op operation on two tiles.
 */
ALWI void your_op_binary_tile_init() {
    MATH(llk_math_eltwise_binary_sfpu_your_op_init<APPROX>());
}

ALWI void your_op_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH(llk_math_eltwise_binary_sfpu_your_op<APPROX>(idst0, idst1, odst));
}
```

> **📋 BOILERPLATE**: This is boilerplate—a simple API wrapper exposing your LLK functions to compute kernels.
>
> **Why it's boilerplate**: The pattern is identical for every binary SFPU operation:
> 1. An `_init()` function that initializes the SFPU for your operation
> 2. A `_tile()` function that performs the operation on tile indices
>
> Both just wrap the LLK functions with the `MATH()` macro (which handles the math thread context).
>
> **What to change**:
> - Function names: `your_op_binary_tile_init`, `your_op_binary_tile`
> - LLK function calls: Match your LLK wrapper function names from Step 2.2
>
> **Note**: `ALWI` means "Always Inline" - these functions are always inlined for performance. `APPROX` is a compile-time constant that determines approximation mode.

---

## Step 4: Update Binary Operation Utilities

### 4.1 Add Operation Mapping

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`

Update `get_defines()` to map your operation to kernel defines:

```cpp
std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<DataType> input_dtype,
    const std::optional<DataType> output_dtype,
    const std::optional<tt::tt_metal::UnaryWithParam>& fused_activations,
    const std::optional<tt::tt_metal::UnaryWithParam>& input_tensor_a_activation) {

    std::map<std::string, std::string> defines;

    switch (op_type) {
        // ... existing cases ...
        case BinaryOpType::YOUR_NEW_OP:
            defines["SFPU_OP_YOUR_OP"] = "1";
            // Add any additional defines needed
            break;
    }

    return defines;
}
```

> **📋 BOILERPLATE**: This is boilerplate—adding a switch case that sets a preprocessor define.
>
> **Why it's boilerplate**: Every operation needs exactly this pattern: a case that sets a unique `SFPU_OP_*` define. The define is used by the compute kernel (`#ifdef`) to select which operation to execute.
>
> **What to change**:
> - `BinaryOpType::YOUR_NEW_OP` → your enum value from Step 1.1
> - `"SFPU_OP_YOUR_OP"` → a unique define name matching what you'll use in Step 5.2
>
> **Slight variation**: If your operation needs additional compile-time parameters (e.g., an exponent value, a threshold), you'd add more defines here. But for most operations, just the single SFPU_OP define suffices.

### 4.2 Add to EltwiseBinaryType Mapping

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.hpp`

If using FPU path as well, add mapping:

```cpp
inline EltwiseBinaryType get_binary_type(BinaryOpType op_type) {
    switch (op_type) {
        // ... existing cases ...
        case BinaryOpType::YOUR_NEW_OP:
            return EltwiseBinaryType::ELWSE_BINARY_YOUR_OP;
    }
}
```

> **📋 BOILERPLATE (if applicable)**: This is boilerplate when needed, but may be skipped entirely.
>
> **Why it might be boilerplate**: If your operation supports both SFPU and FPU paths, you need this mapping. The pattern is a simple switch case.
>
> **Why you might skip this**: If your operation is SFPU-only (many complex math operations are), you don't need an FPU type mapping. The `is_binary_sfpu_op()` check in Step 5.1 will ensure only the SFPU path is used.
>
> **What to change**: Map your `BinaryOpType` to the corresponding `EltwiseBinaryType`.

---

## Step 5: Update Device Operation

### 5.1 Add SFPU Operation Check

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`

Update `is_binary_sfpu_op()` to include your operation:

```cpp
bool is_binary_sfpu_op(BinaryOpType val, DataType a, DataType b) {
    switch (val) {
        // ... existing cases ...
        case BinaryOpType::YOUR_NEW_OP:
            // Specify which data types are supported
            return (a == DataType::FLOAT32 || a == DataType::BFLOAT16) &&
                   (b == DataType::FLOAT32 || b == DataType::BFLOAT16);
        default:
            return false;
    }
}
```

> **⚠️ PARTIALLY BOILERPLATE**: The switch-case structure is boilerplate, but the data type conditions require thought.
>
> **What's boilerplate**: Adding the case statement itself—every SFPU op needs one.
>
> **What's NOT boilerplate - Data Type Support**:
> You must decide which input data type combinations your operation supports. Consider:
>
> 1. **SFPU capabilities**: The SFPU primarily handles floating-point operations. Integer operations may not be supported or may need special handling.
>
> 2. **Mathematical validity**: Some operations are only defined for certain types:
>    - Bitwise operations (`AND`, `OR`) typically need integer types
>    - Transcendental functions (`log`, `exp`) need floating-point
>    - Mixed-type operations may need type promotion rules
>
> 3. **Precision requirements**:
>    - `FLOAT32` offers higher precision but lower throughput
>    - `BFLOAT16` is faster but loses precision (only 7 mantissa bits)
>    - Your algorithm's numerical stability may favor one over the other
>
> 4. **Mixed-type inputs**: Can input A be FLOAT32 while input B is BFLOAT16? The SFPU may or may not handle this efficiently.
>
> **Example considerations**:
> - `POW(a, b)`: Needs float types; integer exponents might use a different fast path
> - `LOGICAL_AND(a, b)`: Might accept any type (interpreting as boolean)
> - `ATAN2(a, b)`: Strictly floating-point

### 5.2 Update Compute Kernel Selection

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

Add your operation to the compute kernel:

```cpp
#ifdef SFPU_OP_YOUR_OP
    your_op_binary_tile_init();
#endif

// In the main compute loop:
#ifdef SFPU_OP_YOUR_OP
    your_op_binary_tile(i, i + 1, i);
#endif
```

> **📋 BOILERPLATE**: This is boilerplate—adding `#ifdef` blocks that call your API functions.
>
> **Why it's boilerplate**: The pattern is identical for every operation:
> 1. An `#ifdef` block in the init section calling `your_op_binary_tile_init()`
> 2. An `#ifdef` block in the compute loop calling `your_op_binary_tile()`
>
> **What to change**:
> - `SFPU_OP_YOUR_OP` → match the define from Step 4.1
> - `your_op_binary_tile_init()` and `your_op_binary_tile()` → match your API from Step 3
>
> **Note on tile indices**: The pattern `(i, i + 1, i)` means:
> - Read input 0 from tile index `i`
> - Read input 1 from tile index `i + 1`
> - Write output to tile index `i` (overwriting input 0)
>
> This is the standard binary operation pattern. The circular buffer management ensures tiles are properly loaded before this point.

---

## Step 6: Register C++ Operation

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp`

Register your operation:

```cpp
// Add operation instantiation
constexpr auto your_op = ttnn::register_operation<
    "ttnn::your_op",
    BinaryOperationSfpu<BinaryOpType::YOUR_NEW_OP>>();
```

> **📋 BOILERPLATE**: This is pure boilerplate—a single line that registers your operation with the TTNN framework.
>
> **Why it's boilerplate**: The template machinery does all the work. You just:
> 1. Provide the Python-visible name (`"ttnn::your_op"`)
> 2. Specify the operation type template parameter
>
> **What to change**:
> - `your_op` → C++ variable name for the operation
> - `"ttnn::your_op"` → Python-visible operation name
> - `BinaryOpType::YOUR_NEW_OP` → your enum value from Step 1.1
>
> **Note**: `BinaryOperationSfpu` vs `BinaryOperation`: Use `BinaryOperationSfpu` for SFPU-only operations. The non-Sfpu variant may include FPU fallback logic.

---

## Step 7: Add Python Bindings

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp`

Add nanobind bindings for Python access:

```cpp
detail::bind_binary_operation(
    module,
    ttnn::your_op,
    R"doc(
    Performs element-wise your_op operation on two tensors.

    Args:
        input_tensor_a (ttnn.Tensor): First input tensor.
        input_tensor_b (ttnn.Tensor): Second input tensor.

    Keyword Args:
        memory_config (ttnn.MemoryConfig, optional): Memory configuration.
        dtype (ttnn.DataType, optional): Output data type.
        activations (List[str], optional): Fused activations.

    Returns:
        ttnn.Tensor: Result tensor.

    Example::
        >>> result = ttnn.your_op(tensor_a, tensor_b)
    )doc",
    "your_op");
```

> **📋 BOILERPLATE**: This is boilerplate—the binding call structure is identical for all binary operations.
>
> **Why it's boilerplate**: The `detail::bind_binary_operation` helper function handles all the complexity of:
> - Argument parsing
> - Type conversion
> - Optional parameter handling
> - Error handling
>
> You just fill in the template.
>
> **What to change**:
> - `ttnn::your_op` → your C++ operation from Step 6
> - The docstring → describe YOUR operation (but the structure/Args/Returns are the same)
> - `"your_op"` → the final Python function name
>
> **Documentation note**: While the binding itself is boilerplate, writing a GOOD docstring is not—clearly explain what the operation does mathematically, any constraints on inputs, and expected behavior.

---

## Step 8: Add Python Golden Function

**File**: `ttnn/ttnn/operations/binary.py`

Add a golden (reference) function for testing:

```python
def _golden_function_your_op(input_tensor_a, input_tensor_b, *args, **kwargs):
    import torch
    # Implement reference computation using PyTorch
    result = torch.your_op(input_tensor_a, input_tensor_b)
    return result

ttnn.attach_golden_function(ttnn.your_op, golden_function=_golden_function_your_op)
```

> **⚠️ NOT BOILERPLATE**: The function structure is templated, but the implementation requires correctness verification.
>
> **Why it's NOT boilerplate**:
>
> 1. **Mathematical correctness**: The golden function IS the definition of correctness. If it's wrong, your tests will pass incorrect implementations.
>
> 2. **PyTorch equivalence**: You must find or implement the equivalent operation in PyTorch:
>    - Direct mapping: `torch.pow`, `torch.atan2` exist directly
>    - Composition: `torch.hypot(a, b)` or `torch.sqrt(a**2 + b**2)`
>    - Custom: Some operations may not exist in PyTorch and need manual implementation
>
> 3. **Edge case handling**: Your golden function should handle:
>    - NaN inputs → what should the output be?
>    - Inf inputs → defined behavior?
>    - Zero inputs → division by zero cases?
>    - Type promotion rules
>
> 4. **Broadcasting semantics**: Ensure your golden function's broadcasting matches TTNN's behavior.
>
> **Example complexities**:
> ```python
> # Simple direct mapping
> def _golden_pow(a, b, *args, **kwargs):
>     return torch.pow(a, b)
>
> # Needs composition
> def _golden_hypot(a, b, *args, **kwargs):
>     return torch.sqrt(a**2 + b**2)
>
> # Needs careful handling
> def _golden_safe_div(a, b, *args, **kwargs):
>     return torch.where(b != 0, a / b, torch.zeros_like(a))
> ```

---

## Step 9: Add Unit Tests

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_binary.py` (or create new test file)

```python
import pytest
import torch
import ttnn

@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 64)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_your_op(device, shape, dtype):
    torch_input_a = torch.randn(shape, dtype=torch.float32)
    torch_input_b = torch.randn(shape, dtype=torch.float32)

    # Expected result using PyTorch
    torch_output = torch.your_op(torch_input_a, torch_input_b)

    # TTNN computation
    input_a = ttnn.from_torch(torch_input_a, dtype=dtype, device=device)
    input_b = ttnn.from_torch(torch_input_b, dtype=dtype, device=device)

    output = ttnn.your_op(input_a, input_b)
    output = ttnn.to_torch(output)

    # Verify results
    assert torch.allclose(output, torch_output, rtol=1e-2, atol=1e-2)
```

> **🔴 NOT BOILERPLATE - CRITICAL FOR CORRECTNESS**: The test structure is templated, but designing good tests requires deep understanding.
>
> **Why it's NOT boilerplate**:
>
> 1. **Test Input Selection**:
>    - `torch.randn()` gives random normal values—but is this appropriate?
>    - For `log(a, b)`: inputs must be positive
>    - For `atan2(y, x)`: test all four quadrants
>    - For `pow(a, b)`: test negative bases with integer vs. fractional exponents
>    - Consider: zeros, ones, very large values, very small values, negative values
>
> 2. **Shape Coverage**:
>    - `(1, 1, 32, 32)` is a single tile—minimum case
>    - Multi-tile shapes test tiling logic
>    - Non-tile-aligned shapes (e.g., `(1, 1, 33, 33)`) test padding
>    - Broadcasting shapes: `(1, 1, 32, 32)` with `(1, 1, 1, 32)`
>
> 3. **Tolerance Selection** (`rtol`, `atol`):
>    - `rtol=1e-2` means 1% relative error allowed
>    - `atol=1e-2` means absolute error up to 0.01
>    - **These values are operation-specific!**
>      - Simple operations (add, mul): tighter tolerance possible
>      - Transcendental operations (log, exp, trig): looser tolerance needed
>      - BFLOAT16: much looser than FLOAT32
>    - Wrong tolerances = flaky tests or missed bugs
>
> 4. **Edge Case Tests** (often missing from basic template):
>    ```python
>    def test_your_op_edge_cases(device):
>        # Test with zeros
>        # Test with infinities
>        # Test with NaN
>        # Test with very large/small values
>        # Test with negative values (if applicable)
>    ```
>
> 5. **Memory Configuration Tests**:
>    - DRAM vs L1 memory
>    - Different shard configurations
>    - These can expose bugs not visible with default configs
>
> 6. **Data Type Combinations**:
>    - Same dtype for both inputs
>    - Mixed dtypes (if supported)
>    - Test all dtypes claimed to be supported in Step 5.1

---

## File Summary

| Step | Files to Modify/Create | Boilerplate? |
|------|------------------------|--------------|
| 1.1 | `binary_op_types.hpp` | ✅ Yes |
| 1.2 | `ckernel_defs.h` | ❌ No - requires architectural decision |
| 2.1 | `ckernel_sfpu_your_op.h` (per arch) | ❌ No - core math implementation |
| 2.2 | `llk_math_eltwise_binary_sfpu_your_op.h` (per arch) | ✅ Yes |
| 3 | `eltwise_binary_sfpu.h` | ✅ Yes |
| 4.1 | `binary_op_utils.cpp` | ✅ Yes |
| 4.2 | `binary_op_utils.hpp` | ✅ Yes (if applicable) |
| 5.1 | `binary_device_operation.cpp` | ⚠️ Partial - dtype decisions needed |
| 5.2 | `eltwise_binary_sfpu_kernel.cpp` | ✅ Yes |
| 6 | `binary.hpp` | ✅ Yes |
| 7 | `binary_nanobind.cpp` | ✅ Yes |
| 8 | `binary.py` | ❌ No - correctness definition |
| 9 | `test_binary.py` | ❌ No - requires test design |

---

## Architecture-Specific Files

For each supported architecture, you need to create/modify LLK files:

### Blackhole
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_your_op.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_your_op.h`

### Wormhole B0
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_your_op.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_your_op.h`

> **Note on architecture-specific code**: While the math wrapper (Step 2.2) is boilerplate across architectures, the SFPU kernel implementation (Step 2.1) may need architecture-specific optimizations or workarounds. Always test on all target architectures.

---

## Tips and Best Practices

1. **Study Existing Operations**: Look at similar operations (e.g., `pow`, `xlogy`) for reference implementations.

2. **Data Type Support**: Carefully consider which data types your operation should support (FLOAT32, BFLOAT16, INT32, etc.).

3. **Approximation Modes**: Some operations support fast approximation modes. Consider implementing both exact and approximate versions.

4. **Broadcasting**: Consider how your operation handles broadcasting (height, width, or both).

5. **Fused Activations**: SFPU operations can support fused activation functions (ReLU, GELU, etc.) for better performance.

6. **Testing**: Test with various tensor shapes, data types, and memory configurations.

7. **Performance**: Profile your operation on hardware to ensure it meets performance requirements.

---

## Summary: Where Your Engineering Effort Goes

| Category | Steps | Effort Level | Why |
|----------|-------|--------------|-----|
| **Core Algorithm** | 2.1 | 🔴 High | This IS your operation - math, precision, edge cases |
| **Correctness Definition** | 8, 9 | 🔴 High | Tests and golden function define "correct" |
| **Architectural Decisions** | 1.2, 5.1 | 🟡 Medium | Data types, hardware enums, capability decisions |
| **Plumbing/Wiring** | 1.1, 2.2, 3, 4, 5.2, 6, 7 | 🟢 Low | Copy-paste with name substitutions |

For a new binary SFPU operation, expect to spend:
- **80% of your time** on Step 2.1 (SFPU implementation) and Steps 8-9 (testing)
- **15% of your time** on architectural decisions and debugging
- **5% of your time** on boilerplate modifications
