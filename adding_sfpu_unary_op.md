# Adding a New SFPU Unary Operation

This guide documents the necessary steps to add a new SFPU (Special Function Processing Unit) unary operation to the tt-metal codebase.

## Overview

Adding a new SFPU unary operation requires modifications across multiple layers of the stack, from low-level hardware kernels to the Python API. The full chain is:

```
Python API → Nanobind Bindings → C++ Executor → Device Operation →
Program Factory → Compute Kernel → Compute API → SFPU Kernel
```

---

## Hardware Data Flow: DRAM → DEST Registers

Understanding how data reaches the DEST registers is essential before implementing SFPU operations. The SFPU (Special Function Processing Unit) can **only operate on data in DEST registers** - it has no direct access to DRAM or Circular Buffers.

### Complete Data Path

```
DRAM → L1 (via NoC) → Circular Buffer → Unpacker → DEST Registers → SFPU → Packer → CB → L1 → DRAM
```

### The Critical Distinction: copy_tile vs. FPU Operations

There are two ways data can reach DEST registers, and understanding this distinction is crucial:

#### Pure SFPU Operations (e.g., ReLU, Sigmoid, Exp)

For operations that **only exist on the SFPU** (no FPU equivalent), data must be **explicitly copied** to DEST via `copy_tile()`:

```cpp
// From eltwise_sfpu.cpp - the standard unary SFPU compute kernel
for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
    tile_regs_acquire();                        // Reserve DEST registers

    cb_wait_front(tt::CBIndex::c_0, 1);         // Wait for data in input CB
    copy_tile(tt::CBIndex::c_0, 0, 0);          // CB[c_0][0] → DEST[0]

    // NOW SFPU can operate on DEST[0]
    #ifdef SFPU_OP_CHAIN_0
    SFPU_OP_CHAIN_0                             // e.g., relu_tile(0)
    #endif

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_2);             // DEST[0] → output CB

    cb_pop_front(tt::CBIndex::c_0, 1);
    tile_regs_release();
}
```

**Why `copy_tile()` is necessary:** The SFPU is a co-processor that can only read from and write to DEST registers. It has no direct path from Circular Buffers. The `copy_tile()` function moves data from CB to DEST, enabling SFPU processing.

#### FPU Operations (e.g., Matmul, Multiply-Accumulate)

For operations implemented in the **FPU** (Floating Point Unit), the FPU naturally outputs results directly to DEST as part of its computation:

```cpp
// From bmm.cpp - matrix multiplication compute kernel
cb_wait_front(tt::CBIndex::c_0, 1);  // Input A in CB
cb_wait_front(tt::CBIndex::c_1, 1);  // Input B in CB

// FPU reads from SrcA/SrcB and outputs DIRECTLY to DEST[0]
// No copy_tile() needed!
matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);

pack_tile(0, tt::CBIndex::c_16);     // DEST → output CB
```

**Why no `copy_tile()` is needed:** The FPU reads operands from source registers (SrcA/SrcB) which are populated by the unpacker, and its output naturally accumulates into DEST registers.

### Chained Operations: FPU → SFPU

When an FPU operation is followed by an SFPU operation (e.g., matmul followed by relu), no intermediate `copy_tile()` is needed because the FPU result is already in DEST:

```cpp
// Matmul outputs to DEST[0]
matmul_tiles(c_0, c_1, 0, 0, 0);

// SFPU can immediately operate on DEST[0] - no copy_tile needed!
relu_tile(0);

// Pack the final result
pack_tile(0, c_out);
```

### Summary Table

| Operation Type | Example Ops | Needs `copy_tile()`? | Reason |
|----------------|-------------|----------------------|--------|
| **Pure SFPU** | relu, sigmoid, exp, tanh | **YES** | SFPU only accesses DEST registers |
| **FPU** | matmul, mul-accumulate | **NO** | FPU naturally outputs to DEST |
| **FPU + SFPU chain** | matmul → relu | **NO** | FPU result already in DEST |

### Detailed Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DRAM: Input tensor data                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    Reader Kernel: noc_async_read_page()
                    (transfers data via Network-on-Chip)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ L1 Memory: Circular Buffer CB[c_0]                                           │
│ - Input staging area in local SRAM                                           │
│ - Reader pushes tiles here, compute kernel consumes them                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    cb_wait_front() - wait for data
                    copy_tile() - for pure SFPU ops
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEST Registers (16 tiles × 2KB = 32KB total)                                 │
│ - DEST[0]: Current tile being processed                                      │
│ - SFPU reads and writes here (in-place computation)                          │
│ - FPU accumulates results here                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    SFPU operation: e.g., relu_tile(0)
                    (modifies DEST[0] in-place)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEST Registers: Result in DEST[0]                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    pack_tile(0, CB[c_2])
                    (DEST → output circular buffer)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ L1 Memory: Circular Buffer CB[c_2]                                           │
│ - Output staging area                                                        │
│ - Compute kernel pushes here, writer kernel consumes                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    Writer Kernel: noc_async_write_page()
                    (transfers data via Network-on-Chip)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DRAM: Output tensor data                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Files in the Data Flow

| Component | File Location | Purpose |
|-----------|---------------|---------|
| **Reader Kernel** | `ttnn/.../unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | DRAM → L1 via NoC |
| **Compute Kernel** | `ttnn/.../unary/device/kernels/compute/eltwise_sfpu.cpp` | copy_tile + SFPU ops |
| **Writer Kernel** | `ttnn/.../unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | L1 → DRAM via NoC |
| **copy_tile API** | `tt_metal/include/compute_kernel_api/tile_move_copy.h` | CB → DEST transfer |
| **Program Factory** | `ttnn/.../unary/device/unary_program_factory.cpp` | Orchestrates all kernels |

### DEST Register Lifecycle

The compute kernel manages DEST registers through a well-defined lifecycle:

```cpp
tile_regs_acquire();    // 1. Reserve DEST registers for this iteration
                        //    (blocks until registers are available)

copy_tile(...);         // 2. Load data: CB → DEST
                        //    (only for pure SFPU ops)

SFPU_OP_CHAIN_0;        // 3. Process: SFPU operates on DEST in-place

tile_regs_commit();     // 4. Signal computation complete
tile_regs_wait();       // 5. Wait for all SFPU operations to finish

pack_tile(...);         // 6. Store result: DEST → output CB

tile_regs_release();    // 7. Release DEST registers for next iteration
```

### Concrete Example 1: Pure SFPU Operation (Standalone ReLU)

**File:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

This is the actual compute kernel used for all standalone unary SFPU operations (relu, sigmoid, exp, etc.). Notice that `copy_tile()` is **required** to move data into DEST before the SFPU operation can execute:

```cpp
// eltwise_sfpu.cpp - Complete kernel for pure SFPU operations
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // ================================================================
            // KEY STEP: copy_tile() moves data from CB to DEST
            // ================================================================
            // Without this, SFPU has no data to operate on!
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);  // CB[c_0] tile 0 → DEST[0]

            // ================================================================
            // SFPU operation executes on data now in DEST[0]
            // ================================================================
            // SFPU_OP_CHAIN_0 expands to e.g., "relu_tile_init(); relu_tile(0);"
            // The operation reads from DEST[0] and writes result back to DEST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();
            tile_regs_wait();

            // Pack result from DEST[0] to output CB
            pack_tile(0, tt::CBIndex::c_2);

            cb_pop_front(tt::CBIndex::c_0, 1);
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
```

**Data flow for this kernel:**
```
DRAM → Reader → CB[c_0] → copy_tile() → DEST[0] → SFPU op → DEST[0] → pack_tile() → CB[c_2] → Writer → DRAM
```

### Concrete Example 2: FPU + SFPU Chain (Matmul with Fused Activation)

**File:** `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

This kernel performs matrix multiplication (FPU operation) followed by an optional SFPU activation (e.g., ReLU, GELU). Notice that **no `copy_tile()` is needed** because `matmul_block()` outputs directly to DEST:

```cpp
// bmm_large_block_zm_fused_bias_activation.cpp - Relevant excerpt
// (simplified for clarity)

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

// SFPU activation is initialized at the start if fused
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION  // e.g., "relu_tile_init();" or "gelu_tile_init();"
#endif

// ... inside the computation loop ...

for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
    tile_regs_acquire();

    // ... reload logic if needed ...

    // ================================================================
    // FPU OPERATION: matmul_block() outputs DIRECTLY to DEST
    // ================================================================
    // No copy_tile() needed! The FPU naturally writes results to DEST.
    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
        matmul_block(
            in0_cb_id, in1_cb_id,
            in0_index, in1_index, dst_index,
            in1_transpose_tile,
            out_subblock_w, out_subblock_h, in0_block_w);
        in0_index++;
        in1_index += in1_block_w;
    }
    // At this point, matmul results are in DEST registers

    if (last_out) {
        // ================================================================
        // SFPU ACTIVATION: Operates on data ALREADY in DEST
        // ================================================================
        // No copy_tile() between matmul and activation!
        // SFPU reads from DEST (where matmul wrote) and writes back to DEST.
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION  // e.g., "relu_tile(i);" or "gelu_tile(i);"
        }
#endif
        tile_regs_commit();

        // Pack final result (after matmul + activation) from DEST to output CB
        cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
        tile_regs_wait();
        pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);
        tile_regs_release();
        cb_push_back(mm_out_cb_id, out_subblock_num_tiles);
    }
    // ...
}
```

**Data flow for this kernel:**
```
DRAM → Reader → CB[c_0], CB[c_1] → matmul_block() → DEST → SFPU activation → DEST → pack_tile() → CB[c_out] → Writer → DRAM
                                   ↑                  ↑
                            FPU outputs here    SFPU operates here
                            (no copy_tile)      (no copy_tile needed)
```

### Key Takeaway

The fundamental difference is **where data originates**:

| Scenario | Data Source for SFPU | copy_tile Required? |
|----------|---------------------|---------------------|
| **Standalone SFPU op** (e.g., `ttnn.relu(x)`) | Circular Buffer (input tensor) | **YES** - must copy CB → DEST |
| **Fused FPU+SFPU** (e.g., `ttnn.matmul(..., activation="relu")`) | DEST (from prior FPU op) | **NO** - data already in DEST |

This fusion pattern is why operations like `matmul` with activation are more efficient than separate `matmul` followed by `relu` - it avoids the round-trip through circular buffers and the extra `copy_tile()` overhead.

---

## Step 1: Implement the SFPU Kernel (Architecture-Specific)

Create the low-level SFPU kernel implementation for each supported architecture.

### File Locations:
- **Wormhole B0**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<op_name>.h`
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_<op_name>.h`

### Template:
```cpp
// ckernel_sfpu_<op_name>.h
#pragma once

// ============================================================================
// BOILERPLATE: Standard includes required for all SFPU operations
// ----------------------------------------------------------------------------
// Why boilerplate? These headers provide:
// - ckernel_sfpu_calculator.h: SFPI intrinsics (vFloat, dst_reg, etc.)
// - llk_math_eltwise_unary_sfpu_params.h: Parameter passing utilities
// - llk_math_eltwise_unary_sfpu_init.h: Initialization macros
// You always need these exact includes for any SFPU unary operation.
// ============================================================================
#include "ckernel_sfpu_calculator.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

// ============================================================================
// BOILERPLATE: Namespace structure
// ----------------------------------------------------------------------------
// Why boilerplate? All SFPU kernels must reside in ckernel::sfpu namespace.
// This is enforced by the LLK (Low-Level Kernel) infrastructure.
// ============================================================================
namespace ckernel {
namespace sfpu {

// ============================================================================
// BOILERPLATE: Function signature and iteration loop
// ----------------------------------------------------------------------------
// Why boilerplate?
// - Template parameters are standard: APPROXIMATION_MODE controls precision
//   vs speed tradeoff, ITERATIONS defaults to 8 (processes 8 vector elements)
// - The for-loop structure iterates through destination register elements
// - dst_reg[0] access pattern and dst_reg++ increment are fixed by hardware
//
// CHANGE REQUIRED: Replace <op_name> with your operation name
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_<op_name>() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        // ====================================================================
        // NOT BOILERPLATE: The actual mathematical computation
        // --------------------------------------------------------------------
        // Why NOT boilerplate? This is the core of your operation and requires
        // understanding of:
        //
        // 1. SFPI Intrinsics: You must use SFPI (SFPU Intrinsic) operations:
        //    - vFloat: Vector float type holding multiple elements
        //    - vInt: Vector integer type
        //    - Arithmetic: +, -, *, / on vFloat
        //    - Built-in functions: sfpu_exp(), sfpu_log(), sfpu_reciprocal(), etc.
        //    - Conditional: vFloat::conditional_assign(), v_if/v_elseif/v_endif
        //
        // 2. Numerical considerations:
        //    - Input range: What values does your op accept?
        //    - Precision: Is approximation acceptable? (APPROXIMATION_MODE)
        //    - Edge cases: Handle inf, nan, zero, negative inputs
        //
        // 3. Performance: SFPU has limited instruction set - complex ops may
        //    need to be decomposed into simpler primitives
        //
        // Example implementations:
        //   abs:     dst_reg[0] = sfpu_abs(v);
        //   neg:     dst_reg[0] = -v;
        //   relu:    dst_reg[0] = vConst0 > v ? vConst0 : v;
        //   sigmoid: dst_reg[0] = sfpu_reciprocal(1.0f + sfpu_exp(-v));
        // ====================================================================
        dst_reg[0] = /* result */;

        // ====================================================================
        // BOILERPLATE: Advance to next destination register
        // --------------------------------------------------------------------
        // Why boilerplate? This increment is required to process all vector
        // elements. The hardware expects this exact pattern.
        // ====================================================================
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

---

## Step 2: Create the Compute Kernel API Wrapper

Add a mid-level C++ wrapper that provides the `<op_name>_tile_init()` and `<op_name>_tile()` functions.

### File Location:
`tt_metal/include/compute_kernel_api/eltwise_unary/<op_name>.h`

### Template:
```cpp
// <op_name>.h
#pragma once

// ============================================================================
// BOILERPLATE: Entire file structure
// ----------------------------------------------------------------------------
// Why boilerplate? This wrapper follows a rigid pattern for ALL unary ops:
// 1. Include common_globals.h for ALWI and other macros
// 2. Conditional include of the LLK header (only for TRISC_MATH processor)
// 3. MATH() macro wrapping to ensure code only compiles for math processor
// 4. Two functions: _tile_init() for setup, _tile() for execution
//
// The ONLY changes needed:
// - Replace <op_name> with your operation name (6 occurrences)
// - If your operation has parameters, modify the _tile() function signature
//   to accept them and pass them through to the SFPU kernel
// ============================================================================
#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_<op_name>.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// ============================================================================
// BOILERPLATE: Initialization function
// ----------------------------------------------------------------------------
// Why boilerplate? SFPU_UNARY_KERNEL_INIT is a standard macro that sets up
// the SFPU for your operation. APPROX is a compile-time constant.
// CHANGE REQUIRED: Only replace <op_name>
// ============================================================================
ALWI void <op_name>_tile_init() {
    MATH(SFPU_UNARY_KERNEL_INIT(<op_name>, APPROX));
}

// ============================================================================
// BOILERPLATE: Execution function
// ----------------------------------------------------------------------------
// Why boilerplate? SFPU_TWO_PARAM_KERNEL macro handles:
// - Calling your calculate_<op_name> function
// - Passing approximation mode (APPROX)
// - Processing 8 iterations
// - Specifying destination index (idst)
// - Vector mode (RC = row-column, meaning full tile)
//
// CHANGE REQUIRED: Replace <op_name>
//
// NOT BOILERPLATE IF: Your operation has parameters!
// For parameterized ops, you need to:
// 1. Add parameters to function signature: <op_name>_tile(uint32_t idst, float param)
// 2. Use different macro or manually call the compute function
// ============================================================================
ALWI void <op_name>_tile(uint32_t idst) {
    MATH(SFPU_TWO_PARAM_KERNEL(_calculate_<op_name>_, APPROX, 8, idst, (int)VectorMode::RC));
}

}  // namespace ckernel
```

---

## Step 3: Add to Conditional Compilation System

Register the new operation in the SFPU split includes system for conditional compilation.

### File: `tt_metal/include/compute_kernel_api/eltwise_unary/sfpu_split_includes.h`

```cpp
// ============================================================================
// BOILERPLATE: Conditional include block
// ----------------------------------------------------------------------------
// Why boilerplate? Every SFPU operation needs exactly this pattern:
// - An #if checking SFPU_OP_<OP_NAME>_INCLUDE
// - An #include of the corresponding header
// - An #endif
//
// The conditional compilation system ensures that only the required SFPU
// operations are compiled into each kernel, reducing code size and compile time.
//
// CHANGE REQUIRED: Replace <OP_NAME> and <op_name> with your operation
// (uppercase for macro, lowercase for filename)
// ============================================================================
#if SFPU_OP_<OP_NAME>_INCLUDE
#include "compute_kernel_api/eltwise_unary/<op_name>.h"
#endif
```

---

## Step 4: Add the Enum Value

Add the new operation type to the `UnaryOpType` enum.

### File: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`

```cpp
// ============================================================================
// BOILERPLATE: Enum entry
// ----------------------------------------------------------------------------
// Why boilerplate? Adding an enum value is purely mechanical:
// - Add the name in SCREAMING_CASE to match existing conventions
// - No logic, no computation, just a unique identifier
//
// The enum serves as the single source of truth for operation identification
// throughout the C++ codebase.
//
// CHANGE REQUIRED: Add <OP_NAME> (follow alphabetical ordering if enforced)
// ============================================================================
enum class UnaryOpType {
    // ... existing operations ...
    <OP_NAME>,  // Add your new operation
};
```

---

## Step 5: Update Unary Op Utils

Map the new operation to its macro definition and kernel functions.

### File: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

#### 5a. Add macro definition mapping in `get_macro_definition()`:
```cpp
// ============================================================================
// BOILERPLATE: Macro definition case
// ----------------------------------------------------------------------------
// Why boilerplate? This is a direct 1:1 mapping:
// - UnaryOpType::<OP_NAME> → "SFPU_OP_<OP_NAME>_INCLUDE"
// - The string must exactly match the macro used in sfpu_split_includes.h
// - No logic involved, just string generation
//
// This enables conditional compilation - only the needed SFPU code gets built.
//
// CHANGE REQUIRED: Replace <OP_NAME> in both places
// ============================================================================
case UnaryOpType::<OP_NAME>:
    return "SFPU_OP_<OP_NAME>_INCLUDE";
```

#### 5b. Add init/function strings in `get_op_init_and_func()`:
```cpp
// ============================================================================
// BOILERPLATE: Init and function string generation
// ----------------------------------------------------------------------------
// Why boilerplate? These strings directly map to the functions created in
// Step 2. The pattern is always:
// - Init: "<op_name>_tile_init();"
// - Func: "<op_name>_tile({dst_index});"
//
// The program factory will embed these strings into the compute kernel,
// which is then compiled for the device.
//
// CHANGE REQUIRED: Replace <op_name> with your operation name
//
// NOT BOILERPLATE IF: Your operation has parameters!
// Parameterized operations need different function signatures:
//   fmt::format("<op_name>_tile({}, {});", idst, param_value)
// ============================================================================
case UnaryOpType::<OP_NAME>:
    return {"<op_name>_tile_init();", fmt::format("<op_name>_tile({});", idst)};
```

#### 5c. If your operation has parameters, add parameter packing logic in `get_op_init_and_func_parameterized()`:

```cpp
// ============================================================================
// NOT BOILERPLATE: Parameter packing logic
// ============================================================================
//
// This section is NOT boilerplate because it requires deep understanding of:
//
// 1. YOUR OPERATION'S MATHEMATICAL DEFINITION
//    - What parameters does your operation need?
//    - What are their types (float, int, enum)?
//    - What are valid ranges?
//
// 2. HOST-TO-DEVICE COMMUNICATION CONSTRAINTS
//    - Parameters are passed as uint32_t values embedded in kernel code strings
//    - Floats must be bit-cast (not converted!) to preserve exact bit patterns
//    - The kernel will bit-cast back to interpret the value
//
// 3. DATA TYPE VARIANTS
//    - Many operations have different implementations for INT32, UINT32, FLOAT
//    - You may need separate code paths based on input_dtype
//
// 4. PERFORMANCE OPTIMIZATIONS
//    - Pre-compute derived values on host to avoid device computation
//    - Example: Pass both `param` and `1.0f/param` to avoid division on device
//
// ============================================================================

// ----------------------------------------------------------------------------
// PATTERN 1: Single float parameter (simplest case)
// ----------------------------------------------------------------------------
// Used by: LEAKY_RELU, ELU, PRELU, HEAVISIDE, RPOW
//
// The float is bit-cast to uint32_t using std::bit_cast<uint32_t>(param).
// The kernel receives the raw bits and bit-casts back to float.
// This preserves exact floating-point representation including special values
// (NaN, Inf, denormals).

case UnaryOpType::LEAKY_RELU:
    return {
        "leaky_relu_tile_init();",
        fmt::format("leaky_relu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))
    };
    // {:#x}u formats as hex (0x3f800000u) for readability in generated code

case UnaryOpType::ELU:
    return {
        "elu_tile_init();",
        fmt::format("elu_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))
    };

// ----------------------------------------------------------------------------
// PATTERN 2: Integer parameter (cast directly)
// ----------------------------------------------------------------------------
// Used by: POWER, BITWISE_XOR, BITWISE_AND, LEFT_SHIFT, RIGHT_SHIFT, ROUND
//
// Integers don't need bit-casting - just cast to uint32_t.

case UnaryOpType::POWER:
    return {
        "power_tile_init();",
        fmt::format("power_tile({}, {}u);", idst, (uint32_t)param0)
    };

case UnaryOpType::BITWISE_XOR:
    return {
        "bitwise_xor_tile_init();",
        fmt::format("bitwise_xor_tile({}, {}u);", idst, (uint)params[0])
    };

// ----------------------------------------------------------------------------
// PATTERN 3: Multiple parameters
// ----------------------------------------------------------------------------
// Used by: SOFTPLUS (beta, threshold), HARDTANH (min, max), SELU (alpha, scale)
//          CLAMP (min, max), THRESHOLD (threshold, value)
//
// Access params[0], params[1], etc. and format each into the function call.

case UnaryOpType::SOFTPLUS: {
    TT_ASSERT(params.size() == 2, "Expected softplus to take 2 parameters");
    float param1 = params[1];  // threshold
    return {
        "softplus_tile_init();",
        fmt::format(
            "softplus_tile({}, {:#x}u, {:#x}u, {:#x}u);",
            idst,
            std::bit_cast<uint32_t>(param0),           // beta
            std::bit_cast<uint32_t>(1.0f / param0),    // 1/beta (pre-computed!)
            std::bit_cast<uint32_t>(param1))           // threshold
    };
}

case UnaryOpType::HARDTANH: {
    float param1 = params[1];  // max value
    return {
        "hardtanh_tile_init();",
        fmt::format(
            "hardtanh_tile({}, {:#x}u, {:#x}u);",
            idst,
            std::bit_cast<uint32_t>(param0),   // min
            std::bit_cast<uint32_t>(param1))   // max
    };
}

// ----------------------------------------------------------------------------
// PATTERN 4: Pre-computed derived values (performance optimization)
// ----------------------------------------------------------------------------
// Used by: REMAINDER, FMOD, DIV_UNARY_SFPU, CELU
//
// Compute values on host that would otherwise require expensive device ops.
// Division is expensive on SFPU, so pass reciprocal instead.

case UnaryOpType::REMAINDER:
    return {
        fmt::format(
            "remainder_tile_init({:#x}u, {:#x}u);",
            std::bit_cast<uint32_t>(param0),
            std::bit_cast<uint32_t>(1.0f / param0)),  // Pre-compute reciprocal!
        fmt::format(
            "remainder_tile({}, {:#x}u, {:#x}u);",
            idst,
            std::bit_cast<uint32_t>(param0),
            std::bit_cast<uint32_t>(1.0f / param0))   // Avoid division on device
    };

case UnaryOpType::DIV_UNARY_SFPU:
    return {
        "binop_with_scalar_tile_init();",
        fmt::format("div_unary_tile({}, {:#x}u);", idst,
            std::bit_cast<uint32_t>(1.0f / param0))  // Pass reciprocal, multiply instead of divide
    };

// ----------------------------------------------------------------------------
// PATTERN 5: Data-type dependent handling
// ----------------------------------------------------------------------------
// Used by: ADD_UNARY_SFPU, RSUB, UNARY_EQ/NE/GT/LT/GE/LE, RELU_MAX, MAXIMUM
//
// Different kernel functions for INT32 vs FLOAT. May also need different
// parameter conversions (int32_t cast vs float bit_cast).

case UnaryOpType::ADD_UNARY_SFPU:
    TT_FATAL(input_dtype.has_value(), "Missing input dtype");
    if (input_dtype == DataType::INT32) {
        return {
            "binop_with_scalar_tile_init();",
            fmt::format(
                "add_unary_tile_int32({}, {}u);",
                idst,
                std::bit_cast<uint32_t>(static_cast<int32_t>(param0_raw)))
        };
    }
    // Float path
    return {
        "binop_with_scalar_tile_init();",
        fmt::format("add_unary_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))
    };

case UnaryOpType::MAXIMUM:
    TT_FATAL(input_dtype.has_value(), "Missing input dtype");
    if (input_dtype == DataType::INT32 || input_dtype == DataType::UINT32) {
        return {
            "unary_max_tile_init();",
            fmt::format("unary_max_int32_tile({}, {}u);", idst, (uint)params[0])
        };
    }
    return {
        "unary_max_tile_init();",
        fmt::format("unary_max_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))
    };

// ----------------------------------------------------------------------------
// PATTERN 6: Template parameter in kernel (compile-time constant)
// ----------------------------------------------------------------------------
// Used by: GELU, EXP, ERF, SIGMOID, TANH (approximation mode)
//
// Some parameters affect kernel compilation (template args) rather than
// runtime behavior. These become template arguments in the generated code.

case UnaryOpType::GELU:
    return {
        fmt::format("gelu_tile_init<{}u>();", (uint32_t)param0),  // Template arg!
        fmt::format("gelu_tile<{1}u>({0});", idst, (uint32_t)param0)
    };

case UnaryOpType::SIGMOID: {
    uint32_t param1 = (uint32_t)params[1];  // approximation mode
    TT_FATAL(
        (int32_t)param0 == (int32_t)VecMode::C || (int32_t)param0 == (int32_t)VecMode::RC,
        "Invalid Vector mode");
    return {
        fmt::format("sigmoid_tile_init<{}u>();", param1),
        fmt::format("sigmoid_tile<{1}, {2}u>({0});", idst, (int32_t)param0, param1)
    };
}

// ----------------------------------------------------------------------------
// PATTERN 7: Enum/mode parameters
// ----------------------------------------------------------------------------
// Used by: RDIV (rounding mode), TYPECAST (input/output dtype)
//
// Enums are converted to their underlying integer values or string names.

case UnaryOpType::RDIV: {
    uint32_t rounding_mode_value = params[1];
    static constexpr const char* rounding_mode_strs[] = {
        "ckernel::RoundingMode::None",
        "ckernel::RoundingMode::Trunc",
        "ckernel::RoundingMode::Floor"
    };
    return {
        "rdiv_tile_init();",
        fmt::format(
            "rdiv_tile<{}>({}, {:#x}u);",
            rounding_mode_strs[rounding_mode_value],  // String enum name!
            idst,
            std::bit_cast<uint32_t>(param0))
    };
}
```

### Understanding the Parameter Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Python: ttnn.leaky_relu(tensor, slope=0.1)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ C++ UnaryWithParam: UnaryOpType::LEAKY_RELU, params=[0.1f]                  │
│                                                                             │
│ The parameter is stored as a float in a std::vector<float>                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ get_op_init_and_func_parameterized():                                       │
│                                                                             │
│ float param0 = params[0];  // 0.1f                                          │
│ uint32_t bits = std::bit_cast<uint32_t>(param0);  // 0x3dcccccd             │
│                                                                             │
│ Returns: {"leaky_relu_tile_init();",                                        │
│           "leaky_relu_tile(0, 0x3dcccccdu);"}                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Program Factory embeds these strings into the compute kernel source:        │
│                                                                             │
│ #define SFPU_OP_INIT_0 leaky_relu_tile_init();                              │
│ #define SFPU_OP_FUNC_0 leaky_relu_tile(0, 0x3dcccccdu);                     │
│                                                                             │
│ The kernel is then compiled with these defines.                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SFPU Kernel receives the parameter as a uint32_t literal:                   │
│                                                                             │
│ void leaky_relu_tile(uint32_t idst, uint32_t slope_bits) {                  │
│     vFloat slope = std::bit_cast<float>(slope_bits);  // Back to 0.1f!      │
│     // Use slope in computation...                                          │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why std::bit_cast Instead of Simple Cast?

```cpp
float value = 0.1f;

// WRONG: Numeric conversion - loses precision, doesn't work for NaN/Inf
uint32_t bad = (uint32_t)value;  // Results in 0!

// CORRECT: Bit-preserving reinterpretation
uint32_t good = std::bit_cast<uint32_t>(value);  // 0x3dcccccd

// The kernel does the reverse:
float restored = std::bit_cast<float>(good);  // Exactly 0.1f again
```

This is critical because:
1. Float-to-int conversion truncates (0.1 → 0)
2. Special values (NaN, Inf, -0.0) have specific bit patterns that must be preserved
3. Denormalized numbers must pass through unchanged

---

## Step 6: Register the C++ Operation

Add the operation registration to expose it as a C++ function.

### File: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
// ============================================================================
// BOILERPLATE: Operation registration macros
// ----------------------------------------------------------------------------
// Why boilerplate? These macros encapsulate ALL the complexity of:
// - Creating a C++ functor class
// - Setting up the invoke() method with proper signatures
// - Handling optional parameters (memory_config, output_tensor, etc.)
// - Connecting to the device operation infrastructure
//
// You just provide the operation name and the macro does everything else.
//
// CHANGE REQUIRED: Only substitute <op_name> and <OP_NAME>
// Choose the appropriate macro variant based on your operation's parameters.
// ============================================================================

// For operations without parameters:
REGISTER_UNARY_OPERATION(<op_name>, <OP_NAME>);

// For operations with a float parameter:
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(<op_name>, <OP_NAME>);

// For operations with an integer parameter:
REGISTER_UNARY_OPERATION_WITH_INTEGER_PARAMETER(<op_name>, <OP_NAME>);
```

---

## Step 7: Add Python Bindings

Expose the operation to Python via nanobind.

### File: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

Add a binding call in `bind_unary_operations()`:
```cpp
// ============================================================================
// MOSTLY BOILERPLATE: Python binding registration
// ----------------------------------------------------------------------------
// Why mostly boilerplate? The function call structure is always the same:
// - detail::bind_unary_operation(module, ttnn::<op_name>, doc, range, dtypes)
//
// BOILERPLATE ASPECTS:
// - The function being called
// - The module parameter
// - The reference to ttnn::<op_name>
//
// NOT BOILERPLATE: The documentation strings!
// ----------------------------------------------------------------------------
// Why documentation is NOT boilerplate? Each operation needs unique docs:
//
// 1. Mathematical description: What does this operation compute?
//    - Formula (e.g., "Computes f(x) = 1 / (1 + e^(-x))" for sigmoid)
//    - Mathematical properties and identities
//
// 2. Input range: What input values are valid/expected?
//    - "[-1, 1]" for acos/asin
//    - "(0, inf)" for log
//    - "any" for abs/neg
//    - Edge cases: What happens at boundaries?
//
// 3. Supported dtypes: Which data types work?
//    - Some ops only support BFLOAT16
//    - Others support BFLOAT8_B, FLOAT32, etc.
//    - INT32 support requires separate _int32 variant
//
// These strings appear in Python help() and documentation, so they must
// be accurate and operation-specific.
// ============================================================================
detail::bind_unary_operation(
    module,
    ttnn::<op_name>,
    R"doc(<mathematical description>)doc",
    R"doc(<input range description, e.g., "[-1, 1]">)doc",
    R"doc(<supported dtypes, e.g., "BFLOAT16, BFLOAT8_B">)doc");
```

---

## Step 8: Register Python Golden Function

Add the PyTorch equivalent for testing.

### File: `ttnn/ttnn/operations/unary.py`

#### 8a. Add to the function list:
```python
# ============================================================================
# BOILERPLATE: Function list registration
# ----------------------------------------------------------------------------
# Why boilerplate? Just appending your operation to an existing list.
# This enables the testing infrastructure to discover your operation.
#
# CHANGE REQUIRED: Add ttnn.<op_name>
# ============================================================================
TTNN_ELTWISE_UNARY_CPP_FUNCTIONS = [
    # ... existing functions ...
    ttnn.<op_name>,
]
```

#### 8b. Add the golden function mapping in `_golden_function()`:
```python
# ============================================================================
# SEMI-BOILERPLATE: Golden function mapping
# ----------------------------------------------------------------------------
# Why semi-boilerplate? The dictionary structure is fixed, but the VALUE
# (the actual golden function) varies based on your operation.
#
# BOILERPLATE IF: Direct PyTorch equivalent exists
#   "<op_name>": torch.abs,
#   "<op_name>": torch.sigmoid,
#   "<op_name>": torch.neg,
#
# NOT BOILERPLATE IF: No direct equivalent, requiring:
# ----------------------------------------------------------------------------
# 1. Lambda expressions for simple transformations:
#    "softplus": lambda x, beta=1.0: torch.log(1 + torch.exp(beta * x)) / beta,
#
# 2. Custom functions for complex operations:
#    def my_op_golden(x, param):
#        # Handle edge cases
#        # Implement the reference algorithm
#        return result
#    "my_op": my_op_golden,
#
# 3. Numerical considerations:
#    - Match the precision characteristics of hardware implementation
#    - Handle inf/nan cases consistently
#    - Consider approximation modes
#
# The golden function serves as ground truth for testing, so it MUST
# accurately represent the intended mathematical operation.
# ============================================================================
name_to_golden_function = {
    # ... existing mappings ...
    "<op_name>": torch.<torch_equivalent>,  # or custom lambda
}
```

---

## Step 9: Add Unit Tests

Create unit tests to verify the operation.

### File: `tests/ttnn/unit_tests/operations/eltwise/test_unary.py` (or similar)

```python
# ============================================================================
# SEMI-BOILERPLATE: Test structure
# ----------------------------------------------------------------------------
# Why semi-boilerplate? The test STRUCTURE follows a pattern:
# 1. Parametrize with shapes
# 2. Create input tensor
# 3. Run ttnn operation
# 4. Compare with expected (golden) output
#
# BOILERPLATE ASPECTS:
# - Test function signature with device and shape parameters
# - Tensor conversion to/from torch
# - Basic assertion structure
#
# NOT BOILERPLATE ASPECTS - Operation-Specific Considerations:
# ============================================================================
@pytest.mark.parametrize("input_shape", [[1, 1, 32, 32], [1, 1, 64, 64]])
def test_<op_name>(device, input_shape):

    # ========================================================================
    # NOT BOILERPLATE: Input tensor generation
    # ------------------------------------------------------------------------
    # Why NOT boilerplate? Consider:
    #
    # 1. Input range: Does your op have domain restrictions?
    #    - sigmoid/tanh: torch.rand() is fine (any real input)
    #    - acos/asin: torch.rand() * 2 - 1 (must be in [-1, 1])
    #    - log: torch.rand() + 0.1 (must be positive)
    #    - sqrt: torch.rand() (must be non-negative)
    #
    # 2. Edge cases: Should you test boundaries?
    #    - Values near zero (potential division issues)
    #    - Very large values (overflow)
    #    - Very small values (underflow)
    #
    # 3. Data type: Does your op support bfloat16? float32?
    #    Some operations lose precision with lower-precision types.
    # ========================================================================
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    output = ttnn.<op_name>(input_tensor)
    output_torch = ttnn.to_torch(output)

    expected = torch.<torch_equivalent>(torch_input)

    # ========================================================================
    # NOT BOILERPLATE: Tolerance selection
    # ------------------------------------------------------------------------
    # Why NOT boilerplate? Tolerance depends on:
    #
    # 1. Operation characteristics:
    #    - Simple ops (abs, neg): tight tolerance (rtol=1e-4)
    #    - Transcendental ops (exp, log, sin): looser (rtol=1e-2 to 1e-3)
    #    - Approximation-based ops: even looser (rtol=1e-2)
    #
    # 2. Data type:
    #    - BFLOAT16 has ~3 decimal digits precision → rtol=1e-2 typical
    #    - FLOAT32 has ~7 decimal digits precision → rtol=1e-5 possible
    #
    # 3. Input range sensitivity:
    #    - Some ops amplify errors near boundaries
    #    - May need atol (absolute tolerance) for near-zero outputs
    #
    # 4. Hardware approximation:
    #    - SFPU implementations may use polynomial approximations
    #    - Check if APPROXIMATION_MODE affects precision
    #
    # Common patterns:
    #   rtol=1e-2: Most bfloat16 transcendental operations
    #   rtol=1e-3: Higher precision ops or float32
    #   atol=1e-5: When outputs can be near zero
    # ========================================================================
    assert torch.allclose(output_torch, expected, rtol=1e-2)
```

---

## Summary: Files to Modify/Create

| Step | File Path | Action | Boilerplate Level |
|------|-----------|--------|-------------------|
| 1 | `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` | **Create** | Structure: Boilerplate, Math: **NOT** |
| 2 | `tt_metal/include/compute_kernel_api/eltwise_unary/<op>.h` | **Create** | Fully Boilerplate |
| 3 | `tt_metal/include/compute_kernel_api/eltwise_unary/sfpu_split_includes.h` | Modify | Fully Boilerplate |
| 4 | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Modify | Fully Boilerplate |
| 5 | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Modify | Mostly Boilerplate* |
| 6 | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | Modify | Fully Boilerplate |
| 7 | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | Modify | Structure: Boilerplate, Docs: **NOT** |
| 8 | `ttnn/ttnn/operations/unary.py` | Modify | Depends on torch equivalent |
| 9 | `tests/ttnn/unit_tests/operations/eltwise/test_unary.py` | Modify | Structure: Boilerplate, Details: **NOT** |

*Step 5 becomes non-boilerplate for parameterized operations

---

## Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python API                               │
│                    ttnn.<op_name>(tensor)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Nanobind Bindings                           │
│                   unary_nanobind.cpp                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       C++ Executor                               │
│          ExecuteUnary<UnaryOpType::<OP_NAME>>::invoke()         │
│                       unary.hpp                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Device Operation                             │
│                  UnaryDeviceOperation                            │
│               unary_device_operation.hpp                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Program Factory                              │
│     Creates GPU program with reader/compute/writer kernels       │
│                  unary_program_factory.cpp                       │
│         Maps UnaryOpType to SFPU_OP_* defines                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Compute Kernel                              │
│                     eltwise_sfpu.cpp                             │
│    Expands SFPU_OP_CHAIN_0 → <op_name>_tile_init/tile()         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Compute Kernel API                            │
│              <op_name>.h (eltwise_unary/)                        │
│         Wraps SFPU calls with MATH() macros                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SFPU Kernel                                │
│              ckernel_sfpu_<op_name>.h                            │
│     Architecture-specific (Wormhole B0 / Blackhole)              │
│           Uses SFPI intrinsics (dst_reg, vFloat)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insight: Boilerplate vs. Custom Code

### Why So Much Boilerplate?

The extensive boilerplate exists because:

1. **Abstraction Layers**: Each layer (Python → C++ → Kernel → Hardware) has its own registration system. These must stay in sync.

2. **Conditional Compilation**: The `SFPU_OP_*_INCLUDE` system avoids compiling unused operations into kernels, saving code size and compile time.

3. **Type Safety**: Enums and registration macros ensure compile-time verification of operation names and parameters.

4. **Consistency**: Uniform patterns make the codebase maintainable and make it easy to add new operations.

### The Non-Boilerplate Parts Are Where The Real Work Is

| Component | What Makes It Non-Boilerplate |
|-----------|------------------------------|
| **SFPU Kernel (Step 1)** | The actual mathematical algorithm using SFPI intrinsics. Must understand hardware capabilities, numerical precision, and edge cases. |
| **Parameter Handling (Step 5c)** | How to pack/unpack operation-specific parameters. Each operation has unique requirements. |
| **Documentation (Step 7)** | Accurate mathematical descriptions, input ranges, and supported types specific to your operation. |
| **Golden Function (Step 8b)** | Reference implementation for testing. May require custom logic if no PyTorch equivalent exists. |
| **Test Details (Step 9)** | Input generation, edge cases, and tolerance values depend on operation characteristics. |

---

## Notes

- **Approximation modes**: Consider adding both `APPROXIMATION_MODE=true` and `APPROXIMATION_MODE=false` implementations for operations that benefit from fast approximations.

- **INT32 variants**: For operations that support integer types, add `<op_name>_tile_int32()` variants.

- **Parameter handling**: Operations with scalar parameters use bit-packing utilities like `param_to_uint32()` and `uint32_to_param()`.

- **Operation chaining**: The program factory supports chaining multiple unary operations together via the `op_chain` vector for fusion optimization.

- **Testing multiple dtypes**: Test with BFLOAT16, BFLOAT8_B, FLOAT32, and other supported data types.
