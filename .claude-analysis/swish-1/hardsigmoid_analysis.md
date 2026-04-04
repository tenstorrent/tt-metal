# SFPU Kernel Analysis: hardsigmoid

**Operation Name:** hardsigmoid
**Math Definition:** `hardsigmoid(x) = max(0, min(1, x/6 + 0.5))`
**Architecture:** Wormhole B0 (SFPU)
**Date:** 2026-04-04

## 1. SFPU Kernel Function Overview

The hardsigmoid operation is implemented as a piecewise-linear activation function using SFPU instructions. It maps all input values to the range [0, 1] using a piece-wise definition:
- For x ≤ -3: output = 0
- For x ≥ 3: output = 1
- For -3 < x < 3: output = x/6 + 0.5

### File Locations
- **SFPU Kernel Implementation:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
- **LLK Math API Wrapper:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
- **Tile API Header:** `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
- **Unary Operation Utils:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (line 66)
- **SFPU Includes:** `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (lines 15-16)

## 2. SFPU Kernel Computation Details

### 2.1 Kernel Function Signature
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid()
```

**Template Parameters:**
- `APPROXIMATION_MODE` (bool): Controls numerical precision mode (currently unused in this kernel, always precise)
- `ITERATIONS` (int, default=8): Number of tiles to process in the loop; determines unroll factor via `#pragma GCC unroll 8`

### 2.2 Core Computation Steps

The kernel performs the following per tile:

1. **Load Input:** `sfpi::vFloat x = sfpi::dst_reg[0];`
   - Reads the current tile from DST register bank (DST-reg [0])
   - DST register is pre-populated by UNPACKER or previous computation

2. **Linear Transformation:** `sfpi::vFloat result = x * one_sixth + 0.5f;`
   - Multiplies input by (1/6) = 0.16666...
   - Adds 0.5 offset
   - This implements the linear segment: f(x) = x/6 + 0.5

3. **Lower Clamp (0):**
   ```cpp
   v_if(result < 0.0f) { result = 0.0f; }
   v_endif;
   ```
   - Uses SFPU conditional to clamp negative values to 0
   - Handles x < -3 case (where x/6 + 0.5 < 0)

4. **Upper Clamp (1):**
   ```cpp
   v_if(result > sfpi::vConst1) { result = sfpi::vConst1; }
   v_endif;
   ```
   - Uses SFPU conditional to clamp values > 1 to 1
   - Handles x > 3 case (where x/6 + 0.5 > 1)
   - Uses predefined constant `sfpi::vConst1` = 1.0f in SFPU format

5. **Write Output:** `sfpi::dst_reg[0] = result;`
   - Stores result back to DST register

6. **Iterate:** `sfpi::dst_reg++;`
   - Advances DST register pointer to next tile
   - Loop repeats ITERATIONS times (default 8)

### 2.3 SFPU Instruction Patterns

**Instruction Classes Used:**
- **Floating-Point Arithmetic:** Multiplication (`x * one_sixth`), Addition (`... + 0.5f`)
- **Comparison:** Less-than (`< 0.0f`), Greater-than (`> 1.0`)
- **Conditional Execution:** `v_if`/`v_endif` blocks for clamping logic
- **Register Movement:** DST register reads/writes and pointer increment

**Data Flow:**
```
DST[0] -> Load x -> * (1/6) -> + 0.5 -> MIN(result, 1) -> MAX(result, 0) -> Store DST[0]
                                                                            -> DST++ -> repeat
```

## 3. LLK Math API Wrapper Layer

### 3.1 Init Function
```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init()
{
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>();
}
```

**Purpose:** Initializes SFPU state for hardsigmoid operations
- Sets SFPU mode to `SfpuType::hardsigmoid`
- Configures approximation mode (currently not used in precise calculation)
- Prepares SFPU ALU for tile processing

### 3.2 Compute Function
```cpp
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardsigmoid(
    uint dst_index, int vector_mode = (int)VectorMode::RC)
{
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardsigmoid<APPROXIMATE, ITERATIONS>,
        dst_index, vector_mode);
}
```

**Parameters:**
- `dst_index` (uint): Target DST register tile index (0-15 typical range)
- `vector_mode` (int, default RC): Vector processing mode
  - RC = Row-Column (tile-wise processing)
  - Controls how tiles are distributed across SFPU lanes

**Behavior:**
- Calls generic `_llk_math_eltwise_unary_sfpu_params_` with:
  - Kernel function: `calculate_hardsigmoid<APPROXIMATE, ITERATIONS>`
  - Destination tile index: dst_index
  - Vector mode: controls lane-wise vs. tile-wise execution

## 4. Tile API Level

### 4.1 Tile Init Macro
```cpp
ALWI void hardsigmoid_tile_init()
{
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()));
}
```

**Purpose:** Public API for initializing hardsigmoid on the tile level
- Invoked once per block before processing tiles
- Expands to LLK init function wrapped in MATH macro
- Uses compile-time constant `APPROX` for approximation mode

### 4.2 Tile Compute Macro
```cpp
ALWI void hardsigmoid_tile(uint32_t idst)
{
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX>(idst)));
}
```

**Parameters:**
- `idst` (uint32_t): DST tile index to process

**Purpose:** Compute hardsigmoid for a single tile
- Called once per tile in the batch
- Invoked after data is loaded to DST register via UNPACKER
- Must be called after `hardsigmoid_tile_init()`

## 5. Integration with Unary Operation Framework

### 5.1 Unary Op Types Registration
From `common/unary_op_utils.cpp:66`:
```cpp
case UnaryOpType::HARDSIGMOID:
    return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
```

**Role:** Maps `UnaryOpType::HARDSIGMOID` to SFPU kernel names for compute kernel code generation

### 5.2 SFPU Include Dispatch
From `sfpu_split_includes.h:15-16`:
```cpp
#if SFPU_OP_HARDSIGMOID_INCLUDE
#include "api/compute/eltwise_unary/hardsigmoid.h"
#endif
```

**Purpose:** Conditional inclusion of hardsigmoid headers based on compile-time flags
- Flag: `SFPU_OP_HARDSIGMOID_INCLUDE`
- Avoids including unused kernels in binary
- Enables cross-file SFPU kernel dependency management

### 5.3 Unary Program Factory
The `UnaryProgramFactory` in `unary_program_factory.cpp`:
1. Reads operation chain from `UnaryParams`
2. Gets kernel path from `utils::get_compute_kernel_path(ops_chain[0].type())`
3. Generates compute kernel with defines from `utils::get_block_defines()`
4. Creates circular buffers: c_0 (input), c_2 (output), c_1 (optional temp)
5. Constructs reader, writer, and compute kernels with appropriate args

## 6. Compute Kernel Usage Pattern

Typical compute kernel usage (from `hardswish_kernel.cpp` which uses hardsigmoid internally):

```cpp
// Step 1: Initialize SFPU for hardsigmoid
hardsigmoid_tile_init();

// Step 2: Compute hardsigmoid on DST[0]
hardsigmoid_tile(0);

// Step 3: Use result in further computation (e.g., multiplication with input)
binary_dest_reuse_tiles<ELWMUL, ...>(cb_input, 0, 0);

// Step 4: Pack result to output CB
pack_tile(0, cb_output);
```

## 7. Register and Memory Layout

### 7.1 Register Usage
- **DST Register:** Input storage and output location
  - Pre-loaded by UNPACKER with tile data
  - Stores intermediate `x * (1/6) + 0.5`
  - Stores final clamped result
  - Advanced via `sfpi::dst_reg++` per iteration

### 7.2 Circular Buffer Configuration (from UnaryProgramFactory)
- **c_0 (Input):**
  - Data format: Input tensor format (BFLOAT16, FLOAT32, etc.)
  - Size: 2 tiles (double-buffered for pipeline overlap)
  - Page size: Adjusted for TILE or ROW_MAJOR layout

- **c_2 (Output):**
  - Data format: Output tensor format
  - Size: 2 tiles (double-buffered)
  - Page size: Matches output layout

- **c_1 (Optional, for HARDSHRINK only):**
  - Not used for hardsigmoid
  - Reserved for ops requiring intermediate storage

### 7.3 DST Register State Machine
```
[UNPACKER reads c_0] -> [DST reg has tile data]
  -> [hardsigmoid_tile() computes]
  -> [DST reg has result]
  -> [PACKER writes to c_2]
```

## 8. Numerics and Precision

### 8.1 Floating-Point Constants
- **`one_sixth`:** `1.0f / 6.0f ≈ 0.16666667`
- **`0.5f`:** Offset constant
- **`sfpi::vConst1`:** SFPU hardware constant for 1.0

### 8.2 Numerical Behavior
- **Input Range:** Unrestricted (uses float32)
- **Output Range:** [0.0, 1.0] (due to clamping)
- **Precision Loss Points:**
  - Division by 6 (limited float precision)
  - Comparison thresholds use exact hardware values
  - Clamp comparisons are exact in floating-point

### 8.3 Special Values Handling
- **NaN Input:** Preserves NaN (comparisons fail, result = NaN)
- **Inf Input:**
  - `+Inf * (1/6) = +Inf` → clamped to 1.0
  - `-Inf * (1/6) = -Inf` → clamped to 0.0
- **Zero:** 0 * (1/6) + 0.5 = 0.5 ✓

## 9. SFPU Instruction Utilization

### 9.1 ALU Operations
- **FP Multiply:** 1 per iteration (x * one_sixth)
- **FP Add:** 1 per iteration (... + 0.5)
- **FP Compare:** 2 per iteration (< 0, > 1)
- **FP Select/Mux:** 2 per iteration (via v_if/v_endif)

### 9.2 Register Pressure
- **Live Values:** 1 (x) + 1 (result) = 2 vFloat registers
- **Register Allocation:** Minimal; uses only working variables
- **Memory Accesses:** DST[0] read/write per tile

### 9.3 Throughput
- **Loop Unroll:** 8 iterations (GCC unroll pragma)
- **Latency per Tile:** ~4-6 cycles (FP ops + conditional branches)
- **Throughput:** ~1 tile per cycle (with pipelining)
- **Total Batch Processing:** 8 tiles in 8-12 cycles (pipeline dependent)

## 10. Known Constraints and Limitations

1. **Approximation Mode Not Used:**
   - Template parameter `APPROXIMATION_MODE` exists but is unused
   - Kernel always computes exact (non-approximate) hardsigmoid
   - No fast approximation variant implemented

2. **Single-Precision Only:**
   - Uses float32 internally
   - Clamp values are exact but division by 6 has float precision limits

3. **No Dynamic Parameters:**
   - Hardcoded (1/6) and 0.5 scaling
   - Cannot adjust clamping range at runtime
   - Use hardtanh if configurable clamp bounds needed

4. **DST Register-Only I/O:**
   - Must be called in sequence after DST has valid data
   - Cannot operate on general-purpose registers

## 11. Integration Points

### 11.1 Dispatch Chain
1. **Python API:** `ttnn.hardsigmoid(tensor)`
2. **C++ Binding:** `bind_unary_operation<"hardsigmoid", &ttnn::hardsigmoid>`
3. **Unary Implementation:** `ttnn::detail::unary_impl()` with `UnaryOpType::HARDSIGMOID`
4. **Device Operation:** `UnaryDeviceOperation` selects `UnaryProgramFactory`
5. **Program Factory:** Creates compute kernel with hardsigmoid defines
6. **Compute Kernel:** Calls `hardsigmoid_tile_init()` and `hardsigmoid_tile(idst)`
7. **SFPU API:** LLK wrappers `llk_math_eltwise_unary_sfpu_hardsigmoid_init` and `llk_math_eltwise_unary_sfpu_hardsigmoid`
8. **SFPU Kernel:** `calculate_hardsigmoid<APPROXIMATE, ITERATIONS>()` executes raw SFPU instructions

### 11.2 Data Flow
```
Input Tensor
  -> Reader Kernel (reads c_0)
  -> UNPACKER (dst_reg[0] = tile)
  -> hardsigmoid_tile_init() [SFPU state]
  -> Loop: hardsigmoid_tile(i) [SFPU compute per tile]
  -> PACKER (tile from dst_reg[0])
  -> Writer Kernel (writes c_2)
  -> Output Tensor
```

## 12. Summary

**hardsigmoid** is a straightforward SFPU unary operation implementing piecewise-linear clamping. It:
- Uses basic FP arithmetic (multiply, add) and comparisons for efficient hardware execution
- Integrates cleanly into the UnaryProgramFactory framework via tile-level APIs
- Maintains register efficiency with minimal working state
- Supports 8-way unrolled iteration for throughput
- Handles edge cases (NaN, Inf) correctly via IEEE float semantics

The operation is well-suited for hardware acceleration and introduces minimal computational overhead while providing the controlled saturation behavior of a hard sigmoid activation.
