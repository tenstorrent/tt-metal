# Testing Plan: Block Variants for Compute API

## 📋 Overview

This document provides a detailed plan for AI agents to implement tests for the newly added block variants (`*_block` functions) in the Compute API. Each agent can independently tackle a different test suite.

**Goal**: Create comprehensive tests that validate block variants produce identical results to tile-by-tile processing.

**Strategy**: Duplicate existing tile-based tests and modify them to use block operations instead.

**Reference Issue**: [#35739](https://github.com/tenstorrent/tt-metal/issues/35739)

---

## 🎯 Test Requirements

### Core Principle
**Block operations must produce IDENTICAL results to tile-by-tile operations.**

The tests will:
1. Create two compute kernels: one using tile-by-tile operations, one using block operations
2. Run both with identical input data
3. Compare outputs using PCC (Pearson Correlation Coefficient) ≥ 0.9999
4. Test various block sizes: 1x1, 2x2, 2x4, 4x4, etc. (always ≤ 16 tiles total)

### Test Infrastructure

**Existing Test Locations**:
- **C++ Gtests**: `tests/ttnn/unit_tests/gtests/udm/eltwise/`
- **Compute Kernel Tests**: `tests/tt_metal/tt_metal/test_eltwise_binary.cpp`
- **TTNN Python Tests**: `tests/ttnn/unit_tests/operations/eltwise/`

**Preferred Approach**: Create compute kernel tests (C++ Gtest style) for direct Compute API validation.

---

## 📦 Test Suites to Implement

### Test Suite 1: Element-wise Binary Block Variants
**Operations**: `add_block`, `sub_block`, `mul_block`
**File**: `tests/tt_metal/tt_metal/test_eltwise_binary_block.cpp`

### Test Suite 2: Broadcast Block Variants
**Operations**: `add_tiles_bcast_block`, `sub_tiles_bcast_block`, `mul_tiles_bcast_block`
**File**: `tests/tt_metal/tt_metal/test_bcast_block.cpp`

### Test Suite 3: Transpose Block Variant
**Operations**: `transpose_wh_block`
**File**: `tests/tt_metal/tt_metal/test_transpose_wh_block.cpp`

### Test Suite 4: Reduce Block Variant
**Operations**: `reduce_block`
**File**: `tests/tt_metal/tt_metal/test_reduce_block.cpp`

### Test Suite 5: Pack Block Variant
**Operations**: `pack_block`
**File**: `tests/tt_metal/tt_metal/test_pack_block.cpp`

---

## 🔧 Implementation Guide

### Step 1: Identify Base Test

Find an existing test for the tile-based version of your operation:

```bash
# Example: Finding add_tiles test
cd /localdev/ncvetkovic/reconfig/tt-metal
find tests -name "*eltwise*" -o -name "*add*" | grep -E '\.(cpp|py)$'
```

**Key Examples**:
- Element-wise: `tests/ttnn/unit_tests/gtests/udm/eltwise/test_udm_add.cpp`
- Binary ops: `tests/tt_metal/tt_metal/test_eltwise_binary.cpp`

### Step 2: Create Duplicate Kernels

For each test, you'll need TWO compute kernels:

#### Kernel A: Tile-by-Tile (Reference)
```cpp
// kernels/compute_add_tiles.cpp
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);  // Block height
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // Block width
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Initialize
    binary_op_init_common(cb_in0, cb_in1, cb_out);
    add_tiles_init(cb_in0, cb_in1);

    // Process each block tile-by-tile
    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for block worth of tiles
        cb_wait_front(cb_in0, Ht * Wt);
        cb_wait_front(cb_in1, Ht * Wt);

        // Acquire DEST for the block
        tile_regs_acquire();

        // Process tile-by-tile
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
                uint32_t tile_idx = h * Wt + w;
                add_tiles(cb_in0, cb_in1, tile_idx, tile_idx, tile_idx);
            }
        }

        tile_regs_commit();

        // Pack results
        cb_reserve_back(cb_out, Ht * Wt);
        tile_regs_wait();

        for (uint32_t i = 0; i < Ht * Wt; i++) {
            pack_tile(i, cb_out);
        }

        tile_regs_release();

        cb_push_back(cb_out, Ht * Wt);
        cb_pop_front(cb_in0, Ht * Wt);
        cb_pop_front(cb_in1, Ht * Wt);
    }
}
}
```

#### Kernel B: Block Operation (Test Subject)
```cpp
// kernels/compute_add_block.cpp
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);  // Block height
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // Block width
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Initialize (same as tile version)
    binary_op_init_common(cb_in0, cb_in1, cb_out);
    add_tiles_init(cb_in0, cb_in1);

    // Process each block using block operation
    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for block worth of tiles
        cb_wait_front(cb_in0, Ht * Wt);
        cb_wait_front(cb_in1, Ht * Wt);

        // Acquire DEST for the block
        tile_regs_acquire();

        // USE BLOCK OPERATION - This is what we're testing!
        add_block<Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);

        tile_regs_commit();

        // Pack results using pack_block
        cb_reserve_back(cb_out, Ht * Wt);
        tile_regs_wait();

        pack_block<Ht, Wt>(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, Ht * Wt);
        cb_pop_front(cb_in0, Ht * Wt);
        cb_pop_front(cb_in1, Ht * Wt);
    }
}
}
```

### Step 3: Create Test Harness

```cpp
// test_eltwise_binary_block.cpp
#include <gtest/gtest.h>
#include "common/command_queue_fixture.hpp"
#include "test_gold_impls.hpp"

namespace {

void run_add_block_test(
    Device* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {

    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block exceeds DEST capacity";

    Program program_tiles = CreateProgram();  // Reference (tile-by-tile)
    Program program_block = CreateProgram();  // Test subject (block operation)

    // Create buffers for both programs
    uint32_t single_tile_size = tile_size(data_format);
    uint32_t buffer_size = single_tile_size * Ht * Wt * num_blocks;

    auto src0_buffer = CreateBuffer(InterleavedBufferConfig{
        device, buffer_size, page_size=single_tile_size, buffer_type=BufferType::DRAM});
    auto src1_buffer = CreateBuffer(InterleavedBufferConfig{
        device, buffer_size, page_size=single_tile_size, buffer_type=BufferType::DRAM});
    auto dst_tiles_buffer = CreateBuffer(...);  // For tile-by-tile result
    auto dst_block_buffer = CreateBuffer(...);  // For block operation result

    // Generate random input data
    std::vector<bfloat16> src0_data = generate_random_bfloat16(buffer_size / sizeof(bfloat16));
    std::vector<bfloat16> src1_data = generate_random_bfloat16(buffer_size / sizeof(bfloat16));

    // Write to device
    EnqueueWriteBuffer(device->command_queue(), src0_buffer, src0_data, false);
    EnqueueWriteBuffer(device->command_queue(), src1_buffer, src1_data, false);

    // Setup programs (CBs, kernels, etc.) for both
    // ... (setup code for program_tiles)
    // ... (setup code for program_block)

    // Run both programs
    EnqueueProgram(device->command_queue(), program_tiles, false);
    EnqueueProgram(device->command_queue(), program_block, false);

    // Read results
    std::vector<bfloat16> result_tiles;
    std::vector<bfloat16> result_block;
    EnqueueReadBuffer(device->command_queue(), dst_tiles_buffer, result_tiles, true);
    EnqueueReadBuffer(device->command_queue(), dst_block_buffer, result_block, true);

    // Compare results using PCC
    float pcc = check_bfloat16_vector_pcc(result_tiles, result_block);
    EXPECT_GE(pcc, 0.9999f) << "Block operation diverged from tile-by-tile reference";

    // Also validate against golden reference
    std::vector<bfloat16> golden = compute_golden_add(src0_data, src1_data);
    float pcc_golden = check_bfloat16_vector_pcc(golden, result_block);
    EXPECT_GE(pcc_golden, 0.9999f) << "Block operation incorrect vs golden";
}

TEST_F(DeviceFixture, AddBlock_1x1) {
    run_add_block_test(device_.get(), 1, 1, 10);
}

TEST_F(DeviceFixture, AddBlock_2x2) {
    run_add_block_test(device_.get(), 2, 2, 10);
}

TEST_F(DeviceFixture, AddBlock_2x4) {
    run_add_block_test(device_.get(), 2, 4, 10);
}

TEST_F(DeviceFixture, AddBlock_4x4) {
    run_add_block_test(device_.get(), 4, 4, 10);
}

TEST_F(DeviceFixture, AddBlock_1x16_MaxWidth) {
    run_add_block_test(device_.get(), 1, 16, 10);
}

TEST_F(DeviceFixture, AddBlock_16x1_MaxHeight) {
    run_add_block_test(device_.get(), 16, 1, 10);
}

}  // namespace
```

### Step 4: Test Matrix

Each test suite should cover:

| Dimension | Values to Test |
|-----------|----------------|
| **Block Height (Ht)** | 1, 2, 4, 8, 16 |
| **Block Width (Wt)** | 1, 2, 4, 8, 16 |
| **Total Tiles (Ht×Wt)** | Must be ≤ 16 |
| **Data Format** | Float16_b, BFP8_b, Float32 |
| **Number of Blocks** | 1, 10, 100 (stress test) |

**Valid Block Sizes** (Ht × Wt ≤ 16):
- 1×1, 1×2, 1×4, 1×8, 1×16
- 2×1, 2×2, 2×4, 2×8
- 4×1, 4×2, 4×4
- 8×1, 8×2
- 16×1

### Step 5: Build and Run

```bash
cd /localdev/ncvetkovic/reconfig/tt-metal

# Build tests
./build_metal.sh --build-tests

# Run specific test suite
./build/test/tt_metal/test_eltwise_binary_block

# Or use pytest for Python tests
pytest tests/ttnn/unit_tests/operations/test_block_variants.py -v
```

---

## 🎨 Operation-Specific Guidance

### 1. Element-wise Binary (`add_block`, `sub_block`, `mul_block`)

**Base Tests**: `tests/tt_metal/tt_metal/test_eltwise_binary.cpp`

**Kernel Pattern**:
```cpp
// Initialize
binary_op_init_common(cb_in0, cb_in1, cb_out);
add_tiles_init(cb_in0, cb_in1);  // or sub/mul

// Block operation
add_block<Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);

// Pack
pack_block<Ht, Wt>(0, cb_out);
```

**Test Cases**:
- ✅ 1×1 (equivalent to single tile)
- ✅ 2×2 (small block)
- ✅ 4×4 (medium block)
- ✅ 1×16 (max width)
- ✅ 16×1 (max height)
- ✅ Different data formats (FP16, BFP8, FP32)

### 2. Broadcast (`add_tiles_bcast_block`, etc.)

**Base Tests**: `tests/tt_metal/tt_metal/test_bcast.cpp`

**Kernel Pattern**:
```cpp
// Initialize with broadcast type
init_bcast<ELWADD, BroadcastType::COL>(cb_in0, cb_in1, cb_out);

// Block operation
add_tiles_bcast_block<BroadcastType::COL, Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);

// Pack
pack_block<Ht, Wt>(0, cb_out);
```

**Test Cases**:
- ✅ BroadcastType::ROW (broadcast along rows)
- ✅ BroadcastType::COL (broadcast along columns)
- ✅ BroadcastType::SCALAR (broadcast single value)
- ✅ Various block sizes

**Special Note**: Broadcast tests need careful input tensor construction to ensure correct broadcast semantics.

### 3. Transpose (`transpose_wh_block`)

**Base Tests**: Look for transpose tests in `tests/`

**Kernel Pattern**:
```cpp
// Initialize
transpose_wh_init(cb_in, cb_out);

// Block operation
transpose_wh_block<Ht, Wt>(cb_in, 0, 0);

// Pack
pack_block<Ht, Wt>(0, cb_out);
```

**Test Cases**:
- ✅ Square blocks (2×2, 4×4)
- ✅ Rectangular blocks (2×4, 4×2)
- ✅ Verify transpose correctness (each 32×32 tile is transposed)

**Golden Reference**: Transpose each 32×32 tile independently.

### 4. Reduce (`reduce_block`)

**Base Tests**: `tests/tt_metal/tt_metal/llk/test_reduce.cpp`

**Kernel Pattern**:
```cpp
// Initialize
reduce_init<REDUCE_OP, REDUCE_COL>(cb_in, cb_scaler, cb_out);

// Block operation
reduce_block<REDUCE_OP, REDUCE_COL, Ht, Wt>(cb_in, cb_scaler, 0, 0, 0);

// Pack
pack_block<Ht, Wt>(0, cb_out);
```

**Test Cases**:
- ✅ PoolType: SUM, AVG, MAX
- ✅ ReduceDim: REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR
- ✅ With/without scaler
- ✅ Various block sizes

**Special Note**: Reduce changes output dimensions. Be careful with expected output shapes.

### 5. Pack (`pack_block`)

**Note**: Pack is typically tested implicitly through other operations, but standalone tests are valuable.

**Kernel Pattern**:
```cpp
// Populate DEST with data (e.g., via add_block)
add_block<Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);

// Pack using block variant
pack_block<Ht, Wt>(0, cb_out);
```

**Test Cases**:
- ✅ Various block sizes
- ✅ Ensure pack_block produces same result as repeated pack_tile

---

## 📁 File Organization

### Recommended Structure

```
tests/tt_metal/tt_metal/
├── block_variants/
│   ├── kernels/
│   │   ├── compute_add_tiles.cpp         (reference: tile-by-tile)
│   │   ├── compute_add_block.cpp         (test: block operation)
│   │   ├── compute_sub_tiles.cpp
│   │   ├── compute_sub_block.cpp
│   │   ├── compute_mul_tiles.cpp
│   │   ├── compute_mul_block.cpp
│   │   ├── compute_bcast_tiles.cpp
│   │   ├── compute_bcast_block.cpp
│   │   ├── compute_transpose_tiles.cpp
│   │   ├── compute_transpose_block.cpp
│   │   ├── compute_reduce_tiles.cpp
│   │   ├── compute_reduce_block.cpp
│   │   └── dataflow_generic.cpp          (shared reader/writer)
│   ├── test_eltwise_binary_block.cpp
│   ├── test_bcast_block.cpp
│   ├── test_transpose_block.cpp
│   ├── test_reduce_block.cpp
│   └── test_pack_block.cpp
```

### CMakeLists.txt Entry

```cmake
# Add block variant tests
add_executable(test_eltwise_binary_block
    tests/tt_metal/tt_metal/block_variants/test_eltwise_binary_block.cpp
)
target_link_libraries(test_eltwise_binary_block PRIVATE tt::metal gtest gtest_main)
add_test(NAME EltwiseBinaryBlock COMMAND test_eltwise_binary_block)
```

---

## ✅ Agent Task Checklist

Each agent implementing a test suite should:

### Planning Phase
- [ ] Read this document completely
- [ ] Identify the base test to duplicate
- [ ] Understand the operation being tested
- [ ] List all test cases to implement

### Implementation Phase
- [ ] Create kernel directory if needed
- [ ] Implement reference kernel (tile-by-tile)
- [ ] Implement test kernel (block operation)
- [ ] Create test harness (Gtest)
- [ ] Implement test cases for all block sizes
- [ ] Add golden reference validation

### Validation Phase
- [ ] Ensure tests compile without errors
- [ ] Run clang-format on all files
- [ ] Run tests and verify they pass
- [ ] Check PCC ≥ 0.9999 for all cases
- [ ] Add test to CMakeLists.txt

### Documentation Phase
- [ ] Add docstring to each test
- [ ] Document any special considerations
- [ ] Update test README if exists

---

## 🚀 Automation Integration

The automation scripts can be extended to support test creation:

### Updated `add_block_variants.py` (Extension)

Add a new phase:

```python
class Phase8_CreateTests:
    """Phase 8: Generate test infrastructure"""

    def generate_test_kernel(self, operation, variant):
        """Generate compute kernel for testing"""
        # Use templates to create kernels
        # ...

    def generate_test_harness(self, operation):
        """Generate Gtest harness"""
        # ...
```

### Run Test Generation

```bash
./run_agent_implementation.sh --phase 8 --operation add_block
```

---

## 📊 Success Criteria

A test suite is complete when:

1. ✅ **All block sizes tested**: 1×1 through valid combinations up to 16 total tiles
2. ✅ **PCC ≥ 0.9999**: Block results match tile-by-tile results
3. ✅ **Golden validation**: Results match mathematical golden reference
4. ✅ **Multiple data formats**: FP16, BFP8, FP32 where applicable
5. ✅ **Edge cases covered**: Min (1×1) and max (16×1, 1×16) block sizes
6. ✅ **CI integration**: Tests run in continuous integration
7. ✅ **Documentation**: Each test is well-documented

---

## 🎯 Priority Order for Agents

Agents should tackle test suites in this order:

### Priority 1 (Foundational - Start Here)
1. **Element-wise Binary** (`add_block`, `sub_block`, `mul_block`)
   - Most basic operations
   - Foundation for other tests

### Priority 2 (High Value)
2. **Reduce** (`reduce_block`)
   - Complex operation, high value
3. **Broadcast** (`*_tiles_bcast_block`)
   - Commonly used, important to validate

### Priority 3 (Complete Coverage)
4. **Transpose** (`transpose_wh_block`)
   - Moderate complexity
5. **Pack** (`pack_block`)
   - Often tested implicitly, but explicit tests valuable

---

## 📚 References

- **Base Tests**: `tests/tt_metal/tt_metal/test_eltwise_binary.cpp`
- **UDM Example**: `tests/ttnn/unit_tests/gtests/udm/eltwise/test_udm_add.cpp`
- **Compute API**: `tt_metal/include/compute_kernel_api/`
- **Test Utils**: `tests/tt_metal/tt_metal/common/`
- **Issue**: [#35739](https://github.com/tenstorrent/tt-metal/issues/35739)
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Task Documentation**: `TASK.md`

---

## 🤖 Agent Execution

### For Parallel Execution

Each agent can work independently on a different test suite:

```bash
# Agent 1: Element-wise Binary
./scripts/generate_tests.sh --operation eltwise_binary --agent-id 1

# Agent 2: Reduce
./scripts/generate_tests.sh --operation reduce --agent-id 2

# Agent 3: Broadcast
./scripts/generate_tests.sh --operation broadcast --agent-id 3

# Agent 4: Transpose
./scripts/generate_tests.sh --operation transpose --agent-id 4

# Agent 5: Pack
./scripts/generate_tests.sh --operation pack --agent-id 5
```

### Serial Execution (Single Agent)

An agent can also implement all tests sequentially:

```bash
./scripts/generate_tests.sh --all --sequential
```

---

**Last Updated**: 2026-01-20
**Status**: Ready for Agent Implementation
**Owner**: AI Agents (Claude Sonnet 4)
