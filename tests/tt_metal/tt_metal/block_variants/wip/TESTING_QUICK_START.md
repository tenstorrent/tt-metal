# 🧪 Testing Block Variants - Quick Start Guide

## Overview

This guide shows you how to generate and run tests for the newly implemented block variant functions. Tests validate that block operations produce **identical results** to tile-by-tile processing.

---

## 🚀 Quick Start (30 seconds)

```bash
cd /localdev/ncvetkovic/reconfig

# Generate all tests
./run_test_generation.sh --all

# Or generate tests for specific operation
./run_test_generation.sh --operation eltwise_binary
```

That's it! Test skeletons are now ready for agents to complete.

---

## 📋 What Operations Can Be Tested?

| Operation | Functions | Priority | Command |
|-----------|-----------|----------|---------|
| **Element-wise Binary** | `add_block`, `sub_block`, `mul_block` | HIGH | `--operation eltwise_binary` |
| **Broadcast** | `*_tiles_bcast_block` (add/sub/mul) | HIGH | `--operation broadcast` |
| **Reduce** | `reduce_block` | HIGH | `--operation reduce` |
| **Transpose** | `transpose_wh_block` | MEDIUM | `--operation transpose` |
| **Pack** | `pack_block` | MEDIUM | `--operation pack` |

---

## 🎯 Test Strategy

### The Core Idea

For each block operation, we create **two compute kernels**:

1. **Reference Kernel** (tile-by-tile) - Known good implementation
2. **Test Kernel** (block operation) - What we're testing

Run both with identical input data → Compare outputs → Must match with PCC ≥ 0.9999

### Example

```cpp
// Reference: tile-by-tile processing
for (uint32_t h = 0; h < Ht; h++) {
    for (uint32_t w = 0; w < Wt; w++) {
        add_tiles(cb_in0, cb_in1, idx, idx, idx);  // Process one tile
    }
}

// Test: block operation
add_block<Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);  // Process all tiles at once

// ✅ Both must produce identical results!
```

---

## 📁 Generated Files

After running the generator, you'll find:

```
tt-metal/tests/tt_metal/tt_metal/block_variants/
├── kernels/
│   ├── compute_add_tiles.cpp        # Reference (tile-by-tile)
│   ├── compute_add_block.cpp        # Test (block operation)
│   ├── compute_sub_tiles.cpp
│   ├── compute_sub_block.cpp
│   └── ...
├── test_eltwise_binary_block.cpp    # Gtest harness
├── test_broadcast_block.cpp
├── test_reduce_block.cpp
├── test_transpose_block.cpp
└── test_pack_block.cpp
```

---

## 🔧 Usage Examples

### List Available Operations

```bash
./run_test_generation.sh --list
```

Output:
```
📋 Available Operations:

  eltwise_binary (Priority 1)
    Functions: add, sub, mul
    Header: compute_kernel_api/eltwise_binary.h

  broadcast (Priority 2)
    Functions: add_tiles_bcast, sub_tiles_bcast, mul_tiles_bcast
    Header: compute_kernel_api/bcast.h

  ...
```

### Generate Tests for Specific Operation

```bash
# Element-wise binary (add, sub, mul)
./run_test_generation.sh --operation eltwise_binary

# Broadcast operations
./run_test_generation.sh --operation broadcast

# Reduce operations
./run_test_generation.sh --operation reduce
```

### Generate All Tests

```bash
./run_test_generation.sh --all
```

### Custom Output Directory

```bash
./run_test_generation.sh --operation reduce --output /path/to/output
```

### Dry Run (Preview)

```bash
./run_test_generation.sh --all --dry-run
```

---

## 🧑‍💻 For AI Agents

### Task Assignment

Each agent can work on a different test suite **independently**:

```bash
# Agent 1: Element-wise Binary (Priority 1)
./run_test_generation.sh --operation eltwise_binary

# Agent 2: Reduce (Priority 1)
./run_test_generation.sh --operation reduce

# Agent 3: Broadcast (Priority 2)
./run_test_generation.sh --operation broadcast

# Agent 4: Transpose (Priority 3)
./run_test_generation.sh --operation transpose

# Agent 5: Pack (Priority 3)
./run_test_generation.sh --operation pack
```

### What Agents Need to Do

The generator creates **skeleton tests** with TODO markers. Agents complete:

1. **Review generated kernels** - Verify tile-by-tile and block variants are correct
2. **Complete test harness** - Fill in TODO sections for:
   - Buffer creation
   - Input data generation
   - Program execution
   - Result comparison (PCC validation)
3. **Add to build system** - Update CMakeLists.txt
4. **Run and validate** - Ensure tests pass

### Detailed Agent Instructions

See **`TESTING_PLAN.md`** for comprehensive 300-line guide with:
- Step-by-step implementation
- Code templates
- Operation-specific guidance
- Test matrix (block sizes, data formats)
- Golden reference examples

---

## 🏗️ Building and Running Tests

### Build Tests

```bash
cd /localdev/ncvetkovic/reconfig/tt-metal

# Build entire project with tests
./build_metal.sh --build-tests

# Or build specific test
cd build
cmake --build . --target test_eltwise_binary_block
```

### Run Tests

```bash
# Run all block variant tests
./build/test/tt_metal/test_eltwise_binary_block
./build/test/tt_metal/test_broadcast_block
./build/test/tt_metal/test_reduce_block
./build/test/tt_metal/test_transpose_block
./build/test/tt_metal/test_pack_block

# Or use Gtest filters
./build/test/tt_metal/test_eltwise_binary_block --gtest_filter="*AddBlock_2x2*"

# Run with verbose output
./build/test/tt_metal/test_eltwise_binary_block --gtest_verbose
```

---

## 📊 Test Coverage

### Block Sizes Tested

All valid combinations where **Ht × Wt ≤ 16**:

| Height (Ht) | Width (Wt) | Total Tiles |
|-------------|------------|-------------|
| 1 | 1, 2, 4, 8, 16 | 1-16 |
| 2 | 1, 2, 4, 8 | 2-16 |
| 4 | 1, 2, 4 | 4-16 |
| 8 | 1, 2 | 8-16 |
| 16 | 1 | 16 |

**Total: 15 test cases per operation**

### Data Formats

- Float16_b (BFloat16)
- BFP8_b (Block Floating Point 8)
- Float32 (where supported)

### Validation Criteria

✅ **PCC (Pearson Correlation Coefficient) ≥ 0.9999**
- Compares block result vs tile-by-tile result
- Also compares against golden mathematical reference

---

## 🐛 Troubleshooting

### "Python not found"

```bash
# Install Python 3
sudo apt-get install python3
```

### "Repository not found"

```bash
# Verify path
ls /localdev/ncvetkovic/reconfig/tt-metal

# Clone if needed
cd /localdev/ncvetkovic/reconfig
git clone https://github.com/tenstorrent/tt-metal.git
```

### "Test generation failed"

```bash
# Check Python script directly
python3 generate_block_tests.py --list

# Verify script is executable
chmod +x generate_block_tests.py run_test_generation.sh
```

### "Build errors"

```bash
# Clean and rebuild
cd tt-metal
rm -rf build
./build_metal.sh --build-tests
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **TESTING_QUICK_START.md** | This file - Quick reference |
| **TESTING_PLAN.md** | Comprehensive 300-line guide for agents |
| **IMPLEMENTATION_SUMMARY.md** | Summary of implemented block variants |
| **TASK.md** | Original implementation task description |
| **COMPLETED_WORK_SUMMARY.md** | Final summary of completed work |

---

## 💡 Tips

### For Sequential Implementation

Generate tests in this order:

```bash
# Priority 1 - Start here
./run_test_generation.sh --operation eltwise_binary
./run_test_generation.sh --operation reduce

# Priority 2 - Next
./run_test_generation.sh --operation broadcast

# Priority 3 - Finish here
./run_test_generation.sh --operation transpose
./run_test_generation.sh --operation pack
```

### For Parallel Implementation

All operations are independent - agents can work simultaneously:

```bash
# 5 agents can work in parallel, one per operation
Agent1: eltwise_binary
Agent2: reduce
Agent3: broadcast
Agent4: transpose
Agent5: pack
```

### Quick Validation

To quickly check if generated tests compile:

```bash
cd tt-metal/tests/tt_metal/tt_metal/block_variants

# Try compiling a single kernel
clang++ -c -I../../../../tt_metal/include kernels/compute_add_block.cpp
```

---

## 🎯 Success Metrics

Tests are complete when:

- ✅ All 15 block sizes tested per operation
- ✅ PCC ≥ 0.9999 for all test cases
- ✅ Tests pass in CI/CD pipeline
- ✅ No memory leaks or undefined behavior
- ✅ Performance is reasonable (not significantly slower than tile-by-tile)

---

## 🚀 Ready to Go!

```bash
# Generate all tests (recommended)
./run_test_generation.sh --all

# Or start with priority operations
./run_test_generation.sh --operation eltwise_binary

# Review generated files
ls -la tt-metal/tests/tt_metal/tt_metal/block_variants/
```

**Next**: See `TESTING_PLAN.md` for detailed implementation guidance.

---

**Last Updated**: 2026-01-20
**Issue**: [#35739](https://github.com/tenstorrent/tt-metal/issues/35739)
**Status**: Ready for Agent Implementation
