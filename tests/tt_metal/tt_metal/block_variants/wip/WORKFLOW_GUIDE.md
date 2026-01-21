# Complete Block Variant Testing Workflow

**Issue #35739** - Adding missing Compute API block variants with automated testing

---

## 🎯 Quick Start (One Command)

```bash
cd /localdev/ncvetkovic/reconfig
./COMPLETE_WORKFLOW.sh
```

This runs the entire pipeline:
1. ✅ Generate test skeletons
2. 🤖 AI agents complete TODOs (parallel)
3. ⚙️ Add tests to CMakeLists.txt
4. 🔨 Build tests
5. 🧪 Run all tests

**Time:** ~5-10 minutes total

---

## 📋 Step-by-Step Workflow

### Step 1: Generate Test Skeletons

Generate test harnesses with TODO sections:

```bash
# All operations
./run_test_generation.sh --all

# Single operation
./run_test_generation.sh --operation eltwise_binary
```

**Output:** Test files in `tt-metal/tests/tt_metal/tt_metal/block_variants/`

---

### Step 2: AI Agents Complete TODOs

Use Claude Sonnet 4 to fill in TODO sections:

```bash
# All operations in parallel (~2 min)
./run_test_completion.sh --parallel

# Single operation
./run_test_completion.sh --operation eltwise_binary

# Dry run (preview only)
./run_test_completion.sh --operation eltwise_binary --dry-run
```

**Requirements:**
- `ANTHROPIC_API_KEY` set in environment (from `~/.bashrc`)
- `anthropic` Python package installed

**Output:** Complete test implementations, 0 TODOs remaining

---

### Step 3: Add Tests to Build System

Add test targets to CMakeLists.txt:

```bash
./add_tests_to_cmake.sh
```

**What it does:**
- Backs up original `CMakeLists.txt`
- Adds 5 test targets using `tt_metal_add_gtest()`
- Shows next steps

**Manual alternative:**
Edit `tt-metal/tests/tt_metal/tt_metal/CMakeLists.txt` and add:

```cmake
# Block variant tests (Issue #35739)
tt_metal_add_gtest(test_eltwise_binary_block
    block_variants/test_eltwise_binary_block.cpp
)

tt_metal_add_gtest(test_reduce_block
    block_variants/test_reduce_block.cpp
)

tt_metal_add_gtest(test_broadcast_block
    block_variants/test_broadcast_block.cpp
)

tt_metal_add_gtest(test_transpose_block
    block_variants/test_transpose_block.cpp
)

tt_metal_add_gtest(test_pack_block
    block_variants/test_pack_block.cpp
)
```

---

### Step 4: Build Tests

Build the test executables:

```bash
cd tt-metal
./build_metal.sh --build-tests
```

**Time:** ~3-5 minutes

**Output:** Test executables in `build/test/tt_metal/test_*_block`

---

### Step 5: Run Tests

Run the block variant tests:

```bash
# All tests
./run_block_tests.sh

# Single test
cd tt-metal
./build/test/tt_metal/test_eltwise_binary_block
```

**Expected output:**
```
[==========] Running tests...
[----------] 17 tests from EltwiseBinaryBlockTest
[ RUN      ] EltwiseBinaryBlockTest.AddBlock_2x2
[       OK ] EltwiseBinaryBlockTest.AddBlock_2x2 (120 ms)
...
[  PASSED  ] 17 tests.
```

---

## 📊 Test Coverage

### Generated Tests (85 total test cases)

| Operation | File | Test Cases | Operations |
|-----------|------|------------|------------|
| Eltwise Binary | `test_eltwise_binary_block.cpp` | 17 | add, sub, mul |
| Reduce | `test_reduce_block.cpp` | 17 | sum, max, min |
| Broadcast | `test_broadcast_block.cpp` | 17 | row, col, scalar |
| Transpose | `test_transpose_block.cpp` | 17 | WH transpose |
| Pack | `test_pack_block.cpp` | 17 | DEST to L1 |

### Test Matrix (per operation)

Each operation tests multiple block sizes:
- 1×1 (single tile)
- 2×2 (4 tiles)
- 4×4 (16 tiles, max DEST capacity)
- 2×4, 4×2 (rectangular blocks)
- Edge cases and error conditions

---

## 🔧 Available Scripts

### Core Workflow Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `COMPLETE_WORKFLOW.sh` | Full end-to-end pipeline | ~5-10 min |
| `run_test_generation.sh` | Generate test skeletons | ~5 sec |
| `run_test_completion.sh` | AI agents complete TODOs | ~2 min |
| `add_tests_to_cmake.sh` | Add to build system | ~1 sec |
| `run_block_tests.sh` | Build and run tests | ~5 min |

### Supporting Scripts

| Script | Purpose |
|--------|---------|
| `generate_block_tests.py` | Python test generator |
| `complete_test_todos.py` | Python AI agent orchestrator |
| `run_agent_implementation.sh` | API implementation (already done) |
| `add_block_variants.py` | API implementation script |

---

## 🎓 Understanding the Tests

### Test Structure

Each test file follows this pattern:

```cpp
// 1. Test fixture class
class EltwiseBinaryBlockTest : public ::testing::Test {
protected:
    void SetUp() override { /* device setup */ }
    void TearDown() override { /* cleanup */ }
    Device* device_;
};

// 2. Helper function to run tests
void run_eltwise_binary_block_test(
    Device* device,
    uint32_t Ht,      // Block height in tiles
    uint32_t Wt,      // Block width in tiles
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b
) {
    // - Validate block size (Ht * Wt <= 16)
    // - Create reference program (tile-by-tile)
    // - Create test program (block-based)
    // - Compare results with PCC
}

// 3. Test cases
TEST_F(EltwiseBinaryBlockTest, AddBlock_2x2) {
    run_eltwise_binary_block_test(device_, 2, 2);
}
```

### What Tests Validate

1. **Correctness:** Block results match tile-by-tile reference
2. **Block Sizes:** 1×1 up to 4×4 (max DEST capacity)
3. **Data Formats:** Float16_b, BFloat16, etc.
4. **Edge Cases:** Single tile, max capacity, rectangular blocks
5. **Numerical Accuracy:** PCC (Pearson Correlation Coefficient) > 0.99

---

## 🐛 Troubleshooting

### Test Generation Issues

**Problem:** `anthropic package not installed`
```bash
pip install anthropic
```

**Problem:** `ANTHROPIC_API_KEY not set`
```bash
# Check ~/.bashrc for:
export ANTHROPIC_API_KEY="your-key"
export ANTHROPIC_BASE_URL="https://your-endpoint"
```

### Build Issues

**Problem:** `Test not found after build`
- Check if test was added to `CMakeLists.txt`
- Run `./add_tests_to_cmake.sh` again

**Problem:** `clang-format violations`
```bash
cd tt-metal
clang-format -i tests/tt_metal/tt_metal/block_variants/*.cpp
```

### Test Failures

**Problem:** Test fails with `DEST capacity exceeded`
- Block size too large (Ht × Wt > 16)
- Check `static_assert` in API functions

**Problem:** PCC too low (< 0.99)
- Numerical precision issue
- Check data format compatibility
- Review compute kernel implementation

---

## 📁 File Structure

```
/localdev/ncvetkovic/reconfig/
├── COMPLETE_WORKFLOW.sh          ⭐ Run everything
├── run_test_generation.sh         📝 Generate skeletons
├── run_test_completion.sh         🤖 AI completion
├── add_tests_to_cmake.sh          ⚙️  Add to build
├── run_block_tests.sh             🧪 Run tests
│
├── generate_block_tests.py        (Python generator)
├── complete_test_todos.py         (Python AI orchestrator)
│
├── TESTING_PLAN.md                (AI agent instructions)
├── WORKFLOW_GUIDE.md              (This file)
│
└── tt-metal/
    ├── tests/tt_metal/tt_metal/
    │   ├── block_variants/        ✅ Generated tests here
    │   │   ├── test_eltwise_binary_block.cpp
    │   │   ├── test_reduce_block.cpp
    │   │   ├── test_broadcast_block.cpp
    │   │   ├── test_transpose_block.cpp
    │   │   └── test_pack_block.cpp
    │   └── CMakeLists.txt         (Add tests here)
    │
    └── tt_metal/include/compute_kernel_api/
        ├── eltwise_binary.h       (add_block, sub_block, mul_block)
        ├── reduce_custom.h        (reduce_block)
        ├── bcast.h                (*_bcast_block)
        ├── transpose_wh.h         (transpose_wh_block)
        └── pack.h                 (pack_block)
```

---

## ✅ Verification Checklist

After running the workflow:

- [ ] All test files generated (5 files)
- [ ] No TODOs remaining in test files
- [ ] Tests added to CMakeLists.txt
- [ ] Build completes without errors
- [ ] All tests pass (85 test cases)
- [ ] No clang-format violations
- [ ] Code ready for PR

---

## 🚀 Next Steps

### For Development

1. **Review generated tests:** Check for correctness
2. **Add more test cases:** Edge cases, error conditions
3. **Performance testing:** Benchmark block vs tile-by-tile
4. **Documentation:** Update API docs with block variants

### For CI/CD

1. **Add to CI pipeline:** Run tests on every commit
2. **Coverage analysis:** Ensure all code paths tested
3. **Regression testing:** Prevent future breakage

### For Production

1. **Code review:** Get team feedback
2. **Create PR:** Submit to tt-metal repository
3. **Update docs:** User-facing documentation
4. **Announce:** Let users know about new APIs

---

## 📞 Support

- **Issue:** https://github.com/tenstorrent/tt-metal/issues/35739
- **Docs:** See `TESTING_PLAN.md` for detailed AI agent instructions
- **Quick Start:** See `TESTING_QUICK_START.md` for condensed guide

---

**Last Updated:** 2026-01-20
**Status:** ✅ Complete - All tests generated and ready to run
