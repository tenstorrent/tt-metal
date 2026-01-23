# Block Variant Testing - Work In Progress

This directory contains automation scripts and documentation for generating and running block variant tests.

## 📍 Location

**Current Directory:**
```
tt-metal/tests/tt_metal/tt_metal/block_variants/wip/
```

All scripts in this directory have been updated to work from this location.

---

## 🚀 Quick Start

### Run All Tests (Build + Run)

```bash
cd /localdev/ncvetkovic/reconfig/tt-metal/tests/tt_metal/tt_metal/block_variants/wip
./run_block_tests.sh
```

### Add Tests to Build System

```bash
./add_tests_to_cmake.sh
```

### Complete Workflow (Generate → AI Complete → Build → Run)

```bash
./COMPLETE_WORKFLOW.sh
```

---

## 📂 Directory Structure

```
wip/                           # This directory (automation & docs)
├── run_block_tests.sh         # Build & run all tests
├── add_tests_to_cmake.sh      # Add tests to CMakeLists.txt
├── run_test_generation.sh     # Generate test skeletons
├── run_test_completion.sh     # AI complete TODOs
├── COMPLETE_WORKFLOW.sh       # Full pipeline
├── generate_block_tests.py    # Python test generator
├── complete_test_todos.py     # Python AI orchestrator
└── *.md                       # Documentation files

../                            # Parent directory (block_variants/)
├── test_eltwise_binary_block.cpp
├── test_reduce_block.cpp
├── test_broadcast_block.cpp
├── test_transpose_block.cpp
├── test_pack_block.cpp
└── kernels/                   # Test compute kernels

../../CMakeLists.txt           # Build configuration (2 levels up)

../../../../..                 # tt-metal repo root (5 levels up)
```

---

## 🔧 Available Scripts

### Testing Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `run_block_tests.sh` | Build & run all tests | `wip/` |
| `add_tests_to_cmake.sh` | Add to build system | `wip/` |

### Test Generation Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `run_test_generation.sh` | Generate test skeletons | `wip/` |
| `run_test_completion.sh` | AI complete TODOs | `wip/` |
| `generate_block_tests.py` | Python generator | `wip/` |
| `complete_test_todos.py` | Python AI orchestrator | `wip/` |

### Workflow Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `COMPLETE_WORKFLOW.sh` | Full end-to-end pipeline | `wip/` |

---

## 📋 Path Configuration

All scripts have been updated to work from the `wip/` directory:

- **tt-metal root:** `../../../../..` (5 levels up)
- **Test files:** `..` (parent directory)
- **CMakeLists.txt:** `../../CMakeLists.txt`
- **Kernels:** `../kernels/`

---

## 🎯 Common Tasks

### 1. Run Tests

```bash
# Quick way
./run_block_tests.sh

# Manual way
cd ../../../../../..           # Go to tt-metal root
./build_metal.sh --build-tests
./build/test/tt_metal/test_eltwise_binary_block
```

### 2. Regenerate Tests

```bash
# Generate skeletons
./run_test_generation.sh --all

# AI complete TODOs (requires ANTHROPIC_API_KEY)
./run_test_completion.sh --parallel

# Add to build
./add_tests_to_cmake.sh
```

### 3. Add to Build System

```bash
./add_tests_to_cmake.sh
```

This adds test targets to `../../CMakeLists.txt`:
```cmake
tt_metal_add_gtest(test_eltwise_binary_block
    block_variants/test_eltwise_binary_block.cpp
)
```

---

## 📝 Test Files Generated

All test files are in the parent directory (`../`):

- `test_eltwise_binary_block.cpp` (17 test cases)
- `test_reduce_block.cpp` (17 test cases)
- `test_broadcast_block.cpp` (17 test cases)
- `test_transpose_block.cpp` (17 test cases)
- `test_pack_block.cpp` (17 test cases)

**Total:** 85 test cases

---

## 🤖 AI Agent Usage

### Prerequisites

```bash
# Check API key is set (from ~/.bashrc)
echo $ANTHROPIC_API_KEY

# Install anthropic package
pip install anthropic
```

### Generate & Complete Tests

```bash
# Single operation
./run_test_completion.sh --operation eltwise_binary

# All operations (parallel)
./run_test_completion.sh --parallel

# Dry run (preview only)
./run_test_completion.sh --operation reduce --dry-run
```

---

## ✅ Verification

Check everything is ready:

```bash
# 1. Test files exist
ls ../*.cpp

# 2. No TODOs remaining
grep -r "TODO" ../*.cpp || echo "✅ No TODOs"

# 3. Tests in CMakeLists.txt
grep "test_eltwise_binary_block" ../../CMakeLists.txt

# 4. Scripts executable
ls -l *.sh
```

---

## 🐛 Troubleshooting

### Problem: Script fails with "directory not found"

**Solution:** Make sure you're in the `wip/` directory:
```bash
cd /localdev/ncvetkovic/reconfig/tt-metal/tests/tt_metal/tt_metal/block_variants/wip
pwd  # Should show path ending in /wip
```

### Problem: Tests not found after build

**Solution:** Add tests to CMakeLists.txt first:
```bash
./add_tests_to_cmake.sh
```

### Problem: AI agent fails (anthropic package)

**Solution:** Install package and check API key:
```bash
pip install anthropic
echo $ANTHROPIC_API_KEY  # Should not be empty
```

---

## 📚 Documentation

All documentation is in this directory:

- `WORKFLOW_GUIDE.md` - Complete workflow documentation
- `TESTING_PLAN.md` - Detailed testing strategy
- `AI_AGENT_TESTING_GUIDE.md` - AI agent instructions
- `TASK.md` - Original problem & requirements
- `QUICK_REFERENCE.txt` - Command cheat sheet

---

## 🎉 Quick Reference

```bash
# From wip/ directory:

# Run tests
./run_block_tests.sh

# Add to build
./add_tests_to_cmake.sh

# Full workflow
./COMPLETE_WORKFLOW.sh

# Generate tests
./run_test_generation.sh --all

# AI complete
./run_test_completion.sh --parallel
```

---

**Status:** ✅ All scripts updated for new location
**Location:** `tt-metal/tests/tt_metal/tt_metal/block_variants/wip/`
**Ready:** All 85 tests generated and ready to run
