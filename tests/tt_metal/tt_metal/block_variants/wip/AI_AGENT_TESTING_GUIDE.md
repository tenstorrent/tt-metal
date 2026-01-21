# 🤖 AI Agent Testing - Complete Automation Guide

## TL;DR - Run This

```bash
cd /localdev/ncvetkovic/reconfig

# Option 1: Everything at once (parallel, ~2 minutes)
./run_test_completion.sh --generate-first --parallel

# Option 2: Step-by-step
./run_test_generation.sh --all            # Generate skeletons
./run_test_completion.sh --parallel       # AI agents complete TODOs
```

**That's it!** AI agents will automatically complete all TODO sections in the tests.

---

## 📋 What This Does

### Step 1: Generate Test Skeletons
Creates test files with TODO placeholders:
```cpp
void run_eltwise_binary_block_test(...) {
    // TODO: Implement full test harness
    // 1. Create programs
    // 2. Create buffers
    // ...
    GTEST_SKIP() << "Test implementation pending";
}
```

### Step 2: AI Agents Complete TODOs
Each AI agent (Claude Sonnet 4):
- Reads the test file with TODOs
- Reviews TESTING_PLAN.md and example tests
- Writes complete test implementation
- Replaces TODO with working code

### Result
Fully implemented tests ready to build and run!

---

## 🚀 Usage Options

### Quick Start (Recommended)

```bash
# Generate + Complete everything in parallel (5 agents, ~2 min)
./run_test_completion.sh --generate-first --parallel
```

### Sequential (One agent, ~10 min)

```bash
# Slower but uses less resources
./run_test_completion.sh --generate-first --all
```

### Single Operation

```bash
# Complete one specific operation
./run_test_generation.sh --operation eltwise_binary
./run_test_completion.sh --operation eltwise_binary
```

### Dry Run (Preview)

```bash
# See what would be done without making changes
./run_test_completion.sh --operation reduce --dry-run
```

---

## 🔧 Command Reference

### Generate Test Skeletons
```bash
./run_test_generation.sh --all                    # All operations
./run_test_generation.sh --operation eltwise_binary  # Specific op
./run_test_generation.sh --list                   # List operations
```

### Complete TODOs with AI Agents
```bash
./run_test_completion.sh --parallel               # Parallel (5 agents)
./run_test_completion.sh --all                    # Sequential (1 agent)
./run_test_completion.sh --operation reduce       # Single operation
./run_test_completion.sh --generate-first --parallel  # Gen + Complete
./run_test_completion.sh --dry-run --operation add   # Preview only
```

---

## 📊 Execution Modes

### Parallel Mode (Recommended)
```bash
./run_test_completion.sh --parallel
```
- **Agents**: 5 (one per operation)
- **Time**: ~2 minutes
- **Resource**: Higher (5 simultaneous API calls)
- **Best for**: Fast completion

**Operations run simultaneously**:
- Agent 1: `eltwise_binary`
- Agent 2: `reduce`
- Agent 3: `broadcast`
- Agent 4: `transpose`
- Agent 5: `pack`

### Sequential Mode
```bash
./run_test_completion.sh --all
```
- **Agents**: 1 (processes operations one by one)
- **Time**: ~10 minutes
- **Resource**: Lower (1 API call at a time)
- **Best for**: Conservative resource usage

**Operations run in order**:
1. `eltwise_binary` → 2. `reduce` → 3. `broadcast` → 4. `transpose` → 5. `pack`

### Single Operation Mode
```bash
./run_test_completion.sh --operation eltwise_binary
```
- **Agents**: 1 (only specified operation)
- **Time**: ~2 minutes per operation
- **Best for**: Testing or focused work

---

## 🔍 What Gets Created

### Before (Generated Skeleton)
```cpp
void run_eltwise_binary_block_test(Device* device, uint32_t Ht, uint32_t Wt, ...) {
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity";

    // TODO: Implement full test harness
    // 1. Create programs (reference and test)
    // 2. Create buffers and CBs
    // 3. Generate input data
    // 4. Run both programs
    // 5. Compare results with PCC >= 0.9999

    GTEST_SKIP() << "Test implementation pending";
}
```

### After (AI Completed)
```cpp
void run_eltwise_binary_block_test(Device* device, uint32_t Ht, uint32_t Wt, ...) {
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity";

    // Create programs
    Program program_ref = CreateProgram();
    Program program_test = CreateProgram();

    // Create buffers
    uint32_t single_tile_size = tile_size(data_format);
    uint32_t total_tiles = Ht * Wt * num_blocks;
    auto src0_buffer = CreateBuffer(/*...*/);
    auto src1_buffer = CreateBuffer(/*...*/);
    auto dst_ref_buffer = CreateBuffer(/*...*/);
    auto dst_test_buffer = CreateBuffer(/*...*/);

    // Setup circular buffers
    CircularBufferConfig cb_config = /*...*/;
    CreateCircularBuffer(program_ref, core, cb_config);
    CreateCircularBuffer(program_test, core, cb_config);

    // Create kernels
    auto kernel_ref = CreateKernel(program_ref, "kernels/compute_add_tiles.cpp", /*...*/);
    auto kernel_test = CreateKernel(program_test, "kernels/compute_add_block.cpp", /*...*/);

    // Generate test data
    std::vector<bfloat16> input_a = generate_random_bfloat16(/*...*/);
    std::vector<bfloat16> input_b = generate_random_bfloat16(/*...*/);

    // Write to device
    EnqueueWriteBuffer(device->command_queue(), src0_buffer, input_a, false);
    EnqueueWriteBuffer(device->command_queue(), src1_buffer, input_b, false);

    // Run both programs
    EnqueueProgram(device->command_queue(), program_ref, false);
    EnqueueProgram(device->command_queue(), program_test, false);

    // Read results
    std::vector<bfloat16> result_ref, result_test;
    EnqueueReadBuffer(device->command_queue(), dst_ref_buffer, result_ref, true);
    EnqueueReadBuffer(device->command_queue(), dst_test_buffer, result_test, true);

    // Validate
    float pcc = check_bfloat16_vector_pcc(result_ref, result_test);
    EXPECT_GE(pcc, 0.9999f) << "Block operation diverged!";

    // Golden reference
    std::vector<bfloat16> golden = compute_golden_add(input_a, input_b);
    EXPECT_GE(check_bfloat16_vector_pcc(golden, result_test), 0.9999f);
}
```

---

## ⚙️ How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ run_test_completion.sh (Orchestrator)                      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Agent 1   │  │   Agent 2   │  │   Agent 3   │  ...  │
│  │ eltwise_bin │  │   reduce    │  │  broadcast  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│         │                 │                 │              │
│         └─────────────────┴─────────────────┘              │
│                          │                                 │
│                          ▼                                 │
│              complete_test_todos.py                        │
│                          │                                 │
│                          ▼                                 │
│              Claude Sonnet 4 API                           │
│                (via ~/.bashrc credentials)                 │
└─────────────────────────────────────────────────────────────┘
```

### Agent Process (per operation)

1. **Read skeleton test file** with TODOs
2. **Load context**:
   - TESTING_PLAN.md (implementation guide)
   - Example tests (test_eltwise_binary.cpp)
   - Generated kernel files
3. **Call Claude API** with:
   - Test file content
   - Context and examples
   - Detailed instructions
4. **Parse response** and extract completed C++ code
5. **Write back** to test file

---

## 📁 Files Involved

### Generated by `run_test_generation.sh`
```
tt-metal/tests/tt_metal/tt_metal/block_variants/
├── kernels/
│   ├── compute_add_tiles.cpp       # Reference (tile-by-tile)
│   ├── compute_add_block.cpp       # Test (block operation)
│   └── ... (more kernels)
├── test_eltwise_binary_block.cpp   # ← HAS TODOs
├── test_broadcast_block.cpp        # ← HAS TODOs
├── test_transpose_block.cpp        # ← HAS TODOs
├── test_reduce_block.cpp           # ← HAS TODOs
└── test_pack_block.cpp             # ← HAS TODOs
```

### Modified by `run_test_completion.sh`
All test files above have TODOs replaced with working implementations.

---

## 🔐 Credentials Setup

The scripts use credentials from `~/.bashrc`:

```bash
export ANTHROPIC_API_KEY="sk-mYNQ0PYwWIeEGRRJO8NVjg"
export ANTHROPIC_BASE_URL="https://litellm-proxy--tenstorrent.workload.tenstorrent.com"
export ANTHROPIC_MODEL="anthropic/claude-sonnet-4-20250514"
```

These are automatically loaded by the scripts. ✅ Already configured!

---

## 🐛 Troubleshooting

### "anthropic package not installed"
```bash
pip install anthropic
```

### "ANTHROPIC_API_KEY not set"
```bash
source ~/.bashrc
echo $ANTHROPIC_API_KEY  # Should show your key
```

### "Test directory not found"
```bash
# Generate test skeletons first
./run_test_generation.sh --all
```

### "Agent failed"
- Check logs in `/tmp/agent_*.log` (parallel mode)
- Try single operation with `--dry-run` first
- Verify API credentials are correct

### Tests don't compile after completion
- Review the generated test file
- Check that kernel paths are correct
- Compare with existing test in `tests/tt_metal/tt_metal/test_eltwise_binary.cpp`

---

## ✅ Verification

### After completion, verify:

```bash
cd tt-metal

# Check generated tests don't have TODOs
grep -r "TODO" tests/tt_metal/tt_metal/block_variants/test_*.cpp
# (Should return no results)

# Check GTEST_SKIP was removed
grep -r "GTEST_SKIP" tests/tt_metal/tt_metal/block_variants/test_*.cpp
# (Should return no results)

# Build tests
./build_metal.sh --build-tests

# Run tests
./build/test/tt_metal/test_eltwise_binary_block
./build/test/tt_metal/test_broadcast_block
./build/test/tt_metal/test_transpose_block
./build/test/tt_metal/test_reduce_block
./build/test/tt_metal/test_pack_block
```

---

## 📊 Expected Timeline

| Mode | Time | Operations |
|------|------|------------|
| **Parallel** | ~2 min | All 5 (simultaneous) |
| **Sequential** | ~10 min | All 5 (one by one) |
| **Single** | ~2 min | One operation |

*Times include API calls, processing, and file writing*

---

## 🎯 Success Criteria

Tests are complete when:
- ✅ No TODO sections remain
- ✅ No GTEST_SKIP() calls
- ✅ All functions implemented:
  - Buffer creation
  - CB setup
  - Kernel creation
  - Data generation
  - Execution
  - Validation (PCC >= 0.9999)
- ✅ Tests compile successfully
- ✅ Tests pass when run

---

## 🚀 Complete Workflow

### End-to-End (Recommended)

```bash
cd /localdev/ncvetkovic/reconfig

# 1. Generate + Complete (parallel, ~2 min)
./run_test_completion.sh --generate-first --parallel

# 2. Build tests
cd tt-metal
./build_metal.sh --build-tests

# 3. Run tests
./build/test/tt_metal/test_eltwise_binary_block
./build/test/tt_metal/test_broadcast_block
./build/test/tt_metal/test_reduce_block
./build/test/tt_metal/test_transpose_block
./build/test/tt_metal/test_pack_block

# 4. If all pass, commit!
git add tests/tt_metal/tt_metal/block_variants/
git commit -m "#35739: Add block variant tests (AI agent generated)"
```

---

## 📚 Additional Resources

- **TESTING_PLAN.md** - Comprehensive test implementation guide
- **TESTING_QUICK_START.md** - Quick testing reference
- **FINAL_SUMMARY.md** - Complete project summary
- **Existing tests** - `tests/tt_metal/tt_metal/test_eltwise_binary.cpp`

---

**🎉 Ready to automate! Just run: `./run_test_completion.sh --generate-first --parallel`**

---

**Last Updated**: 2026-01-20
**Status**: Ready for Use
**Issue**: [#35739](https://github.com/tenstorrent/tt-metal/issues/35739)
