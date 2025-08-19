# 🔧 Automated Hardware Debugging Tool

## 📁 Directory Structure

```
automated_hardware_debugger/
├── automated_hardware_debugger.py        # Main standalone tool
├── example_usage_standalone.py           # Usage examples and demonstrations
├── AUTOMATED_HARDWARE_DEBUGGER_README.md # Comprehensive documentation
├── HACKATHON_TOOL_SUMMARY.md            # Quick hackathon overview
├── TESTING_VERIFICATION_COMPLETE.md     # Testing and verification records
├── TOOL_ORGANIZATION_SUMMARY.md         # Tool organization and development summary
├── MODIFIED_FILES_EXPLANATION.md        # Analysis of code injection process
├── DUPLICATION_FIX_SUMMARY.md          # Details on code duplication prevention fixes
├── modified_files_snapshot/              # Snapshots of modified files (auto-generated)
│   ├── modified_test_permute.py         # Python test with injected debugging loops
│   ├── modified_permute_rm_program_factory.cpp # C++ factory with environment variable handling
│   └── modified_transpose_xw_rm_single_tile_size.cpp # C++ kernel with injected NOPs
└── README.md                             # This file
```

## 🚀 Quick Start

### From Project Root
```bash
# Use the convenient wrapper (recommended)
./debug-tool --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked

# Or call directly
./automated_hardware_debugger/automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked
```

### From This Directory
```bash
cd automated_hardware_debugger/

# Run the tool
./automated_hardware_debugger.py --test-file ../tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked

# View examples
./example_usage_standalone.py

# Read documentation
cat AUTOMATED_HARDWARE_DEBUGGER_README.md
```

## 🔍 Robust File Discovery

This version includes **4 search strategies** that work when glob patterns fail:

### 1. **Git-based Search** (Most Reliable)
- Uses `git ls-files *.cpp` to get all tracked C++ files
- Filters by operation type and file patterns
- Works perfectly in git repositories

### 2. **Find Command Search**
- Uses system `find` command with specific patterns
- Works when git is not available
- More reliable than glob patterns

### 3. **Content-based Search**
- Uses `grep` to find files containing specific code patterns
- Searches for `ComputeConfig`, `CreateKernel`, `MAIN`, etc.
- Finds files that actually implement the functionality

### 4. **Direct Path Search**
- Checks known common locations for files
- Uses `Path.rglob()` for recursive searching
- Fallback when other methods fail

## 🎯 Features

- ✅ **Smart Operation Detection**: Automatically detects operation type (permute, conv, matmul, etc.)
- ✅ **Multi-Strategy File Discovery**: 4 different methods to find files reliably
- ✅ **Comprehensive Error Handling**: Graceful fallbacks when strategies fail
- ✅ **Detailed Progress Reporting**: Shows which strategy found each file
- ✅ **Automatic Backup & Restore**: All modifications are temporary and safe
- ✅ **Modified File Snapshots**: Saves copies of injected code for inspection
- ✅ **Duplicate Prevention**: Ensures clean, single injections without code duplication

## 📊 Example Output

```
🚀 Starting Automated Hardware Debugging Session
================================================================================
🔧 Injecting debugging loops into test_permute_5d_blocked in tests/ttnn/unit_tests/operations/test_permute.py
✅ Successfully injected debugging loops into test_permute_5d_blocked
🔍 Searching for program factory file...
📍 Detected operation type: permute
  🔄 Trying Git-based search...
✅ Found program factory: ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_rm_program_factory.cpp
🔍 Searching for compute kernel file...
📍 Detected operation type: permute
  🔄 Trying Git-based search...
✅ Found compute kernel: ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp

🧪 Running modified test...
[... test execution ...]

🔄 Restoring all modified files...
✅ All files restored to original state
```

## 🛠️ Command Line Options

- `--test-file`: Path to the Python test file
- `--function`: Name of the test function to debug
- `--max-nops`: Maximum number of NOPs to test (default: 100)
- `--iterations`: Number of iterations per configuration (default: 10)
- `--backup-dir`: Directory to store backup files (default: ./debug_backups)

## 📚 More Information

- **Full Documentation**: `AUTOMATED_HARDWARE_DEBUGGER_README.md`
- **Hackathon Summary**: `HACKATHON_TOOL_SUMMARY.md`
- **Usage Examples**: `./example_usage_standalone.py`

---

**🏆 Ready for hackathon demonstrations with robust file discovery!**
