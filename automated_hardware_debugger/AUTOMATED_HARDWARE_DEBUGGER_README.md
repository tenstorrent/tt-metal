# Automated Hardware Debugging Tool

## Overview

A standalone tool that automatically injects debugging code into hardware test functions, program factories, and compute kernels to find minimum failing configurations for hardware debugging and validation.

## 🎯 Purpose

This tool **automatically transforms** any hardware test function by:

1. **Injecting debugging loops** that test various NOP configurations
2. **Finding and modifying** the relevant program factory to pass environment variables
3. **Finding and modifying** the relevant compute kernel to inject NOPs
4. **Running the modified test** and collecting failure statistics
5. **Analyzing results** to find optimal debugging configurations
6. **Restoring all files** to their original state after completion

## 🚀 Quick Start

### Basic Usage

```bash
# Debug a specific test function
./automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked

# With custom parameters
./automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked --max-nops 50 --iterations 5
```

### Command Line Arguments

- `--test-file`: Path to the Python test file
- `--function`: Name of the test function to debug
- `--max-nops`: Maximum number of NOPs to test (default: 100)
- `--iterations`: Number of iterations per configuration (default: 10)
- `--backup-dir`: Directory to store backup files (default: ./debug_backups)
- `--skip-build`: Skip the C++ project build step (faster, but C++ modifications won't take effect)

## 🔧 How It Works

### 1. Code Injection (3 Files Modified)
The tool automatically modifies three types of files:

### 2. Project Build (Critical Step)
After modifying C++ files, the tool automatically builds the project:
```bash
export TT_METAL_ENV=dev && ./build_metal.sh --enable-profiler --build-tests
```
This ensures that C++ modifications take effect before running tests.

### 3. Test Function Transformation

**Original test function:**
```python
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    torch.manual_seed(520)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config)
    tt_output = ttnn.permute(tt_input, perm)
    tt_output = ttnn.to_torch(tt_output)

    assert_equal(torch_output, tt_output)
```

**Automatically injected debugging code:**
```python
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    device.disable_and_clear_program_cache()
    nop_types = ["UNOPS", "MNOPS", "PNOPS"]

    torch.manual_seed(520)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    all_results = []
    for is_risc in range(2):
        os.environ["RISCV"] = str(is_risc)
        for core_nop in nop_types:
            for nops in range(100):  # max_nops
                os.environ[core_nop] = str(nops)
                counter = 0

                for i in range(10):  # iterations
                    try:
                        tt_input = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config)
                        tt_output = ttnn.permute(tt_input, perm)
                        tt_output = ttnn.to_torch(tt_output)

                        if torch.equal(torch_output, tt_output):
                            counter += 1
                    except:
                        pass  # Count as failure

                failures = 10 - counter
                if failures > 0:
                    # Store failure data for analysis
                    all_results.append({...})
```

### 2. Program Factory Modification

**Automatically finds and modifies** the program factory (e.g., `permute_rm_program_factory.cpp`) to:

```cpp
// Injected environment variable handling
std::map<std::string, std::string> compute_defines;
compute_defines["UNOPS"] = std::to_string(std::getenv("UNOPS") ? std::stoi(std::getenv("UNOPS")) : 0);
compute_defines["MNOPS"] = std::to_string(std::getenv("MNOPS") ? std::stoi(std::getenv("MNOPS")) : 0);
compute_defines["PNOPS"] = std::to_string(std::getenv("PNOPS") ? std::stoi(std::getenv("PNOPS")) : 0);
compute_defines["RISCV"] = std::to_string(std::getenv("RISCV") ? std::stoi(std::getenv("RISCV")) : 0);

// Modified ComputeConfig
tt::tt_metal::ComputeConfig{
    .fp32_dest_acc_en = fp32_dest_acc_en,
    .compile_args = compute_kernel_args,
    .defines = compute_defines  // <- Injected
}
```

### 3. Compute Kernel Modification

**Automatically finds and modifies** the compute kernel (e.g., `transpose_xw_rm_single_tile_size.cpp`) to:

```cpp
// Injected NOP functions
template <const int n, const int riscv>
inline void add_nops() {
    for (int i = 0; i < n; i++) {
        if constexpr (riscv) {
            asm("nop");
        } else {
            TTI_NOP;
        }
    }
}

template <const int U, const int M, const int P, const int R>
inline void add_trisc_nops() {
    if constexpr (U) UNPACK((add_nops<U, R>()));
    if constexpr (M) MATH((add_nops<M, R>()));
    if constexpr (P) PACK((add_nops<P, R>()));
}

void MAIN {
    // ... existing code ...

    for (uint32_t n = 0; n < num_blocks; n++) {
        add_trisc_nops<UNOPS, MNOPS, PNOPS, RISCV>();  // <- Injected
        // ... rest of loop ...
    }
}
```

## 📊 Sample Output

```
🚀 Starting Automated Hardware Debugging Session
================================================================================
🔧 Injecting debugging loops into test_permute_5d_blocked in tests/ttnn/unit_tests/operations/test_permute.py
📦 Added missing imports: import os, import json
✅ Successfully injected debugging loops into test_permute_5d_blocked
🔍 Searching for program factory file...
✅ Found program factory: ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_rm_program_factory.cpp
🔧 Injecting program factory modifications into ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_rm_program_factory.cpp
✅ Successfully injected program factory modifications
🔍 Searching for compute kernel file...
✅ Found compute kernel: ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp
🔧 Injecting compute kernel modifications into ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp
✅ Successfully injected compute kernel modifications

🔨 Building project with modified files...
⚙️ Setting environment and building project...
✅ Project built successfully

🧪 Running modified test...
Shape:  (3, 65, 3, 3, 65) Perm:  (4, 0, 3, 2, 1) Memory config:  DRAM_MEMORY_CONFIG Dtype:  ttnn.bfloat16
RISCV  0
NOP TYPE  UNOPS
Nops 47: 7/10 failures (70.00%)
...

🔄 Restoring all modified files...
✅ Restored tests/ttnn/unit_tests/operations/test_permute.py
✅ Restored ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_rm_program_factory.cpp
✅ Restored ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp
🧹 Cleaning up backup files...
🗑️ Removed backup: test_permute.py.backup
🗑️ Removed backup: permute_rm_program_factory.cpp.backup
🗑️ Removed backup: transpose_xw_rm_single_tile_size.cpp.backup
🗑️ Removed empty backup directory: debug_backups
🔄 All files restored and backups cleaned up

================================================================================
DEBUGGING SESSION RESULTS
================================================================================
✅ Debugging session completed successfully!

🎯 OPTIMAL DEBUGGING CONFIGURATIONS:

Configuration 1:
  Shape: (3, 65, 3, 3, 65)
  Permutation: (4, 0, 3, 2, 1)
  Memory Config: DRAM_MEMORY_CONFIG
  Data Type: ttnn.bfloat16
  Nop Type: MNOPS
  RISC Mode: 1
  Nop Count: 47
  Failure Rate: 70.00%

📁 Modified files: 3
  • tests/ttnn/unit_tests/operations/test_permute.py
  • ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_rm_program_factory.cpp
  • ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp

🔄 All files have been restored to their original state.
```

## 🗂️ File Structure

```
.
├── automated_hardware_debugger.py     # Main standalone tool
├── debug_backups/                     # Backup directory (auto-created, auto-removed)
│   ├── test_permute.py.backup         # Temporary backups (auto-cleaned)
│   ├── permute_rm_program_factory.cpp.backup
│   └── transpose_xw_rm_single_tile_size.cpp.backup
└── debug_results_*.json               # Generated result files (auto-cleaned)
```

**Note**: The `debug_backups/` directory and all backup files are **automatically removed** after successful file restoration, leaving no traces behind.

## 🔍 Technical Details

### File Discovery Algorithm

The tool automatically discovers relevant files using these patterns:

**Program Factory:**
- `**/permute*program_factory*.cpp`
- `**/permute*factory*.cpp`
- `**/permute_rm_program_factory.cpp`

**Compute Kernel:**
- `**/kernels/compute/*transpose*.cpp`
- `**/kernels/compute/*permute*.cpp`
- `**/transpose_xw_rm_single_tile_size.cpp`

### Code Injection Strategy

1. **AST Parsing**: Uses Python AST to safely parse and modify test functions
2. **Regex Patterns**: Uses regex to find and modify C++ code structures
3. **Backup & Restore**: Creates backups before modification and restores after completion
4. **Environment Variables**: Uses environment variables to pass debugging parameters

### Safety Features

- ✅ **Automatic Backup**: All files backed up before modification
- ✅ **Automatic Restore**: All files restored after completion (even on failure)
- ✅ **Automatic Cleanup**: Backup files and directories automatically removed after successful restoration
- ✅ **Smart Import Injection**: Automatically adds required imports (`os`, `json`) and includes (`<cstdlib>`, `<string>`, `<map>`) to prevent compilation errors
- ✅ **Real-time Output**: Build and test output streams live to terminal during execution
- ✅ **Error Handling**: Graceful error handling with detailed messages
- ✅ **Timeout Protection**: Build has 10-minute timeout, test execution has 5-minute timeout
- ✅ **Cleanup**: Temporary result files automatically cleaned up

## 🛠️ Usage Examples

### Example 1: Basic Debugging
```bash
# Debug the permute 5D blocked function
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \
    --function test_permute_5d_blocked
```

### Example 2: Custom Parameters
```bash
# Test fewer NOPs and iterations for faster results
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \
    --function test_permute_5d_blocked \
    --max-nops 50 \
    --iterations 5
```

### Example 3: Skip Build (Fast Testing)
```bash
# Skip the build step for faster testing (C++ changes won't take effect)
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \
    --function test_permute_5d_blocked \
    --skip-build
```

### Example 4: Different Operation
```bash
# Debug a different operation (tool will find relevant files automatically)
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_conv2d.py \
    --function test_conv2d_basic
```

### Example 5: Custom Backup Location
```bash
# Use custom backup directory
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \
    --function test_permute_5d_blocked \
    --backup-dir /tmp/debug_backups
```

## 🔬 Integration with CI/CD

```yaml
- name: Run Hardware Debugging Analysis
  run: |
    python automated_hardware_debugger.py \
      --test-file tests/ttnn/unit_tests/operations/test_permute.py \
      --function test_permute_5d_blocked \
      --max-nops 25 \
      --iterations 3
```

## ⚠️ Important Notes

1. **Temporary Modifications**: All code changes are temporary and automatically restored
2. **File Discovery**: Tool automatically finds relevant program factory and kernel files
3. **Environment Variables**: Uses UNOPS, MNOPS, PNOPS, RISCV environment variables
4. **Test Requirements**: Target test function should use pytest parametrize decorators
5. **Backup Safety**: Always creates backups before making any modifications

## 🐛 Troubleshooting

### Tool Can't Find Program Factory
- Make sure you're running from the project root directory
- Check that the operation has a corresponding program factory file
- Manually specify files if needed (feature can be added)

### Test Function Not Found
- Verify the function name is spelled correctly
- Ensure the function exists in the specified file
- Check that the function uses the expected signature

### Compilation Errors
- The tool restores files automatically, even on errors
- Check the backup directory if restoration fails
- Manually restore from backups if needed

## 🤝 Contributing

To extend this tool for new operations:

1. **Add file discovery patterns** in `find_program_factory_file()` and `find_compute_kernel_file()`
2. **Customize injection patterns** in the relevant injection methods
3. **Test with your specific operation** to ensure compatibility
4. **Submit improvements** via pull request

## 📈 Future Enhancements

- **Size Parameter Reduction**: Integrate with size parameter optimization
- **Multi-Device Support**: Support for multi-device debugging
- **GUI Interface**: Web-based interface for easier usage
- **Result Visualization**: Charts and graphs for debugging results
- **Configuration Templates**: Predefined configurations for common operations

---

**This tool automates the entire hardware debugging workflow, from code injection to result analysis, making hardware validation faster and more systematic.**
