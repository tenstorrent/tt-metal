# 🏆 Hackathon Tool: Automated Hardware Debugging Tool

## 🎯 **What We Built**

A **standalone command-line tool** that automatically injects debugging code into hardware test functions to find minimum failing configurations for efficient hardware debugging and validation.

## 🚀 **Key Features**

### ✅ **Fully Automated Code Injection**
- **Finds and modifies test functions** automatically
- **Discovers program factory files** and injects environment variable handling
- **Discovers compute kernel files** and injects NOP functions
- **Automatic project build** - rebuilds C++ code after modifications to ensure changes take effect
- **Smart import injection** - automatically adds required imports and includes to prevent compilation errors
- **Real-time output streaming** - build and test output displays live in terminal during execution
- **Temporary modifications** - all files are automatically restored after completion
- **Automatic cleanup** - backup files and directories are automatically removed, leaving no traces

### ✅ **Intelligent Debugging Strategy**
- Tests various **NOP configurations** (UNOPS, MNOPS, PNOPS)
- Tests different **RISC-V modes** (0 and 1)
- Iterates through **NOP counts** (0-100, configurable)
- Runs **multiple test iterations** per configuration
- **Counts failures** instead of stopping on first failure

### ✅ **Comprehensive Analysis**
- Identifies **optimal debugging parameters** with highest failure rates
- Generates **detailed reports** with actionable recommendations
- Saves **JSON results** for further analysis
- Provides **failure statistics** and success metrics

## 💻 **How to Use It**

### **Basic Command**
```bash
./automated_hardware_debugger.py --test-file tests/ttnn/unit_tests/operations/test_permute.py --function test_permute_5d_blocked
```

### **With Custom Parameters**
```bash
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \
    --function test_permute_5d_blocked \
    --max-nops 50 \
    --iterations 5
```

### **Command Line Options**
- `--test-file`: Path to the Python test file
- `--function`: Name of the test function to debug
- `--max-nops`: Maximum number of NOPs to test (default: 100)
- `--iterations`: Number of iterations per configuration (default: 10)
- `--backup-dir`: Directory to store backup files (default: ./debug_backups)

## 🔧 **What It Does Behind the Scenes**

### 1. **Test Function Transformation**
```python
# Original test function
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    input_a = random_torch_tensor(dtype, shape)
    tt_output = ttnn.permute(tt_input, perm)
    assert_equal(torch_output, tt_output)

# ↓ AUTOMATICALLY BECOMES ↓

# Injected debugging loops
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    for is_risc in range(2):
        os.environ["RISCV"] = str(is_risc)
        for core_nop in ["UNOPS", "MNOPS", "PNOPS"]:
            for nops in range(100):
                os.environ[core_nop] = str(nops)
                # Test multiple iterations and count failures
                failures = 0
                for i in range(10):
                    try:
                        tt_output = ttnn.permute(tt_input, perm)
                        if not torch.equal(torch_output, tt_output):
                            failures += 1
                    except:
                        failures += 1
```

### 2. **Program Factory Modification**
```cpp
// AUTOMATICALLY INJECTED:
std::map<std::string, std::string> compute_defines;
compute_defines["UNOPS"] = std::to_string(std::getenv("UNOPS") ? std::stoi(std::getenv("UNOPS")) : 0);
compute_defines["MNOPS"] = std::to_string(std::getenv("MNOPS") ? std::stoi(std::getenv("MNOPS")) : 0);
compute_defines["PNOPS"] = std::to_string(std::getenv("PNOPS") ? std::stoi(std::getenv("PNOPS")) : 0);
compute_defines["RISCV"] = std::to_string(std::getenv("RISCV") ? std::stoi(std::getenv("RISCV")) : 0);

tt::tt_metal::ComputeConfig{.defines = compute_defines}  // ← INJECTED
```

### 3. **Compute Kernel Modification**
```cpp
// AUTOMATICALLY INJECTED NOP FUNCTIONS:
template <const int U, const int M, const int P, const int R>
inline void add_trisc_nops() {
    if constexpr (U) UNPACK((add_nops<U, R>()));
    if constexpr (M) MATH((add_nops<M, R>()));
    if constexpr (P) PACK((add_nops<P, R>()));
}

void MAIN {
    for (uint32_t n = 0; n < num_blocks; n++) {
        add_trisc_nops<UNOPS, MNOPS, PNOPS, RISCV>();  // ← INJECTED
        // ... rest of kernel logic
    }
}
```

## 📊 **Sample Output**

```
🚀 Starting Automated Hardware Debugging Session
================================================================================
✅ Successfully injected debugging loops into test_permute_5d_blocked
✅ Found program factory: ttnn/cpp/ttnn/operations/.../permute_rm_program_factory.cpp
✅ Successfully injected program factory modifications
✅ Found compute kernel: ttnn/cpp/ttnn/operations/.../transpose_xw_rm_single_tile_size.cpp
✅ Successfully injected compute kernel modifications

🧪 Running modified test...
Shape: (3, 65, 3, 3, 65) Perm: (4, 0, 3, 2, 1) Memory config: DRAM_MEMORY_CONFIG
RISCV 0
NOP TYPE UNOPS
Nops 47: 7/10 failures (70.00%)
NOP TYPE MNOPS
Nops 23: 5/10 failures (50.00%)

🔄 Restoring all modified files...
✅ Restored tests/ttnn/unit_tests/operations/test_permute.py
✅ Restored ttnn/cpp/.../permute_rm_program_factory.cpp
✅ Restored ttnn/cpp/.../transpose_xw_rm_single_tile_size.cpp
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

🎯 OPTIMAL DEBUGGING CONFIGURATION:
  Shape: (3, 65, 3, 3, 65)
  Permutation: (4, 0, 3, 2, 1)
  Memory Config: DRAM_MEMORY_CONFIG
  Data Type: ttnn.bfloat16
  Nop Type: UNOPS
  RISC Mode: 0
  Nop Count: 47
  Failure Rate: 70.00%

📁 Modified files: 3 (all automatically restored)
```

## 🏆 **Perfect for Hackathons**

### **Why This Tool is Hackathon-Ready:**

1. **🚀 Zero Configuration** - Just point it at any test function and go!
2. **📱 Single Command** - One command does everything automatically
3. **🔒 Safe & Reversible** - All modifications are temporary and auto-restored
4. **🧹 Professional Cleanup** - Automatically removes all backup files and directories, leaving no traces
5. **📊 Actionable Results** - Provides specific debugging parameters to use
6. **🔧 Works on Any Test** - Automatically discovers and modifies relevant files
7. **⚡ Configurable Speed** - Adjust parameters for quick vs thorough analysis

### **Perfect Use Cases:**
- **Hardware validation teams** - Find minimal failing configs for simulator debugging
- **Performance optimization** - Identify specific NOP configurations causing issues
- **CI/CD integration** - Automated failure analysis in build pipelines
- **Bug reproduction** - Get exact parameters to reproduce hardware failures
- **Research & development** - Systematic hardware behavior characterization

## 🎁 **Deliverables**

### **Core Files:**
1. **`automated_hardware_debugger.py`** - Main standalone tool (850+ lines)
2. **`AUTOMATED_HARDWARE_DEBUGGER_README.md`** - Comprehensive documentation
3. **`example_usage_standalone.py`** - Usage examples and demonstrations

### **Features Implemented:**
- ✅ **AST-based Python code injection**
- ✅ **Regex-based C++ code modification**
- ✅ **Automatic file discovery algorithms**
- ✅ **Automatic project build system** - ensures C++ modifications take effect
- ✅ **Comprehensive backup & restore system**
- ✅ **Automatic backup cleanup** - leaves no traces behind
- ✅ **Statistical failure analysis**
- ✅ **JSON result export**
- ✅ **Command-line interface with argparse**
- ✅ **Error handling and timeout protection**
- ✅ **Extensible architecture for new operations**

## 🚀 **Ready to Demo!**

```bash
# Quick demo command
./automated_hardware_debugger.py \
    --test-file tests/ttnn/unit_tests/operations/test_permute.py \
    --function test_permute_5d_blocked \
    --max-nops 20 \
    --iterations 3

# Show usage examples
./example_usage_standalone.py

# Read comprehensive documentation
cat AUTOMATED_HARDWARE_DEBUGGER_README.md
```

---

**🏆 This tool transforms manual hardware debugging into an automated, systematic process - perfect for hackathon judges and real-world hardware validation teams!**
