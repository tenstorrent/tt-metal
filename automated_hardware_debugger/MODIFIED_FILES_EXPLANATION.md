# 📸 Modified Files Snapshot - Code Injection Analysis

## 🎯 Overview

This directory contains **exact copies** of the files that were modified during a debugging session, showing **precisely what code was injected and executed**. These snapshots are saved automatically before file restoration.

## 📁 Files in This Snapshot

### 1. **`modified_test_permute.py`**
**Original**: `tests/ttnn/unit_tests/operations/test_permute.py`

**What was injected**: Complete debugging loop system into `test_permute_5d_blocked` function:

```python
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    device.disable_and_clear_program_cache()
    print("Shape: ", shape, "Perm: ", perm, "Memory config: ", memory_config, "Dtype: ", dtype)
    nop_types_sentence = "UNOPS MNOPS PNOPS"
    nop_types = nop_types_sentence.split()

    torch.manual_seed(520)
    input_a = random_torch_tensor(dtype, shape)
    torch_output = torch.permute(input_a, perm)

    # ⬇️ INJECTED DEBUGGING CODE ⬇️
    min_config = {}
    all_results = []

    for is_risc in range(2):  # Test RISC-V modes 0 and 1
        print("RISCV ", is_risc)
        os.environ["RISCV"] = str(is_risc)
        for core_nop in nop_types:  # Test UNOPS, MNOPS, PNOPS
            print("NOP TYPE ", core_nop)
            my_it = 1  # Number of iterations (configurable)
            my_nop = 3  # Max NOPs to test (configurable)

            for nops in range(my_nop):
                os.environ[core_nop] = str(nops)
                counter = 0

                for i in range(my_it):
                    try:
                        # Original test logic with error handling
                        tt_input = ttnn.from_torch(...)
                        tt_output = ttnn.permute(tt_input, perm)
                        tt_output = ttnn.to_torch(tt_output)

                        if torch.equal(torch_output, tt_output):
                            counter = counter + 1
                        else:
                            # Failure detected - collect data
                            pass
                    except:
                        # Exception = failure
                        pass

                failures = my_it - counter
                if failures > 0:
                    # Store failure configuration for analysis
                    all_results.append({
                        'shape': shape, 'perm': perm,
                        'memory_config': str(memory_config),
                        'dtype': str(dtype),
                        'nop_type': core_nop,
                        'is_risc': is_risc,
                        'nop_count': nops,
                        'failures': failures,
                        'total_iterations': my_it
                    })

                print(f"Nops {nops}: {failures}/{my_it} failures")
```

### 2. **`modified_permute_rm_program_factory.cpp`**
**Original**: `ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_rm_program_factory.cpp`

**What was injected**: Environment variable handling and compute defines:

```cpp
// ⬇️ INJECTED CODE ⬇️
std::map<std::string, std::string> compute_defines;
compute_defines["UNOPS"] = std::to_string(std::getenv("UNOPS") ? std::stoi(std::getenv("UNOPS")) : 0);
compute_defines["MNOPS"] = std::to_string(std::getenv("MNOPS") ? std::stoi(std::getenv("MNOPS")) : 0);
compute_defines["PNOPS"] = std::to_string(std::getenv("PNOPS") ? std::stoi(std::getenv("PNOPS")) : 0);
compute_defines["RISCV"] = std::to_string(std::getenv("RISCV") ? std::stoi(std::getenv("RISCV")) : 0);

// Modified CreateKernel call to include defines
auto compute_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp",
    all_cores,
    tt::tt_metal::ComputeConfig{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .compile_args = compute_kernel_args,
        .defines = compute_defines  // ⬅️ INJECTED
    }
);
```

### 3. **`modified_transpose_xw_rm_single_tile_size.cpp`**
**Original**: `ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp`

**What was injected**: NOP functions and calls within the main computation loop:

```cpp
// ⬇️ INJECTED DEBUG INCLUDES ⬇️
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

// ⬇️ INJECTED NOP FUNCTIONS ⬇️
template <const int n, const int riscv>
inline void add_nops() {
    DPRINT << "RISCV " << riscv << " NOPS " << n << ENDL();

    for (int i = 0; i < n; i++) {
        if constexpr (riscv) {
            asm("nop");           // RISC-V NOP
        } else {
            TTI_NOP;              // Tensix NOP
        }
    }
}

template <const int U, const int M, const int P, const int R>
inline void add_trisc_nops() {
    DPRINT << "U " << (uint32_t)U << " M " << (uint32_t)M << " P " << (uint32_t)P << ENDL();
    if constexpr (U) {
        UNPACK((add_nops<U, R>()));   // UNPACK NOPs
    }
    if constexpr (M) {
        MATH((add_nops<M, R>()));     // MATH NOPs
    }
    if constexpr (P) {
        PACK((add_nops<P, R>()));     // PACK NOPs
    }
}

void MAIN {
    // ... original setup code ...

    for (uint32_t n = 0; n < num_blocks; n++) {
        // ⬇️ INJECTED NOP CALL ⬇️
        add_trisc_nops<UNOPS, MNOPS, PNOPS, RISCV>();

        // ... original computation logic continues ...
        tilize_init(cb_in, 1, cb_tilize);
        // ... etc
    }
}
```

## 🔧 How the Injection Works

### **1. Environment Variable Flow**:
```
Python Test → Sets os.environ["UNOPS"] = "47"
     ↓
Program Factory → Reads std::getenv("UNOPS") → Sets compute_defines["UNOPS"] = "47"
     ↓
Compute Kernel → Compiled with UNOPS=47 → add_trisc_nops<47, M, P, R>()
     ↓
Runtime → Executes 47 NOPs in UNPACK stage
```

### **2. Template Instantiation**:
- The kernel templates get **compile-time instantiated** with actual values
- `add_trisc_nops<47, 23, 15, 1>()` becomes specific code
- Different NOPs target different pipeline stages (UNPACK/MATH/PACK)

### **3. Debugging Loop Logic**:
- **Outer loops**: Test different RISC-V modes and NOP types
- **Middle loop**: Iterate through NOP counts (0, 1, 2, 3...)
- **Inner loop**: Run multiple iterations per configuration
- **Failure tracking**: Count and record when `torch.equal()` fails

## 🐛 Code Injection Quality

### **Clean, Single Injections** ✅:
- Each code element is injected **exactly once**
- **Duplicate prevention** checks ensure clean modifications
- **Debug includes**: Appear only once per file
- **NOP functions**: Injected only once per kernel
- **Environment variables**: Added only once per factory
- **Professional code quality** with no redundant injections

### **Template Compile-Time Values**:
- `UNOPS`, `MNOPS`, `PNOPS`, `RISCV` are **compile-time constants**
- Set via environment variables → program factory → kernel compilation
- Each test iteration compiles the kernel with different constant values

## 🎯 Key Insights

### **Why This Approach Works**:
1. **Environment variables** provide clean inter-process communication
2. **Template specialization** ensures compile-time optimization
3. **Multiple pipeline stages** can be tested independently
4. **Statistical approach** captures intermittent hardware issues
5. **Complete automation** removes manual debugging overhead

### **Hardware Debugging Power**:
- Can reproduce **race conditions** and **timing issues**
- Tests **different execution modes** (RISC-V vs Tensix)
- **Quantifies failure rates** instead of binary pass/fail
- **Pinpoints specific configurations** that trigger problems

---

**💡 This snapshot provides the exact view of what code was executed during the debugging session, enabling full reproducibility and deeper analysis of the injection process.**
