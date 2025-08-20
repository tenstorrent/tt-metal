# Reproducibility Steps: Adding Tensor Information Logging to TTNN Add Operations

Here's a step-by-step guide to reproduce the changes made:

## 1. Initial Setup
- Working directory: `/localdev/ppopovic/tt-metal`
- Git repository with main branch
- Goal: Add tensor shape and memory config logging to `ttnn.add()` operations

## 2. File Modifications

### Step 2.1: Modify Test File
**File:** `tests/ttnn/unit_tests/operations/eltwise/test_add.py`
- Simplified the test file to contain only `test_add_scalar` function
- Removed extensive test cases, kept basic tensor + scalar test
- Added commented-out Python tensor info printing code for reference

### Step 2.2: Modify C++ Pybind File
**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary_pybind.cpp`

#### Added includes:
```cpp
#include <fstream>
#include <functional>
#include <string>
```

#### Added utility functions:
1. **`get_python_call_stack()`** - Returns filtered Python call stack as string
2. **`generate_hash()`** - Generates uint32_t hash from input string using `std::hash<std::string>`
3. **`write_operation_info_to_file()`** - Creates `op_<stack_id>.txt` files with formatted output

#### Modified two operation handlers:
1. **Tensor + Scalar operation** (around line 2090):
   - Captures callstack and tensor info
   - Writes to file with format: `op_<hash>.txt`
   - Prints: `"Operation dispatched with stack_id: <id> -> <filename>"`

2. **Tensor + Tensor operation** (around line 2137):
   - Same approach for tensor-tensor operations
   - Includes both input tensors' information

## 3. Build Process
```bash
./build_metal.sh
```
- Build completed successfully with clang-format and linting applied
- No compilation errors

## 4. File Output Format
Each operation creates a file named `op_<hash>.txt` containing:
```
****CALLSTACK****
[Filtered Python call stack]
**** END CALLSTACK ****
****ARGS****
input_tensor_a : shape = <shape> data_type = <dtype> memory_config = <config>
scalar = <value>  // for scalar operations
// or
input_tensor_b : shape = <shape> data_type = <dtype> memory_config = <config>  // for tensor operations
**** END ARGS ***
```

## 5. Key Features
- **Hash-based filenames**: Unique files for unique callstack+args combinations
- **Console output**: Only shows `"Operation dispatched with stack_id: <id> -> <filename>"`
- **Filtered callstack**: Excludes pytest, pybind, and internal frames
- **Tensor information**: Shows logical shape, data type, and memory configuration
- **File persistence**: Information stored in text files rather than console spam

## 6. Git Commit
```bash
git add tests/ttnn/unit_tests/operations/eltwise/test_add.py ttnn/cpp/ttnn/operations/eltwise/binary/binary_pybind.cpp
git commit -m "test_add and pybind"
```

## 7. Usage
When running `ttnn.add()` operations:
- Console shows: `Operation dispatched with stack_id: 1234567890 -> op_1234567890.txt`
- Detailed tensor and callstack info saved to the corresponding file
- Each unique operation context gets its own file for tracking

This approach provides clean console output while maintaining detailed operation logging for debugging and analysis purposes.

## 8. Technical Details

### Hash Function Implementation
- Uses `std::hash<std::string>` to generate hash from combined callstack and arguments
- Converts to `uint32_t` for shorter, cleaner filenames
- Ensures unique files for unique operation contexts

### Callstack Filtering
Filters out these frame types to show only user code:
- `/pytest` - pytest framework frames
- `/pluggy/` - pytest plugin system
- `/python3.10/site-packages/` - installed packages
- `ttnn/__init__.py` - TTNN initialization
- `decorators.py` - TTNN decorators

### Memory Configuration Details
Captures and logs:
- **Tensor shape**: `logical_shape()` method result
- **Data type**: `dtype()` method result
- **Memory config**: `memory_config()` method result
- **Scalar values**: For tensor+scalar operations

### File Management
- Files created in current working directory
- Format: `op_<uint32_hash>.txt`
- No file cleanup - accumulates for analysis
- Each unique operation gets separate file

## 9. Future Extensions
This framework can be extended to:
- Add timestamp information
- Include more tensor metadata (storage_type, layout, etc.)
- Support other TTNN operations beyond add
- Add file rotation or cleanup mechanisms
- Include performance timing information
