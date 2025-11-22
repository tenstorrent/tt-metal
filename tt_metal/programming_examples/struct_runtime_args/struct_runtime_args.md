# Struct-Based Runtime Arguments Example

This example demonstrates the new struct-based runtime arguments API for TT-Metalium kernels, which provides a type-safe and convenient way to pass complex structured data to device kernels.

## Overview

Instead of manually packing and unpacking runtime arguments as vectors of `uint32_t`, you can now define POD (Plain Old Data) structs containing:
- Nested structs
- `std::array`
- `std::pair`
- `std::tuple`
- Enums (including `enum class`)
- Basic types like `uint32_t`
- Any combination of the above

The API automatically handles the conversion between struct format and the underlying `uint32_t` vector representation.

## Key Features Demonstrated

### 1. Complex Struct Definition
Both host and device code define matching struct types:

```cpp
enum class OperationType : uint32_t {
    ADD = 0,
    MULTIPLY = 1,
    SUBTRACT = 2,
    DIVIDE = 3
};

struct NestedData {
    uint32_t id;
    uint32_t value;
    uint32_t multiplier;
};

struct RuntimeArgs {
    uint32_t core_id;
    std::array<uint32_t, 4> vector_data;
    std::pair<uint32_t, uint32_t> range;
    std::tuple<uint32_t, uint32_t, uint32_t> triple;
    NestedData nested;
    OperationType op_mode;
};
```

### 2. Setting Runtime Arguments (Host Side)

**Per-Core Arguments:**
```cpp
RuntimeArgs args = {};
args.core_id = 0;
args.vector_data = {10, 20, 30, 40};
args.range = {0, 100};
args.triple = std::make_tuple(111, 222, 333);
// ... set other fields ...

SetRuntimeArgs(program, kernel_id, core, args);
```

**Common Arguments (Shared by All Cores):**
```cpp
CommonRuntimeArgs common_args;
common_args.global_offset = 1000;
// ... set other fields ...

SetCommonRuntimeArgs(program, kernel_id, common_args);
```

**Multiple Cores at Once:**
```cpp
std::vector<CoreCoord> cores = {core0, core1};
std::vector<RuntimeArgs> args_vec = {args_core0, args_core1};
SetRuntimeArgs(program, kernel_id, cores, args_vec);
```

### 3. Getting Runtime Arguments (Device Side)

```cpp
auto& rt_args = get_runtime_arguments<RuntimeArgs>();
auto& common_args = get_common_runtime_arguments<CommonRuntimeArgs>();

// Access fields directly
uint32_t id = rt_args.core_id;
uint32_t first_elem = rt_args.vector_data[0];
uint32_t start = rt_args.range.first;
uint32_t triple_elem = std::get<0>(rt_args.triple);
OperationType mode = rt_args.op_mode;

// Use enum in switch statement
switch (rt_args.op_mode) {
    case OperationType::ADD:
        // Handle ADD
        break;
    case OperationType::MULTIPLY:
        // Handle MULTIPLY
        break;
    // ...
}
```

## Running the Example

1. Set the DPRINT environment variable to see kernel output:
   ```bash
   export TT_METAL_DPRINT_CORES=(0,0),(1,0)
   ```

2. Build and run:
   ```bash
   ./build/tt_metal/programming_examples/struct_runtime_args
   ```

## Expected Output

The program will:
1. Launch kernels on two cores: (0,0) and (1,0)
2. Each core receives unique runtime arguments via the struct API
3. Both cores share common runtime arguments
4. The kernels print all received arguments using DPRINT
5. Each kernel performs a computation using the structured data

Example output:
```
=== Core (0,0) ===
Runtime Args:
  core_id: 0
  vector_data: [10, 20, 30, 40]
  range: (0, 100)
  triple: (111, 222, 333)
  nested.id: 1
  nested.value: 100
  nested.multiplier: 2
Common Runtime Args:
  global_offset: 1000
  global_scale: 10
  constants: [42, 314, 271]
  shared_data.id: 999
  ...
Computed result: 1020

=== Core (1,0) ===
Runtime Args:
  core_id: 1
  vector_data: [50, 60, 70, 80]
  ...
```

## Benefits of Struct-Based API

1. **Type Safety**: Compile-time checking of argument types
2. **Readability**: Clear structure definition instead of magic indices
3. **Maintainability**: Easy to add/remove fields without updating indices
4. **Flexibility**: Supports complex nested data structures
5. **Performance**: Zero overhead - uses `reinterpret_cast` internally

## Implementation Details

- Structs must be POD (Plain Old Data) types
- Struct size is automatically padded to 4-byte alignment
- The API uses `memcpy` and zero-initialization for safe memory handling
- Both old (`get_arg_val`) and new (`get_runtime_arguments`) APIs can coexist
