# Migrate Device Operation to TMP Pattern

Migrate a device operation from the old vector-based structure to the new TMP (Template Metaprogramming) device operation pattern that eliminates unnecessary heap allocations.

## Usage

When you need to migrate a device operation, use this command and provide:
- The operation name you're migrating (e.g., 'Embedding', 'Unary', 'Dropout')
- The location of the old device operation code

## Migration Guide

Follow the comprehensive guide at: `ttnn/cursor/DEVICE_OPERATION_MIGRATION_GUIDE.md`

The guide includes:
1. Detailed comparison of old vs new operation structures
2. Step-by-step migration process (8 main steps)
3. Complete 15-step checklist for verification
4. Common pitfalls to avoid
5. Reference examples (Dropout operation)
6. File structure guidance
7. Building and testing instructions

## Quick Reference - Migration Steps

**Step 1: Create `operation_attributes_t`**
- Extract all const member variables from old operation structure
- These represent operation configuration (not tensor arguments)

**Step 2: Create `tensor_args_t`**
- Use the operation's `invoke` method signature
- Include all Tensor parameters (required and optional)

**Step 3: Define Return Types**
- Prefer `spec_return_value_t` (TensorSpec) for newer operations
- Use `tensor_return_value_t` for the actual tensor return
- For multiple returns: use `std::vector` or `std::tuple`

**Step 4: Implement `compute_output_specs`**
- Required method that computes output tensor specifications

**Step 5: Implement `select_program_factory`**
- Returns `std::variant` of possible program factory types
- Selection logic based on tensor properties or operation attributes

**Step 6: Implement Validation**
- `validate_on_program_cache_miss` (required)
- `validate_on_program_cache_hit` (optional, can call cache_miss version)

**Step 7: Register Prim**
- Register in `ttnn::prim` namespace
- **Critical:** Always use the registered prim, never call `invoke` directly

**Step 8: Create Program Factory**
- Extract lambda captures → `shared_variables_t` struct
- Convert lambda body → `override_runtime_arguments` method
- Implement `create` method from old `create_program`

**Step 9: Implement `compute_program_hash` (if legacy had it)**
- Include: Circular Buffer setup, kernels, cores, compile-time args
- Exclude: Buffer addresses, offsets, runtime arguments
- **Important:** Include program factory variant index and all values that affect program structure

## Complete Checklist (15 Steps)

- [ ] **Step 1**: Created `operation_attributes_t` struct with all const configuration members
- [ ] **Step 2**: Created `tensor_args_t` struct with all Tensor parameters from invoke signature
- [ ] **Step 3**: Defined `tensor_return_value_t` and `spec_return_value_t` appropriately
- [ ] **Step 4**: Implemented `compute_output_specs`
- [ ] **Step 5**: [Optional] Implemented `create_output_tensors` (if legacy had it)
- [ ] **Step 6**: Implemented `select_program_factory` returning correct variant type
- [ ] **Step 7**: Implemented `validate_on_program_cache_miss`
- [ ] **Step 8**: [Optional] Implemented `validate_on_program_cache_hit` (if legacy had it)
- [ ] **Step 9**: Registered prim in `ttnn::prim` namespace
- [ ] **Step 10**: Updated all call sites to use prim instead of direct invoke or `operation::run`
- [ ] **Step 11**: Created program factory with:
  - [ ] `shared_variables_t` struct (from lambda captures)
  - [ ] `create` method (from old `create_program`)
  - [ ] `override_runtime_arguments` method (from lambda body)
- [ ] **Step 12**: [Optional] Implemented `compute_program_hash` (if legacy had it)
- [ ] **Step 13**: Removed old device operation code (after verification)
- [ ] **Step 14**: Relevant tests pass
- [ ] **Step 15**: Code compiles without warnings

## Common Pitfalls

1. **Forgetting to register the prim**: Always register in `ttnn::prim` namespace and use it instead of direct calls
2. **Including runtime-only values in hash**: Only hash compile-time constants that affect program structure
3. **Not including values that affect the program structure in hash**: Every parameter that has an effect on program structure must be taken into account in the hash

## Example Reference

See the Dropout operation migration:
- Location: `ttnn/cpp/ttnn/operations/experimental/dropout`
- PRs:
  - https://github.com/tenstorrent/tt-metal/pull/11793
  - https://github.com/tenstorrent/tt-metal/pull/11956

## File Structure

After migration, the operation should have this structure:

```
ttnn/cpp/ttnn/operations/<operation>/
├── device/
│   ├── <operation>_device_operation.hpp      # Main device operation struct
│   ├── <operation>_device_operation.cpp      # Implementation
│   ├── <operation>_device_operation_types.hpp # operation_attributes_t, tensor_args_t, return types
│   ├── <operation>_program_factory.hpp       # Program factory structs
│   ├── <operation>_program_factory.cpp       # Program factory implementation
│   └── kernels/                              # Kernel files (if any)
├── <operation>.hpp                           # Public API wrapper
├── <operation>.cpp                           # Public API implementation
└── <operation>_pybind.cpp                    # Python bindings (if any)
```

## Building and Testing

**Build:**
```bash
./build_metal.sh -c -e --debug --without-python-bindings
```

**Test:**
```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/<operation_name>/
```

**Reset device if needed:**
```bash
tt-smi -r
```
