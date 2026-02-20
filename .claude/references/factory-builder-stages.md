# Factory Builder Stage Reference

**Prerequisite Reading**: `.claude/references/ttnn-cb-memory-fundamentals.md` - CB page concepts, sync rules

---

## Section Quick Reference

**Usage**: Grep for the pattern to find current line number, then `Read` with `offset`/`limit`.

| Need | Grep Pattern | ~Lines |
|------|--------------|--------|
| TensorAccessor code snippets | `### TensorAccessor Code Snippets` | 33 |
| Official patterns reference | `## Official TTNN Patterns Reference` | 127 |
| Stage 4 (device op, validation) | `## Stage 4: Device Operation` | 137 |
| Stage 5 (program factory, CBs) | `## Stage 5: Program Factory Structure` | 179 |
| Stage 6 (kernel compilation) | `## Stage 6: Kernel Compilation` | 127 |
| Empty kernel stub templates | `### Empty Kernel Stub Templates` | 63 |
| Debugging guidance | `## Debugging` | 18 |
| Execution logging template | `## Execution Logging (Optional)` | 191 |

---

### TensorAccessor Code Snippets

**Host-side setup** (in program factory):
```cpp
#include <tt-metalium/tensor_accessor_args.hpp>

// Build compile-time args with TensorAccessorArgs
std::vector<uint32_t> reader_compile_time_args = {cb_id};
TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

// Alternative: get args directly
const auto accessor_args = TensorAccessorArgs(buffer);
auto compile_args = accessor_args.get_compile_time_args();
// For runtime args (if using RuntimeX flags):
SetCommonRuntimeArgs(program, kernel_id, accessor_args.get_common_runtime_args());
```

**Device-side usage** (in kernel):
```cpp
#include "api/dataflow/dataflow_api.h"  // TensorAccessor included automatically

// Create from compile-time args starting at index 1
constexpr auto src_tensor_args = TensorAccessorArgs<1>();
const auto s = TensorAccessor(src_tensor_args, base_addr, page_size);

// Use get_noc_addr for address calculation
noc_async_read(s.get_noc_addr(page_id), l1_write_addr, page_size);
// Or use helper functions:
noc_async_read_page(page_id, s, l1_write_addr);
```

---

## Official TTNN Patterns Reference

These patterns show the official code structures used throughout TTNN operations. Use these as templates when implementing your program factory.

### Program Factory Structure

**Shared Variables** (in `device/{operation_name}_program_factory.hpp`):
```cpp
namespace ttnn::operations::{operation_name}::detail {

struct {OperationName}SharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle compute_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> cores;
};

ttnn::device_operation::CachedProgram<{OperationName}SharedVariables> {operation_name}_single_core(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    /* operation-specific params */);

}  // namespace ttnn::operations::{operation_name}::detail
```

**Program Factory** (in `device/{operation_name}_op.hpp`):
```cpp
struct {OperationName}DeviceOperation {
    struct ProgramFactory {
        using shared_variables_t = detail::{OperationName}SharedVariables;
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const {OperationName}Params& operation_attributes,
            const {OperationName}Inputs& tensor_args,
            Tensor& output);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const {OperationName}Params& operation_attributes,
            const {OperationName}Inputs& tensor_args,
            Tensor& output);
    };
    using program_factory_t = std::variant<ProgramFactory>;
};
```

### Work Distribution Pattern

```cpp
// Use split_work_to_cores for even distribution
#include <tt-metalium/work_split.hpp>

auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
uint32_t num_cores_y = compute_with_storage_grid_size.y;
auto [num_cores, all_cores, core_group_1, core_group_2,
      num_work_per_core_group_1, num_work_per_core_group_2] =
    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_work_units);
```

### Bfloat16 Packing Pattern

When you need to pass bfloat16 scalar values as compile-time args (e.g., for scalers, epsilon):

```cpp
#include <tt-metalium/bfloat16.hpp>

// Pack two bfloat16 values into a single uint32 for compile-time args
const float scaler_value = 1.0f / static_cast<float>(width);
const bfloat16 bfloat_scaler(scaler_value);
const uint32_t packed_scaler = pack_two_bfloat16_into_uint32({bfloat_scaler, bfloat_scaler});

// Use in compile-time args
std::vector<uint32_t> compute_compile_args = {
    Wt,
    packed_scaler,  // Packed bfloat16 value
};
```

**Note**: Do NOT use `bfloat16().to_packed()` or `bfloat16().to_uint32()` - these methods don't exist. Always use `pack_two_bfloat16_into_uint32()`.

### Circular Buffer Pattern

```cpp
// Standard CB indices
uint32_t cb_input_idx = tt::CBIndex::c_0;    // Input CB
uint32_t cb_output_idx = tt::CBIndex::c_2;   // Output CB
// Additional CBs: c_1, c_3, c_4, etc.

// Double-buffered CB (2 tiles)
uint32_t num_tiles = 2;
tt::tt_metal::CircularBufferConfig cb_config =
    tt::tt_metal::CircularBufferConfig(
        num_tiles * single_tile_size, {{cb_idx, cb_data_format}})
        .set_page_size(cb_idx, single_tile_size);
tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
```

### Kernel Creation Pattern

```cpp
// Reader kernel (RISCV_0 / BRISC / NOC0)
tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/{category}/{operation}/device/kernels/dataflow/reader_{operation}.cpp",
    all_cores,
    tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

// Writer kernel (RISCV_1 / NCRISC / NOC1)
tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/{category}/{operation}/device/kernels/dataflow/writer_{operation}.cpp",
    all_cores,
    tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

// Compute kernel (optional)
tt::tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/{category}/{operation}/device/kernels/compute/{operation}_compute.cpp",
    all_cores,
    tt::tt_metal::ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .compile_args = compute_compile_time_args});
```

---

## Stage 4: Device Operation

### Goal
Complete device operation with proper validation and factory selection.

### Step 4.1: Write Test First (RED)

**Create test** `test_dev/test_stage4_device_op.py`:
```python
import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -r' to reset device (see CLAUDE.md)

def test_device_op_called(device):
    """Operation should reach program factory, not fail at validation"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exc:
        ttnn.{operation_name}(input_tensor{, required_params})

    # Error should be about program/kernel, not validation
    error_msg = str(exc.value).lower()
    assert "kernel" in error_msg or "program" in error_msg or "factory" in error_msg, \
        f"Expected program/kernel error, got: {exc.value}"

def test_program_factory_selected(device):
    """select_program_factory should return valid factory type"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Operation should not fail at factory selection
    with pytest.raises(RuntimeError) as exc:
        ttnn.{operation_name}(input_tensor{, required_params})

    # Should not mention "select" or "factory selection"
    assert "select" not in str(exc.value).lower()
```

**Run tests to confirm they fail:**
```bash
./build_metal.sh -b Debug && pytest test_dev/test_stage4_device_op.py -v
```

Expected: Tests fail because `select_program_factory` throws or returns wrong type.

### Step 4.2: Write Implementation (GREEN)

**Note**: The scaffolder already creates `device/{operation_name}_op.cpp` with stub implementations. You just need to verify or update as needed.

The scaffolder creates these implementations in `device/{operation_name}_op.cpp`:

```cpp
// select_program_factory (already created by scaffolder)
{OperationName}DeviceOperation::program_factory_t
{OperationName}DeviceOperation::select_program_factory(
    const {OperationName}Params& operation_attributes,
    const {OperationName}Inputs& tensor_args) {
    return ProgramFactory{};
}

// validate_on_program_cache_miss (already created by scaffolder)
void {OperationName}DeviceOperation::validate_on_program_cache_miss(
    const {OperationName}Params& attributes,
    const {OperationName}Inputs& tensor_args) {
    validate_inputs(attributes, tensor_args);
}

// validate_on_program_cache_hit (already created by scaffolder)
void {OperationName}DeviceOperation::validate_on_program_cache_hit(
    const {OperationName}Params& attributes,
    const {OperationName}Inputs& tensor_args) {
    validate_inputs(attributes, tensor_args);
}
```

The scaffolder also creates `ProgramFactory::create` stub that calls the program factory function:
```cpp
// ProgramFactory::create (already created by scaffolder - calls detail::{operation_name}_single_core)
{OperationName}DeviceOperation::ProgramFactory::cached_program_t
{OperationName}DeviceOperation::ProgramFactory::create(
    const {OperationName}Params& operation_attributes,
    const {OperationName}Inputs& tensor_args,
    Tensor& output) {
    return detail::{operation_name}_single_core(
        tensor_args.input,
        output,
        operation_attributes.param1);  // Pass operation-specific params
}

// override_runtime_arguments (already created by scaffolder)
void {OperationName}DeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const {OperationName}Params& operation_attributes,
    const {OperationName}Inputs& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    for (size_t i = 0; i < cores.size(); ++i) {
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
            runtime_args[0] = input_addr;
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
            runtime_args[0] = output_addr;
        }
    }
}
```

**Your main task in Stage 4-6**: Implement `detail::{operation_name}_single_core()` in `device/{operation_name}_program_factory.cpp`

### Step 4.3: Verify Tests Pass (GREEN)
```bash
./build_metal.sh -b Debug && pytest test_dev/test_stage4_device_op.py -v
```

**STOP. Do not proceed until Stage 4 tests pass.**

---

## Stage 5: Program Factory Structure

### Goal
Create program factory with circular buffers and work distribution. Kernels not yet created.

### Step 5.1: Write Test First (RED)

**Create test** `test_dev/test_stage5_program_factory.py`:
```python
import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.
# Before running: 'tt-smi -r' to reset device (see CLAUDE.md)

def test_program_factory_creates_cbs(device):
    """Program factory should create CBs before failing at kernel creation"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exc:
        ttnn.{operation_name}(input_tensor{, required_params})

    error_msg = str(exc.value).lower()
    # Should fail at kernel, not at CB or program
    assert "kernel" in error_msg, f"Expected kernel error, got: {exc.value}"
    assert "circular" not in error_msg, f"Should not fail at CB creation: {exc.value}"

def test_work_distribution(device):
    """Should handle various input sizes"""
    # Small input (1 tile)
    small_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Large input (many tiles)
    large_input = ttnn.from_torch(
        torch.randn(1, 32, 64, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for inp in [small_input, large_input]:
        with pytest.raises(RuntimeError) as exc:
            ttnn.{operation_name}(inp{, required_params})
        # Should reach kernel creation for all sizes
        assert "kernel" in str(exc.value).lower()
```

**Run tests to confirm they fail:**
```bash
./build_metal.sh -b Debug && pytest test_dev/test_stage5_program_factory.py -v
```

Expected: Tests fail because `detail::{operation_name}_single_core` throws "not yet implemented".

### Step 5.2: Write Implementation (GREEN)

**Update** `device/{operation_name}_program_factory.cpp` (the scaffolder created a stub that throws):

```cpp
#include "{operation_name}_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::{operation_name}::detail {

using namespace tt;
using namespace tt::tt_metal;

ttnn::device_operation::CachedProgram<{OperationName}SharedVariables> {operation_name}_single_core(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    float param1 /* Replace with actual params from spec */) {

    const auto& input = input_tensor;
    auto& output = output_tensor;

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    tt::tt_metal::Program program{};

    // Data formats
    tt::DataFormat input_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t input_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_tile_size = tt::tile_size(output_cb_data_format);

    // Work distribution
    // From spec's "Work Distribution" section - extract work unit definition
    uint32_t num_work_units = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2,
          num_work_per_core_group_1, num_work_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_work_units);

    // Create circular buffers from spec's "Circular Buffer Requirements" table
    // CB 0: Input (double-buffered, 2 tiles)
    uint32_t cb_input_idx = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;  // Double-buffered
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * input_tile_size, {{cb_input_idx, input_cb_data_format}})
            .set_page_size(cb_input_idx, input_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    // CB 2: Output (double-buffered, 2 tiles)
    uint32_t cb_output_idx = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;  // Double-buffered
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * output_tile_size, {{cb_output_idx, output_cb_data_format}})
            .set_page_size(cb_output_idx, output_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    // Add additional CBs from spec as needed
    // ...

    // Throw before kernel creation (Stage 5 boundary)
    TT_THROW("{operation_name}: Kernel creation not yet implemented");

    // The code below will be uncommented in Stage 6
    /*
    // Compile-time args for kernels
    std::vector<uint32_t> reader_compile_time_args = {cb_input_idx};
    // Add tensor accessor args as needed...

    std::vector<uint32_t> writer_compile_time_args = {cb_output_idx};
    // Add tensor accessor args as needed...

    // Create kernels
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(...);
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(...);
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(...);

    // Build cores vector for shared_variables
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores.push_back({i / num_cores_y, i % num_cores_y});
    }

    // Set runtime args
    ...

    return {
        std::move(program),
        {OperationName}SharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cores = std::move(cores)}};
    */
}

}  // namespace ttnn::operations::{operation_name}::detail
```

**Note**: The scaffolder already creates `device/{operation_name}_op.hpp` with the `ProgramFactory` struct and `shared_variables_t` type alias pointing to `detail::{OperationName}SharedVariables`. You don't need to modify the header - just implement the function in the `.cpp` file.

### Step 5.3: Verify Tests Pass (GREEN)
```bash
./build_metal.sh -b Debug && pytest test_dev/test_stage5_program_factory.py -v
```

**STOP. Do not proceed until Stage 5 tests pass.**

---

## Stage 6: Kernel Compilation

### Goal
Create **empty** stub kernels that compile at runtime. The kernels should immediately return without doing any work.

**CRITICAL**: Stage 6 stub kernels must be completely empty:
- Kernels should just return immediately (empty `kernel_main()` or `MAIN`)
- Do NOT implement any data movement, CB operations, or compute logic
- All kernel implementation is deferred to Stage 7 (kernel-writer agent)
- The only goal is to verify the program factory infrastructure works (kernel paths, compile-time args, runtime args)

**Important**: Kernels are JIT-compiled at runtime. The only way to verify kernel compilation is to run the operation and check for compilation errors in the output.

### Step 6.1: Write Test First (RED)

**Create test** `test_dev/test_stage6_kernel_compilation.py`:
```python
import pytest
import torch
import ttnn

# NOTE: Use the built-in `device` fixture from conftest.py - do NOT define your own.

def test_kernels_compile_at_runtime(device):
    """Kernels should compile without errors when operation runs"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run the operation - kernel compilation happens here
    # Empty stub kernels will produce garbage output, but should not error
    try:
        result = ttnn.{operation_name}(input_tensor{, required_params})
    except RuntimeError as e:
        error_str = str(e)
        if ".cpp" in error_str or "error:" in error_str.lower():
            pytest.fail(f"Kernel compilation failed: {e}")
        raise

def test_program_executes_without_hang(device):
    """Program should execute without hanging (empty stubs return immediately)"""
    input_tensor = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Empty stub kernels return immediately - no hang possible
    result = ttnn.{operation_name}(input_tensor{, required_params})
    assert result is not None
```

### Step 6.2: Create Kernel Directory Structure

```
device/kernels/
├── dataflow/
│   ├── reader_{operation_name}.cpp
│   └── writer_{operation_name}.cpp
└── compute/
    └── {operation_name}_compute.cpp  (if needed)
```

### Empty Kernel Stub Templates

**Reader kernel stub** `device/kernels/dataflow/reader_{operation_name}.cpp`:
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Empty stub - implementation in Stage 7
}
```

**Writer kernel stub** `device/kernels/dataflow/writer_{operation_name}.cpp`:
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Empty stub - implementation in Stage 7
}
```

**Compute kernel stub** `device/kernels/compute/{operation_name}_compute.cpp`:
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"

namespace NAMESPACE {

void MAIN {
    // Empty stub - implementation in Stage 7
}

}  // namespace NAMESPACE
```

### Step 6.3: Complete Program Factory

In `device/{operation_name}_program_factory.cpp`, remove the `TT_THROW` and add kernel creation.

The program factory should:
1. Create all circular buffers (from spec)
2. Create kernel handles with `CreateKernel()` pointing to the empty stub files
3. Set runtime args (even though empty kernels don't use them)
4. Return the cached program with shared variables

Empty kernels don't need compile-time args, but you should still pass them so the infrastructure is ready for Stage 7.

### Step 6.4: Verify Tests Pass (GREEN)
```bash
./build_metal.sh -b Debug && pytest test_dev/test_stage6_kernel_compilation.py -v
```

If kernel compilation fails at runtime, check the error output for:
- Source file paths (indicates which kernel failed)
- Syntax errors or missing includes

## Debugging

### Build Errors (Host-side C++)
- Missing includes: Add required headers
- Kernel path not found: Verify kernel path string in CreateKernel

### Runtime Kernel Compilation Errors
Kernel compilation happens at runtime. If the operation fails with a compilation error:
- Check the error message for the kernel source file path
- Look for syntax errors or missing includes in that kernel

### Test Failures
- Stage 4: Check select_program_factory returns valid type
- Stage 5: Check CB creation doesn't throw, work distribution is correct
- Stage 6: Check kernel paths exist and kernels have valid syntax

---

## Execution Logging (Optional)

If the caller includes **"enable detailed logging"** or **"with execution log"** in the prompt, you MUST create a detailed execution log file.

### Log File Location
`{operation_name}_factory_builder_execution_log.md` in the operation directory.

### Log Format
```markdown
# Execution Log: {Operation Name} Factory Building

## Session Info
- **Started**: {timestamp or "session start"}
- **Operation**: {operation_name}
- **Spec Path**: {path to spec file}
- **Operation Path**: {target directory}

## Execution Timeline

### Step 1: {Description}
**Action**: {What you did - e.g., "Read spec CB requirements", "Create reader kernel"}
**Command/Tool**: {Tool used and parameters}
**Result**:
```
{Full output - especially important for builds and tests}
```
**Decision**: {What you decided based on this result}

### Step 2: {Description}
...

## Spec Extraction
| Section | Extracted Value | Used For |
|---------|-----------------|----------|
| Circular Buffers | {CB config} | CreateCircularBuffer calls |
| Work Distribution | {strategy} | split_work_to_cores setup |
| Data Flow | {pattern} | Kernel responsibilities |
| Memory Access | {patterns} | Reader/Writer implementation |

## Stage 4: Device Operation
### Files Modified
| File | Change | Reason |
|------|--------|--------|
| {path} | {what changed} | {why} |

### Build Attempt
**Command**: `./build_metal.sh -b Debug`
**Result**: SUCCESS / FAILED
**Output** (if failed):
```
{error output}
```
**Fix Applied**: {if any}

### Test Run
**Command**: `pytest test_dev/test_stage4_device_op.py -v`
**Result**: PASS / FAIL
**Output**:
```
{test output}
```
**Issues**: {if any}

## Stage 5: Program Factory
### CB Configuration Applied
| CB ID | Name | Size | Data Format | Reasoning |
|-------|------|------|-------------|-----------|
| c_0 | cb_input | 2 tiles | input dtype | Double-buffered for... |
| c_2 | cb_output | 2 tiles | output dtype | Double-buffered for... |

### Work Distribution Applied
- **Total work units**: {calculation}
- **Grid size**: {x} x {y}
- **Cores used**: {num_cores}
- **Work per core (group 1)**: {n}
- **Work per core (group 2)**: {n}

### Files Modified
| File | Change | Reason |
|------|--------|--------|
| {path} | {what changed} | {why} |

### Build Attempt
...

### Test Run
**Command**: `pytest test_dev/test_stage5_program_factory.py -v`
...

## Stage 6: Kernel Compilation
### Kernels Created
| Kernel | Type | Path | Compile-Time Args | Runtime Args |
|--------|------|------|-------------------|--------------|
| reader | DataMovement | {path} | {args} | {args} |
| writer | DataMovement | {path} | {args} | {args} |
| compute | Compute | {path} | {args} | N/A |

### Kernel Code Summary
#### Reader Kernel
```cpp
// Key logic summary
{brief code summary}
```

#### Writer Kernel
```cpp
// Key logic summary
{brief code summary}
```

#### Compute Kernel (if applicable)
```cpp
// Key logic summary
{brief code summary}
```

### Build Attempt
...

### Test Run (Kernel JIT Compilation)
**Command**: `pytest test_dev/test_stage6_kernel_compilation.py -v`
**Result**: PASS / FAIL
**Output**:
```
{test output - kernel compilation happens here}
```

### Kernel Compilation Errors (if any)
| Kernel | Error | Line | Fix Applied |
|--------|-------|------|-------------|
| {kernel} | {error message} | {line #} | {fix} |

## Build Attempts Summary
| Stage | Attempt | Result | Duration | Key Error (if failed) |
|-------|---------|--------|----------|----------------------|
| 4 | 1 | PASS/FAIL | {time} | {error} |
| 5 | 1 | PASS/FAIL | {time} | {error} |
| 5 | 2 | PASS/FAIL | {time} | {error} |
| 6 | 1 | PASS/FAIL | {time} | {error} |

## Test Results Summary
| Test File | Tests Run | Passed | Failed | Errors |
|-----------|-----------|--------|--------|--------|
| test_stage4_device_op.py | {n} | {n} | {n} | {n} |
| test_stage5_program_factory.py | {n} | {n} | {n} | {n} |
| test_stage6_kernel_compilation.py | {n} | {n} | {n} | {n} |

## Errors Encountered
| Error | Stage | Context | Resolution |
|-------|-------|---------|------------|
| {error message} | 4/5/6 | {what caused it} | {how fixed} |

## Key Decisions
| Decision | Options | Choice | Rationale |
|----------|---------|--------|-----------|
| {topic} | {options} | {choice} | {why} |

## Deviations from Spec
| Aspect | Spec Said | Actually Implemented | Reason |
|--------|-----------|---------------------|--------|
| {aspect} | {spec value} | {actual value} | {why deviated} |

## Final Status
- **Stage 4 (Device Operation)**: PASS/FAIL
- **Stage 5 (Program Factory)**: PASS/FAIL
- **Stage 6 (Kernel Compilation)**: PASS/FAIL
- **All Builds**: PASS/FAIL
- **All Tests**: PASS/FAIL
- **Output Files**: {list all created/modified files}
- **Issues**: {any unresolved issues}
- **Ready for kernel implementation**: Yes/No
```

### What to Log
1. **Every file created/modified** - path, what changed, why
2. **Every build attempt** - full command, output (especially errors), result
3. **Every test run** - command, full output, pass/fail status
4. **Every build/compilation error** - exact error message, file/line, fix applied
5. **Kernel compilation errors** - these happen at runtime during tests, capture them!
6. **CB configuration decisions** - what was configured and why
7. **Work distribution setup** - how work was divided across cores
8. **Deviations from spec** - any places where implementation differs from spec

### Logging Guidelines
- Log in real-time as you work, not retrospectively
- **ALWAYS capture full build and test output** - this is critical for debugging
- Kernel errors only appear during test runs (JIT compilation) - pay special attention
- Include exact error messages, not paraphrases
- Document the fix for each error before moving on
- If multiple build/test attempts, document what finally worked
- Be explicit about which stage each action belongs to
- Note any deviations from spec with clear rationale
