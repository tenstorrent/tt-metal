# Verify Device Operation Hash

Verify that a device operation's `compute_program_hash` correctly includes all values that affect program structure and excludes runtime-only values.

## Usage

When you need to verify a device operation's hash implementation, use this command and provide:
- The operation name you're verifying (e.g., 'Dropout', 'Conv3d', 'Binary')
- The location of the device operation code

## Overview

The program hash is used by the program cache to determine if a cached program can be reused. The hash must include:
- **All values that affect selection of a program factory**
- **All values that affect program structure** (kernels, kernel groups, compile-time args, defines, CB configs, semaphores)
- **Program factory variant index** (when multiple factories exist)

The hash must exclude:
- **Values that don't affect program structure** (buffer address, offsets, number of tiles to process, etc)

## Quick Decision: Do You Need a Custom Hash?

**Default Implementation Available:**
The infrastructure provides a default `compute_program_hash` that automatically hashes:
- The device operation type
- All of `operation_attributes_t`
- All of `tensor_args_t`

**You can skip implementing `compute_program_hash` if:**
- ✅ All members of `operation_attributes_t` affect program structure (or program factory selection)
- ✅ All relevant properties of `tensor_args_t` affect program structure
- ✅ Single program factory (no factory variant index needed)
- ✅ No runtime-only values in `operation_attributes_t` that don't affect program structure

**You must implement custom `compute_program_hash` if:**
- ❌ You have runtime-only values in `operation_attributes_t` (e.g., `seed` that only affects runtime args)
- ❌ You have multiple program factories (need to include `program_factory.index()`)
- ❌ You need to exclude specific tensor properties that don't affect program structure
- ❌ You need to include derived values not directly in `operation_attributes_t` or `tensor_args_t`

## Analysis Process

### Step 1: Identify Program Factory(ies)

1. Check `select_program_factory` in the device operation to see if there are multiple factories
2. If single factory: analyze that one factory
3. If multiple factories: analyze each factory variant separately

**Example (Dropout):**
```cpp
program_factory_t select_program_factory(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (args.use_per_device_seed) {
        return program::DropoutMeshWorkloadFactory{};
    } else {
        return program::DropoutProgramFactory{};
    }
}
```
This has 2 factories, so analyze both.

### Step 2: Analyze Program Factory `create` Method

For each program factory, examine the `create` method and identify what affects program structure:

#### 2.1 Kernels

**What to look for:**
- Kernel paths (different kernels = different program structure)
- Kernel types (Reader, Writer, Compute)
- Kernel configurations (MathFidelity, fp32_dest_acc_en, math_approx_mode)

**What affects hash:**
- Kernel source file path
- Kernel type (ReaderDataMovementConfig, WriterDataMovementConfig, ComputeConfig)
- Math fidelity settings
- Any compile-time kernel configuration

**Example:**
```cpp
kernels.reader = create_reader_kernel(program, all_cores, reader_compile_args, kReaderKernelPath);
kernels.writer = create_writer_kernel(program, all_cores, writer_compile_args, kWriterKernelPath);
kernels.compute_group_1 = create_compute_kernel(program, core_group_1, compute_group_1_args, kComputeKernelPath, math_approx_mode);
```
- `kReaderKernelPath`, `kWriterKernelPath`, `kComputeKernelPath` → affects hash
- `math_approx_mode` → affects hash (part of ComputeConfig)

#### 2.2 Kernel Groups / Core Ranges

**What to look for:**
- CoreRangeSet assignments to different kernel groups
- Different core ranges for different kernels

**What affects hash:**
- CoreRangeSet structure (which cores run which kernels)
- Number of core groups
- Core group assignments

**Example:**
```cpp
auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
    split_work_to_cores(compute_with_storage_grid_size, num_tiles);
```
- `core_group_1`, `core_group_2` structure → affects hash
- `num_cores`, `num_cores_y` → may affect hash if they change core assignments

#### 2.3 Compile-Time Arguments

**What to look for:**
- Values passed to `CreateKernel` as compile-time args
- Values in `compile_time_args` vectors
- Values from `TensorAccessorArgs` (these are compile-time)

**What affects hash:**
- All compile-time arguments passed to kernels
- Tensor accessor args (buffer layout, strides, etc.)

**Example:**
```cpp
std::vector<uint32_t> compute_group_1_args = {
    num_tiles_per_core_group_1,  // per_core_block_cnt
    1,                           // per_core_block_size
    prob_int,                    // prob
    uscale                       // scale
};
```
- All values in `compute_group_1_args` → affect hash
- These come from `operation_attributes_t` (prob, scale) or `tensor_args_t` (num_tiles derived from input shape)

**Important:** If a value is used in compile-time args, it MUST be in the hash, even if it's also used in runtime args.

#### 2.4 Defines (Preprocessor Defines)

**What to look for:**
- `std::map<std::string, std::string> defines` passed to kernel creation
- Any preprocessor defines that change kernel compilation

**What affects hash:**
- All define keys and values

**Example:**
```cpp
std::map<std::string, std::string> defines;
defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

auto reader_kernels_id = CreateKernel(
    program,
    kernel_path,
    core_grid,
    tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));
```
- All define values → affect hash
- These typically come from `operation_attributes_t` or derived from `tensor_args_t`

#### 2.5 Circular Buffer (CB) Configs

**What to look for:**
- `CircularBufferConfig` creation
- CB indices, data formats, page sizes, number of tiles

**What affects hash:**
- CB index
- Data format
- Page size
- Number of tiles
- Core ranges for each CB

**Example:**
```cpp
CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
                                     .set_page_size(cb_index, single_tile_size);
auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
```
- `cb_index`, `data_format`, `single_tile_size`, `num_tiles` → affect hash
- These come from tensor dtypes (affects data_format) and shapes (affects num_tiles)

#### 2.6 Semaphores

**What to look for:**
- `CreateSemaphore` calls
- Semaphore initialization values

**What affects hash:**
- Semaphore indices
- Initial values (if compile-time)
- Core ranges

**Note:** Most semaphores use runtime values, but if initialization is compile-time, it affects hash.

### Step 3: Trace Values Back to Source

For each value identified above, trace it back to:
- `operation_attributes_t` members
- `tensor_args_t` members (or properties derived from tensors like shape, dtype, memory_config)

**Example (Dropout):**
- `prob_int` (compile-time arg) → comes from `args.prob` (operation_attributes_t)
- `uscale` (compile-time arg) → comes from `args.scale` (operation_attributes_t)
- `num_tiles` (CB config) → derived from `input.physical_volume()` (tensor_args_t)
- `data_fmt_in` (CB config) → derived from `input.dtype()` (tensor_args_t)
- `core_group_1`, `core_group_2` → derived from `num_tiles` and `compute_with_storage_grid_size` (tensor_args_t)

### Step 4: Decide if Custom `compute_program_hash` is Needed

**Default Implementation:**
The infrastructure provides a default `compute_program_hash` that hashes:
- The device operation type
- All of `operation_attributes_t`
- All of `tensor_args_t`

**When to Use Default (No Custom Implementation Needed):**
✅ Use the default if:
- All members of `operation_attributes_t` affect program structure (or program factory selection)
- All relevant properties of `tensor_args_t` affect program structure
- No runtime-only values in `operation_attributes_t` that don't affect program structure
- Single program factory (no factory variant index needed)

**When to Implement Custom `compute_program_hash`:**
❌ Implement custom hash if:
- You have runtime-only values in `operation_attributes_t` that don't affect program structure (e.g., `seed` in Dropout)
- You have multiple program factories and need to include `program_factory.index()`
- You need to exclude specific tensor properties that don't affect program structure
- You need to include derived values not directly in `operation_attributes_t` or `tensor_args_t`

### Step 5: Check Custom `compute_program_hash` Implementation (If Needed)

If you need a custom implementation, verify that it includes:
1. ✅ Program factory variant index (if multiple factories)
2. ✅ All `operation_attributes_t` members that affect program structure
3. ✅ All `tensor_args_t` properties that affect program structure
4. ✅ Any derived values that affect program structure

Verify that it excludes:
1. ❌ Runtime-only values (buffer addresses, offsets)
2. ❌ Values only used in `override_runtime_arguments` (unless also used in compile-time args)
3. ❌ Values that don't affect program structure

**Example 1: Dropout (Custom Hash Needed)**

Dropout needs a custom hash because:
- `seed` in `operation_attributes_t` is runtime-only (only affects runtime args)
- Multiple program factories (needs `program_factory.index()`)

```cpp
tt::stl::hash::hash_t DropoutDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();
    auto args_without_seed = args;
    args_without_seed.seed = 0;  // ✅ Correct: seed only affects runtime args
    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<DropoutDeviceOperation>(
        args_without_seed,           // ✅ Includes prob, scale, output_dtype, output_memory_config
        program_factory.index(),     // ✅ Includes factory variant index
        input_tensor.dtype(),        // ✅ Affects CB data format
        input_tensor.memory_config(), // ✅ Affects program structure
        input_shape.volume());       // ✅ Affects num_tiles, core groups
    return hash;
}
```

**Analysis:**
- ✅ `program_factory.index()` - correct (2 factories)
- ✅ `args_without_seed` - includes prob, scale, output_dtype, output_memory_config (all affect program structure)
- ✅ `input_tensor.dtype()` - affects CB data format
- ✅ `input_tensor.memory_config()` - affects program structure
- ✅ `input_shape.volume()` - affects num_tiles and core group assignments
- ✅ `seed` excluded - correct (only used in runtime args)

**Example 2: Operation with Default Hash (No Custom Implementation)**

If an operation has:
- All `operation_attributes_t` members affect program structure
- All relevant `tensor_args_t` properties affect program structure
- Single program factory
- No runtime-only values in attributes

Then you can **omit** `compute_program_hash` entirely. The default implementation will hash all of `operation_attributes_t` and `tensor_args_t`, which is correct.

## Testing Your Hash Implementation

### Test Pattern 1: Cache Hit Test (Same Structure, Different Addresses)

Test that programs with the same structure but different buffer addresses reuse the cache:

```python
def test_operation_cache_address(device, ...):
    # Test that program cache updates the addresses of the inputs
    grid_size = device.compute_with_storage_grid_size()
    dummy = []
    for _ in range(3):
        # Create new tensors with different addresses but same structure
        dummy.append(ttnn.from_torch(torch.randn(input_shape), device=device, layout=ttnn.TILE_LAYOUT))
        run_operation_test(device, ...)

    # Should have only 1 cache entry (same program structure)
    assert device.num_program_cache_entries() == 1
```

### Test Pattern 2: Cache Miss Test (Different Structure)

Test that programs with different structure create separate cache entries:

```python
def test_operation_cache_hash(device, ...):
    # Test that program cache does not re-use the same program for different inputs
    grid_size = device.compute_with_storage_grid_size()
    dummy = []
    for i in range(2):
        # Change something that affects program structure
        new_shape = (input_shape[0], input_shape[1] * (i + 1), ...)
        dummy.append(ttnn.from_torch(torch.randn(new_shape), device=device, layout=ttnn.TILE_LAYOUT))
        run_operation_test(device, new_shape, ...)

    # Should have 2 cache entries (different program structures)
    assert device.num_program_cache_entries() == 2
```

### Test Pattern 3: Runtime-Only Changes (Cache Hit)

Test that changing runtime-only values (like seed) doesn't create new cache entries:

```python
def test_operation_cache_runtime_only(device, ...):
    # Test that changing runtime-only values doesn't create new cache entries
    for seed in [42, 123, 456]:
        run_operation_test(device, ..., seed=seed)

    # Should have only 1 cache entry (seed doesn't affect program structure)
    assert device.num_program_cache_entries() == 1
```

### Test Pattern 4: Compile-Time Changes (Cache Miss)

Test that changing compile-time values creates new cache entries:

```python
def test_operation_cache_compile_time(device, ...):
    # Test that changing compile-time values creates new cache entries
    for prob in [0.1, 0.5, 0.9]:
        run_operation_test(device, ..., prob=prob)

    # Should have 3 cache entries (prob affects compile-time args)
    assert device.num_program_cache_entries() == 3
```

## Checklist

Use this checklist to verify your hash implementation:

### For Single Program Factory:
- [ ] Identified all kernels and their configurations
- [ ] Identified all kernel groups/core ranges
- [ ] Identified all compile-time arguments
- [ ] Identified all defines
- [ ] Identified all CB configurations
- [ ] Identified all semaphores (if any)
- [ ] Traced all values back to `operation_attributes_t` or `tensor_args_t`
- [ ] **Decided if custom `compute_program_hash` is needed:**
  - [ ] All `operation_attributes_t` members affect program structure? → Can use default
  - [ ] All relevant `tensor_args_t` properties affect program structure? → Can use default
  - [ ] Any runtime-only values in `operation_attributes_t`? → Need custom hash
- [ ] If custom hash needed: Verified it includes all identified values
- [ ] If custom hash needed: Verified it excludes runtime-only values
- [ ] Written cache hit test (same structure, different addresses)
- [ ] Written cache miss test (different structure)
- [ ] Written runtime-only change test (should hit cache)
- [ ] Written compile-time change test (should miss cache)

### For Multiple Program Factories:
- [ ] Analyzed each program factory variant separately
- [ ] Verified `compute_program_hash` includes `program_factory.index()`
- [ ] Verified each factory's hash includes all its structure-affecting values
- [ ] Written tests for each factory variant
- [ ] Written test that verifies different factories create different cache entries

## Common Issues

### Issue 1: Missing Program Factory Index

**Problem:** When multiple factories exist, forgetting to include `program_factory.index()` in hash.

**Solution:** Always include `select_program_factory(...).index()` in the hash.

### Issue 2: Including Runtime-Only Values

**Problem:** Including values that only affect runtime arguments (like seed, buffer addresses).

**Solution:** Only include values used in compile-time args, kernel configs, CB configs, or defines.

### Issue 3: Excluding Compile-Time Values

**Problem:** Forgetting to include values that affect compile-time args but are also used in runtime args.

**Solution:** If a value is used in compile-time args, it MUST be in the hash, even if also used in runtime args.

### Issue 4: Missing Derived Values

**Problem:** Forgetting to include values derived from tensors that affect program structure (like num_tiles from shape).

**Solution:** Include all tensor properties that affect program structure: dtype, memory_config, shape (for num_tiles, core groups).

### Issue 5: Including Values That Don't Affect Structure

**Problem:** Including values that don't actually affect kernels, CBs, or program structure.

**Solution:** Only include values that directly affect: kernel paths/configs, compile-time args, defines, CB configs, core ranges, semaphores.

### Issue 6: Unnecessary Custom Hash Implementation

**Problem:** Implementing a custom `compute_program_hash` when the default would work.

**Solution:** If all `operation_attributes_t` members and all relevant `tensor_args_t` properties affect program structure, and you have a single program factory, you can omit `compute_program_hash` entirely. The default implementation will hash all attributes and tensor args, which is correct.

## Example Reference

**Operations with Custom Hash (Needed):**
- Dropout: Location `ttnn/cpp/ttnn/operations/experimental/dropout`
  - Hash implementation: `dropout_device_operation.cpp` (lines 115-130)
  - Why custom: Has `seed` (runtime-only) and multiple program factories
  - Program factory: `dropout_program_factory.cpp`
  - Tests: `tests/ttnn/unit_tests/operations/dropout/`

- PlusOne: Location `ttnn/cpp/ttnn/operations/experimental/plusone`
  - Hash implementation: `plusone_device_operation.cpp` (lines 35-56)
  - Why custom: Hashes specific tensor properties (dtype, memory_config, shape) rather than whole tensor

**Operations Using Default Hash:**
- Look for operations that don't declare `compute_program_hash` - they use the default implementation

**Test Examples:**
- Conv3d operation tests: Location `tests/ttnn/unit_tests/operations/conv/test_conv3d.py`
  - Cache address test: `test_conv3d_cache_address` (lines 199-208)
  - Cache hash test: `test_conv3d_cache_hash` (lines 216-228)

## Building and Testing

**Build:**
```bash
./build_metal.sh -c -e --debug
```

**Test:**
```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/<operation_name>/ -v
```

**Reset device if needed:**
```bash
tt-smi -r
```
