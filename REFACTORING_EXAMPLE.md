# Program Factory Refactoring Example

This document shows a concrete example of how to refactor a program factory from the old approach (direct `Program` creation) to the new approach (using `ProgramDescriptor`).

## Example: TypecastProgramFactory

### Before Refactoring (Old Way)

```cpp
// ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp

TypecastProgramFactory::cached_program_t TypecastProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

    // OLD: Direct Program creation
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;

    // OLD: Direct circular buffer creation
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // OLD: Direct kernel creation
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)num_tiles_per_core_group_1,
        (std::uint32_t)num_tiles_per_core_group_2
    };

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/typecast_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // ... more kernel creation ...

    return {std::move(program), shared_variables_t{reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}
```

### After Refactoring (New Way)

```cpp
// ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp

TypecastProgramFactory::cached_program_t TypecastProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

    // NEW: Create ProgramDescriptor instead of Program
    tt::tt_metal::ProgramDescriptor descriptor{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;

    // NEW: Add circular buffer to descriptor
    tt::tt_metal::CBDescriptor cb_src0_descriptor;
    cb_src0_descriptor.total_size = num_input_tiles * single_tile_size;
    cb_src0_descriptor.core_ranges = all_cores;
    cb_src0_descriptor.format_descriptors.push_back({
        .buffer_index = src0_cb_index,
        .data_format = cb_data_format,
        .page_size = single_tile_size
    });
    descriptor.cbs.push_back(std::move(cb_src0_descriptor));

    // NEW: Add kernel to descriptor
    tt::tt_metal::KernelDescriptor reader_kernel_descriptor;
    reader_kernel_descriptor.kernel_source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/typecast_reader.cpp";
    reader_kernel_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_descriptor.core_ranges = all_cores;
    reader_kernel_descriptor.compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)num_tiles_per_core_group_1,
        (std::uint32_t)num_tiles_per_core_group_2
    };
    reader_kernel_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};
    descriptor.kernels.push_back(std::move(reader_kernel_descriptor));

    // ... add more kernels to descriptor ...

    // NEW: Convert descriptor to program
    tt::tt_metal::Program program = tt::tt_metal::Program(descriptor);

    return {std::move(program), shared_variables_t{reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}
```

## Key Changes Summary

### 1. Program Creation
```cpp
// OLD
tt::tt_metal::Program program{};

// NEW
tt::tt_metal::ProgramDescriptor descriptor{};
// ... add components to descriptor ...
tt::tt_metal::Program program = tt::tt_metal::Program(descriptor);
```

### 2. Circular Buffer Creation
```cpp
// OLD
tt::tt_metal::CircularBufferConfig cb_config = ...;
tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);

// NEW
tt::tt_metal::CBDescriptor cb_descriptor;
cb_descriptor.total_size = size;
cb_descriptor.core_ranges = all_cores;
cb_descriptor.format_descriptors.push_back({...});
descriptor.cbs.push_back(std::move(cb_descriptor));
```

### 3. Kernel Creation
```cpp
// OLD
auto kernel_id = tt_metal::CreateKernel(program, kernel_path, cores, config);

// NEW
tt::tt_metal::KernelDescriptor kernel_descriptor;
kernel_descriptor.kernel_source = kernel_path;
kernel_descriptor.core_ranges = cores;
kernel_descriptor.compile_time_args = args;
kernel_descriptor.config = config_descriptor;
descriptor.kernels.push_back(std::move(kernel_descriptor));
```

### 4. Semaphore Creation (if needed)
```cpp
// OLD
tt::tt_metal::CreateSemaphore(program, core_ranges, initial_value);

// NEW
tt::tt_metal::SemaphoreDescriptor semaphore_descriptor;
semaphore_descriptor.core_ranges = core_ranges;
semaphore_descriptor.initial_value = initial_value;
descriptor.semaphores.push_back(std::move(semaphore_descriptor));
```

## Benefits of the New Approach

1. **Separation of Concerns**: Program description is separated from program creation
2. **Better Testing**: Can test program descriptions independently
3. **Serialization**: Program descriptions can be serialized/deserialized
4. **Validation**: Can validate program descriptions before creation
5. **Optimization**: Can optimize program descriptions before creating programs
6. **Debugging**: Easier to inspect and debug program structure

## Migration Checklist

For each program factory:

- [ ] Replace `Program program{};` with `ProgramDescriptor descriptor{};`
- [ ] Convert all `CreateCircularBuffer` calls to `CBDescriptor` objects
- [ ] Convert all `CreateKernel` calls to `KernelDescriptor` objects
- [ ] Convert all `CreateSemaphore` calls to `SemaphoreDescriptor` objects
- [ ] Add `Program program = Program(descriptor);` before return
- [ ] Update any runtime argument handling
- [ ] Test the refactored operation
- [ ] Update documentation if needed

## Common Pitfalls

1. **Missing Descriptor Fields**: Ensure all required fields are set in descriptors
2. **Runtime Arguments**: Runtime arguments still need to be set after program creation
3. **Shared Variables**: Kernel handles and other shared variables still need to be tracked
4. **Error Handling**: Ensure proper error handling for descriptor validation

## Testing Strategy

1. **Unit Tests**: Test that the refactored operation produces the same program structure
2. **Integration Tests**: Test that the operation works correctly end-to-end
3. **Performance Tests**: Ensure no performance regression
4. **Memory Tests**: Verify no memory leaks or issues
