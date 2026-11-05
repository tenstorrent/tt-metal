# Matmul (Multi Core -  Basic)

We'll build a program that will perform matmul operations on two tensors with equal-size inner dimension on as many cores as possible on the accelerator chip. This example builds on top of the previous single core example.

In terms of API usage, there isn\'t much change. We will discuss the specific changes to:

-   Using different kernels with their different runtime arguments.
-   Programming in terms of looping over all cores we will use.

All important ways we use the API different are in the new `matmul_multi_core` function.

The full example program is in [matmul_multi_core.cpp](../../../tt_metal/programming_examples/matmul_multi_core/matmul_multi_core.cpp)

To build and execute, you may use the following commands:
Then run the following:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_matmul_multi_core
```
## Mesh setup and accessing cores

We first must get information on the layout of the entire chip. This means the grid of cores and its x/y dimensions in number of cores.

``` cpp
constexpr int device_id = 0;
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

auto core_grid = mesh_device->compute_with_storage_grid_size();
auto num_output_tiles_total = (M * N) / TILE_HW;
```

## Splitting the work across cores

We need to split the work across multiple cores. This means figuring out how many tiles each core will use to compute the matmul.

We use a helper function that the Metal team developed to do this. Given the number of tiles of the multiplication and the entire grid upon which we want to execute, we receive back:

-   The total number of cores, which is the product of the dimensions
-   The `CoreRangeSet` of all cores
-   The first `CoreRangeSet` of cores which will perform a certain count of tile operations, A
-   The second `CoreRangeSet` of cores which will perform another coun of tile operations, B, if applicable
-   A as `uint32_t`
-   B as `uint32_t`, if applicable

``` cpp
auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
    split_work_to_cores(core_grid, num_output_tiles_total);
```

The reason why we may have two separate sets of cores and tile counts is because depending on the grid size, it may not be possible to evenly distribute tiles over the full set of cores. We may need another separate set of cores to perform the `spill-over` number of computations.

This means we will need to be careful when it comes programming each set of cores. We also need to account for the case where we can evenly distribute work, meaning the second set will be empty.

## Using different kernels for reader/writer

We use more complex reader/writer kernels to feed our compute engine.

These kernels will use banking and interleaving techniques to ensure consistent and performant dataflow operations. We must ensure that cores only receive and perform computations on their specific set of tiles, and nothing more or less.

``` cpp
tt_metal::experimental::KernelSpec reader_kernel_spec{
    .unique_id = reader_kernel,
    .source =
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
    .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
    .tensor_bindings =
        {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "a"},
         {.tensor_parameter_name = src1_tensor_param, .accessor_name = "b"}},
    .runtime_arg_schema =
        {.runtime_arg_names = {"Mt", "Kt", "Nt", "output_tile_start_id", "num_output_tiles"}},
    .hw_config =
        tt_metal::experimental::DataMovementHardwareConfig{
            .config_1xx =
                tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        },
};

tt_metal::experimental::KernelSpec writer_kernel_spec{
    .unique_id = writer_kernel,
    .source =
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
    .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
    .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "c"}},
    .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
    .hw_config =
        tt_metal::experimental::DataMovementHardwareConfig{
            .config_1xx =
                tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};
```

## Compute kernel args

We need to account for the fact that we may two separate groups of cores that require different arguments for tile count.
The compute kernel declares the tile count as a named runtime argument, so one `KernelSpec` serves both groups. All three kernels go into one work unit of the `ProgramSpec` that targets `all_cores`.

``` cpp
tt_metal::experimental::KernelSpec compute_kernel_spec{
    .unique_id = compute_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/compute/mm.cpp",
    .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
    .dfb_bindings =
        {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
         tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
         tt_metal::experimental::ProducerOf(output_dfb, "out0")},
    .runtime_arg_schema = {.runtime_arg_names = {"num_output_tiles", "Kt"}},
    .hw_config =
        tt_metal::experimental::ComputeHardwareConfig{
            .fpu_math_fidelity = math_fidelity,
            .sfpu_precision_mode = Precision::Precise,
        },
};

tt_metal::experimental::ProgramSpec program_spec{
    .name = "matmul_multi_core",
    .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
    .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
    .tensor_parameters =
        {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
         {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
         {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
    .work_units =
        {{.name = "matmul_multi_core",
          .kernels = {reader_kernel, writer_kernel, compute_kernel},
          .target_nodes = all_cores}},
};
Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

## Reader/writer kernel runtime args

Here, we introduce the concept of looping over all cores to apply an API.

In this case, we must set named runtime args per core for the reader, writer and compute kernels with `AddRuntimeArgsForNode`, then pass them in `ProgramRunArgs` to `SetProgramRunArgs`. Note that we also must take care to account for the split groups of cores and to use the appropriate tile count when assigning args.

``` cpp
tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel};
tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel};
tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel};
uint32_t work_offset = 0;
auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

// Iterate through each work group and assign work to cores
for (const auto& [ranges, work_per_core] : work_groups) {
    for (const auto& range : ranges.ranges()) {
        for (const auto& core : range) {
            // Set arguments for the reader kernel (data input)
            tt_metal::experimental::AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"Mt", Mt},                                // Number of tiles in M dimension
                 {"Kt", Kt},                                // Number of tiles in K dimension
                 {"Nt", Nt},                                // Number of tiles in N dimension
                 {"output_tile_start_id", work_offset},     // Starting offset for this core's work
                 {"num_output_tiles", work_per_core}});     // Amount of work for this core

            // Set arguments for the writer kernel (data output)
            tt_metal::experimental::AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_tiles", work_per_core}, {"start_id", work_offset}});

            // Set arguments for the compute kernel
            tt_metal::experimental::AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values,
                core,
                {{"num_output_tiles", work_per_core},  // Amount of work for this core
                 {"Kt", Kt}});                         // Number of tiles in K dimension for dot product
            work_offset += work_per_core;              // Update offset for next core
        }
    }
}

// The matrices A, B and C are passed to the kernels through the tensors bound to the tensor parameters.
tt_metal::experimental::ProgramRunArgs run_args{
    .kernel_run_args = {reader_run_args, writer_run_args, compute_run_args},
    .tensor_args =
        {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
         {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
         {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
};
tt_metal::experimental::SetProgramRunArgs(program, run_args);
```

## Conclusion

Those are all the major changes that we made in order to upgrade our single core matmul example into one that will use as many cores as possible. To see a more complicated example using data reuse among these cores, please refer to the `Matmul multi-core data reuse` example.
