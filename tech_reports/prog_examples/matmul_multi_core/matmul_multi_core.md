# Matmul (Multi Core -  Basic)

We'll build a program that will perform matmul operations on two tensors with equal-size inner dimension on as many cores as possible on the accelerator chip. This example builds on top of the previous single core example.

In terms of API usage, there isn\'t much change. We will discuss the specific changes to:

-   Using different kernels with their different runtime arguments.
-   Programming in terms of looping over all cores we will use.

All important ways we use the API different are in the new `matmul_multi_core` function.

The full example program is in [tt_metal/programming_examples/matmul_multi_core/matmul_multi_core.cpp](../../../tt_metal/programming_examples/matmul_multi_core/matmul_multi_core.cpp)

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh
    ./build/programming_examples/matmul_multi_core
```
## Accessing all the cores

We first must get information on the layout of the entire chip. This means the grid of cores and its x/y dimensions in number of cores.

``` cpp
auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
uint32_t num_cores_x = compute_with_storage_grid_size.x;
uint32_t num_cores_y = compute_with_storage_grid_size.y;
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
auto [num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);
```

The reason why we may have two separate sets of cores and tile counts is because depending on the grid size, it may not be possible to evenly distribute tiles over the full set of cores. We may need another separate set of cores to perform the `spill-over` number of computations.

This means we will need to be careful when it comes programming each set of cores. We also need to account for the case where we can evenly distribute work, meaning the second set will be empty.

## Using different kernels for reader/writer

We use more complex reader/writer kernels to feed our compute engine.

These kernels will use banking and interleaving techniques to ensure consistent and performant dataflow operations. We must ensure that cores only receive and perform computations on their specific set of tiles, and nothing more or less.

``` cpp
auto reader_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
    all_cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

auto writer_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
    all_cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});
```

## Compute kernel args

We need to account for the fact that we may two separate groups of cores that require different arguments for tile count.

``` cpp
vector<uint32_t> compute_args_group_1 = {
    1, // B
    1, // Mt
    Kt, // Kt
    num_output_tiles_per_core_group_1 // Nt
}; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

auto matmul_multi_core_kernel_group_1_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
    core_group_1,
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1}
);

if (!core_group_2.ranges().empty()) {
    vector<uint32_t> compute_args_group_2 = {
        1, // B
        1, // Mt
        Kt, // Kt
        num_output_tiles_per_core_group_2 // Nt
    }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

    auto matmul_multi_core_kernel_group_2_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
        core_group_2,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2}
    );
}
```

## Reader/writer kernel runtime args

Here, we introduce the concept of looping over all cores to apply an API.

In this case, we must set runtime args for reader/writer kernels. Note that we also must take care to account for the split groups of cores and to use the appropriate tile count when assigning args.

``` cpp
for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){

    CoreCoord core = {i / num_cores_y, i % num_cores_y};

    uint32_t num_output_tiles_per_core;
    if (core_group_1.core_coord_in_core_ranges(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_1;
    } else if (core_group_2.core_coord_in_core_ranges(core)) {
        num_output_tiles_per_core = num_output_tiles_per_core_group_2;
    } else {
        TT_ASSERT(false, "Core not in specified core ranges");
    }

    tt_metal::SetRuntimeArgs(
        program, reader_id, core,
        {src0_addr,
        src1_addr,
        Mt,
        Kt,
        Nt,
        MtKt,
        KtNt,
        B,
        uint32_t(bcast_batch),
        num_tiles_written,
        num_output_tiles_per_core,
        MtNt }
    );
    tt_metal::SetRuntimeArgs(
        program,
        writer_id,
        core,
        {dst_addr,
        num_output_tiles_per_core,
        num_tiles_written }
    );
    num_tiles_written += num_output_tiles_per_core;
}
```

## Conclusion

Those are all the major changes that we made in order to upgrade our single core matmul example into one that will use as many cores as possible. To see a more complicated example using data reuse among these cores, please refer to the `Matmul multi-core data reuse` example.
