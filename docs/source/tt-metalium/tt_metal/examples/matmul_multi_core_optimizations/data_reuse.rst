.. _MatMul_Multi_Core_Data_Reuse_example:

Data Reuse in `matmul_multicore_reuse`
======================================

Fine-Grained Block Size Control
-------------------------------

Advanced matrix dimension controls are found in the Programming Example's matmul_common directory, namely Block Matrix Multiply Ops (bmm_op.hpp). Including this header allows us advanced dynamic means of defining and retrieving matrix parameters. Our matmul kernels that work out-of-the-box perform on row-major and tile-major layouts, so you have the power to define your own outer-dimensional tile sizes, desired core grid dimensions, as well as your own input block width, all depending on your problem at hand.

In our reuse example, we can employ the ``get_large_matmul_params(...)`` function and pass our inputs as described above. By doing so, we let METALIUM's bmm op utility functions do the heavy lifting for us mathematically, and calculate our matmul's exact work-per-core size and work output size seamlessly. (You can consult the header for the prime factorization method used, plus many other details).

.. code-block:: cpp

    auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
    uint32_t per_core_M = std::get<0>(matmul_params);
    uint32_t per_core_N = std::get<1>(matmul_params);
    uint32_t out_subblock_h = std::get<2>(matmul_params);
    uint32_t out_subblock_w = std::get<3>(matmul_params);

Take note of the example's use of "subblocks" above. Recall that until now, we have optimized matmul by dividing matrices into blocks and subdivided those into tiles, which are laid out neatly on our compute cores. A key optimization here in `matmul_multicore_reuse` is the introduction of an intermediate subdivision of blocks, called subblocks. Below are some optimal subblock layouts already provided for you in the header, which run efficiently on our hardware.

.. code-block:: cpp

    constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
        {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
        {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
    }};

Intermediate Dataflow Buffer Configuration
------------------------------------------

In addition to our double-buffer config, we introduce a third dataflow buffer named ``interm0``. Like the input buffers, it is declared with a ``DataflowBufferSpec``. It shares its L1 memory with the ``output`` dataflow buffer: each of the two specs names the other in ``advanced_options.alias_with``. This buffer acts as a temporary storage for the intermediate results of matrix multiplication before they are combined into the final output.

.. code-block:: cpp

    // The output and the intermediate dataflow buffers share the same memory (aliased).
    tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = out_CB_tiles,
        .data_format_metadata = cb_data_format,
        .advanced_options = {.alias_with = {interm0_dfb}},
    };
    tt_metal::experimental::DataflowBufferSpec interm0_dfb_spec{
        .unique_id = interm0_dfb,
        .entry_size = single_tile_size,
        .num_entries = out_CB_tiles,
        .data_format_metadata = cb_data_format,
        .advanced_options = {.alias_with = {output_dfb}},
    };

Stride Kernel Arguments
-----------------------

The runtime arguments for the read, write, and compute kernels are set up in a certain way to employ data reuse through the intermediate dataflow buffer. This setup aligns with the execution model of the ``bmm_tile_layout.cpp`` reader and writer kernels, and ``bmm_large_block_zm.cpp`` compute kernel. Each kernel is a ``KernelSpec`` that declares its dataflow buffer bindings, its tensor bindings and the names of its runtime arguments. The compute kernel binds ``interm0`` as both producer and consumer.

.. code-block:: cpp

    /*
     * Create Kernels (Reader, Writer, Compute)
     */
    // Create reader and writer kernels per core
    tt_metal::experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp",
        .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"in0_tensor_start_tile_id",
                  "in0_tensor_stride_w",
                  "in0_tensor_stride_h",
                  "in0_tensor_next_block_stride",
                  "in0_block_w",
                  "in0_block_h",
                  "in0_block_num_tiles",
                  "in1_tensor_start_tile_id",
                  "in1_tensor_stride_w",
                  "in1_tensor_stride_h",
                  "in1_tensor_next_block_stride",
                  "in1_block_w",
                  "in1_block_h",
                  "in1_block_num_tiles",
                  "num_blocks",
                  "MtKt",
                  "KtNt",
                  "batch",
                  "bcast_B"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"out_tensor_start_tile_id",
                  "out_tensor_stride_w",
                  "out_tensor_stride_h",
                  "out_tensor_next_subblock_stride_w",
                  "out_tensor_next_subblock_stride_h",
                  "out_subblock_w",
                  "out_subblock_h",
                  "out_subblock_tile_count",
                  "out_num_subblocks_w",
                  "out_num_subblocks_h",
                  "MtNt",
                  "batch"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    // Create compute kernel. The compute kernel both produces and consumes the intermediate dataflow buffer.
    tt_metal::experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
             tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
             tt_metal::experimental::ProducerOf(output_dfb, "out0"),
             tt_metal::experimental::ProducerOf(interm0_dfb, "interm0"),
             tt_metal::experimental::ConsumerOf(interm0_dfb, "interm0")},
        .compile_time_args = compute_kernel_args,
        .hw_config =
            tt_metal::experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = math_fidelity,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

The ``ProgramSpec`` collects the kernels, dataflow buffers and tensor parameters, and one work unit runs all three kernels on ``all_cores``. ``MakeProgramFromSpec`` builds the ``Program`` from it.

.. code-block:: cpp

    tt_metal::experimental::ProgramSpec program_spec{
        .name = "matmul_multicore_reuse",
        .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec, interm0_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
        .work_units =
            {{.name = "matmul_multicore_reuse",
              .kernels = {reader_kernel, writer_kernel, compute_kernel},
              .target_nodes = all_cores}},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);


Recall our compile-time kernel compute args. Each one has a name, and the compute kernel reads it with ``get_arg(args::<name>)``:

.. code-block:: cpp

    tt_metal::experimental::KernelSpec::CompileTimeArgs compute_kernel_args = {
        {"in0_block_w", in0_block_w},
        {"in0_num_subblocks", in0_num_subblocks},
        {"in0_block_num_tiles", in0_block_num_tiles},
        {"in0_subblock_num_tiles", in0_subblock_num_tiles},

        {"in1_num_subblocks", in1_num_subblocks},
        {"in1_block_num_tiles", in1_block_num_tiles},
        {"in1_per_core_w", in1_per_core_w},

        {"num_blocks", num_blocks},

        {"out_subblock_h", out_subblock_h},
        {"out_subblock_w", out_subblock_w},
        {"out_subblock_num_tiles", out_subblock_num_tiles},
        {"batch", B}};

To properly run the reader and writer kernels, we must set up the runtime arguments with this information.  For each block of in0 and in1 matrices, we read the tiles pertaining to a certain subblock from DRAM into that core's L1, and we perform the bmm_large_block_zm on tiles therein using stride arguments.  Recall each tile is a member of a certain subblock, and subblocks are distributed across different cores in the core grid (specifically, in each core's L1). The writer kernel then stores the partial matmul results into its corresponding output subblock. ``AddRuntimeArgsForNode`` adds the named runtime args of one core to the kernel's ``KernelRunArgs``.

Reader:

.. code-block:: cpp

    tt_metal::experimental::AddRuntimeArgsForNode(
        reader_run_args.runtime_arg_values,
        core,
        {
            {"in0_tensor_start_tile_id", (std::uint32_t)Kt * per_core_M * output_idx_y},
            {"in0_tensor_stride_w", (std::uint32_t)1},
            {"in0_tensor_stride_h", (std::uint32_t)Kt},
            {"in0_tensor_next_block_stride", (std::uint32_t)in0_block_w},

            {"in0_block_w", (std::uint32_t)in0_block_w},
            {"in0_block_h", (std::uint32_t)per_core_M},
            {"in0_block_num_tiles", (std::uint32_t)in0_block_w * per_core_M},

            {"in1_tensor_start_tile_id", (std::uint32_t)per_core_N * output_idx_x},
            {"in1_tensor_stride_w", (std::uint32_t)1},
            {"in1_tensor_stride_h", (std::uint32_t)Nt},
            {"in1_tensor_next_block_stride", (std::uint32_t)in0_block_w * Nt},

            {"in1_block_w", (std::uint32_t)per_core_N},
            {"in1_block_h", (std::uint32_t)in0_block_w},
            {"in1_block_num_tiles", (std::uint32_t)per_core_N * in0_block_w},

            {"num_blocks", (std::uint32_t)Kt / in0_block_w},

            {"MtKt", (std::uint32_t)Mt * Kt},
            {"KtNt", (std::uint32_t)Kt * Nt},
            {"batch", (std::uint32_t)B},
            {"bcast_B", (std::uint32_t)bcast_batch},
        });

Writer:

.. code-block:: cpp

    tt_metal::experimental::AddRuntimeArgsForNode(
        writer_run_args.runtime_arg_values,
        core,
        {
            {"out_tensor_start_tile_id",
             ((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt)},
            {"out_tensor_stride_w", (std::uint32_t)1},
            {"out_tensor_stride_h", (std::uint32_t)Nt},
            {"out_tensor_next_subblock_stride_w", (std::uint32_t)out_subblock_w},
            {"out_tensor_next_subblock_stride_h", (std::uint32_t)out_subblock_h * Nt},

            {"out_subblock_w", (std::uint32_t)out_subblock_w},
            {"out_subblock_h", (std::uint32_t)out_subblock_h},
            {"out_subblock_tile_count", (std::uint32_t)(out_subblock_w * out_subblock_h)},
            {"out_num_subblocks_w", (std::uint32_t)(per_core_N / out_subblock_w)},
            {"out_num_subblocks_h", (std::uint32_t)(per_core_M / out_subblock_h)},

            {"MtNt", (std::uint32_t)Mt * Nt},
            {"batch", (std::uint32_t)B},
        });

The DRAM tensors are not runtime arguments. They are passed for the tensor parameters, together with the runtime args, through ``SetProgramRunArgs``:

.. code-block:: cpp

    // Set the runtime args, and the tensors bound to the tensor parameters.
    tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args = {reader_run_args, writer_run_args},
        .tensor_args =
            {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
    };
    tt_metal::experimental::SetProgramRunArgs(program, run_args);

Intermediate Results Handling
-----------------------------

In ``bmm_large_block_zm.cpp``,

a. **Preparing the Intermediate Buffer**:

    - **Reserving Partial Results Space**: For a given block (excluding the last block), we reserve space for intermediate (ie. partial) results in the rear of the intermediate dataflow buffer with ``reserve_back(...)``. Each consecutive subblock within this block will access this space, and contribute their partial results.

    .. code-block:: cpp

        dfb_interm0.reserve_back(out_subblock_num_tiles);

    - **Storing Partial Results**: Partial results are stored via a packing mechanism with ``pack_tile(...)`` into the above reserved space.

    .. code-block:: cpp

        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, dfb::interm0);
        }
        dfb_interm0.push_back(out_subblock_num_tiles);

b. **Computing with Partial Results**:

    - **Result Retrieval**: During block computations after the first block, we retrieve the stored results ``wait_front(...)`` for further computation. This retrieval, also known as "reloading" data, is the heart of our data reuse concept.  It is leveraged only when our flag ``enable_reload`` is set to true.  Recall from our understanding of dataflow buffers that there needs be synchronization that all tile work thus far be finished before contributing more partial results.

    .. code-block:: cpp

        if (enable_reload) {
            copy_init(dfb::interm0);
            dfb_interm0.wait_front(out_subblock_num_tiles);
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                copy_tile(dfb::interm0, i, i);
            }
            dfb_interm0.pop_front(out_subblock_num_tiles);
            matmul_init(dfb::in0, dfb::in1);
        }

    - **Execution with `matmul_tiles`**: Now we are ready to compute partial results and integrate them back into the computation stream (or for the last block of computation, culminate our data reuse to produce the final output tensor).  We call the ``matmul_tiles(...)`` function to execute our matmul on the core's subblocks of tiles.

    .. code-block:: cpp

        // Compute output sub-block from in0_subblock x in1_subblock
        int dst_index = 0;
        int in0_index_h_offset = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {
            for (uint32_t w = 0; w < out_subblock_w; w++) {
                int in1_index_inner_dim_offset = 0;
                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                    int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                    int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                    matmul_tiles(dfb::in0, dfb::in1, in0_index, in1_index, dst_index);
                    in1_index_inner_dim_offset += in1_per_core_w;
                }
                dst_index++;
            }
            in0_index_h_offset += in0_block_w;
        }

c. **Wrapping Up the Intermediate Buffer**:

    - **Freeing Up Space**: After all partial results have been computed and stored in our output subblock, we have completed the cycle of reuse, so now we free up the space in the intermediate dataflow buffer with ``pop_front(...)``.

Conclusion
----------

Those are the additional steps for getting ``matmul_multicore_data_reuse`` operations up and running on the compute engine. To see a more complicated example using core-to-core data movement, please refer to the :ref:`Matmul multi-core data mcast example<MatMul_Multi_Core_Optimized_Data_Mcast_example>`.
