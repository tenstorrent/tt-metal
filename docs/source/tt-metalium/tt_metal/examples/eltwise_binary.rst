.. _Eltwise binary example:

Eltwise binary
==============

We now build a program that performs elementwise binary operations on a pair of equal-sized tensors using the FPU (matrix engine). The FPU can perform a variety of complex operations efficiently, including matrix multiplication and reduction. It can also perform common elementwise operations like addition. This example will add two tensors together using the FPU.

Although looking trivial, this example introduces essential concepts that are used in more complicated programs. Namely the compute kernel and the use of dataflow buffers. Tenstorrent devices are designed with explicit data movement in mind. Leading to a separate compute kernel. And dataflow buffers are the primary way to move data between kernels.

We'll go through any new code section by section. This builds on top of
previous examples. Note that we have this exact, full example program is located under
``tt_metal/programming_examples/metal_example_eltwise_binary``, so you can follow along.

To build and execute, you may use the following commands. Note that we include
the necessary environment variables here, but you may possibly need more
depending on the most up-to-date installation methods.

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/metal_example_eltwise_binary

Program setup
-------------

Initializing Metalium is almost the same as before. To recap, we need to

1. Create a mesh device (even single devices are treated as a 1x1 mesh)
2. Obtain the mesh command queue to issue data down/uploads and program execution, and create a workload to hold the program
3. Define the core we are going to use (in this case, only one at {0, 0})
4. Some basic constants for tile and total buffer sizes

.. code-block:: cpp

    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // A MeshWorkload is a collection of programs that will be executed on the mesh.
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr experimental::NodeCoord core = {0, 0};

    constexpr uint32_t n_tiles = 64;
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

This time, instead of 2 DRAM buffers to copy data out and back, we will create 3 tensors. 2 as data sources and 1 as the output. Each row of the row-major tensor holds one tile's worth of elements, so the page size is one tile. This is the most common setting for buffers in Metalium as the compute engines expect to operate on tiles of data.

.. code-block:: cpp

    TensorSpec dram_tensor_spec(
        Shape({n_tiles, elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM})); // This is a DRAM tensor.

    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    // Each tensor is replicated across the mesh; on a unit mesh this is a single device allocation.

Data preparation and upload is the same as before. For this example, tensor 0 will be filled with random values, and tensor 1 will be filled with a constant of -1 just for demonstration purposes. The data is then uploaded to the device asynchronously like before.

.. code-block:: cpp

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
    for(auto& val : a_data) {
        val = bfloat16(distribution(rng));
    }

    constexpr float val_to_add = -1.0f;
    std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));

    // Upload host vectors into the mesh tensors.
    HostTensor a_host_tensor = HostTensor::from_vector(a_data, dram_tensor_spec);
    HostTensor b_host_tensor = HostTensor::from_vector(b_data, dram_tensor_spec);
    cq.enqueue_write_tensor(a_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(b_host_tensor, src1_dram_tensor);

Dataflow buffers
----------------

Here we introduce a new concept: dataflow buffers. They are communication channels between the different kernel on a Tensix. Conceptually they act as pipes between different kernels. There are in total 32 dataflow buffers supported on a Tensix. To utilize them, the host program must describe the dataflow buffers in the program spec, and each kernel binds a dataflow buffer to a name it uses in the kernel. Kernels, dataflow buffers and tensor parameters are all identified by name within the program spec.



.. code-block:: cpp

    const experimental::KernelSpecName reader{"reader"};
    const experimental::KernelSpecName writer{"writer"};
    const experimental::KernelSpecName compute{"compute"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName src1_dfb{"src1"};
    const experimental::DFBSpecName dst_dfb{"dst"};
    const experimental::TensorParamName src0_tensor_param{"src0"};
    const experimental::TensorParamName src1_tensor_param{"src1"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    constexpr uint32_t tiles_per_cb = 2;
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = tile_size_bytes,                        // Since we will be sending one tile at a time, we set
                                                              // the entry size to the tile size
        .num_entries = tiles_per_cb,                          // The number of entries in the dataflow buffer
        .data_format_metadata = tt::DataFormat::Float16_b,    // The data format it'll hold
    };
    experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = src1_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = tiles_per_cb,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec dst_dfb_spec{
        .unique_id = dst_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = tiles_per_cb,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

The ``DataflowBufferSpec`` describes a dataflow buffer. At the core, there are a few critical parameters:

* The name of the dataflow buffer (``unique_id``)
* The number of entries in the dataflow buffer (``num_entries``)
* The data format of the dataflow buffer (bfloat16, block float4, etc..)
* How large each entry is (``entry_size``)
  * For most cases, this should be the same as the size of a tile of the underlying data format

For instance, to create a dataflow buffer of 2 tiles of bfloat16 data, we need to set the number of entries to ``2``. The entry size set to ``tile_size_bytes`` and the data format is set to bfloat16.

Data movement and compute kernels
---------------------------------

In the previous example (DRAM loopback), we used a single kernel to perform the entire operation; to read data from DRAM and write it back out. The Tensix core in fact contains 5 RISC-V cores. 2 of them are the data movement cores, which connects to the NoC and can issue commands to access other on chip resources (including DRAM). The other 3 are compute cores, which operates cooperatively and runs a single compute kernel. They have access to the matrix and vector engines, which performs the majority of the compute work on a Tensix.

.. note::
    Unlike traditional multi core processors. Where a problem is broken down into subtasks as assigned to the cores while each core runs the same code (SPMD, single program multiple data). The compute cores on a Tensix are designed to run different code. The compute kernel is compiled 3 times. Once for each of the 3 compute cores and generating 3 different binaries.  They work collaboratively to perform a single task. The 3 compute cores are the Unpack, Math and Pack cores. They are responsible for moving data from L1 into the matrix or vector engines, issue commands for computation and moving the results back out to L1. Which can be done at the same time for high throughput.

.. code-block:: cpp

    experimental::KernelSpec reader_spec{
        .unique_id = reader,
        .source = OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/read_tiles.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };
    experimental::KernelSpec writer_spec{
        .unique_id = writer,
        .source = OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/write_tile.cpp",
        .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source = OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/tiles_add.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src0_dfb, "in0"),
             experimental::ConsumerOf(src1_dfb, "in1"),
             experimental::ProducerOf(dst_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            },
    };

    experimental::ProgramSpec program_spec{
        .name = "eltwise_binary",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
        .work_units =
            {{.name = "eltwise_binary",
              .kernels = {reader, writer, compute},
              .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

Each kernel is described by a ``KernelSpec``. ``dfb_bindings`` and ``tensor_bindings`` bind the dataflow buffers and tensor parameters to the names the kernel uses in its source (for example ``in0``), and ``runtime_arg_schema`` declares the named runtime arguments. The ``ProgramSpec`` collects the kernels, dataflow buffers and tensor parameters, and its work units place the kernels on cores. ``MakeProgramFromSpec`` builds the program from it.

Notice the ``ComputeHardwareConfig`` object. This indicates to the framework that a compute kernel should be created. There's a plethora of settings that can be set here. The most important one is the ``fpu_math_fidelity`` setting. This controls how accurate certain floating point operations are on the FPU specifically. Other operations (like the ones in the vector engine) are _not_ affected by this setting.

Now let's look at the kernels. First the reader kernel. This kernel reads both input buffers from DRAM and pushes them into the dataflow buffers for the compute kernel to consume. As discussed above, consider the dataflow buffers as a pipe. Data can be pushed into the pipe and read out of it. And both ends must ensure there is space to do so.

To do so, the reader creates 2 tensor accessors from the tensors bound to it by name (``tensor::in0`` and ``tensor::in1``). Unlike on most processors, the Tensix doesn't just see raw bytes. The tensor accessor contains metadata about the data it is associated to enable the data be read out of the DRAM in the correct format. Then in a loop, the program waits for space to be available in the dataflow buffers. Once there is, it reads a tile from DRAM and pushes it into the dataflow buffer. The ``noc.async_read_barrier()`` call waits for the read to finish before committing the data to the dataflow buffer. This is important as the read is asynchronous and data is not guaranteed to be there when the program continues.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        // The dataflow buffers to read the tiles into
        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);

        const uint32_t tile_size_bytes = dfb_in0.get_entry_size();

        // The input tensors are bound to this kernel by name in the host code.
        const auto in0 = TensorAccessor(tensor::in0);
        const auto in1 = TensorAccessor(tensor::in1);

        // Create Device 2.0 Noc object
        Noc noc;

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.reserve_back(1);
            dfb_in1.reserve_back(1);

            noc.async_read(in0, dfb_in0, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read(in1, dfb_in1, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});

            noc.async_read_barrier();  // Wait until tile reads are done
            dfb_in0.push_back(1);
            dfb_in1.push_back(1);
        }
    }

The compute kernel performs the actual elementwise addition. It initializes the matrix engine for addition, then enters a loop. Inside the loop, it waits for data to be available in the input dataflow buffers, and for the destination registers to be free. The destination registers (16 in total, with 8 usable at a time) are used to transfer data to and from the computation engines. The kernel adds the input tiles, writes the result to destination register 0, waits to be sure the result is ready, and waits for the output dataflow buffer to have space, and pushes the computed tile into it. Finally, it marks the input tiles as consumed, the output tile as produced, and releases the registers.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_in1(dfb::in1);
        DataflowBuffer dfb_out0(dfb::out0);

        constexpr uint32_t dst_reg = 0;

        compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out0);
        add_init(dfb::in0, dfb::in1);

        // Loop over all the tiles and perform the computation
        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.wait_front(1);
            dfb_in1.wait_front(1);

            tile_regs_acquire();

            add_tiles(dfb::in0, dfb::in1, 0, 0, dst_reg);

            tile_regs_commit();
            tile_regs_wait();

            dfb_out0.reserve_back(1);
            pack_tile(dst_reg, dfb::out0);

            dfb_out0.push_back(1);
            dfb_in0.pop_front(1);
            dfb_in1.pop_front(1);
            tile_regs_release();
        }
    }

The writer kernel looks similar to the reader kernel. Instead of reading, it writes data back into DRAM and uses the appropriate API to do so.


.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        // The dataflow buffer that we are going to read from and write to DRAM
        DataflowBuffer dfb_out(dfb::out0);
        const uint32_t tile_size_bytes = dfb_out.get_entry_size();

        // The output tensor is bound to this kernel by name in the host code.
        const auto out0 = TensorAccessor(tensor::out0);

        // Create Device 2.0 Noc object
        Noc noc;

        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_out.wait_front(1);
            noc.async_write(dfb_out, out0, tile_size_bytes, {}, {.page_id = i});
            noc.async_write_barrier();
            dfb_out.pop_front(1);
        }
    }


Then the host program sets the kernel arguments and launches the program. Each kernel gets the named runtime argument ``n_tiles``, and the DRAM tensors are bound to the tensor parameters by name. There is no difference in setting the arguments for the compute kernel.

.. note::
    Unlike OpenCL/CUDA. Each kernel (reader, compute and writer) can have it's own set of arguments. Furthermore, on a multi cored program (i.e. using more then 1 Tensix core), kernels within each core can have different arguments. This enables Metalium to exploit the grid like nature of the Tenstorrent processors to achieve high performance.

.. code-block:: cpp

    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = writer,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = compute,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
        .tensor_args =
            {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

Download the result and verify output
-------------------------------------

Finally, we download the result and verify the output. Again we read the data into a vector and wait for the transfer to finish. The result is then compared to the expected output. Note that we use a loose tolerance for the comparison because of the nature of bfloat16.

.. code-block:: cpp

    std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

    constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
    TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
    for (size_t i = 0; i < result_vec.size(); ++i) {
        const float expected = static_cast<float>(a_data[i]) + val_to_add;
        const float actual = static_cast<float>(result_vec[i]);

        if (std::abs(expected - actual) > eps) {
            pass = false;
            fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
        }
    }

Validation and teardown
-----------------------

.. code-block:: cpp

   pass &= mesh_device->close();

We now use ``mesh_device->close()`` to teardown our mesh device. This releases resources associated with the device.

Next we will explore the use of the vector engine (SFPU) to perform the same operation.
:ref:`Eltwise sfpu example<Eltwise sfpu example>`.
