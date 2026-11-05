.. _Eltwise sfpu example:

Eltwise SFPU
============

We now build a program that will perform operations using the SFPU (vector engine/Special Function Processing Unit). Though not packing as much punch as the FPU (Matrix Engine), the SFPU is a powerful unit that can perform complex element-wise operations. This example will show how to use the SFPU to perform ``exp(x)`` on the input data.

This example is similar to the previous example of adding two vectors using the FPU. But instead of using the FPU, we will use the SFPU. And only 1 input buffer is used instead of 2. The SFPU can perform a variety of operations not just exponential, such as square root, sine, cosine, ReLU and more.

We'll go through this code section by section. The fully source code for this example is available under the ``tt_metal/programming_examples/eltwise_sfpu`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``metal_example_eltwise_sfpu`` executable in the ``build/programming_examples`` directory. For example:


.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh
    ./build/programming_examples/metal_example_eltwise_sfpu

Program setup
-------------

Like the previous examples, setting up programs running on the accelerator is similar

* Allocate tensors for both input and output data
* Describe dataflow buffers for communication between kernels
* Describe both data movement and compute kernels


First, allocate the tensors using the ``MeshTensor`` API. A ``TensorSpec`` describes the shape, data type, layout and memory placement of the tensor. Each row of a row-major tensor is one page, so a row of one tile's worth of elements sets the page size to the tile size (in bytes), which matches how the compute engine expects data. Each tensor is replicated across the mesh. The destination tensor is not initialized as it will be filled with results later.

.. code-block:: cpp

    TensorSpec dram_tensor_spec(
        Shape({n_tiles, elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

    // Initialize the input data with random values and use as the input to the kernel.
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
    for (bfloat16& v : src0_vec) {
        v = bfloat16(dist(rng));
    }

    HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);

Then we describe the dataflow buffers. Again the entry size is set to tile size. There are 2 entries in each dataflow buffer to allow overlapping of data movement and compute operations. (data movement kernels will send/consume 1 tile at a time in this example). Kernels, dataflow buffers and tensor parameters are identified by name within the program spec.

.. code-block:: cpp

    const experimental::KernelSpecName unary_reader_kernel{"unary_reader"};
    const experimental::KernelSpecName unary_writer_kernel{"unary_writer"};
    const experimental::KernelSpecName eltwise_sfpu_kernel{"eltwise_sfpu"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName output_dfb{"output"};
    const experimental::TensorParamName src0_tensor_param{"src0"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    // Describe 2 dataflow buffers for input and output.
    constexpr uint32_t num_input_tiles = 2;
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = tile_size_bytes,
        .num_entries = num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

Next, describe the kernels as ``KernelSpec`` objects and collect them, the dataflow buffers and the tensor parameters in a ``ProgramSpec``. Nothing different from the previous examples besides being different kernels. Each kernel binds the dataflow buffers and tensors to the names it uses in its source (for example ``in0``), and declares its named runtime arguments. ``MakeProgramFromSpec`` then builds the program.

.. code-block:: cpp

    experimental::KernelSpec unary_reader_kernel_spec{
        .unique_id = unary_reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0")},
        .tensor_bindings = {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    experimental::KernelSpec unary_writer_kernel_spec{
        .unique_id = unary_writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
        .dfb_bindings = {experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    experimental::KernelSpec eltwise_sfpu_kernel_spec{
        .unique_id = eltwise_sfpu_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src0_dfb, "in0"), experimental::ProducerOf(output_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    experimental::ProgramSpec program_spec{
        .name = "eltwise_sfpu",
        .kernels = {unary_reader_kernel_spec, unary_writer_kernel_spec, eltwise_sfpu_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
        .work_units =
            {{.name = "eltwise_sfpu",
              .kernels = {unary_reader_kernel, unary_writer_kernel, eltwise_sfpu_kernel},
              .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

The kernels
-----------

The reader kernel takes in the number of tiles to read as a named runtime argument. The source tensor is bound to the kernel by name, so the kernel does not need its address. Then read each tile from the source tensor and write it to the dataflow buffer. The structure should be familiar by now, as it is similar to the previous example but with one less buffer to read from.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp
    #include <cstdint>

    #include "api/dataflow/noc.h"
    #include "api/dataflow/dataflow_buffer.h"
    #include "api/tensor/noc_traits.h"
    #include "experimental/kernel_args.h"

    void kernel_main() {
        // Read parameters from the kernel arguments
        uint32_t n_tiles = get_arg(args::n_tiles);

        // The dataflow buffer to read the tiles into
        DataflowBuffer dfb_in0(dfb::in0);

        const uint32_t tile_size_bytes = dfb_in0.get_entry_size();

        // The input tensor is bound to this kernel by name in the host code.
        const auto in0 = TensorAccessor(tensor::in0);

        // Create Device 2.0 Noc object
        Noc noc;

        // Loop over all the tiles and read them into the dataflow buffer
        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_in0.reserve_back(1);
            noc.async_read(in0, dfb_in0, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});

            noc.async_read_barrier();
            dfb_in0.push_back(1);
        }
    }


The writer kernel is the exact same as the previous example.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/write_tile.cpp
    #include <cstdint>

    #include "api/dataflow/noc.h"
    #include "api/dataflow/dataflow_buffer.h"
    #include "api/tensor/noc_traits.h"
    #include "experimental/kernel_args.h"

    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        // The dataflow buffer that we are going to read from and write to DRAM
        DataflowBuffer dfb_out(dfb::out0);
        const uint32_t tile_size_bytes = dfb_out.get_entry_size();

        // The output tensor is bound to this kernel by name in the host code.
        const auto out0 = TensorAccessor(tensor::out0);

        // Create Device 2.0 Noc object
        Noc noc;

        // Loop over all the tiles and write them to the output buffer
        for (uint32_t i = 0; i < n_tiles; i++) {
            dfb_out.wait_front(1);
            // Write the tile to DRAM
            noc.async_write(dfb_out, out0, tile_size_bytes, {}, {.page_id = i});
            noc.async_write_barrier();
            // Mark the tile as consumed
            dfb_out.pop_front(1);
        }
    }

The compute kernel is the most interesting and different one. The flow is generally the same, but instead of calling functions that interact with the FPU (Matrix Engine), we use ones that invoke the SFPU. Note that some functions are postfixed with ``_sfpu`` to indicate that they are using the SFPU specifically, or they are implied by the fact that they do complex element-wise operations that are not supported by the FPU. The general flow of using the SFPU is as follows:

* Configure the hardware once with ``compute_kernel_hw_startup``, then set up the tile copy with ``copy_init``
* Call the specific SFPU operation initialization function, such as ``exp_tile_init`` for exponential
* Acquire tile registers using ``tile_regs_acquire``
* Wait for data to be available in the dataflow buffer using ``wait_front`` (same as the FPU)
* Copy the tile from the dataflow buffer to the registers using ``copy_tile``
* Perform the SFPU operation using ``exp_tile`` (or other SFPU operations)
* Wait for the result to be written back using ``tile_regs_commit`` and ``tile_regs_wait``
* Reserve space in the dataflow buffer for the result using ``reserve_back`` (same as the FPU)
* Pack the result tile from the registers to the dataflow buffer using ``pack_tile``
* Mark the input tile as consumed using ``pop_front`` (same as the FPU)
* Release the tile registers using ``tile_regs_release``

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp
    #include <cstdint>
    #include "api/compute/common.h"
    #include "api/compute/tile_move_copy.h"
    #include "api/compute/eltwise_unary/eltwise_unary.h"
    #include "api/compute/eltwise_unary/exp.h"
    #include "api/dataflow/dataflow_buffer.h"
    #include "experimental/kernel_args.h"

    void kernel_main() {
        uint32_t n_tiles = get_arg(args::n_tiles);

        DataflowBuffer dfb_in0(dfb::in0);
        DataflowBuffer dfb_out0(dfb::out0);

        // Initialize the SFPU
        compute_kernel_hw_startup(dfb::in0, dfb::out0);
        copy_init(dfb::in0);
        // Telling the SFPU to perform exponential. This is required each time we
        // switch to a different SFPU operation.
        exp_tile_init();
        for (uint32_t i = 0; i < n_tiles; i++) {
            tile_regs_acquire();
            dfb_in0.wait_front(1);
            copy_tile(dfb::in0, /*offset*/ 0, /*register_offset*/ 0);
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            dfb_out0.reserve_back(1);
            pack_tile(0, dfb::out0);  // copy tile 0 from the registers to the dataflow buffer
            dfb_in0.pop_front(1);
            tile_regs_release();
            dfb_out0.push_back(1);
        }
    }

Set up runtime arguments
------------------------

For this program, the runtime arguments are similar to the previous examples. Each kernel gets the named runtime argument ``n_tiles``: the reader and writer know the size of the data to read and write, and the compute kernel knows how much data to expect from the reader and how much data to write to the writer. The source and destination tensors are bound to their tensor parameters by name, instead of passing buffer addresses.

.. code-block:: cpp

    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = eltwise_sfpu_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = unary_reader_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
             {.kernel = unary_writer_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
        .tensor_args =
            {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

Program execution and final check
---------------------------------


Finally we can run the program. The program is enqueued to the mesh command queue and the results are read back from the device. Then compared against the expected results.

.. code-block:: cpp

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

    constexpr float eps = 5e-2f;
    for (uint32_t i = 0; i < result_vec.size(); ++i) {
        float expected = static_cast<float>(bfloat16(std::exp(static_cast<float>(src0_vec[i]))));
        float result = static_cast<float>(result_vec[i]);
        if (std::abs(expected - result) > eps) {
            pass = false;
            fmt::print(stderr, "Result mismatch at index {}: {} != {}\n", i, expected, result);
        }
    }

    // Finally, close the device.
    pass &= mesh_device->close();

Conclusion
----------

This is the step to execute computation on the SFPU. Next we will introduce more complex data movement and running matrix multiplication using the matrix engine. See
:ref:`MatMul Single Core example<MatMul_Single_Core example>`.
