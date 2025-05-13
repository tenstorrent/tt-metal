.. _Eltwise sfpu example:

Eltwise SFPU
============

We now build a program that will perform operations using the SFPU (vector engine/Special Function Processing Unit).

We'll go through this code section by section. The fully source code for this example is available under the ``tt_metal/programming_examples/eltwise_sfpu`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``eltwise_sfpu`` executable in the ``build/programming_examples`` directory. For example:


.. code-block:: bash

    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh
    ./build/programming_examples/eltwise_sfpu

Program setup
-------------

Like the previous examples, setting up programs running on the accelerator is similar

* Allocate buffers for both input and output data
* Allocate circular buffers for communication between kernels
* Setup both data movement and compute kernels

First, allocate the buffers using the interleaved config with page size set to the tile size (in bytes). This is the most common buffer type and configuration used in Metalium as it fits how the compute engine expects data to be sent in. Then data is generated and written to the source buffer. The destination buffer is not initialized as it will be filled with the results later.

.. code-block:: cpp

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = tile_size_bytes * n_tiles,
        .page_size = tile_size_bytes,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    // Fill a host buffer with random data and upload to the device.
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    std::vector<bfloat16> src0_vec(n_tiles * elemnts_per_tile);
    for (bfloat16& v : src0_vec) {
        v = bfloat16(dist(rng));
    }

    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);

Then we allocate the circular buffers. Again page size is set to tile size. There is 2 pages in each circular buffer to allow overlapping of data movement and compute operations. (data movement kernels will send/consume 1 tile at a time in this example).

.. code-block::cpp

    // Allocate 2 circular buffers for input and output.
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, tile_size_bytes);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);



Next, we need to set up the kernels. Nothing different from the previous examples besides being different kernels.

.. code-block:: cpp

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/write_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
        core,
        ComputeConfig{
            .math_approx_mode = false,
        });

The kernels
-----------




Set up runtime arguments
------------------------

For this program, the runtime arguments are similar to the previous examples. The reader gets the source address and size of the data to read. The writer gets the destination address and size of the data to write. The compute kernel simply know how much data to expect from the reader and how much data to write to the writer.

.. code-block:: cpp

    SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, unary_reader_kernel_id, core, {src0_dram_buffer->address(), n_tiles});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), n_tiles});

Program execution and final check
---------------------------------

Finally we can run the program. The program is enqueued to the cinnabd queue and the results are read back from the device and compared against the expected results.

.. code-block:: cpp

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<bfloat16> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    for(uint32_t i = 0; i < result_vec.size(); ++i) {
        float expected = bfloat16(std::exp(src0_vec[i].to_float())).to_float();
        float result = result_vec[i].to_float();
        if (std::abs(expected - result) > eps) {
            pass = false;
            tt::log_error(tt::LogTest, "Result mismatch at index {}: {} != {}", i, expected, result);
        }
    }
    pass &= CloseDevice(device);

Conclusion
----------

This is the step to execute computation on the SFPU. Next we will intoduce more complex data movement and running matrix multiplication using the matrix engine. See
:ref:`MatMul Single Core example<MatMul_Single_Core example>`.
