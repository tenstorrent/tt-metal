.. _Eltwise sfpu example:

Eltwise SFPU
============

We now build a program that will perform operations using the SFPU (vector engine/Special Function Processing Unit). Though not packing as much punch as the FPU (Matrix Engine), the SFPU is a powerful unit that can perform complex element-wise operations. This example will show how to use the SFPU to perform ``exp(x)`` on the input data.

This example is similar to the previous example of adding two vectors using the FPU. But instead of using the FPU, we will use the SFPU. And only 1 input buffer is used instead of 2. The SFPU can perform a variety of operations not just exponential, such as square root, sine, cosine, ReLU and more.

We'll go through this code section by section. The fully source code for this example is available under the ``tt_metal/programming_examples/eltwise_sfpu`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``eltwise_sfpu`` executable in the ``build/programming_examples`` directory. For example:


.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
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
    std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
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



Next, create the kernels. Nothing different from the previous examples besides being different kernels.

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

The reader kernel takes in the address of the source buffer and the number of tiles to read. Then read each tile from the source buffer and write it to the circular buffer. The structure should be familiar by now, as it is similar to the previous example but with one less buffer to read from.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp
    #include <cstdint>

    void kernel_main() {
        uint32_t in0_addr = get_arg_val<uint32_t>(0);
        uint32_t n_tiles = get_arg_val<uint32_t>(1);

        constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

        const uint32_t tile_size_bytes = get_tile_size(cb_in0);
        const InterleavedAddrGenFast<true> in0 = {
            .bank_base_address = in0_addr,         // The base address of the buffer
            .page_size = tile_size_bytes,          // The size of a buffer page
            .data_format = DataFormat::Float16_b,  // The data format of the buffer
        };

        // Read in the data from the source buffer and write to the circular buffer
        // in a loop.
        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_reserve_back(cb_in0, 1);
            uint32_t cb_in0_addr = get_write_ptr(cb_in0);
            noc_async_read_tile(i, in0, cb_in0_addr);

            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }


The writer kernel is the exact same as the previous example.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/write_tile.cpp
    #include <cstdint>

    void kernel_main() {
        uint32_t c_addr = get_arg_val<uint32_t>(0);
        uint32_t n_tiles = get_arg_val<uint32_t>(1);

        // The circular buffer that we are going to read from and write to DRAM
        constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
        const uint32_t tile_size_bytes = get_tile_size(cb_out0);

        // Address of the output buffer
        const InterleavedAddrGenFast<true> out0 = {
            .bank_base_address = c_addr,
            .page_size = tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };

        // Loop over all the tiles and write them to the output buffer
        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_out0, 1);
            uint32_t cb_out0_addr = get_read_ptr(cb_out0);
            // write the tile to DRAM
            noc_async_write_tile(i, out0, cb_out0_addr);
            noc_async_write_barrier();
            // Mark the tile as consumed
            cb_pop_front(cb_out0, 1);
        }
    }

The compute kernel is the most interesting and different one. The flow is generally the same, but instead of calling functions that interact with the FPU (Matrix Engine), we use ones that invoke the SFPU. Note that some functions are postfixed with ``_sfpu`` to indicate that they are using the SFPU specifically, or they are implied by the fact that they do complex element-wise operations that are not supported by the FPU. The general flow of using the SFPU is as follows:

* Initialize the SFPU with the ``init_sfpu`` function
* Call the specific SFPU operation initialization function, such as ``exp_tile_init`` for exponential
* Acquire tile registers using ``tile_regs_acquire``
* Wait for data to be available in the circular buffer using ``cb_wait_front`` (same as the FPU)
* Copy the tile from the circular buffer to the registers using ``copy_tile``
* Perform the SFPU operation using ``exp_tile`` (or other SFPU operations)
* Wait for the result to be written back using ``tile_regs_commit`` and ``tile_regs_wait``
* Reserve space in the circular buffer for the result using ``cb_reserve_back`` (same as the FPU)
* Pack the result tile from the registers to the circular buffer using ``pack_tile``
* Mark the input tile as consumed using ``cb_pop_front`` (same as the FPU)
* Release the tile registers using ``tile_regs_release``

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp
    #include <cstdint>
    #include "compute_kernel_api/common.h"
    #include "compute_kernel_api/tile_move_copy.h"
    #include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
    #include "compute_kernel_api/eltwise_unary/exp.h"

    namespace NAMESPACE {
    void MAIN {
        uint32_t n_tiles = get_arg_val<uint32_t>(0);

        // Initialize the SFPU
        init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
        // Setup the SFPU for exponential operation
        exp_tile_init();
        for (uint32_t i = 0; i < n_tiles; i++) {
            // Make sure and acquire data before running the SFPU operation
            tile_regs_acquire();
            cb_wait_front(tt::CBIndex::c_0, 1);
            // Copy the tile from the circular buffer offset 0 to the tile registers 0
            copy_tile(tt::CBIndex::c_0, /*offset*/ 0, /*register_offset*/ 0);

            // Invoke the SFPU exponential operation on tile 0
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();

            // Clean up and prepare for the next iteration
            cb_reserve_back(tt::CBIndex::c_16, 1);
            pack_tile(0, tt::CBIndex::c_16);  // copy tile 0 from the registers to the CB
            cb_pop_front(tt::CBIndex::c_0, 1);
            tile_regs_release();
            cb_push_back(tt::CBIndex::c_16, 1);
        }
    }
    }

Set up runtime arguments
------------------------

For this program, the runtime arguments are similar to the previous examples. The reader gets the source address and size of the data to read. The writer gets the destination address and size of the data to write. The compute kernel simply know how much data to expect from the reader and how much data to write to the writer.

.. code-block:: cpp

    SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, unary_reader_kernel_id, core, {src0_dram_buffer->address(), n_tiles});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), n_tiles});

Program execution and final check
---------------------------------

Finally we can run the program. The program is enqueued to the command queue and the results are read back from the device. Then compared against the expected results.

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

This is the step to execute computation on the SFPU. Next we will introduce more complex data movement and running matrix multiplication using the matrix engine. See
:ref:`MatMul Single Core example<MatMul_Single_Core example>`.
