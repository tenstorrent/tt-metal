.. _Eltwise binary example:

Eltwise binary
==============

We now build a program that will perform elementwise binary operations on a pair of equal-sized tensors using the FPU (matrix engine). The FPU can perform a variety of complex operations efficiently, including matrix multiplication and reduction. It can also perform common elementwise operations like addition. This example will add two tensors together using the FPU.

Although looking trivial, this example introduces essential concepts that are used in more complicated programs. Namely the compute kernel and the use of circular buffers. Tenstorrent devices are designed with explicit data movement in mind. Leading to a seperate compute kernel. And for circular buffers is the primary way to move data between kernels.

We'll go through any new code section by section. This builds on top of
previous examples. Note that we have this exact, full example program is located under
``tt_metal/programming_examples/eltwise_binary``, so you can follow along.

To build and execute, you may use the following commands. Note that we include
the necessary environment variables here, but you may possibly need more
depending on the most up-to-date installation methods.

.. code-block:: bash

    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/eltwise_binary

Program setup
-------------

Initializing Metalium is almost the same as before. To recap, we need to

1. Open a device (again we are using device 0)
2. Obtain the command queue to issue down/uploads and program execution
3. Create a program object to hold our kernels
4. Define the core we are going to use (in this case, only one at {0, 0})
5. Some basic constants for tile and total buffer sizes

.. code-block:: cpp

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();

    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t n_tiles = 64;
    constexpr uint32_t elemnts_per_tile = TILE_WIDTH * TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elemnts_per_tile;

This time, instead of 2 DRAM buffers to copy data out and back, we will create 3 buffers. 2 as data sources and 1 as the output. Page size is set to one tile. This is the most common setting for buffers in Metalium as the compute engines expect to operate on tiles of data.

.. code-block:: cpp

    InterleavedBufferConfig config{
        .device = device,                       // The device to create the buffer on
        .size = n_tiles * tile_size_bytes,      // The size of the buffer in bytes
        .page_size = tile_size_bytes,           // The page size of the buffer in bytes. In this case, will be
                                                // one tile per page.
        .buffer_type = BufferType::DRAM};       // This is a DRAM buffer.
    auto src0_dram_buffer = CreateBuffer(config);
    auto src1_dram_buffer = CreateBuffer(config);
    auto dst_dram_buffer = CreateBuffer(config);

Data preperation and upload is the same as before. For this example, buffer 0 will be filled with random values, and buffer 1 will be filled with a constant of -1 just for demonstration purposes. The data is then uploaded to the device asynchronously like before.

.. code-block:: cpp

    constexpr float val_to_add = -1.0f;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> src0_data(elemnts_per_tile * n_tiles);
    std::vector<bfloat16> src1_data(elemnts_per_tile * n_tiles, bfloat16(val_to_add));
    for(auto& val : a_data) {
        val = bfloat16(distribution(rng));
    }
    // Upload the data from host to the device.
    EnqueueWriteBuffer(cq, src0_dram_buffer, a_data, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b_data, false);

Data movement and compute kernels
---------------------------------

In the previous example (DRAM loopback), we used a single kernel to perform the entire operation; to read data from DRAM and write it back out. The Tensix core in fact contains 5 RISC-V cores. 2 of them are the data movement cores, which connects to the NoC and can issue commands to access other on chip resources (includign DRAM). The other 3 are compute cores, which operates cooperatively and runs a single compute kernel. They have access to the matrix and vector engines, which performs the majority of the compute work on a Tensix.

.. note::
    Unlike traditional multi core processors. Where a problem is broken down into subtasks as assigned to the cores while each core runs the same code (SPMD, single program multiple data). The compute cores on a Tensix are designed to run different code. The compute kernel is compiled 3 times. Once for each of the 3 compute cores and generating 3 different binaries.  They work collaboratively to perform a single task. The 3 compute cores are the Unpack, Math and Pack cores. They are responsible for moving data from L1 into the matrix or vector engines, issue commands for computation and moving the results back out to L1. Which can be done at the smae time for high throughput.

.. code-block:: cpp

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

Notice the ``ComputeConfig`` object. This indicates to the framework that a compute kernel should be created. There's a plethora of settings that can be set here. The most important one is the ``math_fidelity`` setting. This controls how accurate certain floating point operations are on the FPU specifically. Other operations (like the ones in the vector engine) are _not_ affected by this setting.

Now let's look at the kernels. First the reader kernel. This kernel reads both input buffers from DRAM and pushes them into the circular buffers (will discuss in the following section) for the compute kernel to consume. For now, consider the circular buffers as a pipe. Data can be pushed into the pipe and read out of it. And both ends mush ensure there is space to do so.




Circular buffers
----------------

Here we introduce a new concept: circular buffers. Conceptually they act as pipes between differnt kernels. The
