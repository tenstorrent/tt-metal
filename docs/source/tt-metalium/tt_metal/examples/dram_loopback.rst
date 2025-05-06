.. _DRAM Loopback Example:

DRAM Loopback
=============

The is the simplest example of using the TT-Metal API. A data movement core in the Tensix copies data DRAM into it's L1(SRAM) buffer and back out to DRAM. Hence "loopback".


We'll go through this code section by section. The fully source code for this example is avaliable under the ``tt_metal/programming_examples/loopback`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``loopback`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/loopback

Device initalization
--------------------

.. code-block:: cpp

   constexpr int device_id = 0;
   auto device = CreateDevice(device_id);

First we open a device. This is our gateway to all operations on the accelerator. The device ID is simply an index into the list of available devices (from 0 to N). Thus device 0 is the first device and always available if one is installed.

Program setup
-------------

.. code-block:: cpp

   CommandQueue& cq = device->command_queue();
   Program program = CreateProgram();

Operations in Metalium are almost always capable to be run asynchronously and the ordering of operations is managed by a command queue. The command queue, like the name suggests, is a FIFO queue of commands that are executed in order. And commands are operations that are run on the device, including but not limited to upload/download of data and program execution.

Next, we create a ``Program`` object that we will fill in later. A program is a set of kernels that are executed on the device. If you are familiar with OpenCL, the program in Metalium is different from OpenCL in that All Tensix cores need to to run the exact same kernel at the same time. However in this example, we are only going to use one of all the cores in the device.

Creating a data movement kernel
-------------------------------

Create a kernel that will copy data from DRAM to L1 and back. Since we are only using one Tensix core, ``{0, 0}`` is the only core (core on the most top left) we use. And as we are moving data from DRAM to L1, This is a data movement kernel using the movement processor 0, and the default NoC interface.

.. code-block:: cpp

    constexpr CoreCoord core = {0, 0};

    KernelHandle dram_copy_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

Create buffers in DRAM and L1 (SRAM)
------------------------------------

Next, we need to declare buffers that will hold the actual data and a intermidiate buffer on chip,

There's in total 3 buffers to be created:
* An L1 (SRAM) buffer within the core itself that will act as temporary storage
* A DRAM buffer that will house input data
* A DRAM buffer that will be written to with output data

Note that almost all operations on the Tensix are aligned with tiles. And a tile is a 32x32 grid of values. The data type used in this example is bfloat16 as it is what the math engine uses internally (though we won't touch the math engine in this example). Making each tile 32 x 32 x 2 bytes = 2048 bytes. And we wish to allocate 50 tiles in each buffer.

There are two types of buffers in the Tensix: L1 and DRAM. L1 is a misnomer as it can be mistaken as similar to L1 cache in a CPU. In fact, the L1 is a SRAM scratchpad on the Tensix. Each generation of Tensotrrent processors has a different amount of L1 memory per Tensix. Grayskull had 1MB and Wormhole/Blackhole has 1.5MN.

Note the ``page_size`` argument in the buffer config and the ``Interleaved`` in the buffer type. Both L1 and DRAM are splitted into banks. Each bank is a physical memory unit that can be accessed independently. Howerver, managing banks seperately is trick and not scalable. Interleaved buffers simply round-robin the data across all banks every ``page_size`` bytes. This allows the programmer to treat the buffer as a single unit, while taking advantage of the parallelism of the banks for hifher bandwidth. Setting page size equal to the buffer size means that the entire buffer will live on a single bank. This is not recommended for performance and in most cases, page size is set to the size of a tile. However, this configuration allows easy illustration of NoC operations. However, these are implementation details and the programmer should not be overly concerned with them.

.. code-block:: cpp

  constexpr uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;
  constexpr uint32_t single_tile_size = sizeof(bfloat16) * tile_size;
  constexpr uint32_t num_tiles = 50;
  constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;
  tt_metal::InterleavedBufferConfig l1_config{
                                        .device=device,
                                        .size = dram_buffer_size,
                                        .page_size = dram_buffer_size,
                                        .buffer_type = tt_metal::BufferType::L1
                                        };

  Buffer l1_buffer = CreateBuffer(l1_config);

The only difference between the L1 and DRAM buffer is the ``BufferType``. The L1 buffer is created with a ``BufferType::L1`` and the DRAM buffer is created with a ``BufferType::DRAM``.

.. code-block:: cpp

  tt_metal::InterleavedBufferConfig dram_config{
                                        .device=device,
                                        .size = dram_buffer_size,
                                        .page_size = dram_buffer_size,
                                        .buffer_type = tt_metal::BufferType::DRAM
                                        };

  Buffer input_dram_buffer = CreateBuffer(dram_config);
  const uint32_t input_dram_buffer_addr = input_dram_buffer.address();

  Buffer output_dram_buffer = CreateBuffer(dram_config);
  const uint32_t output_dram_buffer_addr = output_dram_buffer.address();

  const uint32_t input_bank_id = 0;
  const uint32_t output_bank_id = 0;

Sending real data into DRAM
---------------------------

.. code-block:: cpp

  std::vector<bfloat16> input_vec(num_tiles * tile_size);
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
  for (auto& val : input_vec) {
      val = bfloat16(distribution(rng));
  }
  EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);

Send in a randomly-generated BFP16 (Brain 16bit floating point) vector that will act as our input data tensor.

Note the final ``false`` argument. This indicates to tt-Metalium that the upload is non-blocking. The function may retrun as soon as possible while data transfer is still in progress. This is useful for performance, but the profram is responsible for ensuring that the the source buffer is not freed before the transfer is complete. In this case, there are future blocking calls/calls to ``Finish`` that will ensure commands are completed before the program exits, which is also when the source buffer is freed.

Setting runtime arguments for the data movement kernel
------------------------------------------------------

.. code-block:: cpp

  const std::vector<uint32_t> runtime_args = {
      l1_buffer.address(),
      input_dram_buffer.address(),
      input_bank_id,
      output_dram_buffer.address(),
      output_bank_id,
      l1_buffer.size()
  };

  SetRuntimeArgs(
      program,
      dram_copy_kernel_id,
      core,
      runtime_args
  );

We now set runtime arguments for our data movement kernel. The kernel can then access these arguments at runtime. For this specific kernel, we need to pass in the following arguments:

* Where the L1 buffer starts (memory address)
* Where the input DRAM buffer starts (memory address)
* The channel index of the input DRAM buffer
* Where the output DRAM buffer starts (memory address)
* The channel index of the output DRAM buffer
* The size of the copy
  * Which happens to be the same as the size of the L1 buffer

Running the program
-------------------

.. code-block:: cpp

    EnqueueProgram(cq, program, false);
    Finish(cq);
    // Equivalently, we could have done:
    // EnqueueProgram(cq, program, true);


Finally, we launch our program. The ``Finish`` call waits for the the host program only continues execution after everything in the command queue has been completed. The final argument in ``EnqueueProgram`` indicates that the program is non-blocking. Setting it to ``true`` would cause the program to block until the program is finished. Efficiently, this is the same as calling ``Finish`` after the program is enqueued.

Download the result and verify output
-------------------------------------

Then we can finally read back the data from the output buffer and assert that
it matches what we sent. Again the final ``true`` argument causes the data transfer to be blocking. Thus we know that the data is fully avaliable when the function returns.

.. code-block:: cpp

  std::vector<bfloat16> result_vec;
  EnqueueReadBuffer(cq,output_dram_buffer, result_vec, true);

  for (int i = 0; i < input_vec.size(); i++) {
    if (input_vec[i] != result_vec[i]) {
        pass = false;
        break;
    }
  }

Validation and teardown
-----------------------

.. code-block:: cpp

   pass &= CloseDevice(device);

We now use ``CloseDevice`` to teardown our device. This releases resources associated with the device.

Now we can start adding some compute to our program. Please refer to the
:ref:`Eltwise sfpu example<Eltwise sfpu example>`.
