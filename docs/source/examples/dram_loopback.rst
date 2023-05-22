.. _DRAM Loopback Example:

DRAM Loopback
=============

We will build a program in TT-Metal that will simply copy data from one DRAM
buffer to another, using the compute engine and an intermediate L1 buffer to do
so. We call this concept "loopback".

We'll go through this code section by section. Note that we have this exact,
full example program in ``programming_examples/loopback/loopback.cpp``, so you
can follow along.

Silicon accelerator setup
-------------------------

.. code-block:: cpp

   constexpr int pci_express_slot = 0;
   Device *device =
       CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

   pass &= InitializeDevice(device);

We instantiate and initialize a device to control our ``GRAYSKULL`` type
accelerator.

Program pre-compilation setup
-----------------------------

.. code-block:: cpp

   Program program = Program();

We create a ``Program`` to be run on our Grayskull accelerator. This is how
we'll be keeping track of things in our session with the device.

Building a data movement kernel
-------------------------------

Declare a ``DataMovementKernel``. We'll use a pre-written kernel that copies
data from one place to another.

We will be using the accelerator core with coordinates ``{0, 0}``.

.. code-block:: cpp

  constexpr tt_xy_pair core = {0, 0};

  DataMovementKernel *dram_copy_kernel = CreateDataMovementKernel(
      program,
      "programming_examples/kernels/dataflow/loopback_dram_copy.cpp",
      core,
      DataMovementProcessor::RISCV_0,
      NOC::RISCV_0_default
  );

Create buffers in DRAM and L1
-----------------------------

Next, we need to declare buffers that we will use during execution. We will
need

* An L1 buffer within the core itself that will be used to store the compute
  engine's work
* A DRAM buffer that will house input data
* A DRAM buffer that will be written to with output data

.. code-block:: cpp

  constexpr uint32_t single_tile_size = 2 * (32 * 32);
  constexpr uint32_t num_tiles = 50;
  constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;
  constexpr uint32_t l1_buffer_addr = 400 * 1024;

  uint32_t l1_bank_id = device->bank_ids_from_logical_core(core).at(0);
  Buffer l1_buffer = Buffer(device, dram_buffer_size, l1_buffer_addr, l1_bank_id, dram_buffer_size, BufferType::L1);

For simplicity, let's make the size of all our buffers 50 tiles. We'll also put
this particular L1 Buffer at location ``400KB``.

Let's make the input and output DRAM buffers.

.. code-block:: cpp

  constexpr uint32_t input_dram_buffer_addr = 0;
  constexpr int dram_channel = 0;
  Buffer input_dram_buffer = Buffer(device, dram_buffer_size, input_dram_buffer_addr, dram_channel, dram_buffer_size, BufferType::DRAM);

  constexpr uint32_t output_dram_buffer_addr = 512 * 1024;
  Buffer output_dram_buffer = Buffer(device, dram_buffer_size, output_dram_buffer_addr, dram_channel, dram_buffer_size, BufferType::DRAM);

Program compilation
-------------------

.. code-block:: cpp

   pass &= CompileProgram(device, program);

Next we compile our program.

Sending real data into DRAM
---------------------------

.. code-block:: cpp

  std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
      dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
  WriteToBuffer(input_dram_buffer, input_vec);

Send in a randomly-generated FP16 vector that will act as our input data
tensor.

Loading the program with desired settings
-----------------------------------------

.. code-block:: cpp

   pass &= ConfigureDeviceWithProgram(device, program);

We then configure the device with our compiled program. Now it's time for any
runtime arguments or input data.

Sending runtime arguments for the data movement kernel
------------------------------------------------------

.. code-block:: cpp

  const std::vector<uint32_t> runtime_args = {
      l1_buffer.address(),
      input_dram_buffer.address(),
      static_cast<uint32_t>(input_dram_buffer.noc_coordinates().x),
      static_cast<uint32_t>(input_dram_buffer.noc_coordinates().y),
      output_dram_buffer.address(),
      static_cast<uint32_t>(output_dram_buffer.noc_coordinates().x),
      static_cast<uint32_t>(output_dram_buffer.noc_coordinates().y),
      l1_buffer.size()
  };

  pass &= WriteRuntimeArgsToDevice(
      device,
      dram_copy_kernel,
      core,
      runtime_args
  );

We now write runtime arguments for our data movement kernel. For this
particular kernel, we have to provide:

* Where the L1 buffer starts (memory address)
* Where the input DRAM buffer starts (memory address)
* The location of the input DRAM buffer's channel on the NOC
* Where the output DRAM buffer starts (memory address)
* The location of the output DRAM buffer's channel on the NOC
* The size of the buffers

Running the program
-------------------

.. code-block:: cpp

   pass &= LaunchKernels(device, program);

Now we finally launch our program. This is a blocking call which will finish
when the program on the device finishes.

Launch and verify output
------------------------

Then we can finally read back the data from the output buffer and assert that
it matches what we sent!

.. code-block:: cpp

  std::vector<uint32_t> result_vec;
  ReadFromBuffer(output_dram_buffer, result_vec);

  pass &= input_vec == result_vec;

Validation and teardown
-----------------------

.. code-block:: cpp

   pass &= CloseDevice(device);

We now use ``CloseDevice`` to teardown our connection to the Tenstorrent
device.

Now we can start adding some compute to our program. Please refer to the
:ref:`Eltwise binary example<Eltwise binary example>`.
