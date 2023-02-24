.. _DRAM Loopback Example:

DRAM Loopback
=============

Let's understand the memory structres in LL Buda by using an example kernel
that will simply copy data from one input DRAM buffer to another output DRAM
buffer, using the compute engine and an L1 memory buffer to do so. We call this
concept "loopback", or more precisely "DRAM loopback".

We will present this as a series of code additions on top of the :ref:`basic
example<Setting up a basic test>`.

Building a data movement kernel
-------------------------------

First, right after we instantiate a ``Program``, we should declare a
``DataMovementKernel`` that we want to use. We'll use a pre-written kernel that
uses the compute engine to copy data from one place to another in the device
memory. We will need to provide it with some runtime parameters that we will
send to the device later.

We will also be using the accelerator core with coordinates ``{0, 0}`` to do
the execution. This is the first core, in the top left of the die.

*Note*: Everything we do were is under the ``tt::ll_buda`` namespace. We have
a using directive for this: ``using namespace tt::ll_buda``.

.. code-block:: cpp

  tt_xy_pair core = {0, 0};

  DataMovementKernel *dram_copy_kernel = CreateDataMovementKernel(
      program,
      "programming_examples/kernels/dataflow/loopback_dram_copy.cpp",
      core,
      DataMovementProcessor::RISCV_0,
      NOC::RISCV_0_default
  );

When attaching the kernel to the program, we need to declare which processors
within the core we want to do the processing. We will use ``RISCV_0`` to start
with.

Create buffers in DRAM and L1
-----------------------------

Next, we need to declare buffers that we will use during execution. We will
need

* A DRAM buffer that will house input data
* An L1 buffer within the core itself that will be used to store the compute
  engine's work
* A DRAM buffer that will be written to with output data

for a total of 3. Let's create the L1 buffer first. We attach L1 buffer data to
program data.

.. code-block:: cpp

  constexpr uint32_t single_tile_size = 2 * (32 * 32);
  constexpr uint32_t num_tiles = 50;
  constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;
  constexpr uint32_t l1_buffer_addr = 400 * 1024;

  L1Buffer *l1_b0 = CreateL1Buffer(program, core, dram_buffer_size, l1_buffer_addr);

We need to know how large the buffer will be in bytes. Let's say we have 50
"tiles" worth.

Wait, what's a tile? In LL Buda, we generally structure blocks of data as
"tiles". Each tile contains ``32 x 32`` (or 1024) values of FP16 format. Since
FP16 is 2 bytes large, and each tile is 1024 values, 50 tiles is ``50 * (2 * 32
* 32)`` bytes large. For simplicity, let's make all our buffers this size. Our
L1 Buffer is therefore of this size, and will be housed at the ``400KB``
starting address in L1.

We'll make similarly-sized DRAM buffers for input and output data, at different
locations in DRAM. Note that we will be using DRAM channel 0 for now. On
Grayskull, we have 8 channels to use.

.. code-block:: cpp

  constexpr uint32_t input_dram_buffer_addr = 0;
  constexpr int dram_channel = 0;
  DramBuffer *input_dram_buffer = CreateDramBuffer(device, dram_channel, dram_buffer_size, input_dram_buffer_addr);

  constexpr uint32_t output_dram_buffer_addr = 512 * 1024;
  DramBuffer *output_dram_buffer = CreateDramBuffer(device, dram_channel, dram_buffer_size, output_dram_buffer_addr);

Now let's create some dummy data and send it into the input DRAM buffer!

Sending real data into DRAM
---------------------------

.. code-block:: cpp

  std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
      dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
  pass &= WriteToDeviceDRAM(device, input_dram_buffer, input_vec);

Note that for the data generation function to work, you need to include header
``"common/bfloat16.hpp"``.

Sending runtime arguments for the data movement kernel
------------------------------------------------------

Right after configuring the device with the program, now we'll need to tell the
data movement kernel its runtime arguments. Because our kernel is somewhat
generic, we've written it to have the following arguments:

* Where the L1 buffer starts (memory address)
* Where the input DRAM buffer starts (memory address)
* The location of the input DRAM buffer's channel on the NOC
* Where the output DRAM buffer starts (memory address)
* The location of the output DRAM buffer's channel on the NOC
* The size of the buffers

.. code-block:: cpp

  const tt_xy_pair input_dram_noc_xy = input_dram_buffer->noc_coordinates(device);
  const tt_xy_pair output_dram_noc_xy = output_dram_buffer->noc_coordinates(device);

  const std::vector<uint32_t> runtime_args = {
      l1_buffer_addr,
      input_dram_buffer_addr,
      (std::uint32_t)input_dram_noc_xy.x,
      (std::uint32_t)input_dram_noc_xy.y,
      output_dram_buffer_addr,
      (std::uint32_t)output_dram_noc_xy.x,
      (std::uint32_t)output_dram_noc_xy.y,
      dram_buffer_size
  };

  pass &= WriteRuntimeArgsToDevice(
      device,
      dram_copy_kernel,
      core,
      runtime_args
  );

Launch and verify output
------------------------

Now we just ``LaunchKernels`` and wait for it to finish. Then we can finally
read back the data from the output buffer and assert that it matches what we
sent!

.. code-block:: cpp

  std::vector<uint32_t> result_vec;
  ReadFromDeviceDRAM(device, output_dram_buffer, result_vec, output_dram_buffer->size());

  pass &= input_vec == result_vec;

Note that we have this exact, full example program in
``programming_examples/loopback/loopback.cpp``.
