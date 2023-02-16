.. _Setting up a basic test:

Setting up a basic test
=======================

.. code-block:: cpp

   #include "ll_buda/host_api.hpp"

   using namespace tt::ll_buda;

   int main(int argc, char **argv) {
       bool pass = true;

       try {
           constexpr int pci_express_slot = 0;
           Device *device =
               CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

           pass &= InitializeDevice(device);

           Program *program = new Program();

           constexpr bool skip_hlkc = false;
           pass &= CompileProgram(device, program, skip_hlkc);

           pass &= ConfigureDeviceWithProgram(device, program);

           pass &= LaunchKernels(device, program);

           pass &= CloseDevice(device);

       } catch (const std::exception &e) {
           tt::log_error(tt::LogTest, "Test failed with exception!");
           tt::log_error(tt::LogTest, "{}", e.what());

           throw;
       }

       if (pass) {
           tt::log_info(tt::LogTest, "Test Passed");
       } else {
           tt::log_fatal(tt::LogTest, "Test Failed");
       }

       return 0;
   }

The most basic LL Buda test would be to open a single accelerator device and
run some blank kernels. These kernels contain basic RISC-V programs that do
nothing. Each step using the API will return a bool indicating a successful or
failed call. We can use that to ensure that we pass every step.

Let's go through each step to get a better handle of everything.

Silicon accelerator setup
-------------------------

.. code-block:: cpp

   constexpr int pci_express_slot = 0;
   Device *device =
       CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

   pass &= InitializeDevice(device);

We instantiate a device object and direct it to use the first PCI Express slot
where our Tenstorrent accelerator is installed. It's a ``GRAYSKULL`` type
device.  This will be the key API entrypoint to the silicon. We then initialize
it with its default parameters which does some housekeeping for low-level
settings.

Program pre-compilation setup
-----------------------------

.. code-block:: cpp

   Program *program = new Program();

Here, we create a ``Program`` to be run on our Grayskull accelerator. It's at
this point we can attach some kernels to this program to

* move data from device DRAM storage to the device's compute engine memory
* perform specific math operations on the device's compute engine
* move data from result buffers back to DRAM storage for later reading by host

and more. We won't attach anything here for now because we want to run blank programs.

Program compilation
-------------------

.. code-block:: cpp

   constexpr bool skip_hlkc = false;
   pass &= CompileProgram(device, program, skip_hlkc);

Here, we compile our program of empty kernels. If we attached any data movement
or compute kernels, those kernels would be appropriate compiled with any
special settings we declared.

Loading the program with desired settings
-----------------------------------------

.. code-block:: cpp

   pass &= ConfigureDeviceWithProgram(device, program);

We now have our compiled kernels! All that's left is to load it up onto the
accelerator with any appropriate kernel arguments and any input data. Because
we have blank kernels, we have no arguments or data to supply. We simply
configure the accelerator with the program. Get ready to launch!

Running the program
-------------------

.. code-block:: cpp

   pass &= LaunchKernels(device, program);

Let's fire up our kernels! A simple call to ``LaunchKernels`` will do this for
us. This is a blocking call that will finish once the accelerator finishes
executing the loaded program with the arguments you specified in the previous
step (if you had any to supply).

Validation and teardown
-----------------------

.. code-block:: cpp

   pass &= CloseDevice(device);

Once the program finishes executing, we have the opportunity to read any
resulting data on the device from any appropriate output DRAM buffers. You can
do things such as inspect this data, assert its correctness etc.

We now use ``CloseDevice`` to teardown our connection to the Tenstorrent
device.

One can consider this the "Hello World!" of LL Buda. We run no data movement or
math kernels. We also don't create any buffers to link data from the host to
the device, or between cores on the device. However, this still represents a
working program on a Tenstorrent accelerator.

Our next steps will be trying to move some data from the host to the device,
and instructing the device to move around the data to specific locations in
memory. Understanding this will be a key building block to understanding the
memory model of LL Buda. We will jump into this first with a simple :ref:`DRAM
loopback test<DRAM Loopback Example>`.

Note that we have this exact, full example program in
``programming_examples/basic_empty_program/basic_empty_program.cpp``.
