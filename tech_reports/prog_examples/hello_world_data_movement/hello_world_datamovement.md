---
title: Basic Data Movement Kernel
---

This example will build a simple program in TT-Metal that will
demonstrate how to set up a data movement kernel and place it on a given
device, core, and RISC-V processor.

We\'ll go through this code section by section. Note that we have this
exact, full example program in
`tt_metal/programming_examples/hello_world_datamovement_kernel/hello_world_datamovement_kernel.cpp`,
so you can follow along.

To build and execute, you may use the following commands. Note that we
include the necessary environment variables here, but you may possibly
need more depending on the most up-to-date installation methods.

    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh --build-tests
    ./build/programming_examples/hello_world_datamovement_kernel

# Silicon accelerator setup

``` cpp
constexpr CoreCoord core = {0, 0};
int device_id = 0;
Device *device = CreateDevice(device_id);
```

The initial setup for this example follows the same pattern for the
compute kernel example. We instantiate a device object to utilize the
hardware on the `grayskull` accelerator. For this example, we will be
using a single core to focus solely on the data movement kernel setup.

# Program pre-compilation setup

``` cpp
CommandQueue& cq = device->command_queue();
auto program = CreateProgram();
```

Then, we obtain the device\'s `CommandQueue` in order to allow commands
to be dispatched for execution. The `Program` is initialized to
encapsulate the kernels and data.

# Building data movement kernels

We will declare two data movement kernels, one for each of the two
RISC-V processors involved in data movement (`RISCV_0` for RISC-V 1 and
`RISCV_1` for RISC-V 5). The `DataMovementConfig` parameter specifies
the configuration for data movement.

``` cpp
KernelHandle void_dataflow_kernel_noc0_id = CreateKernel(
    program,
    "tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

KernelHandle void_dataflow_kernel_noc1_id = CreateKernel(
    program,
    "tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
```

Since this example focuses on data movement kernel setup, we do not need
to specify any compile-time arguments.

# Data movement kernel function

``` cpp
void kernel_main() {

    // Nothing to move. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
    DPRINT_DATA1(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 1." << ENDL());

}
```

The kernel function, defined in
`tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp`,
simply prints two statements from each of the RISC-V processors involved
with data movement. This kernel is designed to be basic in order to
focus on the data movement kernel setup itself.

# Configure and execute program on device

We set the runtime arguments for each kernel separately, before
dispatching the program to the `CommandQueue` for execution on the
device using `EnqueueProgram()`. Both kernels will run on core `{0, 0}`,
each utilizing different NoC systems.

``` cpp
SetRuntimeArgs(program, void_dataflow_kernel_noc0_id, core, {});
SetRuntimeArgs(program, void_dataflow_kernel_noc1_id, core, {});
EnqueueProgram(cq, program, false);
printf("Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication.\n");
```

We then wait for the program to finish execution with `Finish()`.

``` cpp
Finish(cq);
printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
CloseDevice(device);
```

If the output includes a message from each kernel, then it has been
executed correctly. Make sure to run the command
`export TT_METAL_DPRINT_CORES=0,0` in order to view the second output
statement.

# Expected output

    Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication.
    Hello, Master, I am running a void data movement kernel on NOC 0.
    Hello, Master, I am running a void data movement kernel on NOC 1.
    Thank you, Core {0, 0} on Device 0, for the completed task.

As shown in the kernel function, there will be two statements printed
from the kernel if the example code is executed correctly. Note that
`DPRINT_DATA0()` is used to print messages from RISC-V processor 1 and
`DPRINT_DATA1()` is used to print messages from RISC-V processor 5.

# Summary

The following lays out the general workflow for setting up a host
program that runs a basic data movement kernel.

1.  1.  Specify the device and the coordinates of the cores that will be
        utilized.
2.  2.  Configure and create the collaboration mechanisms (e.g. command
        queue).
3.  3.  Create the program that will contain the kernels.
4.  3.  Specify the data movement kernel configuration and create the
        kernels.
5.  4.  Set up the runtime arguments for the data movement kernel and
        launch the program.
6.  5.  Wait for the program to finish execution before closing the
        device.
