# Basic Compute Kernel

We will build a simple program in TT-Metal that will set up a void compute kernel and place it on a given device, core, and RISC-V processor. In this example, there will be nothing for the kernel to compute; the focus will be on the general setup for a compute kernel.

We'll go through this code section by section. The full example program is at:
[tt_metal/programming_examples/hello_world_compute_kernel/hello_world_compute_kernel.cpp](../../../tt_metal/programming_examples/hello_world_compute_kernel/hello_world_compute_kernel.cpp)
so you can follow along.

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh
    ./build/programming_examples/hello_world_compute_kernel
```
## Silicon accelerator setup

``` cpp
constexpr CoreCoord core = {0, 0};
int device_id = 0;
Device *device = CreateDevice(device_id);
```

We instantiate a device object that will be used to interface with the designated `grayskull` accelerator. The core that we will be using for this example is represented by its coordinates `{0, 0}`. Note that logical coordinates are used to designate the core(s) that will be utilized for the program.

## Program pre-compilation setup

``` cpp
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
```

We first obtain the global `CommandQueue` associated with the device in order to use the fast dispatch capabilities of the software. Thism mechanism enables commands to be run asynchronously between the host and device.

Next, we create a `Program` to be run on our Grayskull accelerator. This object will encapsulate our data and kernels, and be dispatched through the `CommandQueue` to execute on the device.

## Building a compute kernel

Declare a void compute kernel to execute on the device. The kernel code for this example can be found in the file indicated in the code block below. The `ComputeConfig` object parameter indicates that we are initializing a compute kernel.

``` cpp
vector<uint32_t> compute_kernel_args = {};
KernelHandle void_compute_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_kernel_args
    }
);
```

Since we are simply demonstrating how to set up a void compute kernel, we do not need to supply any compile-time arguments to our kernel function.

## Compute kernel function

``` cpp
void MAIN {

    // Nothing to compute. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    DPRINT_MATH(DPRINT << "Hello, Master, I am running a void compute kernel." << ENDL());

}
```

Our kernel function, which is defined in [tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp](../../../tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp),
contains just a single `DPRINT` statement to indicate that the compute kernel has executed. Note that in order to print from the compute kernel, `DPRINT_MATH()` must be used.

## Configure and execute program on device

The next step will be to set the runtime arguments for the program using
`SetRuntimeArgs()`, then run it. This function allows us to also specify which cores will receive these kernel arguments for the given program.
`EnqueueProgram()` will then send the program to the device for execution (the `false` parameter indicates that the operation is not blocking.)

``` cpp
SetRuntimeArgs(program, void_compute_kernel_id, core, {});
EnqueueProgram(cq, program, false);
printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");
```

We then wait for all programs dispatched by the command queue to be executed before closing the device. `Finish()` will block all commands until the dispatched commands have been completed.

``` cpp
Finish(cq);
printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
CloseDevice(device);
```

## Expected Output

If executed correctly, we should have the following output. Make sure to run the command `export TT_METAL_DPRINT_CORES=0,0` in order to view the kernel's output statement.

    Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.
    Hello, Master, I am running a void compute kernel.
    Thank you, Core {0, 0} on Device 0, for the completed task.

## Summary

The following lays out the general workflow for setting up a host program that runs a basic compute kernel.

1. Specify the device and the coordinates of the cores that will be utilized.
2. Obtain the command queue and create the program that will be executed.
3. Specify the compute kernel configuration and create it.
4. Set up the runtime arguments for the compute kernel and launch the program.
5. Wait for the program to finish execution.
