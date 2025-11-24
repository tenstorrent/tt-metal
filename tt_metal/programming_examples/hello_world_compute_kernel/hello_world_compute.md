# Basic Compute Kernel

We will build a simple program in TT-Metal that will set up a void compute kernel and place it on a given device, core, and RISC-V processor. In this example, there will be nothing for the kernel to compute; the focus will be on the general setup for a compute kernel.

We'll go through this code section by section. The full example program is at [hello_world_compute_kernel.cpp](../../../tt_metal/programming_examples/hello_world_compute_kernel/hello_world_compute_kernel.cpp)

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_hello_world_compute_kernel
```

Before running, set `TT_METAL_DPRINT_CORES=0,0` to see the kernel prints.

## Mesh setup

Create a 1x1 mesh device, get the mesh command queue, construct a workload and device range, and create a program.

```cpp
constexpr CoreCoord core = {0, 0};
int device_id = 0;
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program = CreateProgram();
```

## Create the compute kernel

We create a simple compute kernel which prints from the three compute RISC-V cores (UNPACK, MATH, PACK).

```cpp
KernelHandle void_compute_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
    core,
    ComputeConfig{});
```

## Launch and wait

Set runtime args, enqueue the program in a mesh workload (non-blocking), print a host message, then wait for completion and close the device.

```cpp
SetRuntimeArgs(program, void_compute_kernel_id, core, {});
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
fmt::print("Hello, Core (0, 0) on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

distributed::Finish(cq);
printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
mesh_device->close();
```

## Expected output

With `TT_METAL_DPRINT_CORES=0,0`, the compute kernel prints from the UNPACK, MATH, and PACK cores on core `{0,0}`. The host prints the greeting and closing messages shown above.
