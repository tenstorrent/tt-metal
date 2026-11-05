# Basic Compute Kernel

We will build a simple program in TT-Metal that will set up a void compute kernel and place it on a given device, core, and RISC-V processor. In this example, there will be nothing for the kernel to compute; the focus will be on the general setup for a compute kernel.

We'll go through this code section by section. The full example program is at [hello_world_compute_kernel.cpp](../../../tt_metal/programming_examples/hello_world_compute_kernel/hello_world_compute_kernel.cpp)

To build and execute, you may use the following commands:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_hello_world_compute_kernel
```

Before running, set `TT_METAL_DPRINT_CORES=0,0` to see the kernel prints.

## Mesh setup

Create a 1x1 mesh device, get the mesh command queue, and construct a workload and device range.

```cpp
constexpr experimental::NodeCoord core = {0, 0};
int device_id = 0;
std::shared_ptr<distributed::MeshDevice> mesh_device =
    distributed::MeshDevice::create_unit_mesh(device_id);  // 1x1 mesh
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
```

## Create the compute kernel

We describe a simple compute kernel which prints from the three compute RISC-V cores (UNPACK, MATH, PACK) with a `KernelSpec`. A `ProgramSpec` then lists the kernels and the work units (the cores the kernels run on), and `MakeProgramFromSpec` creates the program from it.

```cpp
const experimental::KernelSpecName void_compute_kernel{"void_compute_kernel"};
experimental::KernelSpec void_compute_kernel_spec{
    .unique_id = void_compute_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
    .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
    .hw_config = experimental::ComputeHardwareConfig{},
};

// A program spec describes the program: its kernels and the cores the kernels run on (the work units).
experimental::ProgramSpec program_spec{
    .name = "hello_world_compute_kernel",
    .kernels = {void_compute_kernel_spec},
    .work_units = {{.name = "hello_world_compute_kernel", .kernels = {void_compute_kernel}, .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

## Launch and wait

Set the program run args, enqueue the program in a mesh workload (non-blocking), print a host message, then wait for completion and close the device.

```cpp
experimental::ProgramRunArgs run_args{.kernel_run_args = {{.kernel = void_compute_kernel}}};
experimental::SetProgramRunArgs(program, run_args);
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
fmt::print("Hello, Core (0, 0) on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

distributed::Finish(cq);
printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
mesh_device->close();
```

## Expected output

With `TT_METAL_DPRINT_CORES=0,0`, the compute kernel prints from the UNPACK, MATH, and PACK cores on core `{0,0}`. The host prints the greeting and closing messages shown above.
