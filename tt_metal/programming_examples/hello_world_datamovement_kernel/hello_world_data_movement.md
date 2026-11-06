# Basic Data Movement Kernel

This example will build a simple program in TT-Metal that will demonstrate how to set up a data movement kernel and place it on a given device, core, and RISC-V processor.

We'll go through this code section by section. Note that we have this exact, full example program in
[hello_world_datamovement_kernel.cpp](../../../tt_metal/programming_examples/hello_world_datamovement_kernel/hello_world_datamovement_kernel.cpp),
so you can follow along.

To build and execute, you may use the following commands:
Then run the following:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_hello_world_datamovement_kernel
```

## Mesh setup

Create a 1x1 mesh device, get the mesh command queue, and construct a workload and device range.

```cpp
constexpr experimental::NodeCoord core = {0, 0};
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
```

## Data movement kernels

There are two Data Movement cores per Tensix. For this example, we launch two identical kernels, one on each DM core (RISCV_0 on NOC 0 and RISCV_1 on NOC 1). Each kernel is a `KernelSpec` with the same kernel source; each one names its own Data Movement core and NoC.

```cpp
const experimental::KernelSpecName data_movement_kernel_0{"data_movement_kernel_0"};
const experimental::KernelSpecName data_movement_kernel_1{"data_movement_kernel_1"};

experimental::KernelSpec data_movement_kernel_0_spec{
    .unique_id = data_movement_kernel_0,
    .source = OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};

experimental::KernelSpec data_movement_kernel_1_spec{
    .unique_id = data_movement_kernel_1,
    .source = OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        },
};
```

A `ProgramSpec` lists both kernels and one work unit that runs them on core `{0, 0}`. `MakeProgramFromSpec` creates the program from it.

```cpp
experimental::ProgramSpec program_spec{
    .name = "hello_world_datamovement_kernel",
    .kernels = {data_movement_kernel_0_spec, data_movement_kernel_1_spec},
    .work_units =
        {{.name = "hello_world_datamovement_kernel",
          .kernels = {data_movement_kernel_0, data_movement_kernel_1},
          .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

Set the program run args (no runtime arguments needed here), enqueue the program in a mesh workload (non-blocking), and wait for completion.

```cpp
experimental::ProgramRunArgs run_args{
    .kernel_run_args = {{.kernel = data_movement_kernel_0}, {.kernel = data_movement_kernel_1}},
};
experimental::SetProgramRunArgs(program, run_args);
std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
          << std::endl;

workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

Finally, close the device after the expected prints from both DM cores.

## Expected output

The program prints NC and BR messages from DM cores 1 and 0 on core `{0,0}`, followed by the host closing message as shown in the C++ example.
