# Basic Data Movement Kernel

This example will build a simple program in TT-Metal that will demonstrate how to set up a data movement kernel and place it on a given device, core, and RISC-V processor.

We'll go through this code section by section. Note that we have this exact, full example program in
[hello_world_datamovement_kernel.cpp](../../../tt_metal/programming_examples/hello_world_datamovement_kernel/hello_world_datamovement_kernel.cpp),
so you can follow along.

To build and execute, you may use the following commands:
Then run the following:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_hello_world_datamovement_kernel
```

## Mesh setup

Create a 1x1 mesh device, get the mesh command queue, construct a workload and device range, and create a program.

```cpp
constexpr CoreCoord core = {0, 0};
auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program = CreateProgram();
```

## Data movement kernels

There are two Data Movement cores per Tensix. For this example, we launch two identical kernels, one on each DM core (RISCV_0 on NOC 0 and RISCV_1 on NOC 1).

```cpp
KernelHandle void_dataflow_kernel_noc0_id = CreateKernel(
    program,
    "tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default });

KernelHandle void_dataflow_kernel_noc1_id = CreateKernel(
    program,
    "tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
    core,
    DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default });
```

Set runtime arguments (none needed here), enqueue the program in a mesh workload (non-blocking), and wait for completion.

```cpp
SetRuntimeArgs(program, void_dataflow_kernel_noc0_id, core, {});
SetRuntimeArgs(program, void_dataflow_kernel_noc1_id, core, {});
std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication." << std::endl;

workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

Finally, close the device after the expected prints from both DM cores.

## Expected output

The program prints NC and BR messages from DM cores 1 and 0 on core `{0,0}`, followed by the host closing message as shown in the C++ example.
