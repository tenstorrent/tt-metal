# DRAM Loopback

We will build a program in TT-Metal that will simply copy data from one
DRAM buffer to another, using the compute engine and an intermediate L1
buffer to do so. We call this concept \"loopback\".

We\'ll go through this code section by section. Note that we have this exact, full example program in [loopback.cpp](../../../tt_metal/programming_examples/loopback/loopback.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/loopback
```

## Mesh setup

Create a 1x1 mesh device, obtain the mesh command queue, and construct a workload and coordinate range. The program is created later from a program spec.

```cpp
constexpr int device_id = 0;
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
```

## Create tensors in DRAM and L1

We allocate a single-tile L1 tensor and two DRAM tensors (each 50 tiles). Each row of a row-major tensor is one page, so a row of one tile's worth of elements gives a page size equal to one tile and transfers operate tile-by-tile.

```cpp
constexpr uint32_t num_tiles = 50;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

TensorSpec dram_tensor_spec(
    Shape({num_tiles, elements_per_tile}),
    TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
TensorSpec l1_tensor_spec(
    Shape({1, elements_per_tile}),
    TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));  // This time we allocate on L1

MeshTensor l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
MeshTensor input_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor output_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
```

## Data movement kernel

Describe the data movement kernel on core `{0,0}` that performs the copy. The kernel binds the program's tensor parameters `input` and `output` under the accessor names `src` and `dst`, and declares the names of its runtime arguments. A `ProgramSpec` collects the kernel, the tensor parameters, and the work unit (the cores the kernel runs on), and `MakeProgramFromSpec` creates the program.

```cpp
constexpr experimental::NodeCoord core = {0, 0};

const experimental::KernelSpecName dram_copy_kernel{"dram_copy"};
const experimental::TensorParamName input_tensor_param{"input"};
const experimental::TensorParamName output_tensor_param{"output"};

experimental::KernelSpec dram_copy_kernel_spec{
    .unique_id = dram_copy_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "loopback/kernels/loopback_dram_copy.cpp",
    .tensor_bindings =
        {{.tensor_parameter_name = input_tensor_param, .accessor_name = "src"},
         {.tensor_parameter_name = output_tensor_param, .accessor_name = "dst"}},
    .runtime_arg_schema = {.runtime_arg_names = {"l1_buffer_addr", "num_tiles"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};

experimental::ProgramSpec program_spec{
    .name = "loopback",
    .kernels = {dram_copy_kernel_spec},
    .tensor_parameters =
        {{.unique_id = input_tensor_param, .spec = dram_tensor_spec},
         {.unique_id = output_tensor_param, .spec = dram_tensor_spec}},
    .work_units = {{.name = "loopback", .kernels = {dram_copy_kernel}, .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

## Upload input data

Upload a randomly generated bfloat16 vector to the input DRAM tensor. The upload is non-blocking, to overlap with host setup.

```cpp
std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
// ... fill input_vec ...
HostTensor input_host_tensor = HostTensor::from_vector(input_vec, dram_tensor_spec);
cq.enqueue_write_tensor(input_host_tensor, input_dram_tensor);
```

## Set runtime arguments

Set the named runtime arguments of the kernel, and bind the DRAM tensors to the program's tensor parameters.

```cpp
experimental::ProgramRunArgs run_args{
    .kernel_run_args = {{
        .kernel = dram_copy_kernel,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            core,
            {{"l1_buffer_addr", static_cast<uint32_t>(l1_tensor.address())}, {"num_tiles", num_tiles}}),
    }},
    .tensor_args =
        {{input_tensor_param, experimental::TensorArgument{input_dram_tensor}},
         {output_tensor_param, experimental::TensorArgument{output_dram_tensor}}},
};
experimental::SetProgramRunArgs(program, run_args);
```

## Launch and wait

Enqueue the program as a mesh workload (non-blocking), then wait for completion.

```cpp
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
distributed::Finish(cq);
```

## Read back and verify

Read the output tensor back and compare with the input.

```cpp
std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(output_dram_tensor).to_vector<bfloat16>();
// compare result_vec to input_vec
```

Finally, close the mesh device after validation.
