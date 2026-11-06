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

Create a 1x1 mesh device, obtain the mesh command queue, construct a workload and coordinate range, and create a program.

```cpp
constexpr int device_id = 0;
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program = CreateProgram();
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

Declare the data movement kernel on core `{0,0}` that performs the copy.

```cpp
constexpr CoreCoord core = {0, 0};
std::vector<uint32_t> dram_copy_compile_time_args;
TensorAccessorArgs(input_dram_tensor).append_to(dram_copy_compile_time_args);
TensorAccessorArgs(output_dram_tensor).append_to(dram_copy_compile_time_args);
auto dram_copy_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
    core,
    DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = dram_copy_compile_time_args });
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

```cpp
const std::vector<uint32_t> runtime_args = { l1_tensor.address(), input_dram_tensor.address(), output_dram_tensor.address(), num_tiles };
SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);
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
