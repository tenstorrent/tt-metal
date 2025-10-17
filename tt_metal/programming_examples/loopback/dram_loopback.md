# DRAM Loopback

We will build a program in TT-Metal that will simply copy data from one
DRAM buffer to another, using the compute engine and an intermediate L1
buffer to do so. We call this concept \"loopback\".

We\'ll go through this code section by section. Note that we have this exact, full example program in [loopback.cpp](../../../tt_metal/programming_examples/loopback/loopback.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
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

## Create buffers in DRAM and L1

We allocate a single-tile L1 buffer and two DRAM buffers (each 50 tiles). Use a page size equal to one tile so transfers operate tile-by-tile.

```cpp
constexpr uint32_t num_tiles = 50;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

distributed::DeviceLocalBufferConfig l1_config{ .page_size = tile_size_bytes, .buffer_type = BufferType::L1 };
distributed::DeviceLocalBufferConfig dram_config{ .page_size = tile_size_bytes, .buffer_type = BufferType::DRAM };
distributed::ReplicatedBufferConfig l1_buffer_config{ .size = tile_size_bytes };
distributed::ReplicatedBufferConfig dram_buffer_config{ .size = dram_buffer_size };

auto l1_buffer = distributed::MeshBuffer::create(l1_buffer_config, l1_config, mesh_device.get());
auto input_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
auto output_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
```

## Data movement kernel

Declare the data movement kernel on core `{0,0}` that performs the copy.

```cpp
constexpr CoreCoord core = {0, 0};
std::vector<uint32_t> dram_copy_compile_time_args;
TensorAccessorArgs(*input_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
TensorAccessorArgs(*output_dram_buffer->get_backing_buffer()).append_to(dram_copy_compile_time_args);
auto dram_copy_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
    core,
    DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = dram_copy_compile_time_args });
```

## Upload input data

Upload a randomly generated bfloat16 vector to the input DRAM buffer. Use non-blocking upload to overlap with host setup.

```cpp
std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
// ... fill input_vec ...
distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, /*blocking=*/false);
```

## Set runtime arguments

```cpp
const std::vector<uint32_t> runtime_args = { l1_buffer->address(), input_dram_buffer->address(), output_dram_buffer->address(), num_tiles };
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

Read the output buffer back from the shard at `{0,0}` and compare with the input.

```cpp
std::vector<bfloat16> result_vec;
distributed::ReadShard(cq, result_vec, output_dram_buffer, distributed::MeshCoordinate(0, 0), /*blocking*/ true);
// compare result_vec to input_vec
```

Finally, close the mesh device after validation.
