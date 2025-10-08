# Eltwise binary


We now build a program that will perform eltwise binary operations on a some equal-sized tensors.

We'll go through any new code section by section. This builds on top of previous examples. Note that we have this exact, full example program in [eltwise_binary.cpp](../../../tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_eltwise_binary
```
## New buffers

In terms of DRAM buffers, We just need a new buffer for a 2nd source, because we have two source tensors (vectors).

## Mesh Setup

- **Mesh device**: Create a 1x1 mesh on device 0. The same API scales to larger meshes.
- **Command queue**: All uploads, downloads, and program executions are enqueued here.
- **Workload and range**: A workload wraps one or more programs and a device coordinate range to run them on.

```cpp
constexpr int device_id = 0;
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```

### DRAM buffers (replicated)

We allocate three DRAM-backed buffers: two inputs and one output. The `ReplicatedBufferConfig` means each device in the mesh gets its own identical allocation. On a unit mesh, this is just one device allocation. Set `page_size` to a single-tile size so NoC transfers operate tile-by-tile.

```cpp
constexpr uint32_t n_tiles = 64;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

distributed::DeviceLocalBufferConfig dram_config{
    .page_size = tile_size_bytes,
    .buffer_type = BufferType::DRAM};
distributed::ReplicatedBufferConfig buffer_config{ .size = n_tiles * tile_size_bytes };

auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
auto dst_dram_buffer  = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
```

Upload host data into the input buffers:

```cpp
std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(-1.0f));
// ... fill a_data with random values ...
distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a_data, false);
distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b_data, false);
```

### Circular buffers (L1 FIFOs)

Create three L1-backed circular buffers to move tiles between kernels. Use two entries per CB for double-buffering to overlap producer and consumer work.

```cpp
constexpr uint32_t tiles_per_cb = 2;
tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
CreateCircularBuffer(program, core, CircularBufferConfig(
    tiles_per_cb * tile_size_bytes,
    {{src0_cb_index, tt::DataFormat::Float16_b}})
    .set_page_size(src0_cb_index, tile_size_bytes));

tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
CreateCircularBuffer(program, core, CircularBufferConfig(
    tiles_per_cb * tile_size_bytes,
    {{src1_cb_index, tt::DataFormat::Float16_b}})
    .set_page_size(src1_cb_index, tile_size_bytes));

tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
CreateCircularBuffer(program, core, CircularBufferConfig(
    tiles_per_cb * tile_size_bytes,
    {{dst_cb_index, tt::DataFormat::Float16_b}})
    .set_page_size(dst_cb_index, tile_size_bytes));
```

### Compute kernel declaration and compile-time defines

We instantiate three kernels:
- Reader (RISCV_0): reads DRAM tiles into `cb_src0` and `cb_src1`
- Compute: pops two tiles, adds them, pushes result to `cb_dst`
- Writer (RISCV_1): writes tiles from `cb_dst` back to DRAM

```cpp
std::vector<uint32_t> reader_args;
TensorAccessorArgs(*src0_dram_buffer).append_to(reader_args);
TensorAccessorArgs(*src1_dram_buffer).append_to(reader_args);
auto reader = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp",
    core,
    DataMovementConfig{ .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args });

std::vector<uint32_t> writer_args;
TensorAccessorArgs(*dst_dram_buffer).append_to(writer_args);
auto writer = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp",
    core,
    DataMovementConfig{ .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args });

auto compute = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp",
    core,
    ComputeConfig{ .math_fidelity = MathFidelity::HiFi4 });

SetRuntimeArgs(program, reader,  core, { src0_dram_buffer->address(), src1_dram_buffer->address(), n_tiles });
SetRuntimeArgs(program, writer,  core, { dst_dram_buffer->address(), n_tiles });
SetRuntimeArgs(program, compute, core, { n_tiles });
```

In this program, we have a second source tensor. We will be adding this to the first source tensor.

### Launch and read back

Enqueue the program as a mesh workload. Use non-blocking enqueue and explicitly wait with `Finish(cq)` before reading results.

```cpp
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

Read back from the shard at mesh coordinate `{0, 0}` (on a unit mesh) and validate.

```cpp
std::vector<bfloat16> result_vec;
distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0), true);
```

In this program, we have a second source tensor. We will be adding this to the first source tensor.

## Conclusion

Those are the additional steps for getting eltwise binary operations upmand running on the compute engine. We essentially repeat the same process to chain together two operations, with one DRAM read in the middle to get the intermediate result and hold it in a DRAM buffer. For an example involving matrix multiplication on a single core, please refer to the `Matmul single core` example.
