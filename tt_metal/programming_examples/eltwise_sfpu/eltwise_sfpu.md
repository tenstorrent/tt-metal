# Eltwise SFPU

We now build a program that will perform an eltwise SFPU unary operation on a single tensor.

We'll go through any new code section by section. This builds on top of previous examples. Note that we have this exact, full example program in [eltwise_sfpu.cpp](../../../tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
export TT_METAL_HOME=$(pwd)
./build_metal.sh --build-programming-examples
./build/programming_examples/metal_example_eltwise_sfpu
```

## Mesh setup

- Create a 1x1 mesh on device 0 and obtain the mesh command queue.
- Create a workload and a coordinate range spanning the mesh.

```cpp
constexpr int device_id = 0;
auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```

## DRAM buffers (replicated)

Allocate one input and one output DRAM-backed buffer. The replicated config allocates identical buffers on each device. On a unit mesh this is a single device allocation. Set the page size to a single-tile size so transfers happen tile-by-tile.

```cpp
constexpr uint32_t n_tiles = 64;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

distributed::DeviceLocalBufferConfig dram_config{
    .page_size = tile_size_bytes,
    .buffer_type = tt_metal::BufferType::DRAM};
distributed::ReplicatedBufferConfig buffer_config{ .size = tile_size_bytes * n_tiles };

auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
auto dst_dram_buffer  = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
```

Upload host data into the input buffer (non-blocking to overlap with host setup):

```cpp
std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
// ... fill src0_vec with random values ...
distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);
```

## Circular buffers (L1 FIFOs)

Create two L1 circular buffers: one for input tiles to the compute kernel and one for output tiles from compute to writer. Use two entries for double-buffering.

```cpp
constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
constexpr uint32_t num_input_tiles = 2;
CircularBufferConfig cb_src0_config =
    CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, tile_size_bytes);
tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
CircularBufferConfig cb_output_config =
    CircularBufferConfig(num_input_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(output_cb_index, tile_size_bytes);
tt_metal::CreateCircularBuffer(program, core, cb_output_config);
```

## Kernels and runtime arguments

Instantiate data-movement and compute kernels, then set runtime arguments. The reader pulls tiles from DRAM into `cb_src0`; compute applies SFPU exp and pushes to `cb_output`; writer flushes tiles back to DRAM.

```cpp
std::vector<uint32_t> reader_compile_time_args;
TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
auto unary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

std::vector<uint32_t> writer_compile_time_args;
TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
auto unary_writer_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/write_tile.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = writer_compile_time_args});

auto eltwise_sfpu_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
    core,
    ComputeConfig{ .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false });

SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, { n_tiles });
SetRuntimeArgs(program, unary_reader_kernel_id, core, { src0_dram_buffer->address(), n_tiles });
SetRuntimeArgs(program, unary_writer_kernel_id, core, { dst_dram_buffer->address(), n_tiles });
```

## Launch and read back

Enqueue the program in a mesh workload (non-blocking), then wait for completion and read back results from shard `{0, 0}`.

```cpp
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);

std::vector<bfloat16> result_vec;
distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0), true);
```

Finally, close the mesh device after validation.
