# Eltwise SFPU

We now build a program that will perform an eltwise SFPU unary operation on a single tensor.

We'll go through any new code section by section. This builds on top of previous examples. Note that we have this exact, full example program in [eltwise_sfpu.cpp](../../../tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
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

## DRAM tensors (replicated)

Allocate one input and one output DRAM-backed `MeshTensor`. Each tensor is replicated across the mesh. On a unit mesh this is a single device allocation. Each row of the row-major `TensorSpec` holds one tile's worth of elements, so the page size is a single tile and transfers happen tile-by-tile.

```cpp
constexpr uint32_t n_tiles = 64;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

TensorSpec dram_tensor_spec(
    Shape({n_tiles, elements_per_tile}),
    TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
```

Upload host data into the input tensor (non-blocking to overlap with host setup):

```cpp
std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
// ... fill src0_vec with random values ...
HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
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
TensorAccessorArgs(src0_dram_tensor).append_to(reader_compile_time_args);
auto unary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = reader_compile_time_args});

std::vector<uint32_t> writer_compile_time_args;
TensorAccessorArgs(dst_dram_tensor).append_to(writer_compile_time_args);
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
    ComputeConfig{ .math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .math_approx_mode = false });

SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, { n_tiles });
SetRuntimeArgs(program, unary_reader_kernel_id, core, { src0_dram_tensor.address(), n_tiles });
SetRuntimeArgs(program, unary_writer_kernel_id, core, { dst_dram_tensor.address(), n_tiles });
```

## Launch and read back

Enqueue the program in a mesh workload (non-blocking), then wait for completion and read back results from the output tensor.

```cpp
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);

std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
```

Finally, close the mesh device after validation.
