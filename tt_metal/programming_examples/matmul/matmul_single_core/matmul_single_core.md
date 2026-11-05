# Matmul (Single Core)

We'll build a program that will perform matmul operations on two tensors with equal-size inner dimension. We will then go through specific sections of the program.

The full example program is in
[matmul_single_core.cpp](../../../tt_metal/programming_examples/matmul_single_core/matmul_single_core.cpp)

To build and execute, you may use the following commands:
Then run the following:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_matmul_single_core
```

## Mesh setup (host)

We use the Mesh API: create a 1x1 mesh device, get the mesh command queue, construct a workload and device range, and create a program. Then prepare inputs, tilize, call `matmul_single_core`, untilize, and validate.

## Main blocks in matmul_single_core function

We will go through sections of the `matmul_single_core` function:

-   Program, enqueue and core range settings
-   Create DRAM tensors based on input and output vectors
-   Create L1 Circular buffers
-   Kernels declarations and related compile and runtime arguments
-   Program launch and reading data from DRAM output tensor to result vector

## Create Program, Enqueue initialization, and core range definition

We want a just a single core, so we will restrict the core range to be just one core at (0, 0).

``` cpp
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program{};
CoreCoord core({0, 0});
```

## Create DRAM tensors & Circular buffers

In terms of DRAM tensors, we need two source tensors and one destination tensor.
Each row of a row-major tensor is one page, so a row of one tile's worth of elements gives a page size of one tile.

``` cpp
// MN = MK*KN
uint32_t Mt = M / TILE_HEIGHT;
uint32_t Kt = K / TILE_WIDTH;
uint32_t Nt = N / TILE_WIDTH;

DataFormat cb_data_format = DataFormat::Float16_b;
uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
uint32_t elements_per_tile = single_tile_size / sizeof(bfloat16);
tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;

auto make_dram_tensor_spec = [&](size_t num_elements) {
    return TensorSpec(
        Shape({static_cast<uint32_t>(num_elements / elements_per_tile), elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
};
TensorSpec src0_tensor_spec = make_dram_tensor_spec(a.size());
TensorSpec src1_tensor_spec = make_dram_tensor_spec(b.size());
TensorSpec dst_tensor_spec = make_dram_tensor_spec(output.size());

MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src0_tensor_spec);
MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src1_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);
uint32_t src0_addr = src0_dram_tensor.address();
uint32_t src1_addr = src1_dram_tensor.address();
uint32_t dst_addr = dst_dram_tensor.address();

uint32_t src0_cb_index = CBIndex::c_0; //0
uint32_t num_input_tiles = 2;
tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
    .set_page_size(src0_cb_index, single_tile_size);
auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

uint32_t src1_cb_index = CBIndex::c_1; // 1
tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
    .set_page_size(src1_cb_index, single_tile_size);
auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

uint32_t output_cb_index = tt::CBIndex::c_16;
uint32_t num_output_tiles = 2;
tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
    .set_page_size(output_cb_index, single_tile_size);
auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
```

## Compile-time kernels arguments

We have to declare some compile-time arguments for read/write kernels.
Some default parameters here will suffice.

``` cpp
vector<uint32_t> compute_args = {
    Mt, // Mt
    Kt, // Kt
    Nt // Nt
};
```

## Compute kernel declaration and compile-time defines

We're using a special reader kernel to take in data from DRAM into L1, and a special writer kernel to write out results from the compute engine back to the destination DRAM tensor.

``` cpp
auto reader_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
    core,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

auto writer_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
    core,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

auto matmul_single_core_kernel_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp",
    core,
    tt_metal::ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .compile_args = compute_args}
);
```

## Runtime arguments and program launch

We will now set runtime arguments for the reader and writer kernels to run the matmul operation on a single core and a single tile at a time.

``` cpp
tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});

tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Kt, Nt});
```

Launch program, enqueue & read in output tensor result into the host vector.

``` cpp
HostTensor src0_host_tensor = HostTensor::from_vector(a, src0_tensor_spec);
HostTensor src1_host_tensor = HostTensor::from_vector(b, src1_tensor_spec);
cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
output = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
```

## Conclusion

Those are the additional steps for getting `matmul_single_core` operations up and running on the compute engine. To see a more complicated example using as many cores as possible, please refer to the
`Matmul multi-core` example.
