# Add 2 Integers in Compute Kernel

In this example, we will build a TT-Metal program that will add two vectors containing integers together, using data movement and compute kernels.

This program can be found in
[add_2_integers_in_compute.cpp](../../../tt_metal/programming_examples/add_2_integers_in_compute/add_2_integers_in_compute.cpp).

To build and execute, you may use the following commands:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_add_2_integers_in_compute
```
## Set up device and program/collaboration mechanisms

``` cpp
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
constexpr experimental::NodeCoord core = {0, 0};
```

We follow the standard procedure for the initial steps in setting up the host workload. The device that the workload will execute on is identified, and the corresponding command queue is accessed. The workload is initialized, and the core indicated for utilization in this example is at the coordinates `{0, 0}` in accordance with the logical mesh layout. The program is created later from a program spec.

## Configure and initialize DRAM tensors

``` cpp
constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
constexpr uint32_t single_tile_size = sizeof(bfloat16) * n_elements_per_tile;
TensorSpec dram_tensor_spec(
    Shape({1, n_elements_per_tile}),
    TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));  // Type of buffer (DRAM or L1(SRAM))
```

We define the tile size to fit BFloat16 values before setting up the configuration for the DRAM tensors. Each tile is 32x32 = 1024 bytes; doubling this allows us to tile up BFloat16 values. We specify the data type, the page layout and the type of the buffer in the `TensorSpec`. Each row of a row-major tensor is one page, so a row of one tile's worth of elements gives a page size of one tile. Our DRAM configuration will be interleaved for this example, which makes the data layout row-based. A `MeshTensor` is replicated across all devices in the mesh, which allows the user to seamlessly port a tensor configuration across an arbitrary-sized mesh of devices. Note that our choice of data format and buffer configuration has significant impact on the performance of the application, as we are able to reduce data traffic by packing values.

Next, we allocate a tensor with the specified configuration for each of the input vectors and another tensor for the output vector. The source data will be sent to the corresponding DRAM tensors to be accessed by the cores, and the results of the computation will be sent to the DRAM to be read by the destination vector.

``` cpp
MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
```

## Configure and Initialize Dataflow Buffers

``` cpp
const experimental::DFBSpecName src0_dfb{"src0"};
const experimental::DFBSpecName src1_dfb{"src1"};
const experimental::DFBSpecName dst_dfb{"dst"};

constexpr uint32_t num_tiles = 1;
auto make_dfb_spec = [&](const experimental::DFBSpecName& dfb_name) {
    return experimental::DataflowBufferSpec{
        .unique_id = dfb_name,
        .entry_size = single_tile_size,
        .num_entries = num_tiles,
        .data_format_metadata = DataFormat::Float16_b,
    };
};
```

L1 dataflow buffers will be used communicate data to and from the compute engine. We describe dataflow buffers for the source vectors and destination vector with `DataflowBufferSpec`s, which are added to the program spec below. The source data will be sent from the DRAM tensors to the dataflow buffer of each specified core, then the results for a given core will be stored in another dataflow buffer before being sent to DRAM.

## Kernel setup

``` cpp
const experimental::KernelSpecName binary_reader_kernel{"binary_reader"};
const experimental::KernelSpecName unary_writer_kernel{"unary_writer"};
const experimental::KernelSpecName eltwise_binary_kernel{"eltwise_binary"};

experimental::KernelSpec binary_reader_kernel_spec{
    .unique_id = binary_reader_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
    .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
    .runtime_arg_schema = {.runtime_arg_names = {"in0_addr", "in1_addr"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        },
};

experimental::KernelSpec unary_writer_kernel_spec{
    .unique_id = unary_writer_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
    .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
    .runtime_arg_schema = {.runtime_arg_names = {"dst_addr"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};
```

Data movement kernels are used for reading to and writing from the DRAM.
A `KernelSpec` is described for each of these operations, with a unique RISC-V processor assigned to each kernel. These kernels will read the data from the DRAM tensors into the dataflow buffers prior to the addition operation, then write the output data to the DRAM from the dataflow buffers so that they may be accessed by the host. Each kernel binds a dataflow buffer as its producer or consumer under a local name (for example `"in0"`), which the kernel source uses to access it, and declares the names of its runtime arguments.

``` cpp
experimental::KernelSpec eltwise_binary_kernel_spec{
    .unique_id = eltwise_binary_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
    .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
    .dfb_bindings =
        {experimental::ConsumerOf(src0_dfb, "in0"),
         experimental::ConsumerOf(src1_dfb, "in1"),
         experimental::ProducerOf(dst_dfb, "out0")},
    .hw_config =
        experimental::ComputeHardwareConfig{
            .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .sfpu_precision_mode = Precision::Precise,
            .enable_32_bit_dest = false,
        },
};

experimental::ProgramSpec program_spec{
    .name = "add_2_integers_in_compute",
    .kernels = {binary_reader_kernel_spec, unary_writer_kernel_spec, eltwise_binary_kernel_spec},
    .dataflow_buffers = {make_dfb_spec(src0_dfb), make_dfb_spec(src1_dfb), make_dfb_spec(dst_dfb)},
    .work_units =
        {{.name = "add_2_integers_in_compute",
          .kernels = {binary_reader_kernel, unary_writer_kernel, eltwise_binary_kernel},
          .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

In addition to the data movement kernels, we need to describe a compute kernel for the addition operation. We use the kernel code for adding 2 tiles as specified in the above code block. The kernel function will use the data provided in the dataflow buffers for the computation. The `ProgramSpec` collects the kernels, the dataflow buffers between them, and the work unit (the cores the kernels run on), and `MakeProgramFromSpec` creates the `Program` from it.

## Program execution

``` cpp
std::vector<bfloat16> src0_vec(n_elements_per_tile);
std::vector<bfloat16> src1_vec(n_elements_per_tile);
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> dist1(0.0f, 14.0f);
std::uniform_real_distribution<float> dist2(0.0f, 8.0f);
for (size_t i = 0; i < n_elements_per_tile; ++i) {
    src0_vec[i] = bfloat16(dist1(rng));
    src1_vec[i] = bfloat16(dist2(rng));
}

HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
HostTensor src1_host_tensor = HostTensor::from_vector(src1_vec, dram_tensor_spec);
cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
```

Next, we create two source vectors, each loaded with random values, before queueing the command to feed them to the corresponding DRAM tensors using `enqueue_write_tensor`.

``` cpp
experimental::ProgramRunArgs run_args{
    .kernel_run_args =
        {{.kernel = binary_reader_kernel,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
              core,
              {{"in0_addr", static_cast<uint32_t>(src0_dram_tensor.address())},
               {"in1_addr", static_cast<uint32_t>(src1_dram_tensor.address())}})},
         {.kernel = unary_writer_kernel,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
              core, {{"dst_addr", static_cast<uint32_t>(dst_dram_tensor.address())}})}},
};
experimental::SetProgramRunArgs(program, run_args);

workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

For each of the data movement kernels, we will set up the corresponding named runtime arguments (the compute kernel has none) before adding the program to the `MeshWorkload` and executing it on the device. The reader kernel reads the source data into the dataflow buffers before having the compute kernel run the tile addition operation.

## Reader kernel function

``` cpp
DataflowBuffer dfb_in0(dfb::in0);
DataflowBuffer dfb_in1(dfb::in1);

const uint32_t tile_size_bytes = dfb_in0.get_entry_size();

// read the tiles from DRAM into the dataflow buffers
dfb_in0.reserve_back(1);
uint32_t dfb_in0_addr = dfb_in0.get_write_ptr();
noc_async_read_page(0, in0, dfb_in0_addr);  // read
noc_async_read_barrier();                   // wait until the read is done
dfb_in0.push_back(1);                       // mark the tile as ready.

// same process for the second input (different dataflow buffer and input buffer)
dfb_in1.reserve_back(1);
uint32_t dfb_in1_addr = dfb_in1.get_write_ptr();
noc_async_read_page(0, in1, dfb_in1_addr);
noc_async_read_barrier();
dfb_in1.push_back(1);
```

The reader kernel reads in a one tile from each of the two source vectors that are stored in the DRAM, and stores these values in dataflow buffers in the given core, with each source vector having its own corresponding dataflow buffer. The kernel accesses each dataflow buffer by the name it is bound to in the host code (for example `dfb::in0`).

## Compute kernel function

``` cpp
compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out0);  // Unpack, Math, Pack
add_init(dfb::in0, dfb::in1);                              // Unpack, Math

// wait for a tile to be ready in the input DFBs
dfb_in0.wait_front(1);  // Unpack
dfb_in1.wait_front(1);  // Unpack

// acquire 8 tile registers to perform the addition
tile_regs_acquire();  // Math

add_tiles(dfb::in0, dfb::in1, 0, 0, 0);  // Unpack, Math

// signal the packer
tile_regs_commit();  // Math

// packer waits here
tile_regs_wait();  // Pack
pack_tile(0, dfb::out0);  // Pack
// packer releases
tile_regs_release();  // Pack

dfb_in0.pop_front(1);  // Unpack
dfb_in1.pop_front(1);  // Unpack

dfb_out0.push_back(1);  // Pack
```

In the compute kernel, a single tile is read from each of the dataflow buffers corresponding to the source data. These values are unpacked from their original data formats into unsigned integers. Then, `add_tiles()` computes the result of the addition between the two retrieved tiles. The result is then packed back into the original data format and written back to the corresponding dataflow buffer.

## Writer kernel function

``` cpp
DataflowBuffer dfb_out0(dfb::out0);
const uint32_t tile_size_bytes = dfb_out0.get_entry_size();

const InterleavedAddrGenFast<true> dst = {
    .bank_base_address = dst_addr,
    .page_size = tile_size_bytes,
    .data_format = DataFormat::Float16_b,
};

dfb_out0.wait_front(1);
uint32_t dfb_out0_addr = dfb_out0.get_read_ptr();
noc_async_write_page(0, dst, dfb_out0_addr);
noc_async_write_barrier();
dfb_out0.pop_front(1);
```

At this point, the results of the addition are computed and stored in the dataflow buffers. We can now write these values to DRAM so that they can be accessed by the host.

``` cpp
std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, true).to_vector<bfloat16>();
// ... compare result_vec with the expected values ...
mesh_device->close();
```

When the program is finished with execution, the output data is stored in the DRAM and must be read from the device using `enqueue_read_tensor`.
