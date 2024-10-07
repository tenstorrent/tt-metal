# Add 2 Integers in Compute Kernel

In this example, we will build a TT-Metal program that will add two vectors containing integers together, using data movement and compute kernels.

This program can be found in
[add_2_integers_in_compute.cpp](../../../tt_metal/programming_examples/add_2_integers_in_compute/add_2_integers_in_compute.cpp).

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh --build-tests
    ./build/programming_examples/add_2_integers_in_compute
```
## Set up device and program/collaboration mechanisms

``` cpp
Device *device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
auto program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```

We follow the standard procedure for the initial steps in setting up the host program. The device that the program will execute on is identified, and the corresponding command queue is accessed. The program is initialized, and the core indicated for utilization in this example is at the coordinates `{0, 0}` in accordance with the logical mesh layout.

## Configure and initialize DRAM buffer

``` cpp
constexpr uint32_t single_tile_size = 2 * 1024;
tt_metal::InterleavedBufferConfig dram_config{
            .device= device,
            .size = single_tile_size,
            .page_size = single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM
};
```

We define the tile size to fit BFloat16 values before setting up the configuration for the DRAM buffer. Each tile is 32x32 = 1024 bytes; doubling this allows us to tile up BFloat16 values. We specify the device to create the buffers on as well as the size of the buffers. Our DRAM configuration will be interleaved for this example, which makes the data layout row-based. Note that our choice of data format and buffer configuration has significant impact on the performance of the application, as we are able to reduce data traffic by packing values.

``` cpp
std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
```

Next, we allocate memory for each buffer with the specified configuration for each of the input vectors and another buffer for the output vector. The source data will be sent to the corresponding DRAM buffers to be accessed by the cores, and the results of the computation will be sent to the DRAM to be read by the destination vector.

``` cpp
auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
auto src1_dram_noc_coord = src1_dram_buffer->noc_coordinates();
auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
uint32_t src0_dram_noc_x = src0_dram_noc_coord.x;
uint32_t src0_dram_noc_y = src0_dram_noc_coord.y;
uint32_t src1_dram_noc_x = src1_dram_noc_coord.x;
uint32_t src1_dram_noc_y = src1_dram_noc_coord.y;
uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;
```

For this example, we will also specify the NoC coordinates to pass into the kernel functions as runtime arguments. We will use this to ensure that the kernels will access the data at the correct NoC addresses.

``` cpp
constexpr uint32_t src0_cb_index = CB::c_in0;
constexpr uint32_t num_input_tiles = 1;
CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

constexpr uint32_t src1_cb_index = CB::c_in1;
CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

constexpr uint32_t output_cb_index = CB::c_out0;
constexpr uint32_t num_output_tiles = 1;
CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, single_tile_size);
CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
```

L1 circular buffers will be used communicate data to and from the compute engine. We create circular buffers for the source vectors and destination vector. The source data will be sent from the DRAM buffers to the circular buffer of each specified core, then the results for a given core will be stored at another circular buffer index before being sent to DRAM.

## Kernel setup

``` cpp
KernelHandle binary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

KernelHandle unary_writer_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
```

Data movement kernels are used for reading to and writing from the DRAM.
A kernel is initialized for each of these operations, with a unique RISC-V processor assigned to each kernel. These kernels will read the data from the DRAM buffers into the circular buffers prior to the addition operation, then write the output data to the DRAM from the circular buffers so that they may be accessed by the host.

``` cpp
vector<uint32_t> compute_kernel_args = {};
KernelHandle eltwise_binary_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_kernel_args,
    }
);
```

In addition to the data movement kernels, we need to create a compute kernel for the addition operation. We use the kernel code for adding 2 tiles as specified in the above code block. The kernel function will use the data provided in the circular buffers for the computation.

## Program execution

``` cpp
std::vector<uint32_t> src0_vec;
std::vector<uint32_t> src1_vec;
src0_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);
src1_vec = create_constant_vector_of_bfloat16(single_tile_size, 8.0f);

EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);
```

Next, we create two source vectors, each loaded with a constant value, before queueing the command to feed it to the corresponding DRAM buffers using `EnqueueWriteBuffer`.

``` cpp
SetRuntimeArgs(program, binary_reader_kernel_id, core, { src0_dram_buffer->address(), src1_dram_buffer->address(), src0_dram_noc_x, src0_dram_noc_y, src1_dram_noc_x, src1_dram_noc_y});
SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_dram_noc_x, dst_dram_noc_y});

EnqueueProgram(cq, program, false);
Finish(cq);
```

For each of the kernels, we will set up the corresponding runtime arguments before executing the program on the device. The reader kernel reads the source data into the circular buffers before having the compute kernel run the tile addition operation.

## Reader kernel function

``` cpp
uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

cb_reserve_back(cb_id_in0, 1);
noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
noc_async_read_barrier();
cb_push_back(cb_id_in0, 1);

cb_reserve_back(cb_id_in1, 1);
noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
noc_async_read_barrier();
cb_push_back(cb_id_in1, 1);
```

The reader kernel reads in a one tile from each of the two source vectors that are stored in the DRAM, and stores these values in circular buffers in the given core, with each source vector having its own corresponding circular buffer.

## Compute kernel function

``` cpp
binary_op_init_common(cb_in0, cb_in1, cb_out0);
add_tiles_init();

// wait for a block of tiles in each of input CBs
cb_wait_front(cb_in0, 1);
cb_wait_front(cb_in1, 1);

tile_regs_acquire(); // acquire 8 tile registers

add_tiles(cb_in0, cb_in1, 0, 0, 0);

tile_regs_commit(); // signal the packer

tile_regs_wait(); // packer waits here
pack_tile(0, cb_out0);
tile_regs_release(); // packer releases

cb_pop_front(cb_in0, 1);
cb_pop_front(cb_in1, 1);

cb_push_back(cb_out0, 1);
```

In the compute kernel, a single tile is read from each of the circular buffers corresponding to the source data. These values are unpacked from their original data formats into unsigned integers. Then, `add_tiles()` computes the result of the addition between the two retrieved tiles. The result is then packed back into the original data format and written back to the corresponding circular buffer.

## Writer kernel function

``` cpp
uint64_t dst_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_addr);

constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

cb_wait_front(cb_id_out0, 1);
noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
noc_async_write_barrier();
cb_pop_front(cb_id_out0, 1);
```

At this point, the results of the addition are computed and stored in the circular buffers. We can now write these values to DRAM so that they can be accessed by the host.

``` cpp
std::vector<uint32_t> result_vec;
EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
CloseDevice(device);
```

When the program is finished with execution, the output data is stored in the DRAM and must be read from the device using `EnqueueReadBuffer`.
