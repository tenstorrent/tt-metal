# Add 2 Integers in RISC-V

RISC-V processors 1 and 5 of a Tensix core are used for data movement, yet also have basic computing capabilities. In this example, we will build a TT-Metalium program to add two integers using these processors.

We'll go through this code section by section. Note that we have this exact, full example program in
[add_2_integers_in_riscv.cpp](../../../tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp),
so you can follow along.

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh
    ./build/programming_examples/add_2_integers_in_riscv
```
## Set up accelerator and program/collaboration mechanisms

``` cpp
Device *device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```

We follow the standard procedure for the initial steps in setting up the host program. The device that the program will execute on is identified, and the corresponding command queue is accessed. The program is initialized, and the core indicated for utilization in this example is at the coordinates `{0, 0}` in accordance to the logical mesh layout.

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

We define the tile size to fit BFloat16 values before setting up the configuration for the DRAM buffer. We specify the device to create the buffers on as well as the size of the buffers. In this example, we will use an interleaved DRAM configuration for the buffers.

``` cpp
std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
```

Next, we allocate memory for each buffer with the specified configuration for each of the input vectors and another buffer for the output vector. Source data will move from the host to DRAM, and the output will be sent from the DRAM to host.

## Initialize source data and write to DRAM

``` cpp
std::vector<uint32_t> src0_vec(1, 14);
std::vector<uint32_t> src1_vec(1, 7);

EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);
```

On the host side, we set initialize the source data. In this case, they are represented as vectors with a single integer value. These values are then written to the corresponding DRAM buffers on the device through a dispatch by the command queue.

## Set up circular buffers for input

``` cpp
constexpr uint32_t src0_cb_index = CB::c_in0;
CircularBufferConfig cb_src0_config = CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

constexpr uint32_t src1_cb_index = CB::c_in1;
CircularBufferConfig cb_src1_config = CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
```

L1 circular buffers will be used communicate data to and from the compute engine. We create a circular buffer for each of the source vectors. Each core will have its own segment of the source data stored in its corresponding circular buffer.

## Kernel setup

``` cpp
KernelHandle binary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
```

In this example, we are using data movement processors for basic computation. As such, we create a kernel function for integer addition that utilizes the RISC-V 1 processor, which is designated for data movement, to run itself on. This kernel perform tile reading, addition, and writing.

## Configure and execute program

``` cpp
SetRuntimeArgs(program, binary_reader_kernel_id, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), dst_dram_buffer->address(),});

EnqueueProgram(cq, program, false);
Finish(cq);
```

In order to execute the program, we need to load the runtime arguments for the kernel function. After doing so with the corresponding buffer addresses, we can dispatch the program to the device for execution through the command queue.

## Kernel execution

``` cpp
// NoC coords (x,y) depending on DRAM location on-chip
uint64_t src0_dram_noc_addr = get_noc_addr(src0_dram_noc_x, src0_dram_noc_y, src0_dram);
uint64_t src1_dram_noc_addr = get_noc_addr(src1_dram_noc_x, src1_dram_noc_y, src1_dram);
uint64_t dst_dram_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_dram);

constexpr uint32_t cb_id_in0 = tt::CB::c_in0; // index=0
constexpr uint32_t cb_id_in1 = tt::CB::c_in1; // index=1

// single-tile ublocks
uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
```

We first define the NoC addresses used for reading and writing data to/from DRAM, as well as retrieve the L1 addresses to access for data movement. Each kernel will access a single tile.

``` cpp
// Read data from DRAM -> L1 circular buffers
noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
noc_async_read_barrier();
noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
noc_async_read_barrier();

// Do simple add in RiscV core
uint32_t* dat0 = (uint32_t*) l1_write_addr_in0;
uint32_t* dat1 = (uint32_t*) l1_write_addr_in1;

dat0[0] = dat0[0] + dat1[0];

// Write data from L1 circulr buffer (in0) -> DRAM
noc_async_write(l1_write_addr_in0, dst_dram_noc_addr, ublock_size_bytes_0);
noc_async_write_barrier();
```

In the kernel, tiles corresponding to each of the source vectors will be read from the DRAM into circular buffers. These tiles will then be accessed and added together. The sum is stored in one of the circular buffers temporarily before being written directly to DRAM to be accessed by the host.

## Retrieve compute results

``` cpp
std::vector<uint32_t> result_vec;
EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
printf("Result = %d : Expected = 21\n", result_vec[0]);

CloseDevice(device);
```

After executing the program, we create a destination vector on the host side to store the results of the device execution. Using `EnqueueReadBuffer`, the results are read from DRAM to the destination vector and displayed.
