# Matmul (Single Core)

We'll build a program that will perform matmul operations on two tensors with equal-size inner dimension. We will then go through specific sections of the program.

The full example program is in
[matmul_single_core.cpp](../../../tt_metal/programming_examples/matmul_single_core/matmul_single_core.cpp)

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh --build-tests
    ./build/programming_examples/matmul_single_core
```

## Host Code

The initial level of host-side code can broken up into sections:

-   Create Device
-   Set input and output vector variables, using the user-defined parameters (M, N, K, B)
-   Tilizing the input vector, and untilizing the device output to vector (row-major layout)
-   Call `matmul_single_core()` program and retrieve output results (details in next section)
-   Validate the device computation results vs. golden results on cpu
-   Close Device

``` cpp
/* Create source data */
constexpr uint32_t M = 640;  // user-defined
constexpr uint32_t N = 640;  // user-defined
constexpr uint32_t K = 640;  // user-defined
constexpr uint32_t B = 1;  // user-defined
uint32_t Mt = M / TILE_HEIGHT;
uint32_t Kt = K / TILE_WIDTH;
uint32_t Nt = N / TILE_WIDTH;
constexpr uint32_t single_tile_size = 2 * 1024;
uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

/* input vectors */
std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(dram_buffer_A_size, 1, 123);
std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(dram_buffer_B_size, 1, 12522);
/* Input vector tilizing */
tilize(src0_vec, M, K);
tilize(src1_vec, K, N);
/* Calling the MatMul host program. Read in result into a host vector */
vector<bfloat16> result_vec(dram_buffer_C_size/sizeof(bfloat16));
matmul_single_core(src0_vec, src1_vec, result_vec, false, M, N, K, B, device);
untilize(result_vec, M, N);

CloseDevice(device);
```

We are keeping all code details with specific host API calls inside `matmul_single_core`, allowing for calling consecutive functions in the main function.

## Main blocks in matmul_single_core function

We will go through sections of the `matmul_single_core` function:

-   Program, enqueue and core range settings
-   Create DRAM buffers based on input and output vectors
-   Create L1 Circular buffers
-   Kernels declarations and related compile and runtime arguments
-   Program launch and reading data from DRAM output buffer to result vector

## Create Program, Enqueue initialization, and core range definition

We want a just a single core, so we will restrict the core range to be just one core at (0, 0).

``` cpp
CommandQueue& cq = detail::GetCommandQueue(device);
Program program{};
CoreRange core({0, 0}, {0, 0});
```

## Create DRAM buffers & Circular buffers

In terms of DRAM buffers, we need two source buffers and one destination buffer.

``` cpp
// MN = MK*KN
uint32_t Mt = M / TILE_HEIGHT;
uint32_t Kt = K / TILE_WIDTH;
uint32_t Nt = N / TILE_WIDTH;

DataFormat cb_data_format = DataFormat::Float16_b;
uint32_t single_tile_size = detail::TileSize(cb_data_format);
MathFidelity math_fidelity = MathFidelity::HiFi4;
//uint32_t single_tile_size = detail::TileSize(cb_data_format);
uint32_t single_tile_size = 2 * 1024;

uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

/* DRAM buffer size == input full size */
/* limiting page_size == single tile size; to allow DRAM channels interleaving */

tt_metal::InterleavedBufferConfig buff_A_config{
                                    .device=device,
                                    .size = dram_buffer_A_size,
                                    .page_size = single_tile_size,
                                    .buffer_type = tt_metal::BufferType::DRAM
                                    };
tt_metal::InterleavedBufferConfig buff_B_config{
                                    .device=device,
                                    .size = dram_buffer_B_size,
                                    .page_size = single_tile_size,
                                    .buffer_type = tt_metal::BufferType::DRAM
                                    };
tt_metal::InterleavedBufferConfig buff_C_config{
                                    .device=device,
                                    .size = dram_buffer_C_size,
                                    .page_size = single_tile_size,
                                    .buffer_type = tt_metal::BufferType::DRAM
                                    };
Buffer src0_dram_buffer = CreateBuffer(buff_A_config);
Buffer src1_dram_buffer = CreateBuffer(buff_B_config);
Buffer dst_dram_buffer = CreateBuffer(buff_C_config);
uint32_t src0_addr = src0_dram_buffer.address();
uint32_t src1_addr = src1_dram_buffer.address();
uint32_t dst_addr = dst_dram_buffer.address();
```

We need to declare three circular buffers to enable data transfer between the reader, compute, and writer engines. Input tiles count is 2 because although the computation is a single tile process, we want to get a performance boost by double buffering..

``` cpp
uint32_t src0_cb_index = CB::c_in0; //0
uint32_t num_input_tiles = 2;
tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
    .set_page_size(src0_cb_index, single_tile_size);
auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

uint32_t src1_cb_index = CB::c_in1; // 1
tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
    .set_page_size(src1_cb_index, single_tile_size);
auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
uint32_t num_output_tiles = 2;
tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
    .set_page_size(output_cb_index, single_tile_size);
auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
```

## Compile-time kernels arguments

We have to declare some compile-time arguments for read/write kernels.
Some default parameters here will suffice.

``` cpp
bool src0_is_dram = src0_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
bool src1_is_dram = src1_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

bool dst_is_dram = dst_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

vector<uint32_t> compute_args = {
    B, // B
    Mt, // Mt
    Kt, // Kt
    Nt // Nt
};
```

## Compute kernel declaration and compile-time defines

We're using a special reader kernel to take in data from DRAM into L1, and a special writer kernel to write out results from the compute engine back to the destination DRAM buffer.

``` cpp
auto reader_id = tt_metal::CreateDataMovementKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank.cpp",
    core,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

auto writer_id = tt_metal::CreateDataMovementKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_8bank.cpp",
    core,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

auto matmul_single_core_kernel_id = tt_metal::CreateComputeKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp",
    core,
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}
);
```

## Runtime arguments and program launch

We will now set runtime arguments for the reader and writer kernels to run the matmul operation on a single core and a single tile at a time.

``` cpp
tt_metal::SetRuntimeArgs(
    program, reader_id, core,
    {src0_addr, src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
);

tt_metal::SetRuntimeArgs(
    program, writer_id, core,
    {dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
);
```

Launch program, enqueue & read in output buffer result into the host vector.

``` cpp
EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
EnqueueProgram(cq, program, false);
EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
```

## Conclusion

Those are the additional steps for getting `matmul_single_core` operations up and running on the compute engine. To see a more complicated example using as many cores as possible, please refer to the
`Matmul multi-core` example.
