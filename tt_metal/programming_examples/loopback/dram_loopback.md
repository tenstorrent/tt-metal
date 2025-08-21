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
## Device setup

``` cpp
constexpr int device_id = 0;
Device *device = CreateDevice(device_id);
```

We instantiate a device to control our `GRAYSKULL` type accelerator.

## Program pre-compilation setup

``` cpp
CommandQueue& cq = detail::GetCommandQueue(device);
Program program = CreateProgram();
```

We first obtain the global `CommandQueue` in order to use the fast dispatch capabilities of the software. This will be used when issuing commands for asynchronous reads/writes/program management.

Next, we create a `Program` to be run on our Grayskull accelerator. This is how we'll be keeping track of things in our session with the device.

## Building a data movement kernel

Declare a kernel for data movement. We'll use a pre-written kernel that copies data from one place to another.

We will be using the accelerator core with coordinates `{0, 0}`.

``` cpp
constexpr CoreCoord core = {0, 0};

KernelHandle dram_copy_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
);
```

## Create buffers in DRAM and L1

Next, we need to declare buffers that we will use during execution. We will need:

-   An L1 buffer within the core itself that will be used to store the
    compute engine\'s work
-   A DRAM buffer that will house input data
-   A DRAM buffer that will be written to with output data

``` cpp
constexpr uint32_t single_tile_size = 2 * (32 * 32);
constexpr uint32_t num_tiles = 50;
constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;
tt_metal::InterleavedBufferConfig l1_config{
                                      .device=device,
                                      .size = dram_buffer_size,
                                      .page_size = dram_buffer_size,
                                      .buffer_type = tt_metal::BufferType::L1
                                      };

Buffer l1_buffer = CreateBuffer(l1_config);
```

For simplicity, let's make the size of all our buffers 50 tiles.

Let's make the input and output DRAM buffers.

``` cpp
tt_metal::InterleavedBufferConfig dram_config{
                                      .device=device,
                                      .size = dram_buffer_size,
                                      .page_size = dram_buffer_size,
                                      .buffer_type = tt_metal::BufferType::DRAM
                                      };

Buffer input_dram_buffer = CreateBuffer(dram_config);
const uint32_t input_dram_buffer_addr = input_dram_buffer.address();

Buffer output_dram_buffer = CreateBuffer(dram_config);
const uint32_t output_dram_buffer_addr = output_dram_buffer.address();
```

## Sending real data into DRAM

``` cpp
std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
    dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);
```

Send in a randomly-generated FP16 vector that will act as our input data tensor.

We use a non-blocking call so we can continue setting up our program.

## Setting runtime arguments for the data movement kernel

``` cpp
const std::vector<uint32_t> runtime_args = {
    l1_buffer.address(),
    input_dram_buffer.address(),
    input_bank_id,
    output_dram_buffer.address(),
    output_bank_id,
    l1_buffer.size()
};

SetRuntimeArgs(
    program,
    dram_copy_kernel_id,
    core,
    runtime_args
);
```

We now set runtime arguments for our data movement kernel. For this
particular kernel, we have to provide:

-   Where the L1 buffer starts (memory address)
-   Where the input DRAM buffer starts (memory address)
-   The Bank ID of the input DRAM buffer
-   Where the output DRAM buffer starts (memory address)
-   The Bank ID of the output DRAM buffer
-   The size of the buffers

## Running the program

``` cpp
EnqueueProgram(cq, program, false);
Finish(cq);
```

Now we finally launch our program. The `Finish` call waits for the
program to return a finished status.

## Launch and verify output

Then we can finally read back the data from the output buffer and assert that it matches what we sent!

``` cpp
std::vector<uint32_t> result_vec;
EnqueueReadBuffer(cq,output_dram_buffer, result_vec, true);

pass &= input_vec == result_vec;
```

We use a blocking call this time because we want to get all the data before doing a comparison.

## Validation and teardown

``` cpp
pass &= CloseDevice(device);
```

We now use `CloseDevice` to teardown our connection to the Tenstorrent device.

Now we can start adding some compute to our program. Please refer to the `Eltwise SFPU example`.
