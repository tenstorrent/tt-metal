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
IDevice* device = CreateDevice(device_id);
```

We instantiate a device to control our accelerator.

## Program pre-compilation setup

``` cpp
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
```

We first obtain the device's `CommandQueue` in order to use the fast dispatch capabilities of the software. This will be used when issuing commands for asynchronous reads/writes/program management.

Next, we create a `Program` to be run on our accelerator. This is how we'll be keeping track of things in our session with the device.

## Create buffers in DRAM and L1

Next, we need to declare buffers that we will use during execution. We will need:

-   An L1 buffer within the core itself that will act as a temporary single-tile buffer
-   A DRAM buffer that will house input data (50 tiles)
-   A DRAM buffer that will be written to with output data (50 tiles)

``` cpp
constexpr uint32_t num_tiles = 50;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
constexpr uint32_t dram_buffer_size = tile_size_bytes * num_tiles;

InterleavedBufferConfig l1_config{
    .device = device,
    .size = tile_size_bytes,  // Single tile only
    .page_size = tile_size_bytes,
    .buffer_type = BufferType::L1
};

auto l1_buffer = CreateBuffer(l1_config);
```

The L1 buffer holds just one tile, while the DRAM buffers each hold 50 tiles.

Let's make the input and output DRAM buffers.

``` cpp
InterleavedBufferConfig dram_config{
    .device = device,
    .size = dram_buffer_size,  // 50 tiles
    .page_size = tile_size_bytes,  // Page size is one tile
    .buffer_type = BufferType::DRAM
};

auto input_dram_buffer = CreateBuffer(dram_config);
auto output_dram_buffer = CreateBuffer(dram_config);
```

## Building a data movement kernel

Declare a kernel for data movement. We'll use a pre-written kernel that copies data from one place to another.

We will be using the accelerator core with coordinates `{0, 0}`.

``` cpp
constexpr CoreCoord core = {0, 0};

std::vector<uint32_t> dram_copy_compile_time_args;
TensorAccessorArgs(*input_dram_buffer).append_to(dram_copy_compile_time_args);
TensorAccessorArgs(*output_dram_buffer).append_to(dram_copy_compile_time_args);
KernelHandle dram_copy_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
    core,
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = dram_copy_compile_time_args});
```

## Sending real data into DRAM

``` cpp
std::vector<bfloat16> input_vec(elements_per_tile * num_tiles);
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
for (auto& val : input_vec) {
    val = bfloat16(distribution(rng));
}

EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, /*blocking=*/false);
```

Send in a randomly-generated bfloat16 vector that will act as our input data tensor.

We use a non-blocking call so we can continue setting up our program.

## Setting runtime arguments for the data movement kernel

``` cpp
const std::vector<uint32_t> runtime_args = {
    l1_buffer->address(),
    input_dram_buffer->address(),
    output_dram_buffer->address(),
    num_tiles
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
-   Where the output DRAM buffer starts (memory address)
-   The number of tiles to copy

## Running the program

``` cpp
EnqueueProgram(cq, program, /*blocking=*/false);
Finish(cq);
// NOTE: The above is equivalent to the following single line:
// EnqueueProgram(cq, program, /*blocking=*/true);
```

Now we finally launch our program. The `Finish` call waits for the
program to return a finished status.

## Launch and verify output

Then we can finally read back the data from the output buffer and assert that it matches what we sent!

``` cpp
std::vector<bfloat16> result_vec;
EnqueueReadBuffer(cq, output_dram_buffer, result_vec, /*blocking=*/true);

for (int i = 0; i < input_vec.size(); i++) {
    if (input_vec[i] != result_vec[i]) {
        pass = false;
        break;
    }
}
```

We use a blocking call this time because we want to get all the data before doing a comparison.

## Validation and teardown

``` cpp
pass &= CloseDevice(device);
```

We now use `CloseDevice` to teardown our connection to the Tenstorrent device.

Now we can start adding some compute to our program. Please refer to the `Eltwise SFPU example`.
