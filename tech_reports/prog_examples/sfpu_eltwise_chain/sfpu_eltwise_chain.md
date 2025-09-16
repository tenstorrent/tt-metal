# SFPU Eltwise Chain

This example demonstrates how to build a TT-Metal program that chains **SFPU (Special Function Processing Unit) operations** together. By chaining operations, the output of one SFPU function becomes the direct input for the next operation, enabling efficient computation of complex mathematical functions without intermediate memory transfers. This approach keeps intermediate results in registers rather than repeatedly moving data between circular buffers in L1 memory or DRAM, significantly improving performance.

You can find the full example in
[`sfpu_eltwise_chain.cpp`](/tt_metal/programming_examples/sfpu_eltwise_chain/sfpu_eltwise_chain.cpp).

## Building and Running the Example

To build and execute, you may use the following commands:
```bash
export TT_METAL_HOME=$(pwd)
./build_metal.sh --build-programming-examples
./build/programming_examples/metal_example_sfpu_eltwise_chain
```

## Main Program Overview

This example demonstrates **SFPU operation chaining** by implementing the **softplus activation function**: `softplus(x) = log(1 + exp(x))`.

We could compute softplus by calling separate operations on the tensor: first `ttnn::exp`, storing provisional results into memory, then adding 1, storing again, and finally computing `ttnn::log`. However, each separate operation has overhead from reading and writing memory. To reduce this overhead, we merge these three operations into a single chained operation, so we only read from and write to memory once.

The three chained SFPU operations are:

1. **exp(x)** - Exponential function
2. **exp(x) + 1** - Add constant (1) to the result
3. **log(exp(x) + 1)** - Natural logarithm of the result

The data pipeline is:
1. Input data is read from DRAM into a circular buffer
2. A tile of ones is created in L1 memory for the addition operation
3. The compute kernel performs the chained SFPU operations on a single tile
4. The result is written back to DRAM

---

### Device and Program Setup

```cpp
IDevice* device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
```

---

### Core Configuration

This example uses a single core for computation:

```cpp
constexpr CoreCoord core = {0, 0};
```

---

### Input Data Preparation

We generate random input data and calculate the expected golden results on the CPU for validation (we will produce one tile 32x32 of data for this example):

```cpp
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> dist(0.f, 1.0f);

std::vector<bfloat16> src_vec(constants::TILE_HW);
for (bfloat16& v : src_vec) {
    v = bfloat16(dist(rng));
}

std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
golden_softplus(src_vec, golden_vec);
```

The input data is then tilized to match the hardware's expected tiled layout from flat structure of CPU's vector:

```cpp
src_vec = tilize_nfaces(src_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);
```

To learn more about tile layout, please refer to the [Tiles documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/tiles.html).

---

### Memory Buffers

We create input and output DRAM buffers to store our data:

```cpp
constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
tt_metal::InterleavedBufferConfig dram_config{
    .device = device,
    .size = sizeof(bfloat16) * src_vec.size(),
    .page_size = single_tile_size,
    .buffer_type = tt_metal::BufferType::DRAM};

std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);

EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);
```

---

### Circular Buffers

Three circular buffers are created for the computation:

```cpp
// Input data buffer
constexpr uint32_t src_cb_index = CBIndex::c_0;
CircularBufferConfig cb_src_config =
    CircularBufferConfig(single_tile_size, {{src_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src_cb_index, single_tile_size);
tt_metal::CreateCircularBuffer(program, core, cb_src_config);

// Ones buffer (for addition operation)
constexpr uint32_t ones_cb_index = CBIndex::c_1;
CircularBufferConfig cb_ones_config =
    CircularBufferConfig(single_tile_size, {{ones_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ones_cb_index, single_tile_size);
tt_metal::CreateCircularBuffer(program, core, cb_ones_config);

// Result buffer
constexpr uint32_t result_cb_index = CBIndex::c_2;
CircularBufferConfig cb_result_config =
    CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(result_cb_index, single_tile_size);
tt_metal::CreateCircularBuffer(program, core, cb_result_config);
```

---

### Kernel Setup

Three kernels are created for this example:

```cpp
// Reader kernel - reads input data and creates ones tile
KernelHandle reader_kernel_id = CreateKernel(
    program,
    "sfpu_eltwise_chain/kernels/dataflow/reader.cpp",
    core,
    tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

// Writer kernel - writes result back to DRAM
KernelHandle writer_kernel_id = CreateKernel(
    program,
    "sfpu_eltwise_chain/kernels/dataflow/writer.cpp",
    core,
    tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

// Compute kernel - performs chained SFPU operations
KernelHandle compute_kernel_id = CreateKernel(
    program,
    "sfpu_eltwise_chain/kernels/compute/compute.cpp",
    core,
    tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
```

---

### Running and Validation

After executing the program, we validate the results using Pearson Correlation Coefficient:

```cpp
EnqueueProgram(cq, program, false);
Finish(cq);

std::vector<bfloat16> result_vec(constants::TILE_HW, 0);
EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);
```

---

## Kernel Descriptions

### Reader Kernel

The reader kernel performs two main tasks:
1. **Reads input data** from DRAM into the source circular buffer
2. **Creates a tile of ones** in L1 memory for the addition operation

```cpp
// Read input data
cb_reserve_back(src_cb_index, one_tile);
const uint32_t l1_write_addr = get_write_ptr(src_cb_index);
noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
noc_async_read_barrier();
cb_push_back(src_cb_index, one_tile);

// Create tile with ones
cb_reserve_back(ones_cb_index, one_tile);
const uint32_t ones_l1_write_addr = get_write_ptr(ones_cb_index);
volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ones_l1_write_addr);
for (uint32_t i = 0; i < tt::constants::TILE_HW; i++) {
    ptr[i] = float_to_bfloat16(1.0f);
}
cb_push_back(ones_cb_index, one_tile);
```

Note additional `float_to_bfloat16` function. Since `bfloat16` is not a native C++ type we obtain a `uint16_t` pointer (the same size as `bfloat16`) and save converted values.

---

### Writer Kernel

The writer kernel reads computed values from the circular buffer and writes these results back to DRAM:

```cpp
cb_wait_front(result_cb_index, one_tile);
const uint32_t l1_read_addr = get_read_ptr(result_cb_index);
noc_async_write_tile(0, interleaved_accessor, l1_read_addr);
noc_async_write_barrier();
cb_pop_front(result_cb_index, one_tile);
```

---

### Compute Kernel - The SFPU Chaining

This is the heart of the example, demonstrating **SFPU operation chaining**:

```cpp
// Initialize SFPU and acquire tile registers
init_sfpu(src_cb_index, result_cb_index);
tile_regs_acquire();

// Load data into registers
cb_wait_front(src_cb_index, one_tile);
cb_wait_front(ones_cb_index, one_tile);
copy_tile(src_cb_index, 0, 0);      // Input data → register 0
copy_tile(ones_cb_index, 0, 1);     // Ones tile → register 1

// Chained SFPU operations chain
exp_tile_init();
exp_tile(0);                        // reads 0-th DST register; compute 'exp'; store to 0-th DST register

add_binary_tile_init();
add_binary_tile(0, 1, 0);          // read 0-th and 1-st DST register; add both registers; store output to 0-th register

log_tile_init();
log_tile(0);                       // reads 0-th DST register; compute 'log'; store to 0-th DST register

// Store result and cleanup
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(result_cb_index, one_tile);
pack_tile(0, result_cb_index);
```

### Key SFPU Chaining Concepts

1. **Register Reuse**: The output of each operation stays in the same register and becomes input for the next operation
2. **No Intermediate Memory**: Results don't need to be stored back to circular buffers between operations
3. **Operation Initialization**: Each SFPU operation requires initialization (`*_init()`) before use
4. **Efficient Chaining**: Multiple complex functions can be computed in a single pass through the data

---

## Expected Output

```console
Metalium vs Golden -- PCC = 0.999847
```

The high Pearson Correlation Coefficient (>0.999) indicates that the SFPU chained operations produce results nearly identical to the CPU golden reference, demonstrating both the correctness and precision of the hardware implementation.

---

## Benefits of SFPU Chaining

1. **Memory Bandwidth Reduction**: Eliminates intermediate memory transfers
2. **Lower Latency**: Reduces the number of memory access cycles
3. **Higher Throughput**: Enables computation of complex functions in fewer kernel invocations
4. **Power Efficiency**: Reduces memory I/O operations which are typically power-intensive

This pattern can be extended to implement more complex mathematical functions by chaining additional SFPU operations together.

### Important Notes on SFPU Precision

**Precision Considerations**: The DST (Destination) registers used in SFPU operations can hold data in either BFP16 or FP32 format depending on kernel configuration settings. This precision setting affects the accuracy of chained operations, as precision loss can accumulate through the chain of computations.

For detailed information about DST register precision and configuration, refer to the [Compute Engines and Dataflow documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html#dst-register).

---

© Tenstorrent AI ULC 2025
