# NoC Tile Transfer

In this example, we build a TT-Metal program that demonstrates how to transfer data between two cores using the NoC (Network-on-Chip) for direct **L1-to-L1 memory communication**.

You can find the full example in
[`noc_tile_transfer.cpp`](tt_metal/programming_examples/NoC_tile_transfer/noc_tile_transfer.cpp).

## Building and Running the Example

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

Run the appropriate command for the Tenstorrent card you have installed:

```bash
export TT_METAL_HOME=$(pwd)
./build_metal.sh --build-programming-examples
./build/programming_examples/metal_example_noc_tile_transfer
```

---

## Main Program Overview

This example uses **two cores**, each running a **reader** and a **writer** kernel. The data pipeline is simple:

1. Core 0 reads data from DRAM into its local CB (Circular Buffer).
2. Core 0 sends the data over the NoC to Core 1.
3. Core 1 receives data from Core 0.
4. Core 1 writes the data back to DRAM.

The communication flow is illustrated below:

![Diagram](media/NoC_transfer_example_diagram.drawio.svg)

---

### Device and Program Setup

```cpp
IDevice* device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
```

---

### Core Configuration

In this example, we are transferring data between two cores, so we need to allocate two virtual cores for this purpose. We do this by manually specifying the coordinates of each core as a `CoreCoord` point, since we will use these coordinates later for synchronization. To simplify the creation of shared resources, such as semaphores, we also define a `CoreRange` object that represents the range covering both cores.

The synchronization and data transfer mechanisms inside the kernels operate using physical core coordinates rather than logical ones. We obtain the physical coordinates using the `worker_core_from_logical_cor` function.

```cpp
constexpr CoreCoord core0 = {0, 0};
constexpr CoreCoord core1 = {0, 1};
const auto core0_physical_coord = device->worker_core_from_logical_core(core0);
const auto core1_physical_coord = device->worker_core_from_logical_core(core1);

CoreRange sem_core_range = CoreRange(core0, core1);
```

In this example each kernel will print some information. To see it in terminal remember to export environmental variable: `export TT_METAL_DPRINT_CORES=(0,0),(0,1)`.

---

### Input and Output Buffers

We create an **input buffer** and **output buffer** in DRAM. These simulate a real-world pipeline where data originates from and is written back to memory.

```cpp
constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;

InterleavedBufferConfig dram_config{
    .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM
};

std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);

const bool input_tensor_is_dram = src_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
const bool output_tensor_is_dram = dst_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
```

We initialize the input with the value `14`:

```cpp
const uint16_t input_data = 14;
std::vector<uint16_t> src_vec(1, input_data);
EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);
```

---

### Synchronization Setup

We create a semaphore shared between the two cores:

```cpp
const uint32_t sem_id = CreateSemaphore(program, sem_core_range, 0);
```

---

### Circular Buffers

Each core uses its own dedicated circular buffer for local data storage:

```cpp
constexpr uint32_t src0_cb_index = CBIndex::c_0;
CircularBufferConfig cb_src0_config =
    CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::UInt16}})
        .set_page_size(src0_cb_index, single_tile_size);
CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src0_config);

constexpr uint32_t src1_cb_index = CBIndex::c_1;
CircularBufferConfig cb_src1_config =
    CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::UInt16}})
        .set_page_size(src1_cb_index, single_tile_size);
CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src1_config);
```

---

### Kernel Setup

We assign four kernels—two per core:

```cpp
// Core 0
KernelHandle core0_reader_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/reader0.cpp",
    core0,
    tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, static_cast<uint32_t>(input_tensor_is_dram)}}
);
KernelHandle core0_writer_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/writer0.cpp",
    core0,
    tt::tt_metal::WriterDataMovementConfig{{src0_cb_index, src1_cb_index}}
);

// Core 1
KernelHandle core1_reader_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/reader1.cpp",
    core1,
    tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, src1_cb_index}}
);
KernelHandle core1_writer_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/writer1.cpp",
    core1,
    tt::tt_metal::WriterDataMovementConfig{{src1_cb_index, static_cast<uint32_t>(output_tensor_is_dram)}}
);

SetRuntimeArgs(program, core0_reader_kernel_id, core0, {src_dram_buffer->address()});
SetRuntimeArgs(program, core0_writer_kernel_id, core0, {core1_physical_coord.x, core1_physical_coord.y, sem_id});
SetRuntimeArgs(program, core1_reader_kernel_id, core1, {core0_physical_coord.x, core0_physical_coord.y, sem_id});
SetRuntimeArgs(program, core1_writer_kernel_id, core1, {dst_dram_buffer->address()});
```

---

### Running and Verifying

We enqueue the program, finish execution, and verify the output:

```cpp
EnqueueProgram(cq, program, false);
Finish(cq);

std::vector<uint16_t> result_vec;
EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

std::cout << "Result = " << result_vec[0] << " : Expected = " << input_data << std::endl;

CloseDevice(device);
```

---

## Kernel Descriptions

### Core 0 — Reader 0

* Reads input from DRAM to local CB.
* Pushes data to Writer 0.

```cpp
cb_reserve_back(src0_cb_index, one_tile);
const uint32_t l1_write_addr = get_write_ptr(src0_cb_index);
noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
noc_async_read_barrier();
cb_push_back(src0_cb_index, one_tile);
```

---

### Core 0 — Writer 0

* Waits for Core 1 to signal readiness.
* Sends tile to Core 1 using NoC.
* Updates synchronization semaphore.

```cpp
noc_semaphore_wait(sem_ptr, 1);
noc_async_write(data_l1_ptr, core1_noc_addr, input_data_tile_size_bytes);
noc_async_write_barrier();
noc_semaphore_inc(sem_addr, 1);
noc_semaphore_set(sem_ptr, 0);
```

---

### Core 1 — Reader 1

* Signals readiness to Core 0.
* Waits until data is received.
* Resets semaphore.

```cpp
cb_reserve_back(src1_cb_index, one_tile);
noc_semaphore_inc(sem_addr, 1);
noc_semaphore_wait(sem_ptr, 1);
noc_semaphore_set(sem_ptr, 0);
```

---

### Core 1 — Writer 1

* Waits for tile in CB.
* Writes tile to DRAM.
* Pops tile from CB.

```cpp
cb_wait_front(src1_cb_index, one_tile);
noc_async_write_tile(0, output_tensor_dram, l1_write_addr_output);
noc_async_write_barrier();
cb_pop_front(src1_cb_index, one_tile);
```

---

## Expected Output

```console
0:(x=0,y=0):NC: 1. READER 0: Reading input data to L1 src0 CB
0:(x=0,y=0):NC: 2. READER 0: Data in src0 CB pushed from reader0
0:(x=0,y=0):BR: 3. WRITER 0: Data available in src0 CB
0:(x=0,y=0):BR: 4. WRITER 0: Data sent to core 1 from core 0
0:(x=0,y=1):NC: 5. READER 1: Preparing for data transfer
0:(x=0,y=1):NC: 6. READER 1: Data received and stored in src1
0:(x=0,y=1):BR: 7. WRITER 1: Received data
0:(x=0,y=1):BR: 8. WRITER 1: Data saved

Result = 14 : Expected = 14
```

---

© Tenstorrent AI ULC 2025
