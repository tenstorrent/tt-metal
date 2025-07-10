# NoC tile transfer

In this example, we will build a TT-Metal program that will be able to send data between cores using NoC (network on chip) L1 to L1 memory communication.

This program can be found in [noc_tile_transfer.cpp](tt_metal/programming_examples/NoC_tile_transfer/noc_tile_transfer.cpp)

To build and execute, you may use the following commands. Note that we include the necessary environment variables here, but you may possibly need more depending on the most up-to-date installation methods.

Run the appropriate command for the Tenstorrent card you have installed:

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_noc_tile_transfer
```

## Main program

In this example we will use two cores - each running two kernels (reader and writer). We will create simple data pipeline, where one core will read data from DRAM memory bank, send it to other core and save it back to DRAM. The communication flow can be visualized as:

![Diagram](media/NoC_transfer_example_diagram.drawio.svg)

### Set up device and program mechanisms

``` cpp
// Device setup
IDevice* device = CreateDevice(0);

// Device command queue and program setup
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
```

### Setup core ranges

In this example, we are transferring data between two cores, so we need to allocate two virtual cores for this purpose. We do this by manually specifying the coordinates of each core individually as a point (`CoreCoord`), as we'll use them for the synchronization mechanism later. To facilitate the creation of future variables, such as semaphores, we also create a `CoreRange` object, which points to the range encompassing both cores.

The mechanisms of synchronization and data transfer between cores used in kernels instead of using logical coordinates, use their physical equivalents, we obtain them using the `worker_core_from_logical_core function`.

```cpp
// Core range setup
constexpr CoreCoord core0 = {0, 0};
constexpr CoreCoord core1 = {0, 1};
const auto core0_physical_coord = device->worker_core_from_logical_core(core0);
const auto core1_physical_coord = device->worker_core_from_logical_core(core1);

CoreRange sem_core_range = CoreRange(core0, core1);
```

### Input data preparation

In this example we will place an input value in DRAM memory bank and create additional output SRAM memory bank to simulate real life data pipeline. We start by setting up memory buffers:

```cpp
constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;
InterleavedBufferConfig dram_config{
    .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);  // Input buffer
std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);  // Output buffer

const bool input_tensor_is_dram = src_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
const bool output_tensor_is_dram = dst_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
```

In this example we will send value 14 between cores.

```cpp
const uint16_t input_data = 14;  // Example input data
std::vector<uint16_t> src_vec(1, input_data);
EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);
```

### Synchronization

To synchronize two cores between each other we will use semaphore. Notice that under `core_spec` we place `sem_core_range` and not individual cores.

```cpp
const uint32_t sem_id = CreateSemaphore(program, sem_core_range, 0);
```

### Circullar buffers setup

Each core receives each own individual CB.

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

### Kernel setup

In this example each core handles each own individual task, so we need to set up four different kernels - two on each core.

```cpp
// Core 0 kernels
KernelHandle core0_reader_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/reader0.cpp",
    core0,
    tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, static_cast<uint32_t>(input_tensor_is_dram)}});
KernelHandle core0_writer_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/writer0.cpp",
    core0,
    tt::tt_metal::WriterDataMovementConfig{{src0_cb_index, src1_cb_index}});

// Core 1 kernels
KernelHandle core1_reader_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/reader1.cpp",
    core1,
    tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, src1_cb_index}});
KernelHandle core1_writer_kernel_id = CreateKernel(
    program,
    "NoC_tile_transfer/kernels/dataflow/writer1.cpp",
    core1,
    tt::tt_metal::WriterDataMovementConfig{{src1_cb_index, static_cast<uint32_t>(output_tensor_is_dram)}});

SetRuntimeArgs(program, core0_reader_kernel_id, core0, {src_dram_buffer->address()});
SetRuntimeArgs(program, core0_writer_kernel_id, core0, {core1_physical_coord.x, core1_physical_coord.y, sem_id});
SetRuntimeArgs(program, core1_reader_kernel_id, core1, {core0_physical_coord.x, core0_physical_coord.y, sem_id});
SetRuntimeArgs(program, core1_writer_kernel_id, core1, {dst_dram_buffer->address()});
```

### Program enqueue and final check

As a final step we enqueue the prepared program and add data transfer back to hos from destination buffer to check if the data transfer was successful.

```cpp
// Program enqueue
EnqueueProgram(cq, program, false);
Finish(cq);

// Data transfer back to host machine
std::vector<uint16_t> result_vec;
EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);  // Blocking call to ensure data is read before proceeding

std::cout << "Result = " << result_vec[0] << " : Expected = " << input_data << std::endl;

CloseDevice(device);
```

## Kernels

### Core 0 - reader 0

This kernel is responsible for reading input data from DRAM bank to L1 CB and pushing it to writer0 kernel.

```cpp
// Read input value data
cb_reserve_back(src0_cb_index, one_tile);
const uint32_t l1_write_addr = get_write_ptr(src0_cb_index);
noc_async_read_tile(0, interleaved_accessor, l1_write_addr);
noc_async_read_barrier();

// Push data to writer 0 kernel
cb_push_back(src0_cb_index, one_tile);
```

### Core 0 - writer 0

This kernel receives data
