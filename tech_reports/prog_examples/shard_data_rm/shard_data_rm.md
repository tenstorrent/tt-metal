# Data Sharding (Multicore)

In this example, we will implement a simple TT-Metalium program to demonstrate how sharding works for untilized data. The code for this program can be found in
[shard_data_rm.cpp](../../../tt_metal/programming_examples/sharding/shard_data_rm.cpp).

The following commands will build and execute the code for this example.
Environment variables may be modified based on the latest
specifications.
```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh --build-tests
    ./build/programming_examples/shard_data_rm
```
# Device setup

``` cpp
int device_id = 0;
Device *device = CreateDevice(device_id);
CommandQueue& cq = device->command_queue();
auto program = CreateProgram();
```

We start the source code by creating an object that designates the hardware device that we will be using for the program. For this example, we select the device with an ID of 0.
In order to dispatch commands to the device for execution we must also retrieve the `CommandQueue` object associated with `device`. Commands will be dispatched through this object to be executed on the device.
The `Program` object is created to encapsulate our kernels and buffers.

# Initialize source data

``` cpp
constexpr uint32_t M = 16;
constexpr uint32_t N = 1;
uint32_t num_values = M * N;
tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
std::vector<bfloat16> src_vec(M * N, bfloat16(0.0f));
for (uint32_t i = 0; i < src_vec.size(); i++) {
    src_vec[i] = bfloat16((i + 1) * 2);
}
constexpr size_t data_size = sizeof(bfloat16);
```

For this example, we will be using a tensor with shape `(16, 1)` as our source vector. The vector contains the values `{2, 4, 6, ... , 28, 30, 32}`.
Note that the process of initializing the source data can be done at any point before loading the values into the DRAM and executing the program.

# Core designation

``` cpp
CoreCoord start_core = {0, 0};
CoreCoord end_core = {0, 3};
uint32_t num_cores = 4;
CoreRange cores(start_core, end_core);
```

For simplicity, we will be using just 4 cores in this program. In order to designate which cores to use, we must designate the logical coordinates of the start and end cores.
The start and end cores must form a rectangle in order for the cores to be utilized; the intermediary cores are determined by the start and end cores.

# Sharding specifications

``` cpp
uint32_t shard_height = num_values / num_cores / data_size;
uint32_t shard_width = N;
uint32_t shard_size = shard_height * shard_width;
uint32_t input_unit_size = sizeof(uint32_t);
uint32_t shard_width_bytes = shard_width * data_size;
uint32_t num_units_per_row = shard_width * input_unit_size;
uint32_t padded_offset_bytes = align(input_unit_size, device->get_allocator_alignment());
```

In order to shard the correct data segments to the respective core, we indicate the shard height, width, size, and other data for the kernel function.
For this situation, 16 units of data will be sharded across 4 cores; each core will have 4 units of data in their corresponding circular buffer.
The `padded_offset_bytes` is set to ensure that the correct address is read from the kernel function when moving data to the circular buffer; in this case, the addresses are aligned to L1 memory.
This example demonstrates height sharding; the shard height is therefore set to evenly distribute the number of vector values across the cores.
If the sharding strategy was different (i.e. width sharding or block sharding), the appropriate values for both the shard height and width would need to be set.

Since we are using BFloat16 tensor values and our page sizes are `uint32_t`, each page will contain 2 tensor values.

# Configure interleaved DRAM buffer

``` cpp
uint32_t src_buffer_size = input_unit_size * num_values / data_size;
tt_metal::InterleavedBufferConfig input_dram_config {
    .device = device,
    .size = src_buffer_size,
    .page_size = input_unit_size,
    .buffer_type = tt_metal::BufferType::DRAM
};
std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
uint32_t src_addr = src_buffer->address();
```

Data will be read to the circular buffers on each core through the DRAM buffer, which is in an interleaved format. In the configuration, the size of the buffer in bytes and page size is indicated.

# Configure circular buffers

``` cpp
bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
uint32_t input_cb_index = CB::c_in0;
CircularBufferConfig input_cb_config = CircularBufferConfig(shard_size * input_unit_size, {{input_cb_index, cb_data_format}})
    .set_page_size(input_cb_index, input_unit_size);
auto cb_input = tt_metal::CreateCircularBuffer(program, cores, input_cb_config);
```

Across each core, the `CircularBuffer` indicated by the index corresponding to `CB::c_in0` will be used to store the data. Through the `CircularBufferConfig` object, we specify the total size of the buffer, which is dependent on the shard and data size, and we also specify the page size.
The corresponding `CircularBuffer` objects are then allocated with this configuration across each of the designated cores.

# Create data movement kernels for sharding

``` cpp
std::vector<uint32_t> reader_compile_time_args = {
    (std::uint32_t)input_cb_index,
    (std::uint32_t)src_is_dram};
auto reader_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/sharding/kernels/reader_sharded_rm.cpp",
    cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
```

Sharding is inherently a data movement operation; as such, a data movement kernel will be used for this. RISC-V processor 1 on each core is designated to run this data movement kernel. The `CircularBuffer`
index and the source buffer type are set as compile time arguments for the kernel in order to move the data to the correct destination using the appropriate configuration.

# Set kernel function runtime arguments

``` cpp
uint32_t curr_idx_h = 0;
uint32_t curr_idx_w = 0;
for (uint32_t i = 0; i < num_cores; i++) {
    CoreCoord core = {0, i};
    tt_metal::SetRuntimeArgs(
        program,
        reader_id,
        core,
        {src_addr,
        input_unit_size,
        shard_height,
        shard_width_bytes,
        padded_offset_bytes,
        curr_idx_h,
        i});
    curr_idx_w += input_unit_size;
    if (curr_idx_w >= num_units_per_row) {
        curr_idx_w = 0;
        curr_idx_h += shard_height;
    }
}
```

For each core, the kernel function runtime arguments are set as a prerequisite for program execution on the device. The kernel function uses the shard specifications and the source data configuration in order to determine which data segments are moved to a given core\'s L1 memory.

# Sharding kernel function

``` cpp
const InterleavedAddrGen<src_is_dram> s0 = {
    .bank_base_address = src_addr,
    .page_size = stick_size
};
uint32_t stick_id = start_id;
cb_reserve_back(cb_id_in0, shard_height);
uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
DPRINT_DATA0(DPRINT << "Core (0," << current_core << "): ");
for (uint32_t h = 0; h < shard_height; ++h) {
    uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
    noc_async_read(src_noc_addr, l1_write_addr, stick_size);
    uint32_t* read_ptr = (uint32_t*)l1_write_addr;
    DPRINT_DATA0(DPRINT << (uint16_t)*read_ptr << " ");
    DPRINT_DATA0(DPRINT << (uint16_t)(*read_ptr >> 16) << " ");
    stick_id++;
    l1_write_addr += padded_offset_bytes;
}
DPRINT_DATA0(DPRINT << ENDL());
noc_async_read_barrier();
cb_push_back(cb_id_in0, shard_height);
```

The `InterleavedAddrGen` object allows us to retrieve the data stored in the DRAM by incrementing by stick size. The stick size determines the difference between addresses of each piece of source data in the DRAM buffer; its value in this case is the size of a `uint32_t` data type.
Each stick will then contain 2 of the BFloat16 tensor values. The generator is used to ensure that the correct address is read from the DRAM buffer into a given core\'s L1 memory.

In the indicated kernel function file, we call `cb_reserve_back` for the indicated circular buffer in order to wait for space of the specified data segment size to be free, then load the data into the circular buffer by reading the value from the NoC address (indicated by `src_noc_addr`) to the L1 memory address (indicated by `l1_write_addr`).

In our example, the `shard_height` is 2, so each of the cores will read in 2 values of the given page size at incremental addresses from the source vector. Then, the BFloat16 values are read and printed out from the kernel.
For this example, each core will read in 4 BFloat16 values; there will be 2 pages read, each with 2 BFloat16 values.
Once the data is written to the circular buffer, `cb_push_back` is called to make the data visible.

# Program execution

``` cpp
EnqueueWriteBuffer(cq, src_buffer, src_vec.data(), false);
EnqueueProgram(cq, program, false);
Finish(cq);
CloseDevice(device);
```

`EnqueueWriteBuffer` is called in order to load the source vector into the interleaved DRAM buffer.
`EnqueueProgram` is then called to dispatch the program to the device for execution. Upon conclusion of the program execution, the device is closed.

# Conclusion

This example walks through a basic example of sharding with a simple untilized source vector, with height sharding as the sharding strategy.
For other strategies, such as block and width sharding, modifications will need to be made to the shard specifications, adjusting for the appropriate shard height and width.
