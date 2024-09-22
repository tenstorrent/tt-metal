# Tensor Padding (Multicore)

In this example, we will implement a basic TT-Metalium program for padding an input tensor. The program will have the following steps:

1. Instantiate device and program.
2. Initialize the source data, as well as the pad value.
3. Designate cores for utilization.
4. Create buffers to be used for moving input and output data.
5. Create data movement kernels.
6. Set kernel runtime arguments.
7. Dispatch program to device for execution.
8. Close device.

The code for this program can be found in [pad_multi_core.cpp](../../../tt_metal/programming_examples/pad/pad_multi_core.cpp).

The following commands will build and execute the code for this example. Environment variables may be modified based on the latest specifications.
```bash
    export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh
    ./build/programming_examples/pad_multi_core
```
## Device setup

``` cpp
int device_id = 0;
Device *device = CreateDevice(device_id);
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
```

In order to access the hardware capabilities of the accelerator, we retrieve the `Device` object, referencing the Tenstorrent device with a `device_id` of 0. We also instantiate the `CommandQueue` and `Program` objects in order to later dispatch the program to the device for execution.

## Initialize data

``` cpp
constexpr uint32_t src_M = 8;
constexpr uint32_t src_N = 4;
constexpr uint32_t packed_data_size = sizeof(uint32_t);
constexpr uint32_t unpacked_data_size = sizeof(bfloat16);
constexpr uint32_t packing_ratio = packed_data_size / unpacked_data_size;
uint32_t src_num_values_unpacked = src_M * src_N;
uint32_t src_num_values_packed = src_num_values_unpacked / packing_ratio;
std::vector<uint32_t> src_vec(src_num_values_packed, 0);
for (uint32_t i = 0; i < src_vec.size(); i++) {
    bfloat16 bfloat_val1 = bfloat16(2 * i + 1);
    bfloat16 bfloat_val2 = bfloat16(2 * i + 2);
    src_vec[i] = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat_val1, bfloat_val2));
}
```

For our example, we will be using a simple tensor with shape (8, 4). We fill this vector with an arbitrary set of values; in this case, it will be values that lie in the range \[1, 32\]. This tensor contains the values that we will send to the device to pad. 
Note that we are using a source vector with `uint32_t` values, but the tensor values themselves are `bfloat16`. 

This example demonstrates a manual way to pack `bfloat16` values into `uint32_t` for another layer of parallelism.

``` cpp
bfloat16 pad_value = bfloat16(2);
std::vector<uint32_t> pad_vec(1, pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(pad_value, pad_value)));
```

The program will pad the input tensor with a given pad value; in this case, it is 2. We create a vector containing a single `uint32_t` element, which contains two `bfloat16` values itself. This value will be accessed from the DRAM by the kernels in order to pad the tensors.

``` cpp
constexpr uint32_t dst_M = 8;
constexpr uint32_t dst_N = 8;
uint32_t dst_num_values_unpacked = dst_M * dst_N;
uint32_t dst_num_values_packed = dst_num_values_unpacked / packing_ratio;
std::vector<uint32_t> dst_vec(dst_num_values_packed, 0);
```

The input tensor will be padded to form an output tensor of shape (8, 8). We set this output tensor with the intended dimensions and a constant initial value of 0. 

This output tensor will store the values of the padded tensor retrieved from the device. 
Since this example will demonstrate padding from a tensor of shape (8, 4) to a tensor of shape (8, 8), only the second dimension will receive padding. 
The code in this example will focus on padding of a single dimension.

## Designate cores for utilization

``` cpp
CoreCoord start_core = {0, 0};
CoreCoord end_core = {0, 3};
uint32_t num_cores = 4;
CoreRange cores(start_core, end_core);
uint32_t num_cores = cores.size();
```

This example will send data to 4 cores for the padding operation. We specify the range of cores to be those given by the coordinates (0, 0) through (0, 3).

## Configure and create DRAM buffers

``` cpp
uint32_t src_buffer_size = packed_data_size * src_num_values_packed;
tt_metal::InterleavedBufferConfig input_dram_config {
    .device = device,
    .size = src_buffer_size,
    .page_size = packed_data_size,
    .buffer_type = tt_metal::BufferType::DRAM
};
std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
uint32_t src_addr = src_buffer->address();
```

We configure the DRAM buffer for the source data (input tensor). The page size will be the size of each tensor value and the total buffer size will be the total tensor size in bytes. When each core executes its reader kernel, the values from this buffer will be read into the corresponding `CircularBuffer`.

``` cpp
uint32_t pad_buffer_size = packed_data_size * pad_vec.size();
tt_metal::InterleavedBufferConfig pad_dram_config {
    .device = device,
    .size = pad_buffer_size,
    .page_size = packed_data_size,
    .buffer_type = tt_metal::BufferType::DRAM
};
std::shared_ptr<tt::tt_metal::Buffer> pad_buffer = CreateBuffer(pad_dram_config);
uint32_t pad_addr = pad_buffer->address();
```

We create another DRAM buffer for the pad value. This buffer will only contain a single value (the pad value). The reader kernel will use the value in this buffer to pad the corresponding data in the `CircularBuffer`; once this kernel is executed, the correspoding tensor row will be padded and be stored in the `CircularBuffer`.

``` cpp
uint32_t dst_buffer_size = packed_data_size * dst_num_values_packed;
tt_metal::InterleavedBufferConfig output_dram_config {
    .device = device,
    .size = dst_buffer_size,
    .page_size = packed_data_size,
    .buffer_type = tt_metal::BufferType::DRAM
};
std::shared_ptr<tt::tt_metal::Buffer> dst_buffer = CreateBuffer(output_dram_config);
uint32_t dst_addr = dst_buffer->address();
```

The DRAM buffer configuration for the output tensor is similar to that for the input tensor. The only modification is that the buffer size must be adjusted to account for the shape of the output tensor, which is larger due to padding.

# Configure and create CircularBuffer

``` cpp
uint32_t cb_id = CB::c_in0;
tt::DataFormat cb_data_format = tt::DataFormat::UInt32;
CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(dst_N * packed_data_size * 2, {{cb_id, cb_data_format}})
    .set_page_size(cb_id, packed_data_size);
auto cb_src = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);
```

We designate a `CircularBuffer` index to be accessed across each of the utilized cores for this program. For this program, the first dimension of the input tensor is 8, and we are parallelizing the operation through chunking the first dimension across 4 cores. Therefore, each core will have 2 rows of the input tensor. 
Since each of these rows will be padded to match the size of the output tensor\'s second dimension, we set the circular buffer size accordingly.

# Create data movement kernels

``` cpp
bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
bool pad_is_dram = pad_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
std::vector<uint32_t> reader_compile_time_args = {(uint32_t) src_is_dram, (uint32_t) pad_is_dram};
std::vector<uint32_t> writer_compile_time_args = {(uint32_t) dst_is_dram};
```

We set the compile-time arguments of the respective kernel functions to be the buffer types, in order to generate the correct addresses for manipulating the data stored inside of the DRAM buffers.

``` cpp
KernelHandle reader_id = CreateKernel(program,
    "tt_metal/programming_examples/pad/kernels/pad_reader_dims_rm_interleaved.cpp",
    cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
KernelHandle writer_id = CreateKernel(program,
    "tt_metal/programming_examples/pad/kernels/pad_writer_dims_rm_interleaved.cpp",
    cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});
```

Using the compile-time arguments and the respective kernel functions, which are written in the designated C++ files, we create the kernels to dispatch to the cores for execution. 
Since the operation of padding involves moving data, we specify these kernels with a data movement configuration. The reader kernel will read in the input tensor data from the DRAM and use the stored pad value to pad each of the rows. 
The writer kernel will then read the padded tensor back to host, where it is stored in `dst_vec`. Each kernel uses a different NoC.

# Set kernel runtime arguments

``` cpp
uint32_t start_src_idx = 0;
uint32_t start_dst_idx = 0;
uint32_t num_rows_per_core = src_M / num_cores;
uint32_t row_size_diff = dst_N - src_N;
uint32_t num_packed_row_src = src_N / packing_ratio;
uint32_t num_packed_row_dst = dst_N / packing_ratio;
uint32_t num_src_sticks_per_core = num_packed_row_src * num_rows_per_core;
for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
    CoreCoord core = {0, core_idx};
    tt_metal::SetRuntimeArgs(
        program,
        reader_id,
        core,
        {src_addr,
         pad_addr,
         start_src_idx,
         row_size_diff / packing_ratio,
         num_packed_row_dst,
         packed_data_size,
         num_rows_per_core
        }
    );
    tt_metal::SetRuntimeArgs(
        program,
        writer_id,
        core,
        {dst_addr,
         start_dst_idx,
         num_packed_row_dst,
         packed_data_size,
         num_rows_per_core
        }
    );
    start_src_idx += num_src_sticks_per_core;
    start_dst_idx += num_packed_row_dst * num_rows_per_core;
}
```

We specify `start_src_idx` in order for each core to access the intended input tensor values through the kernel. Similarly, `start_dst_idx` is used in order for each core to access the intended output tensor values.
We iterate through the range of cores that are designated for utilization and set the corresponding runtime arguments for the reader and writer kernels. 

Note that for this example, on the host side, we define the kernel arguments based on the size of the packed data (`uint32_t`).

# Reader kernel function

``` cpp
const InterleavedAddrGen<src_is_dram> s0 = {
    .bank_base_address = src_addr,
    .page_size = data_size_bytes
};
const InterleavedAddrGen<pad_is_dram> s1 = {
    .bank_base_address = pad_addr,
    .page_size = data_size_bytes
};
```

In the reader kernel, we specify the DRAM buffer address generators for the input tensor and the buffer containing the pad value.

``` cpp
uint32_t src_stick_id = start_src_stick_id;
uint32_t src_start_col_idx = row_size_diff / 2;
uint32_t src_end_col_idx = dst_N - src_start_col_idx;
for (uint32_t i = 0; i < num_rows_per_core; i++) {
    for (uint32_t dst_col_idx = 0; dst_col_idx < dst_N; dst_col_idx++) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);
        if (dst_col_idx < src_start_col_idx || dst_col_idx >= src_end_col_idx) {
            uint64_t pad_noc_addr = get_noc_addr(0, s1);
            noc_async_read(pad_noc_addr, l1_addr, data_size_bytes);
        }
        else {
            uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
            noc_async_read(src_noc_addr, l1_addr, data_size_bytes);
            src_stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
        l1_addr += data_size_bytes;
    }
}
```

The reader kernel is designed to focus on tensor padding along the second dimension. Each core will iterate through a given number of rows of the tensor based on the number of cores used and the number of rows in the tensor; in this case, it is 2, since there are 8 rows divided evenly among 4 cores. 
Using the start and end index, the kernel reads the pad value into the circular buffer until it is able to pad the source tensor row into the intended shape, then reads in the source data before padding the rest of the row.

# Writer kernel function

``` cpp
const InterleavedAddrGen<dst_is_dram> s1 = {
    .bank_base_address = dst_addr,
    .page_size = data_size_bytes
};

uint32_t dst_stick_id = start_dst_stick_id;
for (uint32_t row_idx = 0; row_idx < num_rows_per_core; row_idx++) {
    for (uint32_t dst_col_idx = 0; dst_col_idx < dst_N; dst_col_idx++) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_addr = get_read_ptr(cb_id);
        uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s0);
        noc_async_write(l1_addr, dst_noc_addr, data_size_bytes);
        noc_async_write_barrier();
        dst_stick_id++;
        cb_pop_front(cb_id, 1);
    }
}
```

Once the reader kernel is finished padding the tensor and storing the new data into the circular buffer, the writer kernel writes the data stored in the circular buffer into the DRAM buffer corresponding to the destination vector (output tensor).

# Dispatch program to device for execution

``` cpp
EnqueueWriteBuffer(cq, src_buffer, src_vec.data(), false);
EnqueueWriteBuffer(cq, pad_buffer, pad_vec.data(), false);
EnqueueProgram(cq, program, false);
EnqueueReadBuffer(cq, dst_buffer, dst_vec.data(), false);
Finish(cq);
/* ... */
CloseDevice(device);
```

In order to send the program to the device for execution, we call `EnqueueWriteBuffer` to move the input tensor data into its corresponding DRAM buffer, and also move the pad value into its corresponding DRAM buffer. 
We then call `EnqueueReadBuffer` to move the output tensor data from its corresponding DRAM buffer to the destination vector.

# Summary

For this program, the data flow is as follows.

1. Create the input tensor and initialize its data.
2. Designate the pad value and insert it into a pad vector containing only the pad value.
3. Move the input tensor (source vector) into its corresponding DRAM buffer.
4. Move the pad vector into its corresponding DRAM buffer.
5. Each core uses the reader kernel to read its tensor values from the DRAM into the circular buffer, while reading the pad value from the DRAM to pad each row.
6. Each core uses the writer kernel to write the padded tensor rows from the circular buffer to the output tensor\'s DRAM buffer.
7. Move the output tensor data from the DRAM to the destination vector.
