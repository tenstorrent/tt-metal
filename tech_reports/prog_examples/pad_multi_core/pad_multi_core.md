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

To build and execute, you may use the following commands:
Then run the following:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_pad_multi_core
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
    constexpr uint32_t src_M = 64;
    constexpr uint32_t src_N = 32;
    constexpr uint32_t packed_data_size = sizeof(uint32_t);
    constexpr uint32_t unpacked_data_size = sizeof(bfloat16);
    constexpr uint32_t packing_ratio = packed_data_size / unpacked_data_size;
    uint32_t src_num_values_unpacked = src_M * src_N;
    uint32_t src_num_values_packed = src_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> src_vec(src_num_values_packed, 0);
    // source vector = {1, 2, 3, ... , 30, 31, 32, ... , 2048}
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        bfloat16 bfloat_val1 = bfloat16(2 * i + 1);
        bfloat16 bfloat_val2 = bfloat16(2 * i + 2);
        src_vec[i] = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat_val1, bfloat_val2));
    }
```

For our example, we will be using a simple tensor with shape (64, 32). We fill this vector with an arbitrary set of values; in this case, it will be values that lie in the range \[1, 2048\]. This tensor contains the values that we will send to the device to pad.
Note that we are using a source vector with `uint32_t` values, but the tensor values themselves are `bfloat16`.

This example demonstrates a manual way to pack `bfloat16` values into `uint32_t` for another layer of parallelism.

``` cpp
bfloat16 pad_value = bfloat16(2);
std::vector<uint32_t> pad_vec(1, pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(pad_value, pad_value)));
```

The program will pad the input tensor with a given pad value; in this case, it is 2. We create a vector containing a single `uint32_t` element, which contains two `bfloat16` values itself. This value will be accessed from the DRAM by the kernels in order to pad the tensors.

``` cpp
constexpr uint32_t dst_M = 64;
constexpr uint32_t dst_N = 64;
uint32_t dst_num_values_unpacked = dst_M * dst_N;
uint32_t dst_num_values_packed = dst_num_values_unpacked / packing_ratio;
std::vector<uint32_t> dst_vec(dst_num_values_packed, 0);
```

The input tensor will be padded to form an output tensor of shape (64, 64). We set this output tensor with the intended dimensions and a constant initial value of 0.

This output tensor will store the values of the padded tensor retrieved from the device.
Since this example will demonstrate padding from a tensor of shape (64, 32) to a tensor of shape (64, 64), only the second dimension will receive padding.
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
    // configure circular buffers expected by TTNN v2 reader/writer: c_0 (main), c_1 (pad), c_2 (align)
    constexpr uint32_t cb0 = CBIndex::c_0;
    constexpr uint32_t cb1 = CBIndex::c_1;
    constexpr uint32_t cb2 = CBIndex::c_2;
    tt::DataFormat cb_df = tt::DataFormat::UInt32;
    const uint32_t stick_size_bytes = packed_data_size;              // 4 bytes per stick (one packed uint32)
    // Use TTNN v2 row-major minimum of 64B per stick in L1 for padding pattern
    const uint32_t stick_size_padded = 64;
    const uint32_t stick_size_padded_aligned = 64;
    const uint32_t num_packed_row_src = src_N / packing_ratio;
    const uint32_t num_packed_row_dst = dst_N / packing_ratio;
    const uint32_t num_sticks_per_barrier = num_packed_row_dst;      // process one row per barrier
    // c_0 needs capacity for one row of padded sticks
    CircularBufferConfig cb_cfg = tt::tt_metal::CircularBufferConfig(
                                       num_sticks_per_barrier * stick_size_padded_aligned,
                                       {{cb0, cb_df}, {cb1, cb_df}, {cb2, cb_df}})
                                       .set_page_size(cb0, stick_size_padded_aligned)
                                       .set_page_size(cb1, stick_size_padded)
                                       .set_page_size(cb2, stick_size_bytes);
    tt_metal::CreateCircularBuffer(program, cores, cb_cfg);
```

We designate a `CircularBuffer` index to be accessed across each of the utilized cores for this program. For this program, the first dimension of the input tensor is 64, and we are parallelizing the operation through chunking the first dimension across 4 cores. Therefore, each core will have 16 rows of the input tensor.
Since each of these rows will be padded to match the size of the output tensor\'s second dimension, we set the circular buffer size accordingly.

# Create data movement kernels

``` cpp
    std::vector<uint32_t> reader_compile_time_args;
    // N, H, C (treat rows as N, single H=1, columns (packed) as C)
    reader_compile_time_args.push_back(src_M);                // N
    reader_compile_time_args.push_back(1);                    // H
    reader_compile_time_args.push_back(num_packed_row_src);   // C
    reader_compile_time_args.push_back(stick_size_bytes);     // stick_size_bytes
    reader_compile_time_args.push_back(src_M);                // N_padded (same)
    reader_compile_time_args.push_back(1);                    // H_padded
    reader_compile_time_args.push_back(num_packed_row_dst);   // C_padded
    reader_compile_time_args.push_back(stick_size_padded);    // stick_size_padded (32B pad pattern)
    reader_compile_time_args.push_back(0);                    // stick_size_padded_front (left-align)
    reader_compile_time_args.push_back(0);                    // stick_size_padded_end
    reader_compile_time_args.push_back(1);                    // num_zero_pad_sticks_read
    reader_compile_time_args.push_back(stick_size_padded);    // last_zero_stick_size (32)
    reader_compile_time_args.push_back(1);                    // not_pad_by_zero
    // pass packed pad value directly as compile-time arg as TTNN does
    reader_compile_time_args.push_back(pad_vec[0]);           // packed_pad_value
    reader_compile_time_args.push_back(stick_size_padded);    // row_major_min_bytes (32)
    reader_compile_time_args.push_back(0);                    // num_front_pad_sticks_read
    reader_compile_time_args.push_back(0);                    // num_end_pad_sticks_read
    reader_compile_time_args.push_back(1);                    // num_sticks_padded_read per stick
    reader_compile_time_args.push_back(stick_size_padded_aligned); // stick_size_padded_aligned (32)
    reader_compile_time_args.push_back(0);                    // unaligned = false
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        cb0, stick_size_bytes, stick_size_padded_aligned};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

We set the compile-time arguments of the respective kernel functions to be the buffer types, in order to generate the correct addresses for manipulating the data stored inside of the DRAM buffers.

``` cpp
    KernelHandle reader_id = CreateKernel(
        program,
        "tt_metal/programming_examples/pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    KernelHandle writer_id = CreateKernel(
        program,
        "tt_metal/programming_examples/pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});
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
    uint32_t num_src_sticks_per_core = num_packed_row_src * num_rows_per_core;
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = {0, core_idx};
        uint32_t num_sticks_per_core = num_rows_per_core * num_packed_row_dst;
        uint32_t num_sticks_per_barrier_rt = num_packed_row_dst;  // one row per barrier
        // Reader v2 runtime: src_addr, num_sticks_per_core, num_sticks_per_barrier, start_id, front_pad_n,c,h, start_dim_offset[0..3]
        std::vector<uint32_t> reader_rt = {src_addr,
                                           num_sticks_per_core,
                                           num_sticks_per_barrier_rt,
                                           start_src_idx,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0};
        tt_metal::SetRuntimeArgs(program, reader_id, core, reader_rt);
        // Writer v2 runtime: dst_addr, num_sticks_per_core, num_sticks_per_barrier, start_id
        std::vector<uint32_t> writer_rt = {dst_addr, num_sticks_per_core, num_sticks_per_barrier_rt, start_dst_idx};
        tt_metal::SetRuntimeArgs(program, writer_id, core, writer_rt);
        start_src_idx += num_src_sticks_per_core;
        start_dst_idx += num_packed_row_dst * num_rows_per_core;
    }
```

We specify `start_src_idx` in order for each core to access the intended input tensor values through the kernel. Similarly, `start_dst_idx` is used in order for each core to access the intended output tensor values.
We iterate through the range of cores that are designated for utilization and set the corresponding runtime arguments for the reader and writer kernels.

Note that for this example, on the host side, we define the kernel arguments based on the size of the packed data (`uint32_t`).

# Reader kernel function

``` cpp
const auto s = TensorAccessor(src_args, src_addr, stick_size_bytes);
```

In the reader kernel, we specify the DRAM buffer address generators for the input tensor and the buffer containing the pad value.

``` cpp
    uint32_t i_stick = start_id;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_reserve_back(cb_in0, num_sticks_per_barrier);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);
        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            uint64_t read_noc_addr = get_noc_addr(i_stick, s);
            // Seed pad value in first word to guarantee padding when writer writes only stick_size_bytes
            *((volatile tt_l1_ptr uint32_t*)l1_write_addr) = packed_pad_value;
            if (read_stick) {
                if constexpr (front_padding) {
                    noc_async_read(read_noc_addr, get_write_ptr(cb_pad_align), stick_size_bytes);
                    noc_async_read_barrier();
                    memmove((void*)(l1_write_addr + stick_size_padded_front), (void*)(get_read_ptr(cb_pad_align)), (size_t)(stick_size_bytes));
                } else if constexpr (unaligned) {
                    noc_async_read(read_noc_addr, get_write_ptr(cb_pad_align), stick_size_bytes);
                    noc_async_read_barrier();
                    noc_async_read(pad_align_noc_addr, l1_write_addr, stick_size_bytes);
                } else {
                    noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
                }
                i_stick++;
            }
            l1_write_addr += stick_size_padded_aligned;
            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, num_sticks_per_barrier);
    }
```

The reader kernel is designed to focus on tensor padding along the second dimension. Each core will iterate through a given number of rows of the tensor based on the number of cores used and the number of rows in the tensor; in this case, it is 2, since there are 64 rows divided evenly among 4 cores.
Using the start and end index, the kernel reads the pad value into the circular buffer until it is able to pad the source tensor row into the intended shape, then reads in the source data before padding the rest of the row.

# Writer kernel function

``` cpp
    const auto s = TensorAccessor(dst_args, dst_addr, stick_size_bytes);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        cb_wait_front(cb_out0, num_sticks_per_barrier);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);
        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            uint64_t write_noc_addr = get_noc_addr(i_stick, s);
            noc_async_write(l1_read_addr, write_noc_addr, stick_size_bytes);
            l1_read_addr += stick_size_padded_aligned;
            i_stick += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out0, num_sticks_per_barrier);
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
