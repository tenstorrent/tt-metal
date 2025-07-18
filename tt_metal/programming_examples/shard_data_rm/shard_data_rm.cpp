// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // This program demonstrates basic data sharding using TT-Metalium APIs. The example shows how to split a buffer
    // into smaller chunks (shards) and distribute them to multiple cores' local buffers. Sharding can improve access
    // speed and reduce NoC (Network-on-Chip) traffic, at the cost of some setup overhead.
    // Here, a row-major matrix of shape (M, N) is sharded across 4 cores along the M dimension, resulting in 4 shards
    // of shape (M/4, N).

    // Select device 0 and create a command queue and program object for kernel/buffer management.
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);   // Acquire handle to hardware device
    CommandQueue& cq = device->command_queue();  // Command queue for dispatching operations
    Program program = CreateProgram();           // Encapsulates kernels and buffers

    const char* dprint_env = std::getenv("TT_METAL_DPRINT_CORES");
    if (!dprint_env || std::string(dprint_env).empty()) {
        fmt::print(stderr, "[WARNING] TT_METAL_DPRINT_CORES is not set.\n");
        fmt::print(stderr, "          Device-side DPRINT output will not appear.\n");
        fmt::print(stderr, "          To enable output, run:\n");
        fmt::print(stderr, "              export TT_METAL_DPRINT_CORES='(0,0)-(0,3)'\n");
    }

    // Create a vector of shape (16, 1) with values {2, 4, 6, ..., 32} in bfloat16 format.
    // See the following visualization of the data layout (transposed to make showing easier in text)
    // data: xxxxyyyyzzzzwwww ] width
    //       |       |      |
    //       0       8      16
    //       [     height    ]
    // Is split across 4 cores, each core will handle 4 elements of 4x1.
    // core 0: xxxx
    // core 1: yyyy
    // core 2: zzzz
    // core 3: wwww
    // Which in each shard. A group of value is called a "stick" as in "stick of RGB values (in NHWC images/convolution)
    // For instance in this example, each stick is 2 bfloat16 values (4 bytes).
    // core X: x      x      x      x
    //         |  stick 0 | stick 1 |
    // Thus in our example. Each core handles 4 values, or 2 sticks of 2 bfloat16 values each.
    constexpr uint32_t M = 16;  // Number of columns
    constexpr uint32_t N = 1;   // Number of rows
    // Use 4 cores: (0,0), (0,1), (0,2), (0,3) for sharding.
    constexpr uint32_t num_cores = 4;  // Number of cores
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {0, num_cores - 1};
    CoreRange cores(start_core, end_core);  // Rectangle of cores (note: end is inclusive)

    static_assert(M % num_cores == 0, "M must be divisible by 4 (number of cores) for this example");
    const uint32_t num_values = M * N;                          // Total elements
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;  // Data format for circular buffer
    std::vector<bfloat16> src_vec(num_values, bfloat16(0.0f));  // Source vector
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        src_vec[i] = bfloat16((float)(i + 1) * 2);  // Fill with {2, 4, ..., 32}
    }
    constexpr size_t data_size = sizeof(bfloat16);  // Size of each element

    // --- Sharding Specifications ---
    // Each core will receive 4 values (16 total / 4 cores).
    // Shard height: number of rows per core (in units of uint32_t pages)
    // Shard width: number of columns per core
    // Shard size: total elements per core
    // input_unit_size: size of a page (uint32_t, alignment requirement)
    // shard_width_bytes: width in bytes for each shard
    uint32_t shard_height = M / num_cores;                       // Height per shard (see note below)
    uint32_t shard_width = N;                                    // Width per shard
    uint32_t shard_size = shard_height * shard_width;            // Elements per shard
    constexpr uint32_t input_unit_size = sizeof(uint32_t);       // Page size for buffer (alignment need on core)

    // Note: Since bfloat16 is 2 bytes and uint32_t is 4 bytes, each page contains 2 bfloat16 values.
    // The division by data_size ensures correct mapping of values to pages.
    uint32_t padded_offset_bytes = align(input_unit_size, device->allocator()->get_alignment(BufferType::DRAM));

    // configure and create interleaved DRAM buffer to insert source data into
    uint32_t src_buffer_size = num_values * data_size;
    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = src_buffer_size,
        .page_size = input_unit_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
    const uint32_t src_addr = src_buffer->address();
    EnqueueWriteBuffer(cq, src_buffer, src_vec.data(), false);

    // configure and create circular buffers with the same address on each of the designated cores
    // Create a circular buffer on each core to hold its shard of data
    uint32_t input_cb_index = CBIndex::c_0;
    size_t cb_total_size = shard_size * input_unit_size;  // Total buffer size per core
    CircularBufferConfig input_cb_config = CircularBufferConfig(cb_total_size, {{input_cb_index, cb_data_format}})
                                               .set_page_size(input_cb_index, input_unit_size);

    CreateCircularBuffer(program, cores, input_cb_config);

    // create data movement kernel to shard data
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    const uint32_t values_per_stick = input_unit_size / data_size;  // Number of bfloat16 values per stick
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index, (std::uint32_t)src_is_dram, input_unit_size, values_per_stick};
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "shard_data_rm/kernels/reader_sharded_rm.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    // Set the parameters for each core to read its shard of data
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {0, i};
        uint32_t idx_h = i * shard_height;
        // stick_id is the index of the first stick in the shard (recall a stick is 2 bfloat16 values in this example)
        uint32_t stick_id = idx_h * shard_width / values_per_stick;
        tt_metal::SetRuntimeArgs(program, reader_id, core, {src_addr, shard_size, padded_offset_bytes, stick_id});
    }

    fmt::print("Original tensor values: ");
    for (uint32_t src_vec_idx = 0; src_vec_idx < src_vec.size(); src_vec_idx++) {
        fmt::print("{:0.1f} ", src_vec[src_vec_idx].to_float());
    }
    fmt::print("\n");

    // start/finish program and close device
    EnqueueProgram(cq, program, false);
    // Kernel prints to console. No need to print the output here.
    //
    // You should see the following output in the console:
    // 0:(x=0,y=0):BR: Core (0,0): 2 4 6 8
    // 0:(x=0,y=1):BR: Core (0,1): 10 12 14 16
    // 0:(x=0,y=2):BR: Core (0,2): 18 20 22 24
    // 0:(x=0,y=3):BR: Core (0,3): 26 28 30 32
    Finish(cq);
    CloseDevice(device);
    fmt::print("Program finished successfully.\n");
}
