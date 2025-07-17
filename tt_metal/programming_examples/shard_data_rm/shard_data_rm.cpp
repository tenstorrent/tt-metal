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
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // This program demonstrates data sharding at the lowest level using TT-Metalium APIs. This example is more abstract
    // but shows the core concepts distrubuted data across multiple cores's local buffers
    // Sharding splits a buffer into smaller chunks (shards) and distributes them to multiple cores' local buffers.
    // Sharding enables faster access and reduces NoC (Network-on-Chip) traffic. At the cost of having an initial
    // setup overhead.
    // In this example, we shard a matrix of shape (M, N) into 4 cores along the M dimension resulting in 4 shards of
    // shape (M/4, N).

    // Select device 0 and create a command queue and program object for kernel/buffer management.
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);   // Acquire handle to hardware device
    CommandQueue& cq = device->command_queue();  // Command queue for dispatching operations
    Program program = CreateProgram();           // Encapsulates kernels and buffers

    // Create a vector of shape (16, 1) with values {2, 4, 6, ..., 32} in bfloat16 format.
    // See the following visualization of the data layout (transposed to make showing easier in text)
    // data: xxxxxxxxxxxxxxxx ] width
    //       |       |      |
    //       0       8      16
    //       [     height    ]
    // Is split across 4 cores, each core will handle 4 elements of 4x1.
    // core 0: xxxx
    // core 1: xxxx
    // core 2: xxxx
    // core 3: xxxx
    constexpr uint32_t M = 16;  // Number of rows
    constexpr uint32_t N = 1;   // Number of columns
    static_assert(M % 4 == 0, "M must be divisible by 4 for this example");
    uint32_t num_values = M * N;                                // Total elements
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;  // Data format for circular buffer
    std::vector<bfloat16> src_vec(num_values, bfloat16(0.0f));  // Source vector
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        src_vec[i] = bfloat16((i + 1) * 2);  // Fill with {2, 4, ..., 32}
    }
    constexpr size_t data_size = sizeof(bfloat16);  // Size of each element

    // Use 4 cores: (0,0), (0,1), (0,2), (0,3) for sharding.
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {0, 3};
    CoreRange cores(start_core, end_core);                 // Rectangle of cores (note: end is inclusive)
    uint32_t num_cores = CoreRangeSet(cores).num_cores();  // Number of cores

    // --- Sharding Specifications ---
    // Each core will receive 4 values (16 total / 4 cores).
    // Shard height: number of rows per core (in units of uint32_t pages)
    // Shard width: number of columns per core
    // Shard size: total elements per core
    // input_unit_size: size of a page (uint32_t)
    // shard_width_bytes: width in bytes for each shard
    // num_units_per_row: number of uint32_t units per row
    // padded_offset_bytes: alignment for buffer writes
    uint32_t shard_height = num_values / num_cores / data_size;  // Height per shard (see note below)
    uint32_t shard_width = N;                                    // Width per shard
    uint32_t shard_size = shard_height * shard_width;            // Elements per shard
    uint32_t input_unit_size = sizeof(uint32_t);                 // Page size for buffer
    uint32_t shard_width_bytes = shard_width * data_size;        // Bytes per row
    uint32_t num_units_per_row = shard_width * input_unit_size;  // Units per row

    // Note: Since bfloat16 is 2 bytes and uint32_t is 4 bytes, each page contains 2 bfloat16 values.
    // The division by data_size ensures correct mapping of values to pages.
    uint32_t padded_offset_bytes = align(input_unit_size, device->allocator()->get_alignment(BufferType::DRAM));

    // configure and create interleaved DRAM buffer to insert source data into
    uint32_t src_buffer_size = input_unit_size * num_values / data_size;
    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = src_buffer_size,
        .page_size = input_unit_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
    uint32_t src_addr = src_buffer->address();

    // configure and create circular buffers with the same address on each of the designated cores
    uint32_t input_cb_index = CBIndex::c_0;
    CircularBufferConfig input_cb_config =
        CircularBufferConfig(shard_size * input_unit_size, {{input_cb_index, cb_data_format}})
            .set_page_size(input_cb_index, input_unit_size);
    tt_metal::CreateCircularBuffer(program, cores, input_cb_config);

    // create data movement kernel to shard data
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)input_cb_index};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "shard_data_rm/kernels/reader_sharded_rm.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    // set runtime arguments for each core
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {0, i};
        tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src_addr, input_unit_size, shard_height, shard_width_bytes, padded_offset_bytes, curr_idx_h, i});
        curr_idx_w += input_unit_size;
        if (curr_idx_w >= num_units_per_row) {
            curr_idx_w = 0;
            curr_idx_h += shard_height;
        }
    }

    printf("Original tensor values: ");
    for (uint32_t src_vec_idx = 0; src_vec_idx < src_vec.size(); src_vec_idx++) {
        printf("%0.f ", src_vec[src_vec_idx].to_float());
    }
    printf("\n");

    // start/finish program and close device
    EnqueueWriteBuffer(cq, src_buffer, src_vec.data(), false);
    EnqueueProgram(cq, program, false);
    Finish(cq);
    CloseDevice(device);
}
