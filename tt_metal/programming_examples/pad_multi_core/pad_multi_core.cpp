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
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // get program/device
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // initialize source data
    constexpr uint32_t src_M = 64;
    constexpr uint32_t src_N = 32;
    constexpr uint32_t packed_data_size = sizeof(uint32_t);
    constexpr uint32_t unpacked_data_size = sizeof(bfloat16);
    constexpr uint32_t packing_ratio = packed_data_size / unpacked_data_size;
    uint32_t src_num_values_unpacked = src_M * src_N;
    uint32_t src_num_values_packed = src_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> src_vec(src_num_values_packed, 0);
    // source vector = {1, 2, 3, ... , 30, 31, 32,   2048}
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        bfloat16 bfloat_val1 = bfloat16(2 * i + 1);
        bfloat16 bfloat_val2 = bfloat16(2 * i + 2);
        src_vec[i] = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat_val1, bfloat_val2));
    }

    // create pad vector
    bfloat16 pad_value = bfloat16(2);
    std::vector<uint32_t> pad_vec(
        1, pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(pad_value, pad_value)));

    // create destination vector
    constexpr uint32_t dst_M = 64;
    constexpr uint32_t dst_N = 64;
    uint32_t dst_num_values_unpacked = dst_M * dst_N;
    uint32_t dst_num_values_packed = dst_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> dst_vec(dst_num_values_packed, 0);

    // designate cores and core specs
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {0, 3};
    CoreRange cores(start_core, end_core);
    uint32_t num_cores = cores.size();

    // configure and create DRAM buffers for input, pad, output
    uint32_t src_buffer_size = packed_data_size * src_num_values_packed;
    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = src_buffer_size,
        .page_size = packed_data_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
    uint32_t src_addr = src_buffer->address();

    uint32_t pad_buffer_size = packed_data_size * pad_vec.size();
    tt_metal::InterleavedBufferConfig pad_dram_config{
        .device = device,
        .size = pad_buffer_size,
        .page_size = packed_data_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> pad_buffer = CreateBuffer(pad_dram_config);

    uint32_t dst_buffer_size = packed_data_size * dst_num_values_packed;
    tt_metal::InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = dst_buffer_size,
        .page_size = packed_data_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer = CreateBuffer(output_dram_config);
    uint32_t dst_addr = dst_buffer->address();

    // configure circular buffers expected by TTNN reader/writer: c_0 (main), c_1 (pad), c_2 (align)
    constexpr uint32_t cb0 = CBIndex::c_0;
    constexpr uint32_t cb1 = CBIndex::c_1;
    constexpr uint32_t cb2 = CBIndex::c_2;
    tt::DataFormat cb_df = tt::DataFormat::UInt32;
    const uint32_t stick_size_bytes = packed_data_size;              // 4 bytes per stick (one packed uint32)
    // Use TTNN row-major minimum of 64B per stick in L1 for padding pattern
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

    // specify compile time args for TTNN reader/writer
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

    // create kernels (borrowed from TTNN production code)
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

    // set kernel runtime arguments
    uint32_t start_src_idx = 0;
    uint32_t start_dst_idx = 0;
    uint32_t num_rows_per_core = src_M / num_cores;
    uint32_t num_src_sticks_per_core = num_packed_row_src * num_rows_per_core;
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = {0, core_idx};
        uint32_t num_sticks_per_core = num_rows_per_core * num_packed_row_dst;
        uint32_t num_sticks_per_barrier_rt = num_packed_row_dst;  // one row per barrier
        // Reader runtime: src_addr, num_sticks_per_core, num_sticks_per_barrier, start_id, front_pad_n,c,h, start_dim_offset[0..3]
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
        // Writer runtime: dst_addr, num_sticks_per_core, num_sticks_per_barrier, start_id
        std::vector<uint32_t> writer_rt = {dst_addr, num_sticks_per_core, num_sticks_per_barrier_rt, start_dst_idx};
        tt_metal::SetRuntimeArgs(program, writer_id, core, writer_rt);
        start_src_idx += num_src_sticks_per_core;
        start_dst_idx += num_packed_row_dst * num_rows_per_core;
    }

    printf(
        "Padding tensor of shape (%d, %d) to shape (%d, %d) with pad value: %d\n",
        src_M,
        src_N,
        dst_M,
        dst_N,
        std::bit_cast<uint16_t>(pad_value));
    printf("Original tensor with shape (%d, %d):\n", src_M, src_N);
    for (uint32_t m = 0; m < src_M; m++) {
        for (uint32_t n = 0; n < num_packed_row_src; n++) {
            printf("%d ", (uint16_t)src_vec[m * num_packed_row_src + n]);
            printf("%d ", (uint16_t)(src_vec[m * num_packed_row_src + n] >> 16));
        }
        printf("\n");
    }
    printf("\n");

    // dispatch program to device for execution
    EnqueueWriteBuffer(cq, src_buffer, src_vec.data(), false);
    EnqueueWriteBuffer(cq, pad_buffer, pad_vec.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_buffer, dst_vec.data(), false);
    Finish(cq);

    printf("Padded tensor with shape (%d, %d):\n", dst_M, dst_N);
    for (uint32_t m = 0; m < dst_M; m++) {
        for (uint32_t n = 0; n < num_packed_row_dst; n++) {
            printf("%d ", (uint16_t)dst_vec[m * num_packed_row_dst + n]);
            printf("%d ", (uint16_t)(dst_vec[m * num_packed_row_dst + n] >> 16));
        }
        printf("\n");
    }

    CloseDevice(device);
}
