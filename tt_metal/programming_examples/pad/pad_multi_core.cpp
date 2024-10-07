// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // get program/device
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    auto program = CreateScopedProgram();

    // initialize source data
    constexpr uint32_t src_M = 8;
    constexpr uint32_t src_N = 4;
    constexpr uint32_t packed_data_size = sizeof(uint32_t);
    constexpr uint32_t unpacked_data_size = sizeof(bfloat16);
    constexpr uint32_t packing_ratio = packed_data_size / unpacked_data_size;
    uint32_t src_num_values_unpacked = src_M * src_N;
    uint32_t src_num_values_packed = src_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> src_vec(src_num_values_packed, 0);
    // source vector = {1, 2, 3, ... , 30, 31, 32}
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        bfloat16 bfloat_val1 = bfloat16(2 * i + 1);
        bfloat16 bfloat_val2 = bfloat16(2 * i + 2);
        src_vec[i] = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat_val1, bfloat_val2));
    }

    // create pad vector
    bfloat16 pad_value = bfloat16(2);
    std::vector<uint32_t> pad_vec(1, pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(pad_value, pad_value)));

    // create destination vector
    constexpr uint32_t dst_M = 8;
    constexpr uint32_t dst_N = 8;
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
    tt_metal::InterleavedBufferConfig input_dram_config {
        .device = device,
        .size = src_buffer_size,
        .page_size = packed_data_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> src_buffer = CreateBuffer(input_dram_config);
    uint32_t src_addr = src_buffer->address();

    uint32_t pad_buffer_size = packed_data_size * pad_vec.size();
    tt_metal::InterleavedBufferConfig pad_dram_config {
        .device = device,
        .size = pad_buffer_size,
        .page_size = packed_data_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> pad_buffer = CreateBuffer(pad_dram_config);
    uint32_t pad_addr = pad_buffer->address();

    uint32_t dst_buffer_size = packed_data_size * dst_num_values_packed;
    tt_metal::InterleavedBufferConfig output_dram_config {
        .device = device,
        .size = dst_buffer_size,
        .page_size = packed_data_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> dst_buffer = CreateBuffer(output_dram_config);
    uint32_t dst_addr = dst_buffer->address();

    // configure and create circular buffer
    uint32_t cb_id = CB::c_in0;
    tt::DataFormat cb_data_format = tt::DataFormat::UInt32;
    CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(dst_N * packed_data_size * 2, {{cb_id, cb_data_format}})
		.set_page_size(cb_id, packed_data_size);
    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);

    // specify compile time args
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool pad_is_dram = pad_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t) src_is_dram,
                                            (uint32_t) pad_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t) dst_is_dram};

    // create kernels
    KernelHandle reader_id = CreateKernel(program,
                                          "tt_metal/programming_examples/pad/kernels/pad_reader_dims_rm_interleaved.cpp",
                                          cores,
                                          tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
    KernelHandle writer_id = CreateKernel(program,
                                          "tt_metal/programming_examples/pad/kernels/pad_writer_dims_rm_interleaved.cpp",
                                          cores,
                                          tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});

    // set kernel runtime arguments
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

    printf("Padding tensor of shape (%d, %d) to shape (%d, %d) with pad value: %d\n", src_M, src_N, dst_M, dst_N, pad_value.to_uint16());
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
