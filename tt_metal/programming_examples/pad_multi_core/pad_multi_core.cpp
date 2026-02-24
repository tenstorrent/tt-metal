// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstdint>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    DeviceContext ctx(0);

    // Initialize source data — a (64, 32) matrix of bfloat16 packed into uint32 pairs.
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
        bfloat16 bfloat_val1 = bfloat16((2 * i) + 1);
        bfloat16 bfloat_val2 = bfloat16((2 * i) + 2);
        src_vec[i] = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat_val1, bfloat_val2));
    }

    // Create pad vector — pad with bfloat16(2).
    bfloat16 pad_value = bfloat16(2);
    std::vector<uint32_t> pad_vec(
        1, pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(pad_value, pad_value)));

    // Destination tensor shape — padded from (64, 32) to (64, 64).
    constexpr uint32_t dst_M = 64;
    constexpr uint32_t dst_N = 64;
    uint32_t dst_num_values_unpacked = dst_M * dst_N;
    uint32_t dst_num_values_packed = dst_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> dst_vec(dst_num_values_packed, 0);

    // Designate cores and core specs — use 4 cores for multi-core padding.
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {0, 3};
    CoreRange cores(start_core, end_core);
    uint32_t num_cores = cores.size();

    // Configure DRAM buffers for input, pad, and output.
    uint32_t src_buffer_size = packed_data_size * src_num_values_packed;
    auto src_buffer = ctx.dram_buffer(src_buffer_size, packed_data_size);
    uint32_t src_addr = src_buffer->address();

    uint32_t pad_buffer_size = packed_data_size * pad_vec.size();
    auto pad_buffer = ctx.dram_buffer(pad_buffer_size, packed_data_size);

    uint32_t dst_buffer_size = packed_data_size * dst_num_values_packed;
    auto dst_buffer = ctx.dram_buffer(dst_buffer_size, packed_data_size);
    uint32_t dst_addr = dst_buffer->address();

    // Circular buffer configuration expected by TTNN reader/writer: c_0 (main), c_1 (pad), c_2 (align).
    constexpr uint32_t cb0 = CBIndex::c_0;
    constexpr uint32_t cb1 = CBIndex::c_1;
    constexpr uint32_t cb2 = CBIndex::c_2;
    tt::DataFormat cb_df = tt::DataFormat::UInt32;
    const uint32_t stick_size_bytes = packed_data_size;     // 4 bytes per stick (one packed uint32)
    // Use TTNN row-major minimum of 64B per stick in L1 for padding pattern.
    const uint32_t stick_size_padded = 64;
    const uint32_t stick_size_padded_aligned = 64;
    const uint32_t num_packed_row_src = src_N / packing_ratio;
    const uint32_t num_packed_row_dst = dst_N / packing_ratio;
    const uint32_t num_sticks_per_barrier = num_packed_row_dst;  // process one row per barrier

    // Specify compile-time args for the TTNN-derived reader kernel.
    // These 20 args come BEFORE the auto-generated TensorAccessorArgs, matching the kernel's
    // expectation of TensorAccessorArgs<20>() at offset 20.
    std::vector<uint32_t> reader_compile_args = {
        src_M,                     // 0: N
        1,                         // 1: H
        num_packed_row_src,        // 2: C
        stick_size_bytes,          // 3: stick_size_bytes
        src_M,                     // 4: N_padded (same)
        1,                         // 5: H_padded
        num_packed_row_dst,        // 6: C_padded
        stick_size_padded,         // 7: stick_size_padded (64B pad pattern)
        0,                         // 8: stick_size_padded_front (left-align)
        0,                         // 9: stick_size_padded_end
        1,                         // 10: num_zero_pad_sticks_read
        stick_size_padded,         // 11: last_zero_stick_size (64)
        1,                         // 12: not_pad_by_zero
        pad_vec[0],                // 13: packed_pad_value (passed as compile-time arg as TTNN does)
        stick_size_padded,         // 14: row_major_min_bytes (64)
        0,                         // 15: num_front_pad_sticks_read
        0,                         // 16: num_end_pad_sticks_read
        1,                         // 17: num_sticks_padded_read per stick
        stick_size_padded_aligned, // 18: stick_size_padded_aligned (64)
        0,                         // 19: unaligned = false
    };

    // Writer compile-time args: CB index and stick sizes before TensorAccessorArgs.
    std::vector<uint32_t> writer_compile_args = {cb0, stick_size_bytes, stick_size_padded_aligned};

    // Build the program with reader and writer kernels on all cores.
    // The kernel uses three CB indices sharing the same L1 allocation, each with a different page size:
    //   c_0: padded+aligned sticks (main data path)
    //   c_1: padded sticks (pad pattern)
    //   c_2: raw sticks (unpadded source data)
    auto builder = ProgramBuilder(cores);
    builder.cb(CircularBufferConfig(
                   num_sticks_per_barrier * stick_size_padded_aligned,
                   {{cb0, cb_df}, {cb1, cb_df}, {cb2, cb_df}})
                   .set_page_size(cb0, stick_size_padded_aligned)
                   .set_page_size(cb1, stick_size_padded)
                   .set_page_size(cb2, stick_size_bytes));

    auto& reader_ref = builder.reader(
        OVERRIDE_KERNEL_PREFIX "pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp",
        {src_buffer},
        reader_compile_args);

    auto& writer_ref = builder.writer(
        OVERRIDE_KERNEL_PREFIX "pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp",
        {dst_buffer},
        writer_compile_args);

    // Set per-core runtime arguments to distribute rows across cores.
    uint32_t start_src_idx = 0;
    uint32_t start_dst_idx = 0;
    uint32_t num_rows_per_core = src_M / num_cores;
    uint32_t num_src_sticks_per_core = num_packed_row_src * num_rows_per_core;
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        CoreCoord core = {0, core_idx};
        uint32_t num_sticks_per_core = num_rows_per_core * num_packed_row_dst;
        uint32_t num_sticks_per_barrier_rt = num_packed_row_dst;  // one row per barrier

        // Reader runtime: src_addr, num_sticks_per_core, num_sticks_per_barrier, start_id,
        //   front_pad_n, front_pad_c, front_pad_h, start_dim_offset[0..3]
        reader_ref.runtime_args_at(core, {src_addr,
                                          num_sticks_per_core,
                                          num_sticks_per_barrier_rt,
                                          start_src_idx,
                                          0, 0, 0,
                                          0, 0, 0, 0});
        // Writer runtime: dst_addr, num_sticks_per_core, num_sticks_per_barrier, start_id
        writer_ref.runtime_args_at(core, {dst_addr, num_sticks_per_core, num_sticks_per_barrier_rt, start_dst_idx});

        start_src_idx += num_src_sticks_per_core;
        start_dst_idx += num_packed_row_dst * num_rows_per_core;
    }

    printf(
        "Padding tensor of shape (%d, %d) to shape (%d, %d) with pad value: %d\n",
        src_M, src_N, dst_M, dst_N, std::bit_cast<uint16_t>(pad_value));
    printf("Original tensor with shape (%d, %d):\n", src_M, src_N);
    for (uint32_t m = 0; m < src_M; m++) {
        for (uint32_t n = 0; n < num_packed_row_src; n++) {
            printf("%d ", (uint16_t)src_vec[(m * num_packed_row_src) + n]);
            printf("%d ", (uint16_t)(src_vec[(m * num_packed_row_src) + n] >> 16));
        }
        printf("\n");
    }
    printf("\n");

    // Upload inputs (non-blocking), execute program, read back result.
    ctx.write(src_buffer, src_vec);
    ctx.write(pad_buffer, pad_vec);
    ctx.run(builder.build());
    dst_vec = ctx.read<uint32_t>(dst_buffer);

    printf("Padded tensor with shape (%d, %d):\n", dst_M, dst_N);
    for (uint32_t m = 0; m < dst_M; m++) {
        for (uint32_t n = 0; n < num_packed_row_dst; n++) {
            printf("%d ", (uint16_t)dst_vec[(m * num_packed_row_dst) + n]);
            printf("%d ", (uint16_t)(dst_vec[(m * num_packed_row_dst) + n] >> 16));
        }
        printf("\n");
    }
}
