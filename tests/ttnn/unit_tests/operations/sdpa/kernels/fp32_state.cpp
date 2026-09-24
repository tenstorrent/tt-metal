// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/streaming/fp32_state.hpp"

// A primitive harness, not a second attention implementation. Use the stock
// unary reader/writer; exercise publication, wraparound and pack-format changes.
void kernel_main() {
    constexpr uint32_t normalize = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(1);
    constexpr bool identity = get_compile_time_arg_val(2);
    constexpr bool first_column = get_compile_time_arg_val(3);
    constexpr uint32_t jobs = get_compile_time_arg_val(4);
    constexpr uint32_t record_tiles = normalize ? width + 1 : 2 * width + 1;
    compute_kernel_hw_startup(0, 16);
    sdpa::streaming::Fp32PackConfig pack;

    auto copy = [&](uint32_t input_index, uint32_t output_cb, uint32_t output_index) {
        reconfig_data_format_skip_int8(0, 0);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::NONE>(0);
        unary_bcast<BroadcastType::NONE>(0, input_index, 0);
        unary_bcast_uninit<BroadcastType::NONE>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack.single_tile(output_cb);
        pack_tile<true>(0, output_cb, output_index);
        tile_regs_release();
    };

    for (uint32_t job = 0; job < jobs; ++job) {
        CircularBuffer(0).wait_front(record_tiles);
        PACK((llk_pack_reconfig_l1_acc(0)));
        if constexpr (normalize) {
            CircularBuffer(1).reserve_back(1);
            copy(width, 1, 0);
            CircularBuffer(1).push_back(1);
            CircularBuffer(2).reserve_back(width);
            for (uint32_t p = 0; p < width; ++p) {
                copy(p, 2, p);
            }
            CircularBuffer(2).push_back(width);
            sdpa::streaming::normalize_rows<width>(1, 2, 3, 16, 1, pack);
        } else {
            CircularBuffer(16).reserve_back(width);
            for (uint32_t p = 0; p < width; ++p) {
                copy(width + 1 + p, 16, p);
            }
            PACK((llk_pack_reconfig_l1_acc(1)));
            sdpa::streaming::rescale_and_accumulate<width, first_column>(
                0, 16, identity ? 32 : 0, 0, 0, width, identity);
            PACK((llk_pack_reconfig_l1_acc(0)));
            CircularBuffer(16).push_back(width);
        }
        CircularBuffer(0).pop_front(record_tiles);
    }
}
