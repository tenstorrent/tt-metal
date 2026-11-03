// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISC: per group of ct output columns, sum over the distinct experts (their count from cb_cnt) of [rows, 32] h
// tiles @ [kt, ct] w_down (LoFi, fp32 dest) -> ct [rows, 32] bf16 output tiles. Expert j's h holds in row u user u's
// routing-weighted activation (zero when u did not select it), so row u of the sum is user u's output.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "moe_stream_common.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t ct = get_compile_time_arg_val(5);
    constexpr uint32_t groups = get_compile_time_arg_val(6);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(7);
    constexpr uint32_t block_tiles = kt * ct;

    MOE_MARK(60);
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<false, dest_layout>(cb_h, cb_w, cb_out, ct);
    cb_wait_front(cb_cnt, 1);
    const uint32_t n = read_tile_value(cb_cnt, 0, 0);
    cb_pop_front(cb_cnt, 1);
    cb_wait_front(cb_h, n * kt);
    MOE_MARK(61);

    for (uint32_t g = 0; g < groups; g++) {
        tile_regs_acquire();
        for (uint32_t p = 0; p < n - 1; p++) {
            cb_wait_front(cb_w, block_tiles);
            custom_mm_block<dest_layout, false>(cb_h, cb_w, p * kt, 0, 0, kt, ct);
            cb_pop_front(cb_w, block_tiles);
        }
        cb_wait_front(cb_w, block_tiles);
        custom_mm_block<dest_layout>(cb_h, cb_w, (n - 1) * kt, 0, 0, kt, ct);
        cb_pop_front(cb_w, block_tiles);
        tile_regs_commit();

        cb_reserve_back(cb_out, ct);
        tile_regs_wait();
        for (uint32_t c = 0; c < ct; c++) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, ct);
    }

    cb_pop_front(cb_h, n * kt);
    MOE_MARK(62);
}
