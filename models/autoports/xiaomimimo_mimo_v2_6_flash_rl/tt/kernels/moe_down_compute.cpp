// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISC: per group of ct output columns, sum over all pairs of [4, 32] h tiles @ [kt, ct] w_down (LoFi, fp32 dest)
// -> ct [4, 32] bf16 output tiles. Pair (u, k)'s h is zero outside row u, so row u of the sum is user u's output.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t ct = get_compile_time_arg_val(5);
    constexpr uint32_t groups = get_compile_time_arg_val(6);
    constexpr uint32_t block_tiles = kt * ct;

    custom_mm_block_init<false, true, false>(cb_h, cb_w, cb_out, ct);
    cb_wait_front(cb_h, num_pairs * kt);

    for (uint32_t g = 0; g < groups; g++) {
        tile_regs_acquire();
        for (uint32_t p = 0; p < num_pairs - 1; p++) {
            cb_wait_front(cb_w, block_tiles);
            custom_mm_block<false>(cb_h, cb_w, p * kt, 0, 0, kt, ct);
            cb_pop_front(cb_w, block_tiles);
        }
        cb_wait_front(cb_w, block_tiles);
        custom_mm_block<true>(cb_h, cb_w, (num_pairs - 1) * kt, 0, 0, kt, ct);
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

    cb_pop_front(cb_h, num_pairs * kt);
}
