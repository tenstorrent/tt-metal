// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

// out[j] = x[j] * cos[j] + rotate_half(x)[j] * sin[j], where rotate_half(x)[j] = -x[j + half] for the first
// half and x[j - half] for the second. The first-half sin tiles are negated once up front so every output
// tile is two ELWMULs accumulated in DST (ELWMUL adds into DST on WH/BH, and DST is zeroed after each pack).
void kernel_main() {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(3);
    constexpr uint32_t neg_sin_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t dst_block = get_compile_time_arg_val(9);
    constexpr uint32_t HtWt = Ht * Wt;
    constexpr uint32_t num_neg_sin_tiles = Ht * half_Wt;

    uint32_t ht = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_cos(cos_cb);
    CircularBuffer cb_sin(sin_cb);
    CircularBuffer cb_scalar(scalar_cb);
    CircularBuffer cb_neg_sin(neg_sin_cb);
    CircularBuffer cb_out(out_cb);

    compute_kernel_hw_startup(sin_cb, scalar_cb, neg_sin_cb);
    cb_sin.wait_front(HtWt);
    cb_cos.wait_front(HtWt);
    cb_scalar.wait_front(1);

    mul_bcast_scalar_init(sin_cb, scalar_cb);
    cb_neg_sin.reserve_back(num_neg_sin_tiles);
    for (uint32_t h = 0; h < Ht; ++h) {
        for (uint32_t j = 0; j < half_Wt; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(sin_cb, scalar_cb, h * Wt + j, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, neg_sin_cb);
            tile_regs_release();
        }
    }
    cb_neg_sin.push_back(num_neg_sin_tiles);
    cb_neg_sin.wait_front(num_neg_sin_tiles);

    reconfig_data_format(in_cb, cos_cb);
    pack_reconfig_data_format(out_cb);
    mul_init(in_cb, cos_cb);

    for (uint32_t i = 0; i < num_rows; ++i) {
        cb_in.wait_front(Wt);
        cb_out.reserve_back(Wt);
        const uint32_t cs_row = ht * Wt;
        const uint32_t neg_sin_row = ht * half_Wt;
        for (uint32_t j0 = 0; j0 < Wt; j0 += dst_block) {
            tile_regs_acquire();
            for (uint32_t d = 0; d < dst_block; ++d) {
                mul_tiles(in_cb, cos_cb, j0 + d, cs_row + j0 + d, d);
            }
            for (uint32_t d = 0; d < dst_block; ++d) {
                const uint32_t j = j0 + d;
                if (j < half_Wt) {
                    mul_tiles(in_cb, neg_sin_cb, j + half_Wt, neg_sin_row + j, d);
                } else {
                    mul_tiles(in_cb, sin_cb, j - half_Wt, cs_row + j, d);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < dst_block; ++d) {
                pack_tile(d, out_cb);
            }
            tile_regs_release();
        }
        cb_in.pop_front(Wt);
        cb_out.push_back(Wt);
        ht = (ht + 1 == Ht) ? 0 : ht + 1;
    }
}
