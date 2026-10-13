// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RoPE on the Q and K rows of each work unit, adapted from ops/rotary_embedding's compute kernel.
// out[j] = x[j] * cos[j] + rotate_half(x)[j] * sin[j], with the first-half sin tiles negated once up front
// so every output tile is two ELWMULs accumulated in DST. Unit u's rows use cos/sin row (u / num_heads) %
// seq_tiles.
//
// Compile-time args: head_tiles, seq_tiles, num_heads, dst_block
// Runtime args: unit_start, num_units

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t num_heads = get_compile_time_arg_val(2);
    constexpr uint32_t dst_block = get_compile_time_arg_val(3);
    constexpr uint32_t half_Wt = Wt / 2;
    constexpr uint32_t HtWt = Ht * Wt;
    constexpr uint32_t num_neg_sin_tiles = Ht * half_Wt;

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t cos_cb = tt::CBIndex::c_2;
    constexpr uint32_t sin_cb = tt::CBIndex::c_3;
    constexpr uint32_t scalar_cb = tt::CBIndex::c_4;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t neg_sin_cb = tt::CBIndex::c_24;

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

    for (uint32_t unit = unit_start; unit < unit_start + num_units; ++unit) {
        const uint32_t ht = (unit / num_heads) % Ht;
        const uint32_t cs_row = ht * Wt;
        const uint32_t neg_sin_row = ht * half_Wt;
        // Q row, then K row.
        for (uint32_t r = 0; r < 2; ++r) {
            cb_in.wait_front(Wt);
            cb_out.reserve_back(Wt);
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
        }
    }
}
