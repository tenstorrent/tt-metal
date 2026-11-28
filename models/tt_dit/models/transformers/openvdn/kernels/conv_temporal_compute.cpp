// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// OpenVDN linear features in the head layout `[F, H, S, 128]`, one unit (frame f, head, 32-row block) per step:
//
//     x = sum_dt w[dt] y[f + dt - 2]     (TEMPORAL; frames outside [0, F) are zero) else x = y[f]
//     x = silu(x)
//     x = x / sqrt(sum_c x^2 + eps^2)    (L2, over the head's 128 channels)
//     out = bf16(x)
//
// All in fp32 on the SFPU, inputs unpacked to DST exactly. The row sums of x^2 are a matmul with a ones tile on the
// hi/lo bf16 split of the per-row partial sums, so every column of the result holds its row's sum.
// `F.normalize` divides by max(||x||, eps); with eps^2 = 1e-12 added under the root instead, rows with ||x|| >> eps
// are unchanged and zero (pad) rows stay zero.
//
// Units u = (f H + head) RT + rt; the reader sends y tiles in the order (unit, column tile, valid dt ascending).
// Compile-time args: temporal, l2, channel tiles (the weights' column tiles).
// Runtime args: number of units, frames, first unit, heads, RT.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t HEAD_TILES = 4;
constexpr uint32_t TAPS = 5;
// fp32, unpacked to DST
constexpr uint32_t cb_y = 0;
constexpr uint32_t cb_w = 1;
constexpr uint32_t cb_x = 6;    // silu output, 4 tiles
constexpr uint32_t cb_ss = 7;   // row partial sums of x^2
constexpr uint32_t cb_inv = 3;  // 1 / norm, every column
// bf16
constexpr uint32_t cb_ones = 2;
constexpr uint32_t cb_ss_hi = 8;
constexpr uint32_t cb_ss_lo = 9;
constexpr uint32_t cb_out = 16;

FORCE_INLINE void copy_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr bool temporal = get_compile_time_arg_val(0) != 0;
    constexpr bool l2 = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t c_tiles = get_compile_time_arg_val(2);
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t frames = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t heads = get_arg_val<uint32_t>(3);
    const uint32_t rt_count = get_arg_val<uint32_t>(4);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_ss_hi, cb_ones, cb_x);
    if constexpr (temporal) {
        cb_wait_front(cb_w, TAPS * c_tiles);
    }
    if constexpr (l2) {
        cb_wait_front(cb_ones, 1);
    }

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t f = u / (heads * rt_count);
        const uint32_t head = (u / rt_count) % heads;
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            tile_regs_acquire();
            if constexpr (temporal) {
                bool started = false;
                for (uint32_t dt = 0; dt < TAPS; ++dt) {
                    const int32_t src = static_cast<int32_t>(f + dt) - 2;
                    if (src < 0 || src >= static_cast<int32_t>(frames)) {
                        continue;
                    }
                    cb_wait_front(cb_y, 1);
                    copy_to(cb_y, 0, 1);
                    copy_to(cb_w, dt * c_tiles + head * HEAD_TILES + c, 2);
                    mul_binary_tile_init();
                    mul_binary_tile(1, 2, started ? 1 : 0);
                    if (started) {
                        add_binary_tile_init();
                        add_binary_tile(0, 1, 0);
                    }
                    started = true;
                    cb_pop_front(cb_y, 1);
                }
            } else {
                cb_wait_front(cb_y, 1);
                copy_to(cb_y, 0, 0);
                cb_pop_front(cb_y, 1);
            }
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (l2) {
                pack_to(0, cb_x);
            } else {
                pack_to(0, cb_out);
            }
            tile_regs_release();
        }
        if constexpr (!l2) {
            continue;
        }

        // Row partial sums over the 4 column tiles of x^2, split.
        cb_wait_front(cb_x, HEAD_TILES);
        tile_regs_acquire();
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            copy_to(cb_x, c, c == 0 ? 0 : 1);
            mul_binary_tile_init();
            mul_binary_tile(c == 0 ? 0 : 1, c == 0 ? 0 : 1, c == 0 ? 0 : 1);
            if (c > 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_ss);
        pack_to(0, cb_ss_hi);
        tile_regs_release();
        cb_wait_front(cb_ss, 1);
        cb_wait_front(cb_ss_hi, 1);
        tile_regs_acquire();
        copy_to(cb_ss, 0, 0);
        copy_to(cb_ss_hi, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_ss_lo);
        tile_regs_release();
        cb_wait_front(cb_ss_lo, 1);

        // 1 / sqrt(row sum + eps^2), the same value in every column.
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_ss_hi, cb_ones);
        matmul_init(cb_ss_hi, cb_ones);
        matmul_tiles(cb_ss_hi, cb_ones, 0, 0, 0);
        matmul_tiles(cb_ss_lo, cb_ones, 0, 0, 0);
        fill_tile_init();
        fill_tile(1, 1e-12f);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        sqrt_tile_init();
        sqrt_tile(0);
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, cb_inv);
        tile_regs_release();
        cb_pop_front(cb_ss, 1);
        cb_pop_front(cb_ss_hi, 1);
        cb_pop_front(cb_ss_lo, 1);

        cb_wait_front(cb_inv, 1);
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            tile_regs_acquire();
            copy_to(cb_x, c, 0);
            copy_to(cb_inv, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_out);
            tile_regs_release();
        }
        cb_pop_front(cb_inv, 1);
        cb_pop_front(cb_x, HEAD_TILES);
    }
}
