// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// OpenVDN linear-branch readout, one unit (frame f, head, 32 q rows) per step:
//
//     x   = q state^T                       (q bf16 exact; state^T as bf16 hi + lo, accumulated in fp32 DST)
//     out = x / sqrt(mean_c x^2 + eps) * w  (RMSNorm over the head's 128 channels, fp32 on the SFPU)
//
// The same products as the composite `split_matmul(q, state^T)` (q bf16, so its single part times both parts of the
// state), summed in DST instead of by a separate add. The row sums of x^2 are a matmul with a ones tile on the hi/lo
// bf16 split of the per-row partial sums, so every column of the result holds its row's sum.
//
// Units u = (f h + head) RT + rt; the reader sends each (frame, head) group's state tiles before its first unit.
// Runtime args: number of units, first unit, RT.

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
constexpr uint32_t STATE_TILES = 16;
// bf16
constexpr uint32_t cb_q = 0;
constexpr uint32_t cb_hi = 1;
constexpr uint32_t cb_lo = 2;
constexpr uint32_t cb_ones = 4;
constexpr uint32_t cb_ss_hi = 8;
constexpr uint32_t cb_ss_lo = 9;
// fp32, unpacked to DST
constexpr uint32_t cb_w = 3;
constexpr uint32_t cb_x = 6;
constexpr uint32_t cb_ss = 7;
constexpr uint32_t cb_inv = 10;
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
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t rt_count = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_hi, cb_x);
    cb_wait_front(cb_w, HEAD_TILES);
    cb_wait_front(cb_ones, 1);

    for (uint32_t u = first; u < first + count; ++u) {
        if (u == first || u % rt_count == 0) {
            cb_wait_front(cb_hi, STATE_TILES);
            cb_wait_front(cb_lo, STATE_TILES);
        }
        cb_wait_front(cb_q, HEAD_TILES);
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_q, cb_hi);
            matmul_init(cb_q, cb_hi);
            for (uint32_t k = 0; k < HEAD_TILES; ++k) {
                matmul_tiles(cb_q, cb_hi, k, k * HEAD_TILES + c, 0);
            }
            for (uint32_t k = 0; k < HEAD_TILES; ++k) {
                matmul_tiles(cb_q, cb_lo, k, k * HEAD_TILES + c, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_x);
            tile_regs_release();
        }
        cb_pop_front(cb_q, HEAD_TILES);
        if (u + 1 == first + count || (u + 1) % rt_count == 0) {
            cb_pop_front(cb_hi, STATE_TILES);
            cb_pop_front(cb_lo, STATE_TILES);
        }

        // Row partial sums over the 4 column tiles of x^2, split.
        cb_wait_front(cb_x, HEAD_TILES);
        tile_regs_acquire();
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            const uint32_t d = c == 0 ? 0 : 1;
            copy_to(cb_x, c, d);
            mul_binary_tile_init();
            mul_binary_tile(d, d, d);
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

        // 1 / sqrt(row sum / 128 + eps), the same value in every column.
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_ss_hi, cb_ones);
        matmul_init(cb_ss_hi, cb_ones);
        matmul_tiles(cb_ss_hi, cb_ones, 0, 0, 0);
        matmul_tiles(cb_ss_lo, cb_ones, 0, 0, 0);
        fill_tile_init();
        fill_tile(1, 1.0f / 128.0f);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        fill_tile_init();
        fill_tile(1, 1e-6f);
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
            copy_to(cb_w, c, 1);
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
