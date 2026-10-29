// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Causal conv1d + SiLU over one prefill chunk, compute. A row shift by d is a matmul with a 0/1 32x32
// matrix and commutes with the per-channel tap, so per channel column and row tile r
//   W_d(r) = x_r * tap[3 - d]                                  (row broadcast, fp32)
//   y_r    = silu(I @ W_0(r) + sum_{d=1..3} L_d @ W_d(r) + U_d @ W_d(r - 1))
// with W(-1) from the carry moved into rows 29..31 (P = D29 @ carry). For every r the compute also
// emits the carry candidate A @ x_r + B @ x_{r-1} (x_{-1} = P): the 3 rows ending at the chunk's last
// valid row when that row lies in tile r (A, B built by the writer for that row); the writer keeps the
// one of the last valid tile. x_{r-1} is kept as a copy in its own CB: an unpack at a tile index past
// a CB's read pointer does not wrap around the CB's end.
// Compile-time args: 0 Rt, 1 columns of this core.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"

namespace {

constexpr uint32_t cb_x = 0, cb_taps = 1, cb_carry = 2, cb_consts = 3, cb_wa = 4, cb_wb = 5, cb_out = 6, cb_cand = 7,
                   cb_praw = 8, cb_xprev = 9;
// consts: 0 I, 1..3 L_d, 4..6 U_d, 7 D29, 8 A, 9 B
constexpr uint32_t kI = 0, kL = 0, kU = 3, kD29 = 7, kA = 8, kB = 9, kConsts = 10;

inline void mm_setup(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format<SrcOrder::Reverse>(in0, in1);
    matmul_init(in0, in1);
    pack_reconfig_data_format(out);
}

// cb_w[d] = src[i] * tap[3 - d], d = 0..3 (row broadcast of the tap row)
inline void weighted(uint32_t src, uint32_t i, uint32_t cb_w) {
    cb_reserve_back(cb_w, 4);
    reconfig_data_format(src, cb_taps);
    pack_reconfig_data_format(cb_w);
    mul_bcast_rows_init(src, cb_taps);
    for (uint32_t d = 0; d < 4; d++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(src, cb_taps, i, 3 - d, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_w);
        tile_regs_release();
    }
    cb_push_back(cb_w, 4);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_consts, cb_carry, cb_praw);
    cb_wait_front(cb_consts, kConsts);
    for (uint32_t c = 0; c < cols; c++) {
        cb_wait_front(cb_taps, 4);
        cb_wait_front(cb_carry, 1);
        // P = D29 @ carry (the carry rows moved to 29..31), W(-1) = P * taps
        cb_reserve_back(cb_praw, 1);
        mm_setup(cb_consts, cb_carry, cb_praw);
        tile_regs_acquire();
        matmul_tiles(cb_consts, cb_carry, kD29, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_praw);
        tile_regs_release();
        cb_push_back(cb_praw, 1);
        cb_pop_front(cb_carry, 1);
        cb_wait_front(cb_praw, 1);
        weighted(cb_praw, 0, cb_wb);

        for (uint32_t r = 0; r < Rt; r++) {
            const uint32_t cur = (r % 2 == 0) ? cb_wa : cb_wb;
            const uint32_t prv = (r % 2 == 0) ? cb_wb : cb_wa;
            cb_wait_front(cb_x, 1);
            weighted(cb_x, 0, cur);
            cb_wait_front(cur, 4);
            cb_wait_front(prv, 4);

            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            mm_setup(cb_consts, cur, cb_out);
            matmul_tiles(cb_consts, cur, kI, 0, 0);
            for (uint32_t d = 1; d <= 3; d++) {
                matmul_tiles(cb_consts, cur, kL + d, d, 0);
            }
            mm_setup(cb_consts, prv, cb_out);
            for (uint32_t d = 1; d <= 3; d++) {
                matmul_tiles(cb_consts, prv, kU + d, d, 0);
            }
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);

            // carry candidate
            const uint32_t praw = r == 0 ? cb_praw : cb_xprev;
            if (r > 0) {
                cb_wait_front(cb_xprev, 1);
            }
            cb_reserve_back(cb_cand, 1);
            tile_regs_acquire();
            mm_setup(cb_consts, cb_x, cb_cand);
            matmul_tiles(cb_consts, cb_x, kA, 0, 0);
            mm_setup(cb_consts, praw, cb_cand);
            matmul_tiles(cb_consts, praw, kB, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_cand);
            tile_regs_release();
            cb_push_back(cb_cand, 1);
            cb_pop_front(praw, 1);

            // x_r -> cb_xprev for the next row
            cb_reserve_back(cb_xprev, 1);
            reconfig_data_format_srca(cb_x);
            pack_reconfig_data_format(cb_xprev);
            copy_tile_to_dst_init_short(cb_x);
            tile_regs_acquire();
            copy_tile(cb_x, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xprev);
            tile_regs_release();
            cb_push_back(cb_xprev, 1);
            cb_pop_front(cb_x, 1);
            cb_pop_front(prv, 4);
        }
        cb_wait_front(cb_xprev, 1);
        cb_pop_front(cb_xprev, 1);
        cb_pop_front((Rt % 2 == 1) ? cb_wa : cb_wb, 4);
        cb_pop_front(cb_taps, 4);
    }
}
