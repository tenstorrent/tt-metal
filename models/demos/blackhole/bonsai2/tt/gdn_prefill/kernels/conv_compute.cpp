// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill causal conv, compute. Token r of a 32-token tile needs x[r - 3 + k] * w_k for k = 0..3;
// a per-channel (column) scale commutes with a row shift, so
//   out = silu( sum_k M_(3-k) @ (cur * w_k) + M'_(3-k) @ (prev * w_k) )
// with M_0 = I, M_s = S_s (rows shifted down by s inside the tile), M'_s = P_s (the previous tile's last s rows) or,
// for the first tile, Q_s (state rows 0-2 = the 3 tokens before the prompt). Scaled tiles live in
// two 4-tile buffers taking turns, so each time tile's scaled copies serve as the next one's prev. New state = R @ last
// raw tile.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "../../ternary/kernels/core_span.hpp"

namespace {
constexpr uint32_t cb_x = 0, cb_s = 1, cb_w = 2, cb_c = 3, cb_a = 4, cb_b = 5, cb_out = 16, cb_ns = 17;
constexpr uint32_t I0 = 0, S1 = 1, P1 = 4, Q1 = 7, R = 10;
ALWI void put(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    cb_reserve_back(cb, n);
    for (uint32_t d = 0; d < n; ++d) {
        pack_tile<true>(d, cb, d);
    }
    cb_push_back(cb, n);
    tile_regs_release();
}
// 4 tiles: src[idx] * w_k (row-broadcast), k = 0..3, into dst
ALWI void scale4(uint32_t src, uint32_t idx, uint32_t dst) {
    tile_regs_acquire();
    reconfig_data_format(src, cb_w);
    mul_bcast_rows_init(src, cb_w);
    for (uint32_t k = 0; k < 4; ++k) {
        mul_tiles_bcast_rows(src, cb_w, idx, k, k);
    }
    put(dst, 4);
}
}  // namespace

void kernel_main() {
    // common args: Dt, ncores, grid_x, Tt
    const uint32_t Dt = get_common_arg_val<uint32_t>(0);
    uint32_t first, n;
    core_span(Dt, get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), 1, first, n);
    const uint32_t Tt = get_common_arg_val<uint32_t>(3);
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    cb_wait_front(cb_c, 11);
    for (uint32_t col = 0; col < n; ++col) {
        cb_wait_front(cb_s, 1);
        cb_wait_front(cb_w, 4);
        // Scaled tiles alternate between two 4-page buffers (prev / cur): reading a ring past its end would wrap.
        scale4(cb_s, 0, cb_a);  // prev = the state tile, scaled
        for (uint32_t tt = 0; tt < Tt; ++tt) {
            const uint32_t prev = (tt & 1) ? cb_b : cb_a, cur = (tt & 1) ? cb_a : cb_b;
            cb_wait_front(cb_x, 1);
            scale4(cb_x, 0, cur);
            cb_wait_front(cur, 4);
            cb_wait_front(prev, 4);
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_c, cur);
            matmul_init(cb_c, cur);
            matmul_tiles(cb_c, cur, I0, 3, 0);  // k = 3: shift 0
            for (uint32_t k = 0; k < 3; ++k) {
                matmul_tiles(cb_c, cur, S1 + (3 - k) - 1, k, 0);
            }
            matmul_init(cb_c, prev);
            for (uint32_t k = 0; k < 3; ++k) {
                matmul_tiles(cb_c, prev, (tt == 0 ? Q1 : P1) + (3 - k) - 1, k, 0);
            }
            silu_tile_init();
            silu_tile(0);
            put(cb_out, 1);
            cb_pop_front(prev, 4);
            if (tt == Tt - 1) {
                tile_regs_acquire();
                reconfig_data_format<SrcOrder::Reverse>(cb_c, cb_x);
                matmul_init(cb_c, cb_x);
                matmul_tiles(cb_c, cb_x, R, 0, 0);
                put(cb_ns, 1);
            }
            cb_pop_front(cb_x, 1);
        }
        cb_pop_front((Tt & 1) ? cb_b : cb_a, 4);  // the last cur
        cb_pop_front(cb_s, 1);
        cb_pop_front(cb_w, 4);
    }
}
