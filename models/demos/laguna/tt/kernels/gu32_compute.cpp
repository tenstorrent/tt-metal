// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode gate/up compute over column-page weights (Laguna): per unit, DST 0 = x @ Wg[:, n] and
// DST 1 = x @ Wu[:, n] (Kt tile matmuls each), packed to cb_g / cb_u; then DST 1 = up * w (per-token weights
// column-broadcast), DST 0 = gate, DST 0 = silu(DST 0) * DST 1, packed to cb_out.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_wv = 2, cb_meta = 3, cb_g = 5, cb_u = 6, cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t units = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    if (units == 0) {
        return;
    }
    cb_wait_front(cb_x, Kt);
    for (uint32_t u = 0; u < units; ++u) {
        reconfig_data_format(cb_w, cb_x);
        matmul_init(cb_x, cb_w);
        tile_regs_acquire();
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t k0 = 0; k0 < Kt; k0 += chunk) {
                cb_wait_front(cb_w, chunk);
                for (uint32_t i = 0; i < chunk; ++i) {
                    matmul_tiles(cb_x, cb_w, k0 + i, i, half);
                }
                cb_pop_front(cb_w, chunk);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_g, 1);
        cb_reserve_back(cb_u, 1);
        pack_tile(0, cb_g);
        pack_tile(1, cb_u);
        cb_push_back(cb_g, 1);
        cb_push_back(cb_u, 1);
        tile_regs_release();

        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_u, 1);
        cb_wait_front(cb_wv, 1);
        reconfig_data_format(cb_u, cb_wv);
        tile_regs_acquire();
        mul_bcast_cols_init_short(cb_u, cb_wv);
        mul_tiles_bcast_cols(cb_u, cb_wv, 0, 0, 1);
        copy_tile_init(cb_g);
        copy_tile(cb_g, 0, 0);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_u, 1);
        cb_pop_front(cb_wv, 1);
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
    cb_pop_front(cb_x, Kt);
}
