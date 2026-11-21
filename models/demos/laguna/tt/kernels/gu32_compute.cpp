// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode gate/up compute over column-page weights (Laguna): per unit and token tile m, gate_m = x_m @ Wg[:, n]
// and up_m = x_m @ Wu[:, n] (Kt tile matmuls each; one weight chunk serves every token tile), packed to cb_g / cb_u;
// then per m DST 1 = up * w (per-token weights column-broadcast), DST 0 = silu(gate) * DST 1, packed to cb_out.
// skip (Mt > 1): per unit a cb_mask word from the reader; only token tiles m with bit m set are computed (their gate /
// up in DST 0.. in order) and pushed. cb_g / cb_u still move Mt pages per unit (the first na filled): a push of fewer
// pages than the CB holds would leave the write pointer mid-CB and the next unit's multi-tile pack would run past
// the CB's end (into cb 7, the writer's routing rows).

#include <cstdint>
#include "api/compute/reg_api.h"
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
    constexpr uint32_t Mt = get_compile_time_arg_val(2);  // 32-row token tiles (gate in DST 0..Mt-1, then up)
    // Mt > 1: one matmul_block per weight tile (rt_dim = Mt, ct_dim = 1) unpacks the weight tile once for every
    // token tile instead of once per token tile (matmul_tiles)
    constexpr bool skip = get_compile_time_arg_val(4) == 1 && Mt > 1;
    constexpr bool block = get_compile_time_arg_val(3) == 1 && Mt > 1 && !skip;
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_wv = 2, cb_meta = 3, cb_g = 5, cb_u = 6, cb_mask = 8, cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t units = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    if (units == 0) {
        return;
    }
    cb_wait_front(cb_x, Mt * Kt);
    for (uint32_t u = 0; u < units; ++u) {
        uint32_t act[Mt], na = 0;  // the computed token tiles, in order
        uint32_t mask = (1u << Mt) - 1;
        if constexpr (skip) {
            cb_wait_front(cb_mask, 1);
            mask = read_tile_value(cb_mask, 0, 0);
            cb_pop_front(cb_mask, 1);
        }
        for (uint32_t m = 0; m < Mt; ++m) {
            if ((mask >> m) & 1) {
                act[na++] = m;
            }
        }
        for (uint32_t half = 0; half < 2; ++half) {
            // one weight chunk feeds every token tile: DST m = x_m @ W[:, n] (gate column, then up column)
            reconfig_data_format(cb_w, cb_x);
            if constexpr (block) {
                matmul_block_init(cb_x, cb_w, false, 1, Mt, Kt);
            } else {
                matmul_init(cb_x, cb_w);
            }
            tile_regs_acquire();
            for (uint32_t k0 = 0; k0 < Kt; k0 += chunk) {
                cb_wait_front(cb_w, chunk);
                for (uint32_t i = 0; i < chunk; ++i) {
                    if constexpr (block) {
                        matmul_block(cb_x, cb_w, k0 + i, i, 0, false, 1, Mt, Kt);
                    } else {
                        for (uint32_t a = 0; a < na; ++a) {
                            matmul_tiles(cb_x, cb_w, act[a] * Kt + k0 + i, i, a);
                        }
                    }
                }
                cb_pop_front(cb_w, chunk);
            }
            tile_regs_commit();
            tile_regs_wait();
            const uint32_t cb = half == 0 ? cb_g : cb_u;
            cb_reserve_back(cb, Mt);
            for (uint32_t a = 0; a < na; ++a) {
                pack_tile(a, cb);
            }
            cb_push_back(cb, Mt);
            tile_regs_release();
        }
        cb_wait_front(cb_g, Mt);
        cb_wait_front(cb_u, Mt);
        cb_wait_front(cb_wv, Mt);
        for (uint32_t a = 0; a < na; ++a) {
            const uint32_t m = act[a];
            reconfig_data_format(cb_u, cb_wv);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_u, cb_wv);
            mul_tiles_bcast_cols(cb_u, cb_wv, a, m, 1);
            copy_init(cb_g);
            copy_tile(cb_g, a, 0);
            silu_tile_init();
            silu_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_g, Mt);
        cb_pop_front(cb_u, Mt);
        cb_pop_front(cb_wv, Mt);
    }
    cb_pop_front(cb_x, Mt * Kt);
}
