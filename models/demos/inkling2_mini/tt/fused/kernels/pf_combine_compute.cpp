// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE combine, compute. Per unit: for each k, tilize the 32 x 32 CW block of expert rows and multiply it
// by the token weights (column broadcast), packed to bf16 (as ttnn.multiply(yb, wk) rounds); then per output
// tile sum the K products in fp32 dest in k order (as ttnn.sum(dim=1), add with the zero tile, accumulating) and
// pack bf16.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t K = get_named_compile_time_arg_val("k");
    constexpr uint32_t CW = get_named_compile_time_arg_val("cw");
    constexpr uint32_t BW = 4;  // fp32 dest tiles per acquire
    constexpr uint32_t cb_rm = 0, cb_w = 1, cb_z = 2, cb_t = 3, cb_p = 4, cb_out = 16;
    const uint32_t n_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_rm, cb_w, cb_out);
    cb_wait_front(cb_z, 1);
    for (uint32_t u = 0; u < n_units; ++u) {
        cb_wait_front(cb_w, K);
        for (uint32_t k = 0; k < K; ++k) {
            cb_wait_front(cb_rm, CW);
            cb_reserve_back(cb_t, CW);
            reconfig_data_format_srca(cb_rm);
            pack_reconfig_data_format(cb_t);
            tilize_init(cb_rm, CW, cb_t);
            tilize_block(cb_rm, CW, cb_t);
            tilize_uninit(cb_rm, cb_t);
            cb_push_back(cb_t, CW);
            cb_pop_front(cb_rm, CW);

            cb_wait_front(cb_t, CW);
            cb_reserve_back(cb_p, CW);
            reconfig_data_format(cb_t, cb_w);
            pack_reconfig_data_format(cb_p);
            mul_bcast_cols_init(cb_t, cb_w);
            for (uint32_t b = 0; b < CW; b += BW) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < BW && b + w < CW; ++w) {
                    mul_tiles_bcast_cols(cb_t, cb_w, b + w, k, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t w = 0; w < BW && b + w < CW; ++w) {
                    pack_tile(w, cb_p);
                }
                tile_regs_release();
            }
            cb_push_back(cb_p, CW);
            cb_pop_front(cb_t, CW);
        }
        cb_pop_front(cb_w, K);

        cb_wait_front(cb_p, K * CW);
        cb_reserve_back(cb_out, CW);
        reconfig_data_format(cb_p, cb_z);
        pack_reconfig_data_format(cb_out);
        add_init(cb_p, cb_z, true);
        for (uint32_t b = 0; b < CW; b += BW) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < BW && b + w < CW; ++w) {
                for (uint32_t k = 0; k < K; ++k) {
                    add_tiles(cb_p, cb_z, k * CW + b + w, 0, w);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t w = 0; w < BW && b + w < CW; ++w) {
                pack_tile(w, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, CW);
        cb_pop_front(cb_p, K * CW);
    }
}
