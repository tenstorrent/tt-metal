// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode attention epilogue (Laguna), compute: gsp = softplus(g) (column 0 = one value per head row), then
// out_j = attn_j * gsp (column broadcast) for the 4 head_dim tiles.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/softplus.h"

void kernel_main() {
    constexpr uint32_t BETA = get_compile_time_arg_val(0), BETA_RECIP = get_compile_time_arg_val(1);
    constexpr uint32_t THRESHOLD = get_compile_time_arg_val(2);
    constexpr uint32_t TJ = get_compile_time_arg_val(3);  // head_dim tiles on this core
    constexpr uint32_t cb_attn = 0, cb_g = 1, cb_gsp = 3, cb_out = 16;
    compute_kernel_hw_startup(cb_g, cb_gsp);
    cb_wait_front(cb_g, 1);
    copy_tile_init(cb_g);
    tile_regs_acquire();
    copy_tile(cb_g, 0, 0);
    softplus_tile_init();
    softplus_tile(0, BETA, BETA_RECIP, THRESHOLD);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_gsp, 1);
    pack_tile(0, cb_gsp);
    cb_push_back(cb_gsp, 1);
    tile_regs_release();
    cb_pop_front(cb_g, 1);

    cb_wait_front(cb_attn, TJ);
    cb_wait_front(cb_gsp, 1);
    mul_bcast_cols_init_short(cb_attn, cb_gsp);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TJ; ++j) {
        mul_tiles_bcast_cols(cb_attn, cb_gsp, j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, TJ);
    for (uint32_t j = 0; j < TJ; ++j) {
        pack_tile(j, cb_out);
    }
    cb_push_back(cb_out, TJ);
    tile_regs_release();
    cb_pop_front(cb_attn, TJ);
    cb_pop_front(cb_gsp, 1);
}
