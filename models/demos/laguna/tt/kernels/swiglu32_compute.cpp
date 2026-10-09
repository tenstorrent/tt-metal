// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode routed SwiGLU compute (Laguna): per unit DST 1 = up * w (column-broadcast routing weights),
// DST 0 = gate, then DST 0 = silu(DST 0) * DST 1, packed as one bf16 tile.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_g = 0, cb_u = 1, cb_w = 2, cb_meta = 3, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_u, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_u, 1);
        cb_wait_front(cb_w, 1);
        tile_regs_acquire();
#ifndef SW_STAGE
#define SW_STAGE 9
#endif
#if SW_STAGE >= 2
        mul_bcast_cols_init_short(cb_u, cb_w);
        mul_tiles_bcast_cols(cb_u, cb_w, 0, 0, 1);
#endif
        copy_tile_init(cb_g);
        copy_tile(cb_g, 0, 0);
#if SW_STAGE >= 3
        silu_tile_init();
        silu_tile(0);
#endif
#if SW_STAGE >= 4
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
#endif
        tile_regs_commit();
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_u, 1);
        cb_pop_front(cb_w, 1);
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
