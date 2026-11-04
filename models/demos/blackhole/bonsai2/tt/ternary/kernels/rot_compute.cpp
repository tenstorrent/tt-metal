// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode rotation, compute: H_1024 = H_32 (x) H_32 (Sylvester), so for the block's row-major 32x32 X the rotated block
// is H32 X (H32 / 32), row-major. Tilize X, T = H32 @ X, Y = T @ (H32 / 32) (both factors exact in bf16; HiFi4 keeps
// all of X's bits; T is rounded to bf16, finer than the ternary matmul's LoFi read of x'), untilize Y from DEST.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = tt::CBIndex::c_0, cb_h = tt::CBIndex::c_2;
    constexpr uint32_t cb_xt = tt::CBIndex::c_24, cb_t = tt::CBIndex::c_25, cb_y = tt::CBIndex::c_16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_xt, cb_xt);

    tilize_init(cb_x, 1, cb_xt);
    cb_wait_front(cb_x, 1);
    cb_reserve_back(cb_xt, 1);
    tilize_block(cb_x, 1, cb_xt);
    cb_push_back(cb_xt, 1);
    cb_pop_front(cb_x, 1);
    tilize_uninit(cb_x, cb_xt);

    cb_wait_front(cb_h, 2);
    cb_wait_front(cb_xt, 1);
#ifdef ROT_DEBUG_COPY
    copy_init(cb_xt);
    tile_regs_acquire();
    copy_tile(cb_xt, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest_init<1>(cb_y);
    cb_reserve_back(cb_y, 1);
    pack_untilize_dest<1>(cb_y);
    cb_push_back(cb_y, 1);
    tile_regs_release();
    pack_untilize_uninit(cb_y);
    return;
#endif
    matmul_init(cb_h, cb_xt);
    tile_regs_acquire();
    matmul_tiles(cb_h, cb_xt, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_t, 1);
    pack_reconfig_data_format(cb_t);
    pack_tile(0, cb_t);
    tile_regs_release();
    cb_push_back(cb_t, 1);

    cb_wait_front(cb_t, 1);
    matmul_init(cb_t, cb_h);
    tile_regs_acquire();
    matmul_tiles(cb_t, cb_h, 0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest_init<1>(cb_y);
    cb_reserve_back(cb_y, 1);
    pack_untilize_dest<1>(cb_y);
    cb_push_back(cb_y, 1);
    tile_regs_release();
    pack_untilize_uninit(cb_y);
}
