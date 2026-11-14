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
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#ifndef ROT_CB_X  // CB indices (the fused form inside a ternary decode linear moves them off the linear's)
#define ROT_CB_X 0
#define ROT_CB_M 1
#define ROT_CB_H 2
#define ROT_CB_SCR 3
#define ROT_CB_U 4
#define ROT_CB_Y 16
#endif
#ifndef ROT_FIRST  // the fused form's rotation cores start after the linear's
#define ROT_FIRST 0
#endif

void kernel_main() {
    constexpr uint32_t cb_x = ROT_CB_X, cb_h = ROT_CB_H;
    constexpr uint32_t cb_xt = tt::CBIndex::c_24, cb_t = tt::CBIndex::c_25, cb_y = ROT_CB_Y;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_xt, cb_xt);

    tilize_init(cb_x, 1, cb_xt);
    cb_wait_front(cb_x, 1);
    cb_reserve_back(cb_xt, 1);
    tilize_block(cb_x, 1, cb_xt);
    cb_push_back(cb_xt, 1);
    cb_pop_front(cb_x, 1);
    tilize_uninit(cb_x, cb_xt);
#ifdef ROT_GLU
    // silu(x) * (D u) into cb_g, which then takes x's place in the rotation.
    constexpr uint32_t cb_u = ROT_CB_U, cb_ut = tt::CBIndex::c_26, cb_g = tt::CBIndex::c_27;
    tilize_init(cb_u, 1, cb_ut);
    cb_wait_front(cb_u, 1);
    cb_reserve_back(cb_ut, 1);
    tilize_block(cb_u, 1, cb_ut);
    cb_push_back(cb_ut, 1);
    cb_pop_front(cb_u, 1);
    tilize_uninit(cb_u, cb_ut);
    cb_wait_front(cb_xt, 1);
    cb_wait_front(cb_ut, 1);
    reconfig_data_format(cb_xt, cb_xt);
    copy_init(cb_xt);
    tile_regs_acquire();
    copy_tile(cb_xt, 0, 0);
#ifdef ROT_SIGMUL
    sigmoid_tile_init();  // ROT_SIGMUL: sigmoid(x) * (D u) (the attention's gate)
    sigmoid_tile(0);
#else
    silu_tile_init();
    silu_tile(0);
#endif
    reconfig_data_format(cb_ut, cb_ut);
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_ut);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_ut, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_g, 1);
    pack_reconfig_data_format(cb_g);
    pack_tile(0, cb_g);
    tile_regs_release();
    cb_push_back(cb_g, 1);
    cb_pop_front(cb_xt, 1);
    cb_pop_front(cb_ut, 1);
    constexpr uint32_t cb_in = cb_g;
#else
    constexpr uint32_t cb_in = cb_xt;
#endif
    // D: the block's signs (a +-1 tile, tiled like the block; the reader no longer flips sign bits word by word)
    constexpr uint32_t cb_m = ROT_CB_M, cb_d = tt::CBIndex::c_28;
    cb_wait_front(cb_m, 1);
    cb_wait_front(cb_in, 1);
    reconfig_data_format(cb_in, cb_m);
    mul_init(cb_in, cb_m);
    tile_regs_acquire();
    mul_tiles(cb_in, cb_m, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_d, 1);
    pack_reconfig_data_format(cb_d);
    pack_tile(0, cb_d);
    tile_regs_release();
    cb_push_back(cb_d, 1);
    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_m, 1);

    cb_wait_front(cb_h, 2);
    cb_wait_front(cb_d, 1);
#ifdef ROT_DEBUG_COPY
    copy_init(cb_d);
    tile_regs_acquire();
    copy_tile(cb_d, 0, 0);
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
    reconfig_data_format<SrcOrder::Reverse>(cb_h, cb_d);
    matmul_init(cb_h, cb_d);
    tile_regs_acquire();
    matmul_tiles(cb_h, cb_d, 0, 0, 0);
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
