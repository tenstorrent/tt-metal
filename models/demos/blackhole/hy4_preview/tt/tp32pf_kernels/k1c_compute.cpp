// SPDX-License-Identifier: Apache-2.0
// tp32pf K1c TRISC: per unit (expert resident in cb_w = [k][4] tiles, k = 0..191) per real M-tile: tilize the 32
// gathered full rows (cb_rm: 192 tiles wide) into cb_x (192 tiles), dest[1 x 4] = x (1x192) @ w (192x4) accumulated
// over the whole K (no spill), then act = silu(min(g, L)) * clamp(u, -L, L) scaled per row by the score tile
// (mul_tiles_bcast_cols) -> cb_o2 [c] (2 tiles). CT: 0 cb_x, 1 cb_w, 2 cb_o, 3 cb_meta, 4 +L bits, 5 -L bits, 6 -BIG
// bits, 7 cb_rm, 8 cb_sc, 9 cb_o2. RT: 0 me
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(3);
    constexpr uint32_t lim_bits = get_compile_time_arg_val(4);
    constexpr uint32_t nlim_bits = get_compile_time_arg_val(5);
    constexpr uint32_t nbig_bits = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(8);
    constexpr uint32_t cb_o2 = get_compile_time_arg_val(9);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);
    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t nu = plan[0];
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t nreal = plan[304 + u];
        cb_wait_front(cb_w, 768);
        for (uint32_t m = 0; m < nreal; ++m) {
            cb_wait_front(cb_rm, 192);
            cb_reserve_back(cb_x, 192);
            reconfig_data_format_srca(cb_w, cb_rm);
            tilize_init(cb_rm, 192, cb_x);
            tilize_block(cb_rm, 192, cb_x);
            tilize_uninit(cb_rm, cb_x);
            cb_push_back(cb_x, 192);
            cb_pop_front(cb_rm, 192);
            cb_wait_front(cb_x, 192);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_rm, cb_w);
            matmul_block_init(cb_x, cb_w, false, 4, 1, 192);
            for (uint32_t k = 0; k < 192; ++k) {
                matmul_block(cb_x, cb_w, k, k * 4, 0, false, 4, 1, 192);
            }
            cb_pop_front(cb_x, 192);
            clamp_tile_init();
            clamp_tile(0, nbig_bits, lim_bits);
            clamp_tile(1, nbig_bits, lim_bits);
            clamp_tile(2, nlim_bits, lim_bits);
            clamp_tile(3, nlim_bits, lim_bits);
            silu_tile_init();
            silu_tile(0);
            silu_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);
            mul_binary_tile(1, 3, 1);
            tile_regs_commit();
            cb_reserve_back(cb_o, 2);
            tile_regs_wait();
            pack_tile(0, cb_o, 0);
            pack_tile(1, cb_o, 1);
            tile_regs_release();
            cb_push_back(cb_o, 2);
            cb_wait_front(cb_o, 2);
            cb_wait_front(cb_sc, 1);
            reconfig_data_format_srca(cb_w, cb_o);
            mul_bcast_cols_init(cb_o, cb_sc);
            for (uint32_t c = 0; c < 2; ++c) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_o, cb_sc, c, 0, 0);
                tile_regs_commit();
                cb_reserve_back(cb_o2, 1);
                tile_regs_wait();
                pack_tile(0, cb_o2);
                tile_regs_release();
                cb_push_back(cb_o2, 1);
            }
            cb_pop_front(cb_o, 2);
            cb_pop_front(cb_sc, 1);
        }
        cb_pop_front(cb_w, 768);
    }
    cb_pop_front(cb_meta, 1);
}
