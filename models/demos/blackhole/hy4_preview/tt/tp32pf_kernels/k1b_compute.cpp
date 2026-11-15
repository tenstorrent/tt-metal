// SPDX-License-Identifier: Apache-2.0
// tp32pf K1b TRISC: per unit (expert resident in cb_w, whole bank runs [b][k][j]) per pair of M-tiles: for each K block
// b: tilize the gathered rows (cb_rm: 2 tile rows x 24 tiles) into cb_x, dest[2 x 4] (+)= x_b (2x24) @ w_b (24x4); the
// partial of blocks 0..6 is spilled to cb_part (bf16) and reloaded into dest before the next block (tilize needs dest).
// After block 7: act = silu(min(g, L)) * clamp(u, -L, L), then scaled per row by the score tile (mul_tiles_bcast_cols)
// -> cb_o2 [m][c]. CT: 0 cb_x, 1 cb_w, 2 cb_o, 3 cb_meta, 4 +L bits, 5 -L bits, 6 -BIG bits, 7 cb_rm, 8 cb_part, 9
// cb_sc, 10 cb_o2, 11 SCALE RT: 0 me
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
    constexpr uint32_t cb_part = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(9);
    constexpr uint32_t cb_o2 = get_compile_time_arg_val(10);
    constexpr uint32_t SCALE = get_compile_time_arg_val(11);
    constexpr uint32_t DBG = get_compile_time_arg_val(12);
    constexpr uint32_t VAR = get_compile_time_arg_val(13);
    constexpr uint32_t PERF = get_compile_time_arg_val(14);
    constexpr uint32_t cb_rm1 = get_compile_time_arg_val(15);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(16);
    const uint32_t me = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);
    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t nu = plan[0];
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mtc = plan[3 + 3 * u];
        cb_wait_front(cb_w, 768);
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            bool continue_j = false;
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t b = (j + me) & 7;
                if (DBG == 2) {
                    cb_wait_front(cb_rm, 24);
                    if (j == 0) {
                        cb_reserve_back(cb_o2, 4);
                        copy_init(cb_rm);
                        for (uint32_t t = 0; t < 4; ++t) {
                            tile_regs_acquire();
                            copy_tile(cb_rm, t, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(0, cb_o2, t);
                            tile_regs_release();
                        }
                        cb_push_back(cb_o2, 4);
                    }
                    cb_pop_front(cb_rm, 24);
                    cb_wait_front(cb_rm, 24);
                    cb_pop_front(cb_rm, 24);
                    if (j == 7) {
                        cb_wait_front(cb_sc, 2);
                        cb_pop_front(cb_sc, 2);
                    }
                    continue;
                }
                reconfig_data_format_srca(cb_w, cb_rm);
                if (!(PERF & 2)) {
                    tilize_init(cb_rm, 24, cb_x);
                }
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t cbh = (SPLIT && h == 1) ? cb_rm1 : cb_rm;
                    cb_wait_front(cbh, 24);
                    cb_reserve_back(cb_x, 24);
                    if (!(PERF & 2)) {
                        tilize_block(cbh, 24, cb_x);
                    }
                    cb_push_back(cb_x, 24);
                    cb_pop_front(cbh, 24);
                }
                if (!(PERF & 2)) {
                    tilize_uninit(cb_rm, cb_x);
                }
                cb_wait_front(cb_x, 48);
                if (DBG == 1) {
                    if (j == 0) {
                        // debug: output tilized x tiles (m, k=0..1) of K block (me & 7) instead of the act
                        cb_reserve_back(cb_o2, 4);
                        copy_init(cb_x);
                        for (uint32_t t = 0; t < 4; ++t) {
                            tile_regs_acquire();
                            copy_tile(cb_x, (t >> 1) * 24 + (t & 1), 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(0, cb_o2, t);
                            tile_regs_release();
                        }
                        cb_push_back(cb_o2, 4);
                    }
                    cb_pop_front(cb_x, 48);
                    if (j == 7) {
                        cb_wait_front(cb_sc, 2);
                        cb_pop_front(cb_sc, 2);
                    }
                    continue;
                }
                if (VAR == 1) {
                    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);
                }
                tile_regs_acquire();
                if (j > 0 && !(DBG >= 3 && DBG < 20 && (j - 1) >= (DBG - 3))) {
                    cb_wait_front(cb_part, 8);
                    copy_init(cb_part);
                    for (uint32_t i = 0; i < 8; ++i) {
                        copy_tile(cb_part, i, i);
                    }
                    cb_pop_front(cb_part, 8);
                }
                if (VAR == 0) {
                    reconfig_data_format_srca(cb_part, cb_w);
                } else if (VAR == 2) {
                    reconfig_data_format<SrcOrder::Reverse>(cb_x, cb_w);
                    pack_reconfig_data_format(cb_o);
                }
                matmul_block_init(cb_x, cb_w, false, 4, 2, 24);
                for (uint32_t k = 0; k < 24; ++k) {
                    matmul_block(cb_x, cb_w, k, b * 96 + k * 4, 0, false, 4, 2, 24);
                }
                cb_pop_front(cb_x, 48);
                if (DBG >= 3 && DBG < 20 && j == DBG - 3) {
                    tile_regs_commit();
                    cb_reserve_back(cb_o2, 4);
                    tile_regs_wait();
                    pack_tile(0, cb_o2, 0);
                    pack_tile(1, cb_o2, 1);
                    pack_tile(4, cb_o2, 2);
                    pack_tile(5, cb_o2, 3);
                    tile_regs_release();
                    cb_push_back(cb_o2, 4);
                    continue_j = true;
                }
                if (DBG >= 3 && DBG < 20 && continue_j) {
                    if (j == 7) {
                        cb_wait_front(cb_sc, 2);
                        cb_pop_front(cb_sc, 2);
                    }
                    continue;
                }
                if (j < 7) {
                    tile_regs_commit();
                    cb_reserve_back(cb_part, 8);
                    tile_regs_wait();
                    for (uint32_t i = 0; i < 8; ++i) {
                        pack_tile(i, cb_part, i);
                    }
                    tile_regs_release();
                    cb_push_back(cb_part, 8);
                }
            }
            if (DBG >= 1 && DBG != 20) {
                continue;
            }
            clamp_tile_init();
            for (uint32_t m = 0; m < 2; ++m) {
                clamp_tile(m * 4 + 0, nbig_bits, lim_bits);
                clamp_tile(m * 4 + 1, nbig_bits, lim_bits);
                clamp_tile(m * 4 + 2, nlim_bits, lim_bits);
                clamp_tile(m * 4 + 3, nlim_bits, lim_bits);
            }
            silu_tile_init();
            for (uint32_t m = 0; m < 2; ++m) {
                silu_tile(m * 4 + 0);
                silu_tile(m * 4 + 1);
            }
            mul_binary_tile_init();
            for (uint32_t m = 0; m < 2; ++m) {
                mul_binary_tile(m * 4 + 0, m * 4 + 2, m * 4 + 0);
                mul_binary_tile(m * 4 + 1, m * 4 + 3, m * 4 + 1);
            }
            tile_regs_commit();
            if (DBG == 20) {
                cb_reserve_back(cb_o2, 4);
                tile_regs_wait();
                pack_tile(0, cb_o2, 0);
                pack_tile(1, cb_o2, 1);
                pack_tile(4, cb_o2, 2);
                pack_tile(5, cb_o2, 3);
                tile_regs_release();
                cb_push_back(cb_o2, 4);
                cb_wait_front(cb_sc, 2);
                cb_pop_front(cb_sc, 2);
                continue;
            }
            cb_reserve_back(cb_o, 4);
            tile_regs_wait();
            pack_tile(0, cb_o, 0);
            pack_tile(1, cb_o, 1);
            pack_tile(4, cb_o, 2);
            pack_tile(5, cb_o, 3);
            tile_regs_release();
            cb_push_back(cb_o, 4);
            // per-row score scale
            cb_wait_front(cb_o, 4);
            cb_wait_front(cb_sc, 2);
            reconfig_data_format_srca(cb_w, cb_o);
            if (SCALE) {
                mul_bcast_cols_init(cb_o, cb_sc);
            } else {
                copy_init(cb_o);
            }
            for (uint32_t m = 0; m < 2; ++m) {
                for (uint32_t c = 0; c < 2; ++c) {
                    tile_regs_acquire();
                    if (SCALE) {
                        mul_tiles_bcast_cols(cb_o, cb_sc, m * 2 + c, m, 0);
                    } else {
                        copy_tile(cb_o, m * 2 + c, 0);
                    }
                    tile_regs_commit();
                    cb_reserve_back(cb_o2, 1);
                    tile_regs_wait();
                    pack_tile(0, cb_o2);
                    tile_regs_release();
                    cb_push_back(cb_o2, 1);
                }
            }
            cb_pop_front(cb_o, 4);
            cb_pop_front(cb_sc, 2);
        }
        cb_pop_front(cb_w, 768);
    }
    cb_pop_front(cb_meta, 1);
}
