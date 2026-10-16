// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISC of the fused reduce + residual + RMSNorm. Per compact tile, exactly moe_reduce_residual_compute.cpp: the
// chip-ordered fp32 sum of the copies, one bf16 rounding, plus the residual -> y (bf16, to cb_out and cb_y). Then the
// RMSNorm of y's user rows, all in compact form (row r of a compact tile is user r % 4):
//   partial = M @ ((sum_t y_t^2) @ J)    J all ones: row sums broadcast over the columns; M[r][c] = (r % 4 == c % 4):
//                                        each row then holds its user's sum over this core's compact tiles
//   G: rows 4p..4p+3 = rows 0-3 of core p's partial (sent by every core's writer)
//   w = rsqrt(M @ G / hidden + eps)                (the same on every core and chip)
//   h_t = y_t * w * gamma_t                        (gamma packed like the compact tiles)
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

using namespace ckernel;

namespace {

// dst[idst] <- cb[tile] (unpacked through srcA in cb's format).
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t idst) {
    reconfig_data_format_srca(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, tile, idst);
}

// Packs dst[0] into one new page of cb (in cb's format) and releases the registers.
FORCE_INLINE void pack_one(uint32_t cb) {
    tile_regs_commit();
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}

// out <- in0[t0] @ in1[t1] (one tile, fp32).
FORCE_INLINE void mm(uint32_t in0, uint32_t t0, uint32_t in1, uint32_t t1, uint32_t out) {
    reconfig_data_format(in1, in0);  // matmul unpacks in1 to srcA and in0 to srcB
    matmul_init(in0, in1);
    tile_regs_acquire();
    matmul_tiles(in0, in1, t0, t1, 0);
    pack_one(out);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mid = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t tiles = get_compile_time_arg_val(4);
    constexpr uint32_t chips = get_compile_time_arg_val(5);
    constexpr uint32_t cb_y = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_const = get_compile_time_arg_val(8);  // tile 0: J, tile 1: M (fp32)
    constexpr uint32_t cb_rs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_part = get_compile_time_arg_val(10);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(11);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(12);
    constexpr uint32_t cb_h = get_compile_time_arg_val(13);
    constexpr uint32_t inv_hidden_bits = get_compile_time_arg_val(14);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(15);

    // FPU adds: the chips' bf16 copies pairwise into fp32 Dest (A + B + Dest), then the bf16-rounded sum plus the
    // residual.
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    cb_wait_front(cb_res, tiles);
    for (uint32_t t = 0; t < tiles; t++) {
        cb_wait_front(cb_in, chips);
        tile_regs_acquire();
        add_init(cb_in, cb_in, true);
        for (uint32_t c = 0; c < chips; c += 2) {
            add_tiles(cb_in, cb_in, c, c + 1, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_in, chips);

        cb_reserve_back(cb_mid, 1);
        tile_regs_wait();
        pack_tile(0, cb_mid);
        tile_regs_release();
        cb_push_back(cb_mid, 1);

        cb_wait_front(cb_mid, 1);
        tile_regs_acquire();
        add_init(cb_res, cb_mid);
        add_tiles(cb_res, cb_mid, t, 0, 0);
        tile_regs_commit();
        cb_pop_front(cb_mid, 1);

        cb_reserve_back(cb_out, 1);
        cb_reserve_back(cb_y, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        pack_tile(0, cb_y);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_push_back(cb_y, 1);
    }
    cb_pop_front(cb_res, tiles);

    // sum_t y_t^2 (fp32)
    cb_wait_front(cb_y, tiles);
    tile_regs_acquire();
    load(cb_y, 0, 0);
    square_tile_init();
    square_tile(0);
    for (uint32_t t = 1; t < tiles; t++) {
        load(cb_y, t, 1);
        square_tile_init();
        square_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    pack_one(cb_sq);

    // partial = M @ (sq @ J)
    cb_wait_front(cb_const, 2);
    cb_wait_front(cb_sq, 1);
    mm(cb_sq, 0, cb_const, 0, cb_rs);
    cb_pop_front(cb_sq, 1);
    cb_wait_front(cb_rs, 1);
    mm(cb_const, 1, cb_rs, 0, cb_part);
    cb_pop_front(cb_rs, 1);

    // While the partials travel: y_t * gamma_t into dst 1 + 2t. Then w = rsqrt(M @ G / hidden + eps) into dst 0 (G row
    // 4p + u holds core p's partial for user u, so M @ G sums the cores in one matmul, the same on every core and
    // chip) and h_t = (y_t * gamma_t) * w.
    static_assert(tiles <= 2, "dst holds w, a gamma scratch tile and two h tiles");
    cb_wait_front(cb_gamma, tiles);
    tile_regs_acquire();
    for (uint32_t t = 0; t < tiles; t++) {
        const uint32_t d = 1 + 2 * t;
        load(cb_y, t, d);
        load(cb_gamma, t, 2);
        mul_binary_tile_init();
        mul_binary_tile(d, 2, d);
    }
    cb_wait_front(cb_gather, 1);
    reconfig_data_format(cb_gather, cb_const);
    matmul_init(cb_const, cb_gather);
    matmul_tiles(cb_const, cb_gather, 1, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_hidden_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    mul_binary_tile_init();
    for (uint32_t t = 0; t < tiles; t++) {
        mul_binary_tile(1 + 2 * t, 0, 1 + 2 * t);
    }
    tile_regs_commit();
    cb_reserve_back(cb_h, tiles);
    tile_regs_wait();
    pack_reconfig_data_format(cb_h);
    for (uint32_t t = 0; t < tiles; t++) {
        pack_tile(1 + 2 * t, cb_h);
    }
    tile_regs_release();
    cb_push_back(cb_h, tiles);
    cb_pop_front(cb_gather, 1);
    cb_pop_front(cb_const, 2);
    cb_pop_front(cb_gamma, tiles);
    cb_pop_front(cb_y, tiles);
}
