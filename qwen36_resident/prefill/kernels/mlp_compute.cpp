// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill engine MLP stack, compute. Core (r, c) of the R x Q grid holds the residual stream x as rows of
// row group r (mt tiles) by hidden tiles [c * HC, (c + 1) * HC) (rows W tiles wide, W >= HC). Per layer:
//   h = rmsnorm(x)                 row sums of squares: this core's part, summed over the row's Q parts
//   g = h @ G, u = h @ U           [mt, IC] blocks of output columns [c * IC, (c + 1) * IC)
//   a = silu(g) * u
//   x += a @ D                     [mt, W] output columns, the first HC are this core's x tiles
// Each matmul accumulates over its K blocks in L1 through the packer: block b brings in0 = the row's
// activation block b (mt x S0 tiles, kb valid columns) and in1 = the column's weight rows (slot x N tiles).
//
// x, h, a, u and d are fixed L1 buffers that are never pushed: tiles are addressed from the CB base. The
// movers learn that a buffer is ready or free from one-page tokens: cb_h_rdy and cb_a_rdy (h, a written)
// and cb_u_free (u consumed; its memory takes the down weights).
//
// Compile-time args: 0 mt, 1 HC, 2 W, 3 IC, 4 Q, 5 S0, 6 GU blocks per owner, 7 GU slot, 8 D blocks per
// owner, 9 D slot, 10 layers, 11 eps (fp32 bits), 12 1 / hidden (fp32 bits).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

namespace {
constexpr uint32_t cb_in0 = 0, cb_gu = 1, cb_dw = 2, cb_x = 10, cb_h = 11, cb_a = 12, cb_u = 13, cb_d = 14;
constexpr uint32_t cb_part = 16, cb_recv = 17, cb_rs = 18, cb_scaler = 19, cb_sq = 20;
constexpr uint32_t cb_h_rdy = 24, cb_a_rdy = 25, cb_u_free = 26;

constexpr uint32_t mt = get_compile_time_arg_val(0);
constexpr uint32_t HC = get_compile_time_arg_val(1);
constexpr uint32_t W = get_compile_time_arg_val(2);
constexpr uint32_t IC = get_compile_time_arg_val(3);
constexpr uint32_t Q = get_compile_time_arg_val(4);
constexpr uint32_t S0 = get_compile_time_arg_val(5);
constexpr uint32_t gu_blocks = get_compile_time_arg_val(6);
constexpr uint32_t gu_slot = get_compile_time_arg_val(7);
constexpr uint32_t d_blocks = get_compile_time_arg_val(8);
constexpr uint32_t d_slot = get_compile_time_arg_val(9);
constexpr uint32_t layers = get_compile_time_arg_val(10);
constexpr uint32_t eps = get_compile_time_arg_val(11);
constexpr uint32_t inv_hidden = get_compile_time_arg_val(12);
constexpr uint32_t sh = 4, sw = 2;  // output subblock: 8 bf16 tiles of DEST
static_assert(mt % sh == 0 && IC % sw == 0 && W % sw == 0, "output blocks split into subblocks");

inline void pack_to(uint32_t cb, uint32_t idx) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb, idx);
    tile_regs_release();
}

// elementwise passes work kEw tiles per DEST acquire (8 bf16 tiles fit half of DEST)
constexpr uint32_t kEw = 8;
static_assert((mt * IC) % kEw == 0, "a and u split into DEST groups");

inline void pack_block(uint32_t cb, uint32_t idx) {
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < kEw; t++) {
        pack_tile<true>(t, cb, idx + t);
    }
    tile_regs_release();
}

inline void token(uint32_t cb) {
    cb_reserve_back(cb, 1);
    cb_push_back(cb, 1);
}

// acc[mt, N] = sum over the K blocks of in0 @ in1 (pack accumulation in L1); an owner holds `per` K tiles
// in blocks of `slot`, the last one narrower
template <uint32_t N>
void matmul(uint32_t in1, uint32_t acc, uint32_t blocks_per_owner, uint32_t slot, uint32_t per) {
    reconfig_data_format(in1, cb_in0);
    pack_reconfig_data_format(acc);
    matmul_block_init(cb_in0, in1, 0, sw, sh, S0);
    for (uint32_t b = 0; b < Q * blocks_per_owner; b++) {
        const uint32_t i = b % blocks_per_owner;
        const uint32_t kb = i + 1 < blocks_per_owner ? slot : per - slot * (blocks_per_owner - 1);
        cb_wait_front(cb_in0, mt * S0);
        cb_wait_front(in1, slot * N);
        pack_reconfig_l1_acc(b > 0 ? 1 : 0);
        for (uint32_t i0 = 0; i0 < mt; i0 += sh) {
            for (uint32_t j0 = 0; j0 < N; j0 += sw) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < kb; k++) {
                    matmul_block(cb_in0, in1, i0 * S0 + k, k * N + j0, 0, false, sw, sh, S0);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t r = 0; r < sh; r++) {
                    for (uint32_t j = 0; j < sw; j++) {
                        pack_tile<true>(r * sw + j, acc, (i0 + r) * N + j0 + j);
                    }
                }
                tile_regs_release();
            }
        }
        cb_pop_front(cb_in0, mt * S0);
        cb_pop_front(in1, slot * N);
    }
    pack_reconfig_l1_acc(0);
}

void norm() {
    // this core's part of the row sums of x^2
    cb_reserve_back(cb_part, mt);
    for (uint32_t i = 0; i < mt; i++) {
        cb_reserve_back(cb_sq, HC);
        reconfig_data_format(cb_x, cb_x);
        pack_reconfig_data_format(cb_sq);
        mul_tiles_init(cb_x, cb_x);
        for (uint32_t j = 0; j < HC; j++) {
            tile_regs_acquire();
            mul_tiles(cb_x, cb_x, i * W + j, i * W + j, 0);
            pack_to(cb_sq, j);
        }
        cb_push_back(cb_sq, HC);
        cb_wait_front(cb_sq, HC);
        reconfig_data_format(cb_scaler, cb_sq);
        pack_reconfig_data_format(cb_part);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_part);
        tile_regs_acquire();
        for (uint32_t j = 0; j < HC; j++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, j, 0, 0);
        }
        reduce_uninit();
        pack_to(cb_part, i);
        cb_pop_front(cb_sq, HC);
    }
    cb_push_back(cb_part, mt);
    // rs = rsqrt(sum of the row's parts / hidden + eps); cb_recv tile q * mt + i is column q's part of row i
    cb_wait_front(cb_recv, Q * mt);
    cb_reserve_back(cb_rs, mt);
    reconfig_data_format_srca(cb_recv);
    pack_reconfig_data_format(cb_rs);
    for (uint32_t i = 0; i < mt; i++) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_recv);
        copy_tile(cb_recv, i, 0);
        for (uint32_t q = 1; q < Q; q++) {
            copy_tile_to_dst_init_short(cb_recv);
            copy_tile(cb_recv, q * mt + i, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_hidden);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_to(cb_rs, i);
    }
    cb_push_back(cb_rs, mt);
    cb_pop_front(cb_recv, Q * mt);
    // h = x * rs
    cb_wait_front(cb_rs, mt);
    reconfig_data_format(cb_x, cb_rs);
    pack_reconfig_data_format(cb_h);
    mul_bcast_cols_init(cb_x, cb_rs);
    for (uint32_t i = 0; i < mt; i++) {
        for (uint32_t j = 0; j < HC; j++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_x, cb_rs, i * W + j, i, 0);
            pack_to(cb_h, i * W + j);
        }
    }
    cb_pop_front(cb_rs, mt);
    token(cb_h_rdy);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_gu, cb_a);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t l = 0; l < layers; l++) {
        norm();
        // g, then silu(g) in place
        matmul<IC>(cb_gu, cb_a, gu_blocks, gu_slot, HC);
        reconfig_data_format_srca(cb_a);
        pack_reconfig_data_format(cb_a);
        copy_tile_to_dst_init_short(cb_a);
        silu_tile_init();
        for (uint32_t t0 = 0; t0 < mt * IC; t0 += kEw) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < kEw; t++) {
                copy_tile(cb_a, t0 + t, t);
                silu_tile(t);
            }
            pack_block(cb_a, t0);
        }
        // u, then a = silu(g) * u
        matmul<IC>(cb_gu, cb_u, gu_blocks, gu_slot, HC);
        reconfig_data_format(cb_a, cb_u);
        pack_reconfig_data_format(cb_a);
        mul_tiles_init(cb_a, cb_u);
        for (uint32_t t0 = 0; t0 < mt * IC; t0 += kEw) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < kEw; t++) {
                mul_tiles(cb_a, cb_u, t0 + t, t0 + t, t);
            }
            pack_block(cb_a, t0);
        }
        token(cb_a_rdy);
        token(cb_u_free);
        // d = a @ D, x += d
        matmul<W>(cb_dw, cb_d, d_blocks, d_slot, IC);
        reconfig_data_format(cb_x, cb_d);
        pack_reconfig_data_format(cb_x);
        add_tiles_init(cb_x, cb_d);
        for (uint32_t i = 0; i < mt; i++) {
            for (uint32_t j = 0; j < HC; j++) {
                tile_regs_acquire();
                add_tiles(cb_x, cb_d, i * W + j, i * W + j, 0);
                pack_to(cb_x, i * W + j);
            }
        }
    }
}
