// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per-head RMSNorm fused into the QKV head-split. For each work unit the reader
// delivers Q | K | V tiles (unit_tiles) in CB 0; this kernel writes the same
// layout to CB 16 with every Q and K head normalised over head_dim and scaled
// by gamma, and V copied through. Per head (Wt = head_dim_tiles tiles wide):
//   x2  = x * x                        (CB 5)
//   ms  = row-sum(x2) * 1/head_dim     (CB 6, column 0 holds the row values)
//   inv = rsqrt(ms + eps)              (CB 7)
//   y   = x * bcast_cols(inv)          (CB 8)
//   out = y * gamma_tiles              (CB 16)   gamma tiles are row-replicated
// Every phase uses <= Wt DST tiles, so fp32 accumulation (4 DST tiles) is fine.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_in = 0, cb_gq = 1, cb_gk = 2, cb_scaler = 3, cb_eps = 4;
constexpr uint32_t cb_x2 = 5, cb_red = 6, cb_inv = 7, cb_tmp = 8, cb_out = 16;

template <uint32_t Wt>
inline void norm_head(uint32_t in_off, uint32_t out_off, uint32_t cb_gamma) {
    CircularBuffer x2(cb_x2), red(cb_red), inv(cb_inv), tmp(cb_tmp);
    // x^2
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_x2);
    mul_init(cb_in, cb_in);
    x2.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_in, cb_in, in_off + j, in_off + j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_x2, j);
    }
    tile_regs_release();
    x2.push_back(Wt);
    // mean of squares: row-reduce across the Wt tiles, scaler tile = 1/head_dim
    reconfig_data_format(cb_x2, cb_scaler);
    pack_reconfig_data_format(cb_red);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_scaler, cb_red);
    x2.wait_front(Wt);
    red.reserve_back(1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_scaler, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_red);
    tile_regs_release();
    red.push_back(1);
    x2.pop_front(Wt);
    reduce_uninit(cb_x2);
    // inv = rsqrt(ms + eps)
    reconfig_data_format(cb_red, cb_eps);
    pack_reconfig_data_format(cb_inv);
    add_init(cb_red, cb_eps);
    red.wait_front(1);
    inv.reserve_back(1);
    tile_regs_acquire();
    add_tiles(cb_red, cb_eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv);
    tile_regs_release();
    inv.push_back(1);
    red.pop_front(1);
    // y = x * bcast_cols(inv)
    reconfig_data_format(cb_in, cb_inv);
    pack_reconfig_data_format(cb_tmp);
    mul_bcast_cols_init(cb_in, cb_inv);
    inv.wait_front(1);
    tmp.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles_bcast_cols(cb_in, cb_inv, in_off + j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_tmp, j);
    }
    tile_regs_release();
    tmp.push_back(Wt);
    inv.pop_front(1);
    // out = y * gamma
    reconfig_data_format(cb_tmp, cb_gamma);
    pack_reconfig_data_format(cb_out);
    mul_init(cb_tmp, cb_gamma);
    tmp.wait_front(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_tmp, cb_gamma, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_out, out_off + j);
    }
    tile_regs_release();
    tmp.pop_front(Wt);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);  // head_dim_tiles
    const uint32_t num_work_units = get_arg_val<uint32_t>(0);

    constexpr uint32_t q_heads_per_group = heads_per_group * q_heads_per_kv;
    constexpr uint32_t group_q_tiles = q_heads_per_group * Wt;
    constexpr uint32_t group_kv_tiles = heads_per_group * Wt;
    constexpr uint32_t unit_tiles = group_q_tiles + 2 * group_kv_tiles;

    CircularBuffer in(cb_in), out(cb_out), gq(cb_gq), gk(cb_gk), sc(cb_scaler), ep(cb_eps);
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    gq.wait_front(Wt);
    gk.wait_front(Wt);
    sc.wait_front(1);
    ep.wait_front(1);

    for (uint32_t w = 0; w < num_work_units; ++w) {
        in.wait_front(unit_tiles);
        out.reserve_back(unit_tiles);
        for (uint32_t h = 0; h < q_heads_per_group; ++h) {
            norm_head<Wt>(h * Wt, h * Wt, cb_gq);
        }
        for (uint32_t h = 0; h < heads_per_group; ++h) {
            norm_head<Wt>(group_q_tiles + h * Wt, group_q_tiles + h * Wt, cb_gk);
        }
        // V passes through
        reconfig_data_format(cb_in, cb_in);
        pack_reconfig_data_format(cb_out);
        copy_init(cb_in);
        for (uint32_t t0 = 0; t0 < group_kv_tiles; t0 += Wt) {
            const uint32_t n = (group_kv_tiles - t0 < Wt) ? (group_kv_tiles - t0) : Wt;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                copy_tile(cb_in, group_q_tiles + group_kv_tiles + t0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_out, group_q_tiles + group_kv_tiles + t0 + j);
            }
            tile_regs_release();
        }
        out.push_back(unit_tiles);
        in.pop_front(unit_tiles);
    }
}
