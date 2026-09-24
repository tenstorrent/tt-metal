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
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_in = 0, cb_gq = 1, cb_gk = 2, cb_scaler = 3, cb_eps = 4;
constexpr uint32_t cb_x2 = 5, cb_red = 6, cb_inv = 7, cb_tmp = 8, cb_out = 16;
constexpr uint32_t cb_cos = 9, cb_sin = 10, cb_trans = 11, cb_rot = 12, cb_si = 13, cb_ci = 14, cb_norm = 15;
constexpr uint32_t cb_qsep = 17;  // Q output when it has its own dtype (e.g. bfp8 for SDPA); K|V stay in cb_out

template <uint32_t Wt, uint32_t cb_dst>
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
    // out = y * gamma  (to cb_out at out_off, or to cb_norm at 0 when rotary follows)
    reconfig_data_format(cb_tmp, cb_gamma);
    pack_reconfig_data_format(cb_dst);
    mul_init(cb_tmp, cb_gamma);
    cb_wait_front(cb_gamma, Wt);  // resident, never popped: only the first call waits (gamma read after unit 0)
    tmp.wait_front(Wt);
    CircularBuffer nrm(cb_norm);
    if constexpr (cb_dst == cb_norm) {
        nrm.reserve_back(Wt);
    }
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_tmp, cb_gamma, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_dst, (cb_dst == cb_norm ? 0 : out_off) + j);
    }
    tile_regs_release();
    tmp.pop_front(Wt);
    if constexpr (cb_dst == cb_norm) {
        nrm.push_back(Wt);
    }
}

// RoPE on one normalised head sitting in cb_norm: out = x*cos + (x @ T)*sin, where T is the
// single 32x32 tile-local rotation the model's rotary_embedding_llama applies to every tile.
template <uint32_t Wt, uint32_t cb_dst>
inline void rotary_head(uint32_t out_off) {
    CircularBuffer nrm(cb_norm), rot(cb_rot), si(cb_si), ci(cb_ci);
    nrm.wait_front(Wt);
    reconfig_data_format(cb_norm, cb_trans);
    pack_reconfig_data_format(cb_rot);
    matmul_init(cb_norm, cb_trans);
    rot.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        matmul_tiles(cb_norm, cb_trans, j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_rot, j);
    }
    tile_regs_release();
    rot.push_back(Wt);
    reconfig_data_format(cb_rot, cb_sin);
    pack_reconfig_data_format(cb_si);
    mul_init(cb_rot, cb_sin);
    rot.wait_front(Wt);
    si.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_rot, cb_sin, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_si, j);
    }
    tile_regs_release();
    si.push_back(Wt);
    rot.pop_front(Wt);
    reconfig_data_format(cb_norm, cb_cos);
    pack_reconfig_data_format(cb_ci);
    mul_init(cb_norm, cb_cos);
    ci.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_norm, cb_cos, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_ci, j);
    }
    tile_regs_release();
    ci.push_back(Wt);
    nrm.pop_front(Wt);
    reconfig_data_format(cb_ci, cb_si);
    pack_reconfig_data_format(cb_dst);
    add_init(cb_ci, cb_si);
    ci.wait_front(Wt);
    si.wait_front(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        add_tiles(cb_ci, cb_si, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_dst, out_off + j);
    }
    tile_regs_release();
    ci.pop_front(Wt);
    si.pop_front(Wt);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);  // head_dim_tiles
    constexpr uint32_t fuse_rotary = get_compile_time_arg_val(3);
    constexpr uint32_t separate_q = get_compile_time_arg_val(4);  // Q -> cb_qsep (own dtype), K|V -> cb_out
    constexpr uint32_t q_split = get_compile_time_arg_val(8);     // 1, or 2: unit = half the Q heads + (K xor V)
    const uint32_t num_work_units = get_arg_val<uint32_t>(0);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(1);

    constexpr uint32_t q_heads_per_group = heads_per_group * q_heads_per_kv;
    constexpr uint32_t sub_q_heads = q_heads_per_group / q_split;
    constexpr uint32_t group_q_tiles = q_heads_per_group * Wt;
    constexpr uint32_t sub_q_tiles = sub_q_heads * Wt;
    constexpr uint32_t group_kv_tiles = heads_per_group * Wt;
    constexpr uint32_t kv_parts = (q_split == 1) ? 2 : 1;
    constexpr uint32_t unit_tiles = sub_q_tiles + kv_parts * group_kv_tiles;
    constexpr uint32_t cb_qout = separate_q ? cb_qsep : cb_out;
    constexpr uint32_t kv_base = separate_q ? 0 : sub_q_tiles;  // K|V offset inside cb_out
    constexpr uint32_t out_tiles = separate_q ? kv_parts * group_kv_tiles : unit_tiles;
    constexpr uint32_t cb_qnorm = fuse_rotary ? cb_norm : cb_qout;
    constexpr uint32_t cb_knorm = fuse_rotary ? cb_norm : cb_out;

    CircularBuffer in(cb_in), out(cb_out), qout(cb_qout), gq(cb_gq), gk(cb_gk), sc(cb_scaler), ep(cb_eps);
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    sc.wait_front(1);
    ep.wait_front(1);
    if constexpr (fuse_rotary) {
        CircularBuffer ct(cb_trans);
        ct.wait_front(1);
    }

    uint32_t sub = work_unit_start % q_split;  // Q half of the current unit, advanced as a rotating counter
    for (uint32_t w = 0; w < num_work_units; ++w) {
        const bool has_k = (q_split == 1) || sub == 0;
        const bool has_v = (q_split == 1) || sub == 1;
        const uint32_t v_in_off = sub_q_tiles + (has_k ? group_kv_tiles : 0);
        const uint32_t v_out_off = kv_base + (has_k ? group_kv_tiles : 0);
        in.wait_front(unit_tiles);
        out.reserve_back(out_tiles);
        if constexpr (separate_q) {
            qout.reserve_back(sub_q_tiles);
        }
        if constexpr (fuse_rotary) {
            CircularBuffer ccos(cb_cos), csin(cb_sin);
            ccos.wait_front(Wt);
            csin.wait_front(Wt);
        }
        for (uint32_t h = 0; h < sub_q_heads; ++h) {
            norm_head<Wt, cb_qnorm>(h * Wt, h * Wt, cb_gq);
            if constexpr (fuse_rotary) {
                rotary_head<Wt, cb_qout>(h * Wt);
            }
        }
        if (has_k) {
            for (uint32_t h = 0; h < heads_per_group; ++h) {
                norm_head<Wt, cb_knorm>(sub_q_tiles + h * Wt, kv_base + h * Wt, cb_gk);
                if constexpr (fuse_rotary) {
                    rotary_head<Wt, cb_out>(kv_base + h * Wt);
                }
            }
        }
        if constexpr (fuse_rotary) {
            CircularBuffer ccos(cb_cos), csin(cb_sin);
            ccos.pop_front(Wt);
            csin.pop_front(Wt);
        }
        // V passes through
        if (has_v) {
            reconfig_data_format(cb_in, cb_in);
            pack_reconfig_data_format(cb_out);
            copy_init(cb_in);
            for (uint32_t t0 = 0; t0 < group_kv_tiles; t0 += Wt) {
                const uint32_t n = (group_kv_tiles - t0 < Wt) ? (group_kv_tiles - t0) : Wt;
                tile_regs_acquire();
                for (uint32_t j = 0; j < n; ++j) {
                    copy_tile(cb_in, v_in_off + t0 + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < n; ++j) {
                    pack_tile(j, cb_out, v_out_off + t0 + j);
                }
                tile_regs_release();
            }
        }
        if constexpr (separate_q) {
            qout.push_back(sub_q_tiles);
        }
        out.push_back(out_tiles);
        in.pop_front(unit_tiles);
        if (++sub == q_split) {
            sub = 0;
        }
    }
}
