// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// bs1 compute of the fused head-split + RMSNorm + RoPE op (QWEN_FUSED_COMPUTE_V3=1, bs1 default; batched sizes keep
// compute_qkv_heads_norm.cpp). v1's math, tile for tile and in the same order (so the output is bit-identical), but
// each phase runs once per unit over all of the unit's normalised heads (its Q heads and, when the unit carries K, its
// K heads, which sit contiguously in the unit) instead of once per head. Every phase pays its data-format reconfig,
// *_init and CB handshakes once per unit; at bs1 that is 9 phase set-ups for 5 heads instead of 45 (heads op 41.7 ->
// 28.7 us in-model). At bs8/16/32 it is bit-identical too but 0.4-1.2% slower end to end, so it stays bs1-only
// (NEGATIVE_RESULTS 53). Only the gamma multiply (gamma_q vs gamma_k) and the last RoPE step (Q output CB vs K|V
// output CB) split into a Q and a K loop. Per unit, over nh heads of Wt = head_dim_tiles tiles:
//   x2  = x * x                        (CB 5,  nh*Wt tiles)
//   ms  = row-sum(x2) * 1/head_dim     (CB 6,  nh tiles; one DST slot per head, <= 4 heads per acquire)
//   inv = rsqrt(ms + eps)              (CB 7,  nh tiles)
//   y   = x * bcast_cols(inv)          (CB 8,  nh*Wt)
//   xn  = y * gamma                    (CB 15 when rotary follows, else the outputs)
//   rot = xn @ T; si = rot * sin; ci = xn * cos; out = ci + si   (CBs 12 / 13 / 14 -> outputs)
// Every phase moves the CBs' full capacity (CAP heads) even when a unit normalises fewer heads (q_split 2 without K),
// so each CB's pointers return to 0 and the indexed tile accesses never run past the end of a CB. Same CB contract
// with the reader and writer as v1. One instantiation of each phase with compile-time CB ids keeps the binary about
// v1's size: this program and the next (SDPA) must fit the 69 KB kernel-config buffer together, or the dispatcher
// cannot stage the next launch while this one runs.
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
constexpr uint32_t cb_qsep = 17;
constexpr uint32_t kDst = 4;  // DST tiles per acquire (fp32 accumulation, half sync)

// op(h, j, dst_slot) for heads [h_begin, h_end), Wt tiles per acquire; tile j of head h is packed to cb_pack at
// pack_base + (h - h_begin) * Wt + j.
template <uint32_t Wt, uint32_t cb_pack, typename Op>
inline void per_head(uint32_t h_begin, uint32_t h_end, uint32_t pack_base, Op op) {
    static_assert(Wt <= kDst, "one head's tiles must fit one DST acquire");
    for (uint32_t h = h_begin; h < h_end; ++h) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            op(h, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, cb_pack, pack_base + (h - h_begin) * Wt + j);
        }
        tile_regs_release();
    }
}

// All phases for one unit: nq Q heads (tiles [0, nq*Wt) of cb_in) and nk K heads right after them. Q results go to
// cb_qout at 0, K results to cb_out at kv_base.
template <uint32_t CAP, uint32_t Wt, bool rotary, uint32_t cb_qout>
inline void unit_heads(uint32_t nq, uint32_t nk, uint32_t kv_base) {
    CircularBuffer x2(cb_x2), red(cb_red), inv(cb_inv), tmp(cb_tmp), nrm(cb_norm);
    constexpr uint32_t T = CAP * Wt;  // tiles moved per phase (the CB capacity)
    const uint32_t nh = nq + nk;
    // x^2
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_x2);
    mul_init(cb_in, cb_in);
    x2.reserve_back(T);
    per_head<Wt, cb_x2>(
        0, nh, 0, [](uint32_t h, uint32_t j, uint32_t d) { mul_tiles(cb_in, cb_in, h * Wt + j, h * Wt + j, d); });
    x2.push_back(T);
    // mean of squares: head h reduces into DST slot h % kDst
    reconfig_data_format(cb_x2, cb_scaler);
    pack_reconfig_data_format(cb_red);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_scaler, cb_red);
    x2.wait_front(T);
    red.reserve_back(CAP);
    for (uint32_t h0 = 0; h0 < nh; h0 += kDst) {
        const uint32_t n = (nh - h0 < kDst) ? (nh - h0) : kDst;
        tile_regs_acquire();
        for (uint32_t h = 0; h < n; ++h) {
            for (uint32_t j = 0; j < Wt; ++j) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x2, cb_scaler, (h0 + h) * Wt + j, 0, h);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t h = 0; h < n; ++h) {
            pack_tile(h, cb_red, h0 + h);
        }
        tile_regs_release();
    }
    red.push_back(CAP);
    x2.pop_front(T);
    reduce_uninit(cb_x2);
    // inv = rsqrt(ms + eps)
    reconfig_data_format(cb_red, cb_eps);
    pack_reconfig_data_format(cb_inv);
    red.wait_front(CAP);
    inv.reserve_back(CAP);
    for (uint32_t h0 = 0; h0 < nh; h0 += kDst) {
        const uint32_t n = (nh - h0 < kDst) ? (nh - h0) : kDst;
        add_init(cb_red, cb_eps);  // rsqrt_tile_init reprograms the SFPU, so the add is re-initialised per block
        tile_regs_acquire();
        for (uint32_t h = 0; h < n; ++h) {
            add_tiles(cb_red, cb_eps, h0 + h, 0, h);
        }
        rsqrt_tile_init();
        for (uint32_t h = 0; h < n; ++h) {
            rsqrt_tile(h);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t h = 0; h < n; ++h) {
            pack_tile(h, cb_inv, h0 + h);
        }
        tile_regs_release();
    }
    inv.push_back(CAP);
    red.pop_front(CAP);
    // y = x * bcast_cols(inv)
    reconfig_data_format(cb_in, cb_inv);
    pack_reconfig_data_format(cb_tmp);
    mul_bcast_cols_init(cb_in, cb_inv);
    inv.wait_front(CAP);
    tmp.reserve_back(T);
    per_head<Wt, cb_tmp>(
        0, nh, 0, [](uint32_t h, uint32_t j, uint32_t d) { mul_tiles_bcast_cols(cb_in, cb_inv, h * Wt + j, h, d); });
    tmp.push_back(T);
    inv.pop_front(CAP);
    // xn = y * gamma (gamma_q for the Q heads, gamma_k for the K heads)
    constexpr uint32_t cb_xn_q = rotary ? cb_norm : cb_qout;
    constexpr uint32_t cb_xn_k = rotary ? cb_norm : cb_out;
    reconfig_data_format(cb_tmp, cb_gq);
    pack_reconfig_data_format(cb_xn_q);
    mul_init(cb_tmp, cb_gq);
    cb_wait_front(cb_gq, Wt);  // resident, never popped: only the first unit waits
    cb_wait_front(cb_gk, Wt);
    tmp.wait_front(T);
    if constexpr (rotary) {
        nrm.reserve_back(T);
    }
    per_head<Wt, cb_xn_q>(
        0, nq, 0, [](uint32_t h, uint32_t j, uint32_t d) { mul_tiles(cb_tmp, cb_gq, h * Wt + j, j, d); });
    if (nk > 0) {
        if constexpr (!rotary) {
            pack_reconfig_data_format(cb_xn_k);
        }
        per_head<Wt, cb_xn_k>(nq, nh, rotary ? nq * Wt : kv_base, [](uint32_t h, uint32_t j, uint32_t d) {
            mul_tiles(cb_tmp, cb_gk, h * Wt + j, j, d);
        });
    }
    tmp.pop_front(T);
    if constexpr (rotary) {
        nrm.push_back(T);
        // RoPE: out = xn*cos + (xn @ T)*sin
        CircularBuffer rot(cb_rot), si(cb_si), ci(cb_ci);
        nrm.wait_front(T);
        reconfig_data_format(cb_norm, cb_trans);
        pack_reconfig_data_format(cb_rot);
        matmul_init(cb_norm, cb_trans);
        rot.reserve_back(T);
        per_head<Wt, cb_rot>(
            0, nh, 0, [](uint32_t h, uint32_t j, uint32_t d) { matmul_tiles(cb_norm, cb_trans, h * Wt + j, 0, d); });
        rot.push_back(T);
        reconfig_data_format(cb_rot, cb_sin);
        pack_reconfig_data_format(cb_si);
        mul_init(cb_rot, cb_sin);
        rot.wait_front(T);
        si.reserve_back(T);
        per_head<Wt, cb_si>(
            0, nh, 0, [](uint32_t h, uint32_t j, uint32_t d) { mul_tiles(cb_rot, cb_sin, h * Wt + j, j, d); });
        si.push_back(T);
        rot.pop_front(T);
        reconfig_data_format(cb_norm, cb_cos);
        pack_reconfig_data_format(cb_ci);
        mul_init(cb_norm, cb_cos);
        ci.reserve_back(T);
        per_head<Wt, cb_ci>(
            0, nh, 0, [](uint32_t h, uint32_t j, uint32_t d) { mul_tiles(cb_norm, cb_cos, h * Wt + j, j, d); });
        ci.push_back(T);
        nrm.pop_front(T);
        reconfig_data_format(cb_ci, cb_si);
        pack_reconfig_data_format(cb_qout);
        add_init(cb_ci, cb_si);
        ci.wait_front(T);
        si.wait_front(T);
        per_head<Wt, cb_qout>(
            0, nq, 0, [](uint32_t h, uint32_t j, uint32_t d) { add_tiles(cb_ci, cb_si, h * Wt + j, h * Wt + j, d); });
        if (nk > 0) {
            pack_reconfig_data_format(cb_out);
            per_head<Wt, cb_out>(nq, nh, kv_base, [](uint32_t h, uint32_t j, uint32_t d) {
                add_tiles(cb_ci, cb_si, h * Wt + j, h * Wt + j, d);
            });
        }
        ci.pop_front(T);
        si.pop_front(T);
    }
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
    constexpr uint32_t sub_q_tiles = sub_q_heads * Wt;
    constexpr uint32_t group_kv_tiles = heads_per_group * Wt;
    constexpr uint32_t kv_parts = (q_split == 1) ? 2 : 1;
    constexpr uint32_t unit_tiles = sub_q_tiles + kv_parts * group_kv_tiles;
    constexpr uint32_t cb_qout = separate_q ? cb_qsep : cb_out;
    constexpr uint32_t kv_base = separate_q ? 0 : sub_q_tiles;  // K|V offset inside cb_out
    constexpr uint32_t out_tiles = separate_q ? kv_parts * group_kv_tiles : unit_tiles;
    // intermediate CB capacity in heads (the host sizes CBs 5-8 / 12-15 for this many heads)
    constexpr uint32_t cap = sub_q_heads + heads_per_group;

    CircularBuffer in(cb_in), out(cb_out), qout(cb_qout), sc(cb_scaler), ep(cb_eps);
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
        unit_heads<cap, Wt, fuse_rotary != 0, cb_qout>(sub_q_heads, has_k ? heads_per_group : 0, kv_base);
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
