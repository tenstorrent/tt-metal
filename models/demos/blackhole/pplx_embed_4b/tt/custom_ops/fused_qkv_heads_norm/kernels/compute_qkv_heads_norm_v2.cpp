// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// v2 of the fused head-split + RMSNorm + RoPE compute: same CB contract as
// compute_qkv_heads_norm.cpp, fewer passes over the data. Per head (Wt tiles):
//   1  x2   = x * x                              -> CB 5
//   2  ms   = row-sum(x2) * 1/head_dim           -> CB 6
//   3  inv  = rsqrt(ms + eps)                    -> CB 7
//   4  xn   = (x * bcast_cols(inv)) * gamma      -> CB 15 (or straight to the output)
//   5  si   = (xn @ T) * sin                     -> CB 13
//   6  out  = xn * cos + si                      -> output CB
// Passes 4-6 keep the running tile in DST and apply the second operand with the
// dest-reuse binary ops, so the intermediates y = x*inv, rot = xn@T and ci = xn*cos
// never round-trip through L1 (v1: 9 passes, 3 of them only to re-read a temporary).
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
constexpr uint32_t cb_x2 = 5, cb_red = 6, cb_inv = 7, cb_out = 16;
constexpr uint32_t cb_cos = 9, cb_sin = 10, cb_trans = 11, cb_si = 13, cb_norm = 15;
constexpr uint32_t cb_qsep = 17;
constexpr auto D2B = EltwiseBinaryReuseDestType::DEST_TO_SRCB;  // DST -> SrcB, CB tile -> SrcA

// Passes 1-4. Writes the normalised head to cb_dst (cb_norm at offset 0 when rotary follows).
template <uint32_t Wt, uint32_t cb_dst>
inline void norm_head(uint32_t in_off, uint32_t out_off, uint32_t cb_gamma) {
    CircularBuffer x2(cb_x2), red(cb_red), inv(cb_inv), dst(cb_dst);
    // 1: x^2
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
    // 2: mean of squares
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
    // 3: inv = rsqrt(ms + eps)
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
    // 4: xn = (x * bcast_cols(inv)) * gamma, gamma applied on the DST tile
    reconfig_data_format(cb_in, cb_inv);
    pack_reconfig_data_format(cb_dst);
    mul_bcast_cols_init(cb_in, cb_inv);
    inv.wait_front(1);
    if constexpr (cb_dst == cb_norm) {
        dst.reserve_back(Wt);
    }
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles_bcast_cols(cb_in, cb_inv, in_off + j, 0, j);
    }
    reconfig_data_format_srca(cb_gamma);
    mul_reuse_dest_init<D2B>(cb_gamma);
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_reuse_dest_tiles<D2B>(cb_gamma, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_dst, (cb_dst == cb_norm ? 0 : out_off) + j);
    }
    tile_regs_release();
    inv.pop_front(1);
    if constexpr (cb_dst == cb_norm) {
        dst.push_back(Wt);
    }
}

// Passes 5-6 on the normalised head in cb_norm: out = xn*cos + (xn @ T)*sin.
template <uint32_t Wt, uint32_t cb_dst>
inline void rotary_head(uint32_t out_off) {
    CircularBuffer nrm(cb_norm), si(cb_si);
    nrm.wait_front(Wt);
    // 5: si = (xn @ T) * sin
    reconfig_data_format(cb_norm, cb_trans);
    pack_reconfig_data_format(cb_si);
    matmul_init(cb_norm, cb_trans);
    si.reserve_back(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        matmul_tiles(cb_norm, cb_trans, j, 0, j);
    }
    reconfig_data_format_srca(cb_sin);
    mul_reuse_dest_init<D2B>(cb_sin);
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_reuse_dest_tiles<D2B>(cb_sin, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_si, j);
    }
    tile_regs_release();
    si.push_back(Wt);
    // 6: out = xn * cos + si
    reconfig_data_format(cb_norm, cb_cos);
    pack_reconfig_data_format(cb_dst);
    mul_init(cb_norm, cb_cos);
    si.wait_front(Wt);
    tile_regs_acquire();
    for (uint32_t j = 0; j < Wt; ++j) {
        mul_tiles(cb_norm, cb_cos, j, j, j);
    }
    reconfig_data_format_srca(cb_si);
    add_reuse_dest_init<D2B>(cb_si);
    for (uint32_t j = 0; j < Wt; ++j) {
        add_reuse_dest_tiles<D2B>(cb_si, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < Wt; ++j) {
        pack_tile(j, cb_dst, out_off + j);
    }
    tile_regs_release();
    si.pop_front(Wt);
    nrm.pop_front(Wt);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t fuse_rotary = get_compile_time_arg_val(3);
    constexpr uint32_t separate_q = get_compile_time_arg_val(4);
    constexpr uint32_t cache_rot = get_compile_time_arg_val(5);  // cos/sin kept across units of one seq tile
    constexpr uint32_t head_groups = get_compile_time_arg_val(6);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(7);
    const uint32_t num_work_units = get_arg_val<uint32_t>(0);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(1);

    constexpr uint32_t q_heads_per_group = heads_per_group * q_heads_per_kv;
    constexpr uint32_t group_q_tiles = q_heads_per_group * Wt;
    constexpr uint32_t group_kv_tiles = heads_per_group * Wt;
    constexpr uint32_t unit_tiles = group_q_tiles + 2 * group_kv_tiles;
    constexpr uint32_t cb_qout = separate_q ? cb_qsep : cb_out;
    constexpr uint32_t kv_base = separate_q ? 0 : group_q_tiles;
    constexpr uint32_t out_tiles = separate_q ? 2 * group_kv_tiles : unit_tiles;
    constexpr uint32_t cb_qnorm = fuse_rotary ? cb_norm : cb_qout;
    constexpr uint32_t cb_knorm = fuse_rotary ? cb_norm : cb_out;

    CircularBuffer in(cb_in), out(cb_out), qout(cb_qout), gq(cb_gq), gk(cb_gk), sc(cb_scaler), ep(cb_eps);
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    gq.wait_front(Wt);
    gk.wait_front(Wt);
    sc.wait_front(1);
    ep.wait_front(1);
    if constexpr (fuse_rotary) {
        CircularBuffer ct(cb_trans);
        ct.wait_front(1);
    }

    uint32_t last_s = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t s_tile = ((work_unit_start + w) / head_groups) % seq_tiles;  // mirrors the reader
        in.wait_front(unit_tiles);
        out.reserve_back(out_tiles);
        if constexpr (separate_q) {
            qout.reserve_back(group_q_tiles);
        }
        if constexpr (fuse_rotary) {
            CircularBuffer ccos(cb_cos), csin(cb_sin);
            if (!cache_rot || s_tile != last_s) {
                if (cache_rot && last_s != 0xFFFFFFFFu) {
                    ccos.pop_front(Wt);
                    csin.pop_front(Wt);
                }
                ccos.wait_front(Wt);
                csin.wait_front(Wt);
                last_s = s_tile;
            }
        }
        for (uint32_t h = 0; h < q_heads_per_group; ++h) {
            norm_head<Wt, cb_qnorm>(h * Wt, h * Wt, cb_gq);
            if constexpr (fuse_rotary) {
                rotary_head<Wt, cb_qout>(h * Wt);
            }
        }
        for (uint32_t h = 0; h < heads_per_group; ++h) {
            norm_head<Wt, cb_knorm>(group_q_tiles + h * Wt, kv_base + h * Wt, cb_gk);
            if constexpr (fuse_rotary) {
                rotary_head<Wt, cb_out>(kv_base + h * Wt);
            }
        }
        if constexpr (fuse_rotary) {
            if (!cache_rot) {
                CircularBuffer ccos(cb_cos), csin(cb_sin);
                ccos.pop_front(Wt);
                csin.pop_front(Wt);
            }
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
                pack_tile(j, cb_out, kv_base + group_kv_tiles + t0 + j);
            }
            tile_regs_release();
        }
        if constexpr (separate_q) {
            qout.push_back(group_q_tiles);
        }
        out.push_back(out_tiles);
        in.pop_front(unit_tiles);
    }
    if constexpr (fuse_rotary) {
        if (cache_rot && last_s != 0xFFFFFFFFu) {
            CircularBuffer ccos(cb_cos), csin(cb_sin);
            ccos.pop_front(Wt);
            csin.pop_front(Wt);
        }
    }
}
