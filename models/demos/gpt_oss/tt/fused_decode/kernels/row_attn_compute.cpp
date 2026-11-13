// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Row attention, compute (TRISC) (dflash/row_attention.py): for one (row, kv head) and its nk 32-position chunks:
//   Qs = Q * scale (scale = 1 / sqrt(64) = 2^-3: exact);
//   S_j = Qs . K_j^T + mask_j (FP32), S_nk = sink tile * scale (the extra "chunk": column 0 = each head's sink logit);
//   m = rowmax over all S; P_j = exp(S_j - m); l = rowsum(P); O = (sum_{j < nk} P_j . V_j) / l.
// Every head row of the 32-row tiles is computed; the writer keeps this kv head's rows.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr uint32_t cb_aux = get_compile_time_arg_val(3);  // scaler, masks, sink tile, zero tile (row_attn_reader)
    constexpr uint32_t cb_qs = get_compile_time_arg_val(4);
    constexpr uint32_t cb_s = get_compile_time_arg_val(5);
    constexpr uint32_t cb_m = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    constexpr uint32_t cb_l = get_compile_time_arg_val(8);
    constexpr uint32_t cb_o = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(11);  // float bits of the softmax scale

    uint32_t info = 0;
    UNPACK(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(info = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    const uint32_t nk = info & 0xFFFF;
    const bool cut_first = (info >> 16) & 1, cut_last = (info >> 17) & 1;
    // row-broadcast mask of chunk j (cb_aux 1 first, 2 last, 3 single; 6 = the zero tile: no cut)
    auto mask_of = [&](uint32_t j) -> uint32_t {
        if (nk == 1) {
            return (cut_first || cut_last) ? 3 : 6;
        }
        return j == 0 ? (cut_first ? 1 : 6) : (j + 1 == nk ? (cut_last ? 2 : 6) : 6);
    };

    compute_kernel_hw_startup(cb_q, cb_q, cb_qs);
    cb_wait_front(cb_q, 2);
    // Qs = Q * scale
    tile_regs_acquire();
    copy_init(cb_q);
    copy_tile(cb_q, 0, 0);
    copy_tile(cb_q, 1, 1);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale_bits);
    mul_unary_tile(1, scale_bits);
    tile_regs_commit();
    cb_reserve_back(cb_qs, 2);
    tile_regs_wait();
    pack_reconfig_data_format(cb_qs);
    pack_tile(0, cb_qs);
    pack_tile(1, cb_qs);
    tile_regs_release();
    cb_push_back(cb_qs, 2);
    cb_pop_front(cb_q, 2);

    // S_j = Qs . K_j^T (matmul unpacks in1 to SrcA, in0 to SrcB) + the chunk's mask (dest-reuse add; only the edge
    // chunks have one), then the sink tile (column 0 = sink logits / scale, x scale) + mask 4 (columns 1..31)
    cb_wait_front(cb_qs, 2);
    cb_wait_front(cb_aux, 7);
    compute_kernel_hw_startup(cb_k, cb_qs, cb_s);
    cb_reserve_back(cb_s, nk + 1);
    for (uint32_t j = 0; j < nk; ++j) {
        cb_wait_front(cb_k, 2 * (j + 1));  // the block's chunks arrive after the older ones
        const uint32_t mk = mask_of(j);
        tile_regs_acquire();
        matmul_init(cb_qs, cb_k, 1);
        matmul_tiles(cb_qs, cb_k, 0, 2 * j, 0);
        matmul_tiles(cb_qs, cb_k, 1, 2 * j + 1, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux, mk, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s);
        pack_tile(0, cb_s);
        tile_regs_release();
    }
    compute_kernel_hw_startup(cb_aux, cb_aux, cb_s);
    tile_regs_acquire();
    copy_init(cb_aux);
    copy_tile(cb_aux, 5, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale_bits);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_aux, 4, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_s);
    tile_regs_release();
    cb_push_back(cb_s, nk + 1);
    cb_pop_front(cb_k, 2 * nk);

    // m = rowmax (a MAX row reduce unpacks the data to SrcA, the scaler to SrcB: only SUM / AVG row reduces swap them)
    cb_wait_front(cb_s, nk + 1);
    compute_kernel_hw_startup(cb_aux, cb_s, cb_m);
    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_aux, cb_m);
    for (uint32_t j = 0; j <= nk; ++j) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_aux, j, 0, 0);
    }
    reduce_uninit(cb_s);
    tile_regs_commit();
    cb_reserve_back(cb_m, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_m);
    pack_tile(0, cb_m);
    tile_regs_release();
    cb_push_back(cb_m, 1);

    // P_j = exp(S_j - m)
    cb_wait_front(cb_m, 1);
    compute_kernel_hw_startup(cb_s, cb_m, cb_p);
    cb_reserve_back(cb_p, nk + 1);
    sub_bcast_cols_init(cb_s, cb_m);
    exp_tile_init();
    for (uint32_t j = 0; j <= nk; ++j) {
        tile_regs_acquire();
        sub_tiles_bcast_cols(cb_s, cb_m, j, 0, 0);
        exp_tile(0, VectorMode::R);  // rows 0..15 (the heads; rows 16..31 are padding)
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_p);
        pack_tile(0, cb_p);
        tile_regs_release();
    }
    cb_push_back(cb_p, nk + 1);
    cb_pop_front(cb_s, nk + 1);
    cb_pop_front(cb_m, 1);

    // l = rowsum(P), 1 / l
    cb_wait_front(cb_p, nk + 1);
    compute_kernel_hw_startup(cb_aux, cb_p, cb_l);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_aux, cb_l);
    for (uint32_t j = 0; j <= nk; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_aux, j, 0, 0);
    }
    reduce_uninit(cb_p);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();
    cb_reserve_back(cb_l, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_l);
    pack_tile(0, cb_l);
    tile_regs_release();
    cb_push_back(cb_l, 1);

    // O = sum_j P_j . V_j
    compute_kernel_hw_startup(cb_v, cb_p, cb_o);
    tile_regs_acquire();
    matmul_init(cb_p, cb_v, 0);
    for (uint32_t j = 0; j < nk; ++j) {
        cb_wait_front(cb_v, 2 * (j + 1));
        matmul_tiles(cb_p, cb_v, j, 2 * j, 0);
        matmul_tiles(cb_p, cb_v, j, 2 * j + 1, 1);
    }
    tile_regs_commit();
    cb_reserve_back(cb_o, 2);
    tile_regs_wait();
    pack_reconfig_data_format(cb_o);
    pack_tile(0, cb_o);
    pack_tile(1, cb_o);
    tile_regs_release();
    cb_push_back(cb_o, 2);
    cb_pop_front(cb_v, 2 * nk);
    cb_pop_front(cb_p, nk + 1);

    // out = O * (1 / l) (per row)
    cb_wait_front(cb_o, 2);
    cb_wait_front(cb_l, 1);
    compute_kernel_hw_startup(cb_o, cb_l, cb_out);
    tile_regs_acquire();
    mul_bcast_cols_init(cb_o, cb_l);
    mul_tiles_bcast_cols(cb_o, cb_l, 0, 0, 0);
    mul_tiles_bcast_cols(cb_o, cb_l, 1, 0, 1);
    tile_regs_commit();
    cb_reserve_back(cb_out, 2);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    pack_tile(1, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 2);
    cb_pop_front(cb_o, 2);
    cb_pop_front(cb_l, 1);
    cb_pop_front(cb_aux, 7);
}
