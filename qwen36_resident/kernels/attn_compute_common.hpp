// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention chunk math shared by the leader and worker compute kernels (32x32 tiles, head h in row h):
//   scores   s[j] = sum_d q[d] k[j Dt + d]^T                  (cb_s, fp32)
//   row max  m = max_j rowmax(s[j])                          (cb_m)
//   probs    p[j] = exp(s[j] - M)  (M: the global row max)    (cb_p)
//   partial  o[d] = sum_j p[j] v[j Dt + d],  l = sum_j rowsum(p[j])   (cb_o: Dt tiles, then l)
// and the q / k preparation (row rmsnorm, weight, rope) and the tail-tile update.
#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "attn_common.hpp"

namespace resident_attn {

// s[j] for j < n_tiles; with mask, tile mask_tile of cb_mask is added to the last tile (tail columns
// past the position)
FORCE_INLINE void scores(
    uint32_t q_cb, uint32_t k_cb, uint32_t k_base, uint32_t n_tiles, bool mask, uint32_t mask_tile = 0) {
    cb_reserve_back(cb_s, chunk_max);
    reconfig_data_format(k_cb, q_cb);
    pack_reconfig_data_format(cb_s);
    matmul_init(q_cb, k_cb, 1 /* transpose k */);
    for (uint32_t j = 0; j < n_tiles; j++) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; d++) {
            matmul_tiles(q_cb, k_cb, d, k_base + j * Dt + d, 0);
        }
        if (mask && j + 1 == n_tiles) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask, mask_tile, 0);
            matmul_init(q_cb, k_cb, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_s, j);
        tile_regs_release();
    }
    cb_push_back(cb_s, chunk_max);
}

FORCE_INLINE void row_max(uint32_t n_tiles) {
    cb_wait_front(cb_s, chunk_max);
    cb_reserve_back(cb_m, 1);
    reconfig_data_format(cb_s, cb_one);
    pack_reconfig_data_format(cb_m);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_m);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n_tiles; j++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_m);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(cb_m, 1);
}

// p = exp(s - M), o = p v, l = rowsum(p); consumes cb_s and cb_p
FORCE_INLINE void probs_and_partial(uint32_t v_cb, uint32_t v_base, uint32_t n_tiles) {
    cb_wait_front(cb_M, 1);
    cb_reserve_back(cb_p, chunk_max);
    reconfig_data_format(cb_s, cb_M);
    pack_reconfig_data_format(cb_p);
    sub_bcast_cols_init(cb_s, cb_M);
    exp_tile_init();
    for (uint32_t j = 0; j < n_tiles; j++) {
        tile_regs_acquire();
        sub_tiles_bcast_cols(cb_s, cb_M, j, 0, 0);
        exp_tile(0, VectorMode::R);  // head rows live in the top faces
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_p, j);
        tile_regs_release();
    }
    cb_push_back(cb_p, chunk_max);
    cb_pop_front(cb_s, chunk_max);
    cb_pop_front(cb_M, 1);

    cb_wait_front(cb_p, chunk_max);
    cb_reserve_back(cb_o, kPart);
    reconfig_data_format(v_cb, cb_p);
    pack_reconfig_data_format(cb_o);
    matmul_init(cb_p, v_cb, 0);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < n_tiles; j++) {
            matmul_tiles(cb_p, v_cb, j, v_base + j * Dt + d, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_o, d);
        tile_regs_release();
    }
    reconfig_data_format(cb_one, cb_p);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, cb_o);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n_tiles; j++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_o, Dt);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(cb_o, kPart);
    cb_pop_front(cb_p, chunk_max);
}

// cb_osum = own partial (cb_o; the leader has none) + the first `children` partials of cb_parts
template <bool own>
FORCE_INLINE void sum_partials(uint32_t children) {
    if constexpr (own) {
        cb_wait_front(cb_o, kPart);
    }
    cb_wait_front(cb_parts, fanin * kPart);
    cb_reserve_back(cb_osum, kPart);
    reconfig_data_format(cb_parts, cb_parts);
    pack_reconfig_data_format(cb_osum);
    for (uint32_t d = 0; d < kPart; d++) {
        tile_regs_acquire();
        copy_init(own ? cb_o : cb_parts);
        copy_tile(own ? cb_o : cb_parts, d, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts);
        for (uint32_t c = own ? 0 : 1; c < children; c++) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts, c * kPart + d, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_osum);
        tile_regs_release();
    }
    cb_push_back(cb_osum, kPart);
    cb_pop_front(cb_parts, fanin * kPart);
    if constexpr (own) {
        cb_pop_front(cb_o, kPart);
    }
}

// cb_rs = 1 / sqrt(mean over the row of in^2 + eps)
FORCE_INLINE void row_rsqrt(uint32_t in_cb) {
    cb_reserve_back(cb_sq, Dt);
    reconfig_data_format(in_cb, in_cb);
    pack_reconfig_data_format(cb_sq);
    mul_init(in_cb, in_cb, false);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        mul_tiles(in_cb, in_cb, d, d, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
    }
    cb_push_back(cb_sq, Dt);

    cb_wait_front(cb_sq, Dt);
    cb_reserve_back(cb_rs, 1);
    reconfig_data_format(cb_mean, cb_sq);
    pack_reconfig_data_format(cb_rs);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_mean, cb_rs);
    tile_regs_acquire();
    for (uint32_t d = 0; d < Dt; d++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_mean, d, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rs);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_sq, Dt);

    cb_wait_front(cb_rs, 1);
    cb_reserve_back(cb_rs, 1);
    reconfig_data_format(cb_rs, cb_rs);
    copy_init(cb_rs);
    tile_regs_acquire();
    copy_tile(cb_rs, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rs);
    tile_regs_release();
    cb_pop_front(cb_rs, 1);
    cb_push_back(cb_rs, 1);
}

// out = rope(in * rs * w[w_base ..]) into out_a (and out_b if given), rope tables at tile rope of cb_rope;
// pops in_cb
FORCE_INLINE void norm_rope(
    uint32_t in_cb, uint32_t w_cb, uint32_t w_base, uint32_t out_a, uint32_t out_b, uint32_t rope = 0) {
    cb_wait_front(in_cb, Dt);
    row_rsqrt(in_cb);
    cb_wait_front(cb_rs, 1);
    cb_reserve_back(cb_sq, Dt);
    reconfig_data_format(in_cb, cb_rs);
    pack_reconfig_data_format(cb_sq);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        mul_bcast_cols_init(in_cb, cb_rs);
        mul_tiles_bcast_cols(in_cb, cb_rs, d, 0, 0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(w_cb);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(w_cb, w_base + d, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
    }
    cb_push_back(cb_sq, Dt);
    cb_pop_front(cb_rs, 1);
    cb_pop_front(in_cb, Dt);

    // rotate_half on the first 64 dims: t0' = t0 cos - t1 sin, t1' = t1 cos + t0 sin
    cb_wait_front(cb_sq, Dt);
    cb_reserve_back(out_a, Dt);
    if (out_b != out_a) {
        cb_reserve_back(out_b, Dt);
    }
    reconfig_data_format(cb_sq, cb_sq);
    pack_reconfig_data_format(out_a);
    auto pack_both = [&](uint32_t dst) {
        pack_tile(dst, out_a);
        if (out_b != out_a) {
            pack_tile(dst, out_b);
        }
    };
    tile_regs_acquire();
    copy_init(cb_sq);
    copy_tile(cb_sq, 0, 0);
    copy_tile(cb_sq, 1, 1);
    copy_tile(cb_rope, rope, 2);
    copy_tile(cb_rope, rope + 1, 3);
    copy_tile(cb_rope, rope + 2, 4);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 5);
    mul_binary_tile(1, 4, 6);
    add_binary_tile_init();
    add_binary_tile(5, 6, 5);
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 6);
    mul_binary_tile(0, 3, 7);
    add_binary_tile_init();
    add_binary_tile(6, 7, 6);
    tile_regs_commit();
    tile_regs_wait();
    pack_both(5);
    pack_both(6);
    tile_regs_release();
    copy_init(cb_sq);
    for (uint32_t d = rot_tiles; d < Dt; d++) {
        tile_regs_acquire();
        copy_tile(cb_sq, d, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_both(0);
        tile_regs_release();
    }
    cb_push_back(out_a, Dt);
    if (out_b != out_a) {
        cb_push_back(out_b, Dt);
    }
    cb_pop_front(cb_sq, Dt);
}

// row r of the tail tiles <- the new k / v row (row 0 of cb_knew / cb_vraw), kBatch tiles per DST pass;
// R, 1 - R at tiles sel, sel + 1 of cb_rowsel
constexpr uint32_t kBatch = 4;
static_assert(Dt % kBatch == 0, "tail tiles go in DST batches");

FORCE_INLINE void update_tail(uint32_t sel = 0) {
    cb_wait_front(cb_tail8, 2 * Dt);
    cb_wait_front(cb_knew, Dt);
    cb_wait_front(cb_vraw, Dt);
    cb_reserve_back(cb_tailk, Dt);
    cb_reserve_back(cb_tailv, Dt);
    pack_reconfig_data_format(cb_tailk);
    for (uint32_t i0 = 0; i0 < 2 * Dt; i0 += kBatch) {
        const uint32_t row_cb = i0 < Dt ? cb_knew : cb_vraw;
        tile_regs_acquire();
        reconfig_data_format(cb_tail8, cb_rowsel);
        mul_init(cb_tail8, cb_rowsel, false);
        for (uint32_t i = 0; i < kBatch; i++) {
            mul_tiles(cb_tail8, cb_rowsel, i0 + i, sel + 1, i);
        }
        reconfig_data_format(cb_rowsel, row_cb);
        mul_bcast_rows_init(cb_rowsel, row_cb);
        for (uint32_t i = 0; i < kBatch; i++) {
            mul_tiles_bcast_rows(cb_rowsel, row_cb, sel, (i0 + i) % Dt, kBatch + i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < kBatch; i++) {
            add_binary_tile(i, kBatch + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < kBatch; i++) {
            pack_tile(i, i0 < Dt ? cb_tailk : cb_tailv);
        }
        tile_regs_release();
    }
    cb_push_back(cb_tailk, Dt);
    cb_push_back(cb_tailv, Dt);
    cb_pop_front(cb_tail8, 2 * Dt);
    cb_pop_front(cb_knew, Dt);
    cb_pop_front(cb_vraw, Dt);
}

// the updated tail tiles as bf8 for the DRAM write-back; pops them
FORCE_INLINE void pack_tail_bf8() {
    cb_reserve_back(cb_flush, 2 * Dt);
    reconfig_data_format(cb_tailk, cb_tailk);
    pack_reconfig_data_format(cb_flush);
    copy_init(cb_tailk);
    for (uint32_t i = 0; i < 2 * Dt; i++) {
        tile_regs_acquire();
        copy_tile(i < Dt ? cb_tailk : cb_tailv, i % Dt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_flush);
        tile_regs_release();
    }
    cb_push_back(cb_flush, 2 * Dt);
    cb_pop_front(cb_tailk, Dt);
    cb_pop_front(cb_tailv, Dt);
}

}  // namespace resident_attn
