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

// the SFPU exp over a pass's rows: the top faces, or the whole tile when the rows reach the bottom faces
constexpr auto kExpMode = top_faces ? VectorMode::R : VectorMode::RC;

// pages of a scores block in cb_s / cb_p: chunk_max on a chunk worker; the tail core (its own binary, ATTN_TAIL_CORE)
// scores one tail tile (packed: up to two), so its cb_s holds blocks of that size (model.py: O-2 holds two)
#ifdef ATTN_TAIL_CORE
constexpr uint32_t kSBlk = packed ? 2 : 1;
#else
constexpr uint32_t kSBlk = chunk_max;
#endif

// s[j] for j < n_tiles; with mask, tile mask_tile of cb_mask is added to the last tile (tail columns
// past the position; packed: tile mask_tile + j to every tile, the tail tiles under per-row masks)
FORCE_INLINE void scores(
    uint32_t q_cb, uint32_t k_cb, uint32_t k_base, uint32_t n_tiles, bool mask, uint32_t mask_tile = 0) {
    cb_reserve_back(cb_s, kSBlk);
    reconfig_data_format(k_cb, q_cb);
    pack_reconfig_data_format(cb_s);
    matmul_init(q_cb, k_cb, 1 /* transpose k */);
    for (uint32_t j = 0; j < n_tiles; j++) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < Dt; d++) {
            matmul_tiles(q_cb, k_cb, d, k_base + j * Dt + d, 0);
        }
        if (mask && (packed || j + 1 == n_tiles)) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_mask, mask_tile + (packed ? j : 0), 0);
            matmul_init(q_cb, k_cb, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_s, j);
        tile_regs_release();
    }
    cb_push_back(cb_s, kSBlk);
}

// (s_base: the block of cb_s to take — O-2's tail scores user i + 1 while user i's block still waits for M(i))
FORCE_INLINE void row_max(uint32_t n_tiles, uint32_t s_base = 0) {
    cb_wait_front(cb_s, s_base + kSBlk);
    cb_reserve_back(cb_m, 1);
    reconfig_data_format(cb_s, cb_one);
    pack_reconfig_data_format(cb_m);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_m);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n_tiles; j++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, s_base + j, 0, 0);
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
    cb_reserve_back(cb_p, kSBlk);
    reconfig_data_format(cb_s, cb_M);
    pack_reconfig_data_format(cb_p);
    sub_bcast_cols_init(cb_s, cb_M);
    exp_tile_init();
    for (uint32_t j = 0; j < n_tiles; j++) {
        tile_regs_acquire();
        sub_tiles_bcast_cols(cb_s, cb_M, j, 0, 0);
        exp_tile(0, kExpMode);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_p, j);
        tile_regs_release();
    }
    cb_push_back(cb_p, kSBlk);
    cb_pop_front(cb_s, kSBlk);
    cb_pop_front(cb_M, 1);

    cb_wait_front(cb_p, kSBlk);
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
    cb_pop_front(cb_p, kSBlk);
}

// cb_osum = own partial (cb_o; the leader has none) + the first `children` partials of cb_parts, those with their bit
// in `mask` (kSkipZeroParts: the children with KV tiles for the user)
template <bool own>
FORCE_INLINE void sum_partials(uint32_t children, uint32_t mask = ~0u) {
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
            if ((mask >> c) & 1) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts, c * kPart + d, 0);
            }
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
    // (S0-8 kKRot: into cb_rotx, then rotated by HAD into out_a / out_b below)
    const uint32_t rope_a = kKRot ? cb_rotx : out_a, rope_b = kKRot ? cb_rotx : out_b;
    cb_wait_front(cb_sq, Dt);
    cb_reserve_back(rope_a, Dt);
    if (rope_b != rope_a) {
        cb_reserve_back(rope_b, Dt);
    }
    reconfig_data_format(cb_sq, cb_sq);
    pack_reconfig_data_format(rope_a);
    auto pack_both = [&](uint32_t dst) {
        pack_tile(dst, rope_a);
        if (rope_b != rope_a) {
            pack_tile(dst, rope_b);
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
    cb_push_back(rope_a, Dt);
    if (rope_b != rope_a) {
        cb_push_back(rope_b, Dt);
    }
    cb_pop_front(cb_sq, Dt);
    if constexpr (kKRot) {
        // HAD = (H8 x H32) / 16 (tile index x column in tile): y_i = x_i @ (H32 / 16) (cb_had, one tile), then H8's +-1
        // across the Dt tiles as three butterfly stages (stride s: tile j <- y_j + y_j+s, tile j + s <- y_j - y_j+s),
        // ping-ponging cb_rotx -> cb_rot2 -> cb_rotx -> cb_rot2 -> out_a / out_b (8 tile matmuls instead of 64)
        cb_wait_front(cb_rotx, Dt);
        cb_reserve_back(cb_rot2, Dt);
        reconfig_data_format(cb_had, cb_rotx);
        pack_reconfig_data_format(cb_rot2);
        matmul_init(cb_rotx, cb_had, 0);
        for (uint32_t i = 0; i < Dt; i++) {
            tile_regs_acquire();
            matmul_tiles(cb_rotx, cb_had, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_rot2);
            tile_regs_release();
        }
        cb_push_back(cb_rot2, Dt);
        cb_pop_front(cb_rotx, Dt);
        uint32_t src = cb_rot2, dst = cb_rotx;
        for (uint32_t s = 1; s < Dt; s <<= 1) {
            const bool last = (s << 1) >= Dt;
            const uint32_t o_a = last ? out_a : dst, o_b = last ? out_b : dst;
            cb_wait_front(src, Dt);
            cb_reserve_back(o_a, Dt);
            if (o_b != o_a) {
                cb_reserve_back(o_b, Dt);
            }
            reconfig_data_format(src, src);
            pack_reconfig_data_format(o_a);
            for (uint32_t pass = 0; pass < 2; pass++) {  // (the sums, then the differences)
                if (pass == 0) {
                    add_init(src, src);
                } else {
                    sub_init(src, src);
                }
                for (uint32_t j = 0; j < Dt; j++) {
                    if (((j & s) != 0) != (pass == 1)) {
                        continue;
                    }
                    const uint32_t lo = pass == 0 ? j : j - s;
                    tile_regs_acquire();
                    if (pass == 0) {
                        add_tiles(src, src, lo, lo + s, 0);
                    } else {
                        sub_tiles(src, src, lo, lo + s, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, o_a, j);
                    if (o_b != o_a) {
                        pack_tile<true>(0, o_b, j);
                    }
                    tile_regs_release();
                }
            }
            cb_push_back(o_a, Dt);
            if (o_b != o_a) {
                cb_push_back(o_b, Dt);
            }
            cb_pop_front(src, Dt);
            const uint32_t t = src;
            src = dst;
            dst = t;
        }
        reconfig_data_format(cb_sq, cb_sq);  // (the unpack formats norm_rope leaves without the rotation)
    }
}

// row r of the tail tiles <- the new k / v row (row 0 of cb_knew / cb_vraw), kBatch tiles per DST pass;
// R, 1 - R at tiles sel, sel + 1 of cb_rowsel
constexpr uint32_t kBatch = 4;
static_assert(Dt % kBatch == 0, "tail tiles go in DST batches");

// (the tail's k tiles come in cb_tail8 (bf8), its v tiles in cb_tail8v (V's format: bf8, or bfp4 with S0): each
// unpack is reconfigured to its CB's format, and the pack to cb_tailk / cb_tailv's)
FORCE_INLINE void update_tail(uint32_t sel = 0) {
    cb_wait_front(cb_tail8, Dt);
    cb_wait_front(cb_tail8v, Dt);
    cb_wait_front(cb_knew, Dt);
    cb_wait_front(cb_vraw, Dt);
    cb_reserve_back(cb_tailk, Dt);
    cb_reserve_back(cb_tailv, Dt);
    for (uint32_t i0 = 0; i0 < 2 * Dt; i0 += kBatch) {
        const uint32_t row_cb = i0 < Dt ? cb_knew : cb_vraw;
        const uint32_t src = i0 < Dt ? cb_tail8 : cb_tail8v;
        if (i0 % Dt == 0) {
            pack_reconfig_data_format(i0 < Dt ? cb_tailk : cb_tailv);
        }
        tile_regs_acquire();
        reconfig_data_format(src, cb_rowsel);
        mul_init(src, cb_rowsel, false);
        for (uint32_t i = 0; i < kBatch; i++) {
            mul_tiles(src, cb_rowsel, (i0 + i) % Dt, sel + 1, i);
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
    cb_pop_front(cb_tail8, Dt);
    cb_pop_front(cb_tail8v, Dt);
    cb_pop_front(cb_knew, Dt);
    cb_pop_front(cb_vraw, Dt);
}

// packed: the n_tail tail tiles (cb_tail8: k of each tile, then v of each) with every row's new k / v row (rows
// 0 .. batch - 1 of cb_knew / cb_vraw) at its position: tail (x) keep + P @ new, per tile the keep mask and the
// placement matrix P at tiles 2 j + 1, 2 j of cb_rowsel (exact: 0/1 products)
// (verify rows: the user's placement / keep tiles from sel0 of cb_rowsel)
FORCE_INLINE void update_tail_rows(uint32_t n_tail, uint32_t sel0 = 0) {
    const uint32_t tp = tail_pages(n_tail);
    cb_wait_front(cb_tail8, tp * Dt);
    cb_wait_front(cb_tail8v, tp * Dt);
    cb_wait_front(cb_knew, Dt);
    cb_wait_front(cb_vraw, Dt);
    cb_reserve_back(cb_tailk, tp * Dt);
    cb_reserve_back(cb_tailv, tp * Dt);
    for (uint32_t kv = 0; kv < 2; kv++) {
        const uint32_t row_cb = kv == 0 ? cb_knew : cb_vraw;
        const uint32_t src = kv == 0 ? cb_tail8 : cb_tail8v;
        pack_reconfig_data_format(kv == 0 ? cb_tailk : cb_tailv);
        for (uint32_t j = 0; j < n_tail; j++) {
            for (uint32_t d0 = 0; d0 < Dt; d0 += kBatch) {
                tile_regs_acquire();
                reconfig_data_format(src, cb_rowsel);
                mul_init(src, cb_rowsel, false);
                for (uint32_t i = 0; i < kBatch; i++) {
                    mul_tiles(src, cb_rowsel, j * Dt + d0 + i, sel0 + 2 * j + 1, i);
                }
                reconfig_data_format(row_cb, cb_rowsel);
                matmul_init(cb_rowsel, row_cb, 0);
                for (uint32_t i = 0; i < kBatch; i++) {
                    matmul_tiles(cb_rowsel, row_cb, sel0 + 2 * j, d0 + i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < kBatch; i++) {
                    pack_tile(i, kv == 0 ? cb_tailk : cb_tailv);
                }
                tile_regs_release();
            }
        }
    }
    cb_push_back(cb_tailk, tp * Dt);
    cb_push_back(cb_tailv, tp * Dt);
    cb_pop_front(cb_tail8, tp * Dt);
    cb_pop_front(cb_tail8v, tp * Dt);
    cb_pop_front(cb_knew, Dt);
    cb_pop_front(cb_vraw, Dt);
}

// kTailBatched: the user's tail tile (cb_tail8, k then v) with its new row placed from the layer's batched new rows
// (cb_knew / cb_vraw, user u in row u; waited for by the caller, kept for the layer's other users): tail (x) keep +
// P_u @ new, P_u / keep at tiles sel / sel + 1 of cb_rowsel (exact: 0/1 products)
FORCE_INLINE void update_tail_user(uint32_t sel) {
    cb_wait_front(cb_tail8, Dt);
    cb_wait_front(cb_tail8v, Dt);
    cb_reserve_back(cb_tailk, Dt);
    cb_reserve_back(cb_tailv, Dt);
    for (uint32_t kv = 0; kv < 2; kv++) {
        const uint32_t row_cb = kv == 0 ? cb_knew : cb_vraw;
        const uint32_t src = kv == 0 ? cb_tail8 : cb_tail8v;
        pack_reconfig_data_format(kv == 0 ? cb_tailk : cb_tailv);
        for (uint32_t d0 = 0; d0 < Dt; d0 += kBatch) {
            tile_regs_acquire();
            reconfig_data_format(src, cb_rowsel);
            mul_init(src, cb_rowsel, false);
            for (uint32_t i = 0; i < kBatch; i++) {
                mul_tiles(src, cb_rowsel, d0 + i, sel + 1, i);
            }
            reconfig_data_format(row_cb, cb_rowsel);
            matmul_init(cb_rowsel, row_cb, 0);
            for (uint32_t i = 0; i < kBatch; i++) {
                matmul_tiles(cb_rowsel, row_cb, sel, d0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < kBatch; i++) {
                pack_tile(i, kv == 0 ? cb_tailk : cb_tailv);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_tailk, Dt);
    cb_push_back(cb_tailv, Dt);
    cb_pop_front(cb_tail8, Dt);
    cb_pop_front(cb_tail8v, Dt);
}

// the updated tail tiles (n_tail of them) for the DRAM write-back: k as bf8 into cb_flush, v in V's format (bf8, or
// bfp4 with S0) into cb_flushv; pops them
FORCE_INLINE void pack_tail_bf8(uint32_t n_tail = 1) {
    const uint32_t tp = tail_pages(n_tail);  // (verify rows: a fixed 2 tiles' pages)
    cb_reserve_back(cb_flush, tp * Dt);
    cb_reserve_back(cb_flushv, tp * Dt);
    for (uint32_t kv = 0; kv < 2; kv++) {
        const uint32_t src = kv == 0 ? cb_tailk : cb_tailv, dst = kv == 0 ? cb_flush : cb_flushv;
        reconfig_data_format(src, src);
        pack_reconfig_data_format(dst);
        copy_init(src);
        for (uint32_t i = 0; i < n_tail * Dt; i++) {
            tile_regs_acquire();
            copy_tile(src, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst);
            tile_regs_release();
        }
    }
    cb_push_back(cb_flush, tp * Dt);
    cb_push_back(cb_flushv, tp * Dt);
    cb_pop_front(cb_tailk, tp * Dt);
    cb_pop_front(cb_tailv, tp * Dt);
}

}  // namespace resident_attn
