// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention worker compute (see attn_common.hpp). Per attention layer and user:
//   chunk worker: the partial attention of q over this core's KV chunk, block by block with an online
//     softmax (a zero partial without one);
//   tail core: k = rope(rmsnorm_rows(k) * w_k), the new k / v row into the tail tiles, the partial over
//     the tail tile (masked), then the tail tiles as bf8 for the DRAM write-back;
// a group head adds its children's partials to its own.
//
// Runtime args: 0 role (1 chunk worker, 2 tail core), 1 children. The chunk's KV tiles come in cb_ntiles.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/binary_max_min.h"
#include "attn_compute_common.hpp"

using namespace resident_attn;

namespace {

FORCE_INLINE void zero_partial() {
    cb_reserve_back(cb_o, kPart);
    pack_reconfig_data_format(cb_o);
    fill_tile_init();
    for (uint32_t d = 0; d < kPart; d++) {
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
    }
    cb_push_back(cb_o, kPart);
}

// DEST tile 0 = row max of the block's nb score tiles
FORCE_INLINE void block_max(uint32_t nb) {
    reconfig_data_format(cb_s, cb_one);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_mrun);
    for (uint32_t j = 0; j < nb; j++) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, j, 0, 0);
    }
    reduce_uninit();
}

// cb_p = exp(s - m) over the block (m: column 0 of m_cb's front tile); consumes cb_s
FORCE_INLINE void block_probs(uint32_t m_cb, uint32_t nb) {
    cb_reserve_back(cb_p, chunk_max);
    reconfig_data_format(cb_s, m_cb);
    pack_reconfig_data_format(cb_p);
    sub_bcast_cols_init(cb_s, m_cb);
    exp_tile_init();
    for (uint32_t j = 0; j < nb; j++) {
        tile_regs_acquire();
        sub_tiles_bcast_cols(cb_s, m_cb, j, 0, 0);
        exp_tile(0, VectorMode::R);  // head rows live in the top faces
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_p, j);
        tile_regs_release();
    }
    cb_push_back(cb_p, chunk_max);
    cb_pop_front(cb_s, chunk_max);
}

// DEST tile 0 = old * alpha (column broadcast); old is tile i of old_cb
FORCE_INLINE void rescaled(uint32_t old_cb, uint32_t i) {
    reconfig_data_format(old_cb, cb_alpha);
    mul_bcast_cols_init(old_cb, cb_alpha);
    mul_tiles_bcast_cols(old_cb, cb_alpha, i, 0, 0);
}

// one block of the online softmax: m = max(m, rowmax s), alpha = exp(m_old - m), p = exp(s - m),
// o = o alpha + p v, l = l alpha + rowsum p (the first block starts from o = l = 0, alpha unused)
FORCE_INLINE void online_block(uint32_t nb, bool first) {
    cb_wait_front(cb_s, chunk_max);
    cb_reserve_back(cb_mrun, 1);
    if (!first) {
        cb_reserve_back(cb_alpha, 1);
    }
    tile_regs_acquire();
    block_max(nb);
    if (!first) {
        reconfig_data_format_srca(cb_mrun);
        copy_init(cb_mrun);
        copy_tile(cb_mrun, 0, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 2);
        sub_binary_tile_init();
        sub_binary_tile(1, 2, 3);
        exp_tile_init();
        exp_tile(3, VectorMode::R);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_mrun);
    pack_tile(first ? 0 : 2, cb_mrun);
    if (!first) {
        pack_reconfig_data_format(cb_alpha);
        pack_tile(3, cb_alpha);
    }
    tile_regs_release();
    cb_push_back(cb_mrun, 1);
    if (!first) {
        cb_push_back(cb_alpha, 1);
        cb_pop_front(cb_mrun, 1);
        cb_wait_front(cb_alpha, 1);
    }
    cb_wait_front(cb_mrun, 1);
    block_probs(cb_mrun, nb);

    cb_wait_front(cb_p, chunk_max);
    cb_reserve_back(cb_orun, Dt);
    pack_reconfig_data_format(cb_orun);
    for (uint32_t d = 0; d < Dt; d++) {
        tile_regs_acquire();
        if (!first) {
            rescaled(cb_orun, d);
        }
        reconfig_data_format(cb_v, cb_p);
        matmul_init(cb_p, cb_v, 0);
        for (uint32_t j = 0; j < nb; j++) {
            matmul_tiles(cb_p, cb_v, j, j * Dt + d, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_orun, d);
        tile_regs_release();
    }
    cb_push_back(cb_orun, Dt);
    cb_reserve_back(cb_lrun, 1);
    pack_reconfig_data_format(cb_lrun);
    tile_regs_acquire();
    if (!first) {
        rescaled(cb_lrun, 0);
    }
    reconfig_data_format(cb_one, cb_p);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, cb_lrun);
    for (uint32_t j = 0; j < nb; j++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_one, j, 0, 0);
    }
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_lrun);
    tile_regs_release();
    cb_push_back(cb_lrun, 1);
    if (!first) {
        cb_pop_front(cb_orun, Dt);
        cb_pop_front(cb_lrun, 1);
        cb_pop_front(cb_alpha, 1);
    }
    cb_pop_front(cb_p, chunk_max);
}

// sends m (cb_m), then with the global row max M: partial (o, l) * exp(m - M)
FORCE_INLINE void online_finish() {
    cb_reserve_back(cb_m, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_mrun);
    copy_init(cb_mrun);
    copy_tile(cb_mrun, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_m);
    pack_tile(0, cb_m);
    tile_regs_release();
    cb_push_back(cb_m, 1);

    cb_wait_front(cb_M, 1);
    cb_reserve_back(cb_alpha, 1);
    tile_regs_acquire();
    copy_init(cb_mrun);
    copy_tile(cb_mrun, 0, 0);
    reconfig_data_format_srca(cb_M);
    copy_init(cb_M);
    copy_tile(cb_M, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    exp_tile_init();
    exp_tile(0, VectorMode::R);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_alpha);
    pack_tile(0, cb_alpha);
    tile_regs_release();
    cb_push_back(cb_alpha, 1);
    cb_wait_front(cb_alpha, 1);

    cb_reserve_back(cb_o, kPart);
    pack_reconfig_data_format(cb_o);
    for (uint32_t d = 0; d < kPart; d++) {
        tile_regs_acquire();
        rescaled(d < Dt ? cb_orun : cb_lrun, d < Dt ? d : 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_o, d);
        tile_regs_release();
    }
    cb_push_back(cb_o, kPart);
    cb_pop_front(cb_orun, Dt);
    cb_pop_front(cb_lrun, 1);
    cb_pop_front(cb_mrun, 1);
    cb_pop_front(cb_alpha, 1);
    cb_pop_front(cb_M, 1);
}
}  // namespace

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t children = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_one, 1);
    cb_wait_front(cb_ntiles, 1);
    uint32_t n_of[batch];
    for (uint32_t u = 0; u < batch; u++) {
        n_of[u] = read_tile_value(cb_ntiles, 0, u);
    }
    if (role == 1) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
        for (uint32_t i = 0; i < iters; i++) {
            const uint32_t n_tiles = n_of[i % batch];
            cb_wait_front(cb_q, Dt);
            if (n_tiles > 0) {
                for (uint32_t b0 = 0; b0 < n_tiles; b0 += chunk_max) {
                    const uint32_t nb = n_tiles - b0 < chunk_max ? n_tiles - b0 : chunk_max;
                    cb_wait_front(cb_k, chunk_max * Dt);
                    cb_wait_front(cb_v, chunk_max * Dt);
                    scores(cb_q, cb_k, 0, nb, false);
                    online_block(nb, b0 == 0);
                    cb_pop_front(cb_k, chunk_max * Dt);
                    cb_pop_front(cb_v, chunk_max * Dt);
                }
                online_finish();
            } else {
                cb_wait_front(cb_M, 1);
                zero_partial();
                cb_pop_front(cb_M, 1);
            }
            if (children > 0) {
                sum_partials<true>(children);
            }
            cb_pop_front(cb_q, Dt);
        }
        return;
    }
    compute_kernel_hw_startup(cb_kraw, cb_kraw, cb_sq);
    cb_wait_front(cb_mean, 1);
    cb_wait_front(cb_rope, 3 * batch);
    cb_wait_front(cb_mask, batch);
    cb_wait_front(cb_rowsel, 2 * batch);
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t u = i % batch;
        cb_wait_front(cb_kw, Dt);
        norm_rope(cb_kraw, cb_kw, 0, cb_knew, cb_knew, 3 * u);
        cb_pop_front(cb_kw, Dt);
        update_tail(2 * u);
        cb_wait_front(cb_q, Dt);
        cb_wait_front(cb_tailk, Dt);
        cb_wait_front(cb_tailv, Dt);
        scores(cb_q, cb_tailk, 0, 1, true, u);
        row_max(1);
        probs_and_partial(cb_tailv, 0, 1);
        if (children > 0) {
            sum_partials<true>(children);
        }
        cb_pop_front(cb_q, Dt);
        pack_tail_bf8();
    }
}
