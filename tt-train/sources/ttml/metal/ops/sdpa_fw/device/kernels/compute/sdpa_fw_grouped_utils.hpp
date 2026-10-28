// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sdpa_compute_utils.hpp"

// Helpers for the GQA-grouped forward path (GROUPED_HEADS).
//
// In grouped mode one work item is (batch, kv_head, Q tile-row) and the compute kernel runs the
// online softmax for all `heads_per_item` Q heads of that K/V group against the SAME streamed
// K/V chunk. Each head needs its own running state (max, exp-sum, un-normalized O). The state
// lives in three "ring" CBs that are consumed from the front and refilled at the back in the
// same cyclic head order every chunk, so the front of each ring is always the state of the head
// currently being processed:
//
//   cb_ring_max : heads_per_item + 1 tiles (bf16)         running row max per head
//   cb_ring_sum : heads_per_item + 1 tiles (fp32)         running exp-sum per head
//   cb_ring_mm  : (heads_per_item + 1) * Wt tiles (fp32)  running O per head
//
// The "+1" slot lets a head push its new state before its old state has been popped.
//
// Unlike the single-row path, the O correction is fused into the PV matmul:
//   O_new = exp(scale*(m_prev - m_new)) ⊙ O_prev + P·V
// is produced inside one DST acquisition (SFPU rescale, then FPU matmul accumulate), so no
// separate cur_mm_out CB / L1-accumulate pass is needed.

// Chunk row-max for one head. Reduces over Sk_chunk_t score tiles; when `do_eltwise_max` the
// head's running max (front of cb_ring_max) is folded in. The result is packed to BOTH
// cb_cur_max (consumed by exp / exp_max_diff this chunk) and the back of cb_ring_max (the head's
// running max for the next chunk). The old running max stays at the ring front until the caller
// has computed exp(scale*(prev - cur)) and pops it.
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t Sk_chunk_t>
void grouped_update_row_max(
    uint32_t cb_attention_weights,
    uint32_t cb_identity_scaler,
    uint32_t cb_cur_max,
    uint32_t cb_ring_max,
    bool do_eltwise_max) {
    cb_wait_front(cb_attention_weights, Sk_chunk_t);

    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1U;
    reconfig_data_format(cb_attention_weights, cb_identity_scaler);
    reduce_init<pool_type, reduce_dim>(cb_attention_weights, cb_identity_scaler, cb_cur_max);
    tile_regs_acquire();
    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        reduce_tile<pool_type, reduce_dim>(
            cb_attention_weights,
            cb_identity_scaler,
            /* in0 tile_idx */ n,
            /* in1 tile_idx */ 0,
            /* dst_reg_idx */ reduce_dst_idx);
    }
    reduce_uninit();

    if (do_eltwise_max) {
        cb_wait_front(cb_ring_max, onetile);
        reconfig_data_format_srca(cb_attention_weights, cb_ring_max);
        copy_init(cb_ring_max);
        copy_tile(cb_ring_max, /* tile_idx */ 0, /* register idx */ prev_max_dst_idx);

        binary_max_tile_init();
        binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, VectorMode::C);
    }
    tile_regs_commit();

    cb_reserve_back(cb_cur_max, onetile);
    cb_reserve_back(cb_ring_max, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max);
    pack_tile(reduce_dst_idx, cb_cur_max);
    // cb_ring_max has the same data format as cb_cur_max; no pack reconfig needed.
    pack_tile(reduce_dst_idx, cb_ring_max);
    tile_regs_release();
    cb_push_back(cb_cur_max, onetile);
    cb_push_back(cb_ring_max, onetile);
}

// new_sum = prev_sum (front of cb_ring_sum) * exp_max_diff + cur_sum, pushed to the back of
// cb_ring_sum. Same arithmetic as update_cur_exp_sum_inplace, different destination.
void grouped_update_exp_sum(uint32_t cb_ring_sum, uint32_t cb_cur_sum_exp, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_ring_sum, onetile);
    cb_wait_front(cb_cur_sum_exp, onetile);
    cb_wait_front(cb_exp_max_diff, onetile);

    constexpr uint32_t exp_sum_dst_idx = 0;
    reconfig_data_format(cb_ring_sum, cb_exp_max_diff);
    mul_bcast_cols_init(cb_ring_sum, cb_exp_max_diff);
    tile_regs_acquire();
    mul_tiles_bcast_cols(cb_ring_sum, cb_exp_max_diff, 0, 0, exp_sum_dst_idx);

    copy_init(cb_cur_sum_exp);
    copy_tile(cb_cur_sum_exp, /* tile_idx */ 0, /* register idx */ exp_sum_dst_idx + 1U);

    add_binary_tile_init();
    add_binary_tile(exp_sum_dst_idx, exp_sum_dst_idx + 1U, exp_sum_dst_idx);
    tile_regs_commit();

    cb_pop_front(cb_ring_sum, onetile);
    cb_pop_front(cb_cur_sum_exp, onetile);
    cb_reserve_back(cb_ring_sum, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ring_sum);
    pack_tile(exp_sum_dst_idx, cb_ring_sum);
    tile_regs_release();
    cb_push_back(cb_ring_sum, onetile);
}

// O_new = exp_max_diff ⊙ O_prev + P·V for one head, where O_prev is the Wt-tile block at the
// front of cb_ring_mm and O_new is pushed to the back of cb_ring_mm.
//
// Per block of `block_size` feat tiles (block_size + 1 <= dst_size, one scratch for the bcast):
//   1. copy O_prev tiles into DST via UnpackToDestFp32 (full FP32),
//   2. broadcast exp_max_diff (column vector) into DST[block_size] and SFPU-multiply,
//   3. matmul_block accumulates P·V on top of the rescaled tiles (FPU accumulates into DST),
//   4. pack the block to cb_ring_mm.
template <uint32_t Sk_chunk_t>
void grouped_update_mm_out_fused(
    uint32_t Wt,
    uint32_t block_size,
    uint32_t cb_ring_mm,
    uint32_t cb_attention_weights,
    uint32_t cb_value,
    uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_ring_mm, Wt);
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
    cb_wait_front(cb_value, Sk_chunk_t * Wt);
    cb_wait_front(cb_exp_max_diff, onetile);
    cb_reserve_back(cb_ring_mm, Wt);

    pack_reconfig_data_format(cb_ring_mm);
    pack_reconfig_l1_acc(false);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();

        // O_prev block -> DST[0..block_size-1] (UnpackToDestFp32, full FP32 precision)
        reconfig_data_format(cb_ring_mm, cb_ring_mm);
        copy_init(cb_ring_mm);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_ring_mm, tile_idx + block_idx, block_idx);
        }

        // exp_max_diff column-broadcast -> DST[block_size]; DST[i] *= DST[block_size]
        init_unary_bcast_col(cb_exp_max_diff);
        unary_bcast<BroadcastType::COL>(cb_exp_max_diff, 0, block_size);
        mul_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_binary_tile(block_idx, block_size, block_idx);
        }

        // DST[0..block_size-1] += P @ V[:, tile_idx : tile_idx + block_size]
        // Full matmul_block_init (not the short form): the preceding B2D broadcast leaves the
        // unpacker/math state in a form the short init does not fully restore.
        reconfig_data_format(cb_value, cb_attention_weights);
        matmul_block_init(
            cb_attention_weights,
            cb_value,
            /* transpose */ 0,
            /* ct_dim */ block_size,
            /* rt_dim */ 1,
            /* kt_dim */ Sk_chunk_t);
        for (uint32_t k_in_chunk = 0; k_in_chunk < Sk_chunk_t; ++k_in_chunk) {
            matmul_block(
                cb_attention_weights,
                cb_value,
                /* in0 (P) tile_idx */ k_in_chunk,
                /* in1 (V) tile_idx */ k_in_chunk * Wt + tile_idx,
                /* dst_idx */ 0,
                /* transpose */ 0,
                /* ct_dim */ block_size,
                /* rt_dim */ 1,
                /* kt_dim */ Sk_chunk_t);
        }

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_ring_mm);
        }
        tile_regs_release();
    }

    cb_pop_front(cb_ring_mm, Wt);
    cb_push_back(cb_ring_mm, Wt);
}

// Row-reduce the head's exp-sum (front of cb_ring_sum) with the column-of-ones tile into
// cb_out (one tile, value in column 0). Pops the ring front.
template <uint32_t cb_matmul_reduce>
void grouped_row_reduce_sum(uint32_t cb_ring_sum, uint32_t cb_out) {
    cb_wait_front(cb_ring_sum, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);

    constexpr uint32_t reduce_dst_idx = 0;
    reconfig_data_format(cb_matmul_reduce, cb_ring_sum);
    tile_regs_acquire();
    matmul_init(cb_ring_sum, cb_matmul_reduce, 0);
    matmul_tiles(cb_ring_sum, cb_matmul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduce_dst_idx);
    tile_regs_commit();

    cb_pop_front(cb_ring_sum, onetile);
    cb_reserve_back(cb_out, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(reduce_dst_idx, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, onetile);
}
