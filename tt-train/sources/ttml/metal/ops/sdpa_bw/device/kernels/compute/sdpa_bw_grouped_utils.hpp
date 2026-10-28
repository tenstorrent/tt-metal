// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sdpa_bw_compute_utils.hpp"

// Helpers for the "grouped" SDPA backward paths, where one work item carries several
// independent accumulators in a single CB region:
//
//   dQ kernel (GROUPED_HEADS): item = (batch, kv_head, sub-group of heads_per_item Q heads,
//     Q tile-row). dO / O / Q / lse of all heads are staged once, every causal K/V row is
//     streamed once and applied to each head. Per-head state:
//       cb_u_scalar_row       heads_per_item fp32 tiles, all fronted; head j uses tile j.
//       cb_intermediates      heads_per_item fp32 lse tiles, all fronted; head j uses tile j.
//       cb_grad_query_accum   heads_per_item * qWt fp32 tiles; head j's dQ is at [j*qWt, (j+1)*qWt).
//
//   dK/dV kernel (GROUPED_KV_ROWS): item = (batch, kv_head, block of kv_rows_per_item
//     consecutive K/V tile-rows). The K/V rows stay resident, the causal Q / dO / lse / u
//     suffix is streamed once and applied to each K/V row. Row r's dK / dV accumulators are
//     at [r*kWt, (r+1)*kWt) / [r*vWt, (r+1)*vWt) of their accumulator CBs.
//
// In both cases the accumulator region is reserved ONCE per item and pushed once at the end;
// rows are written with out-of-order packs (plain on first touch, L1-accumulate afterwards).

// u = rowsum(dO ⊙ O) for one head. Same arithmetic as compute_u_scalar_row, but the dO / O
// rows are addressed by `tile_base` and the un-reduced product goes through `cb_tmp` (a
// single-tile fp32 CB) instead of in-place in cb_u_scalar_row, so u lands at the BACK of
// cb_u_scalar_row (one tile per head) and a copy goes to cb_u_scaler_output for the KV kernel.
void grouped_compute_u_scalar_row(
    const uint32_t cb_grad_output,
    const uint32_t cb_attn_output,
    const uint32_t tile_base,
    const uint32_t cb_tmp,
    const uint32_t cb_u_scalar_row,
    const uint32_t cb_mat_mul_reduction,
    const uint32_t tiles_per_row,
    const uint32_t cb_u_scaler_output) {
    constexpr uint32_t accum_register = 0;

    reconfig_data_format(cb_grad_output, cb_attn_output);
    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(cb_grad_output, cb_attn_output, /*acc_to_dest*/ true);
    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; ++tile_idx) {
        mul_tiles(
            cb_grad_output,
            cb_attn_output,
            /* tile_idx */ tile_base + tile_idx,
            /* tile_idx */ tile_base + tile_idx,
            /* dst_reg_idx*/ accum_register);
    }
    tile_regs_commit();

    pack_reconfig_data_format(cb_tmp);
    cb_reserve_back(cb_tmp, onetile);
    tile_regs_wait();
    pack_tile(accum_register, cb_tmp);
    tile_regs_release();
    cb_push_back(cb_tmp, onetile);

    cb_wait_front(cb_tmp, onetile);
    tile_regs_acquire();
    reconfig_data_format(cb_tmp, cb_mat_mul_reduction);
    matmul_init(cb_tmp, cb_mat_mul_reduction, /* transpose */ 0);
    matmul_tiles(cb_tmp, cb_mat_mul_reduction, /* tile_idx */ 0, /* tile_idx */ 0, accum_register);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_tmp, onetile);
    cb_reserve_back(cb_u_scalar_row, onetile);
    pack_reconfig_data_format(cb_u_scalar_row);
    pack_tile(accum_register, cb_u_scalar_row);

    cb_reserve_back(cb_u_scaler_output, onetile);
    pack_reconfig_data_format(cb_u_scalar_row, cb_u_scaler_output);
    pack_tile(accum_register, cb_u_scaler_output);
    cb_push_back(cb_u_scaler_output, onetile);

    tile_regs_release();
    cb_push_back(cb_u_scalar_row, onetile);
}

// accum[base .. base+tiles_per_row) (+)= A @ B_row, where A is the single fronted tile of cb_a
// and B_row the fronted `tiles_per_row` tiles of cb_b starting at `b_tile_base`
// (dQ += dS @ K, dV += P^T @ dO, dK += dS^T @ Q). The accumulator region must already be
// reserved by the caller (once per item); tiles are packed out-of-order at
// `accum_tile_base + tile_idx`, with the packer's L1-accumulate enabled when `do_accumulate`.
// Pops the A tile.
void grouped_accumulate_row_matmul(
    const uint32_t cb_a,
    const uint32_t cb_b,
    const uint32_t cb_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const uint32_t accum_tile_base,
    const bool do_accumulate,
    const uint32_t b_tile_base = 0U) {
    cb_wait_front(cb_a, onetile);

    // 2-arg pack_reconfig: skip reprogram if new PACK format matches the previously-configured one.
    pack_reconfig_data_format(cb_a, cb_accum);
    if (do_accumulate) {
        pack_reconfig_l1_acc(true);
    }

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_accum, cb_b);
        matmul_init(cb_a, cb_b, /*transpose*/ 0);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            matmul_tiles(
                cb_a,
                cb_b,
                /* tile_idx */ 0,
                /* tile_idx */ b_tile_base + tile_idx + block_idx,
                /* dst_reg_idx*/ block_idx);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile</* out_of_order */ true>(
                /*dst_reg_idx*/ block_idx, cb_accum, accum_tile_base + tile_idx + block_idx);
        }
        tile_regs_release();
    }

    if (do_accumulate) {
        pack_reconfig_l1_acc(false);
    }
    cb_pop_front(cb_a, onetile);
}

// Copy `num_tiles` tiles starting at `tile_base` of the (fronted) accumulator to cb_output
// (FP32 -> output format). Caller handles wait/pop of cb_source.
void grouped_pack_accum_to_output(
    const uint32_t cb_source, const uint32_t cb_output, const uint32_t tile_base, const uint32_t num_tiles) {
    cb_reserve_back(cb_output, num_tiles);

    pack_reconfig_data_format(cb_output);
    reconfig_data_format(cb_source, cb_source);
    copy_init(cb_source);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(cb_source, tile_base + tile_idx, /* register idx */ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/* register idx */ 0, cb_output);
        tile_regs_release();
    }
    cb_push_back(cb_output, num_tiles);
}
