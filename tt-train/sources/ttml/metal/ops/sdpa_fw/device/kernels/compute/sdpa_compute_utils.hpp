// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"
#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h"
#endif

constexpr uint32_t onetile = 1U;

// Round `a` up to the nearest multiple of `b`. Uses the bitwise fast path when
// `b` is a power of two and the standard divmul form otherwise.
inline constexpr uint32_t round_up(uint32_t a, uint32_t b) {
    if ((b & (b - 1U)) == 0U) {
        return (a + (b - 1U)) & -b;
    }
    return ((a + b - 1U) / b) * b;
}

// SFPU intrinsics for first-column-only operations.
// Column vectors (from row-reduce) only have meaningful data in column 0,
// so we process 4 SFPU iterations (half-face) instead of the standard 8,
// saving ~75% of SFPU cycles.
#ifdef TRISC_MATH
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_recip_first_column, idst, VectorMode::C);
}
#endif  // TRISC_MATH

// Apply an attention mask to a Q@K^T score tile already sitting in DST register `register_idx`.
//
// The input `attn_mask` tile holds 1.0 at valid (keep) positions and 0.0 at masked (drop)
// positions. The function transforms it to (0.0, -1e9) in-place and adds it to the score
// tile so that masked positions become large negative values that vanish under softmax.
//
// Preconditions:
//   - The DST register buffer must be in acquired state via *acquire_dst*.
//   - `cb_attn_mask` must have at least `mask_tile_idx + 1` tiles fronted.
//
// `mask_tile_idx` selects which mask tile in `cb_attn_mask` to apply:
//   - When the writer pre-stages two causal masks in this CB, callers pass 0 for the
//     diagonal-tile pattern and 1 for the all-masked pattern used past the diagonal.
//   - When the host provides an arbitrary mask, callers pass the per-chunk tile offset.
//
// Scaling by the SDPA scaler factor is NOT done here; it is deferred to after the
// row-max subtraction for better numerical precision.
void apply_mask_on_reg(
    uint32_t register_idx,
    uint32_t cb_attn_mask,
    uint32_t minus_one_bits,
    uint32_t custom_inf_bits,
    uint32_t mask_tile_idx = 0U) {
    const uint32_t mask_register = register_idx + 1U;  // mask register should be next to data register
    cb_wait_front(cb_attn_mask, onetile);
    copy_tile_init(cb_attn_mask);
    copy_tile(
        cb_attn_mask,
        /* tile_idx */ mask_tile_idx,
        /* register idx */ mask_register);

    // Apply the attention mask to Q @ K^T scores:
    // masked positions receive 0.0, unmasked positions remain unchanged
    mask_tile_init();
    mask_tile(register_idx, mask_register);

    // Transform mask: 1/0 → 0/-1 → 0/-1e9F, then add to scores.
    // Scaling is NOT applied here — it is deferred to after max-subtraction
    // for better numerical precision (avoids scale * large_raw_score in BF16).
    binop_with_scalar_tile_init();
    add_unary_tile(mask_register, minus_one_bits);   // subtract 1.0 from mask, so it becomes 0.0 and -1.0
    mul_unary_tile(mask_register, custom_inf_bits);  // multiply by 1e9F to transform mask to 0.0 and -1e9F

    // Add mask to scaled matmul result:
    // masked positions receive large negative values (will be 0.0 after softmax),
    // unmasked positions remain unchanged
    add_binary_tile_init();
    add_binary_tile(register_idx, mask_register, register_idx);
}

// TODO: replace FPU reduce_tile(MAX) + sub_tiles_bcast_cols with SFPU equivalents once LLK
// provides SFPU row-reduce-max and sub_bcast_col. This will allow the full softmax (max, sub,
// exp, sum, reciprocal) to stay in DST at FP32 without pack/unpack round-trips through the CB.
//
// Reduces over `Sk_chunk_t` consecutive attention-weight tiles in the same row, accumulating
// the max into a single DST register. For `Sk_chunk_t == 1` this collapses to a single
// reduce_tile call (the original single-tile behavior).
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t Sk_chunk_t = 1U>
void update_cur_row_max_value(
    uint32_t cb_attention_weights,
    uint32_t cb_identity_scaler,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max = false) {
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
        cb_wait_front(cb_prev_max, onetile);
        copy_tile_to_dst_init_short_with_dt(cb_attention_weights, cb_prev_max);
        copy_tile(cb_prev_max, /* tile_idx */ 0, /* register idx */ prev_max_dst_idx);

        // find max value between current max and previous max
        binary_max_tile_init();
        binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, VectorMode::C);
    }
    tile_regs_commit();

    cb_reserve_back(cb_cur_max, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max);
    pack_tile(reduce_dst_idx, cb_cur_max);
    tile_regs_release();
    cb_push_back(cb_cur_max, onetile);
}

// Apply exp(scale * (score - max)) in place on Sk_chunk_t attention-weight tiles, and produce
// the per-row partial chunk sum (sum over the Sk_chunk_t tiles) into cb_cur_exp_sum.
//
// For Sk_chunk_t > 1 the chunk sum is built directly in L1 via the packer's L1-accumulate path:
// the first exp tile overwrites cb_cur_exp_sum[0], subsequent tiles add to it. This avoids an
// extra DST cycle (and an extra row-reduce CB) just to fold the per-tile partial sums together.
//
// DST budget: Sk_chunk_t registers are held live across sub_tiles_bcast_cols + exp_tile +
// (two) pack passes. With fp32_dest_acc_en=true the DST has 4 tiles, so Sk_chunk_t <= 4.
template <uint32_t scaler_fp32, uint32_t Sk_chunk_t = 1U>
void apply_exp_inplace_and_find_exp_sum(uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum) {
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
    cb_wait_front(cb_cur_max, onetile);

    reconfig_data_format(cb_attention_weights, cb_cur_max);
    sub_bcast_cols_init_short(cb_attention_weights, cb_cur_max);

    // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
    // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
    tile_regs_acquire();
    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        sub_tiles_bcast_cols(
            cb_attention_weights, cb_cur_max, /* in0 tile_idx */ n, /* in1 tile_idx */ 0, /* dst_reg_idx */ n);
        sdpa_exp_tile_scaled<scaler_fp32>(n);
    }
    tile_regs_commit();

    // In-place overwrite of cb_attention_weights. The CB is sized exactly Sk_chunk_t (no
    // double-buffering), so pop+reserve aliases to the same physical L1 slots.
    cb_pop_front(cb_attention_weights, Sk_chunk_t);
    cb_reserve_back(cb_attention_weights, Sk_chunk_t);
    cb_reserve_back(cb_cur_exp_sum, onetile);

    tile_regs_wait();
    // Write the Sk_chunk_t exp tiles back to cb_attention_weights at offsets 0..Sk_chunk_t-1.
    pack_reconfig_data_format(cb_attention_weights);
    pack_reconfig_l1_acc(false);
    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        pack_tile(n, cb_attention_weights);
    }
    cb_push_back(cb_attention_weights, Sk_chunk_t);

    // Build the chunk's partial sum directly in cb_cur_exp_sum[0]:
    //   - first exp tile: overwrite slot 0 (L1-acc off);
    //   - subsequent tiles: L1-accumulate onto slot 0.
    pack_reconfig_data_format(cb_cur_exp_sum);
    pack_tile</* out_of_order */ true>(/* dst */ 0, cb_cur_exp_sum, /* offset */ 0);
    if constexpr (Sk_chunk_t > 1U) {
        pack_reconfig_l1_acc(true);
        for (uint32_t n = 1U; n < Sk_chunk_t; ++n) {
            pack_tile</* out_of_order */ true>(/* dst */ n, cb_cur_exp_sum, /* offset */ 0);
        }
        pack_reconfig_l1_acc(false);
    }
    cb_push_back(cb_cur_exp_sum, onetile);
    tile_regs_release();
}

// Computes (attention_weights @ V) where attention_weights is a row of Sk_chunk_t score tiles
// and V is the corresponding Sk_chunk_t × Wt block. The K-dimension reduction (over Sk_chunk_t)
// is accumulated inside the matmul DST register across an inner loop. For Sk_chunk_t == 1 the
// inner loop runs once and indexing collapses to the original single-tile-row form.
//
// Each output block of `block_size` feat tiles is produced by one matmul_block call per K step
// (ct_dim=block_size, rt_dim=1, kt_dim=Sk_chunk_t). V is row-major in cb_value (seq outer, feat
// inner), so the `block_size` V tiles consumed per K step at a fixed seq are already contiguous
// in the CB — no transpose read needed (contrast with QK^T where K must be col-major). The
// matmul_block MOP walks them contiguously starting at in1_idx = k*Wt+tile_idx.
//
// `block_size` here is the caller's choice. `matmul_block` writes ct_dim output tiles into
// DST and needs no scratch register, so callers can pass up to `dst_size` (4 for fp32_acc,
// 8 for bf16_acc) — distinct from the SFPU block_size used elsewhere that needs `+1` for
// `unary_bcast` scratch and is therefore capped at `dst_size - 1`.
template <uint32_t Sk_chunk_t = 1U>
void matmul_qk_by_v(
    uint32_t Wt, uint32_t block_size, uint32_t cb_attention_weights, uint32_t cb_value, uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
    cb_wait_front(cb_value, Sk_chunk_t * Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);

    // matmul maps: in0(attention_weights)→SrcB, in1(value)→SrcA
    reconfig_data_format(cb_value, cb_attention_weights);
    pack_reconfig_data_format(cb_cur_mm_out);
    matmul_block_init(
        cb_attention_weights,
        cb_value,
        /* transpose */ 0,
        /* ct_dim */ block_size,
        /* rt_dim */ 1,
        /* kt_dim */ Sk_chunk_t);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        // K-reduction: at each k step, matmul_block writes ct_dim=block_size output tiles into
        // DST[0..block_size-1], accumulating across k steps. For Sk_chunk_t==1 the loop runs
        // once and behavior matches the legacy single-step matmul.
        for (uint32_t k_in_chunk = 0; k_in_chunk < Sk_chunk_t; ++k_in_chunk) {
            matmul_block(
                cb_attention_weights,
                cb_value,
                /* in0 (A) tile_idx */ k_in_chunk,
                /* in1 (B) tile_idx */ k_in_chunk * Wt + tile_idx,
                /* dst_idx */ 0,
                /* transpose */ 0,
                /* ct_dim */ block_size,
                /* rt_dim */ 1,
                /* kt_dim */ Sk_chunk_t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_cur_mm_out);
        }
        tile_regs_release();
    }
    cb_push_back(cb_cur_mm_out, Wt);
}

template <uint32_t scaler_fp32>
void update_exp_max_diff(uint32_t cb_prev_max_value, uint32_t cb_cur_max_value, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_max_value, onetile);
    cb_wait_front(cb_cur_max_value, onetile);

    cb_reserve_back(cb_exp_max_diff, onetile);

    const uint32_t exp_max_diff_dst_idx = 0;
    reconfig_data_format(cb_prev_max_value, cb_cur_max_value);
    tile_regs_acquire();
    sub_init(cb_prev_max_value, cb_cur_max_value);
    sub_tiles(
        cb_prev_max_value,
        cb_cur_max_value,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* dst_reg_idx */ exp_max_diff_dst_idx);

    // First-column scaled exp: exp(scale * (prev_max - cur_max)).
    // Both max values are column vectors, so the result is a column vector —
    // only column 0 has data. Process 4× fewer SFPU iterations than full-tile exp.
    sdpa_exp_tile_first_column_scaled<scaler_fp32>(exp_max_diff_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_max_diff);
    pack_tile(exp_max_diff_dst_idx, cb_exp_max_diff);
    tile_regs_release();
    cb_push_back(cb_exp_max_diff, onetile);
}

void update_cur_exp_sum_inplace(uint32_t cb_prev_sum_exp, uint32_t cb_cur_sum_exp, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_sum_exp, onetile);
    cb_wait_front(cb_cur_sum_exp, onetile);
    cb_wait_front(cb_exp_max_diff, onetile);

    const uint32_t exp_sum_dst_idx = 0;
    reconfig_data_format(cb_prev_sum_exp, cb_exp_max_diff);  // reconfig data format to precise
    mul_bcast_cols_init_short(cb_prev_sum_exp, cb_exp_max_diff);
    tile_regs_acquire();
    // multiply previous exp sum with exp_max_diff
    mul_tiles_bcast_cols(cb_prev_sum_exp, cb_exp_max_diff, 0, 0, exp_sum_dst_idx);

    // copy current sum exp to next register
    copy_tile_init(cb_cur_sum_exp);
    copy_tile(cb_cur_sum_exp, /* tile_idx */ 0, /* register idx */ exp_sum_dst_idx + 1U);

    // add to updated previous exp sum with current exp sum
    add_binary_tile_init();
    add_binary_tile(exp_sum_dst_idx, exp_sum_dst_idx + 1U, exp_sum_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_cur_sum_exp, onetile);
    cb_reserve_back(cb_cur_sum_exp, onetile);
    pack_reconfig_data_format(cb_cur_sum_exp);
    pack_tile(exp_sum_dst_idx, cb_cur_sum_exp);
    tile_regs_release();
    cb_push_back(cb_cur_sum_exp, onetile);
}

// Lightweight init for unary_bcast<COL> with B2D datacopy path.
// Reads column 0 from a CB tile and replicates it to all 32 columns in a DST register.
// Only reprograms UNPACK and MATH MOPs — safe inside a tile_regs_acquire block.
// Uses reconfig_data_format_srcb so SrcA format register is preserved for the caller.
void init_unary_bcast_col(uint32_t cb_col_vec) {
    reconfig_data_format_srcb(cb_col_vec);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_col_vec)));
    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::COL>(
        cb_col_vec)));
}

// Scale prev_mm_out tiles by a column-broadcast factor and L1-accumulate onto cur_mm_out.
// Uses SFPU path: prev_mm_out is loaded via UnpackToDestFp32 (full FP32 in DST),
// the scale factor is loaded via unary_bcast<COL> (col 0 broadcast to all cols),
// and mul_binary_tile does FP32 SFPU multiply.
void update_cur_mm_out(
    uint32_t Wt, uint32_t block_size, uint32_t cb_prev_mm_out, uint32_t cb_cur_mm_out, uint32_t cb_exp_max_diff) {
    cb_wait_front(cb_prev_mm_out, Wt);
    cb_wait_front(cb_cur_mm_out, Wt);
    cb_wait_front(cb_exp_max_diff, onetile);

    pack_reconfig_data_format(cb_cur_mm_out);
    pack_reconfig_l1_acc(true);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();

        // Load prev_mm_out tiles to DST via UnpackToDestFp32 (full FP32 precision)
        reconfig_data_format(cb_prev_mm_out, cb_prev_mm_out);
        copy_tile_init(cb_prev_mm_out);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_prev_mm_out, tile_idx + block_idx, block_idx);
        }

        // Load exp_max_diff with column broadcast to DST[block_size]
        init_unary_bcast_col(cb_exp_max_diff);
        unary_bcast<BroadcastType::COL>(cb_exp_max_diff, 0, block_size);

        // SFPU element-wise multiply: DST[i] = DST[i] * DST[block_size]
        mul_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_binary_tile(block_idx, block_size, block_idx);
        }

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_cur_mm_out);
        }
        tile_regs_release();
    }
    pack_reconfig_l1_acc(false);

    cb_pop_front(cb_cur_mm_out, Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);
    cb_push_back(cb_cur_mm_out, Wt);
}

// Row-reduce a tile in place via matmul with a column-of-ones tile.
// Result is a column vector (value in column 0 of each row).
template <uint32_t cb_identity_scaler, uint32_t cb_matmul_reduce>
void row_reduce_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);
    cb_wait_front(cb_matmul_reduce, onetile);

    const uint32_t reduce_dst_idx = 0;

    reconfig_data_format(cb_matmul_reduce, cb_in_idx);
    tile_regs_acquire();

    matmul_init(cb_in_idx, cb_matmul_reduce, 0);
    matmul_tiles(cb_in_idx, cb_matmul_reduce, /* tile_idx */ 0, /* tile_idx */ 0, reduce_dst_idx);
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(reduce_dst_idx, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

// Apply reciprocal to a tile in place.
void recip_tile_inplace(uint32_t cb_in_idx) {
    cb_wait_front(cb_in_idx, onetile);

    const uint32_t dst_idx = 0;
    tile_regs_acquire();
    reconfig_data_format(cb_in_idx, cb_in_idx);
    copy_tile_init(cb_in_idx);
    copy_tile(cb_in_idx, /* tile_idx */ 0, dst_idx);
    recip_tile_init</* legacy_compat */ false>();
    MATH((recip_tile_first_column(dst_idx)));
    tile_regs_commit();

    tile_regs_wait();
    cb_pop_front(cb_in_idx, onetile);
    cb_reserve_back(cb_in_idx, onetile);
    pack_reconfig_data_format(cb_in_idx);
    pack_tile(dst_idx, cb_in_idx);
    tile_regs_release();
    cb_push_back(cb_in_idx, onetile);
}

// Compute logsumexp = scale*max + log(sum_exp) and pack to output CB with column-0 masking.
// Max is unscaled (raw QK^T max), so we multiply by scale to get the correct LSE.
// Reads from cb_sum_exp (row-reduced) and cb_max WITHOUT consuming them.
// cb_mask_tile should contain 1.0 in column 0 and 0.0 elsewhere (cb_matmul_reduce).
// The lse tile is computed entirely in FP32 DST registers for maximum precision.
void compute_and_pack_lse(
    uint32_t cb_sum_exp, uint32_t cb_max, uint32_t cb_out_lse, uint32_t cb_mask_tile, uint32_t scaler_bits) {
    // mask_tile HW constraint: mask register must be idst_data + 1
    const uint32_t lse_reg = 0;
    const uint32_t mask_reg = 1U;
    const uint32_t max_reg = 2U;

    tile_regs_acquire();

    reconfig_data_format(cb_sum_exp, cb_sum_exp);
    copy_tile_init(cb_sum_exp);
    copy_tile(cb_sum_exp, /* tile_idx */ 0, lse_reg);

    log_tile_init</* fast_and_approx */ false>();
    log_tile</* fast_and_approx */ false>(lse_reg);

    copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_sum_exp, /* new_cb_idx */ cb_max);
    copy_tile(cb_max, /* tile_idx */ 0, max_reg);

    // lse = scale * max + log(sum_exp)
    binop_with_scalar_tile_init();
    mul_unary_tile(max_reg, scaler_bits);

    add_binary_tile_init();
    add_binary_tile(max_reg, lse_reg, lse_reg);

    copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_max, /* new_cb_idx */ cb_mask_tile);
    copy_tile(cb_mask_tile, /* tile_idx */ 0, mask_reg);

    mask_tile_init();
    mask_tile(lse_reg, mask_reg);

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out_lse, onetile);
    pack_reconfig_data_format(cb_out_lse);
    pack_tile(lse_reg, cb_out_lse);
    tile_regs_release();
    cb_push_back(cb_out_lse, onetile);
}

// Pack intermediate result with masking to ensure only column 0 has value, rest are zeros.
// cb_mask_tile should contain 1.0 in column 0 and 0.0 elsewhere (use cb_matmul_reduce).
// Input tile has reduced value in column 0 after row reduction.
// Output tile will have value only in column 0, all other columns zeroed out.
void pack_intermediate_result(
    uint32_t cb_in_idx, uint32_t cb_out_idx, uint32_t cb_mask_tile, uint32_t tiles_count = 1U) {
    cb_wait_front(cb_in_idx, tiles_count);
    cb_reserve_back(cb_out_idx, tiles_count);

    const uint32_t dst_idx = 0;

    for (uint32_t tile_idx = 0; tile_idx < tiles_count; ++tile_idx) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_mask_tile, /* new_cb_idx */ cb_in_idx);
        copy_tile(cb_in_idx, /* tile_idx */ tile_idx, /* register idx */ dst_idx);

        copy_tile_to_dst_init_short_with_dt(/* old_cb_idx */ cb_in_idx, /* new_cb_idx */ cb_mask_tile);
        copy_tile(cb_mask_tile, /* tile_idx */ 0, /* register idx */ dst_idx + 1U);

        mask_tile_init();
        mask_tile(dst_idx, dst_idx + 1U);

        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_out_idx);
        pack_tile(dst_idx, cb_out_idx);
        tile_regs_release();
    }

    cb_push_back(cb_out_idx, tiles_count);
}
