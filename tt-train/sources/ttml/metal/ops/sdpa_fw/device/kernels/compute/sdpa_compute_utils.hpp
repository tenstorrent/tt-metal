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
#include "api/compute/eltwise_unary/recip.h"
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

// SFPU intrinsics for first-column-only operations.
// Column vectors (from row-reduce) only have meaningful data in column 0,
// so we process 4 SFPU iterations (half-face) instead of the standard 8,
// saving ~75% of SFPU cycles.
#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sdpa_fw_recip_first_column,
        (is_fp32_dest_acc_en),
        idst,
        VectorMode::C);
}
#endif  // TRISC_MATH

// Softmax helpers. Score CB is row-major (row r, K tile n at r * Sk_chunk_t + n); each stat CB is
// rows deep with row r in slot r. The row-range helpers (row0, n) leave the CB waits/reserves/pushes to
// the caller; the block-level ones (apply_exp_inplace_and_find_exp_sum, matmul_qk_by_v) do their own.

// TODO: replace FPU reduce_tile(MAX) + sub_tiles_bcast_cols with SFPU equivalents once LLK
// provides SFPU row-reduce-max and sub_bcast_col. This will allow the full softmax (max, sub,
// exp, sum, reciprocal) to stay in DST at FP32 without pack/unpack round-trips through the CB.
//
// Max over rows [row0, row0 + n) of the score CB, folded with their running max, packed to
// cb_cur_max[row]. One DST pass: slot i holds row i's max, slot n + i its previous max, so n <= 4 without
// the fold and n <= 2 with it.
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t Sk_chunk_t>
void update_row_max_values(
    uint32_t cb_attention_weights,
    uint32_t cb_identity_scaler,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max,
    uint32_t row0,
    uint32_t n) {
    reconfig_data_format(cb_attention_weights, cb_identity_scaler);
    reduce_init<pool_type, reduce_dim>(cb_attention_weights, cb_identity_scaler, cb_cur_max);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < Sk_chunk_t; ++k) {
            reduce_tile<pool_type, reduce_dim>(
                cb_attention_weights,
                cb_identity_scaler,
                /* in0 tile_idx */ (row0 + i) * Sk_chunk_t + k,
                /* in1 tile_idx */ 0,
                /* dst_reg_idx */ i);
        }
    }
    reduce_uninit();

    if (do_eltwise_max) {
        reconfig_data_format_srca(cb_attention_weights, cb_prev_max);
        copy_init(cb_prev_max);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(cb_prev_max, /* tile_idx */ row0 + i, /* register idx */ n + i);
        }
        binary_max_tile_init();
        for (uint32_t i = 0; i < n; ++i) {
            binary_max_tile(i, n + i, i, VectorMode::C);
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_max);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_cur_max, /* offset */ row0 + i);
    }
    tile_regs_release();
}

// P = exp(scale * (score - max)) written over the whole rows x Sk_chunk_t score block, pass_tiles tiles
// (in row-major order) per DST pass. Each row's sum of P is built in cb_cur_exp_sum[row] by the packer: the
// row's first tile is packed normally, the rest with L1-accumulate on, which adds the tile onto the slot.
template <uint32_t scaler_fp32, uint32_t scaler_bf16, uint32_t rows, uint32_t Sk_chunk_t, uint32_t pass_tiles>
void apply_exp_inplace_and_find_exp_sum(uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum) {
    constexpr uint32_t total = rows * Sk_chunk_t;

    reconfig_data_format(cb_attention_weights, cb_cur_max);
    sub_bcast_cols_init(cb_attention_weights, cb_cur_max);

    for (uint32_t t0 = 0; t0 < total; t0 += pass_tiles) {
        const uint32_t n = (total - t0 < pass_tiles) ? (total - t0) : pass_tiles;
        // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
        // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            sub_tiles_bcast_cols(
                cb_attention_weights,
                cb_cur_max,
                /* in0 tile_idx */ t0 + i,
                /* in1 tile_idx (row) */ (t0 + i) / Sk_chunk_t,
                /* dst_reg_idx */ i);
            sdpa_exp_tile_scaled<scaler_fp32, scaler_bf16>(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_attention_weights);
        pack_reconfig_l1_acc(false);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile</* out_of_order */ true>(i, cb_attention_weights, /* offset */ t0 + i);
        }

        // row sums: a row's first tile overwrites its slot, the rest add onto it
        pack_reconfig_data_format(cb_cur_exp_sum);
        bool accumulate = false;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = t0 + i;
            const bool first_of_row = (t % Sk_chunk_t == 0U);
            if (first_of_row == accumulate) {
                accumulate = !first_of_row;
                pack_reconfig_l1_acc(accumulate);
            }
            pack_tile</* out_of_order */ true>(i, cb_cur_exp_sum, /* offset */ t / Sk_chunk_t);
        }
        if (accumulate) {
            pack_reconfig_l1_acc(false);
        }
        tile_regs_release();
    }
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

// O_cur = O_prev * alpha + P @ V for the whole item, one block_h x block_w rectangle of output tiles per
// DST pass. P is rows x Sk_chunk_t tiles (row-major), V is Sk_chunk_t x Wt tiles (row-major, as read
// from DRAM). With `rescale`, the previous accumulator block is loaded into DST first (UnpackToDestFp32),
// multiplied by its row's alpha (broadcast into the spare DST slot block_h * block_w), and the matmul
// accumulates on top of it; so block_h * block_w must leave one DST tile free.
template <uint32_t Sk_chunk_t, uint32_t rows>
void matmul_qk_by_v(
    uint32_t Wt,
    uint32_t block_h,
    uint32_t block_w,
    uint32_t cb_attention_weights,
    uint32_t cb_value,
    uint32_t cb_prev_mm_out,
    uint32_t cb_exp_max_diff,
    uint32_t cb_cur_mm_out,
    bool rescale) {
    cb_wait_front(cb_attention_weights, rows * Sk_chunk_t);
    cb_wait_front(cb_value, Sk_chunk_t * Wt);
    if (rescale) {
        cb_wait_front(cb_prev_mm_out, rows * Wt);
        cb_wait_front(cb_exp_max_diff, rows);
    }
    cb_reserve_back(cb_cur_mm_out, rows * Wt);

    const uint32_t alpha_dst = block_h * block_w;  // spare slot for the broadcast factor

    // matmul maps: in0(attention_weights)->SrcB, in1(value)->SrcA
    reconfig_data_format(cb_value, cb_attention_weights);
    pack_reconfig_data_format(cb_cur_mm_out);
    matmul_block_init(
        cb_attention_weights,
        cb_value,
        /* transpose */ 0,
        /* ct_dim */ block_w,
        /* rt_dim */ block_h,
        /* kt_dim */ Sk_chunk_t);

    for (uint32_t row0 = 0; row0 < rows; row0 += block_h) {
        for (uint32_t col0 = 0; col0 < Wt; col0 += block_w) {
            tile_regs_acquire();
            if (rescale) {
                // DST[i*block_w + j] = O_prev[row0+i][col0+j] * alpha[row0+i]
                reconfig_data_format(cb_prev_mm_out, cb_prev_mm_out);
                copy_init(cb_prev_mm_out);
                for (uint32_t i = 0; i < block_h; ++i) {
                    for (uint32_t j = 0; j < block_w; ++j) {
                        copy_tile(cb_prev_mm_out, (row0 + i) * Wt + col0 + j, i * block_w + j);
                    }
                }
                init_unary_bcast_col(cb_exp_max_diff);
                mul_binary_tile_init();
                for (uint32_t i = 0; i < block_h; ++i) {
                    unary_bcast<BroadcastType::COL>(cb_exp_max_diff, row0 + i, alpha_dst);
                    for (uint32_t j = 0; j < block_w; ++j) {
                        mul_binary_tile(i * block_w + j, alpha_dst, i * block_w + j);
                    }
                }
                // The B2D broadcast and SFPU multiply leave the FPU in a different state: full re-init.
                reconfig_data_format(cb_value, cb_attention_weights);
                matmul_block_init(
                    cb_attention_weights,
                    cb_value,
                    /* transpose */ 0,
                    /* ct_dim */ block_w,
                    /* rt_dim */ block_h,
                    /* kt_dim */ Sk_chunk_t);
            }
            for (uint32_t k = 0; k < Sk_chunk_t; ++k) {
                matmul_block(
                    cb_attention_weights,
                    cb_value,
                    /* in0 (P) tile_idx */ row0 * Sk_chunk_t + k,
                    /* in1 (V) tile_idx */ k * Wt + col0,
                    /* dst_idx */ 0,
                    /* transpose */ 0,
                    /* ct_dim */ block_w,
                    /* rt_dim */ block_h,
                    /* kt_dim */ Sk_chunk_t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_h; ++i) {
                for (uint32_t j = 0; j < block_w; ++j) {
                    pack_tile</* out_of_order */ true>(
                        i * block_w + j, cb_cur_mm_out, /* offset */ (row0 + i) * Wt + col0 + j);
                }
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_cur_mm_out, rows * Wt);
}

// exp(scale * (prev_max - cur_max)) for rows [row0, row0 + n) -> cb_exp_max_diff[row], n <= 4 (one DST slot
// per row).
template <uint32_t scaler_fp32, uint32_t scaler_bf16>
void update_exp_max_diffs(
    uint32_t cb_prev_max_value, uint32_t cb_cur_max_value, uint32_t cb_exp_max_diff, uint32_t row0, uint32_t n) {
    reconfig_data_format(cb_prev_max_value, cb_cur_max_value);
    sub_init(cb_prev_max_value, cb_cur_max_value);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        sub_tiles(cb_prev_max_value, cb_cur_max_value, /* tile_idx */ row0 + i, /* tile_idx */ row0 + i, i);
    }
    // First-column scaled exp: both max values are column vectors, so only column 0 has data and the
    // SFPU runs 4x fewer iterations than a full-tile exp.
    for (uint32_t i = 0; i < n; ++i) {
        sdpa_exp_tile_first_column_scaled<scaler_fp32, scaler_bf16>(i);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_exp_max_diff);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_exp_max_diff, /* offset */ row0 + i);
    }
    tile_regs_release();
}

// cur_sum[row] = prev_sum[row] * exp_max_diff[row] + cur_sum[row] for rows [row0, row0 + n), in place.
// Two DST slots per row (product, current sum), so n <= 2.
void update_cur_exp_sums_inplace(
    uint32_t cb_prev_sum_exp, uint32_t cb_cur_sum_exp, uint32_t cb_exp_max_diff, uint32_t row0, uint32_t n) {
    reconfig_data_format(cb_prev_sum_exp, cb_exp_max_diff);  // reconfig data format to precise
    mul_bcast_cols_init(cb_prev_sum_exp, cb_exp_max_diff);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        mul_tiles_bcast_cols(cb_prev_sum_exp, cb_exp_max_diff, row0 + i, row0 + i, i);
    }
    copy_init(cb_cur_sum_exp);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_cur_sum_exp, /* tile_idx */ row0 + i, /* register idx */ n + i);
    }
    add_binary_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        add_binary_tile(i, n + i, i);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_cur_sum_exp);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_cur_sum_exp, /* offset */ row0 + i);
    }
    tile_regs_release();
}

// Row-reduce tiles [row0, row0 + n) of cb_in_idx in place (matmul with a ones column), n <= 4.
template <uint32_t cb_matmul_reduce>
void row_reduce_tiles_inplace(uint32_t cb_in_idx, uint32_t row0, uint32_t n) {
    reconfig_data_format(cb_matmul_reduce, cb_in_idx);
    tile_regs_acquire();
    matmul_init(cb_in_idx, cb_matmul_reduce, 0);
    for (uint32_t i = 0; i < n; ++i) {
        matmul_tiles(cb_in_idx, cb_matmul_reduce, /* tile_idx */ row0 + i, /* tile_idx */ 0, i);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_in_idx);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_in_idx, /* offset */ row0 + i);
    }
    tile_regs_release();
}

// 1/x on tiles [row0, row0 + n) of cb_in_idx in place, n <= 4.
void recip_tiles_inplace(uint32_t cb_in_idx, uint32_t row0, uint32_t n) {
    tile_regs_acquire();
    reconfig_data_format(cb_in_idx, cb_in_idx);
    copy_init(cb_in_idx);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_in_idx, /* tile_idx */ row0 + i, i);
    }
    recip_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        MATH((recip_tile_first_column(i)));
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_in_idx);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_in_idx, /* offset */ row0 + i);
    }
    tile_regs_release();
}

// Split units. A sender packs its running state, per row [max, unreduced sum, output tiles], into
// cb_partial_out (fp32) for the writer; a merger folds such a partial into its own running state, like
// one more K/V step: m = max(m_own, m_p), alpha_x = exp(scale * (m_x - m)), sum = sum_own * alpha_own +
// sum_p * alpha_p, O = O_own * alpha_own + O_p * alpha_p. One row per DST pass; splits are rare. The
// helpers are kept out of line: every inlined LLK init costs kernel text, and the program has a budget.
#ifdef SPLIT_UNITS
__attribute__((noinline, noclone)) void copy_cb_tile_to_dst(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format(cb, cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

__attribute__((noinline, noclone)) void bcast_col_to_dst(uint32_t cb, uint32_t row, uint32_t dst) {
    init_unary_bcast_col(cb);
    unary_bcast<BroadcastType::COL>(cb, row, dst);
}

__attribute__((noinline, noclone)) void pack_dst_to(uint32_t dst, uint32_t cb, uint32_t offset) {
    pack_reconfig_data_format(cb);
    pack_tile</* out_of_order */ true>(dst, cb, offset);
}

// DST[0] = DST[0] * bcast(alpha_a[row]) + DST[2] * bcast(alpha_b[row])
__attribute__((noinline, noclone)) void scale_and_add_dst(uint32_t cb_alpha_a, uint32_t cb_alpha_b, uint32_t row) {
    bcast_col_to_dst(cb_alpha_a, row, 1);
    bcast_col_to_dst(cb_alpha_b, row, 3);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    mul_binary_tile(2, 3, 2);
    add_binary_tile_init();
    add_binary_tile(0, 2, 0);
}

__attribute__((noinline, noclone)) void pack_dst_inorder(uint32_t dst, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
}

// Partials are streamed one row at a time: the CBs hold two rows of (Wt + 2) tiles.
template <uint32_t rows>
void send_partial(uint32_t Wt, uint32_t cb_mm_out, uint32_t cb_max, uint32_t cb_sum_exp, uint32_t cb_partial_out) {
    const uint32_t tiles_per_row = Wt + 2U;
    cb_wait_front(cb_mm_out, rows * Wt);
    cb_wait_front(cb_max, rows);
    cb_wait_front(cb_sum_exp, rows);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_partial_out, tiles_per_row);
        for (uint32_t t = 0; t < tiles_per_row; ++t) {
            tile_regs_acquire();
            if (t == 0U) {
                copy_cb_tile_to_dst(cb_max, r, 0);
            } else if (t == 1U) {
                copy_cb_tile_to_dst(cb_sum_exp, r, 0);
            } else {
                copy_cb_tile_to_dst(cb_mm_out, r * Wt + t - 2U, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_dst_inorder(0, cb_partial_out);
            tile_regs_release();
        }
        cb_push_back(cb_partial_out, tiles_per_row);
    }
    cb_pop_front(cb_mm_out, rows * Wt);
    cb_pop_front(cb_max, rows);
    cb_pop_front(cb_sum_exp, rows);
}

template <uint32_t scaler_fp32, uint32_t scaler_bf16, uint32_t rows>
void merge_partial(
    uint32_t Wt,
    uint32_t cb_prev_mm_out,
    uint32_t cb_prev_max,
    uint32_t cb_prev_sum_exp,
    uint32_t cb_partial_in,
    uint32_t cb_cur_mm_out,
    uint32_t cb_cur_max,
    uint32_t cb_cur_sum_exp,
    uint32_t cb_alpha_own,
    uint32_t cb_alpha_p) {
    const uint32_t tiles_per_row = Wt + 2U;
    cb_wait_front(cb_prev_mm_out, rows * Wt);
    cb_wait_front(cb_prev_max, rows);
    cb_wait_front(cb_prev_sum_exp, rows);
    cb_reserve_back(cb_cur_mm_out, rows * Wt);  // packed in order, row after row

    for (uint32_t r = 0; r < rows; ++r) {
        cb_wait_front(cb_partial_in, tiles_per_row);

        // (1) m = max(m_own, m_p); alpha_own = exp(scale * (m_own - m)); alpha_p = exp(scale * (m_p - m)).
        //     The maxes are column vectors, so the exp runs on the first column only. Each result goes to
        //     its CB one tile at a time, so it is visible to the next pass.
        tile_regs_acquire();
        copy_cb_tile_to_dst(cb_prev_max, r, 0);
        copy_cb_tile_to_dst(cb_partial_in, 0, 1);
        copy_cb_tile_to_dst(cb_prev_max, r, 2);
        copy_cb_tile_to_dst(cb_partial_in, 0, 3);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0, VectorMode::C);
        sub_binary_tile_init();
        sub_binary_tile(2, 0, 2);
        sub_binary_tile(3, 0, 3);
        sdpa_exp_tile_first_column_scaled<scaler_fp32, scaler_bf16>(2);
        sdpa_exp_tile_first_column_scaled<scaler_fp32, scaler_bf16>(3);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_cur_max, 1);
        pack_dst_inorder(0, cb_cur_max);
        cb_push_back(cb_cur_max, 1);
        cb_reserve_back(cb_alpha_own, 1);
        pack_dst_inorder(2, cb_alpha_own);
        cb_push_back(cb_alpha_own, 1);
        cb_reserve_back(cb_alpha_p, 1);
        pack_dst_inorder(3, cb_alpha_p);
        cb_push_back(cb_alpha_p, 1);
        tile_regs_release();
        cb_wait_front(cb_alpha_own, r + 1U);
        cb_wait_front(cb_alpha_p, r + 1U);

        // (2) sum = sum_own * alpha_own + sum_p * alpha_p, then the same per output tile
        for (uint32_t t = 0; t < Wt + 1U; ++t) {  // t == 0: the sums; t > 0: output tile t - 1
            tile_regs_acquire();
            if (t == 0U) {
                copy_cb_tile_to_dst(cb_prev_sum_exp, r, 0);
            } else {
                copy_cb_tile_to_dst(cb_prev_mm_out, r * Wt + t - 1U, 0);
            }
            copy_cb_tile_to_dst(cb_partial_in, 1U + t, 2);
            scale_and_add_dst(cb_alpha_own, cb_alpha_p, r);
            tile_regs_commit();
            tile_regs_wait();
            if (t == 0U) {
                cb_reserve_back(cb_cur_sum_exp, 1);
                pack_dst_inorder(0, cb_cur_sum_exp);
                cb_push_back(cb_cur_sum_exp, 1);
            } else {
                pack_dst_inorder(0, cb_cur_mm_out);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_partial_in, tiles_per_row);
    }
    cb_push_back(cb_cur_mm_out, rows * Wt);

    cb_wait_front(cb_alpha_own, rows);
    cb_pop_front(cb_alpha_own, rows);
    cb_wait_front(cb_alpha_p, rows);
    cb_pop_front(cb_alpha_p, rows);
    cb_pop_front(cb_prev_mm_out, rows * Wt);
    cb_pop_front(cb_prev_max, rows);
    cb_pop_front(cb_prev_sum_exp, rows);
}
#endif  // SPLIT_UNITS

// lse[row] = scale * max + log(sum_exp), FP32, column 0 only, packed to cb_out_lse[row].
void compute_and_pack_lse(
    uint32_t cb_sum_exp,
    uint32_t cb_max,
    uint32_t cb_out_lse,
    uint32_t cb_mask_tile,
    uint32_t scaler_bits,
    uint32_t row) {
    // mask_tile HW constraint: mask register must be idst_data + 1
    const uint32_t lse_reg = 0;
    const uint32_t mask_reg = 1U;
    const uint32_t max_reg = 2U;

    tile_regs_acquire();

    reconfig_data_format(cb_sum_exp, cb_sum_exp);
    copy_init(cb_sum_exp);
    copy_tile(cb_sum_exp, /* tile_idx */ row, lse_reg);

    log_tile_init</* fast_and_approx */ false>();
    log_tile</* fast_and_approx */ false>(lse_reg);

    reconfig_data_format_srca(/* old_cb_idx */ cb_sum_exp, /* new_cb_idx */ cb_max);
    copy_init(/* new_cb_idx */ cb_max);
    copy_tile(cb_max, /* tile_idx */ row, max_reg);

    // lse = scale * max + log(sum_exp)
    binop_with_scalar_tile_init();
    mul_unary_tile(max_reg, scaler_bits);

    add_binary_tile_init();
    add_binary_tile(max_reg, lse_reg, lse_reg);

    reconfig_data_format_srca(/* old_cb_idx */ cb_max, /* new_cb_idx */ cb_mask_tile);
    copy_init(/* new_cb_idx */ cb_mask_tile);
    copy_tile(cb_mask_tile, /* tile_idx */ 0, mask_reg);

    mask_tile_init();
    mask_tile(lse_reg, mask_reg);

    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format(cb_out_lse);
    pack_tile</* out_of_order */ true>(lse_reg, cb_out_lse, /* offset */ row);
    tile_regs_release();
}
