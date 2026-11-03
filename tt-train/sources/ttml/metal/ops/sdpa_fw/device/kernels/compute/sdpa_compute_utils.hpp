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
#include "tt-train/sources/ttml/metal/common/sdpa_item_math.hpp"
#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h"
#endif

constexpr uint32_t onetile = 1U;

// The packer is reconfigured only when the CB it packs to has another format: a reconfig stalls the packer
// (it waits on PACK and THCON and rewrites the config), and most of a step's packs go to fp32 CBs in a row.
// pack_to tracks the CB the packer was last set to and uses the (old, new) form, which skips equal formats;
// every pack site in the kernel goes through it. The kernel seeds it with the ocb of compute_kernel_hw_startup
// and keeps it right after a helper that sets the packer itself (ttnn's apply_causal_mask_lightweight).
inline uint32_t packer_cb = 0;
inline void pack_to(uint32_t cb) {
    pack_reconfig_data_format(packer_cb, cb);
    packer_cb = cb;
}

// SFPU intrinsics for first-column-only operations.
// Column vectors (from row-reduce) only have meaningful data in column 0,
// so we process 4 SFPU iterations (half-face) instead of the standard 8,
// saving ~75% of SFPU cycles.
#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void sdpa_fw_recip_tile_first_column(uint32_t idst) {
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
// rows deep with row r in slot r. The row-range helpers (row0, n) and apply_exp_inplace_and_find_exp_sum
// (the whole block) leave the CB waits/reserves/pushes to the caller; matmul_qk_by_v does its own.
//
// TODO: these re-implement ttnn's SDPA helpers in
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:
//   update_row_max_values              -> reduce_c
//   apply_exp_inplace_and_find_exp_sum -> sub_exp_block_bcast_cols_inplace (incl. the L1-acc row sum)
//   update_exp_max_diffs               -> sub_exp_block
//   row_reduce_tiles_inplace           -> matmul_reduce
//   recip_tiles_inplace                -> recip_block_inplace
//   matmul_qk_by_v, the QK^T loop      -> matmul_blocks
// The causal mask stamp already comes from there (sdpa_mask_lightweight.hpp). Including the rest needs
// template flags for the training-only differences (accurate exp, fp32 stats, out-of-order packing for
// stacked heads, the sdpa_fw first-column reciprocal) and that header's REDUCE_OP / EXP_APPROX_MODE
// defines; tracked in https://github.com/tenstorrent/tt-metal/issues/63774.

// TODO: replace FPU reduce_tile(MAX) + sub_tiles_bcast_cols with SFPU equivalents once LLK
// provides SFPU row-reduce-max and sub_bcast_col. This will allow the full softmax (max, sub,
// exp, sum, reciprocal) to stay in DST at FP32 without pack/unpack round-trips through the CB.
//
// DST[i] = max(DST[i], cb_max[idx0 + i]) for i < n, column vectors, DST[n + i] as scratch: a running max
// folded with new maxes (the K/V loop's step max, a merger's partial). Inside the caller's DST pass. Not
// inlined and not cloned, like the other helpers the merge shares with the loop: their inits are text the
// kernel, close to its size cap, would otherwise carry once per call site (the compiler's constant-argument
// clones did the same).
__attribute__((noinline, noclone)) void fold_max_in_dst(uint32_t cb_max, uint32_t idx0, uint32_t n) {
    reconfig_data_format_srca(cb_max);
    copy_init(cb_max);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(cb_max, /* tile_idx */ idx0 + i, /* register idx */ n + i);
    }
    binary_max_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        binary_max_tile(i, n + i, i, VectorMode::C);
    }
}

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
        fold_max_in_dst(cb_prev_max, row0, n);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_to(cb_cur_max);
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
    // Fused scale+exp, exp(scale * (score - max)), in one SFPU pass per tile; the exp state is set once
    // here, as nothing below but the subtract (FPU) and the packs runs between the tiles.
    sdpa_exp_tile_scaled_init<scaler_fp32>();

    for (uint32_t t0 = 0; t0 < total; t0 += pass_tiles) {
        const uint32_t n = (total - t0 < pass_tiles) ? (total - t0) : pass_tiles;
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
        // L1-accumulate is off here: every user of it turns it off before it returns, and so does this
        // loop; toggling it again would stall the packer for nothing.
        pack_to(cb_attention_weights);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile</* out_of_order */ true>(i, cb_attention_weights, /* offset */ t0 + i);
        }

        // row sums: a row's first tile overwrites its slot, the rest add onto it
        pack_to(cb_cur_exp_sum);
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

// DST[i * block_w + j] = cb_in[(row0 + i) * Wt + col0 + j] * alpha[row0 + i] for a block_h x block_w block
// of tiles, at full fp32: cb_in unpacks straight to DST (UnpackToDestFp32) and the SFPU multiplies by the
// row's alpha, broadcast into the spare DST slot block_h * block_w. Used by the merge of split units (the
// K/V loop's rescale is the packer's L1-accumulate pass in matmul_qk_by_v); not inlined or cloned, since
// the unpacker inits it carries are most of its text and the compute kernel is close to its size cap.
__attribute__((noinline, noclone)) void scale_block_in_dst(
    uint32_t cb_in, uint32_t Wt, uint32_t row0, uint32_t col0, uint32_t block_h, uint32_t block_w, uint32_t cb_alpha) {
    const uint32_t alpha_dst = block_h * block_w;
    reconfig_data_format(cb_in, cb_in);
    copy_init(cb_in);
    for (uint32_t i = 0; i < block_h; ++i) {
        for (uint32_t j = 0; j < block_w; ++j) {
            copy_tile(cb_in, (row0 + i) * Wt + col0 + j, i * block_w + j);
        }
    }
    init_unary_bcast_col(cb_alpha);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < block_h; ++i) {
        unary_bcast<BroadcastType::COL>(cb_alpha, row0 + i, alpha_dst);
        for (uint32_t j = 0; j < block_w; ++j) {
            mul_binary_tile(i * block_w + j, alpha_dst, i * block_w + j);
        }
    }
}

// O_cur = P @ V (+ O_prev * alpha) for the whole item. P is rows x Sk_chunk_t tiles (row-major), V is
// Sk_chunk_t x Wt tiles (row-major, as read from DRAM). P @ V goes first, one block_h x block_w rectangle of
// output tiles per DST pass (the whole DST), packed to cb_cur_mm_out. With `rescale`, O_prev * alpha is then
// added onto it by the packer with L1-accumulate, as on main: rescale_block tiles of one row per DST pass,
// the previous accumulator loaded at full fp32 (UnpackToDestFp32), its row's alpha broadcast into the spare
// slot rescale_block, and the product formed on the SFPU. The packs of the two loops go through the same
// packer in order, so the accumulate always finds the matmul result in L1.
template <uint32_t Sk_chunk_t, uint32_t rows>
void matmul_qk_by_v(
    uint32_t Wt,
    uint32_t block_h,
    uint32_t block_w,
    uint32_t rescale_block,
    uint32_t cb_attention_weights,
    uint32_t cb_value,
    uint32_t cb_prev_mm_out,
    uint32_t cb_exp_max_diff,
    uint32_t cb_cur_mm_out,
    bool rescale) {
    cb_wait_front(cb_attention_weights, rows * Sk_chunk_t);
    cb_wait_front(cb_value, Sk_chunk_t * Wt);
    cb_reserve_back(cb_cur_mm_out, rows * Wt);

    // matmul maps: in0(attention_weights)->SrcB, in1(value)->SrcA
    reconfig_data_format(cb_value, cb_attention_weights);
    pack_to(cb_cur_mm_out);
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

    if (rescale) {
        cb_wait_front(cb_prev_mm_out, rows * Wt);
        cb_wait_front(cb_exp_max_diff, rows);
        reconfig_data_format(cb_prev_mm_out, cb_prev_mm_out);
        pack_reconfig_l1_acc(1);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t col0 = 0; col0 < Wt; col0 += rescale_block) {
                tile_regs_acquire();
                // DST[j] = O_prev[r][col0 + j] * alpha[r]; the copy and the broadcast reprogram the same
                // unpacker MOP, so each pass sets both up again.
                copy_init(cb_prev_mm_out);
                for (uint32_t j = 0; j < rescale_block; ++j) {
                    copy_tile(cb_prev_mm_out, r * Wt + col0 + j, j);
                }
                init_unary_bcast_col(cb_exp_max_diff);
                unary_bcast<BroadcastType::COL>(cb_exp_max_diff, r, rescale_block);
                mul_binary_tile_init();
                for (uint32_t j = 0; j < rescale_block; ++j) {
                    mul_binary_tile(j, rescale_block, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < rescale_block; ++j) {
                    pack_tile</* out_of_order */ true>(j, cb_cur_mm_out, /* offset */ r * Wt + col0 + j);
                }
                tile_regs_release();
            }
        }
        pack_reconfig_l1_acc(0);
    }
    cb_push_back(cb_cur_mm_out, rows * Wt);
}

// cb_exp_max_diff[out0 + i] = exp(scale * (prev_max[prev0 + i] - cur_max[cur0 + i])) for i < n, n <= 4 (one
// DST slot per row). The K/V loop passes the row for all three; a merger reads the partial's max from its
// slot in the partial row and packs one alpha at a time. One copy for all three call sites (see
// fold_max_in_dst).
template <uint32_t scaler_fp32, uint32_t scaler_bf16>
__attribute__((noinline, noclone)) void update_exp_max_diffs(
    uint32_t cb_prev_max_value,
    uint32_t prev0,
    uint32_t cb_cur_max_value,
    uint32_t cur0,
    uint32_t cb_exp_max_diff,
    uint32_t out0,
    uint32_t n) {
    reconfig_data_format(cb_prev_max_value, cb_cur_max_value);
    sub_init(cb_prev_max_value, cb_cur_max_value);
    sdpa_exp_tile_scaled_init<scaler_fp32>();
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        sub_tiles(cb_prev_max_value, cb_cur_max_value, /* tile_idx */ prev0 + i, /* tile_idx */ cur0 + i, i);
    }
    // First-column scaled exp: both max values are column vectors, so only column 0 has data and the
    // SFPU runs 4x fewer iterations than a full-tile exp.
    for (uint32_t i = 0; i < n; ++i) {
        sdpa_exp_tile_first_column_scaled<scaler_fp32, scaler_bf16>(i);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_to(cb_exp_max_diff);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_exp_max_diff, /* offset */ out0 + i);
    }
    tile_regs_release();
}

// cb_out[out0 + i] = sum_a[a0 + i] * alpha_a[a0 + i] + sum_b[b0 + i] [* alpha_b[a0 + i]] for i < n; the alphas
// sit at the row's slot. Two DST slots per row (the two terms), so n <= 2. The K/V loop runs it in place
// (cb_out = sum_b = the step's sums, no alpha_b); a merger adds the partial's sum scaled by its alpha.
template <bool scale_b>
void update_cur_exp_sums(
    uint32_t cb_sum_a,
    uint32_t cb_alpha_a,
    uint32_t a0,
    uint32_t cb_sum_b,
    uint32_t cb_alpha_b,
    uint32_t b0,
    uint32_t cb_out,
    uint32_t out0,
    uint32_t n) {
    reconfig_data_format(cb_sum_a, cb_alpha_a);  // reconfig data format to precise
    mul_bcast_cols_init(cb_sum_a, cb_alpha_a);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        mul_tiles_bcast_cols(cb_sum_a, cb_alpha_a, a0 + i, a0 + i, i);
    }
    if constexpr (scale_b) {
        reconfig_data_format(cb_sum_a, cb_sum_b, cb_alpha_a, cb_alpha_b);
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles_bcast_cols(cb_sum_b, cb_alpha_b, b0 + i, a0 + i, n + i);
        }
    } else {
        copy_init(cb_sum_b);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(cb_sum_b, /* tile_idx */ b0 + i, /* register idx */ n + i);
        }
    }
    add_binary_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        add_binary_tile(i, n + i, i);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_to(cb_out);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_out, /* offset */ out0 + i);
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
    pack_to(cb_in_idx);
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
        MATH((sdpa_fw_recip_tile_first_column(i)));
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_to(cb_in_idx);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile</* out_of_order */ true>(i, cb_in_idx, /* offset */ row0 + i);
    }
    tile_regs_release();
}

// Split units. A sender packs its running state, per row [max, unreduced sum, output tiles] (the layout
// in sdpa_item_math.hpp), into cb_partial_out (fp32) for the writer; a merger folds such a partial into
// its own running state, like one more K/V step: m = max(m_own, m_p), alpha_x = exp(scale * (m_x - m)),
// sum = sum_own * alpha_own + sum_p * alpha_p, O = O_own * alpha_own + O_p * alpha_p. The partials stream
// through the CBs one row (Wt + 2 tiles) at a time. The partial's tiles and the sums are multiplied on the
// FPU (19-bit inputs, once per partial); the merger's own fp32 output accumulator is scaled in DST so it is
// never truncated.
// A piece whose first step lies past a row's diagonal (Sq_chunk_t > 2 * Sk_chunk_t, or a piece of the
// first chunks) sees only masked scores on that row. The stamp adds the bf16 mask value (-998244352) in
// fp32, where a score of a few hundred survives as the low bits; the row max then rounds to the mask value
// in the bf16 max CB, so the row's P is exp(scale * score), not exp(0): finite garbage of up to exp(scale *
// |score|) per element. The fold removes it exactly, since alpha = exp(scale * (mask - real max))
// underflows to 0. The whole path never does this, since every row has K tile 0 unmasked at step 0.
// Measured finite on Blackhole with raw scores in the thousands (SplitEdge_LargeScoresStayFinite).
// ttnn has this fold twice: ring_joint's K split (merge_into_prev in ring_joint_sdpa.cpp) and flash decode's
// correction_block (compute_common.hpp). Neither fits here. ring_joint's runs on its bf16 streaming path
// only (it rejects fp32_dest_acc_en, and this kernel keeps the fp32 accumulator in DST), its senders hold
// one unit and keep their state in L1 for the reducer to pull after a signal, as three whole-item blocks
// (sum, max, out CBs); here a sender's core goes on to whole items and mergers, so the state leaves L1 for
// DRAM scratch, and the merger streams it back a row at a time, [max, sum, O] in one CB, because a whole
// item's fp32 partial does not fit the L1 the pick leaves. correction_block folds max and sum only, from
// four CBs indexed alike per row; the partial's tiles sit at fixed slots of one row. The fold below is built
// from the K/V loop's own helpers instead, so a later shared helper has one local copy to replace.
#ifdef SPLIT_UNITS
// dst_tiles: output tiles per DST pass when copying the output tiles through.
template <uint32_t rows, uint32_t dst_tiles>
void send_partial(uint32_t Wt, uint32_t cb_mm_out, uint32_t cb_max, uint32_t cb_sum_exp, uint32_t cb_partial_out) {
    const uint32_t tiles_per_row = sdpa_items::partial_row_tiles(Wt);
    cb_wait_front(cb_mm_out, rows * Wt);
    cb_wait_front(cb_max, rows);
    cb_wait_front(cb_sum_exp, rows);
    pack_to(cb_partial_out);
    for (uint32_t r = 0; r < rows; ++r) {
        cb_reserve_back(cb_partial_out, tiles_per_row);
        // [max, sum, O_0, O_1], then the remaining O tiles dst_tiles at a time
        tile_regs_acquire();
        reconfig_data_format_srca(cb_max);
        copy_init(cb_max);
        copy_tile(cb_max, r, 0);
        reconfig_data_format_srca(cb_sum_exp);
        copy_init(cb_sum_exp);
        copy_tile(cb_sum_exp, r, 1);
        // cb_mm_out unpacks straight to DST (UnpackToDestFp32), so its unpack format differs from the
        // sum's even though both CBs are fp32: the reconfig is needed.
        reconfig_data_format_srca(cb_mm_out);
        copy_init(cb_mm_out);
        const uint32_t first = (Wt < 2U) ? Wt : 2U;
        for (uint32_t t = 0; t < first; ++t) {
            copy_tile(cb_mm_out, r * Wt + t, 2U + t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < 2U + first; ++i) {
            pack_tile(i, cb_partial_out);
        }
        tile_regs_release();
        for (uint32_t t0 = first; t0 < Wt; t0 += dst_tiles) {
            const uint32_t n = (Wt - t0 < dst_tiles) ? (Wt - t0) : dst_tiles;
            tile_regs_acquire();
            copy_init(cb_mm_out);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_mm_out, r * Wt + t0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_partial_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_partial_out, tiles_per_row);
    }
    cb_pop_front(cb_mm_out, rows * Wt);
    cb_pop_front(cb_max, rows);
    cb_pop_front(cb_sum_exp, rows);
}

// block_size: output tiles per DST pass (divides Wt, leaves one DST slot for alpha).
template <uint32_t scaler_fp32, uint32_t scaler_bf16, uint32_t rows, uint32_t block_size>
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
    const uint32_t tiles_per_row = sdpa_items::partial_row_tiles(Wt);
    using sdpa_items::kPartialFirstOutTile;
    using sdpa_items::kPartialMaxTile;
    using sdpa_items::kPartialSumTile;
    cb_wait_front(cb_prev_mm_out, rows * Wt);
    cb_wait_front(cb_prev_max, rows);
    cb_wait_front(cb_prev_sum_exp, rows);
    cb_reserve_back(cb_cur_mm_out, rows * Wt);  // packed by offset, row after row

    for (uint32_t r = 0; r < rows; ++r) {
        cb_wait_front(cb_partial_in, tiles_per_row);

        // (1) m = max(m_own, m_p) -> cb_cur_max[r], then alpha_own = exp(scale * (m_own - m)) and
        //     alpha_p = exp(scale * (m_p - m)), the K/V loop's max fold and alpha step with the partial's max
        //     read from its slot in the partial row. The maxes are bf16 values, so the FPU subtraction is
        //     exact. Each CB is pushed a row at a time: the row's tiles are read back within the iteration.
        tile_regs_acquire();
        reconfig_data_format_srca(cb_prev_max);
        copy_init(cb_prev_max);
        copy_tile(cb_prev_max, r, 0);
        fold_max_in_dst(cb_partial_in, kPartialMaxTile, 1U);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_cur_max, 1);
        pack_to(cb_cur_max);
        pack_tile(0, cb_cur_max);
        cb_push_back(cb_cur_max, 1);
        tile_regs_release();
        cb_wait_front(cb_cur_max, r + 1U);
        cb_reserve_back(cb_alpha_own, 1);
        update_exp_max_diffs<scaler_fp32, scaler_bf16>(cb_prev_max, r, cb_cur_max, r, cb_alpha_own, 0U, 1U);
        cb_push_back(cb_alpha_own, 1);
        cb_reserve_back(cb_alpha_p, 1);
        update_exp_max_diffs<scaler_fp32, scaler_bf16>(
            cb_partial_in, kPartialMaxTile, cb_cur_max, r, cb_alpha_p, 0U, 1U);
        cb_push_back(cb_alpha_p, 1);
        cb_wait_front(cb_alpha_own, r + 1U);
        cb_wait_front(cb_alpha_p, r + 1U);

        // (2) sum = sum_own * alpha_own + sum_p * alpha_p on the FPU (the K/V loop's sum update with the
        //     partial's sum as the scaled second term), then the partial's output tiles times alpha_p,
        //     packed into the output row.
        cb_reserve_back(cb_cur_sum_exp, 1);
        update_cur_exp_sums</* scale_b */ true>(
            cb_prev_sum_exp, cb_alpha_own, r, cb_partial_in, cb_alpha_p, kPartialSumTile, cb_cur_sum_exp, 0U, 1U);
        cb_push_back(cb_cur_sum_exp, 1);
        pack_to(cb_cur_mm_out);  // the unpacker keeps the config above: all four CBs are fp32
        // The sum update above ends on an SFPU add; the broadcast multiply below gets its own init rather than
        // relying on the FPU program the sum update left behind.
        mul_bcast_cols_init(cb_partial_in, cb_alpha_p);
        for (uint32_t t0 = 0; t0 < Wt; t0 += block_size) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_size; ++i) {
                mul_tiles_bcast_cols(cb_partial_in, cb_alpha_p, kPartialFirstOutTile + t0 + i, r, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size; ++i) {
                pack_tile</* out_of_order */ true>(i, cb_cur_mm_out, r * Wt + t0 + i);
            }
            tile_regs_release();
        }
        // (3) O += O_own * alpha_own in DST: the merger's accumulator is fp32 and must not pass through
        //     the FPU's 19-bit inputs on every fold, so it is scaled in DST like the in-loop rescale and
        //     the packer adds it onto (2) in L1.
        for (uint32_t t0 = 0; t0 < Wt; t0 += block_size) {
            tile_regs_acquire();
            scale_block_in_dst(cb_prev_mm_out, Wt, r, t0, /* block_h */ 1U, block_size, cb_alpha_own);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(1U);
            for (uint32_t i = 0; i < block_size; ++i) {
                pack_tile</* out_of_order */ true>(i, cb_cur_mm_out, r * Wt + t0 + i);
            }
            pack_reconfig_l1_acc(0U);
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
    pack_to(cb_out_lse);
    pack_tile</* out_of_order */ true>(lse_reg, cb_out_lse, /* offset */ row);
    tile_regs_release();
}
