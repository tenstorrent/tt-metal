// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// [sdpa_prefill_splitk] Split-K causal chunked-prefill attention over a paged prefix. Each core walks its list of
// work pieces (Q head, Sq_chunk_t-tile Q chunk, K-chunk range [k_lo, k_hi)) and emits per piece the
// softmax-normalised partial output and the row statistics: m = row max of the raw scores (the scale is folded into
// the exponentials, as in the stock op) and l = sum_j exp((s_j - m) * scale) over the piece's keys. The per K chunk
// body is the stock sdpa_inner_loop STANDARD body (ttnn .../sdpa/device/kernels/compute/compute_common.hpp) with
// absolute Q / K tile coordinates for the lightweight causal diagonal stamp; the finalisation adds the statistics
// output before the stock normalisation.

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

// copy num_tiles tiles of in_cb to out_cb; pop_input = false keeps them fronted in in_cb (l is copied out, then used
// for the normalisation).
template <bool pop_input>
void copy_stats(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    reconfig_data_format_srca(in_cb);
    copy_init(in_cb);
    pack_reconfig_data_format(out_cb);
    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0 /*dst*/);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
    if constexpr (pop_input) {
        cb_in.pop_front(num_tiles);
    }
}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(11);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(14);

    constexpr uint32_t cb_q_in = get_compile_time_arg_val(15);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(16);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(17);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(18);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(19);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(20);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(21);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(22);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(23);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(24);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(25);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(26);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(27);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(28);
    constexpr uint32_t cb_out = get_compile_time_arg_val(29);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(30);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;

    uint32_t argidx = 0;
    const uint32_t num_pieces = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_base_tile = get_arg_val<uint32_t>(argidx++);  // absolute tile row of Q row 0 (chunk_start / 32)

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_out);
    matmul_init(cb_q_in, cb_k_in);

    // [neginf, causal_diag] from the writer, fronted for the whole program (stock lightweight causal mask)
    LightweightMaskContext lw_mask;
    lw_mask.is_causal = true;
    lw_mask.neginf_tile_idx = 0;
    lw_mask.causal_diag_tile_idx = 1;
    lw_mask.primary_diag_tile_idx = 1;
    CircularBuffer(cb_mask_in).wait_front(2);

    for (uint32_t piece = 0; piece < num_pieces; ++piece) {
        const uint32_t q_chunk = get_arg_val<uint32_t>(argidx++);
        const uint32_t k_lo = get_arg_val<uint32_t>(argidx++);
        const uint32_t k_hi = get_arg_val<uint32_t>(argidx++);
        const uint32_t q_start_tile = q_base_tile + q_chunk * Sq_chunk_t;

        uint32_t alias_prev_sum = cb_sum_A;
        uint32_t alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A;
        uint32_t alias_cur_max = cb_max_B;
        uint32_t alias_mm2_prev_out = cb_out_im_A;
        uint32_t alias_mm2_cur_out = cb_out_im_B;
        uint32_t processed_k_chunks = 0;

        for (uint32_t k_chunk = k_lo; k_chunk < k_hi; ++k_chunk) {
            /* QK = Q_CHUNK @ K_CHUNK (stock STANDARD body from here to the swaps) */
            reconfig_data_format(cb_k_in, cb_q_in);
            pack_reconfig_data_format(cb_qk_im);
            matmul_blocks(
                cb_q_in,
                cb_k_in,
                cb_qk_im,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                qk_in0_num_subblocks,
                qk_in1_num_subblocks,
                qk_in0_block_w,
                qk_subblock_h,
                qk_subblock_w,
                true /*transpose*/);

            // [sdpa_prefill_splitk] absolute K tile of this chunk: the diagonal stamp lines up with q_start_tile
            const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
            const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;
            if (q_start_tile < k_high_idx) {
                reconfig_data_format(cb_qk_im, cb_mask_in);
                CircularBuffer cb_qk_im_obj(cb_qk_im);
                cb_qk_im_obj.wait_front(qk_chunk_tiles);
                cb_qk_im_obj.pop_front(qk_chunk_tiles);
                cb_qk_im_obj.reserve_back(qk_chunk_tiles);
                lw_mask.template apply<dst_size>(
                    cb_mask_in,
                    cb_qk_im,
                    Sk_chunk_t,
                    Sq_chunk_t,
                    k_chunk,
                    0,
                    false,
                    false,
                    false,
                    0,
                    0,
                    0,
                    q_start_tile,
                    k_low_idx,
                    0,
                    0);
                cb_qk_im_obj.push_back(qk_chunk_tiles);
            }

            reconfig_data_format(cb_qk_im, cb_identity_scale_in);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(
                alias_cur_max, alias_prev_max, Sk_chunk_t, processed_k_chunks > 0);

            sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(
                alias_cur_max, alias_cur_sum, Sk_chunk_t);

            reconfig_data_format(cb_v_in, cb_qk_im);
            pack_reconfig_data_format(alias_mm2_cur_out);

            /* OUT_IM = QK @ V_CHUNK */
            matmul_blocks(
                cb_qk_im,
                cb_v_in,
                alias_mm2_cur_out,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w,
                out_subblock_h,
                out_subblock_w,
                false /*transpose*/);

            CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles);
            reconfig_data_format(alias_prev_max, alias_cur_max);

            if (processed_k_chunks > 0) {
                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);
                mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);
                mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
            }

            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);
            processed_k_chunks++;
        }

        /* Final row reduction of the partial sum (stock) */
        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

        // [sdpa_prefill_splitk] statistics: Sq_chunk_t tiles of m, then Sq_chunk_t tiles of l (column 0 holds the row
        // values); m's CB is released by the copy, l stays for the normalisation.
        copy_stats<true>(alias_prev_max, cb_stats, Sq_chunk_t);
        copy_stats<false>(alias_prev_sum, cb_stats, Sq_chunk_t);

        /* cb_cur_sum = 1.0 / cb_cur_sum; out = out_acc * cb_cur_sum (stock) */
        recip_block_inplace(alias_prev_sum, Sq_chunk_t);
        pack_reconfig_data_format(cb_out);
        mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);

        CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
    }
}
