// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/untilize.h"
#include "../rt_args_common.hpp"
#include "compute_common.hpp"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;

namespace NAMESPACE {

void MAIN {
    // Compile time arguments

    // Input dimensions in tiles
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);

    // Matmul configs
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(13);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(16);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(17);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(18);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(19);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(20);

    // Attention-specific parameters (simplified for MLA)
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(21);
    constexpr uint32_t q_tile_height = get_compile_time_arg_val(22);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(23);
    constexpr uint32_t num_tree_reduction_steps =
        get_compile_time_arg_val(24);  // tree reduction steps (3 for 8 S blocks)

    // MLA decode is always causal, no attention mask, no attention sink, no sliding window
    constexpr bool is_causal = true;
    constexpr bool use_attention_mask = false;
    constexpr bool use_attention_sink = false;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // CB index definitions
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_accumulate_im = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_1 = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_2 = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_1 = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_2 = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_21;
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;

    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    // Runtime arguments
    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const bool is_sender_after_reduce =
        get_arg_val<uint32_t>(arg_idx++) == 1;  // Intermediate nodes write to output CBs after reduction
    // Tree reduction info: 3 steps × 2 values (role, partner_s_block_idx) = 6 values
    tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_tree_reduction_steps * 2;

    // Idle core
    // get_arg_val<uint32_t>(0) can go from 0-63 for the core_num; for active cores 65 is out of range so 65 indicates
    // an idle_core
    if (get_arg_val<uint32_t>(0) == 65) {
        return;
    }

    // Get cur_pos from position tensor (MLA decode is always causal)
    uint32_t cur_pos;
    if constexpr (is_causal) {
        constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
        cb_wait_front(cb_index_id, 1);
        cur_pos = read_tile_value(cb_index_id, 0, cur_batch / q_heads_parallel_factor);
        cb_pop_front(cb_index_id, 1);
    } else {
        // Non-causal: use full sequence length
        constexpr uint32_t cur_pos_base = St * 32 - 1;
        cur_pos = cur_pos_base;
    }

    // Get the sequence length assignment (no sliding window for MLA)
    auto [k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size);
    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    // Calculate number of active S blocks based on k_num_chunks
    // With strided distribution, core N gets chunks N, N+num_cores, N+2*num_cores, ...
    // So active cores = min(k_num_chunks, num_cores_per_head)
    uint32_t num_active_s_blocks = (k_num_chunks < num_cores_per_head) ? k_num_chunks : num_cores_per_head;

    // Tree reduction: count actual reductions (only where partner is active)
    // role_code: 0=idle, 1=sender, 2=receiver
    uint32_t num_cores_to_wait = 0;
    for (uint32_t step = 0; step < num_tree_reduction_steps; ++step) {
        uint32_t role_code = tree_reduction_info[step * 2 + 0];
        uint32_t partner_s_block_idx = tree_reduction_info[step * 2 + 1];
        // Count this step only if we're a receiver AND partner is active
        if (role_code == 2 && partner_s_block_idx < num_active_s_blocks) {
            num_cores_to_wait++;
        }
    }

    // We tilize input Q if it is in ROW MAJOR layout
    // Q is always tilized (TILE_LAYOUT)
    mm_init(cb_q_in, cb_k_in, cb_qk_im);
    cb_wait_front(cb_q_in, q_chunk_tiles);

    // Use compile-time matmul configs (dynamic chunk size removed)
    constexpr uint32_t qk_subblock_h_dynamic = qk_subblock_h;
    constexpr uint32_t qk_subblock_w_dynamic = qk_subblock_w;
    constexpr uint32_t qk_in0_num_subblocks_dynamic = qk_in0_num_subblocks;
    constexpr uint32_t qk_in1_num_subblocks_dynamic = qk_in1_num_subblocks;
    constexpr uint32_t out_in0_block_w_dynamic = out_in0_block_w;
    constexpr uint32_t out_num_blocks_dynamic = out_num_blocks;
    constexpr uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t;

    // TODO: Used for legacy sfpu functions
    // - VectorMode::RC is equivalent to 32x32 tiles
    // - VectorMode::R is equivalent to 16x32 tiles (also works for smaller tile heights like 8x32)
    // NOTE: Using VectorMode::RC for smaller tiles will be correct accuracy, just slower due to unnecessary math
    constexpr int vector_mode = (q_tile_height < 32) ? VectorMode::R : VectorMode::RC;

    // We set up Ping Pong intermediate buffers between loops
    uint32_t cb_cur_max = cb_max_1;
    uint32_t cb_prev_max = cb_max_2;
    uint32_t cb_cur_sum = cb_sum_1;
    uint32_t cb_prev_sum = cb_sum_2;

    // Loop through all heads assigned to core
    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        /******************************************************************************
         *                           FLASH ATTENTION LOOP                             *
         ******************************************************************************/
        /**
         * Compute Parameters (most are compile time but some are dynamic):
         * @tparam St - Total sequence length in tiles
         * @tparam DHt - Head dimension in tiles
         * @tparam Sq_chunk_t - Query chunk size in tiles
         * @tparam Sk_chunk_t - Key chunk size in tiles (dynamic)
         * @tparam qk_in0_block_w - QK matmul block width
         * @tparam qk_subblock_w - QK matmul subblock width (dynamic)
         * @tparam qk_subblock_h - QK matmul subblock height (dynamic)
         * @tparam qk_in0_num_subblocks - QK input0 subblocks (dynamic)
         * @tparam qk_in1_num_subblocks - QK input1 subblocks (dynamic)
         * @tparam qk_num_blocks - QK number of blocks
         * @tparam out_in0_block_w - Output matmul block width (dynamic)
         * @tparam out_subblock_w - Output matmul subblock width
         * @tparam out_subblock_h - Output matmul subblock height
         * @tparam out_in0_num_subblocks - Output input0 subblocks
         * @tparam out_in1_num_subblocks - Output input1 subblocks
         * @tparam out_num_blocks - Output number of blocks (dynamic)
         * @tparam is_causal - Whether to use causal attention (if mask is applied)
         * @tparam use_attention_mask - Whether to use attention mask for non-causal attention
         *
         * Circular Buffer Parameters:
         * @tparam cb_q_in - Query input buffer
         * @tparam cb_k_in - Key input buffer
         * @tparam cb_v_in - Value input buffer
         * @tparam cb_mask_in - Mask input buffer
         * @tparam cb_scale_in - Scale input buffer
         * @tparam cb_identity_scale_in - Identity scale buffer
         * @tparam cb_qk_im - QK intermediate buffer
         * @tparam cb_out_im - Output intermediate buffer
         * @tparam cb_out_accumulate_im - Output accumulate buffer
         * @tparam cb_cur_max - Current max buffer
         * @tparam cb_prev_max - Previous max buffer
         * @tparam cb_cur_sum - Current sum buffer
         * @tparam cb_prev_sum - Previous sum buffer
         * @tparam cb_exp_max_diff - Exp max diff buffer
         * @tparam cb_out_o - Output O buffer
         * @tparam cb_out_m - Output M buffer
         * @tparam cb_out_l - Output L buffer
         *
         * Runtime Parameters:
         * @param k_chunk_start - Start index of key chunk
         * @param k_chunk_end - End index of key chunk
         * @param do_reduce - Whether to perform reduction
         * @param qk_chunk_tiles - Number of QK chunk tiles (dynamic)
         * @param out_chunk_tiles - Number of output chunk tiles
         */

        /* START OF FLASH ATTENTION LOOP */
        {
            uint32_t cb_out_mm = cb_out_accumulate_im;

            // Loop through all K chunks (strided iteration for DRAM bank locality)
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += num_cores_per_head) {
                // Reconfig register DF
                reconfig_data_format(cb_q_in, cb_k_in);
                pack_reconfig_data_format(cb_qk_im);

                // OPTIMIZATION: Add the attention mask directly on top of DST if chunk sizes are dynamic
#ifdef DYNAMIC_CHUNK_SIZE
                bool add_causal_mask_fusion = is_causal && k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk;
                bool add_mask_fusion = add_causal_mask_fusion || use_attention_mask;
#else
                bool add_mask_fusion = false;
                bool add_causal_mask_fusion = false;
#endif

                /* QK = Q_CHUNK @ K_CHUNK */
                /* NOTE: skip_in1_pop=true because K buffer is reused for V matmul */

                cb_matmul_blocks(
                    cb_q_in,
                    cb_k_in,
                    cb_qk_im,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    qk_num_blocks,
                    qk_in0_num_subblocks_dynamic,
                    qk_in1_num_subblocks_dynamic,
                    qk_in0_block_w,
                    qk_subblock_h_dynamic,
                    qk_subblock_w_dynamic,
                    true,
                    add_mask_fusion,
                    cb_mask_in,
                    cb_zero_in,
                    true /* skip_in1_pop: K buffer reused for V matmul */);

                /* QK += MASK */
                if (!add_mask_fusion) {
                    if constexpr (is_causal) {
                        // For decode, we only apply mask at the last chunk for causal mode
                        if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    } else {
                        if constexpr (use_attention_mask) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    }
                }

                /**
                 * OPTIMIZATION
                 * Typically, scores are multiplied by a scalar here, but an optimization was employed
                 * where the scaling is fused into exp both in exp(x - max) and exp(prev_max - cur_max).
                 * This gives us scaling for free on the performance-critical exp(x - max) computation.
                 */

                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_max);

                /**
                 * OPTIMIZATION
                 * reduce_c can perform both reduce_max and eltwise max with previous result.
                 * if do_eltwise_max:
                 *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 * else:
                 *  cur_max = max(qk, dim=-1)
                 */
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_max, cb_prev_max, Sk_chunk_t, k_chunk > k_chunk_start);

                /* QK -= cb_cur_max */
                /* QK = exp(QK)*/
                reconfig_data_format(cb_qk_im, cb_cur_max);
                pack_reconfig_data_format(cb_qk_im);

                /**
                 * sub_exp performs `QK = exp((QK - cur_max) * scale)`
                 */
                sub_exp_block_bcast_cols_inplace_reduce<
                    cb_qk_im,
                    Sq_chunk_t,
                    scale_fp32,
                    vector_mode,
                    cb_identity_scale_in>(cb_cur_max, cb_cur_sum, Sk_chunk_t);
                cb_wait_front(cb_qk_im, qk_chunk_tiles_dynamic);

                // Reconfig register DF
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);

                /* reduce_c performs CUR_SUM = sum(QK, dim = -1) */
                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t, false);

                /* OUT_IM = QK @ V_CHUNK
                 * MLA optimization: V is the first vDHt columns of K (which has DHt columns).
                 * Use strided matmul to read V directly from cb_k_in, skipping extra columns.
                 */
                reconfig_data_format(cb_qk_im, cb_k_in);
                pack_reconfig_data_format(cb_out_im);
                cb_matmul_blocks_strided(
                    cb_qk_im,
                    cb_k_in,  // Read V from K buffer directly
                    cb_out_mm,
                    Sq_chunk_t,
                    vDHt,  // Output width (V columns)
                    Sk_chunk_t,
                    DHt,  // in1_row_stride: actual K row width
                    out_num_blocks_dynamic,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_in0_block_w_dynamic,
                    out_subblock_h,
                    out_subblock_w,
                    true,   // skip_in1_wait: K already in CB from Q@K matmul
                    true);  // skip_in1_pop: we pop K manually below after both matmuls

                // Pop K buffer now that both Q@K and QK@V matmuls are complete
                cb_pop_front(cb_k_in, Sk_chunk_t * DHt);

                // Reconfig register DF
                reconfig_data_format_srca(cb_out_im);
                cb_pop_front(cb_qk_im, qk_chunk_tiles_dynamic);

                /* OUT_ACC += OUT_IM */
                if (k_chunk == k_chunk_start) {
                    cb_out_mm = cb_out_im;
                } else {
                    // When there is more than 1 chunk, we perform Lazy Softmax

                    // Reconfig register DF
                    reconfig_data_format(cb_prev_max, cb_cur_max);
                    pack_reconfig_data_format(cb_exp_max_diff);

                    /* EXP_MAX_DIFF = exp(PREV_MAX - CUR_MAX) */
                    sub_exp_block<scale_fp32, vector_mode>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    cb_pop_front(cb_prev_max, Sq_chunk_t);

                    /* PREV_SUM *= EXP_MAX_DIFF */
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    /* OUT_ACC *= EXP_MAX_DIFF */
                    reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    mul_block_bcast_cols(cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im, Sq_chunk_t, vDHt);

                    /* CUR_SUM += PREV_SUM */
                    reconfig_data_format(cb_cur_sum, cb_prev_sum);
                    pack_reconfig_data_format(cb_cur_sum);
                    add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                    /* OUT_ACC += OUT_IM */
                    reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                }

                if (k_chunk < k_chunk_end - 1 || do_reduce) {
                    // Move intermediate sum and max values to appropriate ping pong buffers
                    reconfig_data_format(cb_cur_max, cb_cur_max);
                    pack_reconfig_data_format(cb_prev_max);

                    // PREV_MAX <- CUR_MAX
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);

                    // PREV_SUM <- CUR_SUM
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                } else {
                    // Write results OUT_ACC, CUR_MAX, CUR_SUM to designated
                    // Write o, m, l into cb_out
                    move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
                    move_block<true>(cb_cur_max, cb_out_m, Sq_chunk_t);
                    move_block<true>(cb_cur_sum, cb_out_l, Sq_chunk_t);
                }
            }
        }
        /* END OF FLASH ATTENTION LOOP */

        // Perform tree reduction across intermediates from other cores
        // Tree reduction: this core receives from num_cores_to_wait partners across multiple steps
        if (do_reduce) {
            // cb_out_accumulate_im should contain o_1 (output from FA of itself's core)
            // cb_prev_max and cb_prev_sum should contain m_1 and l_1 (max and sum of logits of itself's core)

            if (num_cores_to_wait > 0) {
                // Tree reduction: perform num_cores_to_wait reduction steps
                // Writer kernel pushes data from each sender as it arrives
                for (uint32_t i = 0; i < num_cores_to_wait; i++) {
                    move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);

                    // Fused Softmax Correction
                    // * Fused Correction is a fused operation that performs the following steps:
                    // * 1. CUR_MAX = max(PREV_MAX, WORKER_MAX)
                    // * 2. EXP_MAX_DIFF_2 = exp((WORKER_MAX - CUR_MAX)*scale)
                    // * 3. PREV_SUM_2 *= EXP_MAX_DIFF_2
                    // * 4. EXP_MAX_DIFF = exp((PREV_MAX - CUR_MAX)*scale)
                    // * 5. PREV_SUM *= EXP_MAX_DIFF
                    // * 6. CUR_SUM = PREV_SUM_2 + PREV_SUM
                    // */
                    correction_block<scale_fp32, vector_mode>(
                        cb_m_in,        // cb worker max
                        cb_prev_sum_2,  // cb worker sum
                        cb_cur_max,
                        cb_prev_max,
                        cb_cur_sum,
                        cb_prev_sum,
                        cb_exp_max_diff,
                        cb_exp_max_diff_2,
                        Sq_chunk_t);

                    // OUT_ACC_2 <- WORKER_OUT
                    move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);

                    // OUT_ACC_2 *= EXP_MAX_DIFF
                    // OUT_ACC *= EXP_MAX_DIFF_2
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, vDHt);
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im_2, cb_exp_max_diff_2, Sq_chunk_t, vDHt);

                    // OUT_ACC = OUT_ACC + OUT_ACC_2
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);

                    // PREV_MAX <- CUR_MAX
                    // PREV_SUM <- CUR_SUM
                    cb_pop_front(cb_prev_max, Sq_chunk_t);
                    cb_pop_front(cb_m_in, Sq_chunk_t);
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }
            }

            // For intermediate nodes (receivers that also send), write to output CBs
            // before final normalization so the next receiver can reduce correctly
            if (is_sender_after_reduce) {
                // Write unnormalized result to output CBs for writer to send
                // cb_out_accumulate_im has output, cb_prev_max has max, cb_prev_sum has sum
                move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
                move_block<true>(cb_prev_max, cb_out_m, Sq_chunk_t);
                move_block<true>(cb_prev_sum, cb_out_l, Sq_chunk_t);
                // Intermediate nodes exit after writing - don't do final normalization
                return;
            }

            /* CUR_SUM = 1.0 / CUR_SUM */
            cb_push_back(cb_cur_sum, Sq_chunk_t);
            reconfig_data_format(cb_cur_sum, cb_cur_sum);
            pack_reconfig_data_format(cb_cur_sum);

            // Handle attention sink here
            if constexpr (use_attention_sink) {
                // m_new
                max_block<vector_mode>(cb_attention_sink, cb_prev_max, cb_cur_max, Sq_chunk_t);

                // exp(m - m_new)
                sub_exp_block<scale_fp32, vector_mode>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);

                // l -> l * exp(m - m_new)
                mul_block_inplace(cb_cur_sum, cb_exp_max_diff, Sq_chunk_t);

                // exp(sink - m_new)
                sub_exp_block<scale_fp32, vector_mode>(cb_attention_sink, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                cb_pop_front(cb_cur_max, Sq_chunk_t);

                // l -> l + exp(sink - m_new)
                add_block_inplace<true>(cb_cur_sum, cb_exp_max_diff_2, Sq_chunk_t);

                // O -> O * exp(m - m_new)
                mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, vDHt);
            }

            reconfig_data_format(cb_cur_sum, cb_cur_sum);
            pack_reconfig_data_format(cb_cur_sum);
            recip_block_inplace<vector_mode>(cb_cur_sum, Sq_chunk_t);

            /* OUT_ACC *= CUR_SUM */
            reconfig_data_format(cb_out_accumulate_im, cb_cur_sum);
            pack_reconfig_data_format(cb_out_accumulate_im);

            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, vDHt);
            pack_reconfig_data_format(cb_out_final);

            // Move output to buffer for the writer (Q is always tilized, so output is also tilized)
            move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
            // Free up cb_prev_max after K chunks
            cb_pop_front(cb_prev_max, Sq_chunk_t);
            cb_pop_front(cb_prev_sum, Sq_chunk_t);
        }
    }

    // Free up cb_q_in after Q chunks
    cb_pop_front(cb_q_in, q_chunk_tiles);
}
}  // namespace NAMESPACE
