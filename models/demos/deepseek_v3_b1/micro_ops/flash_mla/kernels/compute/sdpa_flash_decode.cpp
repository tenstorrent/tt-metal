// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "../rt_args_common.hpp"
#include "compute_common.hpp"

void kernel_main() {
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
    // 17: num_cores_per_batch (unused by compute)
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(18);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(19);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(20);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(21);
    // 22: q_tile_height (unused by compute)
    // 23: scale_fp32 (unused by simplified compute)
    constexpr uint32_t num_tree_reduction_steps = get_compile_time_arg_val(24);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(25);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(26);
    constexpr uint32_t cb_ms_in = get_compile_time_arg_val(27);
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(28);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(29);
    constexpr uint32_t cb_out_im = get_compile_time_arg_val(30);
    constexpr uint32_t cb_out_accumulate_im = get_compile_time_arg_val(31);
    constexpr uint32_t cb_out_o = get_compile_time_arg_val(32);
    constexpr uint32_t cb_out_ms = get_compile_time_arg_val(33);
    constexpr uint32_t cb_out_final = get_compile_time_arg_val(34);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;

    // Runtime arguments
    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const bool is_sender_after_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_tree_reduction_steps * 2;

    // Idle core check
    if (get_arg_val<uint32_t>(0) == 65) {
        return;
    }

    // Get cur_pos from position tensor (MLA decode is always causal)
    uint32_t cur_pos;
    {
        cb_wait_front(cb_index_id, 1);
        cur_pos = read_tile_value(cb_index_id, 0, cur_batch / q_heads_parallel_factor);
        cb_pop_front(cb_index_id, 1);
    }

    // Get the sequence length assignment
    auto [k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size);
    if (k_chunk_start == k_chunk_end) {
        return;
    }

    // Calculate number of active S blocks for tree reduction
    uint32_t num_active_s_blocks = (k_num_chunks < num_cores_per_head) ? k_num_chunks : num_cores_per_head;

    // Count actual tree reductions (only where partner is active)
    uint32_t num_cores_to_wait = 0;
    for (uint32_t step = 0; step < num_tree_reduction_steps; ++step) {
        uint32_t role_code = tree_reduction_info[step * 2 + 0];
        uint32_t partner_s_block_idx = tree_reduction_info[step * 2 + 1];
        if (role_code == 2 && partner_s_block_idx < num_active_s_blocks) {
            num_cores_to_wait++;
        }
    }

    // Init matmul and wait for Q
    mm_init(cb_q_in, cb_k_in, cb_qk_im);
    cb_wait_front(cb_q_in, q_chunk_tiles);

    // Loop through all heads assigned to core
    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        /* SIMPLIFIED COMPUTE: QK matmul → QK@V matmul → accumulate (no softmax) */
        {
            uint32_t cb_out_mm = cb_out_accumulate_im;

            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += num_cores_per_head) {
                /* QK = Q @ K */
                reconfig_data_format(cb_q_in, cb_k_in);
                pack_reconfig_data_format(cb_qk_im);

                cb_matmul_blocks(
                    cb_q_in,
                    cb_k_in,
                    cb_qk_im,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    qk_num_blocks,
                    qk_in0_num_subblocks,
                    qk_in1_num_subblocks,
                    qk_in0_block_w,
                    qk_subblock_h,
                    qk_subblock_w,
                    true,   // transpose
                    false,  // no mask fusion
                    0,
                    0,      // unused mask/zero CBs
                    true);  // skip_in1_pop: K buffer reused for V matmul

                /* OUT_IM = QK @ V (strided matmul, V from K buffer) */
                reconfig_data_format(cb_qk_im, cb_k_in);
                pack_reconfig_data_format(cb_out_im);
                cb_matmul_blocks_strided(
                    cb_qk_im,
                    cb_k_in,
                    cb_out_mm,
                    Sq_chunk_t,
                    vDHt,
                    Sk_chunk_t,
                    DHt,
                    out_num_blocks,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_in0_block_w,
                    out_subblock_h,
                    out_subblock_w,
                    true,
                    true);  // skip_in1_wait, skip_in1_pop

                // Pop K and QK buffers
                cb_pop_front(cb_k_in, Sk_chunk_t * DHt);
                reconfig_data_format_srca(cb_out_im);
                cb_pop_front(cb_qk_im, qk_chunk_tiles);

                /* Accumulate: simple addition (no softmax) */
                if (k_chunk == k_chunk_start) {
                    cb_out_mm = cb_out_im;
                } else {
                    reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                }

                if (!(k_chunk < k_chunk_end - 1 || do_reduce)) {
                    // Last chunk overall and not reducing: write to output CBs
                    move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
                    // Push dummy m/s tile (m and s packed into single tile)
                    cb_reserve_back(cb_out_ms, Sq_chunk_t);
                    cb_push_back(cb_out_ms, Sq_chunk_t);
                }
            }
        }

        /* Simplified tree reduction: just add outputs from other cores */
        if (do_reduce) {
            if (num_cores_to_wait > 0) {
                for (uint32_t i = 0; i < num_cores_to_wait; i++) {
                    // Pop unused m/s tile from sender (m and s packed into single tile)
                    cb_wait_front(cb_ms_in, Sq_chunk_t);
                    cb_pop_front(cb_ms_in, Sq_chunk_t);

                    // Add sender's output to accumulator
                    reconfig_data_format(cb_out_accumulate_im, cb_out_o);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
                }
            }

            if (is_sender_after_reduce) {
                // Intermediate node: write accumulated output for next receiver
                move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
                // Push dummy m/s tile (m and s packed into single tile)
                cb_reserve_back(cb_out_ms, Sq_chunk_t);
                cb_push_back(cb_out_ms, Sq_chunk_t);
                return;
            }

            // Final node: write output directly (no normalization)
            pack_reconfig_data_format(cb_out_final);
            move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
        }
    }

    // Free up cb_q_in after Q chunks
    cb_pop_front(cb_q_in, q_chunk_tiles);
}
