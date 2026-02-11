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

#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm.h"
#include "../../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm_reuse_dest_srcb.h"
#include "api/compute/eltwise_unary/exp.h"

#ifdef TRISC_PACK
#include "../../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

void kernel_main() {
    // Compile time arguments

    // Input dimensions in tiles
    constexpr uint32_t St = get_compile_time_arg_val(0);          // 1024
    constexpr uint32_t DHt = get_compile_time_arg_val(1);         // 18
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);        // 16
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);  // 1
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);  // 4

    // Matmul configs
    // 5: num_cores_per_batch (unused by compute)
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(7);
    // 8: num_heads_per_core (unused by compute)
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(9);
    // 10: q_tile_height (unused by compute)
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(11);
    constexpr uint32_t num_tree_reduction_steps = get_compile_time_arg_val(12);
    constexpr uint32_t dst_size = get_compile_time_arg_val(13);
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(15);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(16);
    constexpr uint32_t cb_interm_out = get_compile_time_arg_val(17);
    constexpr uint32_t cb_interm_ms = get_compile_time_arg_val(18);
    constexpr uint32_t cb_out_in = get_compile_time_arg_val(19);
    constexpr uint32_t cb_ms_in = get_compile_time_arg_val(20);
    constexpr uint32_t cb_out_o = get_compile_time_arg_val(21);
    constexpr uint32_t cb_out_ms = get_compile_time_arg_val(22);
    constexpr uint32_t cb_out_final = get_compile_time_arg_val(23);

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

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    constexpr bool transpose_k = true;
    constexpr bool transpose_v = false;

    // Init matmul and wait for Q
    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(SFPU_FPU, 0, 1));
    PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
    PACK(SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, true, scale_fp32, true));
    sdpa_custom_mm_block_init<transpose_k>(cb_q_in, cb_k_in, cb_out_o, Sk_chunk_t);

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

    // 8 rows per face, 2 faces per tile
    // constexpr uint32_t max_dst_offset = 8 * 2 * chunk_size;
    constexpr uint32_t packed_tile_size = 8 * 2;
    constexpr uint32_t mm2_dst_offset = 0;
    constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
    constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * vDHt;
    constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
    // Second col in the tile containing max
    constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
    // Next tile after max/sum
    constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_tile_offset = mm1_dst_offset / packed_tile_size;

    constexpr bool exp_approx_mode = false;

    bool sdpa_output_is_final = do_output && (!do_reduce || num_cores_to_wait == 0);
    uint32_t sdpa_output_cb = 0;
    uint32_t sdpa_ms_cb = 0;
    if (sdpa_output_is_final) {
        sdpa_output_cb = cb_out_final;
        sdpa_ms_cb = cb_out_ms;
    } else if (num_cores_to_wait > 0) {
        sdpa_output_cb = cb_interm_out;
        sdpa_ms_cb = cb_interm_ms;
    } else {
        sdpa_output_cb = cb_out_o;
        sdpa_ms_cb = cb_out_ms;
    }
    uint32_t num_chunks = (k_chunk_end - k_chunk_start + num_cores_per_head - 1) / num_cores_per_head;
    cb_wait_front(cb_q_in, q_chunk_tiles);
    cb_reserve_back(sdpa_output_cb, vDHt);
    cb_reserve_back(sdpa_ms_cb, Sq_chunk_t);
    tile_regs_acquire();
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        compute_sdpa_chunk<
            Sk_chunk_t,
            q_chunk_tiles,
            out_chunk_tiles,
            scale_fp32,
            scale_bf16,
            transpose_k,
            transpose_v,
            packed_tile_size,
            exp_approx_mode>(
            cb_q_in,
            cb_k_in,
            sdpa_output_cb,
            mm1_dst_offset,
            mm2_dst_offset,
            max_dst_offset,
            sum_dst_offset,
            corr_exp_dst_offset,
            chunk == 0,
            !sdpa_output_is_final && (chunk == (num_chunks - 1)));
    }
    if (!sdpa_output_is_final) {
        // Stall for Red Sum to finish
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile(max_dst_tile_offset, sdpa_ms_cb);
        cb_push_back(sdpa_ms_cb, Sq_chunk_t);
    } else {
        compute_sdpa_recip<out_chunk_tiles, exp_approx_mode, scale_bf16>(cb_q_in, sum_dst_offset, mm2_dst_offset);
    }
    // Sem is incremented once per 2 tiles since sem can only go up to 15
    for (uint32_t i = 0; i < out_chunk_tiles; i += 2) {
        PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
        pack_tile(mm2_dst_tile_offset + i, sdpa_output_cb);
        pack_tile(mm2_dst_tile_offset + i + 1, sdpa_output_cb);
        PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
    }
    cb_push_back(sdpa_output_cb, out_chunk_tiles);
    tile_regs_commit();
    // tile_regs_wait();
    tile_regs_release();
    sdpa_custom_mm_block_uninit();
    // For Safety, wait for SFPU and FPU to finish
    // We switch to running SFPU from math risc for tail operations
    MATH(t6_semaphore_wait_on_max<p_stall::STALL_MATH>(SFPU_FPU));
    MATH(t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU));
    PACK(t6_semaphore_wait_on_max<p_stall::STALL_PACK>(SFPU_FPU));
    PACK(t6_semaphore_wait_on_max<p_stall::STALL_PACK>(semaphore::FPU_SFPU));

    static_assert(vDHt % dst_size == 0, "vDHt must be divisible by dst_size");
    constexpr uint32_t num_blocks = vDHt / dst_size;
    constexpr uint32_t block_size = vDHt / num_blocks;

    /* Simplified tree reduction: just add outputs from other cores */
    if (do_reduce) {
        exp_tile_init<exp_approx_mode, false, scale_fp32>();
        if (num_cores_to_wait > 0) {
            UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, false, p_dim_stride_target::FACE_ROW_MAJOR>(
                cb_ms_in)));
            for (uint32_t i = 0; i < num_cores_to_wait - 1; i++) {
                // EXP_APPROX_MODE, final_reduction, block_size, num_blocks, scale_fp32, vector_mode
                sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                    cb_ms_in,       // worker max (ms1)
                    cb_interm_ms,   // prev max (m2)
                    cb_interm_ms,   // cur max output (m = max(m1, m2))
                    cb_out_in,      // l1 input
                    cb_interm_out,  // l2 input
                    cb_interm_out   // l output
                );
            }
            if (is_sender_after_reduce) {  // Send to next core
                sdpa_tail<exp_approx_mode, false, block_size, num_blocks, scale_fp32, VectorMode::C>(
                    cb_ms_in,       // worker max (ms1)
                    cb_interm_ms,   // prev max (m2)
                    cb_out_ms,      // cur max output (m = max(m1, m2))
                    cb_out_in,      // l1 input
                    cb_interm_out,  // l2 input
                    cb_out_o        // l output
                );
            } else {
                sdpa_tail<exp_approx_mode, true, block_size, num_blocks, scale_fp32, VectorMode::C>(
                    cb_ms_in,       // worker max (ms1)
                    cb_interm_ms,   // prev max (m2)
                    cb_out_ms,      // cur max output (m = max(m1, m2))
                    cb_out_in,      // l1 input
                    cb_interm_out,  // l2 input
                    cb_out_final    // l output
                );
            }
        }
    }

    // Free up cb_q_in after Q chunks
    cb_pop_front(cb_q_in, q_chunk_tiles);
}
