// SPDX-License-Identifier: Apache-2.0
// LOCAL COPY of production SDPA kernel for model-local specialization (README step 5).
// Verbatim copy first to preserve parity; specialize one measured region at a time.

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);

    constexpr uint32_t num_cores = get_compile_time_arg_val(22);

    constexpr bool is_causal = get_compile_time_arg_val(23) == 1;
    constexpr bool use_provided_mask = get_compile_time_arg_val(24) == 1;
    constexpr bool use_padded_mask = get_compile_time_arg_val(25) == 1;
    constexpr bool is_chunked = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr bool use_attention_sink = get_compile_time_arg_val(29) == 1;
    constexpr bool use_streaming_compute = get_compile_time_arg_val(30) == 1;
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(31);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(32);
    // Zigzag remap flag drives the external remap_q_index call on the flat B*NQH*q_num_chunks range.
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(33) == 1;

    const uint32_t core_id = get_arg_val<uint32_t>(0);
    const uint32_t num_phases = get_arg_val<uint32_t>(1);
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(2);
    uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(3);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(4);
    }

    // Global Q scheduling args follow phase_2 slot.
    const uint32_t global_q_start = get_arg_val<uint32_t>(5);
    const uint32_t global_q_count = get_arg_val<uint32_t>(6);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_arg_offset = 34;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_chunk_start_idx = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 9);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(cb_arg_offset + 14);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(cb_arg_offset + 16);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 17);
    // F4 aliased-K/V handshake args (constexpr-off unless kv_alias).
    constexpr bool kv_alias = get_compile_time_arg_val(cb_arg_offset + 18) == 1;
    constexpr uint32_t cb_kv_sync = get_compile_time_arg_val(cb_arg_offset + 19);
    // Aliased single-slot ring capacities in TILES: shared alloc / tile bytes.
    // K bf4 = 576B, V bf8 = 1088B. Shared alloc = 146,880B => K 255, V 135.
    constexpr uint32_t kv_shared_bytes = 146880;
    constexpr uint32_t kv_k_capacity_tiles = kv_alias ? (kv_shared_bytes / 576) : 0;
    constexpr uint32_t kv_v_capacity_tiles = kv_alias ? (kv_shared_bytes / 1088) : 0;
    uint32_t chunked_q_chunk_offset = 0;
    CircularBuffer cb_chunk_start_idx_obj(cb_chunk_start_idx);
    CircularBuffer cb_identity_scale_in_obj(cb_identity_scale_in);
    CircularBuffer cb_mask_in_obj(cb_mask_in);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_out);
    matmul_init(cb_q_in, cb_k_in);

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            cb_chunk_start_idx_obj.wait_front(1);
            uint32_t chunk_start_idx = ckernel::read_tile_value(cb_chunk_start_idx, 0, 0);
            cb_chunk_start_idx_obj.pop_front(1);
            const uint32_t q_chunk_size = Sq_chunk_t * TILE_HEIGHT;
            chunked_q_chunk_offset_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunked_q_chunk_offset_phase_2 = chunked_q_chunk_offset_phase_1;
            }
        }
    }

    if constexpr (use_streaming_compute) {
        // Streaming SDPA v2: direct cb_qkt_im writes via cb_push_back_hold_wr_ptr.
        // No row buffers needed; a dedicated 1-tile CB is used as recip scratch.

        // Wait once for identity scale; v2 removes per-call waits inside reduce_c_row_group
        cb_identity_scale_in_obj.wait_front(1);

        // Lightweight-mask context: writer pre-generates either [neginf, causal_diag, partial?]
        // or, for sliding, [neginf, trailing_primary, leading_prev, leading_current, trailing_next, partial?].
        // primary_diag_tile_idx is the per-layout tile used for the row-local diagonal stamp.
        LightweightMaskContext lw_mask;
        uint32_t lw_mask_tile_count = 1;
        lw_mask.neginf_tile_idx = 0;
        lw_mask.is_causal = is_causal;
        if constexpr (sliding_window_size > 0) {
            lw_mask.primary_diag_tile_idx = 1;
            lw_mask.sliding_leading_prev_tile_idx = 2;
            lw_mask.sliding_leading_tile_idx = 3;
            lw_mask.sliding_trailing_next_tile_idx = 4;
            lw_mask_tile_count = 5;
        } else if constexpr (is_causal) {
            lw_mask.causal_diag_tile_idx = lw_mask_tile_count++;
            lw_mask.primary_diag_tile_idx = lw_mask.causal_diag_tile_idx;
        }
        if constexpr (k_partial_col > 0) {
            lw_mask.global_n_partial_col = k_partial_col;
            lw_mask.global_n_partial_tile_idx = lw_mask_tile_count++;
            // global_n_padded_tiles = Sk_chunk_t - valid_tiles_in_last_chunk
            constexpr uint32_t last_chunk_first_tile =
                (valid_Skt > Sk_chunk_t) ? ((valid_Skt - 1) / Sk_chunk_t) * Sk_chunk_t : 0u;
            constexpr uint32_t valid_tiles_in_last_chunk = valid_Skt - last_chunk_first_tile;
            lw_mask.global_n_padded_tiles = Sk_chunk_t - valid_tiles_in_last_chunk;
        }
        // A user-provided dense mask is streamed per-chunk by the reader and consumed inside the
        // inner loop — it does not use the writer-generated lightweight palette, so skip this wait.
        if constexpr ((is_causal || sliding_window_size > 0 || k_partial_col > 0) && !use_provided_mask) {
            cb_mask_in_obj.wait_front(lw_mask_tile_count);
        }

        // Global Q scheduling: sdpa_standard_v2 walks the per-core flat range over
        // B*NQH*q_num_chunks chunks; the modulo inside its inner loop extracts the per-head q_chunk
        // from each flat index. num_phases==1 is pinned for streaming, so the chunked offset comes
        // from phase 1.
        sdpa_standard_v2<
            Sq_chunk_t,
            Sk_chunk_t,
            valid_Skt,
            DHt,
            vDHt,
            scale_fp32,
            qk_subblock_h,
            qk_subblock_w,
            out_subblock_h,
            out_subblock_w,
            use_padded_mask,
            cb_q_in,
            cb_k_in,
            cb_v_in,
            cb_qk_im,
            cb_identity_scale_in,
            cb_exp_max_diff,
            cb_col_identity,
            cb_recip_scratch,
            cb_out,  // normalized output goes directly to output CB
            cb_mask_in,
            sliding_window_size,
            is_causal,
            use_attention_sink,
            cb_attention_sink,
            use_provided_mask>(
            global_q_count,
            k_num_chunks,
            cb_out_im_A,
            cb_out_im_B,
            cb_max_A,
            cb_max_B,
            cb_sum_A,
            cb_sum_B,
            global_q_start,
            chunked_q_chunk_offset_phase_1,
            lw_mask,
            q_num_chunks,
            use_zigzag_balancing);
    } else {
        // Standard SDPA path (causal, masked, chunked, etc.)
        constexpr bool use_lightweight_causal_mask = is_causal && !use_provided_mask && (sliding_window_size == 0);

        LightweightMaskContext lw_mask;
        if constexpr (use_lightweight_causal_mask) {
            lw_mask.is_causal = true;
            lw_mask.neginf_tile_idx = 0;
            lw_mask.causal_diag_tile_idx = 1;
            lw_mask.primary_diag_tile_idx = lw_mask.causal_diag_tile_idx;
            cb_mask_in_obj.wait_front(2);
        }

        for (uint32_t phase = 0; phase < num_phases; ++phase) {
            if (phase == 0) {
                chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1;
            } else {
                chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2;
            }

            // Global Q scheduling: sdpa_standard walks the per-core flat range over
            // B*NQH*q_num_chunks chunks; the modulo inside its inner loop extracts the per-head
            // q_chunk from each flat index.
            sdpa_standard<
                cb_qk_im,
                cb_identity_scale_in,
                cb_attention_sink,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                vDHt,
                use_attention_sink,
                is_causal,
                use_provided_mask,
                use_padded_mask,
                is_chunked,
                scale_fp32,
                sliding_window_size,
                use_lightweight_causal_mask,
                kv_alias,
                cb_kv_sync,
                kv_k_capacity_tiles,
                kv_v_capacity_tiles>(
                Skt,
                qk_in0_block_w,
                qk_subblock_w,
                qk_subblock_h,
                qk_in0_num_subblocks,
                qk_in1_num_subblocks,
                qk_num_blocks,
                out_in0_block_w,
                out_subblock_w,
                out_subblock_h,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_num_blocks,
                /*iter_q_start=*/0,
                /*iter_q_end=*/global_q_count,
                q_num_chunks,
                /*local_q_start=*/global_q_start,
                chunked_q_chunk_offset,
                k_num_chunks,
                q_chunk_tiles,
                k_chunk_tiles,
                v_chunk_tiles,
                qk_chunk_tiles,
                out_chunk_tiles,
                cb_q_in,
                cb_k_in,
                cb_v_in,
                cb_mask_in,
                cb_col_identity,
                cb_out_im_A,
                cb_out_im_B,
                cb_max_A,
                cb_max_B,
                cb_sum_A,
                cb_sum_B,
                cb_exp_max_diff,
                cb_out,
                lw_mask,
                use_zigzag_balancing);
        }
    }
}
