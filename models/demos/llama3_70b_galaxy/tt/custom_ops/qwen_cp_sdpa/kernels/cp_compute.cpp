// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Qwen context-parallel SDPA compute: both owned ring slices of a mesh column in one program, streaming v2
// compute (16-bit DEST). Derived from the stock sdpa.cpp (stock_compute.cpp here), causal + chunked path only.
// sdpa_items_v2 below is the stock sdpa_standard_v2 (compute_streaming.hpp) with one change: the Q loop walks
// this core's explicit item list (runtime args) instead of a zigzag range, and each item takes its slice's
// causal chunk offset. The K-chunk step is cp_inner_loop_step (cp_step.hpp), the stock sdpa_inner_loop_step
// specialised to this configuration. The per-Q-chunk work (K-chunk loop, bounds, masking, inner step) is unchanged, so
// every Q chunk is computed exactly as by the stock call.
//
// Runtime args: offset_0, offset_1, n_items, item... (item = (p << 16) | (nq << 8) | q_chunk).

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "cp_step.hpp"

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_h,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_h,
    uint32_t qktv_subblock_w,
    bool use_padded_mask,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    uint32_t cb_mask_in,
    uint32_t sliding_window_size = 0,
    bool is_causal_sdpa = false,
    bool use_attention_sink = false,
    uint32_t cb_attention_sink = INVALID_CB,
    bool use_provided_mask = false,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = INVALID_CB>
void sdpa_items_v2(
    const uint32_t n_items,
    const uint32_t item_arg_base,
    const uint32_t offset_0,
    const uint32_t offset_1,
    const uint32_t k_num_chunks,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const LightweightMaskContext& lw_mask) {
    init_sdpa_streaming_semaphores();

    // use_padded_mask + is_causal_sdpa is handled at the host level (mutually exclusive).
    static_assert(
        !(use_padded_mask && is_causal_sdpa), "use_padded_mask and is_causal_sdpa are mutually exclusive in v2");
    // K-loop bound geometry shared with the reader (see sliding_window_geometry.hpp).
    using window_geom = SlidingWindowLoopGeometry<sliding_window_size, is_causal_sdpa, TILE_HEIGHT>;
    constexpr bool has_sliding_window = window_geom::has_sliding_window;
    constexpr uint32_t left_window_tiles = window_geom::left_window_tiles;
    constexpr uint32_t right_window_tiles = window_geom::right_window_tiles;

    // v1 dense-mask scope: a user-provided mask supplies all masking itself (causal/sliding/padding
    // are baked into the tensor by the caller, and the reader neginf-fills padded positions), so it
    // is never combined with the structured stamp paths.
    static_assert(
        !(use_provided_mask && (is_causal_sdpa || has_sliding_window)),
        "use_provided_mask is mutually exclusive with causal/sliding stamping in v2");

    // Neginf tile is permanently fronted by the writer — wait once before any K-chunk loop.
    // Skipped for a user-provided mask: the writer generates no palette; the reader streams the
    // dense mask (with padded positions neginf-filled) per chunk instead.
    constexpr uint32_t padded_k_tiles_inner = (Sk_chunk_t - (Skt % Sk_chunk_t)) % Sk_chunk_t;
    if constexpr (use_padded_mask && padded_k_tiles_inner > 0 && !use_provided_mask) {
        CircularBuffer(cb_mask_in).wait_front(1);
    }

    constexpr uint32_t last_chunk_Sk = Sk_chunk_t - padded_k_tiles_inner;

    for (uint32_t q = 0; q < n_items; q++) {
        // Item q of this core: (slice p << 16) | (head << 8) | q chunk within the slice. Only the q chunk and the
        // slice's causal offset enter the compute; the head selects the reader's / writer's tiles.
        const uint32_t item = get_arg_val<uint32_t>(item_arg_base + q);
        const uint32_t chunked_q_chunk_offset = (item >> 16) ? offset_1 : offset_0;
        AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};
        AccumulatorHalf cur = {cb_sum_B, cb_max_B, cb_out_im_B};

        // reduce_trigger enables early reduce start via semaphore signaling from packer to unpacker.
        // The unpack MOP is split in half (block_ct_dim / 2), so active_Sk must be even,
        // and we need >1 subblock so the semaphore fires before the reduce's second half.
        constexpr bool can_reduce_trigger = reduce_trigger_supported && (Sk_chunk_t % qkt_subblock_w == 0) &&
                                            (Sk_chunk_t / qkt_subblock_w > 1) && (Sk_chunk_t % 2 == 0);

        // Pre-compute subblock width: compile-time for full chunks, hoisted for padded last chunk.
        constexpr uint32_t full_sbw = qkt_subblock_w;
        constexpr uint32_t padded_sbw = (last_chunk_Sk < Sk_chunk_t && last_chunk_Sk % qkt_subblock_w != 0)
                                            ? largest_factor_le(last_chunk_Sk, qkt_subblock_w)
                                            : qkt_subblock_w;

        // With largest_factor_le, padded chunks also have evenly-dividing subblocks,
        // so reduce_trigger can be enabled when the same constraints hold for last_chunk_Sk.
        constexpr bool can_reduce_trigger_padded = reduce_trigger_supported && (padded_k_tiles_inner > 0) &&
                                                   (last_chunk_Sk % padded_sbw == 0) &&
                                                   (last_chunk_Sk / padded_sbw > 1) && (last_chunk_Sk % 2 == 0);

        // Optional zigzag Q-chunk remap plus per-Q K-chunk bounds. Causal uses the
        // diagonal upper bound; sliding-window adds a lower bound and, for non-causal
        // centered windows, an upper bound around the Q chunk.
        uint32_t q_chunk_local = item & 0xFF;
        uint32_t q_start_tile = 0;
        uint32_t k_loop_start = 0;
        uint32_t k_loop_end = k_num_chunks;
        if constexpr (is_causal_sdpa || has_sliding_window) {
            // Reader and writer apply the same remap; compute must agree or causal
            // masks and output positions desync. The mod is a no-op when the input is per-head
            // ([0, q_num_chunks)) and extracts the per-head q_chunk when it's a flat global index
            // (global Q scheduling iterates across batches and heads).
            // q_chunk_global is the absolute Q chunk index (used for the diagonal);
            // chunked-prefill shifts this via chunked_q_chunk_offset.
            const uint32_t q_chunk_global = q_chunk_local + chunked_q_chunk_offset;
            q_start_tile = q_chunk_global * Sq_chunk_t;
            if constexpr (is_causal_sdpa) {
                const uint32_t limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
                k_loop_end = limit < k_num_chunks ? limit : k_num_chunks;
            }
            if constexpr (has_sliding_window) {
                if (q_start_tile > left_window_tiles) {
                    k_loop_start = (q_start_tile - left_window_tiles) / Sk_chunk_t;
                }
                if constexpr (!is_causal_sdpa) {
                    const uint32_t limit =
                        (q_start_tile + Sq_chunk_t + right_window_tiles + Sk_chunk_t - 1) / Sk_chunk_t;
                    k_loop_end = limit < k_num_chunks ? limit : k_num_chunks;
                }
            }
        }
        // Windowed K-range narrowing: this Q chunk's [k_lo, k_hi) comes from the reader's ctrl CB —
        // read via the UNPACK mailbox so all three TRISCs agree. The reader streams exactly this many
        // K/V chunks and the writer produces exactly this many mask chunks; disagreement deadlocks.
        if constexpr (use_windowed_narrowing) {
            CircularBuffer cb_k_range_obj(cb_windowed_k_range);
            cb_k_range_obj.wait_front(1);
            k_loop_start = ckernel::read_tile_value(cb_windowed_k_range, 0, 0);
            k_loop_end = ckernel::read_tile_value(cb_windowed_k_range, 0, 1);
            cb_k_range_obj.pop_front(1);
        }

        for (uint32_t k_chunk = k_loop_start; k_chunk < k_loop_end; k_chunk++) {
            bool is_first = (k_chunk == k_loop_start);
            bool is_last = (k_chunk == k_loop_end - 1);

            // Padded path is non-causal only (use_padded_mask && is_causal_sdpa rejected by static_assert).
            // With sliding-window loop narrowing, the loop's last chunk may not be the tensor's final K chunk.
            // A user-provided mask processes the full Sk_chunk_t (padded cols are neginf in the dense
            // mask), so it never takes the partial-tile narrowing.
            const bool is_padded =
                !is_causal_sdpa && !use_provided_mask && (k_chunk == k_num_chunks - 1) && (padded_k_tiles_inner > 0);
            uint32_t chunk_active_Sk = is_padded ? last_chunk_Sk : Sk_chunk_t;
            bool chunk_reduce_trigger = is_padded ? can_reduce_trigger_padded : can_reduce_trigger;
            uint32_t chunk_sbw = is_padded ? padded_sbw : full_sbw;

            // Last-chunk narrowing: causal and non-causal partial-tile K are mutually exclusive
            // (static_assert at function entry), but both shrink chunk_active_Sk on is_last to
            // skip cols past the diag (causal) or past Sk's last partial tile (non-causal partial).
            //
            // Mask side:
            // - Causal: per-row diagonal/trailing-neginf stamp via apply_lightweight_mask_streaming
            //   (no-stamp / partial-stamp / full-neginf decided per row by diag_col).
            // - Non-causal partial: trailing fully-padded tiles → neginf via num_padded; partial
            //   boundary tile → vertical-bar mask tile via apply_mask + lw_partial_tile_idx.
            bool apply_partial_mask = false;
            bool apply_sliding_mask = false;
            bool apply_causal_mask = is_causal_sdpa;
            uint32_t target_active_Sk = chunk_active_Sk;
            if constexpr (use_padded_mask && !is_causal_sdpa && !use_provided_mask) {
                // Stamp the partial-tile vertical mask whenever the tensor's last K chunk carries a
                // partial tile (Sk % TILE != 0), independent of fully padded tiles. When there are no
                // fully padded tiles, global_n_padded_tiles is 0, so target_active_Sk remains
                // Sk_chunk_t and only the vertical mask stamp is added.
                if ((k_chunk == k_num_chunks - 1) && lw_mask.global_n_partial_col > 0) {
                    target_active_Sk = Sk_chunk_t - lw_mask.global_n_padded_tiles;
                    apply_partial_mask = true;
                }
            }
            if constexpr (has_sliding_window) {
                const uint32_t k_start_tile = k_chunk * Sk_chunk_t;
                const uint32_t trailing_range_end =
                    is_causal_sdpa ? (q_start_tile + Sq_chunk_t) : (q_start_tile + Sq_chunk_t + right_window_tiles);
                apply_sliding_mask = true;
                if constexpr (is_causal_sdpa) {
                    const uint32_t k_end_tile = k_start_tile + Sk_chunk_t;
                    apply_causal_mask = (q_start_tile < k_end_tile) && ((q_start_tile + Sq_chunk_t) > k_start_tile);
                }
                if (is_last && trailing_range_end > k_start_tile) {
                    const uint32_t window_active_Sk = trailing_range_end - k_start_tile;
                    target_active_Sk = target_active_Sk < window_active_Sk ? target_active_Sk : window_active_Sk;
                }
            } else if constexpr (is_causal_sdpa) {
                if (is_last) {
                    target_active_Sk = q_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t;
                }
            }
            if (target_active_Sk < chunk_active_Sk) {
                chunk_active_Sk = target_active_Sk;
                chunk_sbw = largest_factor_le(chunk_active_Sk, qkt_subblock_w);
                // reduce_trigger relies on active_Sk == Sk_chunk_t for the unpack MOP split.
                chunk_reduce_trigger = false;
            }

            (void)apply_partial_mask;
            (void)apply_sliding_mask;
            (void)apply_causal_mask;
            cp_inner_loop_step<
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                vDHt,
                scale_fp32,
                qkt_subblock_h,
                qkt_subblock_w,
                qktv_subblock_h,
                qktv_subblock_w,
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                cb_mask_in>(
                prev,
                cur,
                is_last,
                is_first,
                chunk_active_Sk,
                chunk_reduce_trigger,
                chunk_sbw,
                q_start_tile,
                k_chunk * Sk_chunk_t,
                lw_mask.neginf_tile_idx,
                lw_mask.primary_diag_tile_idx);

            // Post-iteration cleanup
            // prev.out and cb_exp_max_diff are already popped row-by-row inside salad_correct_row.
            if (!is_first) {
                sdpa_cb_pop_front_out_of_line(prev.max, Sq_chunk_t);
                sdpa_cb_pop_front_out_of_line(prev.sum, Sq_chunk_t);
            }

            if (is_last) {
                sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t);
            } else {
                std::swap(prev, cur);
            }
        }
        // Q already popped inside sdpa_inner_loop_step after Phase 1 of the last K chunk.
    }
}

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(9);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(10);

    constexpr uint32_t cb_arg_offset = 11;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 9);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(cb_arg_offset + 14);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 15);

    const uint32_t offset_0 = get_arg_val<uint32_t>(0);
    const uint32_t offset_1 = get_arg_val<uint32_t>(1);
    const uint32_t n_items = get_arg_val<uint32_t>(2);
    constexpr uint32_t item_arg_base = 3;

    CircularBuffer cb_identity_scale_in_obj(cb_identity_scale_in);
    CircularBuffer cb_mask_in_obj(cb_mask_in);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_out);
    matmul_init(cb_q_in, cb_k_in);

    // As the stock streaming path: wait once for the identity scale and the lightweight causal mask palette
    // [neginf, causal_diag] that the writer leaves fronted.
    cb_identity_scale_in_obj.wait_front(1);
    LightweightMaskContext lw_mask;
    uint32_t lw_mask_tile_count = 1;
    lw_mask.neginf_tile_idx = 0;
    lw_mask.is_causal = true;
    lw_mask.causal_diag_tile_idx = lw_mask_tile_count++;
    lw_mask.primary_diag_tile_idx = lw_mask.causal_diag_tile_idx;
    cb_mask_in_obj.wait_front(lw_mask_tile_count);

    sdpa_items_v2<
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
        false,  // use_padded_mask
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_qk_im,
        cb_identity_scale_in,
        cb_exp_max_diff,
        cb_col_identity,
        cb_recip_scratch,
        cb_out,
        cb_mask_in,
        0,           // sliding_window_size
        true,        // is_causal
        false,       // use_attention_sink
        0xFFFFFFFF,  // cb_attention_sink: inactive id, as the stock factory passes it
        false,       // use_provided_mask
        false,       // use_windowed_narrowing
        cb_q_in>(    // cb_windowed_k_range: the stock factory's fallback id (q_in), unused
        n_items,
        item_arg_base,
        offset_0,
        offset_1,
        k_num_chunks,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        lw_mask);
    cp_prof_emit();
}
