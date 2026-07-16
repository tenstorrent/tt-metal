// SPDX-License-Identifier: Apache-2.0
// LOCAL COPY of production SDPA kernel for model-local specialization (README step 5).
// Verbatim copy first to preserve parity; specialize one measured region at a time.

// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"
#include "windowed_mask_gen.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Sqt = get_compile_time_arg_val(3);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(4);
    constexpr uint32_t unpadded_Sk = get_compile_time_arg_val(5);
    constexpr uint32_t DHt = get_compile_time_arg_val(6);
    constexpr uint32_t vDHt = get_compile_time_arg_val(7);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(12);
    constexpr uint32_t scale_val = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t is_causal = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t use_provided_mask = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t use_padded_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t is_chunked = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(19);
    constexpr bool use_lightweight_mask = get_compile_time_arg_val(20) == 1;
    constexpr bool use_streaming_compute = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(22);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(23);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(24) == 1;
    // Windowed (block-diagonal) mask generation flags. Fixed scalar slots BEFORE the tensor-accessor
    // block so the accessor offset chain stays intact for all configs.
    constexpr bool use_windowed_mask = get_compile_time_arg_val(25) == 1;

    // out accessor, then the cu_window accessor chained immediately after it (before the CB-id block).
    constexpr auto out_args = TensorAccessorArgs<26>();
    constexpr auto cu_window_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    const uint32_t num_phases = get_arg_val<uint32_t>(2);
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(3);
    uint32_t chunk_start_t_in_q_chunks_phase_1 = get_arg_val<uint32_t>(4);
    const uint32_t write_offset_phase_1 = get_arg_val<uint32_t>(5);
    uint32_t chunk_start_t_in_q_chunks_phase_2 = 0;
    uint32_t write_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunk_start_t_in_q_chunks_phase_2 = get_arg_val<uint32_t>(6);
        write_offset_phase_2 = get_arg_val<uint32_t>(7);
    }

    // Global Q scheduling args follow phase_2 args.
    const uint32_t global_q_start = get_arg_val<uint32_t>(8);
    const uint32_t global_q_count = get_arg_val<uint32_t>(9);
    const uint32_t cu_window_seqlens_addr = get_arg_val<uint32_t>(10);
    const uint32_t cu_window_seqlens_eles = get_arg_val<uint32_t>(11);

    constexpr uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;  // non-streaming drain only

    constexpr uint32_t cb_arg_offset = cu_window_args.next_compile_time_args_offset();
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_chunk_start_idx = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 4);
    // cu_window CB id lives in the CB-id block (appended by CBIds for windowed mode; inactive otherwise).
    constexpr uint32_t cb_cu_window_in = get_compile_time_arg_val(cb_arg_offset + 5);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    const auto out_writer = TensorAccessor(out_args, out_addr);

    const auto out_tile_shape = TensorTileShape(B, NQH, valid_Sqt, vDHt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    // Lightweight mask: generate template tiles once, leave permanently fronted.
    // Sliding layout: [neginf, trailing_primary, leading_prev, leading_current, trailing_next, k_partial?].
    // Non-sliding layout: [neginf, causal_diag?, k_partial?].
    if constexpr (use_lightweight_mask) {
        // is_causal handles K-partial via causal stamp; skip emitting partial tile in causal mode.
        constexpr uint32_t writer_partial_col = is_causal ? 0u : k_partial_col;
        generate_lightweight_mask_tiles<
            writer_partial_col,
            /*joint_l*/ 0u,
            cb_mask_in,
            is_causal,
            sliding_window_size>(noc);
    }

    // Windowed: load cu_window_seqlens into L1 once; the writer synthesizes the block-diagonal mask per
    // Q chunk from it (so the reader streams Q/K/V only).
    if constexpr (use_windowed_mask) {
        const auto cu_window_reader = TensorAccessor(cu_window_args, cu_window_seqlens_addr);
        constexpr uint32_t cu_tile_bytes = get_tile_size(cb_cu_window_in);
        CircularBuffer cb_cu(cb_cu_window_in);
        cb_cu.reserve_back(1);
        noc.async_read(
            cu_window_reader, CoreLocalMem<uint32_t>(cb_cu.get_write_ptr()), cu_tile_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
        cb_cu.push_back(1);
    }

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            CircularBuffer cb_chunk_start(cb_chunk_start_idx);
            cb_chunk_start.wait_front(1);
            auto chunk_start_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_chunk_start.get_read_ptr());
            uint32_t chunk_start_idx = chunk_start_ptr[0];
            cb_chunk_start.pop_front(1);
            const uint32_t q_chunk_size = Sq_chunk_t * tt::constants::TILE_HEIGHT;
            chunk_start_t_in_q_chunks_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunk_start_t_in_q_chunks_phase_2 = chunk_start_t_in_q_chunks_phase_1;
            }
        }
    }

    uint32_t chunk_start_t_in_q_chunks = 0;
    uint32_t write_offset = 0;
    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_1;
            write_offset = write_offset_phase_1;
        } else {
            chunk_start_t_in_q_chunks = chunk_start_t_in_q_chunks_phase_2;
            write_offset = write_offset_phase_2;
        }
        for (uint32_t global_q_iter = 0; global_q_iter < global_q_count; ++global_q_iter) {
            const auto decoded =
                decompose_global_q_index(global_q_start + global_q_iter, q_num_chunks, NQH, use_zigzag_balancing);
            const uint32_t nb = decoded.nb;
            const uint32_t nq = decoded.nq;
            const uint32_t q_chunk = decoded.q_chunk;

            // Generate masks only for legacy generated-mask variants. No-mask variants do not allocate cb_mask_in.
            if constexpr (
                !use_provided_mask && !use_lightweight_mask &&
                (is_causal || sliding_window_size > 0 || use_padded_mask)) {
                generate_mask<is_chunked, sliding_window_size, use_padded_mask, cb_mask_in>(
                    noc,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    q_chunk,
                    chunk_start_t_in_q_chunks,
                    true,
                    false,
                    unpadded_Sk,
                    0,
                    is_causal);
            }

            // Windowed: synthesize this Q chunk's block-diagonal mask (all K chunks) before draining its
            // output. The call is a template wrapper that only instantiates the generator when
            // use_windowed_mask is true (kernel_main is not a template, so a bare `if constexpr` here
            // would still compile the discarded body). valid_Skt derived from the unpadded K length.
            constexpr uint32_t windowed_valid_Skt =
                (unpadded_Sk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
            windowed_generate_if_enabled<use_windowed_mask, cb_mask_in, cb_cu_window_in>(
                noc,
                q_chunk,
                Sq_chunk_t,
                Sk_chunk_t,
                valid_Sqt,
                windowed_valid_Skt,
                k_num_chunks,
                cu_window_seqlens_eles);

            // Determine how many rows of OUT will be written. Both start and end rows are
            // capped by valid_Sqt, since Sq padding is independent of Sk padding.
            const uint32_t out_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_end_tile = std::min(out_row_start_tile + Sq_chunk_t, valid_Sqt);
            const uint32_t out_row_tile_count = out_row_end_tile - out_row_start_tile;
            uint32_t out_tile_id = out_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile, 0);
            if constexpr (use_streaming_compute) {
                // Streaming: drain per row-group (cb_out is a 2-slot ping-pong).
                // Compute always pushes Sq_chunk_t rows; rows past out_row_tile_count
                // are padding and get popped without being written.
                write_block_row_grouped(
                    noc,
                    out_writer,
                    cb_out,
                    Sq_chunk_t,
                    out_row_tile_count,
                    vDHt,
                    out_tile_id,
                    tile_bytes,
                    out_subblock_h,
                    barrier_threshold);
            } else {
                write_block(
                    noc,
                    out_writer,
                    cb_out,
                    out_chunk_tiles,
                    out_row_tile_count,
                    vDHt,
                    out_tile_id,
                    tile_bytes,
                    barrier_threshold);
            }
        }
    }  // close phase
}
