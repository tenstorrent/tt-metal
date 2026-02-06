// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Reader kernel (sender) with rt_dim=2 support and X Row Caching (Phase 3)
//
// Phase 3 optimization: Cache full X row in L1 to avoid re-reading from DRAM
// for each k_block iteration. This reduces X reads by K_blocks factor.
//
// Flow per row pair:
//   1. Read full X[r:r+2, :] into CB (rt_dim × Wt tiles)
//   2. For each k_block:
//      - For each p_block: mcast W1[p_block, k_block], W3[p_block, k_block]
//      - Compute reuses cached X for all p_blocks
//   3. For each c_block, k_block: mcast W2[k_block, c_block]
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r:r+2, :] - full row cache
constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w2_idx = tt::CBIndex::c_2;
constexpr auto cb_w3_idx = tt::CBIndex::c_3;
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;
constexpr auto cb_m_idx = tt::CBIndex::c_8;
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;
constexpr auto cb_y_idx = tt::CBIndex::c_10;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);
constexpr uint32_t rt_dim = 2;  // Process 2 rows at a time

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t input_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w1_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w2_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w3_address = get_arg_val<uint32_t>(ra++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t num_receivers_excluding_self = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto w1_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    const auto x_address_generator = TensorAccessor(x_args, input_address, tile_bytes);
    const auto w1_address_generator = TensorAccessor(w1_args, w1_address, tile_bytes);
    const auto w2_address_generator = TensorAccessor(w2_args, w2_address, tile_bytes);
    const auto w3_address_generator = TensorAccessor(w3_args, w3_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    const uint32_t end_row_for_sync = start_row + max_rows_for_sync;

    volatile tt_l1_ptr uint32_t* mcast_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);

    // Process rows in pairs (rt_dim=2)
    for (uint32_t r = start_row; r < end_row_for_sync; r += rt_dim) {
        // ---- Phase 3: Read FULL X rows once (cache in L1) ----
        // Read rt_dim full rows of X (2 rows × Wt tiles = 2*Wt tiles)
        for (uint32_t row_offset = 0; row_offset < rt_dim; ++row_offset) {
            uint32_t x_row = r + row_offset;
            // Clamp to valid row range for padding iterations
            if (x_row >= end_row) {
                x_row = end_row - 1;  // Duplicate last valid row for padding
            }
            // Read full row of X (Wt tiles)
            const uint32_t x_tile_start = x_row * Wt;
            read_tiles_by_row(cb_input_idx, x_address_generator, x_tile_start, Wt, tile_bytes, Wt);
        }

        // ---- Phase A: Mcast W1/W3 for all k_blocks (X already cached) ----
        for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
            const uint32_t k_block_size =
                (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

            for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
                const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

                // Batched mcast W1
                const uint32_t w1_first_row_tile_start = p_block_start * hidden_Wt + k_block_start;
                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w1_idx,
                    w1_address_generator,
                    w1_first_row_tile_start,
                    block_size,
                    block_size,
                    k_block_size,
                    p_block_size,
                    hidden_Wt,
                    tile_bytes,
                    mcast_sender_sem_ptr,
                    mcast_receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_receivers_excluding_self);

                // Batched mcast W3
                const uint32_t w3_first_row_tile_start = p_block_start * hidden_Wt + k_block_start;
                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w3_idx,
                    w3_address_generator,
                    w3_first_row_tile_start,
                    block_size,
                    block_size,
                    k_block_size,
                    p_block_size,
                    hidden_Wt,
                    tile_bytes,
                    mcast_sender_sem_ptr,
                    mcast_receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_receivers_excluding_self);
            }

            // ---- Phase C: Mcast W2 for all c_blocks (for this k_block) ----
            for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
                const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

                const uint32_t w2_first_col_start = k_block_start * Wt + c_block_start;
                mcast_sender_read_batched_cols_and_send_loopback(
                    cb_w2_idx,
                    w2_address_generator,
                    w2_first_col_start,
                    block_size,
                    block_size,
                    k_block_size,
                    c_block_size,
                    Wt,
                    tile_bytes,
                    mcast_sender_sem_ptr,
                    mcast_receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_receivers_excluding_self);
            }
        }
    }
}
