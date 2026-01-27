// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k]
// CBs with intermediate computations
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;  // Partial (X @ W1)[r, k_block] between p_blocks
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;  // Partial (X @ W3)[r, k_block] between p_blocks
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;          // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;          // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;            // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;    // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);         // output width (C)
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);  // inner dimension (K==P)

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t input_address = get_arg_val<uint32_t>(ra++);        // DRAM base for X
    const uint32_t w1_address = get_arg_val<uint32_t>(ra++);           // DRAM base for W1
    const uint32_t w2_address = get_arg_val<uint32_t>(ra++);           // DRAM base for W2
    const uint32_t w3_address = get_arg_val<uint32_t>(ra++);           // DRAM base for W3
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);  // Actual data rows
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);    // Sync iterations (for multicast)
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    // Shared multicast runtime args (same topology and semaphores for W1/W2/W3)
    // W1, W3, W2 execute sequentially so they can safely reuse the same semaphores
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_num_dests = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Address generators
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

    // Uniform padding: loop for max_rows_for_sync to maintain multicast sync.
    // For padding rows (r >= end_row), we read the last valid row to keep all
    // cores synchronized without producing extra output.

    // Shared semaphore pointers for synchronization (reused by W1/W2/W3)
    volatile tt_l1_ptr uint32_t* mcast_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);

    // ================== Loop structure with W1/W2/W3 multicast (flash-attention) ==================
    // Flash-attention optimization: for r in rows:
    //     # Phase A: Compute XW1[r, :] and XW3[r, :] - read X[r, p_block] only once!
    //     for p_block in p_blocks:                    # OUTER LOOP - read X once per p_block
    //        read X[r, p_block]
    //        for k_block in k_blocks:                 # INNER LOOP - accumulate across full hidden dimension
    //          # First process W1 for all p in p_block (compute processes W1 first)
    //          for p in p_block:
    //            [SENDER] read W1[p, k_block] from DRAM
    //            [SENDER] wait for all receivers ready
    //            [SENDER] multicast W1[p, k_block] to receivers
    //            [SENDER] signal data ready to receivers
    //          # Then process W3 for all p in p_block (compute processes W3 second)
    //          for p in p_block:
    //            [SENDER] read W3[p, k_block] from DRAM
    //            [SENDER] wait for all receivers ready
    //            [SENDER] multicast W3[p, k_block] to receivers
    //            [SENDER] signal data ready to receivers
    //
    //     # Phase B: Compute M[r,:] once
    //     [no reading required to compute M[r, :]]
    //
    //     # Phase C: Use M[r, :] for all c-blocks to compute Y[r, :]
    //     for c_block in c_blocks:
    //        for k_block in k_blocks:
    //          for c in c_block:
    //            [SENDER] read W2[k_block, c] from DRAM
    //            [SENDER] wait for all receivers ready
    //            [SENDER] multicast W2[k_block, c] to receivers
    //            [SENDER] signal data ready to receivers
    // ============================================================================
    for (uint32_t r = start_row; r < end_row_for_sync; ++r) {
        // ---- Phase A: Compute XW1[r,:] and XW3[r,:] with flash-attention optimization ----
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

            // --- Read p_block_size tiles of X[r, p_block] ONCE per p_block
            // For padding rows (r >= end_row), read last valid row to keep sync
            const uint32_t x_row = (r < end_row) ? r : (end_row - 1);
            const uint32_t x_tile_start = x_row * Wt + p_block_start;
            read_tiles_by_row(cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

            // Stream W1 and W3 data organized by k_blocks to match compute kernel expectations
            // Always use batched mcast with padding for unaligned edge blocks
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // Batched mcast W1: read block_size rows × block_size tiles, padding edge blocks
                const uint32_t w1_first_row_tile_start = p_block_start * hidden_Wt + k_block_start;
                mcast_sender_read_batched_rows_and_send(
                    cb_w1_idx,
                    w1_address_generator,
                    w1_first_row_tile_start,
                    block_size,    // tiles_per_row (always full block)
                    block_size,    // num_rows (always full block)
                    k_block_size,  // valid_tiles_per_row (actual valid tiles, may be < block_size)
                    p_block_size,  // valid_num_rows (actual valid rows, may be < block_size)
                    hidden_Wt,     // row_stride
                    tile_bytes,
                    mcast_sender_sem_ptr,
                    mcast_receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    mcast_num_dests);

                // Batched mcast W3: read block_size rows × block_size tiles, padding edge blocks
                const uint32_t w3_first_row_tile_start = p_block_start * hidden_Wt + k_block_start;
                mcast_sender_read_batched_rows_and_send(
                    cb_w3_idx,
                    w3_address_generator,
                    w3_first_row_tile_start,
                    block_size,    // tiles_per_row (always full block)
                    block_size,    // num_rows (always full block)
                    k_block_size,  // valid_tiles_per_row
                    p_block_size,  // valid_num_rows
                    hidden_Wt,     // row_stride
                    tile_bytes,
                    mcast_sender_sem_ptr,
                    mcast_receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    mcast_num_dests);
            }
        }

        // ---- Phase B: Compute M[r, :] once ----
        // No reading required to compute M[r, :]

        // ---- Phase C: Use M[r, :] for all c_blocks ----
        // Batched W2 multicast: read block_size columns × block_size rows at once
        // This reduces mcast sync overhead from 4× per k_block to 1× per k_block
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // Batched mcast W2: read block_size columns × block_size rows, padding edge blocks
                // W2 is [K, C] = [hidden_Wt, Wt] tiles, stored row-major
                // We need columns c_block_start..c_block_start+block_size, rows k_block_start..k_block_start+block_size
                const uint32_t w2_first_col_start = k_block_start * Wt + c_block_start;
                mcast_sender_read_batched_cols_and_send(
                    cb_w2_idx,
                    w2_address_generator,
                    w2_first_col_start,
                    block_size,    // tiles_per_col (always full block)
                    block_size,    // num_cols (always full block)
                    k_block_size,  // valid_tiles_per_col (actual valid rows, may be < block_size)
                    c_block_size,  // valid_num_cols (actual valid cols, may be < block_size)
                    Wt,            // col_stride (width of W2 matrix)
                    tile_bytes,
                    mcast_sender_sem_ptr,
                    mcast_receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    mcast_num_dests);
            }
        }
    }
}
