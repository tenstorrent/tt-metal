// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k] - received via multicast
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
    const uint32_t input_address = get_arg_val<uint32_t>(ra++);  // DRAM base for X
    // NOTE: W1/W2/W3 all come via multicast - no DRAM addresses needed!
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);  // Actual data rows
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);    // Sync iterations
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    // Shared multicast runtime args (same sender and semaphores for W1/W2/W3)
    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Address generator - only need X, all weights come via multicast
    constexpr auto x_args = TensorAccessorArgs<3>();
    const auto x_address_generator = TensorAccessor(x_args, input_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    const uint32_t end_row_for_sync = start_row + max_rows_for_sync;

    // Uniform padding: loop for max_rows_for_sync to maintain multicast sync.
    // For padding rows (r >= end_row), we read the last valid row to keep all
    // cores synchronized without producing extra output.

    // Shared semaphore pointers for synchronization (reused by W1/W2/W3)
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, mcast_sender_semaphore_addr);

#ifdef ROW_OF_M_FITS_IN_L1
    // ================== Loop structure with W1/W2/W3 multicast receive (flash-attention) ==================
    // Flash-attention optimization: for r in rows:
    //     # Phase A: Compute XW1[r, :] and XW3[r, :] - read X[r, p_block] only once!
    //     for p_block in p_blocks:                    # OUTER LOOP - read X once per p_block
    //        read X[r, p_block]
    //        for k_block in k_blocks:                 # INNER LOOP - accumulate across full hidden dimension
    //          # First process W1 for all p in p_block (compute processes W1 first)
    //          for p in p_block:
    //            [RECEIVER] signal sender ready for W1[p, k_block]
    //            [RECEIVER] wait for W1[p, k_block] multicast to arrive
    //          # Then process W3 for all p in p_block (compute processes W3 second)
    //          for p in p_block:
    //            [RECEIVER] signal sender ready for W3[p, k_block]
    //            [RECEIVER] wait for W3[p, k_block] multicast to arrive
    //
    //     # Phase B: Compute M[r,:] once
    //     [no reading required to compute M[r, :]]
    //
    //     # Phase C: Use M[r, :] for all c-blocks to compute Y[r, :]
    //     for c_block in c_blocks:
    //        for k_block in k_blocks:
    //          for c in c_block:
    //            [RECEIVER] signal sender ready for W2[k_block, c]
    //            [RECEIVER] wait for W2[k_block, c] multicast to arrive
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

            // Receive W1 and W3 data organized by k_blocks to match compute kernel expectations
            // Always receive batched: block_size rows × block_size tiles per mcast
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                constexpr uint32_t tiles_per_batch = block_size * block_size;
                mcast_receiver_reserve_and_receive(
                    cb_w1_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
                mcast_receiver_reserve_and_receive(
                    cb_w3_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }

        // ---- Phase B: Compute M[r, :] once ----
        // No reading required to compute M[r, :]

        // ---- Phase C: Use M[r, :] for all c_blocks ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                [[maybe_unused]] const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                for (uint32_t c = 0; c < c_block_size; ++c) {
                    mcast_receiver_reserve_and_receive(
                        cb_w2_idx, block_size, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
                }
            }
        }
    }

#else
    // ================== Loop structure with W1/W2/W3 multicast receive (non-flash-attention) ==================
    // Standard approach: for r in rows:
    //     for c_block in c_blocks:              # Process output columns in blocks
    //        for k_block in k_blocks:           # Process hidden dimension in blocks
    //          for k in k_block:                # For each element in k_block
    //            for p_block in p_blocks:       # Accumulate across input dimension
    //              [RECEIVER] signal sender ready for W1[p_block, k]
    //              [RECEIVER] wait for W1[p_block, k] multicast to arrive
    //              [RECEIVER] signal sender ready for W3[p_block, k]
    //              [RECEIVER] wait for W3[p_block, k] multicast to arrive
    //          for c in c_block:                # For each column in c_block
    //            [RECEIVER] signal sender ready for W2[k_block, c]
    //            [RECEIVER] wait for W2[k_block, c] multicast to arrive
    // ============================================================================
    for (uint32_t r = start_row; r < end_row_for_sync; ++r) {
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);

            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : (hidden_Wt - k_block_start);

                // STEP A: Stream X + receive W1 for all k in this k_block
                for (uint32_t k = 0; k < k_block_size; ++k) {
                    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
                        const uint32_t p_block_size =
                            (p_block_start + block_size <= Wt) ? block_size : (Wt - p_block_start);

                        // Read X[r, p_block] - for padding rows, read last valid row
                        const uint32_t x_row = (r < end_row) ? r : (end_row - 1);
                        const uint32_t x_tile_start = x_row * Wt + p_block_start;
                        read_tiles_by_row(
                            cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

                        // Receive W1[p_block, k] via multicast
                        mcast_receiver_reserve_and_receive(
                            cb_w1_idx, block_size, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);

                        // Receive W3[p_block, k] via multicast
                        mcast_receiver_reserve_and_receive(
                            cb_w3_idx, block_size, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
                    }
                }

                // STEP B: Receive W2[k_block, c] via multicast for all c in c_block
                for (uint32_t c = 0; c < c_block_size; ++c) {
                    mcast_receiver_reserve_and_receive(
                        cb_w2_idx, block_size, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
                }
            }
        }
    }
#endif  // ROW_OF_M_FITS_IN_L1
}
