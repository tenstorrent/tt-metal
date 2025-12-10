// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <hw/inc/dataflow_api.h>

#include "dataflow_api.h"
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
    uint32_t input_address = get_arg_val<uint32_t>(ra++);  // DRAM base for X
    // NOTE: w1_address is NOT passed to receiver - W1 comes via multicast!
    uint32_t w2_address = get_arg_val<uint32_t>(ra++);  // DRAM base for W2
    uint32_t w3_address = get_arg_val<uint32_t>(ra++);  // DRAM base for W3
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    // NEW: Multicast-related runtime args for W1
    uint32_t w1_mcast_sender_noc_x = get_arg_val<uint32_t>(ra++);
    uint32_t w1_mcast_sender_noc_y = get_arg_val<uint32_t>(ra++);
    uint32_t w1_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    uint32_t w1_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Address generators - NOTE: No W1 address generator needed!
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto w2_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    const auto x_address_generator = TensorAccessor(x_args, input_address, tile_bytes);
    const auto w2_address_generator = TensorAccessor(w2_args, w2_address, tile_bytes);
    const auto w3_address_generator = TensorAccessor(w3_args, w3_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;

    // DPRINT << "RECEIVER kernel starting: start_row=" << start_row << ", num_rows=" << num_rows_to_process
    //        << ", sender_noc=(" << w1_mcast_sender_noc_x << "," << w1_mcast_sender_noc_y << ")\n";

    // Semaphore pointers for synchronization
    volatile tt_l1_ptr uint32_t* w1_mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w1_mcast_receiver_semaphore_addr);
    uint64_t w1_sender_semaphore_noc_addr =
        get_noc_addr(w1_mcast_sender_noc_x, w1_mcast_sender_noc_y, w1_mcast_sender_semaphore_addr);

#define ROW_OF_M_FITS_IN_L1 1

#ifdef ROW_OF_M_FITS_IN_L1
    // ================== Loop structure with W1 multicast receive ==================
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
    //            read W3[p, k_block] (no multicast yet - Phase 1)
    //
    //     # Phase B: Compute M[r,:] once
    //     [no reading required to compute M[r, :]]
    //
    //     # Phase C: Use M[r, :] for all c-blocks to compute Y[r, :]
    //     for c_block in c_blocks:
    //        for k_block in k_blocks:
    //           read W2[k_block, c_block] (no multicast yet - Phase 1)
    // ============================================================================
    for (uint32_t r = start_row; r < end_row; ++r) {
        // ---- Phase A: Compute XW1[r,:] and XW3[r,:] with flash-attention optimization ----
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

            // --- Read p_block_size tiles of X[r, p_block] ONCE per p_block
            uint32_t x_tile_start = r * Wt + p_block_start;
            read_tiles_by_row(cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

            // Receive W1 and read W3 data organized by k_blocks to match compute kernel expectations
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // First, receive all W1 data for this k_block (compute processes W1 first)
                // DPRINT << "Receiver waiting for W1, p_block_size=" << p_block_size << "\n";
                for (uint32_t p = 0; p < p_block_size; ++p) {
                    // DPRINT << "Receiver iteration p=" << p << ", resetting semaphore\n";
                    // 1. Reset receiver semaphore to INVALID (clear previous iteration's VALID)
                    noc_semaphore_set(w1_mcast_receiver_sem_ptr, 0);

                    // 2. Reserve CB space for incoming multicast data
                    cb_reserve_back(cb_w1_idx, block_size);

                    // DPRINT << "Receiver signaling sender ready\n";
                    // 3. Signal to sender that we're ready to receive
                    noc_semaphore_inc(w1_sender_semaphore_noc_addr, 1);
                    noc_async_write_barrier();  // CRITICAL: Wait for semaphore increment to reach sender!

                    // DPRINT << "Receiver waiting for data from sender\n";
                    // 4. Wait for sender to multicast data (writes to our reserved CB write pointer)
                    noc_semaphore_wait(w1_mcast_receiver_sem_ptr, 1);

                    // DPRINT << "Receiver received data, pushing to CB\n";
                    // 5. Push to CB (multicast already wrote to our L1 at CB write address)
                    cb_push_back(cb_w1_idx, block_size);
                }
                // DPRINT << "Receiver finished receiving W1\n";  // Then, read all W3 data for this k_block (compute
                // processes W3 second)
                // NOTE: W3 is NOT multicast in Phase 1 - each core still reads independently
                for (uint32_t p = 0; p < p_block_size; ++p) {
                    uint32_t w3_tile_start = (p_block_start + p) * hidden_Wt + k_block_start;
                    read_tiles_by_row(
                        cb_w3_idx, w3_address_generator, w3_tile_start, k_block_size, tile_bytes, block_size);
                }
            }
        }

        // ---- Phase B: Compute M[r, :] once ----
        // No reading required to compute M[r, :]

        // ---- Phase C: Use M[r, :] for all c_blocks ----
        // NOTE: W2 is NOT multicast in Phase 1 - each core still reads independently
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
                for (uint32_t c = 0; c < c_block_size; ++c) {
                    const uint32_t c_global = c_block_start + c;
                    uint32_t w2_tile_start = k_block_start * Wt + c_global;
                    read_tiles_by_col(
                        cb_w2_idx, w2_address_generator, w2_tile_start, k_block_size, tile_bytes, Wt, block_size);
                }
            }
        }
    }

#else
    // ================== Loop structure with W1 multicast receive (non-flash-attention) ==================
    for (uint32_t r = start_row; r < end_row; ++r) {
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

                        // Read X[r, p_block]
                        uint32_t x_tile_start = r * Wt + p_block_start;
                        read_tiles_by_row(
                            cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

                        // Receive W1[p_block, k] via multicast
                        for (uint32_t p = 0; p < p_block_size; ++p) {
                            // Signal ready
                            noc_semaphore_inc(w1_sender_semaphore_noc_addr, 1);

                            // Wait for data
                            noc_semaphore_wait(w1_mcast_receiver_sem_ptr, VALID);
                            noc_semaphore_set(w1_mcast_receiver_sem_ptr, INVALID);

                            // Reserve and push (multicast already wrote to CB)
                            cb_reserve_back(cb_w1_idx, block_size);
                            cb_push_back(cb_w1_idx, block_size);
                        }

                        // Read W3[p_block, k] - no multicast yet
                        for (uint32_t p = 0; p < p_block_size; ++p) {
                            uint32_t w3_tile_idx = (p_block_start + p) * hidden_Wt + k_block_start + k;
                            cb_reserve_back(cb_w3_idx, block_size);
                            uint32_t w3_l1_addr = get_write_ptr(cb_w3_idx);
                            noc_async_read_tile(w3_tile_idx, w3_address_generator, w3_l1_addr);
                            noc_async_read_barrier();
                            cb_push_back(cb_w3_idx, block_size);
                        }
                    }
                }

                // STEP B: Stream W2[k_block, c] for all c in c_block
                for (uint32_t c = 0; c < c_block_size; ++c) {
                    const uint32_t c_global = c_block_start + c;
                    uint32_t w2_tile_start = k_block_start * Wt + c_global;
                    read_tiles_by_col(
                        cb_w2_idx, w2_address_generator, w2_tile_start, k_block_size, tile_bytes, Wt, block_size);
                }
            }
        }
    }
#endif  // ROW_OF_M_FITS_IN_L1
}
