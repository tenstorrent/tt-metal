// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <hw/inc/dataflow_api.h>

#include "dataflow_api.h"
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
    uint32_t input_address = get_arg_val<uint32_t>(ra++);  // DRAM base for X
    uint32_t w1_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W1
    uint32_t w2_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W2
    uint32_t w3_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W3
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    // Shared multicast runtime args (same topology and semaphores for W1/W2/W3)
    // W1, W3, W2 execute sequentially so they can safely reuse the same semaphores
    uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    uint32_t mcast_num_dests = get_arg_val<uint32_t>(ra++);
    uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

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

    // Shared semaphore pointers for synchronization (reused by W1/W2/W3)
    volatile tt_l1_ptr uint32_t* mcast_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);

#define ROW_OF_M_FITS_IN_L1 1

#ifdef ROW_OF_M_FITS_IN_L1
    // ================== Loop structure with W1 multicast ==================
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

            // Stream W1 and W3 data organized by k_blocks to match compute kernel expectations
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // First, read all W1 data for this k_block (compute processes W1 first)
                if (mcast_num_dests > 0) {
                    // Multicast path: loop over each row, read, multicast
                    for (uint32_t p = 0; p < p_block_size; ++p) {
                        uint32_t w1_tile_start = (p_block_start + p) * hidden_Wt + k_block_start;
                        mcast_sender_read_and_send(
                            cb_w1_idx,
                            w1_address_generator,
                            w1_tile_start,
                            k_block_size,
                            tile_bytes,
                            block_size,
                            mcast_sender_sem_ptr,
                            mcast_receiver_sem_ptr,
                            mcast_receiver_semaphore_addr,
                            mcast_dest_noc_start_x,
                            mcast_dest_noc_start_y,
                            mcast_dest_noc_end_x,
                            mcast_dest_noc_end_y,
                            mcast_num_dests);
                    }
                } else {
                    // Single-core path: read all p_block_size rows at once (same as original reader)
                    // DPRINT << "Single-core path, no multicast\n";
                    for (uint32_t p = 0; p < p_block_size; ++p) {
                        uint32_t w1_tile_start = (p_block_start + p) * hidden_Wt + k_block_start;
                        read_tiles_by_row(
                            cb_w1_idx, w1_address_generator, w1_tile_start, k_block_size, tile_bytes, block_size);
                    }
                    // DPRINT << "Finished reading W1 tiles for single-core path\n";
                }
                // Then, read all W3 data for this k_block (compute processes W3 second)
                if (mcast_num_dests > 0) {
                    // Multicast path for W3: loop over each row, read, multicast
                    for (uint32_t p = 0; p < p_block_size; ++p) {
                        uint32_t w3_tile_start = (p_block_start + p) * hidden_Wt + k_block_start;
                        mcast_sender_read_and_send(
                            cb_w3_idx,
                            w3_address_generator,
                            w3_tile_start,
                            k_block_size,
                            tile_bytes,
                            block_size,
                            mcast_sender_sem_ptr,
                            mcast_receiver_sem_ptr,
                            mcast_receiver_semaphore_addr,
                            mcast_dest_noc_start_x,
                            mcast_dest_noc_start_y,
                            mcast_dest_noc_end_x,
                            mcast_dest_noc_end_y,
                            mcast_num_dests);
                    }
                } else {
                    // Single-core path: read all W3 normally
                    for (uint32_t p = 0; p < p_block_size; ++p) {
                        uint32_t w3_tile_start = (p_block_start + p) * hidden_Wt + k_block_start;
                        read_tiles_by_row(
                            cb_w3_idx, w3_address_generator, w3_tile_start, k_block_size, tile_bytes, block_size);
                    }
                }
            }
        }

        // DPRINT << "Finished Phase A for row " << r << "\n";
        // ---- Phase B: Compute M[r, :] once ----
        // No reading required to compute M[r, :]

        // ---- Phase C: Use M[r, :] for all c_blocks ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                if (mcast_num_dests > 0) {
                    // Multicast path for W2: loop over each column in c_block
                    for (uint32_t c = 0; c < c_block_size; ++c) {
                        const uint32_t c_global = c_block_start + c;
                        uint32_t w2_tile_start = k_block_start * Wt + c_global;

                        // Wait for receivers
                        mcast_sender_wait_for_receivers(mcast_sender_sem_ptr, mcast_num_dests);

                        // Read W2 column from DRAM (by column, not row)
                        read_tiles_by_col<false>(
                            cb_w2_idx, w2_address_generator, w2_tile_start, k_block_size, tile_bytes, Wt, block_size);
                        uint32_t w2_l1_write_addr = get_write_ptr(cb_w2_idx);
                        noc_async_read_barrier();

                        // Multicast data
                        mcast_sender_send_data(
                            w2_l1_write_addr,
                            mcast_dest_noc_start_x,
                            mcast_dest_noc_start_y,
                            mcast_dest_noc_end_x,
                            mcast_dest_noc_end_y,
                            k_block_size * tile_bytes,
                            mcast_num_dests);

                        // Signal receivers
                        mcast_sender_signal_receivers(
                            mcast_receiver_sem_ptr,
                            mcast_receiver_semaphore_addr,
                            mcast_dest_noc_start_x,
                            mcast_dest_noc_start_y,
                            mcast_dest_noc_end_x,
                            mcast_dest_noc_end_y,
                            mcast_num_dests);  // Push to CB
                        cb_push_back(cb_w2_idx, block_size);
                    }
                } else {
                    // Single-core path: read all W2 normally
                    for (uint32_t c = 0; c < c_block_size; ++c) {
                        const uint32_t c_global = c_block_start + c;
                        uint32_t w2_tile_start = k_block_start * Wt + c_global;
                        read_tiles_by_col(
                            cb_w2_idx, w2_address_generator, w2_tile_start, k_block_size, tile_bytes, Wt, block_size);
                    }
                }
            }
        }
        // DPRINT << "Finished Phase C for row " << r << "\n";
    }

#else
    // ================== Loop structure with W1 multicast (non-flash-attention) ==================
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);

            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : (hidden_Wt - k_block_start);

                // STEP A: Stream X + W1 for all k in this k_block
                for (uint32_t k = 0; k < k_block_size; ++k) {
                    const uint32_t w1_block_offset = k_block_start + k;

                    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
                        const uint32_t p_block_size =
                            (p_block_start + block_size <= Wt) ? block_size : (Wt - p_block_start);

                        // Read X[r, p_block]
                        uint32_t x_tile_start = r * Wt + p_block_start;
                        read_tiles_by_row(
                            cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

                        // Read W1[p_block, k] and multicast
                        for (uint32_t p = 0; p < p_block_size; ++p) {
                            uint32_t w1_tile_idx = (p_block_start + p) * hidden_Wt + w1_block_offset;

                            if (w1_mcast_num_dests > 0) {
                                // Multicast path
                                // Wait for receivers ready
                                noc_semaphore_wait(w1_mcast_sender_sem_ptr, w1_mcast_num_dests);
                                noc_semaphore_set(w1_mcast_sender_sem_ptr, 0);

                                // Read single tile
                                cb_reserve_back(cb_w1_idx, block_size);
                                uint32_t w1_l1_write_addr =
                                    get_write_ptr(cb_w1_idx);  // Save write address for multicast
                                noc_async_read_tile(w1_tile_idx, w1_address_generator, w1_l1_write_addr);
                                noc_async_read_barrier();
                                cb_push_back(cb_w1_idx, block_size);

                                // Multicast from the write address (same L1 address on all cores)
                                uint64_t w1_multicast_noc_addr = get_noc_multicast_addr(
                                    w1_mcast_dest_noc_start_x,
                                    w1_mcast_dest_noc_start_y,
                                    w1_mcast_dest_noc_end_x,
                                    w1_mcast_dest_noc_end_y,
                                    w1_l1_write_addr);
                                noc_async_write_multicast(
                                    w1_l1_write_addr, w1_multicast_noc_addr, tile_bytes, w1_mcast_num_dests);

                                // Signal ready
                                uint64_t w1_receiver_sem_noc_addr = get_noc_multicast_addr(
                                    w1_mcast_dest_noc_start_x,
                                    w1_mcast_dest_noc_start_y,
                                    w1_mcast_dest_noc_end_x,
                                    w1_mcast_dest_noc_end_y,
                                    w1_mcast_receiver_semaphore_addr);
                                noc_semaphore_set_multicast(
                                    w1_mcast_receiver_semaphore_addr, w1_receiver_sem_noc_addr, w1_mcast_num_dests);
                                noc_async_write_barrier();

                                // Pop from sender's CB
                                cb_pop_front(cb_w1_idx, block_size);
                            } else {
                                // Single-core path: just read normally
                                cb_reserve_back(cb_w1_idx, block_size);
                                uint32_t w1_l1_addr = get_write_ptr(cb_w1_idx);
                                noc_async_read_tile(w1_tile_idx, w1_address_generator, w1_l1_addr);
                                noc_async_read_barrier();
                                cb_push_back(cb_w1_idx, block_size);
                            }
                        }

                        // Read W3[p_block, k] - no multicast yet
                        for (uint32_t p = 0; p < p_block_size; ++p) {
                            uint32_t w3_tile_idx = (p_block_start + p) * hidden_Wt + w1_block_offset;
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
