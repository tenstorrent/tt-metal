// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TRUE FLASH SwiGLU RECEIVER KERNEL
//
// This kernel implements the "True Flash" dataflow optimization for SwiGLU.
// The key difference from the original: loop order is INVERTED to match sender.
//
// True Flash loop order (k_block outer, p_block inner):
//   for k_block:
//     for p_block: receive W1, W3
//     for c_block: receive W2
//
// All receivers wait for multicast from sender (core 0,0).
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k_block] - via multicast
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block] - via multicast
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k_block] - via multicast
// CBs with intermediate computations
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;
constexpr auto cb_m_idx = tt::CBIndex::c_8;
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);         // embed_dim / 32 (output width)
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);  // hidden_dim / 32

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t input_address = get_arg_val<uint32_t>(ra++);
    // NOTE: W1/W2/W3 come via multicast - no DRAM addresses needed!
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    // Multicast sync parameters
    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Address generator for X only
    constexpr auto x_args = TensorAccessorArgs<3>();
    const auto x_address_generator = TensorAccessor(x_args, input_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    const uint32_t end_row_for_sync = start_row + max_rows_for_sync;

    // Semaphore pointers
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, mcast_sender_semaphore_addr);

    // Calculate block counts
    const uint32_t num_k_blocks = (hidden_Wt + block_size - 1) / block_size;
    const uint32_t num_p_blocks = (Wt + block_size - 1) / block_size;
    const uint32_t num_c_blocks = (Wt + block_size - 1) / block_size;

    constexpr uint32_t tiles_per_batch = block_size * block_size;

    // ================== TRUE FLASH LOOP ORDER: k_block OUTER ==================
    for (uint32_t r = start_row; r < end_row_for_sync; ++r) {
        // For padding rows, use last valid row for X
        const uint32_t x_row = (r < end_row) ? r : (end_row - 1);

        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            // ---- Phase A: Read X and receive W1/W3 for this k_block ----
            for (uint32_t p_block_idx = 0; p_block_idx < num_p_blocks; ++p_block_idx) {
                const uint32_t p_block_start = p_block_idx * block_size;
                const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

                // Read X[r, p_block] (each receiver reads its own X)
                const uint32_t x_tile_start = x_row * Wt + p_block_start;
                read_tiles_by_row(
                    cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

                // Receive W1[p_block, k_block] via multicast
                mcast_receiver_reserve_and_receive(
                    cb_w1_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);

                // Receive W3[p_block, k_block] via multicast
                mcast_receiver_reserve_and_receive(
                    cb_w3_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }

            // ---- Phase B: Compute M (no data transfer needed) ----

            // ---- Phase C: Receive W2[k_block, :] for ALL c_blocks ----
            for (uint32_t c_block_idx = 0; c_block_idx < num_c_blocks; ++c_block_idx) {
                // Receive W2[k_block, c_block] via multicast
                mcast_receiver_reserve_and_receive(
                    cb_w2_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }
    }
}
