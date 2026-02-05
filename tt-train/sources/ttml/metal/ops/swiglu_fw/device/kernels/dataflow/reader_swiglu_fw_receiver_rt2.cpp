// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Reader kernel (receiver) with rt_dim=2 support
// Reads 2 rows of X at a time to match compute kernel's rt_dim=2 processing
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r:r+2, p_block] - 2 rows
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
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    constexpr auto x_args = TensorAccessorArgs<3>();
    const auto x_address_generator = TensorAccessor(x_args, input_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    const uint32_t end_row_for_sync = start_row + max_rows_for_sync;

    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, mcast_sender_semaphore_addr);

    // Process rows in pairs (rt_dim=2)
    for (uint32_t r = start_row; r < end_row_for_sync; r += rt_dim) {
        // ---- Phase A: Read 2 rows of X per p_block, receive W1/W3 ----
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

            // Read rt_dim rows of X (2 rows × block_size tiles = 8 tiles)
            for (uint32_t row_offset = 0; row_offset < rt_dim; ++row_offset) {
                uint32_t x_row = r + row_offset;
                if (x_row >= end_row) {
                    x_row = end_row - 1;  // Duplicate last valid row for padding
                }
                const uint32_t x_tile_start = x_row * Wt + p_block_start;
                read_tiles_by_row(
                    cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);
            }

            // Receive W1 and W3 for all k_blocks
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                constexpr uint32_t tiles_per_batch = block_size * block_size;
                mcast_receiver_reserve_and_receive(
                    cb_w1_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
                mcast_receiver_reserve_and_receive(
                    cb_w3_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }

        // ---- Phase C: Receive W2 for all c_blocks ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                constexpr uint32_t tiles_per_batch = block_size * block_size;
                mcast_receiver_reserve_and_receive(
                    cb_w2_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }
    }
}
