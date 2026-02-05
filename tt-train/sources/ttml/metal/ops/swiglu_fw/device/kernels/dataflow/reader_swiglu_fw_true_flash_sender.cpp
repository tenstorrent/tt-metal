// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TRUE FLASH SwiGLU SENDER KERNEL
//
// This kernel implements the "True Flash" dataflow optimization for SwiGLU.
// The key difference from the original: loop order is INVERTED.
//
// Original (p_block outer, k_block inner):
//   for p_block: for k_block: send W1, W3
//
// True Flash (k_block outer, p_block inner):
//   for k_block: for p_block: send W1, W3; for c_block: send W2
//
// This allows the compute kernel to process M tiles on-demand without
// materializing the full M row in L1.
//
// Trade-off: X is read K_blocks times more often (mitigated by X caching).
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k_block]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k_block]
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
    const uint32_t w1_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w2_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w3_address = get_arg_val<uint32_t>(ra++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);
    const uint32_t start_row = get_arg_val<uint32_t>(ra++);

    // Multicast parameters
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t num_receivers_excluding_self = get_arg_val<uint32_t>(ra++);
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

    // Semaphore pointers
    volatile tt_l1_ptr uint32_t* mcast_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);

    // Calculate block counts
    const uint32_t num_k_blocks = (hidden_Wt + block_size - 1) / block_size;
    const uint32_t num_p_blocks = (Wt + block_size - 1) / block_size;
    const uint32_t num_c_blocks = (Wt + block_size - 1) / block_size;

    // ================== TRUE FLASH LOOP ORDER: k_block OUTER ==================
    for (uint32_t r = start_row; r < end_row_for_sync; ++r) {
        // For padding rows, use last valid row for X
        const uint32_t x_row = (r < end_row) ? r : (end_row - 1);

        for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
            const uint32_t k_block_start = k_block_idx * block_size;
            const uint32_t k_block_size =
                (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

            // ---- Phase A: Read X and send W1/W3 for this k_block ----
            for (uint32_t p_block_idx = 0; p_block_idx < num_p_blocks; ++p_block_idx) {
                const uint32_t p_block_start = p_block_idx * block_size;
                const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

                // Read X[r, p_block] (sender also reads its own X)
                const uint32_t x_tile_start = x_row * Wt + p_block_start;
                read_tiles_by_row(
                    cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

                // Multicast W1[p_block, k_block] with loopback
                const uint32_t w1_first_row_tile_start = p_block_start * hidden_Wt + k_block_start;
                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w1_idx,
                    w1_address_generator,
                    w1_first_row_tile_start,
                    block_size,    // tiles_per_row
                    block_size,    // num_rows
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
                    num_receivers_excluding_self);

                // Multicast W3[p_block, k_block] with loopback
                const uint32_t w3_first_row_tile_start = p_block_start * hidden_Wt + k_block_start;
                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w3_idx,
                    w3_address_generator,
                    w3_first_row_tile_start,
                    block_size,    // tiles_per_row
                    block_size,    // num_rows
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
                    num_receivers_excluding_self);
            }

            // ---- Phase B: Compute M (no data transfer needed) ----

            // ---- Phase C: Send W2[k_block, :] for ALL c_blocks ----
            for (uint32_t c_block_idx = 0; c_block_idx < num_c_blocks; ++c_block_idx) {
                const uint32_t c_block_start = c_block_idx * block_size;
                const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

                // Multicast W2[k_block, c_block] with loopback
                const uint32_t w2_first_col_start = k_block_start * Wt + c_block_start;
                mcast_sender_read_batched_cols_and_send_loopback(
                    cb_w2_idx,
                    w2_address_generator,
                    w2_first_col_start,
                    block_size,    // tiles_per_col
                    block_size,    // num_cols
                    k_block_size,  // valid_tiles_per_col
                    c_block_size,  // valid_num_cols
                    Wt,            // col_stride
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
