// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Dual-NOC Weight Sender Kernel (RISCV_0 / NOC1)
//
// This kernel runs on the SENDER core (0,0) on RISCV_0.
// It handles ONLY weight reading from DRAM + multicast to all cores.
// X reading is handled by the X reader kernel on RISCV_1 (concurrent).
//
// Loop structure (must match compute kernel and X reader):
//   for r in max_rows_for_sync:
//     Phase A: for p_block: for k_block: mcast W1, mcast W3
//     Phase B: idle (SiLU is compute-only)
//     Phase C: for c_block: for k_block: mcast W2
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w2_idx = tt::CBIndex::c_2;
constexpr auto cb_w3_idx = tt::CBIndex::c_3;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t w1_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w2_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w3_address = get_arg_val<uint32_t>(ra++);
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);

    // Multicast bounding box
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t num_receivers_excluding_self = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const uint32_t tile_bytes = get_tile_size(cb_w1_idx);

    // Weight address generators (no X - that's on RISCV_1)
    constexpr auto w1_args = TensorAccessorArgs<3>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    const auto w1_address_generator = TensorAccessor(w1_args, w1_address, tile_bytes);
    const auto w2_address_generator = TensorAccessor(w2_args, w2_address, tile_bytes);
    const auto w3_address_generator = TensorAccessor(w3_args, w3_address, tile_bytes);

    volatile tt_l1_ptr uint32_t* mcast_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);

    // Loop for max_rows_for_sync iterations (same weight data for all cores per row).
    // Weight data is the SAME for every row - the sender reads and multicasts each block
    // once per row iteration so all cores stay synchronized.
    for (uint32_t r = 0; r < max_rows_for_sync; ++r) {
        // ---- Phase A: Mcast W1/W3 for all p_blocks × k_blocks ----
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // Batched mcast W1 with LOOPBACK
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

                // Batched mcast W3 with LOOPBACK
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
        }

        // ---- Phase B: Idle (SiLU is compute-only) ----

        // ---- Phase C: Mcast W2 for all c_blocks × k_blocks ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // W2 in row-major CB layout: [k0_c0..k0_c3, k1_c0..k1_c3, ...]
                // This enables matmul_block in the compute kernel
                const uint32_t w2_first_row_start = k_block_start * Wt + c_block_start;
                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w2_idx,
                    w2_address_generator,
                    w2_first_row_start,
                    block_size,    // tiles_per_row (c tiles)
                    block_size,    // num_rows (k tiles)
                    c_block_size,  // valid_tiles_per_row
                    k_block_size,  // valid_num_rows
                    Wt,            // row_stride (width of W2 matrix)
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
