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

    // Multicast bounding box and semaphores
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t num_receivers_excluding_self = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    const McastLoopbackConfig mcast_cfg = {
        .sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(mcast_sender_semaphore_addr),
        .receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(mcast_receiver_semaphore_addr),
        .receiver_sem_addr = mcast_receiver_semaphore_addr,
        .noc_start_x = mcast_dest_noc_start_x,
        .noc_start_y = mcast_dest_noc_start_y,
        .noc_end_x = mcast_dest_noc_end_x,
        .noc_end_y = mcast_dest_noc_end_y,
        .num_receivers = num_receivers_excluding_self,
    };

    const uint32_t tile_bytes = get_tile_size(cb_w1_idx);

    // Weight address generators (no X - that's on RISCV_1)
    constexpr auto w1_args = TensorAccessorArgs<3>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    const auto w1_address_generator = TensorAccessor(w1_args, w1_address, tile_bytes);
    const auto w2_address_generator = TensorAccessor(w2_args, w2_address, tile_bytes);
    const auto w3_address_generator = TensorAccessor(w3_args, w3_address, tile_bytes);

    for (uint32_t r = 0; r < max_rows_for_sync; ++r) {
        // ---- Phase A: Mcast W1/W3 for all p_blocks × k_blocks ----
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
                const uint32_t weight_tile_start = p_block_start * hidden_Wt + k_block_start;

                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w1_idx,
                    w1_address_generator,
                    weight_tile_start,
                    block_size,
                    block_size,
                    k_block_size,
                    p_block_size,
                    hidden_Wt,
                    tile_bytes,
                    mcast_cfg);

                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w3_idx,
                    w3_address_generator,
                    weight_tile_start,
                    block_size,
                    block_size,
                    k_block_size,
                    p_block_size,
                    hidden_Wt,
                    tile_bytes,
                    mcast_cfg);
            }
        }

        // ---- Phase B: Idle (SiLU is compute-only) ----

        // ---- Phase C: Mcast W2 for all c_blocks × k_blocks ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // W2 in row-major CB layout for matmul_block compatibility
                const uint32_t w2_first_row_start = k_block_start * Wt + c_block_start;
                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w2_idx,
                    w2_address_generator,
                    w2_first_row_start,
                    block_size,
                    block_size,
                    c_block_size,
                    k_block_size,
                    Wt,
                    tile_bytes,
                    mcast_cfg);
            }
        }
    }
}
