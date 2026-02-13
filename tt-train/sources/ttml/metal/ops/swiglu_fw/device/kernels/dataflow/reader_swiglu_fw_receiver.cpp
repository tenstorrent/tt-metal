// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// Dual-NOC Weight Receiver Kernel (RISCV_0 / NOC1)
//
// This kernel runs on RECEIVER cores (all cores except sender 0,0) on RISCV_0.
// It handles ONLY weight receiving via multicast.
// X reading and Y writing are handled by the X reader kernel on RISCV_1 (concurrent).
//
// Loop structure (must match sender and compute):
//   for r in max_rows_for_sync:
//     Phase A: for p_block: for k_block: receive W1, receive W3
//     Phase B: idle (SiLU is compute-only)
//     Phase C: for c_block: for k_block: receive W2
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
    // No X or Y addresses - those are on RISCV_1
    // No W addresses - all weights come via multicast
    const uint32_t max_rows_for_sync = get_arg_val<uint32_t>(ra++);

    // Sender location for semaphore signaling
    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(ra++));

    volatile tt_l1_ptr uint32_t* mcast_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, mcast_sender_semaphore_addr);

    for (uint32_t r = 0U; r < max_rows_for_sync; ++r) {
        // ---- Phase A: Receive W1/W3 for all p_blocks × k_blocks ----
        for (uint32_t p_block_start = 0U; p_block_start < Wt; p_block_start += block_size) {
            for (uint32_t k_block_start = 0U; k_block_start < hidden_Wt; k_block_start += block_size) {
                constexpr uint32_t tiles_per_batch = block_size * block_size;
                mcast_receiver_reserve_and_receive(
                    cb_w1_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
                mcast_receiver_reserve_and_receive(
                    cb_w3_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }

        // ---- Phase B: Idle (SiLU is compute-only) ----

        // ---- Phase C: Receive W2 for all c_blocks × k_blocks ----
        for (uint32_t c_block_start = 0U; c_block_start < Wt; c_block_start += block_size) {
            for (uint32_t k_block_start = 0U; k_block_start < hidden_Wt; k_block_start += block_size) {
                constexpr uint32_t tiles_per_batch = block_size * block_size;
                mcast_receiver_reserve_and_receive(
                    cb_w2_idx, tiles_per_batch, mcast_receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }
    }
}
