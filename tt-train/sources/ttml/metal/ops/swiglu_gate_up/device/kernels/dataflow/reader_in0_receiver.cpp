// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// IN0 Receiver (RISCV_1) — Non-left-column cores
//
// Receives X tiles via row multicast from the left-column sender.
// ============================================================================

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_in0_idx = tt::CBIndex::c_0;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(3);
constexpr uint32_t in0_mcast_sender_semaphore_id = get_compile_time_arg_val(4);
constexpr uint32_t in0_mcast_receiver_semaphore_id = get_compile_time_arg_val(5);

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(ra++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(ra++);

    const uint32_t mcast_sender_semaphore_addr = get_semaphore(in0_mcast_sender_semaphore_id);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(in0_mcast_receiver_semaphore_id);

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, mcast_sender_semaphore_addr);

    const uint32_t num_k_blocks = (Wt + block_size - 1U) / block_size;

    for (uint32_t m = 0U; m < per_core_M; ++m) {
        for (uint32_t k = 0U; k < num_k_blocks; ++k) {
            mcast_receiver_reserve_and_receive(cb_in0_idx, block_size, receiver_sem_ptr, sender_semaphore_noc_addr);
        }
    }
}
