// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// IN1 Receiver + M Writer (RISCV_0) — Non-top-row cores
//
// Outer loop iterates num_m_blocks times. For each m_block, receives W1/W3
// weight batches ONCE, then writes block_h rows of M tiles to DRAM.
//
// N-block batching: receives n_blocks_per_batch N-blocks per multicast.
// ============================================================================

#include <algorithm>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w3_idx = tt::CBIndex::c_2;
constexpr auto cb_m_out_idx = tt::CBIndex::c_5;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);
constexpr uint32_t block_h = get_compile_time_arg_val(3);
constexpr uint32_t num_m_blocks = get_compile_time_arg_val(4);
constexpr uint32_t per_core_N = get_compile_time_arg_val(5);
constexpr uint32_t per_core_N_rounded = get_compile_time_arg_val(6);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(7);
constexpr uint32_t n_blocks_per_batch = get_compile_time_arg_val(8);
constexpr uint32_t num_n_batch_iters = get_compile_time_arg_val(9);
constexpr uint32_t in1_mcast_sender_semaphore_id = get_compile_time_arg_val(10);
constexpr uint32_t in1_mcast_receiver_semaphore_id = get_compile_time_arg_val(11);

constexpr uint32_t tiles_per_n_block = block_size * block_size;
constexpr uint32_t tiles_per_receive = n_blocks_per_batch * tiles_per_n_block;
constexpr uint32_t num_k_blocks = (Wt + block_size - 1U) / block_size;

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t m_address = get_arg_val<uint32_t>(ra++);
    const uint32_t m_start = get_arg_val<uint32_t>(ra++);
    const uint32_t n_start = get_arg_val<uint32_t>(ra++);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(ra++);
    const uint32_t actual_n_tiles = get_arg_val<uint32_t>(ra++);
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(ra++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_w1_idx);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(in1_mcast_sender_semaphore_id);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(in1_mcast_receiver_semaphore_id);

    constexpr auto m_args = TensorAccessorArgs<12>();
    const auto m_addr_gen = TensorAccessor(m_args, m_address, tile_bytes);

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, mcast_sender_semaphore_addr);

    for (uint32_t mb = 0U; mb < num_m_blocks; ++mb) {
        // ---- Receive W1/W3 for all K-blocks (once per m_block), batched N-blocks ----
        for (uint32_t k = 0U; k < num_k_blocks; ++k) {
            for (uint32_t n_batch = 0U; n_batch < num_n_batch_iters; ++n_batch) {
                mcast_receiver_reserve_and_receive(
                    cb_w1_idx, tiles_per_receive, receiver_sem_ptr, sender_semaphore_noc_addr);
                mcast_receiver_reserve_and_receive(
                    cb_w3_idx, tiles_per_receive, receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }

        // ---- Write block_h rows of M to DRAM ----
        for (uint32_t m_sub = 0U; m_sub < block_h; ++m_sub) {
            uint32_t m = mb * block_h + m_sub;
            cb_wait_front(cb_m_out_idx, per_core_N_rounded);
            if (m < actual_m_tiles) {
                uint32_t m_row = m_start + m;
                uint32_t l1_read_addr = get_read_ptr(cb_m_out_idx);
                for (uint32_t n = 0U; n < actual_n_tiles; ++n) {
                    uint32_t tile_idx = m_row * hidden_Wt + n_start + n;
                    noc_async_write_page(tile_idx, m_addr_gen, l1_read_addr);
                    l1_read_addr += tile_bytes;
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_m_out_idx, per_core_N_rounded);
        }
    }
}
