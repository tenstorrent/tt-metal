// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// IN1 Receiver + M Writer (RISCV_0) — Non-top-row cores
//
// Receives W1/W3 weight tiles via column multicast from the top-row sender.
// After compute produces M tiles, writes them to DRAM.
//
// Loop structure:
//   for m in per_core_M:
//     for k_block in K_blocks:
//       for n_sub in num_n_blocks:
//         receive W1 batch, push to CB
//         receive W3 batch, push to CB
//     write M[m, 0:actual_n_tiles] to DRAM
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
constexpr uint32_t per_core_M = get_compile_time_arg_val(3);
constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
constexpr uint32_t per_core_N_rounded = get_compile_time_arg_val(5);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(6);
constexpr uint32_t in1_mcast_sender_semaphore_id = get_compile_time_arg_val(7);
constexpr uint32_t in1_mcast_receiver_semaphore_id = get_compile_time_arg_val(8);

constexpr uint32_t tiles_per_batch = block_size * block_size;

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

    constexpr auto m_args = TensorAccessorArgs<9>();
    const auto m_addr_gen = TensorAccessor(m_args, m_address, tile_bytes);

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    const uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, mcast_sender_semaphore_addr);

    const uint32_t num_k_blocks = (Wt + block_size - 1U) / block_size;

    for (uint32_t m = 0U; m < per_core_M; ++m) {
        // ---- Receive W1/W3 for all K-blocks and N-sub-blocks ----
        for (uint32_t k = 0U; k < num_k_blocks; ++k) {
            for (uint32_t n_sub = 0U; n_sub < num_n_blocks; ++n_sub) {
                mcast_receiver_reserve_and_receive(
                    cb_w1_idx, tiles_per_batch, receiver_sem_ptr, sender_semaphore_noc_addr);
                mcast_receiver_reserve_and_receive(
                    cb_w3_idx, tiles_per_batch, receiver_sem_ptr, sender_semaphore_noc_addr);
            }
        }

        // ---- Write M tiles to DRAM ----
        if (m < actual_m_tiles) {
            const uint32_t m_row = m_start + m;
            cb_wait_front(cb_m_out_idx, per_core_N_rounded);
            uint32_t l1_read_addr = get_read_ptr(cb_m_out_idx);
            for (uint32_t n = 0U; n < actual_n_tiles; ++n) {
                uint32_t tile_idx = m_row * hidden_Wt + n_start + n;
                noc_async_write_page(tile_idx, m_addr_gen, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_m_out_idx, per_core_N_rounded);
        } else {
            cb_wait_front(cb_m_out_idx, per_core_N_rounded);
            cb_pop_front(cb_m_out_idx, per_core_N_rounded);
        }
    }
}
