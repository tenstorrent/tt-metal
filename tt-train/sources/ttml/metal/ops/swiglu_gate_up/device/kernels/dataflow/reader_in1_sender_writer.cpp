// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// IN1 Sender + M Writer (RISCV_0) — Top row cores
//
// Outer loop iterates num_m_blocks times. For each m_block, reads and
// multicasts W1/W3 one N-block at a time (reused by compute across block_h M
// rows), then writes block_h rows of M tiles to DRAM.
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
constexpr uint32_t in1_mcast_sender_semaphore_id = get_compile_time_arg_val(8);
constexpr uint32_t in1_mcast_receiver_semaphore_id = get_compile_time_arg_val(9);

constexpr uint32_t tiles_per_n_block = block_size * block_size;
constexpr uint32_t cb_row_width = block_size;

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t w1_address = get_arg_val<uint32_t>(ra++);
    const uint32_t w3_address = get_arg_val<uint32_t>(ra++);
    const uint32_t m_address = get_arg_val<uint32_t>(ra++);
    const uint32_t m_start = get_arg_val<uint32_t>(ra++);
    const uint32_t n_start = get_arg_val<uint32_t>(ra++);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(ra++);
    const uint32_t actual_n_tiles = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t num_receivers = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_w1_idx);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(in1_mcast_sender_semaphore_id);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(in1_mcast_receiver_semaphore_id);

    constexpr auto w1_args = TensorAccessorArgs<10>();
    constexpr auto w3_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<w3_args.next_compile_time_args_offset()>();
    const auto w1_addr_gen = TensorAccessor(w1_args, static_cast<size_t>(w1_address), tile_bytes);
    const auto w3_addr_gen = TensorAccessor(w3_args, static_cast<size_t>(w3_address), tile_bytes);
    const auto m_addr_gen = TensorAccessor(m_args, static_cast<size_t>(m_address), tile_bytes);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);

    const McastLoopbackConfig mcast_cfg = {
        .sender_sem_ptr = sender_sem_ptr,
        .receiver_sem_ptr = receiver_sem_ptr,
        .receiver_sem_addr = mcast_receiver_semaphore_addr,
        .noc_start_x = mcast_dest_noc_start_x,
        .noc_start_y = mcast_dest_noc_start_y,
        .noc_end_x = mcast_dest_noc_end_x,
        .noc_end_y = mcast_dest_noc_end_y,
        .num_receivers = num_receivers,
    };

    for (uint32_t mb = 0U; mb < num_m_blocks; ++mb) {
        // ---- Send W1/W3 for all K-blocks (once per m_block) ----
        for (uint32_t k_block_start = 0U; k_block_start < Wt; k_block_start += block_size) {
            const uint32_t k_block_size = std::min(block_size, Wt - k_block_start);

            for (uint32_t n_block = 0U; n_block < num_n_blocks; ++n_block) {
                const uint32_t n_offset = n_start + n_block * block_size;
                const uint32_t valid_tiles_per_row = std::min(block_size, hidden_Wt - n_offset);
                const uint32_t w_tile_start = k_block_start * hidden_Wt + n_offset;

                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w1_idx,
                    w1_addr_gen,
                    w_tile_start,
                    cb_row_width,
                    block_size,
                    valid_tiles_per_row,
                    k_block_size,
                    hidden_Wt,
                    tile_bytes,
                    mcast_cfg);

                mcast_sender_read_batched_rows_and_send_loopback(
                    cb_w3_idx,
                    w3_addr_gen,
                    w_tile_start,
                    cb_row_width,
                    block_size,
                    valid_tiles_per_row,
                    k_block_size,
                    hidden_Wt,
                    tile_bytes,
                    mcast_cfg);
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
