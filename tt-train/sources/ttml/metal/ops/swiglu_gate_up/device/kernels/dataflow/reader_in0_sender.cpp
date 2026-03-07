// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// IN0 Sender (RISCV_1) — Left column cores
//
// Reads block_h rows of X tiles from DRAM per K-block and multicasts across
// the row. Outer loop over m_blocks, inner loop over k_blocks.
// ============================================================================

#include <algorithm>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_in0_idx = tt::CBIndex::c_0;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t block_h = get_compile_time_arg_val(2);
constexpr uint32_t num_m_blocks = get_compile_time_arg_val(3);
constexpr uint32_t in0_mcast_sender_semaphore_id = get_compile_time_arg_val(4);
constexpr uint32_t in0_mcast_receiver_semaphore_id = get_compile_time_arg_val(5);

constexpr uint32_t x_tiles_per_block = block_h * block_size;
constexpr uint32_t num_k_blocks = (Wt + block_size - 1U) / block_size;

void kernel_main() {
    uint32_t ra = 0U;
    const uint32_t x_address = get_arg_val<uint32_t>(ra++);
    const uint32_t m_start = get_arg_val<uint32_t>(ra++);
    const uint32_t actual_m_tiles = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(ra++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(ra++);
    const uint32_t num_receivers = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_in0_idx);
    const uint32_t mcast_sender_semaphore_addr = get_semaphore(in0_mcast_sender_semaphore_id);
    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(in0_mcast_receiver_semaphore_id);

    constexpr auto x_args = TensorAccessorArgs<6>();
    const auto x_addr_gen = TensorAccessor(x_args, x_address, tile_bytes);

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
        for (uint32_t k_block = 0U; k_block < num_k_blocks; ++k_block) {
            const uint32_t k_block_start = k_block * block_size;
            const uint32_t k_block_size = std::min(block_size, Wt - k_block_start);

            if (num_receivers > 0U) {
                mcast_sender_wait_for_receivers(sender_sem_ptr, num_receivers);
                cb_reserve_back(cb_in0_idx, x_tiles_per_block);
                uint32_t l1_addr = get_write_ptr(cb_in0_idx);

                for (uint32_t m_sub = 0U; m_sub < block_h; ++m_sub) {
                    uint32_t m = mb * block_h + m_sub;
                    uint32_t m_row = std::min(m_start + m, m_start + actual_m_tiles - 1U);
                    uint32_t x_row_start = m_row * Wt;
                    for (uint32_t t = 0U; t < k_block_size; ++t) {
                        uint64_t noc_addr = x_addr_gen.get_noc_addr(x_row_start + k_block_start + t);
                        noc_async_read(noc_addr, l1_addr, tile_bytes);
                        l1_addr += tile_bytes;
                    }
                    if (k_block_size < block_size) {
                        uint64_t pad_addr = x_addr_gen.get_noc_addr(x_row_start + k_block_start + k_block_size - 1U);
                        for (uint32_t t = k_block_size; t < block_size; ++t) {
                            noc_async_read(pad_addr, l1_addr, tile_bytes);
                            l1_addr += tile_bytes;
                        }
                    }
                }
                noc_async_read_barrier();
                mcast_sender_send_data_loopback(
                    get_write_ptr(cb_in0_idx),
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    x_tiles_per_block * tile_bytes,
                    num_receivers + 1U);
                mcast_sender_signal_receivers_loopback(
                    receiver_sem_ptr,
                    mcast_receiver_semaphore_addr,
                    mcast_dest_noc_start_x,
                    mcast_dest_noc_start_y,
                    mcast_dest_noc_end_x,
                    mcast_dest_noc_end_y,
                    num_receivers + 1U);
                cb_push_back(cb_in0_idx, x_tiles_per_block);
            } else {
                cb_reserve_back(cb_in0_idx, x_tiles_per_block);
                uint32_t l1_addr = get_write_ptr(cb_in0_idx);
                for (uint32_t m_sub = 0U; m_sub < block_h; ++m_sub) {
                    uint32_t m = mb * block_h + m_sub;
                    uint32_t m_row = std::min(m_start + m, m_start + actual_m_tiles - 1U);
                    uint32_t x_row_start = m_row * Wt;
                    for (uint32_t t = 0U; t < k_block_size; ++t) {
                        uint64_t noc_addr = x_addr_gen.get_noc_addr(x_row_start + k_block_start + t);
                        noc_async_read(noc_addr, l1_addr, tile_bytes);
                        l1_addr += tile_bytes;
                    }
                    if (k_block_size < block_size) {
                        uint64_t pad_addr = x_addr_gen.get_noc_addr(x_row_start + k_block_start + k_block_size - 1U);
                        for (uint32_t t = k_block_size; t < block_size; ++t) {
                            noc_async_read(pad_addr, l1_addr, tile_bytes);
                            l1_addr += tile_bytes;
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_in0_idx, x_tiles_per_block);
            }
        }
    }
}
