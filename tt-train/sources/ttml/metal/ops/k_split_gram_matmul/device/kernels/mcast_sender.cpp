// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast sender — runs on RISCV_0 (row senders) or RISCV_1 (col senders).
// Reads input tiles from DRAM and multicasts to receiver cores.
// Row senders (c_0): RISCV_0/NOC_0, read M_block rows indexed by m_sub.
// Col senders (c_1): RISCV_1/NOC_1, read N_block rows indexed by n_sub.
// Even K-columns → lower/diag, odd K-columns → upper. One handshake per K-block batch.
// Pushes own-parity blocks to local CB before multicast (critical for avoiding deadlock).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    uint32_t sender_sem_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t receiver_sem_addr = get_semaphore(get_compile_time_arg_val(3));
    uint32_t sender_sem2_addr = get_semaphore(get_compile_time_arg_val(4));
    uint32_t receiver_sem2_addr = get_semaphore(get_compile_time_arg_val(5));
    constexpr uint32_t cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t block_size = get_compile_time_arg_val(7);
    constexpr uint32_t cb_size_tiles = get_compile_time_arg_val(8);

#ifdef SENDER_REDUCE_SEND
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(11);
    uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(12));
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(13);
    constexpr uint32_t M_block = get_compile_time_arg_val(14);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(15);
    constexpr auto tensor_args = TensorAccessorArgs<16>();
#else
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(10);
    constexpr auto tensor_args = TensorAccessorArgs<11>();
#endif

    uint32_t argidx = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t lower_noc_x_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t lower_noc_y_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t lower_noc_x_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t lower_noc_y_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t lower_num_dests = get_arg_val<uint32_t>(argidx++);
    const uint32_t upper_noc_x_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t upper_noc_y_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t upper_noc_x_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t upper_noc_y_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t upper_num_dests = get_arg_val<uint32_t>(argidx++);
    const uint32_t injector_keeps_odd = get_arg_val<uint32_t>(argidx++);
    const uint32_t lower_loopback = get_arg_val<uint32_t>(argidx++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(argidx++);
    const uint32_t Mpc = get_arg_val<uint32_t>(argidx++);
    const uint32_t K_tiles = get_arg_val<uint32_t>(argidx++);  // padded K for loop structure
    const uint32_t logical_M_tiles = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_K_tiles = get_arg_val<uint32_t>(argidx++);

#ifdef SENDER_REDUCE_SEND
    const uint32_t partner_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_noc_y = get_arg_val<uint32_t>(argidx++);
#endif

    volatile tt_l1_ptr uint32_t* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);
    volatile tt_l1_ptr uint32_t* sender_sem2_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem2_addr);

    *(receiver_sem_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* receiver_sem2_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem2_addr);
    *(receiver_sem2_ptr) = VALID;

    const auto reader = TensorAccessor(tensor_args, src_addr, tile_size);

    constexpr uint32_t block_bytes = block_size * tile_size;
    constexpr uint32_t num_blocks = num_tiles / block_size;
    constexpr uint32_t cb_capacity_bytes = cb_size_tiles * tile_size;

    // M_block = num_tiles / K_tiles, K_block_tiles = block_size / M_block
    uint32_t rows_per_block = num_tiles / K_tiles;
    uint32_t K_block_tiles = block_size / rows_per_block;

    uint32_t recv_cb_base = get_write_ptr(cb_id);

    for (uint32_t m_sub = 0; m_sub < num_m_blocks; m_sub++) {
        for (uint32_t n_sub = 0; n_sub < num_n_blocks; n_sub++) {
            // Row sender (c_0): reads rows m_sub*M_block+m; col sender (c_1): reads rows n_sub*N_block+n
            uint32_t row_base = (cb_id == 0) ? m_sub * rows_per_block : n_sub * rows_per_block;

            uint32_t lower_recv_offset = 0;
            uint32_t upper_recv_offset = 0;

            for (uint32_t blk = 0; blk < num_blocks; blk++) {
                bool is_lower_block = (blk % 2 == 0);

                uint32_t batch_idx = blk / 2;
                uint32_t first_k_col = batch_idx * K_block_tiles * 2 + (is_lower_block ? 0 : 1);

                cb_reserve_back(cb_id, block_size);
                uint32_t base_addr = get_write_ptr(cb_id);
                for (uint32_t kb = 0; kb < K_block_tiles; kb++) {
                    uint32_t k_col = first_k_col + kb * 2;
                    for (uint32_t m = 0; m < rows_per_block; m++) {
                        uint32_t cb_offset = (kb * rows_per_block + m) * tile_size;
                        uint32_t global_row = tile_offset + row_base + m;
                        if (global_row < logical_M_tiles && k_col < logical_K_tiles) {
                            uint32_t dram_tile = global_row * logical_K_tiles + k_col;
                            noc_async_read_tile(dram_tile, reader, base_addr + cb_offset);
                        } else {
                            fill_tile_zeros(base_addr + cb_offset, tile_size);
                        }
                    }
                }
                noc_async_read_barrier();

                // Push to own CB before multicast — critical to let compute start
                // while we wait for the multicast handshake
                bool is_own = (is_lower_block != (bool)injector_keeps_odd);
                if (is_own) {
                    cb_push_back(cb_id, block_size);
                }

                uint32_t recv_dst;

                if (is_lower_block && lower_num_dests > 0) {
                    recv_dst = recv_cb_base + lower_recv_offset;

                    if (lower_loopback && lower_num_dests == 1) {
                        if (base_addr != recv_dst) {
                            noc_async_write(base_addr, get_noc_addr(my_x[0], my_y[0], recv_dst), block_bytes);
                            noc_async_write_barrier();
                        }
                    } else if (lower_loopback) {
                        noc_semaphore_wait(sender_sem_ptr, lower_num_dests - 1);
                        noc_semaphore_set(sender_sem_ptr, 0);

                        uint64_t mcast_addr = get_noc_multicast_addr(
                            lower_noc_x_start, lower_noc_y_start, lower_noc_x_end, lower_noc_y_end, recv_dst);
                        noc_async_write_multicast_loopback_src(base_addr, mcast_addr, block_bytes, lower_num_dests);

                        uint64_t sem_mcast_addr = get_noc_multicast_addr(
                            lower_noc_x_start, lower_noc_y_start, lower_noc_x_end, lower_noc_y_end, receiver_sem_addr);
#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif
                        noc_semaphore_set_multicast_loopback_src(receiver_sem_addr, sem_mcast_addr, lower_num_dests);

                        noc_semaphore_wait(receiver_sem_ptr, VALID);
                        noc_semaphore_set(receiver_sem_ptr, INVALID);
                    } else {
                        noc_semaphore_wait(sender_sem_ptr, lower_num_dests);
                        noc_semaphore_set(sender_sem_ptr, 0);

                        uint64_t mcast_addr = get_noc_multicast_addr(
                            lower_noc_x_start, lower_noc_y_start, lower_noc_x_end, lower_noc_y_end, recv_dst);
                        noc_async_write_multicast(base_addr, mcast_addr, block_bytes, lower_num_dests);

                        uint64_t sem_mcast_addr = get_noc_multicast_addr(
                            lower_noc_x_start, lower_noc_y_start, lower_noc_x_end, lower_noc_y_end, receiver_sem_addr);
#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif
                        noc_semaphore_set_multicast(receiver_sem_addr, sem_mcast_addr, lower_num_dests);
                    }

                    lower_recv_offset += block_bytes;
                    if (lower_recv_offset >= cb_capacity_bytes)
                        lower_recv_offset = 0;
                }

                if (!is_lower_block && upper_num_dests > 0) {
                    recv_dst = recv_cb_base + upper_recv_offset;

                    noc_semaphore_wait(sender_sem2_ptr, upper_num_dests);
                    noc_semaphore_set(sender_sem2_ptr, 0);

                    uint64_t mcast_addr = get_noc_multicast_addr(
                        upper_noc_x_start, upper_noc_y_start, upper_noc_x_end, upper_noc_y_end, recv_dst);
                    noc_async_write_multicast(base_addr, mcast_addr, block_bytes, upper_num_dests);

                    uint64_t sem_mcast_addr = get_noc_multicast_addr(
                        upper_noc_x_start, upper_noc_y_start, upper_noc_x_end, upper_noc_y_end, receiver_sem2_addr);
#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif
                    noc_semaphore_set_multicast(receiver_sem2_addr, sem_mcast_addr, upper_num_dests);

                    upper_recv_offset += block_bytes;
                    if (upper_recv_offset >= cb_capacity_bytes)
                        upper_recv_offset = 0;
                }
            }

            noc_async_write_barrier();
            noc_async_atomic_barrier();

#ifdef SENDER_REDUCE_SEND
            {
                uint32_t M_start = m_sub * rows_per_block;
                uint32_t current_M_block = (rows_per_block < Mpc - M_start) ? rows_per_block : (Mpc - M_start);
                uint32_t N_start = n_sub * rows_per_block;
                uint32_t current_N = (rows_per_block < Mpc - N_start) ? rows_per_block : (Mpc - N_start);
                uint32_t block_tiles = current_M_block * current_N;

                uint32_t partner_reduce_addr = get_write_ptr(reduce_cb);
                uint64_t partner_sem_noc = get_noc_addr(partner_noc_x, partner_noc_y, reduce_sem_addr);

                cb_wait_front(cb_out, block_tiles);
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                uint64_t partner_noc_addr = get_noc_addr(partner_noc_x, partner_noc_y, partner_reduce_addr);
                noc_async_write(l1_read_addr, partner_noc_addr, block_tiles * out_tile_size);
                noc_async_write_barrier();
                noc_semaphore_inc(partner_sem_noc, 1);
                cb_pop_front(cb_out, block_tiles);
            }
#endif
        }
    }
}
