// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast receiver + output/reduce kernel with M_block x N_block streaming.
// Per (m_sub, n_sub): receives K-blocks via multicast, then handles output:
//   Default:      writes compute output to DRAM.
//   REDUCE_SEND:  NOC-writes compute partial to partner's reduce CB.
//   REDUCE_RECV:  waits for partner's partial, then writes combined output to DRAM.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    uint32_t sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t Mpc = get_compile_time_arg_val(8);

#ifdef REDUCE_SEND
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(9);
    uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t M_block = get_compile_time_arg_val(12);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(13);
#else
    constexpr uint32_t padded_out_tiles = get_compile_time_arg_val(9);
#ifdef REDUCE_RECV
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(10);
    uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t M_block = get_compile_time_arg_val(13);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(14);
    constexpr auto out_tensor_args = TensorAccessorArgs<15>();
#else
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t M_block = get_compile_time_arg_val(11);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(12);
    constexpr auto out_tensor_args = TensorAccessorArgs<13>();
#endif
#endif

    uint32_t argidx = 0;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(argidx++);

#ifdef REDUCE_SEND
    const uint32_t partner_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_noc_y = get_arg_val<uint32_t>(argidx++);
#else
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_M_tiles = get_arg_val<uint32_t>(argidx++);
#ifdef MIRROR_OUTPUT
    const uint32_t mirror_M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t mirror_N_start_tile = get_arg_val<uint32_t>(argidx++);
    constexpr uint32_t mirror_cb = tt::CBIndex::c_4;
#endif
#endif

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_semaphore_addr);

    constexpr uint32_t num_blocks = num_tiles / block_size;
    constexpr uint32_t N_block = M_block;  // N_block = M_block always

    for (uint32_t m_sub = 0; m_sub < num_m_blocks; m_sub++) {
        uint32_t M_start = m_sub * M_block;
        uint32_t current_M_block = (M_block < Mpc - M_start) ? M_block : (Mpc - M_start);

        for (uint32_t n_sub = 0; n_sub < num_n_blocks; n_sub++) {
            uint32_t N_start = n_sub * N_block;
            uint32_t current_N = (N_block < Mpc - N_start) ? N_block : (Mpc - N_start);
            uint32_t block_tiles = current_M_block * current_N;

            // --- Receive K-blocks via multicast ---
            for (uint32_t blk = 0; blk < num_blocks; blk++) {
                cb_reserve_back(cb_id, block_size);
                noc_semaphore_set(receiver_sem_ptr, INVALID);
                noc_semaphore_inc(sender_sem_noc_addr, 1);
                noc_semaphore_wait(receiver_sem_ptr, VALID);
                cb_push_back(cb_id, block_size);
            }

            // --- Output / reduce ---
#ifdef REDUCE_SEND
            {
                uint32_t partner_reduce_addr = get_write_ptr(reduce_cb);
                uint64_t partner_noc_addr = get_noc_addr(partner_noc_x, partner_noc_y, partner_reduce_addr);
                uint64_t partner_sem_noc = get_noc_addr(partner_noc_x, partner_noc_y, reduce_sem_addr);

                cb_wait_front(cb_out, block_tiles);
                uint32_t l1_addr = get_read_ptr(cb_out);
                noc_async_write(l1_addr, partner_noc_addr, block_tiles * out_tile_size);
                noc_async_write_barrier();
                noc_semaphore_inc(partner_sem_noc, 1);
                cb_pop_front(cb_out, block_tiles);
            }
#else
#ifdef REDUCE_RECV
            {
                volatile tt_l1_ptr uint32_t* reduce_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sem_addr);
                cb_reserve_back(reduce_cb, block_tiles);
                noc_semaphore_wait(reduce_sem_ptr, 1);
                noc_semaphore_set(reduce_sem_ptr, 0);
                cb_push_back(reduce_cb, block_tiles);
            }
#endif
            {
                const auto out_writer = TensorAccessor(out_tensor_args, out_addr, out_tile_size);

                // Row-major write: compute pushes rows of current_N tiles
                for (uint32_t m = 0; m < current_M_block; m++) {
                    cb_wait_front(cb_out, current_N);
                    uint32_t l1_read_addr = get_read_ptr(cb_out);
                    uint32_t row = M_start_tile + M_start + m;
                    for (uint32_t n = 0; n < current_N; n++) {
                        uint32_t col = N_start_tile + N_start + n;
                        if (row < logical_M_tiles && col < logical_M_tiles) {
                            uint32_t tile_id = row * padded_out_tiles + col;
                            noc_async_write_tile(tile_id, out_writer, l1_read_addr + n * out_tile_size);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, current_N);
                }
#ifdef MIRROR_OUTPUT
                for (uint32_t n = 0; n < current_N; n++) {
                    cb_wait_front(mirror_cb, current_M_block);
                    uint32_t l1_read_addr = get_read_ptr(mirror_cb);
                    uint32_t col = mirror_N_start_tile + N_start + n;
                    for (uint32_t m = 0; m < current_M_block; m++) {
                        uint32_t row = mirror_M_start_tile + M_start + m;
                        if (row < logical_M_tiles && col < logical_M_tiles) {
                            uint32_t tile_id = row * padded_out_tiles + col;
                            noc_async_write_tile(tile_id, out_writer, l1_read_addr + m * out_tile_size);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(mirror_cb, current_M_block);
                }
#endif
            }
#endif
        }
    }
}
