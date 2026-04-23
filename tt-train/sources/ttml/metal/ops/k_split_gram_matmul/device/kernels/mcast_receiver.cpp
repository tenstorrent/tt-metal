// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Plain multicast receiver — runs on RISCV_1 (col interior/upper) or RISCV_0 (helper row receiver).
// Only receives multicast tiles into CB; does not write output.
// When REDUCE_RECV is defined, also waits for partner's reduce partial per (m_sub, n_sub) block.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    uint32_t sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);

#ifdef REDUCE_RECV
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(6);
    uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(7));
    constexpr uint32_t Mpc = get_compile_time_arg_val(8);
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t M_block = get_compile_time_arg_val(10);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t N_block = M_block;
#else
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(7);
#endif

    uint32_t argidx = 0;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(argidx++);

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_semaphore_addr);

    constexpr uint32_t num_blocks = num_tiles / block_size;

    for (uint32_t m_sub = 0; m_sub < num_m_blocks; m_sub++) {
        for (uint32_t n_sub = 0; n_sub < num_n_blocks; n_sub++) {
            for (uint32_t blk = 0; blk < num_blocks; blk++) {
                cb_reserve_back(cb_id, block_size);

                noc_semaphore_set(receiver_sem_ptr, INVALID);
                noc_semaphore_inc(sender_sem_noc_addr, 1);

                noc_semaphore_wait(receiver_sem_ptr, VALID);

                cb_push_back(cb_id, block_size);
            }

#ifdef REDUCE_RECV
            {
                volatile tt_l1_ptr uint32_t* reduce_sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sem_addr);
                uint32_t M_start = m_sub * M_block;
                uint32_t current_M_block = std::min(M_block, Mpc - M_start);
                uint32_t N_start = n_sub * N_block;
                uint32_t current_N = std::min(N_block, Mpc - N_start);
                uint32_t block_tiles = current_M_block * current_N;
                cb_reserve_back(reduce_cb, block_tiles);
                noc_semaphore_wait(reduce_sem_ptr, 1);
                noc_semaphore_set(reduce_sem_ptr, 0);
                cb_push_back(reduce_cb, block_tiles);
            }
#endif
        }
    }

    noc_async_atomic_barrier();
}
