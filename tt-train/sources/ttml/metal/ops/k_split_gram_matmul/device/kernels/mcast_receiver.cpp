// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Plain multicast receiver — runs on RISCV_1 (col interior/upper) or RISCV_0 (helper row receiver).
// Only receives multicast tiles into CB; does not write output.
// When REDUCE_RECV is defined, also receives the partner's reduce partial per
// (m_sub, n_sub) block, using the credit handshake described in mcast_receiver_writer.cpp:
// reserve the c_5 slot, credit the partner via its reduce_ack_sem, then wait for the data
// semaphore. Every c_5 transaction is a full M_block × N_block block.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    const uint32_t sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    const uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);

#ifdef REDUCE_RECV
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(6);
    const uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(7));
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t M_block = get_compile_time_arg_val(9);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(10);
    const uint32_t reduce_ack_sem_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t N_block = M_block;
    constexpr uint32_t reduce_block_capacity = M_block * N_block;
#else
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(7);
#endif

    uint32_t argidx = 0;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(argidx++);
#ifdef REDUCE_RECV
    const uint32_t partner_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_noc_y = get_arg_val<uint32_t>(argidx++);
#endif

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
                const uint64_t partner_ack_noc = get_noc_addr(partner_noc_x, partner_noc_y, reduce_ack_sem_addr);
                cb_reserve_back(reduce_cb, reduce_block_capacity);
                // c_5 slot is free — credit the partner to send the next block
                noc_semaphore_inc(partner_ack_noc, 1);
                noc_semaphore_wait_min(reduce_sem_ptr, 1);
                noc_semaphore_set(reduce_sem_ptr, 0);
                cb_push_back(reduce_cb, reduce_block_capacity);
            }
#endif
        }
    }

    noc_async_atomic_barrier();
}
