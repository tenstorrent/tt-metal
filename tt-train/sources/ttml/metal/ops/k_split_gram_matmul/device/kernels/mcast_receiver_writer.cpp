// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast receiver + output writer — runs on RISCV_0 (row direction) or RISCV_1 (col edge).
// Receives multicast K-blocks, then per (m_sub, n_sub) handles output:
//   REDUCE_SEND:  NOC-writes compute partial to partner's reduce CB.
//   REDUCE_RECV:  waits for partner's partial, then writes combined output to DRAM.
//
// Reduce channel protocol (one block in flight):
//   receiver: cb_reserve_back(c_5 capacity) → inc sender's reduce_ack_sem → wait
//             reduce_sem ≥ 1 → clear → push.
//   sender:   wait local reduce_ack_sem ≥ 1 → clear → NOC-write block → barrier →
//             inc receiver's reduce_sem.
// The ack credit guarantees the receiver has freed c_5 before the sender overwrites it.
// Every c_5 transaction is a full M_block × N_block block (partial edge blocks occupy a
// valid prefix), so both CBs' pointers stay pinned to base and the sender's local
// get_write_ptr(c_5) is always the partner's landing address.
//
// REDUCE_SEND cores' compute produces transposed sub-blocks column-major
// (n_sub outer, m_sub inner) so they arrive in the partner's row-major consume order;
// this kernel's loop mirrors that to size partial edge blocks correctly.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    const uint32_t sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    const uint32_t receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t Mpc = get_compile_time_arg_val(8);

#ifdef REDUCE_SEND
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(9);
    const uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t M_block = get_compile_time_arg_val(12);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(13);
    const uint32_t reduce_ack_sem_addr = get_semaphore(get_compile_time_arg_val(14));
#else  // REDUCE_RECV
    constexpr uint32_t padded_out_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(10);
    const uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t num_m_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t M_block = get_compile_time_arg_val(13);
    constexpr uint32_t num_n_blocks = get_compile_time_arg_val(14);
    const uint32_t reduce_ack_sem_addr = get_semaphore(get_compile_time_arg_val(15));
    constexpr auto out_tensor_args = TensorAccessorArgs<16>();
#endif

    uint32_t argidx = 0;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(argidx++);

#ifdef REDUCE_SEND
    const uint32_t partner_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_noc_y = get_arg_val<uint32_t>(argidx++);
#else  // REDUCE_RECV
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_M_tiles = get_arg_val<uint32_t>(argidx++);
#ifdef MIRROR_OUTPUT
    const uint32_t mirror_M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t mirror_N_start_tile = get_arg_val<uint32_t>(argidx++);
    constexpr uint32_t mirror_cb = tt::CBIndex::c_4;
#endif
    const uint32_t partner_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t partner_noc_y = get_arg_val<uint32_t>(argidx++);
#endif

    volatile tt_l1_ptr uint32_t* receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore_addr);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_semaphore_addr);

    constexpr uint32_t num_blocks = num_tiles / block_size;
    constexpr uint32_t N_block = M_block;
    // Fixed c_5 transaction size (= its capacity); partial edge blocks occupy a prefix.
    constexpr uint32_t reduce_block_capacity = M_block * N_block;

    for (uint32_t outer = 0; outer < num_m_blocks; outer++) {
        for (uint32_t inner = 0; inner < num_n_blocks; inner++) {
#ifdef REDUCE_SEND
            // Sender compute iterates column-major (see header comment)
            const uint32_t m_sub = inner;
            const uint32_t n_sub = outer;
#else
            const uint32_t m_sub = outer;
            const uint32_t n_sub = inner;
#endif
            const uint32_t M_start = m_sub * M_block;
            const uint32_t current_M_block = std::min(M_block, Mpc - M_start);
            const uint32_t N_start = n_sub * N_block;
            const uint32_t current_N = std::min(N_block, Mpc - N_start);

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
                volatile tt_l1_ptr uint32_t* reduce_ack_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_ack_sem_addr);
                const uint32_t partner_reduce_addr = get_write_ptr(reduce_cb);
                uint64_t partner_noc_addr = get_noc_addr(partner_noc_x, partner_noc_y, partner_reduce_addr);
                uint64_t partner_sem_noc = get_noc_addr(partner_noc_x, partner_noc_y, reduce_sem_addr);

                const uint32_t block_tiles = current_M_block * current_N;
                cb_wait_front(cb_out, reduce_block_capacity);
                // Wait for the partner's credit: its c_5 slot is free
                noc_semaphore_wait_min(reduce_ack_ptr, 1);
                noc_semaphore_set(reduce_ack_ptr, 0);
                const uint32_t l1_addr = get_read_ptr(cb_out);
                // Only the valid prefix carries data; the padding tail is never read
                noc_async_write(l1_addr, partner_noc_addr, block_tiles * out_tile_size);
                noc_async_write_barrier();
                noc_semaphore_inc(partner_sem_noc, 1);
                cb_pop_front(cb_out, reduce_block_capacity);
            }
#else  // REDUCE_RECV
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
            {
                const auto out_writer = TensorAccessor(out_tensor_args, out_addr, out_tile_size);

                for (uint32_t m = 0; m < current_M_block; m++) {
                    cb_wait_front(cb_out, N_block);
                    const uint32_t l1_read_addr = get_read_ptr(cb_out);
                    const uint32_t row = M_start_tile + M_start + m;
                    for (uint32_t n = 0; n < current_N; n++) {
                        const uint32_t col = N_start_tile + N_start + n;
                        if (row < logical_M_tiles && col < logical_M_tiles) {
                            const uint32_t tile_id = row * padded_out_tiles + col;
                            noc_async_write_page(tile_id, out_writer, l1_read_addr + n * out_tile_size);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, N_block);
                }
#ifdef MIRROR_OUTPUT
                // Mirror group n holds source column n transposed: tile (m, n) of this
                // block belongs at mirror position (col-of-tile, row-of-tile) globally.
                for (uint32_t n = 0; n < current_N; n++) {
                    cb_wait_front(mirror_cb, M_block);
                    const uint32_t l1_read_addr = get_read_ptr(mirror_cb);
                    const uint32_t row = mirror_M_start_tile + N_start + n;
                    for (uint32_t m = 0; m < current_M_block; m++) {
                        const uint32_t col = mirror_N_start_tile + M_start + m;
                        if (row < logical_M_tiles && col < logical_M_tiles) {
                            const uint32_t tile_id = row * padded_out_tiles + col;
                            noc_async_write_page(tile_id, out_writer, l1_read_addr + m * out_tile_size);
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(mirror_cb, M_block);
                }
#endif
            }
#endif
        }
    }

    // Flush every outstanding NOC transaction before the kernel exits: an
    // unflushed multicast/semaphore write or non-posted atomic can land after the
    // next dispatch re-initialises semaphores and strand a counted wait forever.
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
