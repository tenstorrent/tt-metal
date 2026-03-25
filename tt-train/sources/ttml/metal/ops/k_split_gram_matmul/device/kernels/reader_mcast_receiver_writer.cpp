// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast receiver + output/reduce kernel with M_block x N_block streaming.
// Phase 1: receives block_size tiles per handshake (for msb: for nsb: for blk:).
// Phase 2 (default):      writes compute output to DRAM row by row.
// Phase 2 (REDUCE_SEND):  NOC-writes compute output to partner's reduce CB.
// Phase 2 (REDUCE_RECV):  waits for partner's partial in reduce CB, then writes combined to DRAM.
// Phase 2 happens per-msb (after all nsb for that msb complete).

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
    constexpr uint32_t M_num_subblocks = get_compile_time_arg_val(11);
    constexpr uint32_t M_block = get_compile_time_arg_val(12);
    constexpr uint32_t N_num_subblocks = get_compile_time_arg_val(13);
#else
    constexpr uint32_t padded_out_tiles = get_compile_time_arg_val(9);
#ifdef REDUCE_RECV
    constexpr uint32_t reduce_cb = get_compile_time_arg_val(10);
    uint32_t reduce_sem_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t M_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t M_block = get_compile_time_arg_val(13);
    constexpr uint32_t N_num_subblocks = get_compile_time_arg_val(14);
    constexpr auto out_tensor_args = TensorAccessorArgs<15>();
#else
    constexpr uint32_t M_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t M_block = get_compile_time_arg_val(11);
    constexpr uint32_t N_num_subblocks = get_compile_time_arg_val(12);
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

    for (uint32_t msb = 0; msb < M_num_subblocks; msb++) {
        uint32_t M_start = msb * M_block;
        uint32_t current_M_block = (M_block < Mpc - M_start) ? M_block : (Mpc - M_start);
        uint32_t msb_tiles = current_M_block * Mpc;

        // --- Phase 1: receive tiles via multicast (for all nsb) ---
        for (uint32_t nsb = 0; nsb < N_num_subblocks; nsb++) {
            for (uint32_t blk = 0; blk < num_blocks; blk++) {
                cb_reserve_back(cb_id, block_size);

                noc_semaphore_set(receiver_sem_ptr, INVALID);
                noc_semaphore_inc(sender_sem_noc_addr, 1);

                noc_semaphore_wait(receiver_sem_ptr, VALID);

                cb_push_back(cb_id, block_size);
            }
        }

        // --- Phase 2 (per-msb) ---
#ifdef REDUCE_SEND
        // Send own partial to partner's reduce CB via NOC write
        uint32_t partner_reduce_addr = get_write_ptr(reduce_cb);
        uint64_t partner_noc_addr = get_noc_addr(partner_noc_x, partner_noc_y, partner_reduce_addr);
        uint64_t partner_sem_noc = get_noc_addr(partner_noc_x, partner_noc_y, reduce_sem_addr);

        cb_wait_front(cb_out, msb_tiles);
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write(l1_addr, partner_noc_addr, msb_tiles * out_tile_size);
        noc_async_write_barrier();
        noc_semaphore_inc(partner_sem_noc, 1);
        cb_pop_front(cb_out, msb_tiles);
#else
        // Receive partner's partial into reduce CB (if REDUCE_RECV)
#ifdef REDUCE_RECV
        volatile tt_l1_ptr uint32_t* reduce_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sem_addr);
        cb_reserve_back(reduce_cb, msb_tiles);
        noc_semaphore_wait(reduce_sem_ptr, 1);
        noc_semaphore_set(reduce_sem_ptr, 0);
        cb_push_back(reduce_cb, msb_tiles);
#endif

        const auto out_writer = TensorAccessor(out_tensor_args, out_addr, out_tile_size);

#ifdef PER_NSB_REDUCTION
        // Row-major write: compute pushes rows of Mpc tiles each
        for (uint32_t m = 0; m < current_M_block; m++) {
            cb_wait_front(cb_out, Mpc);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t row = M_start_tile + M_start + m;
            for (uint32_t n = 0; n < Mpc; n++) {
                uint32_t col = N_start_tile + n;
                if (row < logical_M_tiles && col < logical_M_tiles) {
                    uint32_t tile_id = row * padded_out_tiles + col;
                    noc_async_write_tile(tile_id, out_writer, l1_read_addr + n * out_tile_size);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, Mpc);
        }
#ifdef MIRROR_OUTPUT
        for (uint32_t n = 0; n < Mpc; n++) {
            cb_wait_front(mirror_cb, current_M_block);
            uint32_t l1_read_addr = get_read_ptr(mirror_cb);
            uint32_t col = mirror_N_start_tile + n;
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
#else
        // Column-major write: Mpc × M_block column-major layout in c_out
        for (uint32_t n = 0; n < Mpc; n++) {
            cb_wait_front(cb_out, current_M_block);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t col = N_start_tile + n;
            for (uint32_t m = 0; m < current_M_block; m++) {
                uint32_t row = M_start_tile + M_start + m;
                if (row < logical_M_tiles && col < logical_M_tiles) {
                    uint32_t tile_id = row * padded_out_tiles + col;
                    noc_async_write_tile(tile_id, out_writer, l1_read_addr + m * out_tile_size);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, current_M_block);
        }

#ifdef MIRROR_OUTPUT
        for (uint32_t n = 0; n < Mpc; n++) {
            cb_wait_front(mirror_cb, current_M_block);
            uint32_t l1_read_addr = get_read_ptr(mirror_cb);
            uint32_t col = mirror_N_start_tile + n;
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
#endif
#endif
    }
}
