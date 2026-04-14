// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Standalone kernel replicating the gather + reduce portion of GatherReduce
// for ND issue reproduction.
//
// 96 sender cores (12x8) each write 64B to a single receiver core's scratch CB
// using the same half-based offset and NOC API sequence as GatherReduce.
// TRISC on the receiver core reduces the two halves: out[i] = half0[i] + half1[i]
//
// Sender (NCRISC): computes half-based offset, NOC writes 64B to receiver scratch CB, sem inc
// Receiver (BRISC): cb_reserve_back on scratch, waits on semaphore, cb_push_back
// TRISC (receiver only): add_half_tiles reduction

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (is_sender_core) {
        constexpr uint32_t src_cb = get_named_compile_time_arg_val("src_cb");
        constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("src_num_pages");
        constexpr uint32_t dest_noc_x = get_named_compile_time_arg_val("dest_noc_x");
        constexpr uint32_t dest_noc_y = get_named_compile_time_arg_val("dest_noc_y");
        constexpr uint32_t data_size_bytes = get_named_compile_time_arg_val("data_size_bytes");
        constexpr uint32_t grid_start_x = get_named_compile_time_arg_val("grid_start_x");
        constexpr uint32_t grid_start_y = get_named_compile_time_arg_val("grid_start_y");
        constexpr uint32_t grid_end_x = get_named_compile_time_arg_val("grid_end_x");
        constexpr uint32_t grid_end_y = get_named_compile_time_arg_val("grid_end_y");
        constexpr uint32_t half_num_cores = get_named_compile_time_arg_val("half_num_cores");
        constexpr uint32_t half_size_bytes = get_named_compile_time_arg_val("half_size_bytes");
        constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");

        uint32_t receiver_semaphore_addr = get_semaphore(get_named_compile_time_arg_val("receiver_semaphore_id"));

        // Setup sharded input buffer
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);

        // Compute half-based offset (identical to GatherReduce)
        const auto half_info = unified_kernels::get_split_half_core_info<true>(
            grid_start_x, grid_start_y, grid_end_x, grid_end_y, half_num_cores);
        uint32_t dst_offset = (half_info.is_half0 ? 0 : half_size_bytes) + half_info.half_local_idx * data_size_bytes;

        // Get scratch CB base address (same L1 addr on all cores)
        uint32_t dst_base_addr = get_write_ptr(scratch_cb);

        // Build NOC addresses
        const uint64_t dst_noc_coord = get_noc_addr(dest_noc_x, dest_noc_y, 0);
        uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(dst_base_addr + dst_offset);
        uint64_t dst_sem_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;

        // Wait for source CB data
        cb_wait_front(src_cb, src_num_pages);
        uint32_t src_addr = get_read_ptr(src_cb);

        // NOC write sequence (identical to GatherReduce)
        noc_async_write_one_packet<true, true>(src_addr, dst_data_noc_addr, data_size_bytes);
        // BH does not support posted atomics due to a bug
        noc_semaphore_inc(dst_sem_noc_addr, 1);
        noc_async_posted_writes_flushed();

        cb_pop_front(src_cb, src_num_pages);
        noc_async_atomic_barrier();
    }
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (is_receiver_core) {
        constexpr uint32_t noc0_num_senders = get_named_compile_time_arg_val("noc0_num_senders");
        constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
        constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
        uint32_t noc0_receiver_semaphore_addr =
            get_semaphore(get_named_compile_time_arg_val("noc0_receiver_semaphore_id"));

        volatile tt_l1_ptr uint32_t* sem_ptr = (volatile tt_l1_ptr uint32_t*)noc0_receiver_semaphore_addr;

        // Reserve space in scratch CB for both halves (2 * num_tiles)
        cb_reserve_back(scratch_cb, 2 * num_tiles);

        // Wait for all senders to complete their NOC writes
        noc_semaphore_wait(sem_ptr, noc0_num_senders);
        noc_semaphore_set(sem_ptr, 0);

        // Push data to make it available for TRISC
        cb_push_back(scratch_cb, 2 * num_tiles);
    }
#elif defined(COMPILE_FOR_TRISC)
    if constexpr (is_receiver_core) {
        constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
        constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
        constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

        // WORKAROUND: Reset out_cb pack state on TRISC2 — firmware may not
        // fully reinitialize LocalCBInterface between generic_op executions.
#if defined(UCK_CHLKC_PACK)
        {
            LocalCBInterface& oi = get_local_cb_interface(out_cb);
            oi.fifo_wr_ptr = oi.fifo_limit - oi.fifo_size;
            oi.fifo_wr_tile_ptr = 0;
            oi.fifo_num_pages = num_tiles;
            oi.tiles_acked_received_init = 0;
        }
#endif

        // Reduce: out[i] = scratch[i] + scratch[i + num_tiles] for i in [0, num_tiles)
        // Identical to GatherReduce::add_half_tiles
        reconfig_data_format<false, true>(scratch_cb, scratch_cb);
        pack_reconfig_data_format<true>(out_cb);
        add_tiles_init(scratch_cb, scratch_cb);

        cb_wait_front(scratch_cb, 2 * num_tiles);
        cb_reserve_back(out_cb, num_tiles);

        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            add_tiles(scratch_cb, scratch_cb, i, num_tiles + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles; i++) {
            pack_tile(i, out_cb);
        }
        cb_push_back(out_cb, num_tiles);
        tile_regs_release();

        cb_pop_front(scratch_cb, 2 * num_tiles);

        // Consume output CB to reset tiles_acked (no downstream consumer in this test)
        cb_wait_front(out_cb, num_tiles);
        cb_pop_front(out_cb, num_tiles);
    }
#endif
}
