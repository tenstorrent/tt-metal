// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Standalone kernel replicating GatherReduce3 from bliu/deepseek attention block.
//
// 112 sender cores (o_proj grid: 12x8 + 8x2) each write 64B to a single
// receiver core (11,9) scratch CB using UsePerCoreSenderIdx mode with
// noc_async_write + noc_async_write_barrier (matching production exactly).
// TRISC on the receiver reduces the two halves: out[i] = half0[i] + half1[i]
// using 32x32 tiles (2 tiles per half, ceil(56/32) = 2).
//
// Sender (NCRISC): uses per-core sender_idx to compute half-based offset,
//                   noc_async_write + noc_async_write_barrier, sem inc
// Receiver (BRISC): cb_reserve_back on scratch, waits on semaphore, cb_push_back
// TRISC (receiver only): add_half_tiles reduction (32x32 tiles)

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
        constexpr uint32_t half_num_cores = get_named_compile_time_arg_val("half_num_cores");
        constexpr uint32_t half_size_bytes = get_named_compile_time_arg_val("half_size_bytes");
        constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("scratch_cb");
        constexpr uint32_t sender_idx = get_named_compile_time_arg_val("sender_idx");

        uint32_t receiver_semaphore_addr = get_semaphore(get_named_compile_time_arg_val("receiver_semaphore_id"));

        // Setup sharded input buffer
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);

        // Compute half-based offset using per-core sender index (UsePerCoreSenderIdx mode)
        constexpr bool is_half0 = sender_idx < half_num_cores;
        constexpr uint32_t half_local_idx = is_half0 ? sender_idx : (sender_idx - half_num_cores);
        uint32_t dst_offset = (is_half0 ? 0 : half_size_bytes) + half_local_idx * data_size_bytes;

        // Get scratch CB base address
        uint32_t dst_base_addr = get_write_ptr(scratch_cb);

        // Build NOC addresses
        const uint64_t dst_noc_coord = get_noc_addr(dest_noc_x, dest_noc_y, 0);
        uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(dst_base_addr + dst_offset);
        uint64_t dst_sem_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;

        // Wait for source CB data
        cb_wait_front(src_cb, src_num_pages);
        uint32_t src_addr = get_read_ptr(src_cb);

        // NOC write sequence (matching bliu/deepseek: non-posted write + barrier)
        noc_async_write(src_addr, dst_data_noc_addr, data_size_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(dst_sem_noc_addr, 1);

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

        // Full HW init required for standalone kernel (production gets this from earlier micro-ops)
        binary_op_init_common(scratch_cb, scratch_cb, out_cb);

        // Match production add_half_tiles exactly
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
        tile_regs_release();
        cb_push_back(out_cb, num_tiles);

        cb_pop_front(scratch_cb, 2 * num_tiles);
    }
#endif
}
