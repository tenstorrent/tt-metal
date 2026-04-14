// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Standalone kernel replicating the gather portion of GatherReduce
// (no TRISC reduction) for ND issue reproduction.
//
// 96 sender cores (12x8) each write 64B to a single receiver core
// using the same half-based offset and NOC API sequence as GatherReduce.
//
// Sender (NCRISC): computes half-based offset, NOC writes 64B to receiver, sem inc
// Receiver (BRISC): waits on semaphore for all senders
// TRISC: no-op

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (is_sender_core) {
        // Compile-time args (same names as GatherReduce in attention block)
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

        // Semaphore address (resolved from semaphore ID)
        uint32_t receiver_semaphore_addr = get_semaphore(get_named_compile_time_arg_val("receiver_semaphore_id"));

        // Runtime arg: receiver's output buffer L1 address
        uint32_t receiver_data_addr = get_common_arg_val<uint32_t>(0);

        // Setup sharded input buffer (makes data available for cb_wait_front)
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);

        // Compute half-based offset (identical to GatherReduce)
        const auto half_info = unified_kernels::get_split_half_core_info<true>(
            grid_start_x, grid_start_y, grid_end_x, grid_end_y, half_num_cores);
        uint32_t dst_offset = (half_info.is_half0 ? 0 : half_size_bytes) + half_info.half_local_idx * data_size_bytes;

        // Build NOC addresses
        const uint64_t dst_noc_coord = get_noc_addr(dest_noc_x, dest_noc_y, 0);
        uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(receiver_data_addr + dst_offset);
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
        uint32_t noc0_receiver_semaphore_addr =
            get_semaphore(get_named_compile_time_arg_val("noc0_receiver_semaphore_id"));

        volatile tt_l1_ptr uint32_t* sem_ptr = (volatile tt_l1_ptr uint32_t*)noc0_receiver_semaphore_addr;

        // Wait for all senders to complete their NOC writes
        noc_semaphore_wait(sem_ptr, noc0_num_senders);
        noc_semaphore_set(sem_ptr, 0);
    }
#elif defined(COMPILE_FOR_TRISC)
    // No-op: gather only, no reduction
#endif
}
