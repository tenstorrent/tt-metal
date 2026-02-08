// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Unified CCL Broadcast kernel
// - NCRISC: Broadcast reader (reads local data into CB)
// - BRISC: Broadcast writer (sends to fabric / waits for data)
// - TRISC: No-op (dataflow only)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/broadcast.hpp"

void kernel_main() {
    using Broadcast = deepseek_b1_ops::Broadcast;

#if defined(COMPILE_FOR_NCRISC)
    // Reader CTArgs
    using BcastCTArgs = Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("cb0_id"),
        get_named_compile_time_arg_val("packet_size_in_pages"),
        get_named_compile_time_arg_val("tensor0_page_size"),
        get_named_compile_time_arg_val("is_sender"),
        get_named_compile_time_arg_val("core_noc_x"),
        get_named_compile_time_arg_val("core_noc_y"),
        get_named_compile_time_arg_val("is_secondary_sender"),
        get_named_compile_time_arg_val("is_active_broadcaster")>;

    // Runtime args:
    Broadcast::ReaderArgs bcast_args{
        get_common_arg_val<uint32_t>(0),  // tensor_address0
        get_common_arg_val<uint32_t>(1),  // tile_id_start
        get_common_arg_val<uint32_t>(2),  // tile_id_end
    };

#elif defined(COMPILE_FOR_BRISC)
    // Writer CTArgs
    using BcastCTArgs = Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("cb0_id"),
        get_named_compile_time_arg_val("packet_size_in_pages"),
        get_named_compile_time_arg_val("tensor0_page_size"),
        get_named_compile_time_arg_val("num_targets_forward_direction"),
        get_named_compile_time_arg_val("num_targets_backward_direction"),
        get_named_compile_time_arg_val("is_sender"),
        get_named_compile_time_arg_val("core_noc_x"),
        get_named_compile_time_arg_val("core_noc_y"),
        get_named_compile_time_arg_val("is_secondary_sender"),
        get_named_compile_time_arg_val("has_secondary_target"),
        get_named_compile_time_arg_val("has_reverse_secondary_connection"),
        get_named_compile_time_arg_val("start_distance_in_hops_forward"),
        get_named_compile_time_arg_val("range_hops_forward"),
        get_named_compile_time_arg_val("start_distance_in_hops_backward"),
        get_named_compile_time_arg_val("range_hops_backward"),
        get_named_compile_time_arg_val("using_persistent_buffers")>;

    // Writer runtime args
    Broadcast::WriterArgs bcast_args{
        get_common_arg_val<uint32_t>(0),   // tensor_address0
        get_common_arg_val<uint32_t>(1),   // out_ready_sem_bank_addr
        get_common_arg_val<uint32_t>(2),   // tile_id_start
        get_common_arg_val<uint32_t>(3),   // tile_id_end
        get_common_arg_val<uint32_t>(4),   // wait_output_semaphore
        get_common_arg_val<uint32_t>(5),   // reset_global_semaphore
        get_common_arg_val<uint32_t>(6),   // out_ready_sem_noc0_x
        get_common_arg_val<uint32_t>(7),   // out_ready_sem_noc0_y
        get_common_arg_val<uint32_t>(8),   // out_ready_sem_wait_value
        get_common_arg_val<uint32_t>(9),   // barrier_sem
        get_common_arg_val<uint32_t>(10),  // barrier_sem_noc0_x
        get_common_arg_val<uint32_t>(11),  // barrier_sem_noc0_y
        get_common_arg_val<uint32_t>(12),  // ring_index
        get_common_arg_val<uint32_t>(13),  // secondary_sync_sem
        get_common_arg_val<uint32_t>(14),  // num_connections
    };

#elif defined(COMPILE_FOR_TRISC)
    // TRISC: Compute args unused for broadcast
    Broadcast::ComputeArgs bcast_args{};
    Broadcast::ComputeCTArgs BcastCTArgs = {};
#endif

    // Execute ccl broadcast op
    Broadcast::Op<BcastCTArgs, true> bcast;
    bcast(bcast_args);
}
