// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Gather unified kernel
// Single kernel file, compiles correctly for both BRISC (receiver) and NCRISC (sender)
// Note: This is a dataflow-only op - no compute kernel

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/gather.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;
};

KERNEL_ENTRY {
    using Gather = deepseek_b1_ops::Gather;

// ============================================================================
// NCRISC (Sender) - DataMovementProcessor.RISCV_1
// Named compile-time args: dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id,
//                          src_cb, src_num_pages, sender_grid_start_x/y, sender_grid_end_x/y,
//                          row_major
// Runtime args: receiver_data_addr (output tensor buffer address, passed from Python)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded input buffer (makes data available for cb_wait_front)
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t src_cb = get_named_compile_time_arg_val("gather_src_cb");
        constexpr uint32_t src_num_pages = get_named_compile_time_arg_val("gather_src_num_pages");
        unified_kernels::setup_sharded_buffer(src_cb, src_num_pages);
    }

    // Get receiver data address from runtime arg (dst CB doesn't exist on sender cores)
    uint32_t receiver_data_addr = get_arg_val<uint32_t>(0);

    // Gather sender args (from compile-time args, passed to op as runtime args)
    Gather::SenderArgs gather_args{
        get_named_compile_time_arg_val("gather_dest_noc_x"),
        get_named_compile_time_arg_val("gather_dest_noc_y"),
        get_named_compile_time_arg_val("gather_data_size_bytes"),
        get_named_compile_time_arg_val("gather_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_src_cb"),
        get_named_compile_time_arg_val("gather_src_num_pages"),
        get_named_compile_time_arg_val("gather_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather_row_major"),
        receiver_data_addr,  // receiver_data_addr from runtime arg (output tensor buffer address)
    };

// ============================================================================
// BRISC (Receiver) - DataMovementProcessor.RISCV_0
// Named compile-time args: noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id,
//                          noc1_receiver_semaphore_id, dst_cb, dst_num_pages
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Gather receiver args (from compile-time args, passed to op as runtime args)
    Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

// ============================================================================
// TRISC (Compute) - No-op (gather is dataflow only)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Gather compute args (no-op for TRISC)
    Gather::ComputeArgs gather_args{};
#endif

    // Execute gather operation
    // pop_src = true (input is consumed after gather)
    Gather::Op<Core::is_sender_core, Core::is_receiver_core, true> gather;
    gather(gather_args);
}
KERNEL_END
