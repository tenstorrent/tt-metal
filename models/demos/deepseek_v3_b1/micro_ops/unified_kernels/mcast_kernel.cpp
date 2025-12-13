// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Mcast unified kernel
// Single kernel file, compiles correctly for both NCRISC (sender) and BRISC (receiver)
// Note: This is a dataflow-only op - no compute kernel

#include "kernel_op_api.hpp"
#include "mcast.hpp"

KERNEL_ENTRY {
    using Mcast = deepseek_b1_ops::Mcast;

// ============================================================================
// NCRISC (Sender) - DataMovementProcessor.RISCV_1
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_loopback"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast0_data_size_bytes")>;

    Mcast::Op<CTArgs>::SenderArgs rt_args;
    rt_args.input_data_addr = get_common_arg_val<uint32_t>(0);
    rt_args.mcast_receiver_data_addr = get_common_arg_val<uint32_t>(1);

// ============================================================================
// BRISC (Receiver) - DataMovementProcessor.RISCV_0
// Compile-time args: [data_receiver_semaphore_id]
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = Mcast::ReceiverCTArgs<get_compile_time_arg_val(0)>;

    Mcast::Op<CTArgs>::ReceiverArgs rt_args;
#endif

    Mcast::Op<CTArgs> mcast;
    mcast(rt_args);
}
KERNEL_END
