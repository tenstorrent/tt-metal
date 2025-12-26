// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Gather unified kernel
// Single kernel file, compiles correctly for both BRISC (receiver) and NCRISC (sender)
// Note: This is a dataflow-only op - no compute kernel

#include "kernel_op_api.hpp"
#include "gather.hpp"

KERNEL_ENTRY {
    using Gather = deepseek_b1_ops::Gather;

// ============================================================================
// NCRISC (Sender) - DataMovementProcessor.RISCV_1
// Compile-time args: [dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id]
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = Gather::SenderCTArgs<
        get_compile_time_arg_val(0),  // dest_noc_x
        get_compile_time_arg_val(1),  // dest_noc_y
        get_compile_time_arg_val(2),  // data_size_bytes
        get_compile_time_arg_val(3)   // receiver_semaphore_id
        >;

    // Runtime args
    uint32_t input_data_addr = get_arg_val<uint32_t>(0);
    uint32_t receiver_data_addr = get_arg_val<uint32_t>(1);
    uint32_t offset = get_arg_val<uint32_t>(2);

    Gather::Op<CTArgs> gather;
    gather({input_data_addr, receiver_data_addr, offset});
// ============================================================================
// BRISC (Receiver) - DataMovementProcessor.RISCV_0
// Compile-time args: [noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id]
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = Gather::ReceiverCTArgs<
        get_compile_time_arg_val(0),  // noc0_num_senders
        get_compile_time_arg_val(1),  // noc1_num_senders
        get_compile_time_arg_val(2),  // noc0_receiver_semaphore_id
        get_compile_time_arg_val(3)   // noc1_receiver_semaphore_id
        >;

    Gather::Op<CTArgs> gather;
    gather();
#endif
}
KERNEL_END
