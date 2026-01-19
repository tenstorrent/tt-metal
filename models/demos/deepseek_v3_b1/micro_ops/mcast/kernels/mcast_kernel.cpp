// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Mcast unified kernel
// Single kernel file, compiles correctly for all RISC cores (NCRISC, BRISC, TRISC)
// Note: This is a dataflow-only op - TRISC is a no-op
//
// Sender runs on BRISC, Receiver runs on NCRISC
// - BRISC: Mcast sender (on sender core), no-op (on receiver cores)
// - NCRISC: Mcast receiver (on receiver cores), no-op (on sender core if not part of receiver grid)
// - TRISC: No-op (dataflow-only operation)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/mcast.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_receiver_core = get_named_compile_time_arg_val("is_receiver_core") == 1;
};

KERNEL_ENTRY {
    using Mcast = deepseek_b1_ops::Mcast;

// ============================================================================
// NCRISC (Receiver) - ReaderConfigDescriptor compiles as NCRISC
// Named compile-time args: mcast receiver params
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using McastCTArgs = Mcast::ReceiverCTArgs;

    // Setup sharded input buffer on sender core so BRISC can read it
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast_src_cb, mcast_src_num_pages);
    }

    // Mcast receiver args (from compile-time args, passed to op as runtime args)
    Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

// ============================================================================
// BRISC (Sender) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: mcast sender params
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using McastCTArgs = Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_sender_core && Core::is_receiver_core>;  // loopback = sender is also a receiver

    // Mcast CB index from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");

    // Mcast receiver data address (passed from Python as runtime arg, this is the output tensor's buffer address)
    uint32_t mcast_receiver_data_addr = get_arg_val<uint32_t>(0);

    // Mcast sender args (from compile-time args, passed to op as runtime args)
    Mcast::SenderArgs mcast_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        mcast_src_cb,
        get_named_compile_time_arg_val("mcast_src_num_pages"),
        get_read_ptr(mcast_src_cb),
        mcast_receiver_data_addr,
    };

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Mcast is a dataflow-only op - TRISC is a no-op
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using McastCTArgs = Mcast::ComputeCTArgs;

    // Mcast compute args (no-op for TRISC)
    Mcast::ComputeArgs mcast_args{};
#endif

    // Execute mcast operation
    // Template params: <CTArgsT, IsSenderCore, IsMcastGridCore, IsReceiverCore, pop_src>
    // - IsSenderCore: this core sends mcast data (BRISC sender)
    // - IsMcastGridCore: this core is part of the mcast destination grid (receives semaphore)
    // - IsReceiverCore: this core receives data into CB (NCRISC receiver)
    // - pop_src: whether to pop the source CB after sending
    Mcast::Op<McastCTArgs, Core::is_sender_core, Core::is_receiver_core, Core::is_receiver_core, true> mcast;
    mcast.init(mcast_args);
    mcast(mcast_args);
    mcast.teardown();
}
KERNEL_END
