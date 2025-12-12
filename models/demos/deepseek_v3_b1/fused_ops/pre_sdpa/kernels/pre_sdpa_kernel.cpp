// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Pre-SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: RMSNorm + Mcast + Matmul + Gather
// - NCRISC: RMSNorm reader + Mcast sender (on input core), Matmul reader + Gather sender (on matmul cores)
// - BRISC: RMSNorm writer + Mcast receiver (on matmul cores), Matmul writer (on matmul cores), Gather receiver (on
// input core)
// - TRISC: RMSNorm compute (on input core), Matmul compute (on matmul cores)

#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/kernel_op_api.hpp"
#include "models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/unified_core_descriptor.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/rmsnorm.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/mcast.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/matmul.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/gather.hpp"

KERNEL_ENTRY {
    // Use UnifiedCoreDescriptor for compile-time role checks
    using Core = pre_sdpa::UnifiedCoreDescriptor;

// ============================================================================
// NCRISC (Reader + Mcast Sender) - ReaderConfigDescriptor compiles as NCRISC
// Named compile-time args: rmsnorm reader, mcast sender, matmul reader, gather sender
// Runtime args: [epsilon, scalar, gather_addr]
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs =
        deepseek_b1_ops::RMSNorm::ReaderCTArgs<get_named_compile_time_arg_val("rmsnorm_tiny_tile") == 1>;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_rt_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_arg_val<uint32_t>(0),  // epsilon
        get_arg_val<uint32_t>(1),  // scalar
    };

    // Mcast sender compile-time args (only what must be compile-time)
    using McastCTArgs = Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_loopback") == 1,
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1>;

    // Mcast CB indices from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

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
        get_write_ptr(mcast_dst_cb),
    };

    // Matmul reader args (from compile-time args, passed to op as runtime args)
    using Matmul = deepseek_b1_ops::Matmul;
    Matmul::ReaderArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_num_tiles"),
    };

    // Gather sender args (from compile-time args, passed to op as runtime args)
    using Gather = deepseek_b1_ops::Gather;
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
        get_arg_val<uint32_t>(2),  // receiver_data_addr
    };

// ============================================================================
// BRISC (Writer + Mcast Receiver) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: rmsnorm writer, mcast receiver, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;

    // RMSNorm writer runtime args
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_rt_args{
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
    };

    // Mcast receiver compile-time args (none needed for receiver)
    using McastCTArgs = Mcast::ReceiverCTArgs;

    // Mcast receiver args (from compile-time args, passed to op as runtime args)
    Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul writer args (from compile-time args, passed to op as runtime args)
    using Matmul = deepseek_b1_ops::Matmul;
    Matmul::WriterArgs matmul_args{
        get_named_compile_time_arg_val("matmul_out"),
    };

    // Gather receiver args (from compile-time args, passed to op as runtime args)
    using Gather = deepseek_b1_ops::Gather;
    Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: rmsnorm compute, matmul compute
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        true  // pop_input
        >;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_rt_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_interm_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_epsilon_index"),
        get_named_compile_time_arg_val("rmsnorm_scalar_index"),
    };

    // Mcast compute compile-time args (none needed, no-op for TRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using McastCTArgs = Mcast::ComputeCTArgs;
    Mcast::ComputeArgs mcast_args{};

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    using Matmul = deepseek_b1_ops::Matmul;
    Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    using Gather = deepseek_b1_ops::Gather;
    Gather::ComputeArgs gather_args{};
#endif

    // ========================================================================
    // Input core: RMSNorm + Mcast send
    // ========================================================================
    {
        DeviceZoneScopedN("RMSNORM");
        deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core> rmsnorm;
        rmsnorm(rmsnorm_rt_args);
    }

    {
        DeviceZoneScopedN("MCAST");
        // Mcast: NCRISC sends from input core, BRISC receives on matmul cores, TRISC no-op
        Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core> mcast;
        mcast(mcast_args);
    }

    // ========================================================================
    // Matmul operation
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        Matmul::Op<Core::is_matmul_core> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Gather: matmul cores (senders) -> input core (receiver)
    // NCRISC sends from matmul cores, BRISC receives on input core, TRISC no-op
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        Gather::Op<Core::is_matmul_core, Core::is_input_core> gather;
        gather(gather_args);
    }
}
KERNEL_END
