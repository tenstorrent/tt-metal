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
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs<
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_tiny_tile")>;

    // Mcast CB indices and page counts from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");

    // Mcast sender CTArgs from named compile-time args
    using McastCTArgs = Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_loopback") == 1,
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        mcast_src_cb,
        mcast_src_num_pages>;

    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs>::ReaderArgs rmsnorm_rt_args;
    rmsnorm_rt_args.epsilon = get_arg_val<uint32_t>(0);
    rmsnorm_rt_args.scalar = get_arg_val<uint32_t>(1);

    Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core>::SenderArgs mcast_rt_args;
    mcast_rt_args.input_data_addr = get_read_ptr(mcast_src_cb);
    mcast_rt_args.mcast_receiver_data_addr = get_write_ptr(mcast_dst_cb);

    // Matmul reader CTArgs from named compile-time args
    using Matmul = deepseek_b1_ops::Matmul;
    using MatmulCTArgs = Matmul::ReaderCTArgs<
        get_named_compile_time_arg_val("matmul_in0_cb"),
        get_named_compile_time_arg_val("matmul_in1_cb"),
        get_named_compile_time_arg_val("matmul_num_tiles_k")>;

    // Gather sender CTArgs from named compile-time args
    using Gather = deepseek_b1_ops::Gather;
    using GatherCTArgs = Gather::SenderCTArgs<
        get_named_compile_time_arg_val("gather_dest_noc_x"),
        get_named_compile_time_arg_val("gather_dest_noc_y"),
        get_named_compile_time_arg_val("gather_data_size_bytes"),
        get_named_compile_time_arg_val("gather_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_src_cb"),
        get_named_compile_time_arg_val("gather_src_num_pages"),
        get_named_compile_time_arg_val("gather_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather_sender_grid_end_y")>;
    Gather::Op<GatherCTArgs, Core::is_matmul_core, Core::is_input_core>::SenderArgs gather_rt_args;
    gather_rt_args.receiver_data_addr = get_arg_val<uint32_t>(2);  // gather receiver data address

// ============================================================================
// BRISC (Writer + Mcast Receiver) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: rmsnorm writer, mcast receiver, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs<
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles")>;

    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs>::WriterArgs rmsnorm_rt_args;

    // Mcast CB info for receiver
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    constexpr uint32_t mcast_dst_num_pages = get_named_compile_time_arg_val("mcast_dst_num_pages");

    // Mcast receiver CTArgs from named compile-time args
    using McastCTArgs = Mcast::ReceiverCTArgs<
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        mcast_dst_cb,
        mcast_dst_num_pages>;
    Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core>::ReceiverArgs mcast_rt_args;

    // Matmul writer CTArgs from named compile-time args
    using Matmul = deepseek_b1_ops::Matmul;
    using MatmulCTArgs = Matmul::WriterCTArgs<get_named_compile_time_arg_val("matmul_out_cb")>;

    // Gather receiver CTArgs from named compile-time args
    using Gather = deepseek_b1_ops::Gather;
    using GatherCTArgs = Gather::ReceiverCTArgs<
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages")>;
    Gather::Op<GatherCTArgs, Core::is_matmul_core, Core::is_input_core>::ReceiverArgs gather_rt_args;

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: rmsnorm compute, matmul compute
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_interm_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_named_compile_time_arg_val("rmsnorm_fp32_acc"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_epsilon_index"),
        get_named_compile_time_arg_val("rmsnorm_scalar_index"),
        true  // pop_input
        >;

    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs>::ComputeArgs rmsnorm_rt_args;

    // Mcast compute CTArgs (no-op for TRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using McastCTArgs = Mcast::ComputeCTArgs;
    Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core>::ComputeArgs mcast_rt_args;

    // Matmul compute CTArgs from named compile-time args
    using Matmul = deepseek_b1_ops::Matmul;
    using MatmulCTArgs = Matmul::ComputeCTArgs<
        get_named_compile_time_arg_val("matmul_in0_cb"),
        get_named_compile_time_arg_val("matmul_in1_cb"),
        get_named_compile_time_arg_val("matmul_out_cb"),
        get_named_compile_time_arg_val("matmul_interm_cb"),
        get_named_compile_time_arg_val("matmul_num_tiles_k"),
        get_named_compile_time_arg_val("matmul_fp32_acc")>;

    // Gather compute CTArgs (no-op for TRISC)
    using Gather = deepseek_b1_ops::Gather;
    using GatherCTArgs = Gather::ComputeCTArgs;
    Gather::Op<GatherCTArgs, Core::is_matmul_core, Core::is_input_core>::ComputeArgs gather_rt_args;
#endif

    // ========================================================================
    // Input core: RMSNorm + Mcast send
    // ========================================================================
    {
        DeviceZoneScopedN("RMSNORM");
        if constexpr (Core::is_input_core) {
            deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs> rmsnorm;
            rmsnorm(rmsnorm_rt_args);
        }
    }

    {
        DeviceZoneScopedN("MCAST");
        // Mcast: NCRISC sends from input core, BRISC receives on matmul cores, TRISC no-op
        Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core> mcast;
        mcast(mcast_rt_args);
    }

    // ========================================================================
    // Matmul operation
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        if constexpr (Core::is_matmul_core) {
            Matmul::Op<MatmulCTArgs> matmul;
            matmul();
        }
    }

    // ========================================================================
    // Gather: matmul cores (senders) -> input core (receiver)
    // NCRISC sends from matmul cores, BRISC receives on input core, TRISC no-op
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        Gather::Op<GatherCTArgs, Core::is_matmul_core, Core::is_input_core> gather;
        gather(gather_rt_args);
    }
}
KERNEL_END
