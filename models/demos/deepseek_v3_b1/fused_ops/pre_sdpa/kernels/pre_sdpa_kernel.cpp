// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Pre-SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: RMSNorm + Mcast
// - NCRISC: RMSNorm reader + Mcast sender (on input core)
// - BRISC: RMSNorm writer + Mcast receiver (on matmul cores)
// - TRISC: RMSNorm compute

#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/kernel_op_api.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/rmsnorm.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/mcast.hpp"
#include "models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/unified_core_descriptor.hpp"

KERNEL_ENTRY {
    using RMSNorm = deepseek_b1_ops::RMSNorm;

// ============================================================================
// NCRISC (Reader + Mcast Sender) - ReaderConfigDescriptor compiles as NCRISC
// Compile-time args: [input_cb, scalars_cb, gamma_cb, num_tiles, tiny_tile]
// Named compile-time args: mcast sender parameters
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = RMSNorm::ReaderCTArgs<
        get_compile_time_arg_val(0),  // input_cb
        get_compile_time_arg_val(1),  // scalars_cb
        get_compile_time_arg_val(2),  // gamma_cb
        get_compile_time_arg_val(3),  // num_tiles
        get_compile_time_arg_val(4)   // tiny_tile
        >;

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
        get_named_compile_time_arg_val("mcast_data_size_bytes")>;

    RMSNorm::Op<RMSNormCTArgs>::ReaderArgs rmsnorm_rt_args;
    rmsnorm_rt_args.epsilon = get_arg_val<uint32_t>(0);
    rmsnorm_rt_args.scalar = get_arg_val<uint32_t>(1);

    // Mcast CB indices from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    Mcast::Op<McastCTArgs>::SenderArgs mcast_rt_args;
    mcast_rt_args.input_data_addr = get_read_ptr(mcast_src_cb);
    mcast_rt_args.mcast_receiver_data_addr = get_write_ptr(mcast_dst_cb);

// ============================================================================
// BRISC (Writer + Mcast Receiver) - WriterConfigDescriptor compiles as BRISC
// Compile-time args: [output_cb, num_tiles]
// Named compile-time args: mcast receiver parameters
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = RMSNorm::WriterCTArgs<
        get_compile_time_arg_val(0),  // output_cb
        get_compile_time_arg_val(1)   // num_tiles
        >;

    RMSNorm::Op<RMSNormCTArgs>::WriterArgs rmsnorm_rt_args;

    // Mcast receiver CTArgs from named compile-time args
    using McastCTArgs = Mcast::ReceiverCTArgs<get_named_compile_time_arg_val("mcast_data_receiver_semaphore")>;
    Mcast::Op<McastCTArgs>::ReceiverArgs mcast_rt_args;

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Compile-time args: [input_cb, scalars_cb, interm_cb, gamma_cb, output_cb,
//                     fp32_acc, num_tiles, epsilon_index, scalar_index]
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using RMSNormCTArgs = RMSNorm::ComputeCTArgs<
        get_compile_time_arg_val(0),  // input_cb
        get_compile_time_arg_val(1),  // scalars_cb
        get_compile_time_arg_val(2),  // interm_cb
        get_compile_time_arg_val(3),  // gamma_cb
        get_compile_time_arg_val(4),  // output_cb
        get_compile_time_arg_val(5),  // fp32_acc
        get_compile_time_arg_val(6),  // num_tiles
        get_compile_time_arg_val(7),  // epsilon_index
        get_compile_time_arg_val(8),  // scalar_index
        true                          // pop_input
        >;

    RMSNorm::Op<RMSNormCTArgs>::ComputeArgs rmsnorm_rt_args;
#endif

    // Print out all compile time args by RISC
    // #if defined(COMPILE_FOR_NCRISC)
    //     DPRINT << "NCRISC Compile-time Args:" << ENDL();
    //     DPRINT << "mcast_src_cb: " << mcast_src_cb << ENDL();
    //     DPRINT << "mcast_dst_cb: " << mcast_dst_cb << ENDL();
    // #elif defined(COMPILE_FOR_BRISC)
    //     DPRINT << "BRISC Compile-time Args:" << ENDL();
    //     DPRINT << "output_cb: " << get_compile_time_arg_val(0) << ENDL();
    //     DPRINT << "num_tiles: " << get_compile_time_arg_val(1) << ENDL();
    // #elif defined(COMPILE_FOR_TRISC)
    //     DPRINT << "TRISC Compile-time Args:" << ENDL();
    //     DPRINT << "input_cb: " << get_compile_time_arg_val(0) << ENDL();
    //     DPRINT << "scalars_cb: " << get_compile_time_arg_val(1) << ENDL();
    //     DPRINT << "interm_cb: " << get_compile_time_arg_val(2) << ENDL();
    //     DPRINT << "gamma_cb: " << get_compile_time_arg_val(3) << ENDL();
    //     DPRINT << "output_cb: " << get_compile_time_arg_val(4) << ENDL();
    //     DPRINT << "fp32_acc: " << get_compile_time_arg_val(5) << ENDL();
    //     DPRINT << "num_tiles: " << get_compile_time_arg_val(6) << ENDL();
    //     DPRINT << "epsilon_index: " << get_compile_time_arg_val(7) << ENDL();
    //     DPRINT << "scalar_index: " << get_compile_time_arg_val(8) << ENDL();
    // #endif

    // Use UnifiedCoreDescriptor for compile-time role checks
    using Core = pre_sdpa::UnifiedCoreDescriptor;

    // ========================================================================
    // Input core: RMSNorm + Mcast send
    // ========================================================================
    if constexpr (Core::is_input_core) {
        RMSNorm::Op<RMSNormCTArgs> rmsnorm;
        rmsnorm(rmsnorm_rt_args);
    }

    // Matmul cores: mcast (dataflow only - NCRISC sends, BRISC receives)
    // #if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    //     if constexpr (Core::is_matmul_core) {
    //         Mcast::Op<McastCTArgs> mcast;
    //         mcast(mcast_rt_args);
    //     }
    // #endif

    DPRINT << "Pre-SDPA kernel completed" << ENDL();
}
KERNEL_END
