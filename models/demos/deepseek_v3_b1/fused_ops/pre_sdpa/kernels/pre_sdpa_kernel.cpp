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
#include "models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/kernel_setup.hpp"
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
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs =
        deepseek_b1_ops::RMSNorm::ReaderCTArgs<get_named_compile_time_arg_val("rmsnorm_tiny_tile") == 1>;
    using McastCTArgs = deepseek_b1_ops::Mcast::
        SenderCTArgs<get_named_compile_time_arg_val("mcast_num_cores"), Core::is_input_core && Core::is_matmul_core>;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_arg_val<uint32_t>(0),  // epsilon
        get_arg_val<uint32_t>(1),  // scalar (1/sqrt(7168))
    };

    // Mcast CB indices from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

    // Mcast sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::SenderArgs mcast_args{
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

    // Matmul reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // Gather sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::SenderArgs gather_args{
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

    // Gather dst CB args for copy operation (receiver CB, used on input core)
    constexpr uint32_t gather_dst_cb = get_named_compile_time_arg_val("gather_dst_cb");
    constexpr uint32_t gather_dst_num_pages = get_named_compile_time_arg_val("gather_dst_num_pages");

    // RMSNorm2 reader args (uses same scalars_cb, different scalar value)
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm2_args{
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_arg_val<uint32_t>(0),  // epsilon (same as rmsnorm1)
        get_arg_val<uint32_t>(3),  // scalar2 (1/sqrt(1536))
    };

// ============================================================================
// BRISC (Writer + Mcast Receiver) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: rmsnorm writer, mcast receiver, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;

    // RMSNorm writer args (BRISC is no-op)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

    // RMSNorm2 writer args (BRISC is no-op)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm2_args{};

    // Mcast receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    // Gather receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::ReceiverArgs gather_args{
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
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs =
        deepseek_b1_ops::RMSNorm::ComputeCTArgs<get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1>;
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),
        get_named_compile_time_arg_val("rmsnorm_interm_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_epsilon_index"),
        get_named_compile_time_arg_val("rmsnorm_scalar_index"),
    };

    // Mcast compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    // RMSNorm2 compute args (separate CBs with exact sizes for testing)
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm2_args{
        get_named_compile_time_arg_val("rmsnorm2_input_cb"),   // separate input CB (2 tiles)
        get_named_compile_time_arg_val("rmsnorm_scalars_cb"),  // reuse scalars cb
        get_named_compile_time_arg_val("rmsnorm2_interm_cb"),  // separate interm CB (3 tiles)
        get_named_compile_time_arg_val("rmsnorm2_gamma_cb"),   // new gamma for 1536 elements
        get_named_compile_time_arg_val("rmsnorm2_output_cb"),  // separate output CB (2 tiles)
        get_named_compile_time_arg_val("rmsnorm2_num_tiles"),  // 2 tiles
        get_named_compile_time_arg_val("rmsnorm_epsilon_index"),
        get_named_compile_time_arg_val("rmsnorm_scalar_index"),
    };
#endif

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_input_core) {
        // RMSNorm input and gamma buffers
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        deepseek_b1_ops::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        deepseek_b1_ops::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);

        // RMSNorm2 gamma buffer (2 tiles, padded from 1536 elements)
        constexpr uint32_t rmsnorm2_gamma_cb = get_named_compile_time_arg_val("rmsnorm2_gamma_cb");
        constexpr uint32_t rmsnorm2_num_tiles = get_named_compile_time_arg_val("rmsnorm2_num_tiles");
        deepseek_b1_ops::setup_sharded_buffer(rmsnorm2_gamma_cb, rmsnorm2_num_tiles);
    }
    if constexpr (Core::is_matmul_core) {
        // Matmul weights
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_num_tiles = get_named_compile_time_arg_val("matmul_num_tiles");
        deepseek_b1_ops::setup_sharded_buffer(matmul_in1, matmul_num_tiles);
    }
#endif

    // Set up reusable operations
    // pop_input = true (input is consumed after RMSNorm)
    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core, true> rmsnorm;

    // ========================================================================
    // Input core: RMSNorm + Mcast send
    // ========================================================================
    {
        DeviceZoneScopedN("RMSNORM");
        rmsnorm(rmsnorm_args);
    }

    {
        DeviceZoneScopedN("MCAST");
        // Mcast: NCRISC sends from input core, BRISC receives on matmul cores, TRISC no-op
        // pop_src = true (input is consumed after mcast)
        deepseek_b1_ops::Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core, true> mcast;
        mcast(mcast_args);
    }

    // ========================================================================
    // Matmul operation
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
        deepseek_b1_ops::Matmul::Op<Core::is_matmul_core, true, false> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Gather: matmul cores (senders) -> input core (receiver)
    // NCRISC sends from matmul cores, BRISC receives on input core, TRISC no-op
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        // pop_src = true (matmul output is consumed after gather)
        deepseek_b1_ops::Gather::Op<Core::is_matmul_core, Core::is_input_core, true> gather;
        gather(gather_args);
    }

    // ========================================================================
    // Copy gather output to RMSNorm input CB with zero padding
    // Gather dst cb has 1536 bytes (1.5 half tiles), pad to 2048 bytes (32x32 tile)
    // Only runs on BRISC on input core (where gather receiver runs)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_input_core) {
        DeviceZoneScopedN("GATHER_TO_RMSNORM_HACK");

        constexpr uint32_t rmsnorm2_input_cb = get_named_compile_time_arg_val("rmsnorm2_input_cb");
        constexpr uint32_t gather_data_size_bytes = 1536 * 2;  // 1.5 half tiles
        constexpr uint32_t rmsnorm_tile_size_bytes = 4096;     // 2 tiles (32x32 each) in bfloat16
        constexpr uint32_t padding_size_bytes = rmsnorm_tile_size_bytes - gather_data_size_bytes;

        // Wait for gather dst cb data (already pushed by gather receiver)
        cb_wait_front(gather_dst_cb, gather_dst_num_pages);

        // Reserve space in rmsnorm2 input cb (2 tiles)
        cb_reserve_back(rmsnorm2_input_cb, 2);

        // Get source and destination addresses
        uint32_t src_addr = get_read_ptr(gather_dst_cb);
        uint32_t dst_addr = get_write_ptr(rmsnorm2_input_cb);

        // Copy gather data to rmsnorm2 cb using local NOC read
        uint64_t src_noc_addr = get_noc_addr(src_addr);
        noc_async_read(src_noc_addr, dst_addr, gather_data_size_bytes);
        noc_async_read_barrier();

        // Zero-pad the remaining bytes (last half tile)
        volatile tt_l1_ptr uint16_t* pad_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_addr + gather_data_size_bytes);
        constexpr uint32_t padding_elements = padding_size_bytes / sizeof(uint16_t);
        for (uint32_t i = 0; i < padding_elements; ++i) {
            pad_ptr[i] = 0;
        }

        // Push the completed 2 tiles to rmsnorm2 input cb
        cb_push_back(rmsnorm2_input_cb, 2);

        // Pop the gather dst cb
        cb_pop_front(gather_dst_cb, gather_dst_num_pages);
    }
#endif

    // ========================================================================
    // RMSNorm2: Apply RMSNorm to the gathered/padded data (1536 elements -> 2 tiles)
    // Uses SEPARATE CBs with exact sizes:
    //   - Input: rmsnorm2_input_cb (2 tiles from copy)
    //   - Interm: rmsnorm2_interm_cb (3 tiles)
    //   - Output: rmsnorm2_output_cb (2 tiles)
    //   - Gamma: rmsnorm2_gamma_cb (2 tiles, padded from 1536 elements)
    //   - Scalars: reuses scalars_cb (same epsilon, different scalar)
    // ========================================================================
    {
        DeviceZoneScopedN("RMSNORM2");
        rmsnorm(rmsnorm2_args);
    }
}
KERNEL_END
