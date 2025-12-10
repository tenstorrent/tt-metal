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
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/rmsnorm.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/mcast.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/matmul.hpp"
#include "models/demos/deepseek_v3_b1/micro_ops/unified_kernels/gather.hpp"
#include "models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/unified_core_descriptor.hpp"
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#else
#include "tools/profiler/kernel_profiler.hpp"
#endif

KERNEL_ENTRY {
    // Use UnifiedCoreDescriptor for compile-time role checks
    using Core = pre_sdpa::UnifiedCoreDescriptor;

// ============================================================================
// NCRISC (Reader + Mcast Sender) - ReaderConfigDescriptor compiles as NCRISC
// Compile-time args: [input_cb, scalars_cb, gamma_cb, num_tiles, tiny_tile]
// Named compile-time args: mcast sender parameters
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs<
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

    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs>::ReaderArgs rmsnorm_rt_args;
    rmsnorm_rt_args.epsilon = get_arg_val<uint32_t>(0);
    rmsnorm_rt_args.scalar = get_arg_val<uint32_t>(1);

    // Mcast CB indices and page counts from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
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
        get_named_compile_time_arg_val("gather_receiver_semaphore_id")>;

    // Gather grid info for computing per-core offset
    constexpr uint32_t gather_sender_grid_start_x = get_named_compile_time_arg_val("gather_sender_grid_start_x");
    constexpr uint32_t gather_sender_grid_start_y = get_named_compile_time_arg_val("gather_sender_grid_start_y");
    constexpr uint32_t gather_sender_grid_size_x = get_named_compile_time_arg_val("gather_sender_grid_size_x");

// ============================================================================
// BRISC (Writer + Mcast Receiver) - WriterConfigDescriptor compiles as BRISC
// Compile-time args: [output_cb, num_tiles]
// Named compile-time args: mcast receiver parameters
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    using Mcast = deepseek_b1_ops::Mcast;
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs<
        get_compile_time_arg_val(0),  // output_cb
        get_compile_time_arg_val(1)   // num_tiles
        >;

    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs>::WriterArgs rmsnorm_rt_args;

    // Mcast receiver CTArgs from named compile-time args
    using McastCTArgs = Mcast::ReceiverCTArgs<get_named_compile_time_arg_val("mcast_data_receiver_semaphore")>;
    Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core>::ReceiverArgs mcast_rt_args;

    // Mcast CB info for receiver
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    constexpr uint32_t mcast_dst_num_pages = get_named_compile_time_arg_val("mcast_dst_num_pages");

    // Matmul writer CTArgs from named compile-time args
    using Matmul = deepseek_b1_ops::Matmul;
    using MatmulCTArgs = Matmul::WriterCTArgs<get_named_compile_time_arg_val("matmul_out_cb")>;

    // Gather receiver CTArgs from named compile-time args
    using Gather = deepseek_b1_ops::Gather;
    using GatherCTArgs = Gather::ReceiverCTArgs<
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id")>;

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Compile-time args: [input_cb, scalars_cb, interm_cb, gamma_cb, output_cb,
//                     fp32_acc, num_tiles, epsilon_index, scalar_index]
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
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

    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs>::ComputeArgs rmsnorm_rt_args;

    // Matmul compute CTArgs from named compile-time args
    using Matmul = deepseek_b1_ops::Matmul;
    using MatmulCTArgs = Matmul::ComputeCTArgs<
        get_named_compile_time_arg_val("matmul_in0_cb"),
        get_named_compile_time_arg_val("matmul_in1_cb"),
        get_named_compile_time_arg_val("matmul_out_cb"),
        get_named_compile_time_arg_val("matmul_interm_cb"),
        get_named_compile_time_arg_val("matmul_num_tiles_k"),
        get_named_compile_time_arg_val("matmul_fp32_acc")>;
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
        // Mcast: NCRISC sends from input core, BRISC receives on matmul cores
#if defined(COMPILE_FOR_NCRISC)
        // Sender: wait for source CB data, send, then pop
        if constexpr (Core::is_input_core) {
            // Wait for RMSNorm output to be ready in source CB
            cb_wait_front(mcast_src_cb, mcast_src_num_pages);

            // Send mcast
            Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core> mcast;
            mcast(mcast_rt_args);

            // Pop the source CB after sending
            cb_pop_front(mcast_src_cb, mcast_src_num_pages);
        }
#elif defined(COMPILE_FOR_BRISC)
        // Receiver: reserve space, wait for mcast, then push
        if constexpr (Core::is_matmul_core) {
            // Reserve space in destination CB before mcast writes to it
            cb_reserve_back(mcast_dst_cb, mcast_dst_num_pages);

            // Wait for mcast data (semaphore signals arrival)
            Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul_core> mcast;
            mcast(mcast_rt_args);

            // Push to destination CB after data arrived
            cb_push_back(mcast_dst_cb, mcast_dst_num_pages);
        }
#endif
    }

    // Debug: Print received mcast data on receiver cores
    // #if defined(COMPILE_FOR_BRISC)
    //     if constexpr (Core::is_matmul_core) {
    //         DPRINT << TileSlice(mcast_dst_cb, 0, SliceRange::hw0_32_8(), true, true) << ENDL();
    //     }
    // #endif

    // ========================================================================
    // Input core: Copy mcast output to output CB
    // ========================================================================
    // #if defined(COMPILE_FOR_BRISC)
    //     if constexpr (Core::is_input_core) {
    //         // Get output CB from named compile-time args
    //         constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");

    //         // Reserve space in output CB
    //         cb_reserve_back(output_cb, mcast_dst_num_pages);
    //         DPRINT << "reserve output cb" << ENDL();

    //         // Copy data from mcast_dst_cb to output_cb
    //         uint32_t src_addr = get_read_ptr(mcast_dst_cb);
    //         uint32_t dst_addr = get_write_ptr(output_cb);
    //         uint32_t size_bytes = mcast_dst_num_pages * get_tile_size(mcast_dst_cb);

    //         noc_async_read(get_noc_addr(src_addr), dst_addr, size_bytes);
    //         noc_async_read_barrier();

    //         // Push output data and pop mcast data
    //         cb_push_back(output_cb, mcast_dst_num_pages);
    //     }
    // #endif

    // ========================================================================
    // iles from RMSNorm writer args Matmul operation
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        if constexpr (Core::is_matmul_core) {
#if defined(COMPILE_FOR_NCRISC)
            // Push weights (in1_cb) - backed by sharded tensor
            constexpr uint32_t matmul_in1_cb = get_named_compile_time_arg_val("matmul_in1_cb");
            constexpr uint32_t matmul_num_tiles_k = get_named_compile_time_arg_val("matmul_num_tiles_k");
            cb_reserve_back(matmul_in1_cb, matmul_num_tiles_k);
            cb_push_back(matmul_in1_cb, matmul_num_tiles_k);
#endif
            Matmul::Op<MatmulCTArgs> matmul;
            matmul();
        }
    }

    // #if defined(COMPILE_FOR_BRISC)
    //     if constexpr (Core::is_input_core) {
    //         // Get output CB from named compile-time args
    //         constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");
    //
    //         // Reserve space in output CB
    //         DPRINT << "reserve output cb" << ENDL();
    //
    //         // Copy data from mcast_dst_cb to output_cb
    //         uint32_t src_addr = get_read_ptr(MatmulCTArgs::out_cb);
    //         uint32_t dst_addr = get_write_ptr(output_cb);
    //         uint32_t size_bytes = mcast_dst_num_pages * get_tile_size(MatmulCTArgs::out_cb);
    //
    //         noc_async_read(get_noc_addr(src_addr), dst_addr, size_bytes);
    //         noc_async_read_barrier();
    //     }
    // #endif

    // Debug: Print received matmul data on receiver cores
    // #if defined(COMPILE_FOR_BRISC)
    //     if constexpr (Core::is_matmul_core) {
    //         DPRINT << TileSlice(MatmulCTArgs::out_cb, 0, SliceRange::hw0_32_8(), true, true) << ENDL();
    //     }
    // #endif

    // ========================================================================
    // Gather: matmul cores (senders) -> input core (receiver)
    // NCRISC sends from matmul cores, BRISC receives on input core
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
#if defined(COMPILE_FOR_NCRISC)
        // Sender: compute offset, wait for matmul output, send data
        if constexpr (Core::is_matmul_core) {
            // Get source CB (matmul output)
            constexpr uint32_t gather_src_cb = get_named_compile_time_arg_val("gather_src_cb");

            // Wait for matmul output to be ready (matmul writer pushes to out_cb)
            cb_wait_front(gather_src_cb, 1);

            // Get source address (matmul output on this core)
            uint32_t input_data_addr = get_read_ptr(gather_src_cb);

            // Get destination address (output tensor on receiver core, passed as runtime arg)
            uint32_t receiver_data_addr = get_arg_val<uint32_t>(2);

            // Compute per-core offset based on core coordinates
            // Note: This assumes contiguous coordinates within the grid
            uint32_t core_x = my_x[0] - gather_sender_grid_start_x;
            uint32_t core_y = my_y[0] - gather_sender_grid_start_y;
            uint32_t core_index = core_y * gather_sender_grid_size_x + core_x;
            uint32_t offset = core_index * GatherCTArgs::data_size_bytes;

            // Send data to receiver
            Gather::Op<GatherCTArgs> gather;
            Gather::Op<GatherCTArgs>::SenderArgs gather_rt_args;
            gather_rt_args.input_data_addr = input_data_addr;
            gather_rt_args.receiver_data_addr = receiver_data_addr;
            gather_rt_args.offset = offset;
            gather(gather_rt_args);

            // Pop the source CB after sending
            cb_pop_front(gather_src_cb, 1);
        }
#elif defined(COMPILE_FOR_BRISC)
        // Receiver: wait for all senders to complete
        if constexpr (Core::is_input_core) {
            Gather::Op<GatherCTArgs> gather;
            gather();
        }
#endif
    }

#if defined(COMPILE_FOR_NCRISC)
    DPRINT << "NCRISC gather completed" << ENDL();
#endif

    DPRINT << "Pre-SDPA kernel completed" << ENDL();
}
KERNEL_END
