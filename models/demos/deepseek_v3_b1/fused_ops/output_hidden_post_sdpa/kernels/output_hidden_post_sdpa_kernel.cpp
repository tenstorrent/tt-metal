// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Output Hidden Post SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: Matmul + Gather + Mcast
// - NCRISC: Matmul reader (setup sharded buffers), Gather sender (on matmul cores), Mcast receiver (on mcast grid)
// - BRISC: Matmul writer (no-op), Gather receiver (on gather core), Mcast sender (on gather core)
// - TRISC: Matmul compute (on matmul cores), Gather/Mcast compute (no-op)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/mcast.hpp"

// Compile-time role flags for dead code elimination via if constexpr
// Defined at namespace scope (local classes cannot have static data members)
struct Core {
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
// Named compile-time args: matmul reader, gather sender, mcast receiver
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Matmul CTArgs type alias (NCRISC uses ReaderCTArgs)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Matmul reader args (NCRISC is no-op for compute, just setup)
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
        get_named_compile_time_arg_val("gather_receiver_data_addr"),  // receiver's output tensor address
    };

    // Mcast CTArgs (NCRISC uses ReceiverCTArgs)
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;

    // Mcast receiver args (from compile-time args)
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: matmul writer (no-op), gather receiver, mcast sender
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Matmul CTArgs type alias (BRISC uses WriterCTArgs)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

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

    // Mcast CTArgs (BRISC uses SenderCTArgs)
    // Gather core (11,9) is NOT in mcast grid (0,0)-(7,11), so no loopback needed
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;  // loopback = false (gather core is not in mcast grid)

    // Mcast source CB = gather destination CB (mcast reads what gather wrote)
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");

    // Mcast receiver data address (runtime arg - output tensor buffer address)
    uint32_t mcast_receiver_data_addr = get_arg_val<uint32_t>(0);

    // Mcast sender args (from compile-time args + runtime arg)
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
        get_read_ptr(mcast_src_cb),  // Read from gather output CB
        mcast_receiver_data_addr,    // Write to output tensor on mcast grid
    };

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: matmul compute
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Matmul CTArgs type alias (out_w is compile-time for TRISC)
    using MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    // Mcast CTArgs (TRISC uses ComputeCTArgs - no-op)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;

    // Mcast compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};
#endif
    DPRINT << "Output Hidden Post SDPA kernel started" << ENDL();

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_matmul_core) {
        // Matmul input (already sharded on each core)
        constexpr uint32_t matmul_in0 = get_named_compile_time_arg_val("matmul_in0");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_num_tiles");
        unified_kernels::setup_sharded_buffer(matmul_in0, matmul_k_num_tiles);

        // Matmul weights (width sharded, persistent)
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_out_w_per_core = get_named_compile_time_arg_val("matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w_per_core);
    }
#endif

    // ========================================================================
    // Matmul operation: [1, 512] x [512, 128] -> [1, 128] per core
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        // pop_in0 = true (input consumed), pop_in1 = false (weights are persistent)
        deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Gather: matmul cores (senders) -> gather core (receiver)
    // NCRISC sends from matmul cores, BRISC receives on gather core, TRISC no-op
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        // pop_src = true (matmul output is consumed after gather)
        deepseek_b1_ops::Gather::Op<Core::is_matmul_core, Core::is_gather_receiver_core, true> gather;
        gather(gather_args);
    }

    // ========================================================================
    // Mcast: gather core (sender) -> mcast grid cores (receivers)
    // BRISC sends from gather core, NCRISC receives on mcast grid, TRISC no-op
    // Source CB = gather_dst_cb (CB 3), Destination CB = mcast_dst_cb (CB 4)
    // ========================================================================
    // pop_src = true (gather output is consumed after mcast)
    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_gather_receiver_core, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, true>
            mcast;
#if defined(COMPILE_FOR_BRISC)
    // Initialize mcast sender (only on gather core)
    if constexpr (Core::is_gather_receiver_core) {
        mcast.init(mcast_args);
    }
#endif
    {
        DeviceZoneScopedN("MCAST");
        mcast(mcast_args);
    }
#if defined(COMPILE_FOR_BRISC)
    // Teardown mcast sender (only on gather core)
    if constexpr (Core::is_gather_receiver_core) {
        mcast.teardown();
    }
#endif

    DPRINT << "Output Hidden Post SDPA kernel finished" << ENDL();
}
