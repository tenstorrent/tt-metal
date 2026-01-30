// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Output Hidden unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: Matmul + Gather
// - NCRISC: Matmul reader (setup sharded buffers), Gather sender (on matmul cores)
// - BRISC: Matmul writer (no-op), Gather receiver (on gather core)
// - TRISC: Matmul compute (on matmul cores), Gather compute (no-op)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"

// Compile-time role flags for dead code elimination via if constexpr
// Defined at namespace scope (local classes cannot have static data members)
struct Core {
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
// Named compile-time args: matmul reader, gather sender
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
        get_write_ptr(get_named_compile_time_arg_val("gather_dst_cb")),  // receiver_data_addr from CB write ptr
    };

// ============================================================================
// BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: matmul writer (no-op), gather receiver
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
#endif

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
    // Matmul operation: [1, 8192] x [8192, 64] -> [1, 64] per core
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
}
