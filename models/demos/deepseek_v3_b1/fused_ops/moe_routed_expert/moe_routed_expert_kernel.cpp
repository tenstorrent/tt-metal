// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// MoE Routed Expert fused kernel
// Single kernel file, compiles correctly for all RISC cores
//
// Implements: Mcast -> Matmul+Activation (extendable with more fusions)
// - Sender core: Mcast sender only (outside compute grid)
// - Compute cores: Mcast receiver + Matmul+Activation compute
// - Mcast grid: Rectangle encompassing sender and compute cores
//
// RISC assignments:
// - BRISC: Mcast sender (sender core only)
// - NCRISC: Mcast receiver + setup sharded buffers
// - TRISC: Matmul+Activation compute (compute cores only)

#include "../../unified_kernels/kernel_op_api.hpp"
#include "../../unified_kernels/kernel_utils.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/matmul.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define CTArgs and args per RISC (different compile-time arg layout per processor)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Mcast CTArgs and args
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul CTArgs and args
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // Setup sharded persistent buffers
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast_src_cb, mcast_src_num_pages);
    }
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t matmul_out_w = get_named_compile_time_arg_val("matmul_out_w");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w);
    }

#elif defined(COMPILE_FOR_BRISC)
    // Mcast CTArgs and args (loopback if sender is in the mcast grid)
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_sender_core && Core::is_mcast_grid_core>;

    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
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

    // Matmul CTArgs and args
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

#elif defined(COMPILE_FOR_TRISC)
    // Mcast CTArgs and args (no-op for TRISC)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul CTArgs and args
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<
        get_named_compile_time_arg_val("matmul_out_w"),
        get_named_compile_time_arg_val("matmul_fused_activation")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };
#endif

    // ========================================================================
    // Mcast: Broadcast input from sender core to all matmul cores
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<McastCTArgs, Core::is_sender_core, Core::is_mcast_grid_core, Core::is_matmul_core, true>
        mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        mcast(mcast_args);
    }
    mcast.teardown();

    // ========================================================================
    // Matmul + Activation: Compute on all matmul cores
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
        deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
        matmul(matmul_args);
    }
}
