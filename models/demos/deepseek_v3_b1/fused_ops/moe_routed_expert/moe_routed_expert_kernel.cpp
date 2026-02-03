// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// MoE Routed Expert fused kernel
// Single kernel file, compiles correctly for all RISC cores
//
// Implements: Mcast -> Matmul+Activation -> Gather -> Gate
// - Sender core: Mcast sender, Gather receiver, Gate compute (outside compute grid)
// - Compute cores: Mcast receiver, Matmul+Activation compute, Gather sender
// - Mcast grid: Rectangle encompassing sender and compute cores
//
// RISC assignments:
// - BRISC: Mcast sender (sender core), Gather receiver (sender core), Gate writer (sender core)
// - NCRISC: Mcast receiver, setup sharded buffers, Gather sender (compute cores), Gate reader (sender core)
// - TRISC: Matmul+Activation compute (compute cores), Gate compute (sender core)

#include "../../unified_kernels/kernel_op_api.hpp"
#include "../../unified_kernels/kernel_utils.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/matmul.hpp"
#include "../../unified_kernels/gather.hpp"
#include "../../unified_kernels/deepseek_moe_gate.hpp"

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

    // Gather sender args (compute cores send to sender core)
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
        get_named_compile_time_arg_val("gather_receiver_data_addr"),
    };

    // Gate CTArgs (NCRISC: reader on sender core) - empty, setup done below
    using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs;

    // Setup sharded persistent buffers
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast_src_cb, mcast_src_num_pages);

        // Gate tensor-backed CBs (bias and indices)
        // Note: gate_input_cb is NOT setup here - gather already pushes to it
        constexpr uint32_t gate_bias_cb = get_named_compile_time_arg_val("gate_bias_cb");
        constexpr uint32_t gate_input_indices_cb = get_named_compile_time_arg_val("gate_input_indices_cb");
        unified_kernels::setup_sharded_buffer(gate_bias_cb, 1);
        unified_kernels::setup_sharded_buffer(gate_input_indices_cb, 1);
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

    // Gather receiver args (sender core receives from compute cores)
    deepseek_b1_ops::Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

    // Gate CTArgs (BRISC: writer on sender core)
    using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::WriterCTArgs<
        get_named_compile_time_arg_val("gate_output_cb"),
        get_named_compile_time_arg_val("gate_output_indices_cb")>;

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

    // Gather args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    // Gate CTArgs (TRISC: compute on sender core)
    using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ComputeCTArgs<
        get_named_compile_time_arg_val("gate_input_cb"),
        get_named_compile_time_arg_val("gate_bias_cb"),
        get_named_compile_time_arg_val("gate_input_indices_cb"),
        get_named_compile_time_arg_val("gate_output_cb"),
        get_named_compile_time_arg_val("gate_output_indices_cb"),
        get_named_compile_time_arg_val("gate_eps"),
        get_named_compile_time_arg_val("gate_scaling_factor"),
        get_named_compile_time_arg_val("gate_enable_sigmoid")>;
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

    // ========================================================================
    // Gather: Collect matmul outputs from compute cores to sender core
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        // pop_src = true (matmul output consumed after gather)
        deepseek_b1_ops::Gather::Op<Core::is_matmul_core, Core::is_sender_core, true> gather;
        gather(gather_args);
    }

    // ========================================================================
    // Gate: Top-8 expert selection with normalized scores (on sender core only)
    // ========================================================================
    {
        DeviceZoneScopedN("GATE");
        deepseek_b1_ops::DeepseekMoeGate::Op<GateCTArgs, Core::is_sender_core> gate;
        gate();
    }

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
