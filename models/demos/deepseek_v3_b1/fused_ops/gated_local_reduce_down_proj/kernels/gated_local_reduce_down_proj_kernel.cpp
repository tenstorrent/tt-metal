// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Gated Local Reduce + Down Projection unified kernel
//
// Fuses: Input Gather + GatedLocalReduce + Mcast1 + Mcast2 + Matmul + ResidualAdd + Output Gather
//
// Sender core (12,9):
//   BRISC:  input_gather_g1 receive, input_gather_g2 receive,
//           mcast1(mcast_src) → 130-core grid,
//           mcast2(residual) → 130-core grid,
//           output gather_receive
//   TRISC:  gated reduce → mcast_src
//   NCRISC: idle (for gated reduce); setup_sharded_buffer for mcast1/mcast2 sources
//
// Matmul cores (112):
//   NCRISC: setup_sharded_buffer(input_src, weights), input_gather send,
//           mcast1 receive, mcast2 receive, output gather send
//   TRISC:  matmul + residual add
//
// CB Layout:
//   CB 0:  group1 gather dest / reduce input (sender core, tensor-backed)
//   CB 1:  group2 gather dest / reduce input (sender core, tensor-backed)
//   CB 2:  intermediate (sender core, 2 tiles, manual)
//   CB 3:  mcast1 source (sender core, manual, TRISC-filled)
//   CB 4:  mcast1 destination / matmul in0 (all 130 cores, manual)
//   CB 5:  matmul weights (112 matmul cores, tensor-backed)
//   CB 6:  matmul output (112 matmul cores, manual)
//   CB 7:  output gather destination (sender core, tensor-backed)
//   CB 8:  input source group1 (g1 source cores, tensor-backed)
//   CB 9:  input source group2 (g2 source cores, tensor-backed)
//   CB 10: mcast2 source / residual input (sender core, tensor-backed)
//   CB 11: mcast2 destination (all 130 cores, manual)
//   CB 12: residual add output (112 matmul cores, manual)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/gated_reduce.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/residual_add.hpp"

struct Core {
    static constexpr bool is_input_gather_sender_g1 = get_named_compile_time_arg_val("is_input_gather_sender_g1") == 1;
    static constexpr bool is_input_gather_sender_g2 = get_named_compile_time_arg_val("is_input_gather_sender_g2") == 1;
    static constexpr bool is_gated_reduce_core = get_named_compile_time_arg_val("is_gated_reduce_core") == 1;
    static constexpr bool is_mcast_sender_core = get_named_compile_time_arg_val("is_mcast_sender_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    static constexpr bool gather_use_per_core_sender_idx =
        get_named_compile_time_arg_val("gather_use_per_core_sender_idx") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Input gather g1 sender args
    deepseek_b1_ops::Gather::SenderArgs ig_g1_args{
        get_named_compile_time_arg_val("ig_g1_dest_noc_x"),
        get_named_compile_time_arg_val("ig_g1_dest_noc_y"),
        get_named_compile_time_arg_val("ig_g1_data_size_bytes"),
        get_named_compile_time_arg_val("ig_g1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ig_g1_src_cb"),
        get_named_compile_time_arg_val("ig_g1_src_num_pages"),
        0,
        0,
        0,
        0,
        1,
        get_named_compile_time_arg_val("ig_g1_receiver_data_addr"),
        get_named_compile_time_arg_val("ig_g1_sender_idx"),
    };

    // Input gather g2 sender args
    deepseek_b1_ops::Gather::SenderArgs ig_g2_args{
        get_named_compile_time_arg_val("ig_g2_dest_noc_x"),
        get_named_compile_time_arg_val("ig_g2_dest_noc_y"),
        get_named_compile_time_arg_val("ig_g2_data_size_bytes"),
        get_named_compile_time_arg_val("ig_g2_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ig_g2_src_cb"),
        get_named_compile_time_arg_val("ig_g2_src_num_pages"),
        0,
        0,
        0,
        0,
        1,
        get_named_compile_time_arg_val("ig_g2_receiver_data_addr"),
        get_named_compile_time_arg_val("ig_g2_sender_idx"),
    };

    // Gated reduce reader args (NCRISC no-op)
    using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ReaderCTArgs;
    deepseek_b1_ops::GatedReduce::ReaderArgs gated_reduce_args{};

    // Mcast1 receiver args
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Mcast2 receiver args
    using Mcast2CTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast2_args{
        get_named_compile_time_arg_val("mcast2_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast2_dst_cb"),
        get_named_compile_time_arg_val("mcast2_dst_num_pages"),
    };

    // Matmul CTArgs (NCRISC is no-op for matmul compute)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // Output gather sender args
    deepseek_b1_ops::Gather::SenderArgs og_args{
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
        get_named_compile_time_arg_val("gather_sender_idx"),
    };

// ============================================================================
// BRISC (Writer)
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Input gather g1 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs ig_g1_args{
        get_named_compile_time_arg_val("ig_g1_noc0_num_senders"),
        0,
        get_named_compile_time_arg_val("ig_g1_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ig_g1_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ig_g1_dst_cb"),
        get_named_compile_time_arg_val("ig_g1_dst_num_pages"),
    };

    // Input gather g2 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs ig_g2_args{
        get_named_compile_time_arg_val("ig_g2_noc0_num_senders"),
        0,
        get_named_compile_time_arg_val("ig_g2_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ig_g2_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ig_g2_dst_cb"),
        get_named_compile_time_arg_val("ig_g2_dst_num_pages"),
    };

    // Gated reduce writer args (BRISC no-op)
    using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::WriterCTArgs;
    deepseek_b1_ops::GatedReduce::WriterArgs gated_reduce_args{};

    // Mcast1 sender args
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        /*Loopback=*/false>;

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

    // Mcast2 sender args (same grid, different semaphores and CBs)
    using Mcast2CTArgs = McastCTArgs;

    constexpr uint32_t mcast2_src_cb = get_named_compile_time_arg_val("mcast2_src_cb");
    constexpr uint32_t mcast2_dst_cb = get_named_compile_time_arg_val("mcast2_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast2_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast2_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast2_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast2_data_size_bytes"),
        mcast2_src_cb,
        get_named_compile_time_arg_val("mcast2_src_num_pages"),
        get_read_ptr(mcast2_src_cb),
        get_write_ptr(mcast2_dst_cb),
    };

    // Matmul CTArgs (BRISC is no-op for matmul)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    // Output gather receiver args
    deepseek_b1_ops::Gather::ReceiverArgs og_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

// ============================================================================
// TRISC (Compute)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Input gather compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs ig_g1_args{};
    deepseek_b1_ops::Gather::ComputeArgs ig_g2_args{};

    // Mcast1 CTArgs (no-op for TRISC)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Mcast2 CTArgs (no-op for TRISC)
    using Mcast2CTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast2_args{};

    // Gated reduce compute args
    using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ComputeCTArgs<
        get_named_compile_time_arg_val("gated_reduce_tiles_per_k"),
        get_named_compile_time_arg_val("gated_reduce_k_num_tiles")>;
    deepseek_b1_ops::GatedReduce::ComputeArgs gated_reduce_args{
        get_named_compile_time_arg_val("gated_reduce_group1_cb"),
        get_named_compile_time_arg_val("gated_reduce_group2_cb"),
        get_named_compile_time_arg_val("gated_reduce_intermed_cb"),
        get_named_compile_time_arg_val("gated_reduce_mcast_src_cb"),
    };

    // Matmul CTArgs
    using MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // Output gather compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs og_args{};
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Setup sharded buffers (NCRISC only)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Input source buffers on source cores
    if constexpr (Core::is_input_gather_sender_g1) {
        constexpr uint32_t ig_g1_src_cb = get_named_compile_time_arg_val("ig_g1_src_cb");
        unified_kernels::setup_sharded_buffer(ig_g1_src_cb, 1);
    }
    if constexpr (Core::is_input_gather_sender_g2) {
        constexpr uint32_t ig_g2_src_cb = get_named_compile_time_arg_val("ig_g2_src_cb");
        unified_kernels::setup_sharded_buffer(ig_g2_src_cb, 1);
    }
    // Mcast2 source (residual input on sender core)
    if constexpr (Core::is_mcast_sender_core) {
        constexpr uint32_t m2_src_cb = get_named_compile_time_arg_val("mcast2_src_cb");
        constexpr uint32_t m2_src_pages = get_named_compile_time_arg_val("mcast2_src_num_pages");
        unified_kernels::setup_sharded_buffer(m2_src_cb, m2_src_pages);
    }
    // Matmul weight buffers (112 matmul cores)
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t matmul_out_w_per_core = get_named_compile_time_arg_val("matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w_per_core);
    }
#endif

    // ========================================================================
    // Input Gather Group1: source cores -> (12,9) group1_cb
    // ========================================================================
    {
        DeviceZoneScopedN("INPUT_GATHER_G1");
        deepseek_b1_ops::Gather::Op<
            Core::is_input_gather_sender_g1,
            Core::is_gated_reduce_core,
            /*pop_src=*/true,
            Core::is_input_gather_sender_g1>
            input_gather_g1;
        input_gather_g1(ig_g1_args);
    }

    // ========================================================================
    // Input Gather Group2: source cores -> (12,9) group2_cb
    // ========================================================================
    {
        DeviceZoneScopedN("INPUT_GATHER_G2");
        deepseek_b1_ops::Gather::Op<
            Core::is_input_gather_sender_g2,
            Core::is_gated_reduce_core,
            /*pop_src=*/true,
            Core::is_input_gather_sender_g2>
            input_gather_g2;
        input_gather_g2(ig_g2_args);
    }

    // ========================================================================
    // Gated Local Reduce (TRISC on sender core)
    // ========================================================================
    {
        DeviceZoneScopedN("GATED_REDUCE");
        deepseek_b1_ops::GatedReduce::Op<GatedReduceCTArgs, Core::is_gated_reduce_core> gated_reduce;
        gated_reduce(gated_reduce_args);
    }

    // ========================================================================
    // Mcast1: sender (12,9) -> 129 receivers in 13x10 grid
    // Broadcasts hidden state [1, K] to all cores
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core,
        Core::is_mcast_receiver_core,
        /*pop_src=*/true>
        mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST1");
        mcast(mcast_args);
    }

    // ========================================================================
    // Mcast2: sender (12,9) -> 129 receivers in 13x10 grid
    // Broadcasts residual input [1, N] to all cores
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<
        Mcast2CTArgs,
        Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core,
        Core::is_mcast_receiver_core,
        /*pop_src=*/true>
        mcast2;
    {
        DeviceZoneScopedN("MCAST2");
        mcast2(mcast2_args);
    }
    mcast.teardown();

    // ========================================================================
    // Matmul: [1, K] x [K, N_per_core] -> [1, N_per_core] on 112 cores
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, /*pop_in0=*/true, /*pop_in1=*/false> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Residual add: matmul_out + shard(residual) -> residual_add_out on 112 matmul cores
    // Each core indexes into the full [1, N] CB11 at offset core_idx * out_w_per_core
    // ========================================================================
    using ResidualAddCTArgs =
        deepseek_b1_ops::ResidualAdd::ComputeCTArgs<get_named_compile_time_arg_val("residual_add_out_w")>;
    deepseek_b1_ops::ResidualAdd::RTArgs residual_add_args{
#if defined(COMPILE_FOR_TRISC)
        get_named_compile_time_arg_val("residual_add_in0"),
        get_named_compile_time_arg_val("residual_add_in1"),
        get_named_compile_time_arg_val("residual_add_out"),
        get_named_compile_time_arg_val("residual_add_total_in1_tiles"),
        get_named_compile_time_arg_val("gather_sender_idx"),
#endif
    };
    {
        DeviceZoneScopedN("ADD");
        deepseek_b1_ops::ResidualAdd::Op<ResidualAddCTArgs, Core::is_matmul_core> residual_add;
        residual_add(residual_add_args);
    }

    // ========================================================================
    // Output Gather: 112 matmul cores -> (12,9)
    // Reads from residual_add_out_cb (CB 12)
    // ========================================================================
    {
        DeviceZoneScopedN("OUTPUT_GATHER");
        deepseek_b1_ops::Gather::Op<
            Core::is_matmul_core,
            Core::is_gather_receiver_core,
            /*pop_src=*/true,
            Core::gather_use_per_core_sender_idx>
            output_gather;
        output_gather(og_args);
    }
}
