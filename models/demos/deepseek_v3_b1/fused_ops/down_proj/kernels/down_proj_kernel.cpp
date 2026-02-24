// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Down Projection (W_down) unified kernel
// Single kernel file, compiles correctly for all RISC cores
//
// Implements: Mcast1 + Mcast2 + Matmul + ResidualAdd + Gather
// - Mcast1: Input [1, K] on (12,9) -> broadcast to 130-core grid (13x10)
// - Mcast2: Residual input [1, N] on (12,9) -> broadcast to 130-core grid (13x10)
// - Matmul: [1, K] x [K, N_per_core] -> [1, N_per_core] on 112 cores
// - Residual add: matmul_out + shard(residual) -> [1, N_per_core] on 112 cores
// - Gather: Collect [1, N_per_core] from 112 cores to [1, N] on (12,9)
//
// Core roles in 13x10 = 130 core grid:
//   M = Mcast sender + Gather receiver (12,9) — dual role
//   R = Matmul core (112 cores)
//   D = DRAM worker — receives mcast semaphore, skips matmul & gather (8 cores)
//   P = Phantom — receives mcast semaphore, skips matmul & gather (9 cores, col 12 rows 0-8)
//
// Note: Mcast sender (12,9) IS in the mcast grid (is_part_of_receiver_grid=true)
// but is NOT an mcast receiver (is_mcast_receiver_core=false) to avoid deadlock.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/residual_add.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    // Mcast sender (12,9) only
    static constexpr bool is_mcast_sender_core = get_named_compile_time_arg_val("is_mcast_sender_core") == 1;
    // Mcast receiver: 129 cores (full 130-core grid MINUS sender at 12,9)
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    // Active matmul cores (112 cores: excludes 8 DRAM workers + 9 phantoms + sender)
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    // Gather receiver (12,9) only
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    // Whether this core uses per-core sender index for scattered gather
    static constexpr bool gather_use_per_core_sender_idx =
        get_named_compile_time_arg_val("gather_use_per_core_sender_idx") == 1;
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader)
// - Matmul reader (112 cores): setup sharded weight buffers
// - Mcast receiver (129 cores): receive mcast data
// - Gather sender (112 cores): send matmul output to gather core
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
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

    // Gather sender args
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
        get_named_compile_time_arg_val("gather_sender_idx"),
    };

// ============================================================================
// BRISC (Writer)
// - Mcast sender (12,9): broadcast input to 130-core grid
// - Gather receiver (12,9): receive from 112 matmul cores
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)
    // Mcast1 sender args
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;  // loopback = false (sender does not need its own mcast data)

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

    // Gather receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

// ============================================================================
// TRISC (Compute)
// - Matmul compute (112 cores)
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // Mcast1 CTArgs (no-op for TRISC)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Mcast2 CTArgs (no-op for TRISC)
    using Mcast2CTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast2_args{};

    // Matmul CTArgs
    using MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Setup sharded buffers (NCRISC only)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Mcast source buffer setup (sender core only)
    if constexpr (Core::is_mcast_sender_core) {
        constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast_src_cb, mcast_src_num_pages);

        constexpr uint32_t mcast2_src_cb = get_named_compile_time_arg_val("mcast2_src_cb");
        constexpr uint32_t mcast2_src_num_pages = get_named_compile_time_arg_val("mcast2_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast2_src_cb, mcast2_src_num_pages);
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
    // Mcast1: (12,9) -> 129 receivers in 13x10 grid
    // Broadcasts input [1, K] to all cores
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core,  // IsMcastGridCore (for semaphore wait on non-receiver cores)
        Core::is_mcast_receiver_core,  // IsReceiverCore (for CB reserve/push)
        true>                          // pop_src
        mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST1");
        mcast(mcast_args);
    }

    // ========================================================================
    // Mcast2: (12,9) -> 129 receivers in 13x10 grid
    // Broadcasts residual input [1, N] to all cores
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<
        Mcast2CTArgs,
        Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core,  // IsMcastGridCore
        Core::is_mcast_receiver_core,  // IsReceiverCore
        true>                          // pop_src
        mcast2;
    {
        DeviceZoneScopedN("MCAST2");
        mcast2(mcast2_args);
    }
    mcast.teardown();

    // ========================================================================
    // Matmul: [1, K] x [K, N_per_core] -> [1, N_per_core] on 112 cores
    // Input: mcast_dst_cb (CB 1), Weights: matmul_in1 (CB 2), Output: matmul_out (CB 3)
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        // pop_in0 = true (mcast output consumed), pop_in1 = false (weights persistent)
        deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Residual add: matmul_out + shard(residual) -> residual_add_out on 112 matmul cores
    // Each core indexes into the full [1, N] CB6 at offset core_idx * out_w_per_core
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
    // Gather: 112 matmul cores -> (12,9)
    // Collects [1, N_per_core] * 112 = [1, N]
    // Uses UsePerCoreSenderIdx=true for scattered (non-rectangular) core layout
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        deepseek_b1_ops::Gather::Op<
            Core::is_matmul_core,                  // IsSenderCore
            Core::is_gather_receiver_core,         // IsReceiverCore
            true,                                  // pop_src
            Core::gather_use_per_core_sender_idx>  // UsePerCoreSenderIdx
            gather;
        gather(gather_args);
    }
}
