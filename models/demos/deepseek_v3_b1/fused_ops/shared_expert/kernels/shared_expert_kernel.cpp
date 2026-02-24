// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Shared expert unified kernel
//
// Fuses: Activation Mcast + Gate/Up Matmul + Gather + GatedLocalReduce
//        + Mcast1 + Mcast2 + Down Proj Matmul + ResidualAdd + Output Gather
//
// Sender core (12,9):
//   BRISC:  act_mcast send → 130-core grid,
//           A_gather receive, B_gather receive,
//           mcast1(mcast_src) → 130-core grid,
//           mcast2(residual) → 130-core grid,
//           output gather receive
//   TRISC:  gated reduce → mcast_src
//   NCRISC: setup_sharded_buffer(act_mcast_src), act_mcast recv (grid only),
//           setup_sharded_buffer(down_weights), mcast1 recv, mcast2 recv,
//           output gather send
//
// Gate compute cores (64 A cores):
//   NCRISC: setup_sharded_buffer(gate_up_weights), act_mcast recv,
//           gate_gather send, mcast1 recv, mcast2 recv, output gather send (if matmul)
//   TRISC:  gate matmul [1, k_per_core] x [k_per_core, 1] → 1 tile,
//           down proj matmul (if matmul core), residual add (if matmul core)
//
// Up compute cores (64 B cores):
//   NCRISC: setup_sharded_buffer(gate_up_weights), act_mcast recv,
//           up_gather send, mcast1 recv, mcast2 recv, output gather send (if matmul)
//   TRISC:  up matmul [1, k_per_core] x [k_per_core, 1] → 1 tile,
//           down proj matmul (if matmul core), residual add (if matmul core)
//
// Idle core (12,8):
//   NCRISC: act_mcast recv (grid only), mcast1 recv (grid only), mcast2 recv (grid only)
//
// CB Layout:
//   CB 0:  A_gather_dst / reduce_g1 (sender, 64 face tiles, manual, face-view)
//   CB 1:  B_gather_dst / reduce_g2 (sender, 64 face tiles, manual, face-view)
//   CB 2:  intermediate (sender, 2 face tiles, manual)
//   CB 3:  mcast1 source (sender, 1 face tile, manual, TRISC-filled)
//   CB 4:  mcast1 dest / down_matmul_in0 (all 130 cores, manual)
//   CB 5:  down_weights (112 matmul cores, tensor-backed)
//   CB 6:  down_matmul_out (112 matmul cores, manual)
//   CB 7:  output_gather_dst (sender, tensor-backed)
//   CB 8:  act_mcast_src (sender, tensor-backed)
//   CB 9:  act_mcast_recv (all 130 cores, manual)
//   CB 10: residual_mcast_src (sender, tensor-backed)
//   CB 11: residual_mcast_dst (all 130 cores, manual)
//   CB 12: residual_add_out (112 matmul cores, manual)
//   CB 13: gate_up_weights (128 compute cores, tensor-backed)
//   CB 14: gate_up_matmul_out (128 compute cores, 1 tile, manual)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/gated_reduce.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/kn_sliced_matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/residual_add.hpp"

struct Core {
    static constexpr bool is_gate_compute_core = get_named_compile_time_arg_val("is_gate_compute_core") == 1;
    static constexpr bool is_up_compute_core = get_named_compile_time_arg_val("is_up_compute_core") == 1;
    static constexpr bool is_compute_core = is_gate_compute_core || is_up_compute_core;
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
    // Activation mcast receiver args
    using ActMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs act_mcast_args{
        get_named_compile_time_arg_val("act_mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("act_mcast_dst_cb"),
        get_named_compile_time_arg_val("act_mcast_dst_num_pages"),
    };

    // Gated reduce reader args (NCRISC no-op)
    using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ReaderCTArgs;
    deepseek_b1_ops::GatedReduce::ReaderArgs gated_reduce_args{};

    // Gate gather (A) sender args
    deepseek_b1_ops::Gather::SenderArgs ag_args{
        get_named_compile_time_arg_val("ag_dest_noc_x"),
        get_named_compile_time_arg_val("ag_dest_noc_y"),
        get_named_compile_time_arg_val("ag_data_size_bytes"),
        get_named_compile_time_arg_val("ag_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ag_src_cb"),
        get_named_compile_time_arg_val("ag_src_num_pages"),
        0,  // sender_grid_start_x (unused, per-core idx)
        0,  // sender_grid_start_y
        0,  // sender_grid_end_x
        0,  // sender_grid_end_y
        1,  // row_major
        get_named_compile_time_arg_val("ag_receiver_data_addr"),
        get_named_compile_time_arg_val("ag_sender_idx"),
    };

    // Up gather (B) sender args
    deepseek_b1_ops::Gather::SenderArgs bg_args{
        get_named_compile_time_arg_val("bg_dest_noc_x"),
        get_named_compile_time_arg_val("bg_dest_noc_y"),
        get_named_compile_time_arg_val("bg_data_size_bytes"),
        get_named_compile_time_arg_val("bg_receiver_semaphore_id"),
        get_named_compile_time_arg_val("bg_src_cb"),
        get_named_compile_time_arg_val("bg_src_num_pages"),
        0,  // sender_grid_start_x (unused, per-core idx)
        0,  // sender_grid_start_y
        0,  // sender_grid_end_x
        0,  // sender_grid_end_y
        1,  // row_major
        get_named_compile_time_arg_val("bg_receiver_data_addr"),
        get_named_compile_time_arg_val("bg_sender_idx"),
    };

    // Mcast1 receiver args (down proj activation)
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Mcast2 receiver args (residual, shares init/teardown with mcast1)
    deepseek_b1_ops::Mcast::ReceiverArgs mcast2_args{
        get_named_compile_time_arg_val("mcast2_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast2_dst_cb"),
        get_named_compile_time_arg_val("mcast2_dst_num_pages"),
    };

    // Gate/Up sliced matmul reader args (NCRISC no-op)
    using KNSlicedMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ReaderCTArgs;
    deepseek_b1_ops::KNSlicedMatmul::ReaderArgs gu_matmul_args{};

    // Down proj matmul reader args (NCRISC no-op)
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
    // Activation mcast sender args
    using ActMcastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("act_mcast_num_cores"),
        get_named_compile_time_arg_val("act_mcast_is_part_of_receiver_grid") == 1,
        /*Loopback=*/false>;

    constexpr uint32_t act_mcast_src_cb = get_named_compile_time_arg_val("act_mcast_src_cb");
    constexpr uint32_t act_mcast_dst_cb = get_named_compile_time_arg_val("act_mcast_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs act_mcast_args{
        get_named_compile_time_arg_val("act_mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("act_mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("act_mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("act_mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("act_mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("act_mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("act_mcast_data_size_bytes"),
        act_mcast_src_cb,
        get_named_compile_time_arg_val("act_mcast_src_num_pages"),
        get_read_ptr(act_mcast_src_cb),
        get_write_ptr(act_mcast_dst_cb),
    };

    // Gated reduce writer args (BRISC no-op)
    using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::WriterCTArgs;
    deepseek_b1_ops::GatedReduce::WriterArgs gated_reduce_args{};

    // Gate gather (A) receiver args
    deepseek_b1_ops::Gather::ReceiverArgs ag_args{
        get_named_compile_time_arg_val("ag_noc0_num_senders"),
        0,
        get_named_compile_time_arg_val("ag_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ag_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("ag_dst_cb"),
        get_named_compile_time_arg_val("ag_dst_num_pages"),
    };

    // Up gather (B) receiver args
    deepseek_b1_ops::Gather::ReceiverArgs bg_args{
        get_named_compile_time_arg_val("bg_noc0_num_senders"),
        0,
        get_named_compile_time_arg_val("bg_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("bg_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("bg_dst_cb"),
        get_named_compile_time_arg_val("bg_dst_num_pages"),
    };

    // Mcast1 sender args (down proj activation)
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

    // Mcast2 sender args (residual, shares init/teardown with mcast1)
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

    // Gate/Up sliced matmul writer args (BRISC no-op)
    using KNSlicedMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::WriterCTArgs;
    deepseek_b1_ops::KNSlicedMatmul::WriterArgs gu_matmul_args{};

    // Down proj matmul writer args (BRISC no-op)
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
    // Activation mcast (no-op for TRISC)
    using ActMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs act_mcast_args{};

    // Gate/Up gather compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs ag_args{};
    deepseek_b1_ops::Gather::ComputeArgs bg_args{};

    // Mcast1 CTArgs (no-op for TRISC)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Mcast2 (no-op for TRISC, shares init/teardown with mcast1, residual)
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

    // Gate/Up sliced matmul CTArgs
    using KNSlicedMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ComputeCTArgs<>;
    deepseek_b1_ops::KNSlicedMatmul::ComputeArgs gu_matmul_args{
        get_named_compile_time_arg_val("gu_act_cb"),
        get_named_compile_time_arg_val("gu_weights_cb"),
        get_named_compile_time_arg_val("gu_out_cb"),
        get_named_compile_time_arg_val("gu_k_offset"),
        get_named_compile_time_arg_val("gu_k_per_core"),
        get_named_compile_time_arg_val("gu_act_total_tiles"),
    };

    // Down proj matmul CTArgs
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
    // Activation mcast source on sender core
    if constexpr (Core::is_mcast_sender_core) {
        constexpr uint32_t act_src_cb = get_named_compile_time_arg_val("act_mcast_src_cb");
        constexpr uint32_t act_src_pages = get_named_compile_time_arg_val("act_mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(act_src_cb, act_src_pages);
    }
    // Gate/Up weights on 128 compute cores
    if constexpr (Core::is_compute_core) {
        constexpr uint32_t gu_weights_cb = get_named_compile_time_arg_val("gu_weights_cb");
        constexpr uint32_t gu_weights_num_pages = get_named_compile_time_arg_val("gu_weights_num_pages");
        unified_kernels::setup_sharded_buffer(gu_weights_cb, gu_weights_num_pages);
    }
#endif

    // ========================================================================
    // Phase 1: Activation Mcast — sender (12,9) → all 130 cores
    // Broadcasts [1, 7168] activation to all cores
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<
        ActMcastCTArgs,
        Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core,
        Core::is_mcast_receiver_core,
        /*pop_src=*/true>
        act_mcast;
    act_mcast.init(act_mcast_args);
    {
        DeviceZoneScopedN("ACT_MCAST");
        act_mcast(act_mcast_args);
    }

    // ========================================================================
    // Phase 2: Gate/Up Matmul on 128 compute cores
    // A cores: activation[k_offset..k_offset+k_per_core] @ gate_weights → 1 tile
    // B cores: activation[k_offset..k_offset+k_per_core] @ up_weights → 1 tile
    // ========================================================================
    {
        DeviceZoneScopedN("GATE_UP_MATMUL");
        deepseek_b1_ops::KNSlicedMatmul::
            Op<KNSlicedMatmulCTArgs, Core::is_compute_core, /*pop_act=*/true, /*pop_weights=*/false>
                gu_matmul;
        gu_matmul(gu_matmul_args);
    }

    // ========================================================================
    // Phase 3: Gate Gather (A) — 64 A cores → CB0 on (12,9)
    // ========================================================================
    {
        DeviceZoneScopedN("GATE_GATHER");
        deepseek_b1_ops::Gather::
            Op<Core::is_gate_compute_core, Core::is_gated_reduce_core, /*pop_src=*/true, Core::is_gate_compute_core>
                gate_gather;
        gate_gather(ag_args);
    }

    // ========================================================================
    // Phase 3b: Up Gather (B) — 64 B cores → CB1 on (12,9)
    // ========================================================================
    {
        DeviceZoneScopedN("UP_GATHER");
        deepseek_b1_ops::Gather::
            Op<Core::is_up_compute_core, Core::is_gated_reduce_core, /*pop_src=*/true, Core::is_up_compute_core>
                up_gather;
        up_gather(bg_args);
    }

    // ========================================================================
    // Phase 4: Gated Local Reduce (TRISC on sender core)
    // SiLU(sum(gate_partials)) * sum(up_partials) → [1, 256]
    // ========================================================================
    {
        DeviceZoneScopedN("GATED_REDUCE");
        deepseek_b1_ops::GatedReduce::Op<GatedReduceCTArgs, Core::is_gated_reduce_core> gated_reduce;
        gated_reduce(gated_reduce_args);
    }

    // ========================================================================
    // Phases 5-6: Mcast1 + Mcast2 — sender (12,9) → 129 receivers in 13x10 grid
    // Shared init/teardown: same grid, same semaphores, sequential sends
    // Mcast1: gated reduce output [1, K_down], Mcast2: residual [1, N]
    // ========================================================================
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_mcast_sender_core,
        Core::is_mcast_receiver_core,
        Core::is_mcast_receiver_core,
        /*pop_src=*/true>
        mcast;
    {
        DeviceZoneScopedN("MCAST1");
        mcast(mcast_args);
    }

    // ========================================================================
    // Setup down proj weights (NCRISC, after mcast1 since we need buffer space)
    // ========================================================================
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t matmul_out_w_per_core = get_named_compile_time_arg_val("matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w_per_core);
    }
    // Mcast2 source (residual on sender core)
    if constexpr (Core::is_mcast_sender_core) {
        constexpr uint32_t m2_src_cb = get_named_compile_time_arg_val("mcast2_src_cb");
        constexpr uint32_t m2_src_pages = get_named_compile_time_arg_val("mcast2_src_num_pages");
        unified_kernels::setup_sharded_buffer(m2_src_cb, m2_src_pages);
    }
#endif

    {
        DeviceZoneScopedN("MCAST2");
        mcast(mcast2_args);
    }
    act_mcast.teardown();

    // ========================================================================
    // Phase 7: Down Proj Matmul — [1, K_down] x [K_down, N_per_core] on 112 cores
    // ========================================================================
    {
        DeviceZoneScopedN("DOWN_MATMUL");
        deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, /*pop_in0=*/true, /*pop_in1=*/false> matmul;
        matmul(matmul_args);
    }

    // ========================================================================
    // Phase 8: Residual add — matmul_out + shard(residual) → residual_add_out on 112 cores
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
    // Phase 9: Output Gather — 112 matmul cores → (12,9)
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
