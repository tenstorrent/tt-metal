// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LM Head Sampling Unified Kernel: CCL Broadcast + Mcast + Matmul for Vocab Projection
//
// Single .cpp compiled for all three RISC processors (NCRISC, BRISC, TRISC).
// Compile-time role flags (is_input_core, is_mcast_receiver_core, is_matmul_core, skip_ccl)
// enable dead code elimination via `if constexpr`, so each core only runs its assigned path.
//
// Data flow:
//   1. CCL Broadcast (multi-device only): Sender device broadcasts input [1, K] to all
//      devices in the mesh via the fabric interconnect. Skipped when skip_ccl=true.
//   2. Mcast:  Sender core multicasts input [1, K] to all cores in the device grid
//   3. Matmul: Each matmul core computes [1, K] x [K, N_per_core] -> [1, N_per_core]
//
// RISC responsibilities:
//   NCRISC: CCL broadcast writer (fabric multicast to remote devices) + mcast receiver
//           (semaphore wait + CB push) + sharded buffer setup (mcast_src on sender core, weight shards on
//           matmul cores)
//   BRISC:  CCL broadcast reader + mcast sender
//           (reads mcast_src CB, NOC multicasts to all receiver cores)
//   TRISC:  Matmul compute (reads in0 from mcast_dst CB, in1 from weights CB, writes to out CB)
//
// CB layout (see op.py LMHeadSampling class for index definitions):
//   CB 0  (mcast_src):   Input tensor on sender core (tensor-backed).
//                         In multi-device mode, backed by intermediate_tensor (CCL broadcast
//                         destination). In single-device mode, backed by input_tensor directly.
//   CB 1  (mcast_dst):   Mcast destination / matmul in0 on all cores (intermediate)
//   CB 2  (matmul_in1):  Vocab weights on matmul cores (tensor-backed)
//   CB 16 (matmul_out):  Matmul output on matmul cores (tensor-backed)
//   CB 30 (bcast_pkt):   CCL broadcast packet buffer (multi-device mode only)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/broadcast.hpp"
#include "../../../unified_kernels/argmax.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"

// Per-core role flags set by UnifiedCompileTimeCoreDescriptor in op.py.
// Each flag is specialized per core group at compile time, enabling if constexpr
// to eliminate dead code paths (e.g., sender-only code on receiver cores).
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;
    static constexpr bool enable_argmax = get_named_compile_time_arg_val("enable_argmax") == 1;
    static constexpr bool is_argmax_core = is_matmul_core;
    static constexpr bool is_argmax_final_core = get_named_compile_time_arg_val("is_argmax_final_core") == 1;
    static constexpr bool is_argmax_mesh_sender_core =
        get_named_compile_time_arg_val("is_argmax_mesh_sender_core") == 1;
    static constexpr bool is_rmsnorm_core = get_named_compile_time_arg_val("is_rmsnorm_core") == 1;
};

void kernel_main() {
// ============================================================================
// Per-RISC compile-time arg setup
// Each RISC receives different named compile-time args from op.py and
// constructs the appropriate Broadcast/Mcast/Matmul arg structs for its role.
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    uint32_t ncrisc_rt_arg_idx = 0;
    // --- NCRISC: CCL broadcast writer + mcast receiver + sharded buffer setup ---

    // CCL Broadcast CTArgs type alias
    using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("bcast_num_targets_forward_direction"),
        get_named_compile_time_arg_val("bcast_num_targets_backward_direction"),
        get_named_compile_time_arg_val("bcast_is_sender"),
        get_named_compile_time_arg_val("bcast_core_noc_x"),
        get_named_compile_time_arg_val("bcast_core_noc_y"),
        get_named_compile_time_arg_val("bcast_is_secondary_sender"),
        get_named_compile_time_arg_val("bcast_has_secondary_target"),
        get_named_compile_time_arg_val("bcast_start_distance_in_hops_forward"),
        get_named_compile_time_arg_val("bcast_range_hops_forward"),
        get_named_compile_time_arg_val("bcast_start_distance_in_hops_backward"),
        get_named_compile_time_arg_val("bcast_range_hops_backward")>;

    // CCL Broadcast writer runtime args (only populated when not skip_ccl)
    deepseek_b1_ops::Broadcast::WriterArgs bcast_args{};
    if constexpr (!Core::skip_ccl) {
        bcast_args = deepseek_b1_ops::Broadcast::WriterArgs{
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // tensor_address0
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_bank_addr
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // wait_output_semaphore
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // reset_global_semaphore
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_noc0_x
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_noc0_y
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // out_ready_sem_wait_value
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem_noc0_x
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // barrier_sem_noc0_y
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // ring_index
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // secondary_sync_sem
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // num_connections (computed from len(dst_nodes))
        };
    }

    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul reader args (NCRISC is a no-op for matmul; compute runs on TRISC)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};
    using ArgmaxCTArgs = deepseek_b1_ops::Sampling::ReaderCTArgs<
        get_named_compile_time_arg_val("argmax_num_values"),
        get_named_compile_time_arg_val("argmax_winner_page_bytes"),
        get_named_compile_time_arg_val("argmax_num_senders"),
        get_named_compile_time_arg_val("argmax_expected_remote_incs"),
        get_named_compile_time_arg_val("argmax_receiver_semaphore_id"),
        get_named_compile_time_arg_val("argmax_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("argmax_mesh_mode"),
        get_named_compile_time_arg_val("argmax_stage1_sender"),
        get_named_compile_time_arg_val("argmax_stage1_receiver"),
        get_named_compile_time_arg_val("argmax_stage2_sender"),
        get_named_compile_time_arg_val("argmax_stage2_receiver"),
        get_named_compile_time_arg_val("argmax_stage1_slot_base_offset"),
        get_named_compile_time_arg_val("argmax_stage1_num_slots"),
        get_named_compile_time_arg_val("argmax_stage1_expected_remote_incs"),
        get_named_compile_time_arg_val("argmax_stage1_local_slot_offset"),
        get_named_compile_time_arg_val("argmax_stage2_slot_base_offset"),
        get_named_compile_time_arg_val("argmax_stage2_num_slots"),
        get_named_compile_time_arg_val("argmax_stage2_expected_remote_incs"),
        get_named_compile_time_arg_val("argmax_stage2_local_slot_offset"),
        get_named_compile_time_arg_val("argmax_mesh_local_send_slot_offset"),
        get_named_compile_time_arg_val("argmax_sender_idx"),
        get_named_compile_time_arg_val("argmax_socket_mode"),
        get_named_compile_time_arg_val("argmax_socket_cb"),
        get_named_compile_time_arg_val("argmax_socket_page_size_bytes")>;

    // Matmul cores: register matmul_in1 CB (CB 2) backed by vocab weight shards
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
        unified_kernels::setup_sharded_buffer(in1_cb, num_tiles_k * out_w);
    }
#elif defined(COMPILE_FOR_BRISC)
    uint32_t brisc_rt_arg_idx = 0;
    // --- BRISC: CCL broadcast reader + mcast sender ---
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender")>;

    // CCL Broadcast reader runtime args (empty payload by design)
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{};

    // Template params: <num_cores, is_sender_in_receiver_grid, loopback>
    // loopback=false because sender does not consume its own multicast data
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;

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
        Core::is_input_core ? get_read_ptr(mcast_src_cb) : 0,
        get_write_ptr(mcast_dst_cb),
    };

    // Matmul writer args (BRISC is a no-op for matmul; compute runs on TRISC)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};
    using ArgmaxCTArgs = deepseek_b1_ops::Sampling::WriterCTArgs<
        get_named_compile_time_arg_val("argmax_winner_page_bytes"),
        get_named_compile_time_arg_val("argmax_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("argmax_socket_mode"),
        get_named_compile_time_arg_val("argmax_socket_cb"),
        get_named_compile_time_arg_val("argmax_socket_page_size_bytes")>;

#elif defined(COMPILE_FOR_TRISC)
    // --- TRISC: Matmul compute ---
    // CCL Broadcast CTArgs (no-op for TRISC)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
    deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};

    // Mcast is a no-op on TRISC (data movement handled by NCRISC/BRISC)
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul compute: [1, K] x [K, N_per_core] -> [1, N_per_core]
    // out_w (output tiles per core) is a compile-time template param for loop unrolling
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb")>;
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(1),     // scalar (1/sqrt(numel))
    };

    using MatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w")>;

    // CB indices and tile count from op.py compile-time args
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("matmul_in0");  // CB 1: mcast_dst
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");  // CB 2: vocab weights
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("matmul_out");  // CB 16: matmul output
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");

    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        .in0 = in0_cb,
        .in1 = in1_cb,
        .out = out_cb,
        .k_num_tiles = num_tiles_k,
    };
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // Phase 0 (multi-device only): CCL Broadcast — replicate input from sender
    // device to all devices in the mesh via the fabric interconnect.
    // Only the input core participates (writer on NCRISC, reader on BRISC).
    // After this phase, every device has the input in its intermediate tensor
    // (which backs CB 0 / mcast_src).
    // ========================================================================
    if constexpr (!Core::skip_ccl) {
        deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
        {
            DeviceZoneScopedN("CCL_BROADCAST");
            bcast(bcast_args);
        }
    }

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers so BRISC/TRISC can access tensor data.
    // Sender core: register RMSNorm input CB backed by input_tensor (skip_ccl)
    // or intermediate_tensor (CCL mode, where broadcast placed the data)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
    }
#endif

#if defined(COMPILE_FOR_TRISC)
    deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_rmsnorm_core, true> rmsnorm;
    {
        DeviceZoneScopedN("RMSNORM");
        rmsnorm(rmsnorm_args);
    }
#endif

    // ========================================================================
    // Phase 1: Mcast — multicast input from sender core to all device cores
    //
    // Template params: <CTArgs, IsSender, IsMcastGridCore, IsReceiverCore, PopSrc>
    //   IsMcastGridCore: participates in semaphore-based sync (all receivers)
    //   IsReceiverCore:  performs CB reserve/push for incoming data (all receivers)
    //   PopSrc:          sender pops mcast_src CB after send (frees tensor-backed buffer)
    // ========================================================================
    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_input_core, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, true>
            mcast;

    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        mcast(mcast_args);
    }

    mcast.teardown();

    // ========================================================================
    // Phase 2: Matmul — each matmul core computes local GEMM with its weight shard
    //
    // Template params: <CTArgs, IsActive, PopIn0, PopIn1>
    //   IsActive: only matmul cores execute; others are no-ops
    //   PopIn0:   pop mcast_dst CB (CB 1) after read (frees intermediate buffer)
    //   PopIn1:   pop matmul_in1 CB (CB 2) after read (frees weight buffer)
    // ========================================================================
    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, true> matmul;
    {
        DeviceZoneScopedN("MATMUL");
        matmul(matmul_args);
    }

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#if defined(COMPILE_FOR_NCRISC)
        constexpr uint32_t gather_cb = get_named_compile_time_arg_val("argmax_gather_cb");
        uint32_t scores_addr = 0;
        if constexpr (Core::is_matmul_core) {
            // Matmul (TRISC) pushes matmul_out CB; wait before NCRISC consumes scores.
            constexpr uint32_t matmul_out_cb = get_named_compile_time_arg_val("matmul_out");
            constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
            cb_wait_front(matmul_out_cb, out_w);
            scores_addr = get_read_ptr(matmul_out_cb);
        }
        deepseek_b1_ops::Sampling::ReaderArgs sampling_args{
            scores_addr,
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
            get_write_ptr(gather_cb),
        };
#elif defined(COMPILE_FOR_BRISC)
    deepseek_b1_ops::Sampling::WriterArgs sampling_args{
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
    };
#endif
        // k=1 fast path: fused sampling invocation matches micro-op style.
        deepseek_b1_ops::Sampling::
            Op<ArgmaxCTArgs, Core::is_matmul_core, Core::is_argmax_final_core, Core::is_argmax_mesh_sender_core>
                sampling_op;
        {
            DeviceZoneScopedN("ARGMAX");
            sampling_op(sampling_args);
        }
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_matmul_core) {
            constexpr uint32_t matmul_out_cb = get_named_compile_time_arg_val("matmul_out");
            constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
            cb_pop_front(matmul_out_cb, out_w);
        }
#endif
#endif
}
