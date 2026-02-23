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
//   NCRISC: CCL broadcast reader (on sender device) + mcast receiver (semaphore wait +
//           CB push) + sharded buffer setup (mcast_src on sender core, weight shards on
//           matmul cores)
//   BRISC:  CCL broadcast writer (fabric multicast to remote devices) + mcast sender
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

// Per-core role flags set by UnifiedCompileTimeCoreDescriptor in op.py.
// Each flag is specialized per core group at compile time, enabling if constexpr
// to eliminate dead code paths (e.g., sender-only code on receiver cores).
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;
};

void kernel_main() {
// ============================================================================
// Per-RISC compile-time arg setup
// Each RISC receives different named compile-time args from op.py and
// constructs the appropriate Broadcast/Mcast/Matmul arg structs for its role.
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // --- NCRISC: CCL broadcast reader + mcast receiver + sharded buffer setup ---

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
            get_common_arg_val<uint32_t>(0),   // tensor_address0
            get_common_arg_val<uint32_t>(1),   // out_ready_sem_bank_addr
            get_common_arg_val<uint32_t>(2),   // wait_output_semaphore
            get_common_arg_val<uint32_t>(3),   // reset_global_semaphore
            get_common_arg_val<uint32_t>(4),   // out_ready_sem_noc0_x
            get_common_arg_val<uint32_t>(5),   // out_ready_sem_noc0_y
            get_common_arg_val<uint32_t>(6),   // out_ready_sem_wait_value
            get_common_arg_val<uint32_t>(7),   // barrier_sem
            get_common_arg_val<uint32_t>(8),   // barrier_sem_noc0_x
            get_common_arg_val<uint32_t>(9),   // barrier_sem_noc0_y
            get_common_arg_val<uint32_t>(10),  // ring_index
            get_common_arg_val<uint32_t>(11),  // secondary_sync_sem
            get_common_arg_val<uint32_t>(12),  // num_connections (computed from len(dst_nodes))
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

    // Matmul cores: register matmul_in1 CB (CB 2) backed by vocab weight shards
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
        unified_kernels::setup_sharded_buffer(in1_cb, num_tiles_k * out_w);
    }

#elif defined(COMPILE_FOR_BRISC)
    // --- BRISC: CCL broadcast reader + mcast sender ---
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender")>;

    // CCL Broadcast reader runtime args (only populated when not skip_ccl)
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
        get_read_ptr(mcast_src_cb),
        get_write_ptr(mcast_dst_cb),
    };

    // Matmul writer args (BRISC is a no-op for matmul; compute runs on TRISC)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

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
    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);
#endif

    // ========================================================================
    // Phase 0 (multi-device only): CCL Broadcast — replicate input from sender
    // device to all devices in the mesh via the fabric interconnect.
    // Only the input core participates (reader on NCRISC, writer on BRISC).
    // After this phase, every device has the input in its intermediate tensor
    // (which backs CB 0 / mcast_src).
    // ========================================================================
    if constexpr (!Core::skip_ccl) {
        deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
        bcast(bcast_args);
    }

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers so BRISC/TRISC can access tensor data.
    // Sender core: register mcast_src CB (CB 0) backed by input_tensor (skip_ccl)
    // or intermediate_tensor (CCL mode, where broadcast placed the data)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast_src_cb, mcast_src_num_pages);
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
    mcast(mcast_args);
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
    matmul(matmul_args);
}
