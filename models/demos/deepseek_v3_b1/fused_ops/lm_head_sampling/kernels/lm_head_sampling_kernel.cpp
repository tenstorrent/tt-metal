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
//   4. Argmax: Fused k=1 sampling across all matmul cores (and optionally across devices)
//   5. MTP Fusion (optional): Embedding lookup, RMSNorm, concat, mcast, EH projection matmul
//
// RISC responsibilities:
//   NCRISC: CCL broadcast writer (fabric multicast to remote devices) + mcast receiver
//           (semaphore wait + CB push) + sharded buffer setup (mcast_src on sender core, weight shards on
//           matmul cores) + argmax reader + MTP token transfer + embedding DRAM fetch + concat assembly
//   BRISC:  CCL broadcast reader + mcast sender + argmax writer + MTP mcast sender
//   TRISC:  RMSNorm compute + Matmul compute + MTP h/e RMSNorm + EH matmul
//
// CB layout (see op.py LMHeadSampling class for index definitions):
//   CB 0  (mcast_src):   Input tensor on sender core (tensor-backed).
//                         In multi-device mode, backed by intermediate_tensor (CCL broadcast
//                         destination). In single-device mode, backed by input_tensor directly.
//   CB 1  (mcast_dst):   Mcast destination / matmul in0 on all cores (intermediate)
//   CB 2  (matmul_in1):  Vocab weights on matmul cores (tensor-backed)
//   CB 9  (matmul_eh):   [MTP] EH projection weights on matmul cores (tensor-backed)
//   CB 10 (embedding):   [MTP] Embedding row for e_rmsnorm input
//   CB 11 (h_gamma):     [MTP] RMSNorm gamma for hidden states (tensor-backed)
//   CB 12 (e_gamma):     [MTP] RMSNorm gamma for embeddings (tensor-backed)
//   CB 14 (e_norm):      [MTP] e_rmsnorm output (intermediate)
//   CB 15 (mcast_eh_src):[MTP] [h_norm|e_norm] (intermediate)
//   CB 16 (matmul_out):  Matmul output on matmul cores (tensor-backed)
//   CB 17 (matmul_eh_out):[MTP] EH matmul output (tensor-backed)
//   CB 18 (mcast_eh_dst):[MTP] Mcast destination for concat on all cores(intermediate)
//   CB 30 (bcast_pkt):   CCL broadcast packet buffer (multi-device mode only)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/broadcast.hpp"
#include "../../../unified_kernels/argmax.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"

// Per-core role flags set by UnifiedCompileTimeCoreDescriptor in op.py.
// Each flag is specialized per core group at compile time, enabling if constexpr
// to eliminate dead code paths (e.g., sender-only code on receiver cores).
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;
    static constexpr bool enable_argmax = get_named_compile_time_arg_val("enable_argmax") == 1;
    static constexpr uint32_t input_socket_mode = get_named_compile_time_arg_val("input_socket_mode");
    static constexpr uint32_t input_socket_mode_none = 0;
    static constexpr uint32_t input_socket_mode_d2d = 2;
    static constexpr bool bcast_use_socket_input = input_socket_mode == input_socket_mode_d2d;
    static constexpr bool is_argmax_core = is_matmul_core;
    static constexpr bool is_argmax_final_core = get_named_compile_time_arg_val("is_argmax_final_core") == 1;
    static constexpr bool is_argmax_mesh_sender_core =
        get_named_compile_time_arg_val("is_argmax_mesh_sender_core") == 1;
    static constexpr bool is_rmsnorm_core = get_named_compile_time_arg_val("is_rmsnorm_core") == 1;
    static constexpr bool persistent_mode = get_named_compile_time_arg_val("persistent_mode") == 1;
    static constexpr uint32_t mesh_row = get_named_compile_time_arg_val("mesh_row");
    static constexpr uint32_t mesh_col = get_named_compile_time_arg_val("mesh_col");
    static_assert(input_socket_mode != 1, "lm_head_sampling input socket mode=1 is invalid");
    static constexpr bool enable_mtp = get_named_compile_time_arg_val("enable_mtp") == 1;
    static constexpr bool enable_mtp_verification = get_named_compile_time_arg_val("enable_mtp_verification") == 1;
    static constexpr bool is_eh_matmul_core = enable_mtp && get_named_compile_time_arg_val("is_eh_matmul_core") == 1;
    static constexpr bool has_bypass_socket_output = get_named_compile_time_arg_val("has_bypass_socket_output") == 1;
    static constexpr bool has_bypass_socket_input = get_named_compile_time_arg_val("has_bypass_socket_input") == 1;
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
    uint32_t mtp_token_addr = 0;

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

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

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
        get_semaphore(get_named_compile_time_arg_val("mcast_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Setup MTP tensor-backed CBs on sender core
    if constexpr (Core::enable_mtp) {
        if constexpr (Core::is_input_core) {
            constexpr uint32_t h_gamma_cb = get_named_compile_time_arg_val("h_gamma_cb");
            constexpr uint32_t rmsnorm_h_num_tiles = get_named_compile_time_arg_val("rmsnorm_h_num_tiles");
            unified_kernels::setup_sharded_buffer(h_gamma_cb, rmsnorm_h_num_tiles);
            constexpr uint32_t e_gamma_cb = get_named_compile_time_arg_val("e_gamma_cb");
            constexpr uint32_t rmsnorm_e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");
            unified_kernels::setup_sharded_buffer(e_gamma_cb, rmsnorm_e_num_tiles);
        }
    }

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
        get_named_compile_time_arg_val("argmax_socket_page_size_bytes"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_out_w"),
        get_named_compile_time_arg_val("argmax_gather_cb")>;

    // Matmul cores: register matmul_in1 CB (CB 2) backed by vocab weight shards
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
        unified_kernels::setup_sharded_buffer(in1_cb, num_tiles_k * out_w);
    }

    deepseek_b1_ops::Sampling::ReaderArgs sampling_args{
        .scores_addr = 0,
        .indices_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .output_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .final_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .final_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .scratch_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .global_sem_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .global_stage2_sem_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .gather_addr = 0,
    };

    // Pre-consume MTP / bypass / verification runtime args before the main
    // Consumption order must match op.py's ncrisc_bcast_common_args layout:
    //   MTP(4) → BYPASS_SEND(2) → VERIFY(3) → BYPASS_RECV(1)
    uint32_t mtp_input_core_noc_x = 0;
    uint32_t mtp_input_core_noc_y = 0;
    uint32_t mtp_embedding_base = 0;
    if constexpr (Core::enable_mtp) {
        mtp_input_core_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_input_core_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_embedding_base = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    uint32_t bypass_sender_config_addr = 0;
    uint32_t bypass_staging_addr = 0;
    if constexpr (Core::has_bypass_socket_output) {
        bypass_sender_config_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        bypass_staging_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    uint32_t mtp_reference_token_addr = 0;
    uint32_t mtp_verification_result_addr = 0;
    uint32_t mtp_speculative_token_addr = 0;
    if constexpr (Core::enable_mtp_verification) {
        mtp_reference_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_verification_result_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_speculative_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    uint32_t bypass_recv_config_addr = 0;
    if constexpr (Core::has_bypass_socket_input) {
        bypass_recv_config_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    // Setup sharded persistent buffers so BRISC/TRISC can access tensor data.
    // Sender core: register RMSNorm input CB backed by input_tensor (skip_ccl)
    // or intermediate_tensor (CCL mode, where broadcast placed the data)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        // In skip_ccl + socket mode BRISC owns CB push for rmsnorm_input_cb.
        if constexpr (!(Core::skip_ccl && Core::bcast_use_socket_input)) {
            unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        }
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
    }

    // [MTP] Second mcast CTArgs + args (NCRISC receiver)
    using McastEhCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_eh_args{
        get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_eh_dst_cb"),
        get_named_compile_time_arg_val("mcast_eh_dst_num_pages"),
    };

    // [MTP] EH DRAM streaming matmul CTArgs (NCRISC reader)
    constexpr uint32_t eh_in1_cb = get_named_compile_time_arg_val("matmul_eh_in1");
    constexpr uint32_t eh_out_cb = get_named_compile_time_arg_val("matmul_eh_out");
    constexpr uint32_t eh_out_w = get_named_compile_time_arg_val("matmul_eh_out_w");
    constexpr uint32_t eh_cb_in1_buf_addr = get_named_compile_time_arg_val("eh_matmul_cb_in1_buf_addr");
    using EHDRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
        eh_in1_cb,
        eh_out_cb,
        get_named_compile_time_arg_val("matmul_eh_dram_in1_tensor_addr"),
        get_named_compile_time_arg_val("matmul_eh_dram_in1_page_size"),
        get_named_compile_time_arg_val("matmul_eh_dram_in1_num_pages"),
        get_named_compile_time_arg_val("matmul_eh_subblock_k"),
        eh_out_w,
        get_named_compile_time_arg_val("matmul_eh_dram_in1_block_size_bytes"),
        get_named_compile_time_arg_val("matmul_eh_out_num_tiles"),
        get_named_compile_time_arg_val("matmul_eh_num_subblocks_k"),
        get_named_compile_time_arg_val("matmul_eh_bank_id"),
        get_named_compile_time_arg_val("matmul_eh_vc")>;

#elif defined(COMPILE_FOR_BRISC)
    uint32_t brisc_rt_arg_idx = 0;
    // --- BRISC: CCL broadcast reader + optional socket-reader path + mcast sender ---
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender"),
        (get_named_compile_time_arg_val("input_socket_mode") == 2 ? 1 : 0)>;

    // BRISC common args layout:
    // [0..3] argmax writer args, [4..6] optional socket-input reader args,
    // [7..12] persistent signal routing metadata.
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{
        get_common_arg_val<uint32_t>(4),  // socket_config_addr
        get_common_arg_val<uint32_t>(5),  // socket_page_size
        get_common_arg_val<uint32_t>(6),  // socket_num_pages
    };

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

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
        get_semaphore(get_named_compile_time_arg_val("mcast_data_sender_semaphore")),
        get_semaphore(get_named_compile_time_arg_val("mcast_data_receiver_semaphore")),
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

    deepseek_b1_ops::Sampling::WriterArgs sampling_args{
        .final_noc_x = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .final_noc_y = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .scratch_addr = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .socket_config_addr = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_enable = get_common_arg_val<uint32_t>(7),
        .persistent_dst_noc_x = get_common_arg_val<uint32_t>(8),
        .persistent_dst_noc_y = get_common_arg_val<uint32_t>(9),
        .persistent_dst_mesh_id = get_common_arg_val<uint32_t>(10),
        .persistent_dst_chip_id = get_common_arg_val<uint32_t>(11),
        .persistent_dst_sem_addr = get_common_arg_val<uint32_t>(12),
    };
    const uint32_t persistent_next_iter_global_sem_addr = get_common_arg_val<uint32_t>(12);

    // [MTP] Second mcast CTArgs + args (BRISC sender)
    using McastEhCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_eh_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;
    constexpr uint32_t mcast_eh_src_cb = get_named_compile_time_arg_val("mcast_eh_src_cb");
    constexpr uint32_t mcast_eh_dst_cb = get_named_compile_time_arg_val("mcast_eh_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast_eh_args{
        get_named_compile_time_arg_val("mcast_eh_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_eh_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_eh_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_eh_dest_noc_end_y"),
        get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_sender_semaphore")),
        get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_eh_data_size_bytes"),
        mcast_eh_src_cb,
        get_named_compile_time_arg_val("mcast_eh_src_num_pages"),
        Core::is_input_core ? get_read_ptr(mcast_eh_src_cb) : 0,
        get_write_ptr(mcast_eh_dst_cb),
    };

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

    using HRMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_h_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_h_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_h_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_h_output_cb")>;

    using ERMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_e_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_e_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_e_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_e_output_cb")>;

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

    using ArgmaxCTArgs = deepseek_b1_ops::Sampling::ComputeCTArgs;
    deepseek_b1_ops::Sampling::ComputeArgs sampling_args{};

    // [MTP] Second mcast is a no-op on TRISC
    using McastEhCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_eh_args{};

    // [MTP] EH DRAM streaming matmul CTArgs (TRISC compute)
    constexpr uint32_t eh_in0_cb = get_named_compile_time_arg_val("matmul_eh_in0");
    constexpr uint32_t eh_in1_cb = get_named_compile_time_arg_val("matmul_eh_in1");
    constexpr uint32_t eh_out_cb = get_named_compile_time_arg_val("matmul_eh_out");
    constexpr uint32_t eh_out_w = get_named_compile_time_arg_val("matmul_eh_out_w");
    constexpr uint32_t eh_cb_in1_buf_addr = get_named_compile_time_arg_val("eh_matmul_cb_in1_buf_addr");
    using EHDRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
        eh_in0_cb,
        eh_in1_cb,
        eh_out_cb,
        get_named_compile_time_arg_val("matmul_eh_subblock_k"),
        eh_out_w,
        get_named_compile_time_arg_val("matmul_eh_subblock_w"),
        get_named_compile_time_arg_val("matmul_eh_num_subblocks_k"),
        1,
        0,
        0>;

    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);
#endif

    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_input_core, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, true>
            mcast;
    deepseek_b1_ops::Mcast::Op<
        McastEhCTArgs,
        Core::enable_mtp && Core::is_input_core,
        Core::enable_mtp && Core::is_mcast_receiver_core,
        Core::enable_mtp && Core::is_mcast_receiver_core,
        true>
        mcast_eh;
    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
    deepseek_b1_ops::Sampling::
        Op<ArgmaxCTArgs, Core::is_matmul_core, Core::is_argmax_final_core, Core::is_argmax_mesh_sender_core>
            sampling_op;

    uint32_t iteration_count = 0;
    mcast.init(mcast_args);
    mcast_eh.init(mcast_eh_args);
    while (true) {
        iteration_count++;
        // ====================================================================
        // Phase 0: CCL Broadcast (multi-device only)
        // ====================================================================
        if constexpr (!Core::skip_ccl || Core::bcast_use_socket_input) {
#if defined(COMPILE_FOR_BRISC)
            constexpr bool is_sender = get_named_compile_time_arg_val("bcast_is_sender") == 1;
            if constexpr (Core::persistent_mode && is_sender && Core::is_input_core) {
                auto next_iteration_semaphore =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(persistent_next_iter_global_sem_addr);
                noc_semaphore_wait(next_iteration_semaphore, 1);
                noc_semaphore_set(next_iteration_semaphore, 0);
            }
#endif
            deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
            {
                DeviceZoneScopedN("CCL_BROADCAST");
                bcast(bcast_args);
            }
        }

#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_input_core && Core::persistent_mode) {
            constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
            constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
            if (iteration_count > 1) {
                unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
            }
        }
#endif

        // ====================================================================
        // Phase 0.5: First RMSNorm (TRISC only)
        // When MTP is enabled, don't pop the input so CB 0 data persists for h_rmsnorm reuse.
        // ====================================================================
        deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_rmsnorm_core, !Core::enable_mtp> rmsnorm;
        {
            DeviceZoneScopedN("RMSNORM");
            rmsnorm(rmsnorm_args);
        }

        // ====================================================================
        // Phase 1: Mcast — multicast input from sender core to all device cores
        // ====================================================================
        {
            DeviceZoneScopedN("MCAST");
            mcast(mcast_args);
        }

        // ====================================================================
        // Phase 2: Matmul — each matmul core computes local GEMM with its weight shard
        // ====================================================================
        {
            DeviceZoneScopedN("MATMUL");
            matmul(matmul_args);
        }

        // ====================================================================
        // [MTP] h_rmsnorm on TRISC — starts immediately after the LM head matmul,
        // overlapping with argmax on NCRISC/BRISC. CB 0 still has hidden states
        // (first RMSNorm used pop_input=false when MTP is enabled).
        // Output writes directly to mcast_eh_src_cb (first half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
        if constexpr (Core::enable_mtp && Core::is_rmsnorm_core) {
            deepseek_b1_ops::RMSNorm::Op<HRMSNormCTArgs, true, true> h_rmsnorm;
            {
                DeviceZoneScopedN("MTP_H_RMSNORM");
                h_rmsnorm(rmsnorm_args);
            }
        }
#endif

        // ====================================================================
        // Phase 3: Argmax Sampling
        // ====================================================================
        {
            DeviceZoneScopedN("ARGMAX");
            sampling_op(sampling_args);
        }

        // ====================================================================
        // [BYPASS] Fan-out: forward T_base to a downstream bypass stage via
        // a dedicated sender socket.  Runs on argmax_final_core only.
        // NCRISC copies the 4-byte token into a 64-byte staging buffer then
        // pushes one page through the bypass sender socket (local NOC write
        // to the bypass d2d_exchange relay core on the same device).
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::has_bypass_socket_output && Core::is_argmax_final_core) {
            DeviceZoneScopedN("BYPASS_SEND");
            auto staging = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bypass_staging_addr);
            staging[0] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sampling_args.output_addr);

            SocketSenderInterface bypass_socket = create_sender_socket_interface(bypass_sender_config_addr);
            set_sender_socket_page_size(bypass_socket, 64);
            socket_reserve_pages(bypass_socket, 1);
            for (uint32_t i = 0; i < bypass_socket.num_downstreams; i++) {
                sender_downstream_encoding enc = get_downstream_encoding(bypass_socket, i);
                noc_async_write(
                    bypass_staging_addr,
                    get_noc_addr(
                        enc.d2d.downstream_noc_x,
                        enc.d2d.downstream_noc_y,
                        bypass_socket.write_ptr + bypass_socket.downstream_fifo_addr),
                    64);
            }
            noc_async_write_barrier();
            socket_push_pages(bypass_socket, 1);
            socket_notify_receiver(bypass_socket);
            update_socket_config(bypass_socket);
        }
#endif

        // ====================================================================
        // [MTP] Token transfer + Embedding lookup + e_rmsnorm + EH matmul
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_matmul_core) {
            constexpr uint32_t matmul_out_cb = get_named_compile_time_arg_val("matmul_out");
            constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
            cb_pop_front(matmul_out_cb, out_w);
        }

        // ================================================================
        // [MTP] Token transfer: argmax_final_core writes token to input_core
        //
        // Runtime args (input_core_noc_x/y, mtp_token_addr, embedding_base)
        // were pre-consumed before the loop.
        // ================================================================
        if constexpr (Core::enable_mtp) {
            if constexpr (Core::is_argmax_final_core) {
                uint64_t dst = get_noc_addr(mtp_input_core_noc_x, mtp_input_core_noc_y, mtp_token_addr);
                noc_async_write(sampling_args.output_addr, dst, 4);
                noc_async_write_barrier();

                uint64_t sem_addr = get_noc_addr(
                    mtp_input_core_noc_x,
                    mtp_input_core_noc_y,
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_inc(sem_addr, 1);
            }

            if constexpr (Core::is_input_core) {
                volatile tt_l1_ptr uint32_t* mtp_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_wait(mtp_ready_sem, 1);
                noc_semaphore_set(mtp_ready_sem, 0);
            }
        }
#endif

        // ====================================================================
        // [MTP] Embedding lookup (NCRISC on input_core)
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_input_core) {
            constexpr uint32_t embedding_size_bytes = get_named_compile_time_arg_val("embedding_size_bytes");
            constexpr uint32_t emb_cb = get_named_compile_time_arg_val("embedding_cb");
            constexpr uint32_t e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");

            const InterleavedAddrGen<true> embedding_addr_gen = {
                .bank_base_address = mtp_embedding_base,
                .page_size = embedding_size_bytes,
            };

            uint32_t token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_token_addr);
            cb_reserve_back(emb_cb, e_num_tiles);
            uint64_t dram_addr = embedding_addr_gen.get_noc_addr(token_id);
            noc_async_read(dram_addr, get_write_ptr(emb_cb), embedding_size_bytes);
            noc_async_read_barrier();
            cb_push_back(emb_cb, e_num_tiles);
        }
#endif
        // ====================================================================
        // [MTP] Second mcast — multicast [h_norm|e_norm] from sender to all cores
        // ====================================================================
        {
            DeviceZoneScopedN("MTP_EH_MCAST");
            mcast_eh(mcast_eh_args);
        }

        // ====================================================================
        // [MTP] e_rmsnorm on TRISC (after embedding arrives in CB)
        // Output writes directly to mcast_eh_src_cb (second half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
        if constexpr (Core::enable_mtp && Core::is_rmsnorm_core) {
            deepseek_b1_ops::RMSNorm::Op<ERMSNormCTArgs, Core::is_rmsnorm_core, true> e_rmsnorm;
            {
                DeviceZoneScopedN("MTP_E_RMSNORM");
                e_rmsnorm(rmsnorm_args);
            }
        }
#endif

        // ====================================================================
        // [MTP] EH matmul using DRAM streaming
        // ====================================================================
#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_eh_matmul_core) {
            deepseek_b1_ops::DRAMStreamingMatmul::Op<EHDRAMMMCTArgs, true, true, false, eh_cb_in1_buf_addr> eh_matmul;
            {
                DeviceZoneScopedN("MTP_EH_DRAM_MATMUL");
                eh_matmul();
            }
        }
#endif

        // ====================================================================
        // [MTP Verification] Compare speculative token with reference token.
        //
        // After argmax produces T_spec, the argmax_final_core:
        // 1. Reads the reference token (T_base forwarded from base LM head stage)
        // 2. Stores T_spec in speculative_tokens_tensor for future verification
        // 3. Compares T_spec == T_base and writes result to verification_result_tensor
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp_verification) {
            // Runtime args (reference/verification/speculative addrs, bypass
            // recv config) were pre-consumed before the loop.

            if constexpr (Core::has_bypass_socket_input && Core::is_argmax_final_core) {
                DeviceZoneScopedN("BYPASS_RECV");
                SocketReceiverInterface bypass_socket = create_receiver_socket_interface(bypass_recv_config_addr);
                set_receiver_socket_page_size(bypass_socket, 64);
                while (!socket_wait_for_pages(bypass_socket, 1, 1000)) {
                }

                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_reference_token_addr) =
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bypass_socket.read_ptr);

                socket_pop_pages(bypass_socket, 1);
                socket_notify_sender(bypass_socket);
            }

            if constexpr (Core::is_argmax_final_core) {
                DeviceZoneScopedN("MTP_VERIFICATION");

                uint32_t speculative_token = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sampling_args.output_addr);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_speculative_token_addr) = speculative_token;

                uint32_t reference_token = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_reference_token_addr);

                uint32_t match = (speculative_token == reference_token) ? 1 : 0;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_verification_result_addr) = match;
            }
        }
#endif

        if constexpr (!Core::persistent_mode) {
            break;
        }
    }
    mcast.teardown();
    mcast_eh.teardown();
}
