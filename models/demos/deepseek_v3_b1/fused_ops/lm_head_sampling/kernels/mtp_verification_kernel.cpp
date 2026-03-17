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
//   CB 15 (mcast_eh_src):[MTP] [e_norm|h_norm] — embedding loaded here, e_rmsnorm in-place, h_rmsnorm appends
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
#include "api/debug/dprint.h"

// ============================================================================
// Core role flags (set per core group by UnifiedCompileTimeCoreDescriptor in op.py)
//
// Each flag is specialized at compile time. `if constexpr` eliminates dead
// code paths so each physical core only runs its assigned role.
// ============================================================================
struct Core {
    // ── Global configuration (same value on all cores) ──────────────
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;
    static constexpr bool persistent_mode = get_named_compile_time_arg_val("persistent_mode") == 1;
    static constexpr bool enable_argmax = get_named_compile_time_arg_val("enable_argmax") == 1;
    static constexpr uint32_t mesh_row = get_named_compile_time_arg_val("mesh_row");
    static constexpr uint32_t mesh_col = get_named_compile_time_arg_val("mesh_col");

    // ── Per-core role flags (1 on assigned cores, 0 elsewhere) ──────
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_rmsnorm_core = get_named_compile_time_arg_val("is_rmsnorm_core") == 1;
    static constexpr bool is_argmax_core = is_matmul_core;
    static constexpr bool is_argmax_final_core = get_named_compile_time_arg_val("is_argmax_final_core") == 1;
    static constexpr bool is_argmax_mesh_sender_core =
        get_named_compile_time_arg_val("is_argmax_mesh_sender_core") == 1;

    // ── Socket / IO configuration ───────────────────────────────────
    static constexpr uint32_t input_socket_mode = get_named_compile_time_arg_val("input_socket_mode");
    static constexpr uint32_t input_socket_mode_none = 0;
    static constexpr uint32_t input_socket_mode_d2d = 2;
    static constexpr bool bcast_use_socket_input = input_socket_mode == input_socket_mode_d2d;
    static_assert(input_socket_mode != 1, "lm_head_sampling input socket mode=1 is invalid");

    // ── MTP (Multi-Token Prediction) ────────────────────────────────
    static constexpr bool enable_mtp = get_named_compile_time_arg_val("enable_mtp") == 1;
    static constexpr bool enable_mtp_verification = get_named_compile_time_arg_val("enable_mtp_verification") == 1;
    static constexpr bool is_eh_matmul_core = enable_mtp && get_named_compile_time_arg_val("is_eh_matmul_core") == 1;
    static constexpr bool has_mtp_logits_socket_output =
        get_named_compile_time_arg_val("has_mtp_logits_socket_output") == 1;
    static constexpr bool has_token_input_socket = get_named_compile_time_arg_val("has_token_input_socket") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // ========================================================================
    // NCRISC — CCL broadcast writer, mcast receiver, argmax reader,
    //          MTP token transfer + embedding DRAM fetch, EH matmul reader
    //
    // Runtime args (consumed sequentially via ncrisc_rt_arg_idx++):
    //   ┌──────────────────────┬───────┬─────────────────────────────────────┐
    //   │ Section              │ Count │ Condition                           │
    //   ├──────────────────────┼───────┼─────────────────────────────────────┤
    //   │ CCL Broadcast writer │  13   │ !skip_ccl                           │
    //   │ Argmax reader        │   7   │ always                              │
    //   │ MTP token transfer   │   4   │ enable_mtp                          │
    //   │ MTP logits gather    │ 2+2N  │ enable_mtp                          │
    //   │ MTP logits socket    │   1   │ has_mtp_logits_socket_output        │
    //   │ Token input socket   │   1   │ has_token_input_socket              │
    //   │ Verification         │  11   │ enable_mtp_verification             │
    //   │ Fabric routing       │  var  │ !skip_ccl (per-core appended)       │
    //   └──────────────────────┴───────┴─────────────────────────────────────┘
    // ========================================================================
    uint32_t ncrisc_rt_arg_idx = 0;
    uint32_t mtp_token_addr = 0;

    // ── CCL Broadcast writer (all cores, !skip_ccl only) ────────────
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
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // num_connections
        };
    }

    // ── Mcast receiver (all cores) ──────────────────────────────────
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_semaphore(get_named_compile_time_arg_val("mcast_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // ── RMSNorm + Matmul (no-ops on NCRISC, compute on TRISC) ──────
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // ── Argmax reader (matmul cores) ────────────────────────────────
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
        get_named_compile_time_arg_val("argmax_gather_cb"),
        get_named_compile_time_arg_val("argmax_defer_socket_output")>;

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

    // ── MTP runtime args (all cores, enable_mtp) ────────────────────
    uint32_t mtp_input_core_noc_x = 0;
    uint32_t mtp_input_core_noc_y = 0;
    uint32_t mtp_embedding_base = 0;
    if constexpr (Core::enable_mtp) {
        mtp_input_core_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_input_core_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_embedding_base = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    // ── MTP logits gather runtime args (enable_mtp) ─────────────────
    // Gather addresses (staging=CB1, shard base, core NOC coords) are
    // emitted whenever enable_mtp; the socket config addr is only
    // present when has_mtp_logits_socket_output.
    uint32_t mtp_logits_socket_config_addr = 0;
    uint32_t mtp_logits_staging_addr = 0;
    uint32_t mtp_logits_eh_shard_base_addr = 0;
    uint32_t mtp_logits_eh_core_noc_x[8] = {};
    uint32_t mtp_logits_eh_core_noc_y[8] = {};
    if constexpr (Core::enable_mtp) {
        mtp_logits_staging_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_logits_eh_shard_base_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        constexpr uint32_t num_eh = get_named_compile_time_arg_val("mtp_logits_num_eh_cores");
        static_assert(num_eh <= 8, "mtp_logits_num_eh_cores exceeds max array size");
        for (uint32_t i = 0; i < num_eh; i++) {
            mtp_logits_eh_core_noc_x[i] = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        }
        for (uint32_t i = 0; i < num_eh; i++) {
            mtp_logits_eh_core_noc_y[i] = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        }
        if constexpr (Core::has_mtp_logits_socket_output) {
            mtp_logits_socket_config_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        }
    }

    // ── Token input socket (receives T_base from upstream via D2D) ───
    uint32_t token_input_socket_config_addr = 0;
    if constexpr (Core::has_token_input_socket) {
        token_input_socket_config_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    // ── Verification runtime args (enable_mtp_verification) ─────────
    // State tensors are on input_core; output staging is on argmax_final_core.
    uint32_t mtp_reference_token_addr = 0;
    uint32_t mtp_verification_result_addr = 0;
    uint32_t mtp_speculative_token_addr = 0;
    uint32_t mtp_unverified_spec_addr = 0;
    uint32_t mtp_verified_spec_addr = 0;
    uint32_t verify_output_staging_addr = 0;
    uint32_t verify_argmax_output_addr = 0;
    uint32_t verify_argmax_core_noc_x = 0;
    uint32_t verify_argmax_core_noc_y = 0;
    uint32_t verify_input_core_noc_x = 0;
    uint32_t verify_input_core_noc_y = 0;
    if constexpr (Core::enable_mtp_verification) {
        mtp_reference_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_verification_result_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_speculative_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_unverified_spec_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_verified_spec_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_output_staging_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_argmax_output_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_argmax_core_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_argmax_core_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_input_core_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_input_core_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    // ── Sharded buffer setup (registers tensor-backed CBs before main loop) ──
    //   input_core:  CB 0 (rmsnorm_input), CB 7 (rmsnorm_gamma)
    //                CB 11 (h_gamma), CB 12 (e_gamma)  [MTP only]
    //   matmul_core: CB 2 (matmul_in1 / vocab weights)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        if constexpr (!(Core::skip_ccl && Core::bcast_use_socket_input)) {
            unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        }
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
    }
    if constexpr (Core::enable_mtp && Core::is_input_core) {
        constexpr uint32_t h_gamma_cb = get_named_compile_time_arg_val("h_gamma_cb");
        constexpr uint32_t rmsnorm_h_num_tiles = get_named_compile_time_arg_val("rmsnorm_h_num_tiles");
        unified_kernels::setup_sharded_buffer(h_gamma_cb, rmsnorm_h_num_tiles);
        constexpr uint32_t e_gamma_cb = get_named_compile_time_arg_val("e_gamma_cb");
        constexpr uint32_t rmsnorm_e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");
        unified_kernels::setup_sharded_buffer(e_gamma_cb, rmsnorm_e_num_tiles);
    }
    if constexpr (Core::is_matmul_core) {
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");
        unified_kernels::setup_sharded_buffer(in1_cb, num_tiles_k * out_w);
    }

    // ── MTP: EH mcast receiver (all cores, enable_mtp) ─────────────
    using McastEhCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_eh_args{
        get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast_eh_dst_cb"),
        get_named_compile_time_arg_val("mcast_eh_dst_num_pages"),
    };

    // ── MTP: EH DRAM streaming matmul reader (eh_matmul_core) ───────
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
    // ========================================================================
    // BRISC — CCL broadcast reader, mcast sender, argmax writer,
    //         MTP EH mcast sender, persistent signal routing
    //
    // Runtime args (fixed-index access from op.py brisc_bcast_common_args):
    //   ┌─────────────────────┬─────────┬────────────────────────────────────┐
    //   │ Section             │ Indices │ Notes                              │
    //   ├─────────────────────┼─────────┼────────────────────────────────────┤
    //   │ Argmax writer       │  [0..3] │ final_noc_x/y, scratch, socket    │
    //   │ Socket input reader │  [4..6] │ config_addr, page_size, num_pages │
    //   │ Persistent routing  │ [7..12] │ enable, dst noc/mesh/chip, sem    │
    //   │ Mcast dst override  │   [13]  │ tensor-backed CB 1 addr (0=CB)    │
    //   │ MTP mcast override  │   [14]  │ tensor-backed CB 18 addr (0=CB)   │
    //   │ Fabric routing      │  [15+]  │ per-core appended                 │
    //   └─────────────────────┴─────────┴────────────────────────────────────┘
    // ========================================================================
    uint32_t brisc_rt_arg_idx = 0;

    // ── CCL Broadcast reader (input_core only) ──────────────────────
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender"),
        (get_named_compile_time_arg_val("input_socket_mode") == 2 ? 1 : 0)>;

    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{
        get_common_arg_val<uint32_t>(4),  // socket_config_addr
        get_common_arg_val<uint32_t>(5),  // socket_page_size
        get_common_arg_val<uint32_t>(6),  // socket_num_pages
    };

    // ── RMSNorm + Matmul (no-ops on BRISC) ──────────────────────────
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    // ── Mcast sender (input_core) ───────────────────────────────────
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;

    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    const uint32_t mcast_dst_addr_override = get_common_arg_val<uint32_t>(13);
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
        mcast_dst_addr_override != 0 ? mcast_dst_addr_override : get_write_ptr(mcast_dst_cb),
    };

    // ── Argmax writer (matmul cores, argmax_final_core) ─────────────
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

    // ── MTP: EH mcast sender (input_core, enable_mtp) ──────────────
    using McastEhCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_eh_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;
    constexpr uint32_t mcast_eh_src_cb = get_named_compile_time_arg_val("mcast_eh_src_cb");
    constexpr uint32_t mcast_eh_dst_cb = get_named_compile_time_arg_val("mcast_eh_dst_cb");
    const uint32_t mcast_eh_dst_addr_override = get_common_arg_val<uint32_t>(14);
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
        mcast_eh_dst_addr_override != 0 ? mcast_eh_dst_addr_override : get_write_ptr(mcast_eh_dst_cb),
    };

#elif defined(COMPILE_FOR_TRISC)
    // ========================================================================
    // TRISC — RMSNorm compute, matmul compute, MTP h/e RMSNorm, EH matmul
    //
    // Runtime args (fixed-index from op.py trisc_common_runtime_args):
    //   [0] epsilon          (uint32 packed float)
    //   [1] scalar           (float, 1/sqrt(numel))
    //   [2] MTP e_scalar     (float, 1/sqrt(embedding_dim), 0 if MTP disabled)
    //
    // No-ops on TRISC: CCL broadcast, mcast, argmax (data movement only)
    // ========================================================================

    // ── CCL Broadcast / Mcast / Argmax (no-ops on TRISC) ────────────
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
    deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};
    using McastEhCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_eh_args{};
    using ArgmaxCTArgs = deepseek_b1_ops::Sampling::ComputeCTArgs;
    deepseek_b1_ops::Sampling::ComputeArgs sampling_args{};

    // ── RMSNorm compute (input_core / rmsnorm_core) ─────────────────
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

    // ── MTP: h_rmsnorm + e_rmsnorm compute (input_core, enable_mtp) ─
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

    // ── Matmul compute (matmul_core) ────────────────────────────────
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w")>;

    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("matmul_in0");
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("matmul_in1");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("matmul_out");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("matmul_k_num_tiles");

    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        .in0 = in0_cb,
        .in1 = in1_cb,
        .out = out_cb,
        .k_num_tiles = num_tiles_k,
    };

    // ── MTP: EH DRAM streaming matmul compute (eh_matmul_core) ──────
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
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " START" << ENDL();
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " START" << ENDL();
        }
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
            // Both BRISC and NCRISC must wait for the mtp_done semaphore before
            // entering the broadcast.  The argmax_final_core increments by 2
            // (one credit per processor); each processor atomically decrements.
            // This prevents BRISC from racing into the broadcast with stale CB
            // data while NCRISC is still blocked on the upstream socket read.
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::enable_mtp && Core::is_input_core && Core::persistent_mode) {
                if (iteration_count > 1) {
                    constexpr uint32_t mtp_done_sem_id = get_named_compile_time_arg_val("mtp_done_semaphore_id");
                    volatile tt_l1_ptr uint32_t* mtp_done_sem =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(mtp_done_sem_id));
                    DPRINT << "LMH iter=" << iteration_count << " WAIT_MTP_DONE" << ENDL();
                    while (__atomic_load_n(mtp_done_sem, __ATOMIC_RELAXED) < 1) {
                    }
                    unified_kernels::semaphore_dec(mtp_done_sem);
                    DPRINT << "LMH iter=" << iteration_count << " WAIT_MTP_DONE_OK" << ENDL();
                }
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
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " P0.5_RMSNORM" << ENDL();
        }
        deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_rmsnorm_core, !Core::enable_mtp> rmsnorm;
        {
            DeviceZoneScopedN("RMSNORM");
            rmsnorm(rmsnorm_args);
        }

        // ====================================================================
        // Phase 1: Mcast — multicast input from sender core to all device cores
        // ====================================================================
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " P1_MCAST" << ENDL();
        }
        {
            DeviceZoneScopedN("MCAST");
            mcast(mcast_args);
        }

        // ====================================================================
        // Phase 2: Matmul — each matmul core computes local GEMM with its weight shard
        // ====================================================================
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " P2_MATMUL" << ENDL();
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " P2_MATMUL" << ENDL();
        }
        {
            DeviceZoneScopedN("MATMUL");
            matmul(matmul_args);
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " P2_MATMUL_DONE" << ENDL();
        }

        // ====================================================================
        // [MTP] h_rmsnorm on TRISC — starts immediately after the LM head matmul,
        // overlapping with argmax on NCRISC/BRISC. CB 0 still has hidden states
        // (first RMSNorm used pop_input=false when MTP is enabled).
        // Output writes directly to mcast_eh_src_cb (first half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
        if constexpr (Core::enable_mtp && Core::is_rmsnorm_core) {
            if constexpr (Core::is_input_core) {
                DPRINT << "LMH iter=" << iteration_count << " MTP_H_RMSNORM" << ENDL();
            }
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
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " P3_ARGMAX" << ENDL();
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " P3_ARGMAX" << ENDL();
        }
        {
            DeviceZoneScopedN("ARGMAX");
            sampling_op(sampling_args);
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " P3_ARGMAX_DONE" << ENDL();
        }

        // ====================================================================
        // [MTP] Token transfer + Embedding lookup + e_rmsnorm + EH matmul
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        // CB 16 (matmul_out) is popped internally by argmax.hpp (scores_cb pop).
        // Do NOT pop it again here — double pop underflows the page counter and
        // causes a hang after ~4 persistent iterations.

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

            // Signal TRISC that embedding data is ready in the CB.
            // Required because emb_cb shares the same CB index as the rmsnorm output:
            // without this gate, TRISC e_rmsnorm can race ahead and consume the stale
            // rmsnorm tiles before BRISC mcast pops them and NCRISC pushes embedding data.
            constexpr uint32_t emb_done_cb = get_named_compile_time_arg_val("mtp_embedding_done_cb");
            cb_reserve_back(emb_done_cb, 1);
            cb_push_back(emb_done_cb, 1);
        }
#endif
        // ====================================================================
        // [MTP] Second mcast — multicast [h_norm|e_norm] from sender to all cores
        // ====================================================================
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " MTP_EH_MCAST" << ENDL();
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " MTP_EH_MCAST" << ENDL();
        }
        {
            DeviceZoneScopedN("MTP_EH_MCAST");
            mcast_eh(mcast_eh_args);
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " MTP_EH_MCAST_DONE" << ENDL();
        }

        // ====================================================================
        // [MTP] e_rmsnorm on TRISC (after embedding arrives in CB)
        // Output writes directly to mcast_eh_src_cb (second half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " MTP_E_RMSNORM" << ENDL();
        }
        if constexpr (Core::enable_mtp && Core::is_rmsnorm_core) {
            // Wait for NCRISC embedding push before reading CB — the embedding CB is
            // reused from the rmsnorm output CB, so without this gate TRISC can race
            // ahead and consume the stale rmsnorm tiles instead of the embedding tiles.
            constexpr uint32_t emb_done_cb = get_named_compile_time_arg_val("mtp_embedding_done_cb");
            cb_wait_front(emb_done_cb, 1);
            cb_pop_front(emb_done_cb, 1);

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
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " MTP_EH_MATMUL" << ENDL();
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " MTP_EH_MATMUL" << ENDL();
        }
        if constexpr (Core::enable_mtp && Core::is_eh_matmul_core) {
            deepseek_b1_ops::DRAMStreamingMatmul::
                Op<EHDRAMMMCTArgs, true, true, false, eh_cb_in1_buf_addr, false, false, 2>
                    eh_matmul;
            {
                DeviceZoneScopedN("MTP_EH_DRAM_MATMUL");
                eh_matmul();
            }
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " MTP_EH_MATMUL_DONE" << ENDL();
        }
#endif

#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_eh_matmul_core) {
            cb_pop_front(eh_out_cb, eh_out_w);
        }
        if constexpr (Core::enable_mtp && Core::is_mcast_receiver_core && !Core::is_eh_matmul_core) {
            constexpr uint32_t mcast_eh_dst_cb_drain = get_named_compile_time_arg_val("mcast_eh_dst_cb");
            constexpr uint32_t mcast_eh_dst_pages_drain = get_named_compile_time_arg_val("mcast_eh_dst_num_pages");
            cb_wait_front(mcast_eh_dst_cb_drain, mcast_eh_dst_pages_drain);
            cb_pop_front(mcast_eh_dst_cb_drain, mcast_eh_dst_pages_drain);
        }

        // ====================================================================
        // [MTP Logits Gather] Read EH matmul shards from all DRAM-bank
        // cores into CB 1 backing memory (mcast_dst, freed after Phase 2).
        // Always runs when enable_mtp, even without socket output.
        // ====================================================================
        if constexpr (Core::enable_mtp && Core::is_argmax_final_core) {
            DeviceZoneScopedN("MTP_LOGITS_GATHER");
            constexpr uint32_t num_eh = get_named_compile_time_arg_val("mtp_logits_num_eh_cores");
            constexpr uint32_t eh_shard_size = get_named_compile_time_arg_val("mtp_logits_eh_shard_size");

            for (uint32_t i = 0; i < num_eh; i++) {
                uint64_t src_addr = get_noc_addr(
                    mtp_logits_eh_core_noc_x[i], mtp_logits_eh_core_noc_y[i], mtp_logits_eh_shard_base_addr);
                noc_async_read(src_addr, mtp_logits_staging_addr + i * eh_shard_size, eh_shard_size);
            }
            noc_async_read_barrier();
        }

        // ====================================================================
        // [MTP Logits Socket Send] Forward gathered logits via D2D socket.
        // Only when has_mtp_logits_socket_output; data already in staging.
        // ====================================================================
        if constexpr (Core::has_mtp_logits_socket_output && Core::is_argmax_final_core) {
            DeviceZoneScopedN("MTP_LOGITS_SOCKET_SEND");
            constexpr uint32_t num_eh = get_named_compile_time_arg_val("mtp_logits_num_eh_cores");
            constexpr uint32_t eh_shard_size = get_named_compile_time_arg_val("mtp_logits_eh_shard_size");
            constexpr uint32_t mtp_logits_page_size = num_eh * eh_shard_size;

            SocketSenderInterface logits_socket = create_sender_socket_interface(mtp_logits_socket_config_addr);
            set_sender_socket_page_size(logits_socket, mtp_logits_page_size);
            socket_reserve_pages(logits_socket, 1);
            for (uint32_t i = 0; i < logits_socket.num_downstreams; i++) {
                sender_downstream_encoding enc = get_downstream_encoding(logits_socket, i);
                noc_async_write(
                    mtp_logits_staging_addr,
                    get_noc_addr(
                        enc.d2d.downstream_noc_x,
                        enc.d2d.downstream_noc_y,
                        logits_socket.write_ptr + logits_socket.downstream_fifo_addr),
                    mtp_logits_page_size);
            }
            noc_async_write_barrier();
            socket_push_pages(logits_socket, 1);
            socket_notify_receiver(logits_socket);
            update_socket_config(logits_socket);
        }
#endif

        // ====================================================================
        // [Token Input] Receive T_base reference token from upstream via D2D
        // socket.  The token page (64 bytes) lands in the socket FIFO; we
        // copy the first 3 uint32s (token_index, token_type, token_pos) into
        // mtp_reference_token_addr so the verification block can read them.
        // Placed after argmax (which produces this_token) but before the
        // verification comparison that needs both this_token and T_base.
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::has_token_input_socket) {
            if constexpr (Core::is_input_core) {
                DeviceZoneScopedN("TOKEN_INPUT_RECV");
                SocketReceiverInterface recv = create_receiver_socket_interface(token_input_socket_config_addr);
                set_receiver_socket_page_size(recv, get_named_compile_time_arg_val("token_input_page_size"));
                socket_wait_for_pages(recv, 1);

                auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv.read_ptr);
                auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_reference_token_addr);
                dst[0] = src[0];  // token_index
                dst[1] = src[1];  // token_type
                dst[2] = src[2];  // token_pos

                socket_pop_pages(recv, 1);
                socket_notify_sender(recv);
                update_socket_config(recv);
            }
        }
#endif

        // ====================================================================
        // [MTP Verification] Split across two cores:
        //   input_core (10,9):        state + logic + skip decision
        //   argmax_final_core (0,0):  this_token + socket CB push
        //
        // Flow:
        //   1. argmax_final_core signals input_core that this_token is ready
        //   2. input_core NOC-reads this_token from argmax_final_core
        //   3. input_core runs accept/reject logic (all state is local)
        //   4. input_core NOC-writes output page to staging buffer on
        //      argmax_final_core
        //   5. input_core signals argmax_final_core that output is ready
        //   6. argmax_final_core pushes staging data to socket CB
        //
        // Output page layout (7 uint32s):
        //   [0] primary token index   [1] primary token type
        //   [2] primary token pos     [3] num_tokens (0/1/2)
        //   [4] secondary token index [5] secondary token type
        //   [6] secondary token pos
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp_verification) {
            // ── argmax_final_core: signal input_core that this_token is ready ──
            if constexpr (Core::is_argmax_final_core) {
                constexpr uint32_t verify_argmax_done_sem_id =
                    get_named_compile_time_arg_val("verify_argmax_done_semaphore_id");
                uint64_t input_core_sem_addr = get_noc_addr(
                    verify_input_core_noc_x, verify_input_core_noc_y, get_semaphore(verify_argmax_done_sem_id));
                noc_semaphore_inc(input_core_sem_addr, 1);
            }

            // ── input_core: full verification logic ───────────────────────
            if constexpr (Core::is_input_core) {
                DeviceZoneScopedN("MTP_VERIFICATION");

                // Wait for argmax to finish producing this_token.
                constexpr uint32_t verify_argmax_done_sem_id =
                    get_named_compile_time_arg_val("verify_argmax_done_semaphore_id");
                volatile tt_l1_ptr uint32_t* argmax_done_sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(verify_argmax_done_sem_id));
                noc_semaphore_wait(argmax_done_sem, 1);
                noc_semaphore_set(argmax_done_sem, 0);

                // NOC-read this_token from argmax_final_core's output_addr
                // into the local speculative token tensor (already in L1).
                uint64_t argmax_output_noc =
                    get_noc_addr(verify_argmax_core_noc_x, verify_argmax_core_noc_y, verify_argmax_output_addr);
                noc_async_read(argmax_output_noc, mtp_speculative_token_addr, sizeof(uint32_t));
                noc_async_read_barrier();

                uint32_t this_token = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_speculative_token_addr);

                constexpr uint32_t MAX_SPEC_DEPTH = get_named_compile_time_arg_val("mtp_max_spec_depth");
                constexpr uint32_t SENTINEL = 0xFFFFFFFF;
                constexpr uint32_t TOKEN_TYPE_BASE = 0;
                constexpr uint32_t TOKEN_TYPE_SPEC = 1;

                auto ref_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_reference_token_addr);
                uint32_t reference_token = ref_ptr[0];
                uint32_t ref_token_type = ref_ptr[1];
                uint32_t ref_token_pos = ref_ptr[2];

                auto unverified = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_unverified_spec_addr);
                auto verified = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_verified_spec_addr);

                uint32_t match = 0;
                uint32_t num_tokens = 0;
                uint32_t tok0_index = 0, tok0_type = 0, tok0_pos = 0;
                uint32_t tok1_index = 0, tok1_type = 0, tok1_pos = 0;

                if (ref_token_type == TOKEN_TYPE_BASE) {
                    uint32_t verified_pos = (ref_token_pos + 1) % MAX_SPEC_DEPTH;
                    uint32_t expected_spec = unverified[verified_pos];

                    if (expected_spec != SENTINEL && reference_token == expected_spec) {
                        // ── ACCEPT ──────────────────────────────────────
                        verified[verified_pos] = expected_spec;
                        unverified[verified_pos] = SENTINEL;
                        match = 1;

                        uint32_t new_spec_pos = (verified_pos + 1) % MAX_SPEC_DEPTH;
                        unverified[new_spec_pos] = this_token;

                        num_tokens = 1;
                        tok0_index = this_token;
                        tok0_type = TOKEN_TYPE_SPEC;
                        tok0_pos = verified_pos + 1;
                    } else {
                        // ── REJECT ──────────────────────────────────────
                        uint32_t next_spec_pos = (ref_token_pos + 1) % MAX_SPEC_DEPTH;
                        unverified[next_spec_pos] = this_token;
                        match = 0;

                        num_tokens = 2;
                        tok0_index = reference_token;
                        tok0_type = TOKEN_TYPE_BASE;
                        tok0_pos = ref_token_pos;
                        tok1_index = this_token;
                        tok1_type = TOKEN_TYPE_SPEC;
                        tok1_pos = ref_token_pos + 1;
                    }
                } else {
                    // ── SPEC token ──────────────────────────────────
                    uint32_t spec_pos = ref_token_pos % MAX_SPEC_DEPTH;
                    if (verified[spec_pos] != SENTINEL) {
                        match = 1;
                        verified[spec_pos] = SENTINEL;

                        uint32_t new_spec_pos = (ref_token_pos + 1) % MAX_SPEC_DEPTH;
                        unverified[new_spec_pos] = this_token;

                        num_tokens = 2;
                        tok0_index = this_token;
                        tok0_type = TOKEN_TYPE_BASE;
                        tok0_pos = ref_token_pos + 1;
                        tok1_index = this_token;
                        tok1_type = TOKEN_TYPE_SPEC;
                        tok1_pos = ref_token_pos + 2;
                    } else {
                        // ── STALE ───────────────────────────────────
                        num_tokens = 0;
                    }
                }

                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_verification_result_addr) = match;

                // NOC-write the 7-uint32 output page to the staging
                // buffer on argmax_final_core, then signal it.
                // Assemble locally in the reference_token tensor (3 uint32s
                // no longer needed after verification) extended into the
                // verification_result area. Since these may not be contiguous,
                // use a dedicated L1 local buffer.
                if constexpr (ArgmaxCTArgs::defer_socket_output && ArgmaxCTArgs::socket_mode != 0) {
                    // Use reference_token_addr as 8-uint32 scratch (it's at
                    // least 12 bytes = 3 uint32s; we allocated (1,3) in
                    // stage.py but L1 alignment gives us 32 bytes).
                    auto staging = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_reference_token_addr);
                    staging[0] = tok0_index;
                    staging[1] = tok0_type;
                    staging[2] = tok0_pos;
                    staging[3] = num_tokens;
                    staging[4] = tok1_index;
                    staging[5] = tok1_type;
                    staging[6] = tok1_pos;

                    uint64_t staging_noc =
                        get_noc_addr(verify_argmax_core_noc_x, verify_argmax_core_noc_y, verify_output_staging_addr);
                    noc_async_write(mtp_reference_token_addr, staging_noc, 28);
                    noc_async_write_barrier();

                    constexpr uint32_t verify_output_ready_sem_id =
                        get_named_compile_time_arg_val("verify_output_ready_semaphore_id");
                    uint64_t argmax_sem_addr = get_noc_addr(
                        verify_argmax_core_noc_x, verify_argmax_core_noc_y, get_semaphore(verify_output_ready_sem_id));
                    noc_semaphore_inc(argmax_sem_addr, 1);
                }
            }

            // ── argmax_final_core: receive output and push to socket CB ──
            if constexpr (Core::is_argmax_final_core) {
                if constexpr (ArgmaxCTArgs::defer_socket_output && ArgmaxCTArgs::socket_mode != 0) {
                    constexpr uint32_t verify_output_ready_sem_id =
                        get_named_compile_time_arg_val("verify_output_ready_semaphore_id");
                    volatile tt_l1_ptr uint32_t* output_ready_sem =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(verify_output_ready_sem_id));
                    noc_semaphore_wait(output_ready_sem, 1);
                    noc_semaphore_set(output_ready_sem, 0);

                    cb_reserve_back(ArgmaxCTArgs::socket_cb_id, 1);
                    auto out_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(ArgmaxCTArgs::socket_cb_id));
                    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(verify_output_staging_addr);
                    for (uint32_t i = 0; i < 7; i++) {
                        out_ptr[i] = src[i];
                    }
                    cb_push_back(ArgmaxCTArgs::socket_cb_id, 1);
                }
            }
        }
#endif

        // ====================================================================
        // [MTP] Signal input_core that all MTP phases are complete.
        // The argmax's persistent_next_iter_semaphore fires during argmax
        // (before MTP), so the input_core can race ahead.  This second gate
        // ensures NCRISC on the input_core blocks until the argmax_final_core
        // finishes CB pops, EH matmul, and verification for this iteration.
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::enable_mtp && Core::is_argmax_final_core && Core::persistent_mode) {
            constexpr uint32_t mtp_done_sem_id = get_named_compile_time_arg_val("mtp_done_semaphore_id");
            uint64_t mtp_done_dst =
                get_noc_addr(mtp_input_core_noc_x, mtp_input_core_noc_y, get_semaphore(mtp_done_sem_id));
            noc_semaphore_inc(mtp_done_dst, 2);
            DPRINT << "MC iter=" << iteration_count << " MTP_DONE_SIG" << ENDL();
        }
#endif
        if constexpr (Core::is_input_core) {
            DPRINT << "LMH iter=" << iteration_count << " DONE" << ENDL();
        }
        if constexpr (Core::is_argmax_final_core) {
            DPRINT << "MC iter=" << iteration_count << " DONE" << ENDL();
        }
        if constexpr (!Core::persistent_mode) {
            break;
        }
    }
    mcast.teardown();
    mcast_eh.teardown();
}
