// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
//   CB 1  (mcast_dst):   Mcast destination / matmul in0 on all cores (intermediate).
//                         [MTP] Also reused for EH mcast destination and gather destination
//                         (sequential use — sized to max(num_tiles_k, eh_num_tiles_k)).
//   CB 2  (matmul_in1):  Vocab weights on matmul cores (tensor-backed)
//   CB 9  (matmul_eh):   [MTP] EH projection weights on matmul cores (tensor-backed)
//   CB 10 (embedding):   [MTP] Embedding row for e_rmsnorm input
//   CB 11 (h_gamma):     [MTP] RMSNorm gamma for hidden states (tensor-backed)
//   CB 12 (e_gamma):     [MTP] RMSNorm gamma for embeddings (tensor-backed)
//   CB 15 (mcast_eh_src):[MTP] [e_norm|h_norm] — embedding loaded here, e_rmsnorm in-place, h_rmsnorm appends
//   CB 16 (matmul_out):  Matmul output on matmul cores (tensor-backed)
//   CB 17 (matmul_eh_out):[MTP] EH matmul output (tensor-backed)
//   CB 30 (bcast_pkt):   CCL broadcast packet buffer (multi-device mode only)
//
// [MTP] CBs that intersect the output-token path and can cause corruption when enable_mtp:
//   - CB 0 (rmsnorm_input_cb): Backing L1 = mtp_token_addr. When MTP, primary rmsnorm does
//     not pop; h_rmsnorm then reads and pops. Argmax writes token to same address. Next
//     iteration broadcast overwrites it. Stale push here (e.g. setup_sharded_buffer on
//     sender) → wrong rmsnorm input → wrong tokens.
//   - CB 8 (mcast_src_cb = embedding_cb): Reused as (1) rmsnorm output → mcast source,
//     then (2) embedding_cb for e_rmsnorm. Order: mcast pops → NCRISC pushes embedding
//     → e_rmsnorm reads. Wrong token_id (e.g. token write commented out) → wrong
//     embedding in CB 8 does not affect next iteration primary rmsnorm (that reads CB 0),
//     but breaks e_rmsnorm/EH path and can leave bad state.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/broadcast.hpp"
// NOTE: rmsnorm.hpp must come before sampling.hpp because sampling.hpp transitively
// includes api/compute/reduce.h, whose template default arguments reference the
// REDUCE_OP / REDUCE_DIM macros that rmsnorm.hpp defines. Reordering avoids
// "REDUCE_OP was not declared in this scope" errors at TRISC compile time.
#include "../../../unified_kernels/rmsnorm.hpp"
#include "../../../unified_kernels/sampling.hpp"
#include "../../../unified_kernels/socket_send.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"
#include "../../../unified_kernels/reduce_to_one_b1.hpp"
#include "../../../unified_kernels/persistent_loop.hpp"
#include "../../../metadata/metadata.hpp"
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

    // ── Per-core role flags (1 on assigned cores, 0 elsewhere) ──────
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_mcast_receiver_core = get_named_compile_time_arg_val("is_mcast_receiver_core") == 1;
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_rmsnorm_core = get_named_compile_time_arg_val("is_rmsnorm_core") == 1;
    // Sampling per-core role flags (previously argmax_*; renamed to match
    // sampling.hpp + micro_ops/sampling/op.py naming).
    static constexpr bool sampling_is_active_core = get_named_compile_time_arg_val("sampling_is_active_core") == 1;
    static constexpr bool sampling_is_final_core = get_named_compile_time_arg_val("sampling_is_final_core") == 1;
    static constexpr bool sampling_mesh_sender_core = get_named_compile_time_arg_val("sampling_mesh_sender_core") == 1;
    static constexpr uint32_t fabric_gate_bcast_turn_semaphore_id =
        get_named_compile_time_arg_val("fabric_gate_bcast_turn_semaphore_id");
    static constexpr uint32_t fabric_gate_argmax_turn_semaphore_id =
        get_named_compile_time_arg_val("fabric_gate_argmax_turn_semaphore_id");
    static constexpr uint32_t fabric_gate_bcast_noc_x = get_named_compile_time_arg_val("fabric_gate_bcast_noc_x");
    static constexpr uint32_t fabric_gate_bcast_noc_y = get_named_compile_time_arg_val("fabric_gate_bcast_noc_y");
    static constexpr uint32_t fabric_gate_argmax_noc_x = get_named_compile_time_arg_val("fabric_gate_argmax_noc_x");
    static constexpr uint32_t fabric_gate_argmax_noc_y = get_named_compile_time_arg_val("fabric_gate_argmax_noc_y");
    static constexpr uint32_t mesh_row = get_named_compile_time_arg_val("mesh_row");
    static constexpr uint32_t mesh_col = get_named_compile_time_arg_val("mesh_col");

    // ── Socket / IO configuration ───────────────────────────────────
    static constexpr uint32_t input_socket_mode = get_named_compile_time_arg_val("input_socket_mode");
    static constexpr uint32_t input_socket_mode_none = 0;
    static constexpr uint32_t input_socket_mode_d2d = 2;
    static constexpr bool bcast_use_socket_input = input_socket_mode == input_socket_mode_d2d;
    static_assert(input_socket_mode != 1, "lm_head_sampling input socket mode=1 is invalid");

    // ── RMSNorm folding (gamma pre-multiplied into weight matrices) ─
    static constexpr bool fold_rmsnorm = get_named_compile_time_arg_val("fold_rmsnorm") == 1;

    // ── MTP (Multi-Token Prediction) ────────────────────────────────
    static constexpr bool enable_mtp = get_named_compile_time_arg_val("enable_mtp") == 1;
    static constexpr bool is_base_stage = get_named_compile_time_arg_val("is_mtp_base_stage") == 1;
    static constexpr bool is_spec_stage = get_named_compile_time_arg_val("is_mtp_verify_stage") == 1;
    static constexpr bool is_eh_matmul_core = enable_mtp && get_named_compile_time_arg_val("is_eh_matmul_core") == 1;
    static constexpr bool is_eh_reduce_worker_core =
        enable_mtp && get_named_compile_time_arg_val("is_reduce_worker_core") == 1;
    static constexpr bool is_eh_reduce_fabric_core =
        enable_mtp && get_named_compile_time_arg_val("is_reduce_fabric_core") == 1;
    static constexpr bool is_e_norm_device = enable_mtp && get_named_compile_time_arg_val("is_e_norm_device") == 1;
    static constexpr uint32_t reduce_gate_semaphore_id = get_named_compile_time_arg_val("reduce_gate_semaphore_id");

    // ── Verify stage metadata transfer ───────────────────────────────
    static constexpr bool is_exit_device = get_named_compile_time_arg_val("is_exit_device") == 1;
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
    //   │ MTP / verify addrs   │   —   │ named compile-time args             │
    //   │ Fabric routing       │  var  │ !skip_ccl (per-core appended)       │
    //   └──────────────────────┴───────┴─────────────────────────────────────┘
    // ========================================================================
    uint32_t ncrisc_rt_arg_idx = 0;
    uint32_t per_core_rta_arg_idx = 0;
    // --- NCRISC: CCL broadcast writer + mcast receiver + sharded buffer setup ---

    // CCL Broadcast CTArgs type alias
#if !defined(SKIP_CCL)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("bcast_data_cb_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("bcast_num_neighbors"),
        get_named_compile_time_arg_val("bcast_num_links"),
        get_named_compile_time_arg_val("bcast_is_root"),
        get_named_compile_time_arg_val("bcast_chunk_size_bytes"),
        get_named_compile_time_arg_val("bcast_last_chunk_size_bytes"),
        get_named_compile_time_arg_val("bcast_num_chunks")>;
#endif

    deepseek_b1_ops::Broadcast::WriterArgs bcast_args{};
    constexpr uint32_t bcast_writer_common_rt_count = 5;
#if !defined(SKIP_CCL)
    if constexpr (!Core::skip_ccl) {
        uint32_t bcast_rta_num_args = 0;
        uint32_t bcast_rta_offset = 0;
        if constexpr (Core::is_input_core) {
            bcast_rta_num_args = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            bcast_rta_offset = per_core_rta_arg_idx;
            per_core_rta_arg_idx += bcast_rta_num_args;
        }
        bcast_args = deepseek_b1_ops::Broadcast::WriterArgs{
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // tensor_address0
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // my_noc_x
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // my_noc_y
            {
                get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // sem_bank_addrs[0]
                get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // sem_bank_addrs[1] (dummy when num_links=1)
            },
            bcast_rta_offset,
            bcast_rta_num_args,
        };
    }
#endif
    if constexpr (Core::skip_ccl) {
        ncrisc_rt_arg_idx = bcast_writer_common_rt_count;
    }

    // ── Mcast receiver (all cores) ──────────────────────────────────
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::DMArgs mcast_args{
        .sender = {},
        .receiver =
            {
                get_semaphore(get_named_compile_time_arg_val("mcast_data_receiver_semaphore")),
                get_named_compile_time_arg_val("mcast_dst_cb"),
                get_named_compile_time_arg_val("mcast_dst_num_pages"),
            },
    };

    // ── RMSNorm + Matmul (no-ops on NCRISC, compute on TRISC) ──────
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // ── Sampling reader (matmul cores) ──────────────────────────────
    // Full CT-arg list from sampling.hpp :: TopKSampling::ReaderCTArgs<>
    // (mirrors micro_ops/sampling/kernels/sampling_kernel.cpp).
    //
    // ScoresCBId/ScoresNumPages enable the CB-backed scores read path
    // inside sampling.hpp (cb_wait_front + get_read_ptr + cb_pop_front).
    // We wire them to the matmul-output CB so phase-1 sees the matmul
    // result via cb_wait, not via a raw RT address.
    using SamplingCTArgs = deepseek_b1_ops::TopKSampling::ReaderCTArgs<
        get_named_compile_time_arg_val("sampling_num_values"),
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        get_named_compile_time_arg_val("sampling_num_senders"),
        get_named_compile_time_arg_val("sampling_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_receiver_semaphore_id"),
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage1_sender"),
        get_named_compile_time_arg_val("sampling_stage1_receiver"),
        get_named_compile_time_arg_val("sampling_stage2_sender"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_stage1_slot_base_offset"),
        get_named_compile_time_arg_val("sampling_stage1_num_slots"),
        get_named_compile_time_arg_val("sampling_stage1_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_stage1_local_slot_offset"),
        get_named_compile_time_arg_val("sampling_stage2_slot_base_offset"),
        get_named_compile_time_arg_val("sampling_stage2_num_slots"),
        get_named_compile_time_arg_val("sampling_stage2_expected_remote_incs"),
        get_named_compile_time_arg_val("sampling_stage2_local_slot_offset"),
        get_named_compile_time_arg_val("sampling_mesh_local_send_slot_offset"),
        get_named_compile_time_arg_val("sampling_sender_idx"),
        get_named_compile_time_arg_val("sampling_socket_mode"),
        get_named_compile_time_arg_val("sampling_socket_cb"),
        get_named_compile_time_arg_val("sampling_socket_page_size_bytes"),
        get_named_compile_time_arg_val("matmul_out"),    // ScoresCBId   — matmul output CB
        get_named_compile_time_arg_val("matmul_out_w"),  // ScoresNumPages — matmul output tile count
        get_named_compile_time_arg_val("sampling_winner_cb"),
        get_named_compile_time_arg_val("sampling_softmax_in_cb"),
        get_named_compile_time_arg_val("sampling_softmax_out_cb"),
        get_named_compile_time_arg_val("sampling_softmax_exp_cb"),
        get_named_compile_time_arg_val("sampling_scaler_cb"),
        get_named_compile_time_arg_val("sampling_temp_cb"),
        get_named_compile_time_arg_val("sampling_inv_temp_bf16"),
        get_named_compile_time_arg_val("sampling_topk_in_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_in_indices_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_indices_cb"),
        get_named_compile_time_arg_val("sampling_phase2_scores_byte_offset"),
        get_named_compile_time_arg_val("sampling_phase2_indices_byte_offset"),
        get_named_compile_time_arg_val("sampling_mesh_stage_scores_cb"),
        get_named_compile_time_arg_val("sampling_mesh_stage_indices_cb"),
        get_named_compile_time_arg_val("sampling_scores_scratch_stage2_offset"),
        get_named_compile_time_arg_val("sampling_indices_scratch_stage2_offset"),
        get_named_compile_time_arg_val("sampling_scores_scratch_addr"),
        get_named_compile_time_arg_val("sampling_indices_scratch_addr")>;

    // Layout matches sampling.hpp :: TopKSampling::ReaderArgs.
    deepseek_b1_ops::TopKSampling::ReaderArgs sampling_args{
        .scores_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .indices_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .output_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .final_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .final_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .global_sem_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
        .global_stage2_sem_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),
    };

    constexpr uint32_t mtp_embedding_base = get_named_compile_time_arg_val("mtp_embedding_dram_base");
    constexpr uint32_t metadata_output_l1_addr = get_named_compile_time_arg_val("metadata_output_l1_addr");
    constexpr uint32_t mtp_token_addr = get_named_compile_time_arg_val("mtp_token_l1_addr");

    // ── Sharded buffer setup (registers tensor-backed CBs before main loop) ──
    //   input_core:  CB 0 (rmsnorm_input), CB 7 (rmsnorm_gamma)
    //                CB 11 (h_gamma), CB 12 (e_gamma)  [MTP only]
    //   matmul_core: CB 2 (matmul_in1 / vocab weights)
    if constexpr (Core::is_input_core && !Core::fold_rmsnorm) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
    }
    if constexpr (Core::enable_mtp && Core::is_input_core && !Core::is_e_norm_device && !Core::fold_rmsnorm) {
        constexpr uint32_t h_gamma_cb = get_named_compile_time_arg_val("h_gamma_cb");
        constexpr uint32_t rmsnorm_h_num_tiles = get_named_compile_time_arg_val("rmsnorm_h_num_tiles");
        unified_kernels::setup_sharded_buffer(h_gamma_cb, rmsnorm_h_num_tiles);
    }
    if constexpr (Core::enable_mtp && Core::is_input_core && Core::is_e_norm_device && !Core::fold_rmsnorm) {
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
    deepseek_b1_ops::Mcast::DMArgs mcast_eh_args{
        .sender = {},
        .receiver =
            {
                get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
                get_named_compile_time_arg_val("mcast_eh_dst_cb"),
                get_named_compile_time_arg_val("mcast_eh_dst_num_pages"),
            },
    };

    // ── MTP: EH DRAM streaming matmul reader (eh_matmul_core) ───────
    constexpr uint32_t eh_in1_cb = get_named_compile_time_arg_val("matmul_eh_in1");
    constexpr uint32_t eh_out_cb = get_named_compile_time_arg_val("matmul_eh_out");
    constexpr uint32_t eh_out_w = get_named_compile_time_arg_val("matmul_eh_out_w");
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

    // ------------------------------------------------------------------------
    // ReduceToOneB1 (reader - receives data from fabric via semaphore waits)
    // ------------------------------------------------------------------------
    using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ReaderCTArgs<
        get_named_compile_time_arg_val("reduce_device_role"),
        get_named_compile_time_arg_val("reduce_num_tiles"),
        get_named_compile_time_arg_val("reduce_local_cb"),
        get_named_compile_time_arg_val("reduce_received_cb"),
        get_named_compile_time_arg_val("is_reduce_fabric_core")>;

    // Reader runtime args
    deepseek_b1_ops::ReduceToOneB1::ReaderArgs reduce_rt_args{
        get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // recv_sem_round1
        get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // recv_sem_round2
        get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // recv_sem_round3
    };

    // ── MTP token broadcast (tree-based, uses Broadcast::Op) ─────────
#if !defined(SKIP_CCL)
    using MtpBcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("mtp_bcast_data_cb_id"),
        get_named_compile_time_arg_val("mtp_bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("mtp_bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("mtp_bcast_num_neighbors"),
        get_named_compile_time_arg_val("mtp_bcast_num_links"),
        get_named_compile_time_arg_val("mtp_bcast_is_root"),
        get_named_compile_time_arg_val("mtp_bcast_chunk_size_bytes"),
        get_named_compile_time_arg_val("mtp_bcast_last_chunk_size_bytes"),
        get_named_compile_time_arg_val("mtp_bcast_num_chunks")>;
#endif
    constexpr uint32_t mtp_bcast_writer_common_rt_count = 5;
    deepseek_b1_ops::Broadcast::WriterArgs mtp_bcast_args{};
#if !defined(SKIP_CCL)
    if constexpr (!Core::skip_ccl && Core::enable_mtp) {
        uint32_t mtp_bcast_rta_num_args = 0;
        uint32_t mtp_bcast_rta_offset = 0;
        if constexpr (Core::is_input_core) {
            mtp_bcast_rta_num_args = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            mtp_bcast_rta_offset = per_core_rta_arg_idx;
            per_core_rta_arg_idx += mtp_bcast_rta_num_args;
        }
        mtp_bcast_args = deepseek_b1_ops::Broadcast::WriterArgs{
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // tensor_address0
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // my_noc_x
            get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // my_noc_y
            {
                get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // sem_bank_addrs[0]
                get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++),  // sem_bank_addrs[1]
            },
            mtp_bcast_rta_offset,
            mtp_bcast_rta_num_args,
        };
    }
#endif
    if constexpr (!Core::enable_mtp || Core::skip_ccl) {
        ncrisc_rt_arg_idx += mtp_bcast_writer_common_rt_count;
    }

#elif defined(COMPILE_FOR_BRISC)
    // ========================================================================
    // BRISC — CCL broadcast reader, mcast sender, argmax writer,
    //         MTP EH mcast sender, persistent signal routing
    //
    // Runtime args (fixed-index access from op.py brisc_bcast_common_args):
    //   ┌──────────────────────┬─────────┬────────────────────────────────────┐
    //   │ Section              │ Indices │ Notes                              │
    //   ├──────────────────────┼─────────┼────────────────────────────────────┤
    //   │ Argmax writer        │  [0..3] │ final_noc_x/y, scratch, socket    │
    //   │ Socket input reader  │  [4..6] │ config_addr, page_size, num_pages │
    //   │ Persistent routing   │ [7..12] │ enable, dst noc/mesh/chip, sem    │
    //   │ Verify / mcast dst   │   —     │ named compile-time args           │
    //   │ Fabric routing       │  [13+]  │ per-core appended                 │
    //   └──────────────────────┴─────────┴────────────────────────────────────┘
    // ========================================================================
    uint32_t brisc_rt_arg_idx = 0;
    // --- BRISC: CCL broadcast reader + optional socket-reader path + mcast sender ---
#if !defined(SKIP_CCL) || defined(ENABLE_SOCKET_READER)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_data_cb_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_root"),
        get_named_compile_time_arg_val("bcast_use_socket")>;
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),  // socket_config_addr
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),  // socket_page_size
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),  // socket_num_pages
    };
#else
    // Keep BRISC common-arg offsets stable when broadcast reader is compiled out
    // (SKIP_CCL without socket-reader). Host still prefixes 3 reader-common slots.
    brisc_rt_arg_idx = 3;
#endif
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

    // Matmul writer args (BRISC is a no-op for matmul; compute runs on TRISC)
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    // ── Sampling writer (matmul cores, sampling_is_final_core) ──────
    // Full CT-arg list from sampling.hpp :: TopKSampling::WriterCTArgs<>.
    // Template slot 18 (DeferSocketOutput) is the only arg this fused op
    // uses that isn't in the micro-op's kernel; we wire it explicitly.
    using SamplingCTArgs = deepseek_b1_ops::TopKSampling::WriterCTArgs<
        get_named_compile_time_arg_val("sampling_winner_page_bytes"),
        get_named_compile_time_arg_val("sampling_local_ready_semaphore_id"),
        get_named_compile_time_arg_val("sampling_socket_mode"),
        get_named_compile_time_arg_val("sampling_socket_cb"),
        get_named_compile_time_arg_val("sampling_socket_page_size_bytes"),
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_softmax_out_cb"),
        get_named_compile_time_arg_val("sampling_rand_cb"),
        get_named_compile_time_arg_val("sampling_winner_cb"),
        get_named_compile_time_arg_val("sampling_p_bf16"),
        get_named_compile_time_arg_val("sampling_topk_scores_slot_bytes"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_output_addr"),
        get_named_compile_time_arg_val("sampling_rand_output_addr"),
        get_named_compile_time_arg_val("sampling_inv_temp_bf16"),
        get_named_compile_time_arg_val("sampling_softmax_in_cb"),
        get_named_compile_time_arg_val("sampling_temp_cb"),
        get_named_compile_time_arg_val("sampling_defer_socket_output"),
        get_named_compile_time_arg_val("sampling_enable_metadata"),
        get_named_compile_time_arg_val("sampling_copy_probabilities"),
        get_named_compile_time_arg_val("metadata_output_l1_addr")>;

    deepseek_b1_ops::TopKSampling::WriterArgs sampling_args{
        .final_noc_x = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .final_noc_y = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .socket_config_addr = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_enable = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_dst_noc_x = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_dst_noc_y = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_dst_mesh_id = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_dst_chip_id = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
        .persistent_dst_sem_addr = get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),
    };
    const uint32_t persistent_next_iter_global_sem_addr = sampling_args.persistent_dst_sem_addr;

    // ── MTP token broadcast reader (BRISC side of Broadcast::Op) ─────
#if !defined(SKIP_CCL)
    using MtpBcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("mtp_bcast_data_cb_id"),
        get_named_compile_time_arg_val("mtp_bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("mtp_bcast_is_root"),
        0>;  // no socket for MTP token bcast
#endif
    deepseek_b1_ops::Broadcast::ReaderArgs mtp_bcast_args{
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),  // socket_config_addr (0)
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),  // socket_page_size (0)
        get_common_arg_val<uint32_t>(brisc_rt_arg_idx++),  // socket_num_pages (0)
    };

    constexpr uint32_t metadata_output_l1_addr = get_named_compile_time_arg_val("metadata_output_l1_addr");
    constexpr uint32_t mtp_token_addr = get_named_compile_time_arg_val("mtp_token_l1_addr");
    constexpr uint32_t mtp_input_core_noc_x = get_named_compile_time_arg_val("mtp_input_core_noc_x");
    constexpr uint32_t mtp_input_core_noc_y = get_named_compile_time_arg_val("mtp_input_core_noc_y");
    constexpr uint32_t mtp_argmax_output_addr = get_named_compile_time_arg_val("mtp_argmax_output_addr");

    // ── Mcast sender (input_core) ───────────────────────────────────
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;

    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
    // mcast dst CB addr (needed since this CB not allocated on sender)

    deepseek_b1_ops::Mcast::DMArgs mcast_args{
        .sender =
            {
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
            },
        .receiver = {},
    };
    // ── MTP: EH mcast sender (input_core, enable_mtp) ──────────────
    using McastEhCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_eh_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;
    constexpr uint32_t mcast_eh_src_cb = get_named_compile_time_arg_val("mcast_eh_src_cb");
    constexpr uint32_t mcast_eh_dst_cb = get_named_compile_time_arg_val("mcast_eh_dst_cb");
    constexpr uint32_t eh_norm_slice_offset_bytes = get_named_compile_time_arg_val("eh_norm_slice_offset_bytes");

    deepseek_b1_ops::Mcast::DMArgs mcast_eh_args{
        .sender =
            {
                get_named_compile_time_arg_val("mcast_eh_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_eh_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_eh_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_eh_dest_noc_end_y"),
                get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_sender_semaphore")),
                get_semaphore(get_named_compile_time_arg_val("mcast_eh_data_receiver_semaphore")),
                get_named_compile_time_arg_val("mcast_eh_data_size_bytes"),
                mcast_eh_src_cb,
                get_named_compile_time_arg_val("mcast_eh_src_num_pages"),
                Core::is_input_core ? get_read_ptr(mcast_eh_src_cb) + eh_norm_slice_offset_bytes : 0,
                get_write_ptr(mcast_eh_dst_cb),
            },
        .receiver = {},
    };
    // ------------------------------------------------------------------------
    // ReduceToOneB1 (writer - sends data via fabric or NOC)
    // ------------------------------------------------------------------------
    using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::WriterCTArgs<
        get_named_compile_time_arg_val("reduce_device_role"),
        get_named_compile_time_arg_val("reduce_num_tiles"),
        get_named_compile_time_arg_val("reduce_payload_size_bytes"),
        get_named_compile_time_arg_val("reduce_local_cb"),
        get_named_compile_time_arg_val("reduce_scratch_cb"),
        get_named_compile_time_arg_val("reduce_packet_cb"),
        get_named_compile_time_arg_val("reduce_num_hops"),
        get_named_compile_time_arg_val("reduce_dst_fabric_node_chip_id"),
        get_named_compile_time_arg_val("reduce_dst_fabric_node_mesh_id"),
        get_named_compile_time_arg_val("reduce_output_core_noc_x"),
        get_named_compile_time_arg_val("reduce_output_core_noc_y"),
        get_named_compile_time_arg_val("reduce_num_workers"),
        get_named_compile_time_arg_val("is_reduce_fabric_core"),
        0>;  // enable_downstream_socket

    // Writer runtime args for worker cores (skip past argmax mesh sender args if co-located)
    constexpr size_t reduce_brisc_arg_start = Core::sampling_mesh_sender_core ? 5 : 0;
    deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs reduce_rt_args{};
    if constexpr (Core::is_eh_reduce_worker_core) {
        reduce_rt_args = deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs{
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 0),  // fabric_core_noc_x
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 1),  // fabric_core_noc_y
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 2),  // my_slot_idx
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 3),  // worker_sem_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 4),  // dst_l1_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 5),  // dst_sem_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 6),  // output_base_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 7),  // shard_idx
        };
    }

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
    // ── Sampling compute (matmul cores; phase-1 everywhere, phase-2/final only on final core) ──
    using SamplingCTArgs = deepseek_b1_ops::TopKSampling::ComputeCTArgs<
        get_named_compile_time_arg_val("sampling_softmax_in_cb"),
        get_named_compile_time_arg_val("sampling_softmax_out_cb"),
        get_named_compile_time_arg_val("sampling_softmax_exp_cb"),
        get_named_compile_time_arg_val("sampling_softmax_sub_cb"),
        get_named_compile_time_arg_val("sampling_max_cb"),
        get_named_compile_time_arg_val("sampling_sum_cb"),
        get_named_compile_time_arg_val("sampling_scaler_cb"),
        get_named_compile_time_arg_val("sampling_temp_cb"),
        get_named_compile_time_arg_val("sampling_rand_cb"),
        get_named_compile_time_arg_val("sampling_seed"),
        get_named_compile_time_arg_val("sampling_topk_k"),
        get_named_compile_time_arg_val("sampling_mesh_mode"),
        get_named_compile_time_arg_val("sampling_stage1_receiver"),
        get_named_compile_time_arg_val("sampling_stage2_receiver"),
        get_named_compile_time_arg_val("sampling_num_values"),
        get_named_compile_time_arg_val("sampling_num_senders"),
        get_named_compile_time_arg_val("sampling_topk_in_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_in_indices_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_scores_cb"),
        get_named_compile_time_arg_val("sampling_topk_out_indices_cb"),
        get_named_compile_time_arg_val("sampling_mesh_stage_scores_cb"),
        get_named_compile_time_arg_val("sampling_mesh_stage_indices_cb"),
        get_named_compile_time_arg_val("sampling_stage1_row_elements"),
        get_named_compile_time_arg_val("sampling_stage1_num_input_tiles"),
        get_named_compile_time_arg_val("sampling_stage2_row_elements"),
        get_named_compile_time_arg_val("sampling_stage2_num_input_tiles")>;
    deepseek_b1_ops::TopKSampling::ComputeArgs sampling_args{};

    // ── ReduceToOneB1 compute (eh_reduce cores) ─────────────────────
    using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ComputeCTArgs<
        get_named_compile_time_arg_val("reduce_device_role"),
        get_named_compile_time_arg_val("reduce_num_tiles"),
        get_named_compile_time_arg_val("reduce_local_cb"),
        get_named_compile_time_arg_val("reduce_received_cb"),
        get_named_compile_time_arg_val("reduce_output_cb"),
        get_named_compile_time_arg_val("reduce_scratch_cb"),
        get_named_compile_time_arg_val("is_reduce_fabric_core")>;
    deepseek_b1_ops::ReduceToOneB1::ComputeArgs reduce_rt_args{};

    // ── RMSNorm compute (input_core / rmsnorm_core) ─────────────────
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        !Core::fold_rmsnorm>;
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
        get_named_compile_time_arg_val("rmsnorm_h_output_cb"),
        !Core::fold_rmsnorm>;

    using ERMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_e_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_e_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_e_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_e_output_cb"),
        !Core::fold_rmsnorm>;

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
        DST_ACCUM_MODE>;

    // ── HW compute startup ──────────────────────────────────────────
    // Sampling needs fp32_dest_acc_en and an SFPU/FPU semaphore pair (see
    // micro_ops/sampling/kernels/sampling_kernel.cpp). On non-sampling
    // compute cores we fall back to the bare default init so we don't
    // read metadata from sampling CBs that aren't allocated there.
    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(ckernel::SFPU_FPU, 0, 1));
    deepseek_compute_kernel_hw_startup<true>(
        SamplingCTArgs::topk_in_scores_cb, SamplingCTArgs::topk_in_scores_cb, SamplingCTArgs::topk_out_scores_cb);
#endif

    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
    deepseek_b1_ops::TopKSampling::
        Op<SamplingCTArgs, Core::sampling_is_active_core, Core::sampling_is_final_core, Core::sampling_mesh_sender_core>
            sampling_op;

#if defined(COMPILE_FOR_TRISC)
    // Mirrors micro_ops/sampling/kernels/sampling_kernel.cpp:183 — without
    // this, rand_tile_init() is never called and the LLK random tile
    // generator falls back to its uninitialised default state, making the
    // `sampling_seed` CT arg a no-op. Gated on sampling_is_active_core so
    // non-sampling cores don't touch the SFPU rand state.
    if constexpr (Core::sampling_is_active_core) {
        sampling_op.set_seed(SamplingCTArgs::seed);
    }
#endif

    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_input_core, Core::is_mcast_receiver_core, Core::is_matmul_core, true>
            mcast;

    constexpr uint32_t termination_semaphore_addr = get_named_compile_time_arg_val("termination_semaphore_addr");
    deepseek_b1_ops::PersistentLoop<Core::persistent_mode> loop(termination_semaphore_addr);

    constexpr uint32_t TOKEN_TYPE_BASE = 0;
    constexpr uint32_t TOKEN_TYPE_SPEC = 1;

#if defined(COMPILE_FOR_BRISC)
    // Write the full DeepseekMetadata output page into the given CB.
    //
    // Header (words 0-8):
    //   [tok0_id, tok0_type, tok0_pos, tok1_id, tok1_type, tok1_pos,
    //    slot_id, 0, input_pos_id]
    //
    // p_indices / p_scores (words 16-63):
    //   When metadata_src_addr != 0 the trailing arrays are copied from that
    //   L1 address (used by the spec stage to forward the base stage's
    //   probabilities).  When 0 the caller guarantees they are already
    //   in-place (base stage writes them via copy_probabilities directly
    //   into the CB page).
    auto write_token_metadata_to_socket_cb = [](uint32_t cb,
                                                uint32_t tok0_id,
                                                uint32_t tok0_type,
                                                uint32_t tok0_pos,
                                                uint32_t tok1_id = 0,
                                                uint32_t tok1_type = 0,
                                                uint32_t tok1_pos = 0,
                                                uint32_t input_pos_id = 0,
                                                uint32_t slot_id = 0,
                                                uint32_t k = 0,
                                                uint32_t temperature = 0,
                                                uint32_t probability_mass_threshold = 0,
                                                uint32_t metadata_src_addr = 0) {
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
        page[0] = tok0_id;
        page[1] = tok0_type;
        page[2] = tok0_pos;
        page[3] = tok1_id;
        page[4] = tok1_type;
        page[5] = tok1_pos;
        page[6] = slot_id;
        page[7] = 0; // input token id
        page[8] = input_pos_id;  // position id
        page[9] = 0;             // prefill token id
        page[10] = temperature;
        page[11] = k;
        page[12] = probability_mass_threshold;
        if (metadata_src_addr != 0) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_src_addr);
            for (uint32_t i = 16; i < 64; ++i) {
                page[i] = src[i];
            }
        }
    };
#endif

    // ====================================================================
    // LM HEAD SAMPLING Lambda
    // ====================================================================
    auto lm_head_sampling = [&]() {
        // ====================================================================
        // Phase 0: CCL Broadcast (multi-device only) (NCRISC only)
        // ====================================================================
#if defined(COMPILE_FOR_BRISC) && !defined(SKIP_CCL)
        constexpr bool is_root = get_named_compile_time_arg_val("bcast_is_root") == 1;
        if constexpr (Core::persistent_mode && is_root && Core::is_input_core) {
            auto next_iteration_semaphore =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(persistent_next_iter_global_sem_addr);
            noc_semaphore_wait(next_iteration_semaphore, 1);
            noc_semaphore_set(next_iteration_semaphore, 0);
        }
#endif

#if defined(COMPILE_FOR_NCRISC)
        // Device-local fabric gate (pre-bcast acquire).
        if constexpr (Core::is_input_core && !Core::skip_ccl) {
            auto* bcast_turn_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(Core::fabric_gate_bcast_turn_semaphore_id));
            noc_semaphore_wait(bcast_turn_sem, 1);
            noc_semaphore_set(bcast_turn_sem, 0);
        }
#endif

        // Keep broadcast symbol usage out of SKIP_CCL builds unless BRISC
        // socket-reader mode is compiled in.
#if !defined(SKIP_CCL) || (defined(ENABLE_SOCKET_READER) && defined(COMPILE_FOR_BRISC))
        if constexpr (!Core::skip_ccl || Core::bcast_use_socket_input) {
            deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
            {
                DeviceZoneScopedN("CCL_BROADCAST");
                bcast(bcast_args);
            }
        }
#endif

#if defined(COMPILE_FOR_NCRISC)
        // Device-local fabric gate (post-bcast release to argmax side).
        if constexpr (Core::is_input_core && !Core::skip_ccl) {
            auto argmax_turn_sem_noc_addr = get_noc_addr(
                Core::fabric_gate_argmax_noc_x,
                Core::fabric_gate_argmax_noc_y,
                get_semaphore(Core::fabric_gate_argmax_turn_semaphore_id));
            noc_semaphore_inc(argmax_turn_sem_noc_addr, 1);
            noc_async_atomic_barrier();
        }
#endif

#if defined(COMPILE_FOR_NCRISC)
        // On the broadcast sender, BRISC already pushes to CB 0 during the broadcast
        // (reserve_back + socket read into write_ptr + push_back). Do not push again
        // or we leave a stale tile that rmsnorm reads on the next iteration (wrong tokens).
        if constexpr (Core::is_input_core && (!Core::skip_ccl || !Core::bcast_use_socket_input)) {
            constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
            constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
            unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        }

        if constexpr (Core::is_input_core && Core::is_exit_device) {
            constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
            constexpr uint32_t argmax_noc_x = get_named_compile_time_arg_val("argmax_core_noc_x");
            constexpr uint32_t argmax_noc_y = get_named_compile_time_arg_val("argmax_core_noc_y");
            constexpr uint32_t metadata_size = sizeof(deepseek_b1_ops::DeepseekMetadata);
            constexpr uint32_t activation_size_bytes = 14336;
            uint32_t rmsnorm_buffer_addr = get_read_ptr(rmsnorm_input_cb);
            uint32_t metadata_src = rmsnorm_buffer_addr + activation_size_bytes;

            // // DEBUG: Print metadata as received from decoder pipeline (before unicast to argmax)
            // {
            //     volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata* md =
            //         reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(metadata_src);
            //     invalidate_l1_cache();
            //     DPRINT << "MD_INPUT iter=" << iteration_count << " pos=" << md->position_id << " slot=" <<
            //     md->slot_id
            //            << " tok=" << md->token_id << " k=" << md->k << ENDL();
            // }

            uint64_t metadata_dst = get_noc_addr(argmax_noc_x, argmax_noc_y, metadata_output_l1_addr);
            noc_async_write(metadata_src, metadata_dst, metadata_size);
            noc_async_write_barrier();
            uint64_t sem_addr = get_noc_addr(
                argmax_noc_x,
                argmax_noc_y,
                get_semaphore(get_named_compile_time_arg_val("metadata_ready_semaphore_id")));
            noc_semaphore_inc(sem_addr, 1);
            noc_async_atomic_barrier();
        }
#endif

        {
            DeviceZoneScopedN("RMSNORM");
            constexpr bool pop_rmsnorm_src = Core::is_e_norm_device || !Core::enable_mtp;
            deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_rmsnorm_core, pop_rmsnorm_src> rmsnorm;
            rmsnorm(rmsnorm_args);
        }

        {
            DeviceZoneScopedN("MCAST");
            mcast(mcast_args);
        }

#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::is_rmsnorm_core && !Core::is_e_norm_device && Core::enable_mtp) {
            constexpr uint32_t hnorm_ready_cb = get_named_compile_time_arg_val("hnorm_ready_cb");
            cb_reserve_back(hnorm_ready_cb, 1);
            cb_push_back(hnorm_ready_cb, 1);
        }
#endif

        {
            DeviceZoneScopedN("MATMUL");
            matmul(matmul_args);
        }

#if defined(COMPILE_FOR_BRISC)
        // Device-local fabric gate (pre-sampling acquire on argmax final core).
        if constexpr (Core::sampling_is_final_core && !Core::skip_ccl) {
            auto* argmax_turn_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(Core::fabric_gate_argmax_turn_semaphore_id));
            noc_semaphore_wait(argmax_turn_sem, 1);
            noc_semaphore_set(argmax_turn_sem, 0);
        }

        // Pre-sampling metadata barrier (single source of truth for both stages).
        // The exit-device input core unicasts the DeepseekMetadata struct and
        // increments `metadata_ready_semaphore_id` above. Sampling.hpp reads
        // temperature / k / probability_mass_threshold off this struct on the
        // base stage (enable_metadata=True), and the downstream `mtp` /
        // `update_speculative_state` lambdas read tok0_*/slot_id from it
        // unconditionally. Waiting + clearing here means neither downstream
        // path needs its own wait, and avoids the prior race where sampling
        // started before metadata had landed.
        if constexpr (Core::sampling_is_final_core && Core::is_exit_device) {
            volatile tt_l1_ptr uint32_t* metadata_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_semaphore(get_named_compile_time_arg_val("metadata_ready_semaphore_id")));
            noc_semaphore_wait(metadata_ready_sem, 1);
            noc_semaphore_set(metadata_ready_sem, 0);
        }
#endif

        {
            DeviceZoneScopedN("SAMPLING");
            sampling_op(sampling_args);
        }
    };

    // ====================================================================
    // MTP Lambda
    // ====================================================================
    auto mtp = [&]() {

    // ====================================================================
    // [MTP] Token unicast from argmax_final_core to input_core on exit device
    // ====================================================================
#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::sampling_is_final_core && Core::is_exit_device) {
            constexpr uint32_t mtp_input_core_noc_x = get_named_compile_time_arg_val("mtp_input_core_noc_x");
            constexpr uint32_t mtp_input_core_noc_y = get_named_compile_time_arg_val("mtp_input_core_noc_y");
            constexpr uint32_t mtp_token_addr = get_named_compile_time_arg_val("mtp_token_l1_addr");
            constexpr uint32_t mtp_argmax_output_addr = get_named_compile_time_arg_val("mtp_argmax_output_addr");
            uint64_t dst = get_noc_addr(mtp_input_core_noc_x, mtp_input_core_noc_y, mtp_token_addr);
            noc_async_write(mtp_argmax_output_addr, dst, 4);
            noc_async_write_barrier();
            uint64_t sem_addr = get_noc_addr(
                mtp_input_core_noc_x,
                mtp_input_core_noc_y,
                get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
            noc_semaphore_inc(sem_addr, 1);
            noc_async_atomic_barrier();
        }
#endif

        // MTP token broadcast via tree-based Broadcast::Op.
        // Root (exit device): wait for token from argmax_final_core, then
        // BRISC pushes CB and NCRISC runs the broadcast writer tree.
        // Non-root: NCRISC broadcast writer waits for fabric semaphore.
#if defined(COMPILE_FOR_BRISC)
#if !defined(SKIP_CCL)
        if constexpr (Core::is_input_core) {
            deepseek_b1_ops::Broadcast::Op<MtpBcastCTArgs, Core::is_input_core> token_bcast_sender;
            token_bcast_sender(mtp_bcast_args);
        }
#endif
#endif

#if defined(COMPILE_FOR_NCRISC)
#if !defined(SKIP_CCL)
        if constexpr (Core::is_input_core) {
            if constexpr (Core::is_exit_device) {
                volatile tt_l1_ptr uint32_t* mtp_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_wait(mtp_ready_sem, 1);
                noc_semaphore_set(mtp_ready_sem, 0);
            }
            deepseek_b1_ops::Broadcast::Op<MtpBcastCTArgs, Core::is_input_core> token_bcast_receiver;
            token_bcast_receiver(mtp_bcast_args);

            // Signal BRISC that the MTP token broadcast has completed. The
            // bcast tree traces back through the exit-device input_core's
            // mtp_ready_sem wait, which itself only fires after sampling on
            // argmax_final_core has completed -- and sampling phase 1 only
            // reaches the per-core-send step after that core's TRISC matmul
            // has popped CB 1.  By gating BRISC's mcast_eh on this push we
            // guarantee that no receiver matmul core is still reading L1
            // region X (CB 1) when mcast_eh writes into the aliased CB 18.
            if constexpr (Core::enable_mtp) {
                constexpr uint32_t mcast_eh_ready_cb = get_named_compile_time_arg_val("mcast_eh_ready_cb");
                cb_reserve_back(mcast_eh_ready_cb, 1);
                cb_push_back(mcast_eh_ready_cb, 1);
            }
        }
#endif
#endif

#if defined(COMPILE_FOR_NCRISC)
        // Pre-reduce gate: BRISC on argmax_final_core is the last to close
        // fabric connections (it runs the argmax writer / fabric sender).
        // Signal the 2 reduce fabric cores that it is safe to open theirs.
        if constexpr (Core::is_input_core && !Core::skip_ccl) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_named_compile_time_arg_val("reduce_fc_noc_0_x"),
                    get_named_compile_time_arg_val("reduce_fc_noc_0_y"),
                    get_semaphore(Core::reduce_gate_semaphore_id)),
                1);
            noc_async_atomic_barrier();
            noc_semaphore_inc(
                get_noc_addr(
                    get_named_compile_time_arg_val("reduce_fc_noc_1_x"),
                    get_named_compile_time_arg_val("reduce_fc_noc_1_y"),
                    get_semaphore(Core::reduce_gate_semaphore_id)),
                1);
            noc_async_atomic_barrier();
        }
#endif

        // ====================================================================
        // [MTP] Embedding lookup (NCRISC on input_core)
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_input_core && Core::is_e_norm_device) {
            constexpr uint32_t embedding_size_bytes = get_named_compile_time_arg_val("embedding_size_bytes");
            constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
            constexpr uint32_t emb_cb = get_named_compile_time_arg_val("embedding_cb");
            constexpr uint32_t e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");
            auto embedding_addr_gen = TensorAccessor(
                tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), mtp_embedding_base, embedding_size_bytes);
            invalidate_l1_cache();
            uint32_t metadata_src_addr = get_read_ptr(rmsnorm_input_cb) + embedding_size_bytes;
            auto* metadata_ptr =
                reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(metadata_src_addr);
            uint32_t token_id = (metadata_ptr->prefill_token_id != static_cast<uint32_t>(-1))
                                    ? metadata_ptr->prefill_token_id
                                    : *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_token_addr);
            cb_reserve_back(emb_cb, e_num_tiles);
            noc_async_read(embedding_addr_gen.get_noc_addr(token_id), get_write_ptr(emb_cb), embedding_size_bytes);
            noc_async_read_barrier();
            cb_push_back(emb_cb, e_num_tiles);
        }
#endif

        // ====================================================================
        // [MTP] h_rmsnorm and e_rmsnorm on TRISC — starts immediately after the LM head matmul,
        // overlapping with argmax on NCRISC/BRISC. CB 0 still has hidden states
        // (first RMSNorm used pop_input=false when MTP is enabled).
        // Output writes directly to mcast_eh_src_cb (first half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
        if constexpr (Core::is_rmsnorm_core) {
            if constexpr (Core::is_e_norm_device) {
                DeviceZoneScopedN("MTP_E_RMSNORM");
                deepseek_b1_ops::RMSNorm::Op<ERMSNormCTArgs, Core::is_rmsnorm_core, true> e_rmsnorm;
                e_rmsnorm(rmsnorm_args);
            } else {
                constexpr uint32_t hnorm_ready_cb = get_named_compile_time_arg_val("hnorm_ready_cb");
                cb_wait_front(hnorm_ready_cb, 1);
                cb_pop_front(hnorm_ready_cb, 1);
                DeviceZoneScopedN("MTP_H_RMSNORM");
                deepseek_b1_ops::RMSNorm::Op<HRMSNormCTArgs, Core::is_rmsnorm_core, true> h_rmsnorm;
                h_rmsnorm(rmsnorm_args);
            }
        }
#endif

        // ====================================================================
        // [MTP] Second mcast — multicast e norm or h norm chunks from sender to all cores
        // ====================================================================
        // Gate BRISC's mcast_eh on NCRISC's `mcast_eh_ready_cb` push above
        // (which fires after `token_bcast_receiver` returns).  This is the
        // performance-independent cure for the CB1 / CB18 L1 alias race: the
        // token broadcast cannot complete on any input_core's NCRISC until
        // every matmul core's TRISC has popped CB 1 (transitively, via
        // sampling phase 1 -> phase 2 -> sampling_final_core BRISC unicast ->
        // mtp_ready_sem -> exit-device fabric send).  Without this wait,
        // BRISC fires mcast_eh as soon as TRISC pushes CB 15, which is gated
        // only on h_rmsnorm time and races receiver matmul.  Sender-only;
        // receivers' BRISC falls through (mcast_eh receive runs on NCRISC).
#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::is_input_core && Core::enable_mtp) {
            constexpr uint32_t mcast_eh_ready_cb = get_named_compile_time_arg_val("mcast_eh_ready_cb");
            cb_wait_front(mcast_eh_ready_cb, 1);
            cb_pop_front(mcast_eh_ready_cb, 1);
        }
#endif

        {
            deepseek_b1_ops::Mcast::
                Op<McastEhCTArgs, Core::is_input_core, Core::is_mcast_receiver_core, Core::is_eh_matmul_core, true>
                    mcast_eh;
            DeviceZoneScopedN("MTP_EH_MCAST");
            mcast_eh(mcast_eh_args);
        }

#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_eh_matmul_core) {
            deepseek_b1_ops::DRAMStreamingMatmul::Op<EHDRAMMMCTArgs, true, true, false, 0, false, false, 3> eh_matmul;
            {
                DeviceZoneScopedN("MTP_EH_DRAM_MATMUL");
                eh_matmul();
            }
        }
#endif

#if defined(COMPILE_FOR_TRISC)
        // Bridge: drain CB17 (EH matmul output, 28 [1,32] tiles) and expose through
        // reduce_local_cb (CB21, aliased L1, 1 page of payload_size_bytes).
        // This lets the reduce see the full shard as 1 compute tile (num_tiles=1).
        if constexpr (Core::is_eh_matmul_core) {
            cb_wait_front(eh_out_cb, eh_out_w);
            cb_pop_front(eh_out_cb, eh_out_w);
            constexpr uint32_t reduce_lcb = get_named_compile_time_arg_val("reduce_local_cb");
            cb_reserve_back(reduce_lcb, 1);
            cb_push_back(reduce_lcb, 1);
        }
#endif

#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::is_eh_reduce_fabric_core) {
            auto* reduce_gate =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(Core::reduce_gate_semaphore_id));
            noc_semaphore_wait(reduce_gate, 1);
            noc_semaphore_set(reduce_gate, 0);
        }
#endif

        {
            DeviceZoneScopedN("REDUCE_TO_ONE");
            constexpr bool is_reduce_core = Core::is_eh_reduce_worker_core || Core::is_eh_reduce_fabric_core;
            deepseek_b1_ops::ReduceToOneB1::Op<ReduceToOneCTArgs, is_reduce_core, true> reduce_op;
            reduce_op(reduce_rt_args);
        }

        // Reset received_cb fifo pointers after reduce_to_one.
        // Fabric writes to fixed L1 slots (0/1/2), but ROOT2/ROOT3 consume
        // fewer than 3 CB pages per iteration, causing the CB's internal
        // rd/wr pointers to drift out of alignment with those fixed addresses.
        // Each RISC independently resets its own local CB interface pointers
        // to the base address, matching reconfig_cbs_for_mask behaviour.
        // NCRISC also resets the shared stream registers after confirming
        // TRISC has finished all pops (tiles_acked == tiles_received).
        constexpr bool needs_received_cb_reset =
            Core::is_eh_reduce_worker_core && ReduceToOneCTArgs::device_role != deepseek_b1_ops::MESH_LEAF;
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (needs_received_cb_reset) {
            constexpr uint32_t rcb = ReduceToOneCTArgs::received_cb;
            volatile uint32_t* rcv_ptr = get_cb_tiles_received_ptr(rcb);
            uintptr_t ack_addr = (uintptr_t)get_cb_tiles_acked_ptr(rcb);
            uint16_t tiles_pushed = (uint16_t)rcv_ptr[0];
            while (true) {
                invalidate_l1_cache();
                if ((uint16_t)reg_read(ack_addr) == tiles_pushed)
                    break;
            }
            *get_cb_tiles_received_ptr(rcb) = 0;
            *get_cb_tiles_acked_ptr(rcb) = 0;
            auto& nc_iface = get_local_cb_interface(rcb);
            uint32_t base = nc_iface.fifo_limit - nc_iface.fifo_size;
            nc_iface.fifo_rd_ptr = base;
            nc_iface.fifo_wr_ptr = base;
        }
#elif defined(COMPILE_FOR_TRISC)
        if constexpr (needs_received_cb_reset) {
            UNPACK({
                constexpr uint32_t rcb_id = ReduceToOneCTArgs::received_cb;
                auto& tr_iface = get_local_cb_interface(rcb_id);
                uint32_t base = tr_iface.fifo_limit - tr_iface.fifo_size;
                tr_iface.fifo_rd_ptr = base;
                tr_iface.tiles_acked_received_init = 0;
            });
        }
#endif

#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::is_eh_reduce_fabric_core || Core::is_eh_reduce_worker_core) {
            constexpr uint32_t argmax_noc_x = get_named_compile_time_arg_val("argmax_core_noc_x");
            constexpr uint32_t argmax_noc_y = get_named_compile_time_arg_val("argmax_core_noc_y");
            uint64_t sync_noc_addr =
                get_noc_addr(argmax_noc_x, argmax_noc_y, get_semaphore(Core::reduce_gate_semaphore_id));
            noc_semaphore_inc(sync_noc_addr, 1);
            noc_async_atomic_barrier();
        }
#endif

#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::sampling_is_final_core) {
            constexpr uint32_t reduce_done_num = get_named_compile_time_arg_val("reduce_gate_num_targets");
            auto* reduce_done_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(Core::reduce_gate_semaphore_id));
            noc_semaphore_wait(reduce_done_sem, reduce_done_num);
            noc_semaphore_set(reduce_done_sem, 0);
        }
#endif

#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::sampling_is_final_core && Core::is_exit_device) {
            // Metadata is guaranteed valid here: lm_head_sampling() waited and
            // cleared `metadata_ready_semaphore_id` before sampling_op ran.
            volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata* metadata_ptr =
                reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(metadata_output_l1_addr);

            invalidate_l1_cache();
            uint32_t base_token_type = metadata_ptr->tok0_type;
            uint32_t base_token_pos = metadata_ptr->position_id;
            uint32_t input_pos_id = metadata_ptr->tok0_pos + 1;
            uint32_t slot_id = metadata_ptr->slot_id;
            uint32_t k = metadata_ptr->k;
            volatile tt_l1_ptr uint32_t* metadata_raw =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_output_l1_addr);
            uint32_t temperature = metadata_raw[10];
            uint32_t probability_mass_threshold = metadata_raw[12];

            constexpr uint32_t eh_gather_dst_cb = get_named_compile_time_arg_val("gather_dst_cb");
            constexpr uint32_t sampling_socket_cb = get_named_compile_time_arg_val("sampling_socket_cb");
            constexpr uint32_t eh_gather_num_pages = get_named_compile_time_arg_val("gather_dst_num_pages");

            cb_reserve_back(eh_gather_dst_cb, eh_gather_num_pages);
            cb_push_back(eh_gather_dst_cb, eh_gather_num_pages);

            cb_wait_front(sampling_socket_cb, 1);
            uint32_t base_token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(sampling_socket_cb));
            cb_reserve_back(eh_gather_dst_cb, 1);
            write_token_metadata_to_socket_cb(
                eh_gather_dst_cb,
                base_token_id,
                base_token_type,
                base_token_pos,
                0,
                0,
                0,
                input_pos_id,
                slot_id,
                k,
                temperature,
                probability_mass_threshold,
                metadata_output_l1_addr);
            cb_push_back(eh_gather_dst_cb, 1);
            cb_pop_front(sampling_socket_cb, 1);
        }
#endif
    };

    // ========================================================================
    // update_speculative_state: runs on BRISC for the spec stage (argmax_final_core).
    //
    // Must run on BRISC (not NCRISC) to avoid racing with the socket send section
    // for CB 6. After lm_head_sampling(), the argmax writer (BRISC) has already pushed
    // the speculative token to CB 6. This function consumes that page, reads the base
    // token from metadata L1 (transferred by NCRISC during the broadcast phase), and
    // writes a TOKEN_META page with both tokens back to CB 6.
    //
    // Metadata layout from base stage (at metadata_output_l1_addr):
    //   [0] = num_tokens, [1] = tok0_id, [2] = tok0_type, [3] = tok0_pos, ...
    // ========================================================================
    auto update_speculative_state = [&]() {
#if defined(COMPILE_FOR_BRISC)
        if constexpr (
            Core::sampling_is_final_core && SamplingCTArgs::defer_socket_output && SamplingCTArgs::socket_mode != 0) {
            DeviceZoneScopedN("MTP_VERIFY_SEND");

            // Metadata is guaranteed valid here: lm_head_sampling() waited and
            // cleared `metadata_ready_semaphore_id` before sampling_op ran.

            // Read the speculative token from the sampling socket CB (produced by spec stage)
            constexpr uint32_t sampling_socket_cb = SamplingCTArgs::socket_cb_id;
            cb_wait_front(sampling_socket_cb, 1);
            invalidate_l1_cache();
            uint32_t spec_token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(sampling_socket_cb));

            // Read the base token from metadata L1 (transferred by NCRISC during the broadcast phase)
            volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata* metadata_ptr =
                reinterpret_cast<volatile tt_l1_ptr deepseek_b1_ops::DeepseekMetadata*>(metadata_output_l1_addr);
            invalidate_l1_cache();
            uint32_t base_token_id = metadata_ptr->tok0_id;
            uint32_t base_token_type = metadata_ptr->tok0_type;
            uint32_t base_token_pos = metadata_ptr->tok0_pos + 1;
            uint32_t slot_id = metadata_ptr->slot_id;
            uint32_t spec_token_type = TOKEN_TYPE_SPEC;
            uint32_t spec_token_pos = metadata_ptr->tok0_pos + 2;
            uint32_t input_pos_id = metadata_ptr->position_id;
            uint32_t k = metadata_ptr->k;
            volatile tt_l1_ptr uint32_t* metadata_raw2 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_output_l1_addr);
            uint32_t temperature = metadata_raw2[10];
            uint32_t probability_mass_threshold = metadata_raw2[12];
            cb_pop_front(sampling_socket_cb, 1);

            cb_reserve_back(sampling_socket_cb, 1);
            write_token_metadata_to_socket_cb(
                sampling_socket_cb,
                base_token_id,
                base_token_type,
                base_token_pos,
                spec_token_id,
                spec_token_type,
                spec_token_pos,
                input_pos_id,
                slot_id,
                k,
                temperature,
                probability_mass_threshold,
                metadata_output_l1_addr);
            cb_push_back(sampling_socket_cb, 1);
        }
#endif
    };

    mcast.init(mcast_args);

    // Persistent loop
    while (loop.next()) {
        // Base Stage: run LM head sampling and MTP ops
        if constexpr (Core::is_base_stage) {
            lm_head_sampling();
            if constexpr (Core::enable_mtp) {
                mtp();
            }
        }

        // Spec Stage: run MTP LM head sampling and update speculative state
        if constexpr (Core::is_spec_stage) {
            lm_head_sampling();
            update_speculative_state();
        }

        // ====================================================================
        // Socket Send and Signal
        // ====================================================================
#if defined(COMPILE_FOR_BRISC)
        if constexpr (
            Core::sampling_is_final_core && SamplingCTArgs::defer_socket_output && SamplingCTArgs::socket_mode != 0 &&
            Core::persistent_mode) {
            if constexpr (Core::is_base_stage) {
                if constexpr (Core::enable_mtp) {
                    constexpr uint32_t eh_gather_dst_cb = get_named_compile_time_arg_val("gather_dst_cb");
                    constexpr uint32_t eh_gather_num_pages = get_named_compile_time_arg_val("gather_dst_num_pages") + 1;
                    constexpr uint32_t eh_gather_total_bytes =
                        get_named_compile_time_arg_val("gather_send_total_bytes");
                    unified_kernels::socket_send_from_cb<SamplingCTArgs::socket_mode>(
                        sampling_args.socket_config_addr, eh_gather_dst_cb, eh_gather_num_pages, eh_gather_total_bytes);
                    {
                        auto& iface = get_local_cb_interface(eh_gather_dst_cb);
                        uint32_t base = iface.fifo_limit - iface.fifo_size;
                        iface.fifo_rd_ptr = base;
                        iface.fifo_wr_ptr = base;
                        *get_cb_tiles_received_ptr(eh_gather_dst_cb) = 0;
                        *get_cb_tiles_acked_ptr(eh_gather_dst_cb) = 0;
                    }
                } else {
                    unified_kernels::socket_send_from_cb<SamplingCTArgs::socket_mode>(
                        sampling_args.socket_config_addr,
                        SamplingCTArgs::socket_cb_id,
                        1,
                        SamplingCTArgs::socket_page_size_bytes);
                }

            } else if constexpr (Core::is_spec_stage) {
                unified_kernels::socket_send_from_cb<SamplingCTArgs::socket_mode>(
                    sampling_args.socket_config_addr,
                    SamplingCTArgs::socket_cb_id,
                    1,
                    SamplingCTArgs::socket_page_size_bytes);
            }
            size_t fabric_arg_idx = sampling_op.persistent_fabric_arg_idx;
            sampling_op.send_persistent_next_iter_inc_via_fabric_brisc(sampling_args, fabric_arg_idx);
        }
#endif

#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::sampling_is_final_core && !Core::skip_ccl) {
            auto bcast_turn_sem_noc_addr = get_noc_addr(
                Core::fabric_gate_bcast_noc_x,
                Core::fabric_gate_bcast_noc_y,
                get_semaphore(Core::fabric_gate_bcast_turn_semaphore_id));
            noc_semaphore_inc(bcast_turn_sem_noc_addr, 1);
            noc_async_atomic_barrier();
        }
#endif
    }
    mcast.teardown(mcast_args);
}
