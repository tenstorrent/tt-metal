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
//   CB 19 (eh_gather_dst):[MTP] EH output gather destination on argmax_final_core (tensor-backed)
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
#include "../../../unified_kernels/argmax.hpp"
#include "../../../unified_kernels/socket_send.hpp"
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
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;
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
    static constexpr bool has_bypass_socket_output = get_named_compile_time_arg_val("has_bypass_socket_output") == 1;
    static constexpr bool has_bypass_socket_input = get_named_compile_time_arg_val("has_bypass_socket_input") == 1;
    static_assert(input_socket_mode != 1, "lm_head_sampling input socket mode=1 is invalid");

    // ── MTP (Multi-Token Prediction) ────────────────────────────────
    static constexpr bool enable_mtp = get_named_compile_time_arg_val("is_mtp_base_stage") == 1;
    static constexpr bool enable_mtp_verification = get_named_compile_time_arg_val("is_mtp_verify_stage") == 1;
    static constexpr bool is_eh_matmul_core = enable_mtp && get_named_compile_time_arg_val("is_eh_matmul_core") == 1;
    static constexpr bool gather_use_per_core_sender_idx =
        get_named_compile_time_arg_val("gather_use_per_core_sender_idx") == 1;

    // -- MTP Stage Identification --------------------------------------------
    static constexpr bool is_base_stage = !enable_mtp_verification;
    static constexpr bool is_spec_stage = enable_mtp_verification;

    // ── Verify stage metadata transfer ───────────────────────────────
    static constexpr bool is_exit_device = get_named_compile_time_arg_val("is_exit_device") == 1;
    static constexpr uint32_t bcast_activation_size_bytes =
        get_named_compile_time_arg_val("bcast_activation_size_bytes");
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
    //   │ MTP (after argmax)   │   5   │ enable_mtp: noc_x, noc_y,         │
    //   │                      │       │ mtp_token_l1_addr, embedding_dram   │
    //   │ Bypass send          │   2   │ has_bypass_socket_output            │
    //   │ Verification         │   4   │ enable_mtp_verification             │
    //   │ Bypass recv          │   1   │ has_bypass_socket_input             │
    //   │ Fabric routing       │  var  │ !skip_ccl (per-core appended)       │
    //   └──────────────────────┴───────┴─────────────────────────────────────┘
    // ========================================================================
    uint32_t ncrisc_rt_arg_idx = 0;

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

    uint32_t mtp_embedding_base = 0;
    uint32_t mtp_token_addr = 0;
    uint32_t mtp_input_core_noc_x = 0;
    uint32_t mtp_input_core_noc_y = 0;
    uint32_t mtp_argmax_output_addr = 0;
    if constexpr (Core::enable_mtp) {
        mtp_embedding_base = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_token_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_input_core_noc_x = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_input_core_noc_y = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        mtp_argmax_output_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    uint32_t verify_output_staging_addr = 0;
    uint32_t verify_bcast_buffer_addr = 0;
    if constexpr (Core::enable_mtp_verification) {
        verify_output_staging_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
        verify_bcast_buffer_addr = get_common_arg_val<uint32_t>(ncrisc_rt_arg_idx++);
    }

    // ── Sharded buffer setup (registers tensor-backed CBs before main loop) ──
    //   input_core:  CB 0 (rmsnorm_input), CB 7 (rmsnorm_gamma)
    //                CB 11 (h_gamma), CB 12 (e_gamma)  [MTP only]
    //   matmul_core: CB 2 (matmul_in1 / vocab weights)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
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

    // ── Output gather sender args
    deepseek_b1_ops::Gather::SenderArgs eh_logits_gather_args{
        get_named_compile_time_arg_val("gather_dest_noc_x"),
        get_named_compile_time_arg_val("gather_dest_noc_y"),
        get_named_compile_time_arg_val("gather_data_size_bytes"),
        get_semaphore(get_named_compile_time_arg_val("gather_receiver_semaphore_id")),
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
    //   │ MTP args (base+mtp)  │ [13..16]│ input_core xy, token/argmax addr  │
    //   │ Verify staging (spec)│   [13]  │ verify_output_staging_addr        │
    //   │ Mcast dst override   │ [13/14/17] │ depends on stage config        │
    //   │ MTP mcast override   │   [18]  │ tensor-backed CB 18 addr (0=CB)   │
    //   │ Fabric routing       │  [N+]   │ per-core appended                 │
    //   └──────────────────────┴─────────┴────────────────────────────────────┘
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
    constexpr uint32_t mcast_dst_override_idx = Core::enable_mtp ? 17 : (Core::enable_mtp_verification ? 14 : 13);
    const uint32_t mcast_dst_addr_override = get_common_arg_val<uint32_t>(mcast_dst_override_idx);

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
        get_named_compile_time_arg_val("argmax_socket_page_size_bytes"),
        get_named_compile_time_arg_val("argmax_defer_socket_output")>;

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

    // ── MTP runtime args (all cores, enable_mtp) ────────────────────
    uint32_t mtp_input_core_noc_x = 0;
    uint32_t mtp_input_core_noc_y = 0;
    uint32_t mtp_token_addr = 0;
    uint32_t mtp_argmax_output_addr = 0;
    if constexpr (Core::enable_mtp) {
        mtp_input_core_noc_x = get_common_arg_val<uint32_t>(13);
        mtp_input_core_noc_y = get_common_arg_val<uint32_t>(14);
        mtp_token_addr = get_common_arg_val<uint32_t>(15);
        mtp_argmax_output_addr = get_common_arg_val<uint32_t>(16);
    }

    // ── Verify stage runtime arg (BRISC, for reading base token metadata)
    uint32_t brisc_verify_output_staging_addr = 0;
    if constexpr (Core::enable_mtp_verification) {
        brisc_verify_output_staging_addr = get_common_arg_val<uint32_t>(13);
    }

    // ── MTP: EH mcast sender (input_core, enable_mtp) ──────────────
    using McastEhCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_eh_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid") == 1,
        false>;
    constexpr uint32_t mcast_eh_src_cb = get_named_compile_time_arg_val("mcast_eh_src_cb");
    constexpr uint32_t mcast_eh_dst_cb = get_named_compile_time_arg_val("mcast_eh_dst_cb");
    const uint32_t mcast_eh_dst_addr_override = get_common_arg_val<uint32_t>(18);
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

    // Output gather receiver args
    deepseek_b1_ops::Gather::ReceiverArgs eh_logits_gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_semaphore(get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
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
    deepseek_b1_ops::Gather::ComputeArgs eh_logits_gather_args{};

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

    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, true, false> matmul;
    deepseek_b1_ops::Sampling::
        Op<ArgmaxCTArgs, Core::is_matmul_core, Core::is_argmax_final_core, Core::is_argmax_mesh_sender_core>
            sampling_op;

    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_input_core, Core::is_mcast_receiver_core, Core::is_mcast_receiver_core, true>
            mcast;

    uint32_t iteration_count = 0;
    constexpr uint32_t SENTINEL = 0xFFFFFFFF;
    constexpr uint32_t TOKEN_TYPE_BASE = 0;
    constexpr uint32_t TOKEN_TYPE_SPEC = 1;

    // Pack up to 2 tokens into a single TOKEN_META page (64 bytes) in the given CB.
    // Layout: [num_tokens, tok0_id, tok0_type, tok0_pos, tok1_id, tok1_type, tok1_pos, ...]
    auto write_token_metadata_to_socket_cb = [](uint32_t cb,
                                                uint32_t num_tokens,
                                                uint32_t tok0_id,
                                                uint32_t tok0_type,
                                                uint32_t tok0_pos,
                                                uint32_t tok1_id = 0,
                                                uint32_t tok1_type = 0,
                                                uint32_t tok1_pos = 0) {
        cb_reserve_back(cb, 1);
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
        page[0] = num_tokens;
        page[1] = tok0_id;
        page[2] = tok0_type;
        page[3] = tok0_pos;
        page[4] = tok1_id;
        page[5] = tok1_type;
        page[6] = tok1_pos;
        cb_push_back(cb, 1);
    };

    // ====================================================================
    // LM HEAD SAMPLING Lambda
    // ====================================================================
    auto lm_head_sampling =
        [&]() {
            // ====================================================================
            // Phase 0: CCL Broadcast (multi-device only) (NCRISC only)
            // ====================================================================
            if constexpr (!Core::skip_ccl || Core::bcast_use_socket_input) {

#if defined(COMPILE_FOR_BRISC)
                constexpr bool is_sender = get_named_compile_time_arg_val("bcast_is_sender") == 1;
                if constexpr (Core::persistent_mode && is_sender && Core::is_input_core) {
                    auto next_iteration_semaphore =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(persistent_next_iter_global_sem_addr);
                    // DPRINT << "BRISC argmax final core wait for next iteration semaphore" << ENDL();
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
            // On the broadcast sender, BRISC already pushes to CB 0 during the broadcast
            // (reserve_back + socket read into write_ptr + push_back). Do not push again
            // or we leave a stale tile that rmsnorm reads on the next iteration (wrong tokens).
            if constexpr (Core::is_input_core && (!Core::skip_ccl || !Core::bcast_use_socket_input)) {
                constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
                constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
                unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
            }

            // [Verify stage] Transfer 64-byte metadata from input_core to argmax_final_core.
            // Metadata sits at verify_bcast_buffer + activation_size in L1.
            // For skip_ccl, BRISC's socket read may still be in flight — wait for
            // the activation pages to appear in CB 0 (implies DMA completed).
            // For non-skip_ccl, the broadcast writer already completed above.
            if constexpr (Core::is_spec_stage && Core::is_input_core && Core::is_exit_device) {
                if constexpr (Core::skip_ccl && Core::bcast_use_socket_input) {
                    constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
                    constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
                    cb_wait_front(rmsnorm_input_cb, rmsnorm_num_tiles);
                }
                constexpr uint32_t argmax_noc_x = get_named_compile_time_arg_val("argmax_core_noc_x");
                constexpr uint32_t argmax_noc_y = get_named_compile_time_arg_val("argmax_core_noc_y");
                constexpr uint32_t metadata_size = 64;
                uint32_t metadata_src = verify_bcast_buffer_addr + Core::bcast_activation_size_bytes;
                uint64_t dst = get_noc_addr(argmax_noc_x, argmax_noc_y, verify_output_staging_addr);
                noc_async_write(metadata_src, dst, metadata_size);
                noc_async_write_barrier();
            }
#endif

            {
                DeviceZoneScopedN("RMSNORM");
                deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_rmsnorm_core, !Core::enable_mtp> rmsnorm;
                rmsnorm(rmsnorm_args);
            }

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
            // Phase 3: Argmax Sampling
            // ====================================================================
            {
                DeviceZoneScopedN("ARGMAX");
                sampling_op(sampling_args);
            }
        }

    // ====================================================================
    // MTP Lambda
    // ====================================================================
    auto mtp =
        [&]() {
    // ====================================================================
    // [MTP] Token transfer + Embedding lookup + e_rmsnorm + EH matmul
    // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::is_argmax_final_core) {
                uint64_t dst = get_noc_addr(mtp_input_core_noc_x, mtp_input_core_noc_y, mtp_token_addr);
                noc_async_write(mtp_argmax_output_addr, dst, 4);
                noc_async_write_barrier();
                uint64_t sem_addr = get_noc_addr(
                    mtp_input_core_noc_x,
                    mtp_input_core_noc_y,
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_inc(sem_addr, 1);
            }
#endif

#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // [MTP] Token transfer: argmax_final_core writes token to input_core
            //
            // Runtime args (input_core_noc_x/y, mtp_token_addr, embedding_base)
            // were pre-consumed before the loop.
            // ================================================================
            if constexpr (Core::is_input_core) {
                volatile tt_l1_ptr uint32_t* mtp_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_semaphore(get_named_compile_time_arg_val("mtp_ready_semaphore_id")));
                noc_semaphore_wait(mtp_ready_sem, 1);
                noc_semaphore_set(mtp_ready_sem, 0);
            }
#endif

        // ====================================================================
        // [MTP] Embedding lookup (NCRISC on input_core)
        // ====================================================================
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::is_input_core) {
                constexpr uint32_t embedding_size_bytes = get_named_compile_time_arg_val("embedding_size_bytes");
                constexpr uint32_t emb_cb = get_named_compile_time_arg_val("embedding_cb");
                constexpr uint32_t e_num_tiles = get_named_compile_time_arg_val("rmsnorm_e_num_tiles");
                const InterleavedAddrGen<true> embedding_addr_gen = {
                    .bank_base_address = mtp_embedding_base,
                    .page_size = embedding_size_bytes,
                };
                invalidate_l1_cache();
                uint32_t token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mtp_token_addr);
                cb_reserve_back(emb_cb, e_num_tiles);
                uint64_t dram_addr = embedding_addr_gen.get_noc_addr(token_id);
                noc_async_read(dram_addr, get_write_ptr(emb_cb), embedding_size_bytes);
                noc_async_read_barrier();
                cb_push_back(emb_cb, e_num_tiles);
            }
#endif

        // ====================================================================
        // [MTP] h_rmsnorm on TRISC — starts immediately after the LM head matmul,
        // overlapping with argmax on NCRISC/BRISC. CB 0 still has hidden states
        // (first RMSNorm used pop_input=false when MTP is enabled).
        // Output writes directly to mcast_eh_src_cb (first half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
            if constexpr (Core::is_rmsnorm_core) {
                {
                    deepseek_b1_ops::RMSNorm::Op<HRMSNormCTArgs, Core::is_rmsnorm_core, true> h_rmsnorm;
                    DeviceZoneScopedN("MTP_H_RMSNORM");
                    h_rmsnorm(rmsnorm_args);
                    DPRINT << "hD" << ENDL();
                }
            }
#endif

        // ====================================================================
        // [MTP] e_rmsnorm on TRISC (after embedding arrives in CB)
        // Output writes directly to mcast_eh_src_cb (second half of concat buffer).
        // ====================================================================
#if defined(COMPILE_FOR_TRISC)
            if constexpr (Core::is_rmsnorm_core) {
                {
                    DeviceZoneScopedN("MTP_E_RMSNORM");
                    deepseek_b1_ops::RMSNorm::Op<ERMSNormCTArgs, Core::is_rmsnorm_core, true> e_rmsnorm;
                    e_rmsnorm(rmsnorm_args);
                }
            }
#endif

            // ====================================================================
            // [MTP] Second mcast — multicast [h_norm|e_norm] from sender to all cores
            // ====================================================================
            {
                deepseek_b1_ops::Mcast::Op<
                    McastEhCTArgs,
                    Core::enable_mtp && Core::is_input_core,
                    Core::enable_mtp && Core::is_mcast_receiver_core,
                    Core::enable_mtp && Core::is_eh_matmul_core,
                    true>
                    mcast_eh;
                DeviceZoneScopedN("MTP_EH_MCAST");
                mcast_eh(mcast_eh_args);
            }

        // ====================================================================
        // [MTP] EH matmul using DRAM streaming
        // ====================================================================
#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::is_eh_matmul_core) {
                deepseek_b1_ops::DRAMStreamingMatmul::Op<EHDRAMMMCTArgs, true, true, false, 0, false, false, 2>
                    eh_matmul;
                {
                    DeviceZoneScopedN("MTP_EH_DRAM_MATMUL");
                    eh_matmul();
                }
            }
#endif

            // ====================================================================
            // [MTP] EH logits output gather
            // ====================================================================
            {
                DeviceZoneScopedN("OUTPUT_GATHER");
                deepseek_b1_ops::Gather::Op<
                    Core::is_eh_matmul_core,
                    Core::is_argmax_final_core,
                    /*pop_src=*/true,
                    Core::gather_use_per_core_sender_idx>
                    eh_logits_gather;
                eh_logits_gather(eh_logits_gather_args);
            }

#if defined(COMPILE_FOR_BRISC)
            if constexpr (Core::is_argmax_final_core) {
                constexpr uint32_t eh_gather_dst_cb = get_named_compile_time_arg_val("gather_dst_cb");
                constexpr uint32_t argmax_socket_cb = get_named_compile_time_arg_val("argmax_socket_cb");
                cb_wait_front(argmax_socket_cb, 1);
                uint32_t token_id = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(argmax_socket_cb));
                write_token_metadata_to_socket_cb(eh_gather_dst_cb, 1, token_id, TOKEN_TYPE_BASE, 0);
                cb_pop_front(argmax_socket_cb, 1);
            }
#endif
        }

    // ========================================================================
    // update_speculative_state: runs on BRISC for the spec stage (argmax_final_core).
    //
    // Must run on BRISC (not NCRISC) to avoid racing with the socket send section
    // for CB 6. After lm_head_sampling(), the argmax writer (BRISC) has already pushed
    // the speculative token to CB 6. This function consumes that page, reads the base
    // token from metadata L1 (transferred by NCRISC during the broadcast phase), and
    // writes a TOKEN_META page with both tokens back to CB 6.
    //
    // Metadata layout from base stage (at verify_output_staging_addr):
    //   [0] = num_tokens, [1] = tok0_id, [2] = tok0_type, [3] = tok0_pos, ...
    // ========================================================================
    auto update_speculative_state = [&]() {
#if defined(COMPILE_FOR_BRISC)
        if constexpr (
            Core::is_argmax_final_core && ArgmaxCTArgs::defer_socket_output && ArgmaxCTArgs::socket_mode != 0) {
            constexpr uint32_t argmax_socket_cb = ArgmaxCTArgs::socket_cb_id;

            cb_wait_front(argmax_socket_cb, 1);
            uint32_t speculative_token =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(argmax_socket_cb));
            cb_pop_front(argmax_socket_cb, 1);

            auto meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(brisc_verify_output_staging_addr);
            uint32_t base_token = meta[1];

            DeviceZoneScopedN("MTP_VERIFY_SEND");
            write_token_metadata_to_socket_cb(
                argmax_socket_cb, 2, base_token, TOKEN_TYPE_BASE, 0, speculative_token, TOKEN_TYPE_SPEC, 1);
        }
#endif
    };

    // Initialize Mcast
    mcast.init(mcast_args);

    // Persistent loop
    while (true) {
        iteration_count++;

        // Base Stage: run LM head sampling and MTP ops
        if constexpr (Core::is_base_stage) {
            // Stage Input: [1, 7168] activations
            // Stage Output: [1, 7168] activations + 64 bytes metadata
            lm_head_sampling();
            // if MTP is enabled, we run MTP ops to produce logits for next MTP decoder stage
            if constexpr (Core::enable_mtp) {
                mtp();
            }
        }

        // Spec Stage: run LM head sampling and update speculative state
        if constexpr (Core::is_spec_stage) {
            // Stage Input: [1, 7168] activations + 64 bytes metadata
            // Stage Output: 64 bytes metadata
            lm_head_sampling();
            update_speculative_state();
        }

        // ====================================================================
        // Socket Send and Signal
        // ====================================================================
#if defined(COMPILE_FOR_BRISC)
        if constexpr (Core::is_argmax_final_core && ArgmaxCTArgs::defer_socket_output && Core::persistent_mode) {
            if constexpr (Core::is_base_stage) {
                if constexpr (Core::enable_mtp) {
                    // Base stage with MTP: send ACTIVATION_W_TOKEN_META from gather CB
                    constexpr uint32_t eh_gather_dst_cb = get_named_compile_time_arg_val("gather_dst_cb");
                    constexpr uint32_t eh_gather_num_pages = get_named_compile_time_arg_val("gather_dst_num_pages") + 1;
                    constexpr uint32_t eh_gather_total_bytes =
                        get_named_compile_time_arg_val("gather_send_total_bytes");
                    unified_kernels::socket_send_from_cb<ArgmaxCTArgs::socket_mode>(
                        sampling_args.socket_config_addr, eh_gather_dst_cb, eh_gather_num_pages, eh_gather_total_bytes);
                } else {
                    // Base stage, no MTP: forward the token from argmax socket CB
                    unified_kernels::socket_send_from_cb<ArgmaxCTArgs::socket_mode>(
                        sampling_args.socket_config_addr,
                        ArgmaxCTArgs::socket_cb_id,
                        1,
                        ArgmaxCTArgs::socket_page_size_bytes);
                }

            } else if constexpr (Core::is_spec_stage) {
                // Spec stage: send TOKEN_META from argmax socket CB
                // (1 page written by update_speculative_state via write_token_metadata_to_socket_cb)
                unified_kernels::socket_send_from_cb<ArgmaxCTArgs::socket_mode>(
                    sampling_args.socket_config_addr,
                    ArgmaxCTArgs::socket_cb_id,
                    1,
                    ArgmaxCTArgs::socket_page_size_bytes);
            }
            // Persistent next iteration semaphore increment
            size_t fabric_arg_idx = sampling_op.persistent_fabric_arg_idx;
            sampling_op.send_persistent_next_iter_inc_via_fabric_brisc(sampling_args, fabric_arg_idx);
        }
#endif

        // ====================================================================
        // Phase 4: Persistent mode (if enabled)
        // ====================================================================
        if constexpr (!Core::persistent_mode) {
            break;
        }
    }
    mcast.teardown();
}
