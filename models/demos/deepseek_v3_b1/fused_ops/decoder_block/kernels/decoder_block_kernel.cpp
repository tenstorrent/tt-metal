// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused Decoder Block kernel: AttentionBlock (pre_sdpa + post_sdpa) + MoE (routed + shared expert)
// Single kernel file, compiles correctly for all RISC cores.
//
// Phase 1: AttentionBlock — CCL Broadcast + RMSNorm + Matmul + Gather + RMSNorm2 + Mcast2 + Matmul2
//          + Matmul3 + RoPE + CreateQHeads + KV Cache + Flash MLA + Post-SDPA (Reduce + Matmul4/5 + Gather + CCL)
// Phase 2: CB Reconfiguration (attention_block layout → MOE layout)
// Phase 3: MoE — Residual Mcast + RMSNorm + Input Mcast + Gate + TopK + Routing + DRAM Matmuls + Reduce-to-One

// === AttentionBlock includes ===
#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/rmsnorm.hpp"
#include "../../../unified_kernels/mcast.hpp"
#include "../../../unified_kernels/matmul.hpp"
#include "../../../unified_kernels/gather.hpp"
#include "../../../unified_kernels/gather_reduce.hpp"
#include "../../../unified_kernels/kn_sliced_matmul.hpp"
#include "../../../unified_kernels/create_q_heads.hpp"
#include "../../../unified_kernels/rope.hpp"
#include "../../../unified_kernels/broadcast.hpp"
#include "../../../unified_kernels/kv_cache_update.hpp"
#include "../../../unified_kernels/flash_mla.hpp"

#include "../../../micro_ops/flash_mla/kernels/rt_args_common.hpp"

#include "../../../unified_kernels/sdpa_reduce_worker.hpp"
#include "../../../unified_kernels/sdpa_reduce_forwarder.hpp"
#include "../../../unified_kernels/all_reduce_sender.hpp"
#include "../../../unified_kernels/all_reduce_receiver.hpp"

// === MoE-specific includes ===
#include "../../../unified_kernels/moe_gather.hpp"
#if defined(COMPILE_FOR_TRISC)
#undef REDUCE_OP
#undef REDUCE_DIM
#endif
#include "../../../unified_kernels/deepseek_moe_gate.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"
#include "../../../unified_kernels/eltwise_mul.hpp"
#include "../../../unified_kernels/eltwise_add.hpp"
#include "../../../unified_kernels/gated_reduce.hpp"
#include "../../../unified_kernels/residual_add.hpp"
#ifdef ENABLE_REDUCE_TO_ONE
#include "../../../unified_kernels/reduce_to_one_b1.hpp"
#endif

// Compile-time role flags for dead code elimination via if constexpr.
// Merged from AttentionBlock and MoE kernels.
struct Core {
    // === AttentionBlock roles ===
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_matmul2_core = get_named_compile_time_arg_val("is_matmul2_core") == 1;
    static constexpr bool is_qnope_core = get_named_compile_time_arg_val("is_qnope_core") == 1;
    static constexpr bool is_qrope_core = get_named_compile_time_arg_val("is_qrope_core") == 1;
    static constexpr bool is_sdpa_input_core = get_named_compile_time_arg_val("is_sdpa_input_core") == 1;
    static constexpr bool is_dkv_matmul_core = get_named_compile_time_arg_val("is_dkv_matmul_core") == 1;
    static constexpr bool is_kv_rmsnorm_core = get_named_compile_time_arg_val("is_kv_rmsnorm_core") == 1;
    static constexpr bool is_knope_core = get_named_compile_time_arg_val("is_knope_core") == 1;
    static constexpr bool is_krope_core = get_named_compile_time_arg_val("is_krope_core") == 1;
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;

    // MLA
    static constexpr bool is_mla_core = get_named_compile_time_arg_val("is_mla_core") == 1;

    // Sequence Parallel configs
    static constexpr uint32_t kv_cache_device_chunk_size = get_named_compile_time_arg_val("kv_cache_device_chunk_size");
    static constexpr uint32_t kv_cache_sp_device_idx = get_named_compile_time_arg_val("kv_cache_sp_device_idx");
    static constexpr uint32_t kv_cache_num_sp_devices = get_named_compile_time_arg_val("kv_cache_num_sp_devices");

    // Post-SDPA roles
    static constexpr bool is_sdpa_worker_core = get_named_compile_time_arg_val("is_sdpa_worker_core") == 1;
    static constexpr bool is_sdpa_forwarder_core = get_named_compile_time_arg_val("is_sdpa_forwarder_core") == 1;
    static constexpr bool is_matmul4_core = get_named_compile_time_arg_val("is_matmul4_core") == 1;
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    static constexpr bool is_mcast3_receiver_core = get_named_compile_time_arg_val("is_mcast3_receiver_core") == 1;
    static constexpr bool is_matmul5_core = get_named_compile_time_arg_val("is_matmul5_core") == 1;
    static constexpr bool is_ccl_sender_core = get_named_compile_time_arg_val("is_ccl_sender_core") == 1;
    static constexpr bool is_ccl_receiver_core = get_named_compile_time_arg_val("is_ccl_receiver_core") == 1;

    // === MoE roles ===
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;

    struct Routed {
        static constexpr bool is_gate_mm_core = get_named_compile_time_arg_val("is_gate_mm_core") == 1;
        static constexpr bool is_gate_proj_core = get_named_compile_time_arg_val("is_gate_proj_core") == 1;
    };
    struct Shared {
        static constexpr bool is_compute_core = get_named_compile_time_arg_val("is_shared_compute_core") == 1;
        static constexpr bool is_gate_compute_core = get_named_compile_time_arg_val("is_shared_gate_compute_core") == 1;
        static constexpr bool is_up_compute_core = get_named_compile_time_arg_val("is_shared_up_compute_core") == 1;
        static constexpr bool is_gated_reduce_core = get_named_compile_time_arg_val("is_shared_gated_reduce_core") == 1;
        static constexpr bool is_mcast_receiver_core =
            get_named_compile_time_arg_val("is_shared_mcast_receiver_core") == 1;
    };
    static constexpr bool is_input_mcast_receiver =
        Routed::is_gate_mm_core || Routed::is_gate_proj_core || Shared::is_compute_core;

    static constexpr bool is_reduce_worker_core = get_named_compile_time_arg_val("is_reduce_worker_core") == 1;
    static constexpr bool is_reduce_fabric_core = get_named_compile_time_arg_val("is_reduce_fabric_core") == 1;
};

void kernel_main() {
    DPRINT << "DECODER BLOCK KERNEL MAIN" << ENDL();
    // ============================================================================
    // NCRISC (Reader + Mcast Receiver) - ReaderConfigDescriptor compiles as NCRISC
    // Named compile-time args: rmsnorm reader, mcast receiver, matmul reader, gather sender
    // Runtime args: []
    // ============================================================================
    uint32_t per_core_rta_arg_idx = 0;
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type aliases (required for Op templates)
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

    deepseek_b1_ops::Broadcast::WriterArgs bcast_args{};

    if constexpr (!Core::skip_ccl && Core::is_input_core) {
        DPRINT << " BCAST ARGS " << per_core_rta_arg_idx << ENDL();
        uint32_t offset_fabric_args = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
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
            per_core_rta_arg_idx,
        };
        per_core_rta_arg_idx += offset_fabric_args;
        DPRINT << " BCAST ARGS AFTER " << per_core_rta_arg_idx << ENDL();
    }

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

    // Mcast receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // Matmul CTArgs type alias (NCRISC uses ReaderCTArgs)
    using MatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ReaderCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    using Matmul3CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Matmul reader args (NCRISC is no-op)
    deepseek_b1_ops::KNSlicedMatmul::ReaderArgs matmul_args{};

    // Gather sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::GatherReduce::SenderArgs gather_reduce_args{
        get_named_compile_time_arg_val("gather_reduce_dest_noc_x"),
        get_named_compile_time_arg_val("gather_reduce_dest_noc_y"),
        get_named_compile_time_arg_val("gather_reduce_data_size_bytes"),
        get_named_compile_time_arg_val("gather_reduce_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("gather_reduce_src_cb"),
        get_named_compile_time_arg_val("gather_reduce_src_num_pages"),
        get_named_compile_time_arg_val("gather_reduce_grid_start_x"),
        get_named_compile_time_arg_val("gather_reduce_grid_start_y"),
        get_named_compile_time_arg_val("gather_reduce_grid_end_x"),
        get_named_compile_time_arg_val("gather_reduce_grid_end_y"),
        get_named_compile_time_arg_val("gather_reduce_half_num_cores"),
        get_named_compile_time_arg_val("gather_reduce_half0_cb_id"),
        get_named_compile_time_arg_val("gather_reduce_half1_cb_id"),
    };

    // RMSNorm2 reader args
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm2_args{};

    // Matmul2 reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul2_args{};

    // Mcast2 receiver args (for matmul2 cores to receive matmul2 input from input core)
    // Uses same semaphore as first mcast
    deepseek_b1_ops::Mcast::ReceiverArgs mcast2_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("matmul2_in0"),
        get_named_compile_time_arg_val("mcast2_dst_num_pages"),
    };

    // Matmul3 reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul3_args{};

    // Qrope CTArgs type alias (NCRISC uses ReaderCTArgs)
    using QRopeCTArgs = deepseek_b1_ops::Rope::ReaderCTArgs<
        get_named_compile_time_arg_val("qrope_Wt"),
        get_named_compile_time_arg_val("qrope_Ht"),
        get_named_compile_time_arg_val("qrope_cos_sin_page_size"),
        get_named_compile_time_arg_val("qrope_total_Wt"),
        get_named_compile_time_arg_val("qrope_start_tile_offset")>;

    deepseek_b1_ops::Rope::ReaderArgs qrope_args{
        .in_cb = get_named_compile_time_arg_val("qrope_in_cb"),
        .cos_cb = get_named_compile_time_arg_val("qrope_cos_cb"),
        .sin_cb = get_named_compile_time_arg_val("qrope_sin_cb"),
        .cos_tensor_address = get_named_compile_time_arg_val("qrope_cos_tensor_address"),
        .sin_tensor_address = get_named_compile_time_arg_val("qrope_sin_tensor_address"),
        .position_ids_tensor_address = get_named_compile_time_arg_val("qrope_position_ids_tensor_address"),
        .trans_mat_cb = get_named_compile_time_arg_val("qrope_trans_mat_cb"),
    };

    // NCRISC: Sender args for QNOPE/QROPE cores
    // Senders write to intermediate CB, then compute tilizes to output CB
    // 3-phase synchronization: nope_phase1, nope_phase2, rope semaphores
    constexpr uint32_t cqh_receiver_in_cb = get_named_compile_time_arg_val("cqh_receiver_in_cb");
    using CreateQHeadsCTArgs = deepseek_b1_ops::CreateQHeads::SenderCTArgs<
        get_named_compile_time_arg_val("cqh_qnope_data_size_bytes"),
        get_named_compile_time_arg_val("cqh_qrope_head_size_bytes")>;
    deepseek_b1_ops::CreateQHeads::SenderArgs create_q_heads_args{
        0,  // sender_grid_start_x (logical 0)
        0,  // sender_grid_start_y (logical 0)
        get_named_compile_time_arg_val("cqh_head_stride_bytes"),
        get_named_compile_time_arg_val("cqh_qnope_cols"),
        get_named_compile_time_arg_val("cqh_qnope_src_cb"),
        get_named_compile_time_arg_val("cqh_qrope_src_cb"),
        Core::is_qnope_core ? get_named_compile_time_arg_val("cqh_qnope_src_num_pages")
                            : get_named_compile_time_arg_val("cqh_qrope_src_num_pages"),
        get_named_compile_time_arg_val("cqh_nope_phase1_semaphore_addr"),
        get_named_compile_time_arg_val("cqh_nope_phase2_semaphore_addr"),
        get_named_compile_time_arg_val("cqh_rope_semaphore_addr"),
        {
            get_named_compile_time_arg_val("cqh_target_noc_coords_row0"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row1"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row2"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row3"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row4"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row5"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row6"),
            get_named_compile_time_arg_val("cqh_target_noc_coords_row7"),
        },
        get_write_ptr(cqh_receiver_in_cb),
    };

    // Matmul CTArgs type alias (NCRISC uses ReaderCTArgs)
    using DKV_MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Matmul reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs dkv_matmul_args{};

    // Gather sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::SenderArgs dkv_gather_args{
        get_named_compile_time_arg_val("dkv_gather_dest_noc_x"),
        get_named_compile_time_arg_val("dkv_gather_dest_noc_y"),
        get_named_compile_time_arg_val("dkv_gather_data_size_bytes"),
        get_named_compile_time_arg_val("dkv_gather_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("dkv_gather_src_cb"),
        get_named_compile_time_arg_val("dkv_gather_src_num_pages"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_start_x"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_start_y"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_end_x"),
        get_named_compile_time_arg_val("dkv_gather_sender_grid_end_y"),
        get_named_compile_time_arg_val("dkv_gather_row_major"),
        get_write_ptr(get_named_compile_time_arg_val(
            "kv_rmsnorm_input_cb")),  // receiver_data_addr from CB write ptr (single-buffered)
    };

    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    // kv cache rmsnorm reader args
    deepseek_b1_ops::RMSNorm::ReaderArgs kv_rmsnorm_args{};

    using K_RopeCTArgs = deepseek_b1_ops::Rope::ReaderCTArgs<
        get_named_compile_time_arg_val("krope_Wt"),
        get_named_compile_time_arg_val("krope_Ht"),
        get_named_compile_time_arg_val("krope_cos_sin_page_size"),
        get_named_compile_time_arg_val("krope_total_Wt"),
        get_named_compile_time_arg_val("krope_start_tile_offset")>;

    deepseek_b1_ops::Rope::ReaderArgs krope_args{
        .in_cb = get_named_compile_time_arg_val("krope_in_cb"),
        .cos_cb = get_named_compile_time_arg_val("krope_cos_cb"),
        .sin_cb = get_named_compile_time_arg_val("krope_sin_cb"),
        .cos_tensor_address = get_named_compile_time_arg_val("krope_cos_tensor_address"),
        .sin_tensor_address = get_named_compile_time_arg_val("krope_sin_tensor_address"),
        .position_ids_tensor_address = get_named_compile_time_arg_val("krope_position_ids_tensor_address"),
        .trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb"),
    };

    deepseek_b1_ops::KVCacheUpdate::ReaderArgs kv_cache_update_args{};

    deepseek_b1_ops::FlashMLADecode::ReaderArgs flash_mla_args;
    if constexpr (Core::is_mla_core) {
        flash_mla_args = {
            .k_addr = get_common_arg_val<uint32_t>(13),
            .local_cur_pos = 0,  // set via flash_mla.set_local_cur_pos() below
            .cur_batch = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .core_num_in_reduce = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .is_mcast_sender = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .mcast_start_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .mcast_start_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .vc = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .St = get_named_compile_time_arg_val("St"),
            .DHt = get_named_compile_time_arg_val("DHt"),
            .Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t"),
            .num_cores_per_head = get_named_compile_time_arg_val("num_cores_per_head"),
            .k_chunk_size = get_named_compile_time_arg_val("k_chunk_size"),
            .mcast_semaphore_addr = get_named_compile_time_arg_val("mla_mcast_semaphore_addr"),
            .k_page_size = get_named_compile_time_arg_val("k_page_size"),
            .k_num_pages = get_named_compile_time_arg_val("k_num_pages"),
            .ncrisc_brisc_sync_semaphore_addr = get_named_compile_time_arg_val("mla_ncrisc_brisc_sync_semaphore_addr"),
            .receiver_ready_semaphore_addr = get_named_compile_time_arg_val("mla_receiver_ready_semaphore_addr"),
            .kv_cache_cur_pos_ready_semaphore_addr =
                get_named_compile_time_arg_val("mla_kv_cache_cur_pos_ready_semaphore_addr"),
            .kv_cache_cur_pos_ready_value = get_named_compile_time_arg_val("mla_kv_cache_cur_pos_ready_value"),
            .cb_k_in = get_named_compile_time_arg_val("mla_k_in_cb"),
        };
    }

    using FlashMLACTArgs = deepseek_b1_ops::FlashMLADecode::ReaderCTArgs;

    // Matmul4 CTArgs
    using Matmul4CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul4_args{};

    // Gather2 sender args (UsePerCoreSenderIdx: each core gets a contiguous index
    // via gather2_sender_idx, avoiding gaps from the non-rectangular kv_b2 grid)
    deepseek_b1_ops::Gather::SenderArgs gather2_args{
        get_named_compile_time_arg_val("gather2_dest_noc_x"),
        get_named_compile_time_arg_val("gather2_dest_noc_y"),
        get_named_compile_time_arg_val("gather2_data_size_bytes"),
        get_semaphore(get_named_compile_time_arg_val("gather2_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather2_src_cb"),
        get_named_compile_time_arg_val("gather2_src_num_pages"),
        get_named_compile_time_arg_val("gather2_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather2_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather2_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather2_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather2_row_major"),
        get_named_compile_time_arg_val("gather2_receiver_data_addr"),
        get_named_compile_time_arg_val("gather2_sender_idx"),
    };

    // Mcast3 receiver args
    using Mcast3CTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast3_args{
        get_semaphore(get_named_compile_time_arg_val("mcast3_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast3_dst_cb"),
        get_named_compile_time_arg_val("mcast3_dst_num_pages"),
    };

    // Matmul5 CTArgs
    using Matmul5CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs matmul5_args{};

    // Gather3 sender args (UsePerCoreSenderIdx: each core gets a contiguous index
    // via gather3_sender_idx, avoiding gaps from the non-rectangular o_proj grid)
    deepseek_b1_ops::Gather::SenderArgs gather3_args{
        get_named_compile_time_arg_val("gather3_dest_noc_x"),
        get_named_compile_time_arg_val("gather3_dest_noc_y"),
        get_named_compile_time_arg_val("gather3_data_size_bytes"),
        get_semaphore(get_named_compile_time_arg_val("gather3_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather3_src_cb"),
        get_named_compile_time_arg_val("gather3_src_num_pages"),
        get_named_compile_time_arg_val("gather3_sender_grid_start_x"),
        get_named_compile_time_arg_val("gather3_sender_grid_start_y"),
        get_named_compile_time_arg_val("gather3_sender_grid_end_x"),
        get_named_compile_time_arg_val("gather3_sender_grid_end_y"),
        get_named_compile_time_arg_val("gather3_row_major"),
        get_named_compile_time_arg_val("gather3_receiver_data_addr"),
        get_named_compile_time_arg_val("gather3_sender_idx"),
    };

    using SdpaReduceWorkerCTArgs = deepseek_b1_ops::SdpaReduceWorker::ReaderCTArgs<
        get_named_compile_time_arg_val("sdpa_cb_local_l"),
        get_named_compile_time_arg_val("sdpa_cb_local_ms"),
        get_named_compile_time_arg_val("sdpa_cb_neighbor_l"),
        get_named_compile_time_arg_val("sdpa_cb_neighbor_ms"),
        get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes"),
        get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes"),
        get_named_compile_time_arg_val("sdpa_num_l_chunks"),
        get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
        get_named_compile_time_arg_val("sdpa_position_enabled"),
        get_named_compile_time_arg_val("sdpa_per_device_chunk_size")>;

    deepseek_b1_ops::SdpaReduceWorker::ReaderArgs sdpa_reduce_worker_args;
    if constexpr (Core::is_sdpa_worker_core) {
        sdpa_reduce_worker_args = {
            .r1_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_recv_buffer_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_recv_buffer_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
        if constexpr (SdpaReduceWorkerCTArgs::position_enabled) {
            sdpa_reduce_worker_args.pos_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            sdpa_reduce_worker_args.r1_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            sdpa_reduce_worker_args.r2_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
            sdpa_reduce_worker_args.r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        }
    }

    using SdpaReduceForwarderCTArgs = deepseek_b1_ops::SdpaReduceForwarder::CTArgs<
        get_named_compile_time_arg_val("sdpa_fwd_slots_per_round"),
        get_named_compile_time_arg_val("sdpa_fwd_slot_size"),
        get_named_compile_time_arg_val("sdpa_fwd_r2_buffer_offset")>;

    deepseek_b1_ops::SdpaReduceForwarder::ForwarderArgs sdpa_reduce_forwarder_args;
    if constexpr (Core::is_sdpa_forwarder_core) {
        sdpa_reduce_forwarder_args = {
            .buffer_base = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .buffer_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
        };
        uint32_t fabric_args_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        sdpa_reduce_forwarder_args.rta_offset = per_core_rta_arg_idx;
        per_core_rta_arg_idx += fabric_args_offset;
        DPRINT << " SDPA REDUCE FORWARDER ARGS AFTER " << per_core_rta_arg_idx << ENDL();
    }

    // CCL Sender NCRISC CTArgs (reads from gather core)
    using CCLSenderReaderCTArgs = deepseek_b1_ops::AllReduceSender::ReaderCTArgs<
        get_named_compile_time_arg_val("ccl_sender_cb0_id"),
        get_named_compile_time_arg_val("ccl_sender_num_tiles"),
        get_named_compile_time_arg_val("ccl_sender_tensor_page_size"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_x"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_y")>;

    // CCL Receiver NCRISC CTArgs (waits for remote data)
    // Note: skip_local_push=1 because gather3 already pushed to CB7 (gather3_dst_cb)
    using CCLReceiverReaderCTArgs = deepseek_b1_ops::AllReduceReceiver::ReaderCTArgs<
        get_named_compile_time_arg_val("ccl_receiver_packet_header_cb_id"),
        get_named_compile_time_arg_val("ccl_receiver_cb_in1"),
        get_named_compile_time_arg_val("ccl_receiver_l1_alignment"),
        get_named_compile_time_arg_val("ccl_receiver_cb_in2"),
        get_named_compile_time_arg_val("ccl_receiver_remote_sender_noc_x"),
        get_named_compile_time_arg_val("ccl_receiver_remote_sender_noc_y"),
        get_named_compile_time_arg_val("ccl_receiver_num_standard_tiles"),
        get_named_compile_time_arg_val("ccl_receiver_cb_residual"),
        get_named_compile_time_arg_val("ccl_receiver_has_residual"),
        get_named_compile_time_arg_val("ccl_receiver_skip_local_push")>;

    // Dummy WriterCTArgs - not used by NCRISC but needed for Op template
    using DummyWriterCTArgs = deepseek_b1_ops::AllReduceSender::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
    // Dummy ComputeCTArgs - not used by NCRISC but needed for Op template
    using DummyComputeCTArgs = deepseek_b1_ops::AllReduceReceiver::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0>;
    deepseek_b1_ops::AllReduceSender::RTArgs ccl_sender_args{};
    deepseek_b1_ops::AllReduceReceiver::RTArgs ccl_receiver_args{};

    if constexpr (Core::is_ccl_sender_core) {
        DPRINT << " CCL SENDER ARGS " << per_core_rta_arg_idx << ENDL();
        ccl_sender_args = {
            .tensor_address = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
        DPRINT << " CCL SENDER ARGS AFTER " << per_core_rta_arg_idx << ENDL();
    }

    if constexpr (Core::is_ccl_receiver_core) {
        DPRINT << " CCL RECEIVER ARGS " << per_core_rta_arg_idx << ENDL();
        ccl_receiver_args = {
            .sender_semaphore_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
        DPRINT << " CCL RECEIVER ARGS AFTER " << per_core_rta_arg_idx << ENDL();
    }
    DPRINT << " PER CORE RTA ARG IDX FINAL " << per_core_rta_arg_idx << ENDL();
    // ========================================================================
    // MoE NCRISC args (nested in struct Moe to avoid naming conflicts)
    // ========================================================================
    struct Moe {
        struct Routed {
            using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
                get_named_compile_time_arg_val("moe_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("moe_mcast_dst_cb"),
                get_named_compile_time_arg_val("moe_mcast_dst_num_pages"),
            };

#ifdef ENABLE_ROUTING
            using GateMMCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
            deepseek_b1_ops::Matmul::ReaderArgs gate_mm_args{};

            deepseek_b1_ops::MoeGather::ReceiverArgs gather_args{
                get_named_compile_time_arg_val("gather_noc0_num_senders"),
                get_named_compile_time_arg_val("gather_noc1_num_senders"),
                get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("gather_dst_cb"),
                get_named_compile_time_arg_val("gather_dst_num_pages"),
            };

            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs;

            deepseek_b1_ops::Mcast::ReceiverArgs index_mcast_args{
                get_named_compile_time_arg_val("index_mcast_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_index"),
                get_named_compile_time_arg_val("index_mcast_num_pages"),
            };

            deepseek_b1_ops::Mcast::ReceiverArgs expert_scale_mcast_args{
                get_named_compile_time_arg_val("expert_scale_mcast_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("mul_cb_scalar_src"),
                get_named_compile_time_arg_val("expert_scale_mcast_num_pages"),
            };
#endif

            using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
                get_named_compile_time_arg_val("gate_proj_cb_in1"),
                get_named_compile_time_arg_val("gate_proj_cb_out"),
                get_named_compile_time_arg_val("gate_proj_in1_tensor_addr"),
                get_named_compile_time_arg_val("gate_proj_in1_page_size"),
                get_named_compile_time_arg_val("gate_proj_in1_num_pages"),
                get_named_compile_time_arg_val("gate_proj_subblock_k"),
                get_named_compile_time_arg_val("gate_proj_per_core_n"),
                get_named_compile_time_arg_val("gate_proj_in1_block_size_bytes"),
                get_named_compile_time_arg_val("gate_proj_out_num_tiles"),
                get_named_compile_time_arg_val("gate_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("gate_proj_bank_id"),
                get_named_compile_time_arg_val("gate_proj_vc"),
                get_named_compile_time_arg_val("enable_routing"),
                get_named_compile_time_arg_val("gate_proj_cb_index"),
                get_named_compile_time_arg_val("gate_proj_index_offset"),
                get_named_compile_time_arg_val("use_hardcoded_expert_index")>;

            using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
                get_named_compile_time_arg_val("up_proj_cb_in1"),
                get_named_compile_time_arg_val("up_proj_cb_mm_out"),
                get_named_compile_time_arg_val("up_proj_in1_tensor_addr"),
                get_named_compile_time_arg_val("up_proj_in1_page_size"),
                get_named_compile_time_arg_val("up_proj_in1_num_pages"),
                get_named_compile_time_arg_val("up_proj_subblock_k"),
                get_named_compile_time_arg_val("up_proj_per_core_n"),
                get_named_compile_time_arg_val("up_proj_in1_block_size_bytes"),
                get_named_compile_time_arg_val("up_proj_out_num_tiles"),
                get_named_compile_time_arg_val("up_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("up_proj_bank_id"),
                get_named_compile_time_arg_val("up_proj_vc"),
                get_named_compile_time_arg_val("enable_routing"),
                get_named_compile_time_arg_val("up_proj_cb_index"),
                get_named_compile_time_arg_val("up_proj_index_offset"),
                get_named_compile_time_arg_val("use_hardcoded_expert_index")>;

            using MulCTArgs = deepseek_b1_ops::EltwiseMul::ReaderCTArgs;

            deepseek_b1_ops::MoeGather::ReceiverArgs down_proj_gather_args{
                get_named_compile_time_arg_val("down_proj_gather_noc0_num_senders"),
                get_named_compile_time_arg_val("down_proj_gather_noc1_num_senders"),
                get_named_compile_time_arg_val("down_proj_gather_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_gather_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_gather_dst_cb"),
                get_named_compile_time_arg_val("down_proj_gather_dst_num_pages"),
            };

            deepseek_b1_ops::Mcast::ReceiverArgs down_proj_mcast_args{
                get_named_compile_time_arg_val("down_proj_mcast_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_mcast_dst_cb"),
                get_named_compile_time_arg_val("down_proj_mcast_dst_num_pages"),
            };

            using DownProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
                get_named_compile_time_arg_val("down_proj_cb_in1"),
                get_named_compile_time_arg_val("down_proj_cb_out"),
                get_named_compile_time_arg_val("down_proj_in1_tensor_addr"),
                get_named_compile_time_arg_val("down_proj_in1_page_size"),
                get_named_compile_time_arg_val("down_proj_in1_num_pages"),
                get_named_compile_time_arg_val("down_proj_subblock_k"),
                get_named_compile_time_arg_val("down_proj_per_core_n"),
                get_named_compile_time_arg_val("down_proj_in1_block_size_bytes"),
                get_named_compile_time_arg_val("down_proj_out_num_tiles"),
                get_named_compile_time_arg_val("down_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("down_proj_bank_id"),
                get_named_compile_time_arg_val("down_proj_vc"),
                get_named_compile_time_arg_val("enable_routing"),
                get_named_compile_time_arg_val("down_proj_cb_index"),
                get_named_compile_time_arg_val("down_proj_index_offset"),
                get_named_compile_time_arg_val("use_hardcoded_expert_index")>;

            using AddCTArgs = deepseek_b1_ops::EltwiseAdd::ReaderCTArgs;

            using ResidualMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs residual_mcast_args{
                get_named_compile_time_arg_val("shared_residual_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_residual_cb"),
                get_named_compile_time_arg_val("shared_residual_num_pages"),
            };

            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
            deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

#ifdef ENABLE_REDUCE_TO_ONE
            using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ReaderCTArgs<
                get_named_compile_time_arg_val("reduce_device_role"),
                get_named_compile_time_arg_val("reduce_num_tiles"),
                get_named_compile_time_arg_val("reduce_local_cb"),
                get_named_compile_time_arg_val("reduce_received_cb_r1"),
                get_named_compile_time_arg_val("reduce_received_cb_r2"),
                get_named_compile_time_arg_val("reduce_received_cb_r3"),
                get_named_compile_time_arg_val("is_reduce_fabric_core")>;

            deepseek_b1_ops::ReduceToOneB1::ReaderArgs reduce_rt_args{
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("reduce_ncrisc_common_rt_arg_base") + 0),
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("reduce_ncrisc_common_rt_arg_base") + 1),
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("reduce_ncrisc_common_rt_arg_base") + 2),
            };
#endif
        } routed;

        struct Shared {
            using GUMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ReaderCTArgs;
            deepseek_b1_ops::KNSlicedMatmul::ReaderArgs gu_matmul_args{};

            deepseek_b1_ops::MoeGather::ReceiverArgs ag_args{
                get_named_compile_time_arg_val("shared_ag_noc0_num_senders"),
                0,
                get_named_compile_time_arg_val("shared_ag_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_ag_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_ag_dst_cb"),
                get_named_compile_time_arg_val("shared_ag_dst_num_pages"),
            };

            deepseek_b1_ops::MoeGather::ReceiverArgs bg_args{
                get_named_compile_time_arg_val("shared_bg_noc0_num_senders"),
                0,
                get_named_compile_time_arg_val("shared_bg_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_bg_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_bg_dst_cb"),
                get_named_compile_time_arg_val("shared_bg_dst_num_pages"),
            };

            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ReaderCTArgs;
            deepseek_b1_ops::GatedReduce::ReaderArgs gated_reduce_args{};

            using DownMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs down_mcast_args{
                get_named_compile_time_arg_val("shared_down_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_down_mcast_dst_cb"),
                get_named_compile_time_arg_val("shared_down_mcast_dst_num_pages"),
            };

            using DownMatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
            deepseek_b1_ops::Matmul::ReaderArgs down_matmul_args{};

            using ResidualAddCTArgs = deepseek_b1_ops::ResidualAdd::ReaderCTArgs;
            deepseek_b1_ops::ResidualAdd::ReaderArgs residual_add_args{};

            deepseek_b1_ops::MoeGather::ReceiverArgs og_args{
                get_named_compile_time_arg_val("shared_og_noc0_num_senders"),
                get_named_compile_time_arg_val("shared_og_noc1_num_senders"),
                get_named_compile_time_arg_val("shared_og_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_og_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_og_dst_cb"),
                get_named_compile_time_arg_val("shared_og_dst_num_pages"),
            };

            using OutputMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs output_mcast_args{
                get_named_compile_time_arg_val("shared_output_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("add_cb_in1"),
                get_named_compile_time_arg_val("shared_output_mcast_dst_num_pages"),
            };
        } shared;
    } moe;

    auto setup_moe_sharded_buffers = [&]() {
        if constexpr (Core::is_sender_core) {
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("moe_rmsnorm_gamma_cb"),
                get_named_compile_time_arg_val("moe_rmsnorm_gamma_num_pages"));
#ifdef ENABLE_ROUTING
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("gate_bias_cb"), 1);
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("gate_input_indices_cb"), 1);
#endif
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("shared_residual_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages"));
        }
#ifdef ENABLE_ROUTING
        if constexpr (Core::Routed::is_gate_mm_core) {
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("gate_mm_in1"),
                get_named_compile_time_arg_val("gate_mm_k_num_tiles") *
                    get_named_compile_time_arg_val("gate_mm_out_w"));
        }
#endif
        if constexpr (Core::Routed::is_gate_proj_core) {
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("mul_cb_in1"), get_named_compile_time_arg_val("mul_num_tiles"));
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("add_cb_in0"), get_named_compile_time_arg_val("add_cb_in0_wait_tiles"));
        }
        if constexpr (Core::Shared::is_compute_core) {
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("shared_gu_weights_cb"),
                get_named_compile_time_arg_val("shared_gu_weights_num_pages"));
        }
        if constexpr (Core::Shared::is_mcast_receiver_core) {
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("shared_down_matmul_in1"),
                get_named_compile_time_arg_val("shared_down_matmul_k_num_tiles") *
                    get_named_compile_time_arg_val("shared_down_matmul_out_w_per_core"));
        }
    };

// ============================================================================
// BRISC (Writer + Mcast Sender) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: bcast writer + rmsnorm writer, mcast sender, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)

    // CCL Broadcast CTArgs type alias
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_num_pages_to_read"),
        get_named_compile_time_arg_val("bcast_is_sender")>;

    // CCL Broadcast reader runtime args (only populated when not skip_ccl)
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{};

    if constexpr (!Core::skip_ccl) {
        bcast_args = deepseek_b1_ops::Broadcast::ReaderArgs{};
    }
    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;  // BRISC is no-op
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_input_core && Core::is_matmul2_core>;  // Always mcast to the main grid

    // RMSNorm writer args (BRISC is no-op)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

    // RMSNorm2 writer args (BRISC is no-op)
    deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm2_args{};

    // Mcast CB indices from named compile-time args
    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");

    // Mcast sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::SenderArgs mcast_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore_addr"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("mcast_data_size_bytes"),
        mcast_src_cb,
        get_named_compile_time_arg_val("mcast_src_num_pages"),
        get_read_ptr(mcast_src_cb),
        get_write_ptr(mcast_dst_cb),
    };

    // Matmul CTArgs type alias (BRISC uses WriterCTArgs)
    using MatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::WriterCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul3CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

    // Matmul writer args (BRISC is no-op)
    deepseek_b1_ops::KNSlicedMatmul::WriterArgs matmul_args{};

    // Gather receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::GatherReduce::ReceiverArgs gather_reduce_args{
        get_named_compile_time_arg_val("gather_reduce_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_reduce_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_reduce_noc0_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("gather_reduce_noc1_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("gather_reduce_half0_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_half1_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_dst_num_tiles"),
    };

    // BRISC: Receiver args for SDPA input cores
    using CreateQHeadsCTArgs = deepseek_b1_ops::CreateQHeads::ReceiverCTArgs;
    deepseek_b1_ops::CreateQHeads::ReceiverArgs create_q_heads_args{
        get_named_compile_time_arg_val("cqh_nope_phase1_semaphore_addr"),
        get_named_compile_time_arg_val("cqh_nope_phase2_semaphore_addr"),
        get_named_compile_time_arg_val("cqh_rope_semaphore_addr"),
        get_named_compile_time_arg_val("cqh_num_nope_senders"),
        get_named_compile_time_arg_val("cqh_num_rope_senders"),
        get_named_compile_time_arg_val("cqh_receiver_in_cb"),
        get_named_compile_time_arg_val("cqh_out_cb"),
        get_named_compile_time_arg_val("cqh_nope_tiles"),
        get_named_compile_time_arg_val("cqh_rope_tiles"),
    };

    // Matmul2 writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul2_args{};

    // Matmul2 CB indices and parameters from named compile-time args
    constexpr uint32_t matmul2_in0 = get_named_compile_time_arg_val("matmul2_in0");

    // Matmul3 writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul3_args{};

    // Qrope CTArgs type alias (BRISC uses WriterCTArgs, no-op)
    using QRopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;

    // Qrope writer args (BRISC is no-op)
    deepseek_b1_ops::Rope::WriterArgs qrope_args{};

    // Mcast2 sender args (for input core to mcast rmsnorm2 output to all matmul2 cores)
    // Uses same grid and semaphores as first mcast
    // Reads from rmsnorm2_output_cb, writes to matmul2_in0 with loopback
    constexpr uint32_t mcast2_src_cb = get_named_compile_time_arg_val("rmsnorm2_output_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast2_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore_addr"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("mcast2_data_size_bytes"),
        mcast2_src_cb,  // Wait for rmsnorm2_output_cb
        get_named_compile_time_arg_val("mcast2_src_num_pages"),
        get_read_ptr(mcast2_src_cb),  // Read from rmsnorm2_output_cb
        get_write_ptr(matmul2_in0),   // Write to matmul2_in0 (loopback)
    };

    // Matmul writer args (BRISC is no-op)
    using DKV_MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs dkv_matmul_args{};

    // Gather receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::ReceiverArgs dkv_gather_args{
        get_named_compile_time_arg_val("dkv_gather_noc0_num_senders"),
        get_named_compile_time_arg_val("dkv_gather_noc1_num_senders"),
        get_named_compile_time_arg_val("dkv_gather_noc0_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("dkv_gather_noc1_receiver_semaphore_addr"),
        get_named_compile_time_arg_val("dkv_gather_dst_cb"),
        get_named_compile_time_arg_val("dkv_gather_dst_num_pages"),
    };

    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    deepseek_b1_ops::RMSNorm::WriterArgs kv_rmsnorm_args{};

    using K_RopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;

    // Writer args (empty - no-op)
    deepseek_b1_ops::Rope::WriterArgs krope_args{};

    deepseek_b1_ops::KVCacheUpdate::WriterArgs kv_cache_update_args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(0),
        .local_cur_pos = 0,  // set via kv_cache_update.set_local_cur_pos() below
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
        .krope_output_cb = get_named_compile_time_arg_val("krope_output_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
        .full_grid_mcast_start_x = get_named_compile_time_arg_val("full_grid_mcast_start_x"),
        .full_grid_mcast_start_y = get_named_compile_time_arg_val("full_grid_mcast_start_y"),
        .full_grid_mcast_end_x = get_named_compile_time_arg_val("full_grid_mcast_end_x"),
        .full_grid_mcast_end_y = get_named_compile_time_arg_val("full_grid_mcast_end_y"),
        .full_grid_mcast_num_dests = get_named_compile_time_arg_val("full_grid_mcast_num_dests"),
        .kv_cache_cur_pos_ready_semaphore_addr =
            get_named_compile_time_arg_val("kv_cache_cur_pos_ready_semaphore_addr"),
    };

    deepseek_b1_ops::FlashMLADecode::WriterArgs flash_mla_args;
    if constexpr (Core::is_mla_core) {
        DPRINT << " THIS IS MLA CORE " << ENDL();
        DPRINT << " PER CORE RTA ARG IDX " << per_core_rta_arg_idx << ENDL();
        constexpr uint32_t num_tree_reduction_steps = get_named_compile_time_arg_val("num_tree_reduction_steps");
        uint32_t cur_batch = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t core_num_in_reduce = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t is_output_core = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t is_mcast_sender = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t output_core_noc_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t output_core_noc_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t mcast_start_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t mcast_start_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t mcast_end_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t mcast_end_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(per_core_rta_arg_idx));
        per_core_rta_arg_idx += num_tree_reduction_steps * 4;

        DPRINT << " AFTTER MLA ARGS " << per_core_rta_arg_idx << ENDL();

        flash_mla_args = {
            .local_cur_pos = 0,  // set via flash_mla.set_local_cur_pos() below
            .cur_batch = cur_batch,
            .core_num_in_reduce = core_num_in_reduce,
            .is_output_core = is_output_core,
            .is_mcast_sender = is_mcast_sender,
            .output_core_noc_x = output_core_noc_x,
            .output_core_noc_y = output_core_noc_y,
            .mcast_start_x = mcast_start_x,
            .mcast_start_y = mcast_start_y,
            .mcast_end_x = mcast_end_x,
            .mcast_end_y = mcast_end_y,
            .tree_reduction_info = tree_reduction_info,
            .Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t"),
            .num_cores_per_head = get_named_compile_time_arg_val("num_cores_per_head"),
            .reducer_semaphore_addr = get_named_compile_time_arg_val("mla_reducer_semaphore_addr"),
            .k_chunk_size = get_named_compile_time_arg_val("k_chunk_size"),
            .q_chunk_size_bytes = get_named_compile_time_arg_val("q_chunk_size_bytes"),
            .DHt = get_named_compile_time_arg_val("DHt"),
            .num_mcast_dests = get_named_compile_time_arg_val("num_mcast_dests"),
            .full_grid_mcast_start_x = get_named_compile_time_arg_val("full_grid_mcast_start_x"),
            .full_grid_mcast_start_y = get_named_compile_time_arg_val("full_grid_mcast_start_y"),
            .full_grid_mcast_end_x = get_named_compile_time_arg_val("full_grid_mcast_end_x"),
            .full_grid_mcast_end_y = get_named_compile_time_arg_val("full_grid_mcast_end_y"),
            .full_grid_mcast_num_dests = get_named_compile_time_arg_val("full_grid_mcast_num_dests"),
            .q_input_mcast_semaphore_addr = get_named_compile_time_arg_val("mla_q_input_mcast_semaphore_addr"),
            .mcast_semaphore_addr = get_named_compile_time_arg_val("mla_mcast_semaphore_addr"),
            .ncrisc_brisc_sync_semaphore_addr = get_named_compile_time_arg_val("mla_ncrisc_brisc_sync_semaphore_addr"),
            .k_num_pages = get_named_compile_time_arg_val("k_num_pages"),
            .num_tree_reduction_steps = num_tree_reduction_steps,
            .receiver_ready_semaphore_addr = get_named_compile_time_arg_val("mla_receiver_ready_semaphore_addr"),
            .cb_k_in = get_named_compile_time_arg_val("mla_k_in_cb"),
            .cb_q_in = get_named_compile_time_arg_val("mla_q_in_cb"),
            .cb_mask = get_named_compile_time_arg_val("mla_mask_cb"),
            .cb_out_in = get_named_compile_time_arg_val("mla_out_in_cb"),
            .cb_ms_in = get_named_compile_time_arg_val("mla_ms_in_cb"),
            .cb_out_ms = get_named_compile_time_arg_val("mla_out_ms_cb"),
        };
    }

    using FlashMLACTArgs = deepseek_b1_ops::FlashMLADecode::WriterCTArgs<
        get_named_compile_time_arg_val("k_page_size"),
        get_named_compile_time_arg_val("vDHt"),
        get_named_compile_time_arg_val("mla_out_o_cb")>;

    // Matmul4/5 CTArgs (BRISC is no-op for matmul)
    using Matmul4CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul5CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs matmul4_args{};
    deepseek_b1_ops::Matmul::WriterArgs matmul5_args{};

    // Gather2 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather2_args{
        get_named_compile_time_arg_val("gather2_noc0_num_senders"),
        get_named_compile_time_arg_val("gather2_noc1_num_senders"),
        get_semaphore(get_named_compile_time_arg_val("gather2_noc0_receiver_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("gather2_noc1_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather2_dst_cb"),
        get_named_compile_time_arg_val("gather2_dst_num_pages"),
    };

    // Mcast3 sender args
    using Mcast3CTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast3_num_cores"),
        get_named_compile_time_arg_val("mcast3_is_part_of_receiver_grid") == 1,
        false>;  // loopback = false

    constexpr uint32_t mcast3_src_cb = get_named_compile_time_arg_val("mcast3_src_cb");
    constexpr uint32_t mcast3_dst_cb = get_named_compile_time_arg_val("mcast3_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast3_args{
        get_named_compile_time_arg_val("mcast3_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast3_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast3_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast3_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast3_data_sender_semaphore_addr"),
        get_semaphore(get_named_compile_time_arg_val("mcast3_data_receiver_semaphore")),
        get_named_compile_time_arg_val("mcast3_data_size_bytes"),
        mcast3_src_cb,
        get_named_compile_time_arg_val("mcast3_src_num_pages"),
        get_read_ptr(mcast3_src_cb),
        get_write_ptr(mcast3_dst_cb),  // CB 4 address (now allocated on gather core too)
    };

    // Gather3 receiver args
    deepseek_b1_ops::Gather::ReceiverArgs gather3_args{
        get_named_compile_time_arg_val("gather3_noc0_num_senders"),
        get_named_compile_time_arg_val("gather3_noc1_num_senders"),
        get_semaphore(get_named_compile_time_arg_val("gather3_noc0_receiver_semaphore_id")),
        get_semaphore(get_named_compile_time_arg_val("gather3_noc1_receiver_semaphore_id")),
        get_named_compile_time_arg_val("gather3_dst_cb"),
        get_named_compile_time_arg_val("gather3_dst_num_pages"),
    };

    using SdpaReduceWorkerCTArgs = deepseek_b1_ops::SdpaReduceWorker::WriterCTArgs<
        get_named_compile_time_arg_val("sdpa_cb_local_l"),
        get_named_compile_time_arg_val("sdpa_cb_local_ms"),
        get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
        get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
        get_named_compile_time_arg_val("sdpa_cb_packet_slot"),
        get_named_compile_time_arg_val("sdpa_l1_alignment"),
        get_named_compile_time_arg_val("sdpa_page_size_bytes"),
        get_named_compile_time_arg_val("sdpa_slot_size"),
        get_named_compile_time_arg_val("sdpa_ms_tile_size_bytes"),
        get_named_compile_time_arg_val("sdpa_l_chunk_size_bytes"),
        get_named_compile_time_arg_val("sdpa_num_l_chunks"),
        get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
        get_named_compile_time_arg_val("sdpa_cb_l_out"),
        get_named_compile_time_arg_val("sdpa_scatter_num_tiles"),
        get_named_compile_time_arg_val("sdpa_scatter_src_tile_size"),
        get_named_compile_time_arg_val("sdpa_scatter_dst_tile_size"),
        get_named_compile_time_arg_val("sdpa_scatter_face_size"),
        get_named_compile_time_arg_val("sdpa_scatter_row_face_size"),
        get_named_compile_time_arg_val("sdpa_scatter_num_rows"),
        1>;  // scatter_arrival_enabled=1 (signal matmul4 cores after each scatter row)

    deepseek_b1_ops::SdpaReduceWorker::WriterArgs sdpa_reduce_worker_args;
    if constexpr (Core::is_sdpa_worker_core) {
        sdpa_reduce_worker_args = {
            .r1_dst_mesh_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_dst_chip_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_neighbor_dst_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_dst_mesh_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_dst_chip_id = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_neighbor_dst_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_neighbor_sem_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .current_core_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .current_core_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .fwd_core_x = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .fwd_core_y = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_fwd_slot_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r1_base_slot_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_fwd_slot_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r2_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r2_base_slot_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .scatter_dest_l1_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .scatter_dest_coords_addr = get_arg_addr(per_core_rta_arg_idx),
            .scatter_arrival_sem_addr = get_semaphore(get_named_compile_time_arg_val("scatter_arrival_semaphore_id")),
        };
        per_core_rta_arg_idx += SdpaReduceWorkerCTArgs::scatter_num_rows * 2;
        DPRINT << " SDPA REDUCE WORKER ARGS " << per_core_rta_arg_idx << ENDL();
    }

    using SdpaReduceForwarderCTArgs = deepseek_b1_ops::SdpaReduceForwarder::CTArgs<
        get_named_compile_time_arg_val("sdpa_fwd_slots_per_round"),
        get_named_compile_time_arg_val("sdpa_fwd_slot_size"),
        get_named_compile_time_arg_val("sdpa_fwd_r2_buffer_offset")>;

    deepseek_b1_ops::SdpaReduceForwarder::ForwarderArgs sdpa_reduce_forwarder_args;
    if constexpr (Core::is_sdpa_forwarder_core) {
        DPRINT << " SDPA REDUCE FORWARDER ARGS " << per_core_rta_arg_idx << ENDL();
        sdpa_reduce_forwarder_args = {
            .buffer_base = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .buffer_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
            .r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(per_core_rta_arg_idx++)),
        };
        uint32_t fabric_args_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        sdpa_reduce_forwarder_args.rta_offset = per_core_rta_arg_idx;
        per_core_rta_arg_idx += fabric_args_offset;
        DPRINT << " SDPA REDUCE FORWARDER ARGS AFTER " << per_core_rta_arg_idx << ENDL();
    }

    // CCL Sender BRISC CTArgs (sends via fabric)
    using CCLSenderWriterCTArgs = deepseek_b1_ops::AllReduceSender::WriterCTArgs<
        get_named_compile_time_arg_val("ccl_sender_packet_header_cb_id"),
        get_named_compile_time_arg_val("ccl_sender_packet_cb_id"),
        get_named_compile_time_arg_val("ccl_sender_l1_alignment"),
        get_named_compile_time_arg_val("ccl_sender_input_num_tiles"),
        get_named_compile_time_arg_val("ccl_sender_page_size_bytes"),
        get_named_compile_time_arg_val("ccl_sender_payload_size_bytes"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_x"),
        get_named_compile_time_arg_val("ccl_sender_data_noc_y"),
        get_named_compile_time_arg_val("ccl_sender_remote_receiver_noc_x"),
        get_named_compile_time_arg_val("ccl_sender_remote_receiver_noc_y"),
        get_named_compile_time_arg_val("ccl_sender_dst_num_hops"),
        get_named_compile_time_arg_val("ccl_sender_num_connections")>;

    // Dummy ReaderCTArgs - not used by BRISC but needed for Op template
    using DummyReaderCTArgs = deepseek_b1_ops::AllReduceSender::ReaderCTArgs<0, 0, 0, 0, 0>;
    deepseek_b1_ops::AllReduceSender::RTArgs ccl_sender_args{};
    if constexpr (Core::is_ccl_sender_core) {
        DPRINT << " CCL SENDER ARGS " << per_core_rta_arg_idx << ENDL();
        ccl_sender_args = {
            .receiver_base_address = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .receive_semaphore_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
        uint32_t fabric_args_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        ccl_sender_args.fabric_args_start_index = per_core_rta_arg_idx;
        per_core_rta_arg_idx += fabric_args_offset;
        DPRINT << " CCL SENDER ARGS AFTER " << per_core_rta_arg_idx << ENDL();
    }
    DPRINT << " PER CORE RTA ARG IDX FINAL " << per_core_rta_arg_idx << ENDL();

    // ========================================================================
    // MoE BRISC args
    // ========================================================================
    struct Moe {
        struct Routed {
            using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
                get_named_compile_time_arg_val("moe_mcast_num_cores"),
                get_named_compile_time_arg_val("moe_mcast_is_part_of_receiver_grid"),
                Core::is_sender_core && Core::is_mcast_grid_core>;
            deepseek_b1_ops::Mcast::SenderArgs mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("moe_mcast_data_sender_semaphore_addr"),
                get_named_compile_time_arg_val("moe_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("moe_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("moe_mcast_src_cb"),
                get_named_compile_time_arg_val("moe_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("moe_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("moe_mcast_dst_cb")),
            };

#ifdef ENABLE_ROUTING
            using GateMMCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
            deepseek_b1_ops::Matmul::WriterArgs gate_mm_args{};

            deepseek_b1_ops::MoeGather::SenderArgs gather_args{
                get_named_compile_time_arg_val("gather_dest_noc_x"),
                get_named_compile_time_arg_val("gather_dest_noc_y"),
                get_named_compile_time_arg_val("gather_data_size_bytes"),
                get_named_compile_time_arg_val("gather_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("gather_src_cb"),
                get_named_compile_time_arg_val("gather_src_num_pages"),
                get_named_compile_time_arg_val("gather_sender_grid_start_x"),
                get_named_compile_time_arg_val("gather_sender_grid_start_y"),
                get_named_compile_time_arg_val("gather_sender_grid_end_x"),
                get_named_compile_time_arg_val("gather_sender_grid_end_y"),
                get_named_compile_time_arg_val("gather_row_major"),
                get_named_compile_time_arg_val("gather_receiver_data_addr"),
                0,
            };

            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::WriterCTArgs<
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("gate_output_indices_cb")>;

            deepseek_b1_ops::Mcast::SenderArgs index_mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("index_mcast_sender_semaphore_addr"),
                get_named_compile_time_arg_val("index_mcast_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("index_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("gate_output_indices_cb"),
                get_named_compile_time_arg_val("index_mcast_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("gate_output_indices_cb")),
                get_write_ptr(get_named_compile_time_arg_val("gate_proj_cb_index")),
            };

            deepseek_b1_ops::Mcast::SenderArgs expert_scale_mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("expert_scale_mcast_sender_semaphore_addr"),
                get_named_compile_time_arg_val("expert_scale_mcast_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("expert_scale_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("expert_scale_mcast_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("gate_output_cb")),
                get_write_ptr(get_named_compile_time_arg_val("mul_cb_scalar_src")),
            };
#endif

            using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;
            using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;

            using MulCTArgs = deepseek_b1_ops::EltwiseMul::WriterCTArgs<
                get_named_compile_time_arg_val("mul_cb_out"),
                get_named_compile_time_arg_val("mul_num_tiles"),
                get_named_compile_time_arg_val("mul_cb_scalar"),
                get_named_compile_time_arg_val("mul_cb_scalar_src"),
                get_named_compile_time_arg_val("mul_scalar_index_offset"),
                get_named_compile_time_arg_val("enable_routing")>;

            deepseek_b1_ops::MoeGather::SenderArgs down_proj_gather_args{
                get_named_compile_time_arg_val("down_proj_gather_dest_noc_x"),
                get_named_compile_time_arg_val("down_proj_gather_dest_noc_y"),
                get_named_compile_time_arg_val("down_proj_gather_data_size_bytes"),
                get_named_compile_time_arg_val("down_proj_gather_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_gather_src_cb"),
                get_named_compile_time_arg_val("down_proj_gather_src_num_pages"),
                get_named_compile_time_arg_val("down_proj_gather_sender_grid_start_x"),
                get_named_compile_time_arg_val("down_proj_gather_sender_grid_start_y"),
                get_named_compile_time_arg_val("down_proj_gather_sender_grid_end_x"),
                get_named_compile_time_arg_val("down_proj_gather_sender_grid_end_y"),
                get_named_compile_time_arg_val("down_proj_gather_row_major"),
                get_named_compile_time_arg_val("down_proj_gather_receiver_data_addr"),
                get_named_compile_time_arg_val("down_proj_gather_sender_idx"),
            };

            deepseek_b1_ops::Mcast::SenderArgs down_proj_mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("down_proj_mcast_sender_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_mcast_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("down_proj_mcast_src_cb"),
                get_named_compile_time_arg_val("down_proj_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("down_proj_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("down_proj_mcast_dst_cb")),
            };

            using DownProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;
            using AddCTArgs = deepseek_b1_ops::EltwiseAdd::WriterCTArgs;

            using ResidualMcastCTArgs = McastCTArgs;
            deepseek_b1_ops::Mcast::SenderArgs residual_mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("shared_residual_mcast_data_sender_semaphore_addr"),
                get_named_compile_time_arg_val("shared_residual_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_residual_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("shared_residual_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("shared_residual_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("shared_residual_mcast_dst_cb")),
            };

            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
            deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

#ifdef ENABLE_REDUCE_TO_ONE
            using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::WriterCTArgs<
                get_named_compile_time_arg_val("reduce_device_role"),
                get_named_compile_time_arg_val("reduce_num_tiles"),
                get_named_compile_time_arg_val("reduce_payload_size_bytes"),
                get_named_compile_time_arg_val("reduce_local_cb"),
                get_named_compile_time_arg_val("reduce_scratch_cb"),
                get_named_compile_time_arg_val("reduce_packet_cb"),
                get_named_compile_time_arg_val("reduce_packet_header_cb"),
                get_named_compile_time_arg_val("reduce_num_hops"),
                get_named_compile_time_arg_val("reduce_dst_fabric_node_chip_id"),
                get_named_compile_time_arg_val("reduce_dst_fabric_node_mesh_id"),
                get_named_compile_time_arg_val("reduce_output_core_noc_x"),
                get_named_compile_time_arg_val("reduce_output_core_noc_y"),
                get_named_compile_time_arg_val("reduce_num_workers"),
                get_named_compile_time_arg_val("reduce_slot_size_bytes"),
                get_named_compile_time_arg_val("is_reduce_fabric_core"),
                get_named_compile_time_arg_val("reduce_brisc_fabric_rt_arg_base")>;

            deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs reduce_rt_args{};
#endif
        } routed;

        struct Shared {
            using GUMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::WriterCTArgs;
            deepseek_b1_ops::KNSlicedMatmul::WriterArgs gu_matmul_args{};

            deepseek_b1_ops::MoeGather::SenderArgs ag_args{
                get_named_compile_time_arg_val("shared_ag_dest_noc_x"),
                get_named_compile_time_arg_val("shared_ag_dest_noc_y"),
                get_named_compile_time_arg_val("shared_ag_data_size_bytes"),
                get_named_compile_time_arg_val("shared_ag_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_ag_src_cb"),
                get_named_compile_time_arg_val("shared_ag_src_num_pages"),
                0,
                0,
                0,
                0,
                0,
                get_named_compile_time_arg_val("shared_ag_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_ag_sender_idx"),
            };

            deepseek_b1_ops::MoeGather::SenderArgs bg_args{
                get_named_compile_time_arg_val("shared_bg_dest_noc_x"),
                get_named_compile_time_arg_val("shared_bg_dest_noc_y"),
                get_named_compile_time_arg_val("shared_bg_data_size_bytes"),
                get_named_compile_time_arg_val("shared_bg_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_bg_src_cb"),
                get_named_compile_time_arg_val("shared_bg_src_num_pages"),
                0,
                0,
                0,
                0,
                0,
                get_named_compile_time_arg_val("shared_bg_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_bg_sender_idx"),
            };

            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::WriterCTArgs;
            deepseek_b1_ops::GatedReduce::WriterArgs gated_reduce_args{};

            using DownMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::SenderArgs down_mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("shared_down_mcast_data_sender_semaphore_addr"),
                get_named_compile_time_arg_val("shared_down_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_down_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("shared_down_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_down_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("shared_down_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("shared_down_mcast_dst_cb")),
            };

            using DownMatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
            deepseek_b1_ops::Matmul::WriterArgs down_matmul_args{};

            using ResidualAddCTArgs = deepseek_b1_ops::ResidualAdd::WriterCTArgs;
            deepseek_b1_ops::ResidualAdd::WriterArgs residual_add_args{};

            deepseek_b1_ops::MoeGather::SenderArgs og_args{
                get_named_compile_time_arg_val("shared_og_dest_noc_x"),
                get_named_compile_time_arg_val("shared_og_dest_noc_y"),
                get_named_compile_time_arg_val("shared_og_data_size_bytes"),
                get_named_compile_time_arg_val("shared_og_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_og_src_cb"),
                get_named_compile_time_arg_val("shared_og_src_num_pages"),
                0,
                0,
                0,
                0,
                0,
                get_named_compile_time_arg_val("shared_og_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_residual_add_core_idx"),
            };

            using OutputMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::SenderArgs output_mcast_args{
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("shared_output_mcast_data_sender_semaphore_addr"),
                get_named_compile_time_arg_val("shared_output_mcast_data_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_output_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("shared_output_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_output_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("shared_output_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("add_cb_in1")),
            };
        } shared;
    } moe;

#ifdef ENABLE_REDUCE_TO_ONE
    constexpr size_t reduce_brisc_arg_start = get_named_compile_time_arg_val("reduce_brisc_rt_arg_base");
    if constexpr (Core::is_reduce_worker_core) {
        moe.routed.reduce_rt_args = deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs{
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 0),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 1),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 2),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 3),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 4),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 5),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 6),
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 7),
        };
    }
#endif

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: rmsnorm compute, matmul compute
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type aliases (required for Op templates)

    // CCL Broadcast CTArgs (no-op for TRISC)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
    deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb")>;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm2_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("rmsnorm2_input_cb"),
        get_named_compile_time_arg_val("rmsnorm2_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm2_output_cb")>;
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(1),     // scalar (1/sqrt(7168))
    };

    // Mcast compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul CTArgs type alias (out_w is compile-time for TRISC)
    const auto matmul_half_info = unified_kernels::get_split_half_core_info<true>(
        get_named_compile_time_arg_val("matmul_grid_start_x"),
        get_named_compile_time_arg_val("matmul_grid_start_y"),
        get_named_compile_time_arg_val("matmul_grid_end_x"),
        get_named_compile_time_arg_val("matmul_grid_end_y"),
        get_named_compile_time_arg_val("matmul_half_num_cores"));
    constexpr uint32_t matmul_k_offset_half1 = get_named_compile_time_arg_val("matmul_k_offset_half1");
    uint32_t k_offset = matmul_half_info.is_half0 ? 0 : matmul_k_offset_half1;

    using MatmulCTArgs =
        deepseek_b1_ops::KNSlicedMatmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::KNSlicedMatmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        k_offset,
        get_named_compile_time_arg_val("matmul_k_per_core"),
        get_named_compile_time_arg_val("matmul_act_total_tiles"),
        get_common_arg_val<uint32_t>(8),  // matmul_weights_addr
    };

    // Gather reduce compute args
    deepseek_b1_ops::GatherReduce::ComputeArgs gather_reduce_args{
        get_named_compile_time_arg_val("gather_reduce_half0_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_half1_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_dst_num_tiles"),
    };

    // RMSNorm2 compute args (separate CBs with exact sizes for testing)
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm2_args{
        get_common_arg_val<uint32_t>(0),  // epsilon (same as rmsnorm1)
        get_common_arg_val<float>(2),     // scalar (1/sqrt(1536))
    };

    // Matmul2 CTArgs type alias (out_w is compile-time for TRISC)
    using Matmul2CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul2_out_w_per_core")>;

    // Matmul2 compute args (from compile-time args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul2_args{
        get_named_compile_time_arg_val("matmul2_in0"),
        get_named_compile_time_arg_val("matmul2_in1"),
        get_named_compile_time_arg_val("matmul2_out"),
        get_named_compile_time_arg_val("matmul2_k_num_tiles"),
        get_common_arg_val<uint32_t>(9),  // matmul2_weights_addr
    };

    // Mcast2 compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast2_args{};

    // Matmul3 CTArgs type alias (out_w is compile-time for TRISC)
    using Matmul3CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul3_out_w_per_core")>;

    // Matmul3 compute args (from compile-time args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul3_args{
        get_named_compile_time_arg_val("matmul3_in0"),
        get_named_compile_time_arg_val("matmul3_in1"),
        get_named_compile_time_arg_val("matmul3_out"),
        get_named_compile_time_arg_val("matmul3_k_num_tiles"),
    };

    // Qrope CTArgs type alias
    using QRopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("qrope_Wt"), get_named_compile_time_arg_val("qrope_Ht")>;

    // Qrope compute args (from compile-time args)
    deepseek_b1_ops::Rope::ComputeArgs qrope_args{
        get_named_compile_time_arg_val("qrope_in_cb"),  // Input from matmul2 output
        get_named_compile_time_arg_val("qrope_cos_cb"),
        get_named_compile_time_arg_val("qrope_sin_cb"),
        get_named_compile_time_arg_val("qrope_trans_mat_cb"),
        get_named_compile_time_arg_val("qrope_rotated_in_interm_cb"),
        get_named_compile_time_arg_val("qrope_cos_interm_cb"),
        get_named_compile_time_arg_val("qrope_sin_interm_cb"),
        get_named_compile_time_arg_val("qrope_output_cb"),
    };

    // CreateQHeads compute args (tilization on SDPA input cores)
    using CreateQHeadsCTArgs = deepseek_b1_ops::CreateQHeads::ComputeCTArgs;
    deepseek_b1_ops::CreateQHeads::ComputeArgs create_q_heads_args{
        get_named_compile_time_arg_val("cqh_receiver_in_cb"),
        get_named_compile_time_arg_val("cqh_out_cb"),
        get_named_compile_time_arg_val("cqh_nope_tiles"),
        get_named_compile_time_arg_val("cqh_rope_tiles"),
    };

    // DKV Matmul compute args
    using DKV_MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("dkv_matmul_out_w_per_core")>;

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Matmul::ComputeArgs dkv_matmul_args{
        get_named_compile_time_arg_val("dkv_matmul_in0"),
        get_named_compile_time_arg_val("dkv_matmul_in1"),
        get_named_compile_time_arg_val("dkv_matmul_out"),
        get_named_compile_time_arg_val("dkv_matmul_k_num_tiles"),
        get_common_arg_val<uint32_t>(10),  // dkv_matmul_weights_addr
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs dkv_gather_args{};

    // CTArgs type aliases (required for Op templates)
    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("kv_rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
        get_named_compile_time_arg_val("kv_rmsnorm_input_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_output_cb")>;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs kv_rmsnorm_args{
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(3),     // kv_scalar (1/sqrt(512))
    };

    using K_RopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("krope_Wt"), get_named_compile_time_arg_val("krope_Ht")>;

    // CB indices (passed as runtime args to ComputeArgs)
    constexpr uint32_t krope_input_cb = get_named_compile_time_arg_val("krope_in_cb");
    constexpr uint32_t krope_cos_cb = get_named_compile_time_arg_val("krope_cos_cb");
    constexpr uint32_t krope_sin_cb = get_named_compile_time_arg_val("krope_sin_cb");
    constexpr uint32_t krope_trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb");
    constexpr uint32_t krope_rotated_in_interm_cb = get_named_compile_time_arg_val("krope_rotated_in_interm_cb");
    constexpr uint32_t krope_cos_interm_cb = get_named_compile_time_arg_val("krope_cos_interm_cb");
    constexpr uint32_t krope_sin_interm_cb = get_named_compile_time_arg_val("krope_sin_interm_cb");
    constexpr uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");

    // Compute args: all CB indices
    deepseek_b1_ops::Rope::ComputeArgs krope_args{
        .in_cb = krope_input_cb,
        .cos_cb = krope_cos_cb,
        .sin_cb = krope_sin_cb,
        .trans_mat_cb = krope_trans_mat_cb,
        .rotated_in_interm_cb = krope_rotated_in_interm_cb,
        .cos_interm_cb = krope_cos_interm_cb,
        .sin_interm_cb = krope_sin_interm_cb,
        .out_cb = krope_output_cb,
    };

    deepseek_b1_ops::KVCacheUpdate::ComputeArgs kv_cache_update_args{
        .kv_cache_input_cb = get_common_arg_val<uint32_t>(4),
        .kv_cache_output_cb = get_common_arg_val<uint32_t>(5),
        .kv_cache_intermed_cb = get_common_arg_val<uint32_t>(6),
    };
    deepseek_b1_ops::FlashMLADecode::ComputeArgs flash_mla_args;
    if constexpr (Core::is_mla_core) {
        constexpr uint32_t num_tree_reduction_steps = get_named_compile_time_arg_val("num_tree_reduction_steps");
        uint32_t do_reduce = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t do_output = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t cur_batch = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t core_num_in_reduce = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        uint32_t is_sender_after_reduce = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(per_core_rta_arg_idx));
        per_core_rta_arg_idx += num_tree_reduction_steps * 2;

        flash_mla_args = {
            .local_cur_pos = 0,  // set via flash_mla.set_local_cur_pos() below
            .do_reduce = do_reduce,
            .do_output = do_output,
            .cur_batch = cur_batch,
            .core_num_in_reduce = core_num_in_reduce,
            .is_sender_after_reduce = is_sender_after_reduce,
            .tree_reduction_info = tree_reduction_info,
            .k_chunk_size = get_named_compile_time_arg_val("k_chunk_size"),
            .num_cores_per_head = get_named_compile_time_arg_val("num_cores_per_head"),
            .num_tree_reduction_steps = num_tree_reduction_steps,
        };
    }

    using FlashMLACTArgs = deepseek_b1_ops::FlashMLADecode::ComputeCTArgs<
        get_named_compile_time_arg_val("mla_q_in_cb"),
        get_named_compile_time_arg_val("mla_k_in_cb"),
        get_named_compile_time_arg_val("mla_mask_cb"),
        get_named_compile_time_arg_val("mla_interm_out_cb"),
        get_named_compile_time_arg_val("mla_interm_ms_cb"),
        get_named_compile_time_arg_val("mla_out_in_cb"),
        get_named_compile_time_arg_val("mla_ms_in_cb"),
        get_named_compile_time_arg_val("mla_out_o_cb"),
        get_named_compile_time_arg_val("mla_out_ms_cb"),
        get_named_compile_time_arg_val("mla_out_final_cb")>;

    // Matmul4 CTArgs
    using Matmul4CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul4_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul4_args{
        get_named_compile_time_arg_val("matmul4_in0"),
        get_named_compile_time_arg_val("matmul4_in1"),
        get_named_compile_time_arg_val("matmul4_out"),
        get_named_compile_time_arg_val("matmul4_k_num_tiles"),
    };

    // Gather2 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather2_args{};

    // Mcast3 CTArgs (no-op)
    using Mcast3CTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast3_args{};

    // Matmul5 CTArgs
    using Matmul5CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul5_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul5_args{
        get_named_compile_time_arg_val("matmul5_in0"),
        get_named_compile_time_arg_val("matmul5_in1"),
        get_named_compile_time_arg_val("matmul5_out"),
        get_named_compile_time_arg_val("matmul5_k_num_tiles"),
    };

    // Gather3 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather3_args{};

    using SdpaReduceWorkerCTArgs = deepseek_b1_ops::SdpaReduceWorker::ComputeCTArgs<
        get_named_compile_time_arg_val("sdpa_cb_local_l"),
        get_named_compile_time_arg_val("sdpa_cb_local_ms"),
        get_named_compile_time_arg_val("sdpa_cb_neighbor_l"),
        get_named_compile_time_arg_val("sdpa_cb_neighbor_ms"),
        get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
        get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
        get_named_compile_time_arg_val("sdpa_cb_l_out"),
        get_named_compile_time_arg_val("sdpa_scale_fp32"),
        get_named_compile_time_arg_val("sdpa_tiles_per_l_chunk"),
        get_named_compile_time_arg_val("sdpa_num_l_chunks"),
        get_named_compile_time_arg_val("sdpa_position_enabled"),
        get_named_compile_time_arg_val("sdpa_per_device_chunk_size"),
        1>;  // final_reduction=1 (always normalize in post_sdpa, untilize constraint)
    deepseek_b1_ops::SdpaReduceWorker::ComputeArgs sdpa_reduce_worker_args;
    if constexpr (Core::is_sdpa_worker_core) {
        sdpa_reduce_worker_args.pos_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        sdpa_reduce_worker_args.device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        sdpa_reduce_worker_args.r1_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        sdpa_reduce_worker_args.r2_neighbor_device_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        sdpa_reduce_worker_args.r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
    }

    using SdpaReduceForwarderCTArgs = deepseek_b1_ops::SdpaReduceForwarder::CTArgs<0, 0, 0>;
    deepseek_b1_ops::SdpaReduceForwarder::ForwarderArgs sdpa_reduce_forwarder_args;

    // CCL Receiver compute CTArgs (reduction)
    using CCLReceiverComputeCTArgs = deepseek_b1_ops::AllReduceReceiver::ComputeCTArgs<
        get_named_compile_time_arg_val("ccl_receiver_cb_in0"),
        get_named_compile_time_arg_val("ccl_receiver_cb_in1"),
        get_named_compile_time_arg_val("ccl_receiver_cb_out0"),
        get_named_compile_time_arg_val("ccl_receiver_cb_residual"),
        get_named_compile_time_arg_val("ccl_receiver_cb_temp"),
        get_named_compile_time_arg_val("ccl_receiver_has_residual"),
        get_named_compile_time_arg_val("ccl_receiver_num_tiles")>;

    using DummyReaderCTArgs = deepseek_b1_ops::AllReduceReceiver::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
    deepseek_b1_ops::AllReduceReceiver::RTArgs ccl_receiver_args{};

    DPRINT << " PER CORE RTA ARG IDX FINAL " << per_core_rta_arg_idx << ENDL();

    // ========================================================================
    // MoE TRISC args
    // ========================================================================
    struct Moe {
        struct Routed {
            using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

#ifdef ENABLE_ROUTING
            using GateMMCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<
                get_named_compile_time_arg_val("gate_mm_out_w"),
                false,
                get_named_compile_time_arg_val("gate_mm_fused_activation")>;
            deepseek_b1_ops::Matmul::ComputeArgs gate_mm_args{
                get_named_compile_time_arg_val("gate_mm_in0"),
                get_named_compile_time_arg_val("gate_mm_in1"),
                get_named_compile_time_arg_val("gate_mm_out"),
                get_named_compile_time_arg_val("gate_mm_k_num_tiles"),
            };

            deepseek_b1_ops::MoeGather::ComputeArgs gather_args{};

            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ComputeCTArgs<
                get_named_compile_time_arg_val("gate_input_cb"),
                get_named_compile_time_arg_val("gate_bias_cb"),
                get_named_compile_time_arg_val("gate_input_indices_cb"),
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("gate_output_indices_cb"),
                get_named_compile_time_arg_val("gate_eps"),
                get_named_compile_time_arg_val("gate_scaling_factor"),
                get_named_compile_time_arg_val("gate_enable_sigmoid")>;

            deepseek_b1_ops::Mcast::ComputeArgs index_mcast_args{};
            deepseek_b1_ops::Mcast::ComputeArgs expert_scale_mcast_args{};
#endif

            using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
                get_named_compile_time_arg_val("gate_proj_cb_in0"),
                get_named_compile_time_arg_val("gate_proj_cb_in1"),
                get_named_compile_time_arg_val("gate_proj_cb_out"),
                get_named_compile_time_arg_val("gate_proj_subblock_k"),
                get_named_compile_time_arg_val("gate_proj_per_core_n"),
                get_named_compile_time_arg_val("gate_proj_subblock_w"),
                get_named_compile_time_arg_val("gate_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("gate_proj_tile_r_dim"),
                get_named_compile_time_arg_val("gate_proj_fuse_silu"),
                get_named_compile_time_arg_val("gate_proj_fp32_dest_acc_en")>;

            using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
                get_named_compile_time_arg_val("up_proj_cb_in0"),
                get_named_compile_time_arg_val("up_proj_cb_in1"),
                get_named_compile_time_arg_val("up_proj_cb_mm_out"),
                get_named_compile_time_arg_val("up_proj_subblock_k"),
                get_named_compile_time_arg_val("up_proj_per_core_n"),
                get_named_compile_time_arg_val("up_proj_subblock_w"),
                get_named_compile_time_arg_val("up_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("up_proj_tile_r_dim"),
                get_named_compile_time_arg_val("up_proj_fuse_silu"),
                get_named_compile_time_arg_val("up_proj_fp32_dest_acc_en")>;

            using MulCTArgs = deepseek_b1_ops::EltwiseMul::ComputeCTArgs<
                get_named_compile_time_arg_val("mul_cb_in0"),
                get_named_compile_time_arg_val("mul_cb_in1"),
                get_named_compile_time_arg_val("mul_cb_out"),
                get_named_compile_time_arg_val("mul_num_tiles"),
                get_named_compile_time_arg_val("up_proj_cb_mm_out"),
                get_named_compile_time_arg_val("up_proj_per_core_n"),
                get_named_compile_time_arg_val("gate_proj_cb_out"),
                get_named_compile_time_arg_val("gate_proj_per_core_n"),
                get_named_compile_time_arg_val("mul_cb_scalar"),
                get_named_compile_time_arg_val("mul_fp32_dest_acc_en"),
                get_named_compile_time_arg_val("enable_routing")>;

            deepseek_b1_ops::MoeGather::ComputeArgs down_proj_gather_args{};
            deepseek_b1_ops::Mcast::ComputeArgs down_proj_mcast_args{};

            using DownProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
                get_named_compile_time_arg_val("down_proj_cb_in0"),
                get_named_compile_time_arg_val("down_proj_cb_in1"),
                get_named_compile_time_arg_val("down_proj_cb_out"),
                get_named_compile_time_arg_val("down_proj_subblock_k"),
                get_named_compile_time_arg_val("down_proj_per_core_n"),
                get_named_compile_time_arg_val("down_proj_subblock_w"),
                get_named_compile_time_arg_val("down_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("down_proj_tile_r_dim"),
                get_named_compile_time_arg_val("down_proj_fuse_silu"),
                get_named_compile_time_arg_val("down_proj_fp32_dest_acc_en")>;

            using AddCTArgs = deepseek_b1_ops::EltwiseAdd::ComputeCTArgs<
                get_named_compile_time_arg_val("add_cb_in0"),
                get_named_compile_time_arg_val("add_cb_in1"),
                get_named_compile_time_arg_val("add_cb_out"),
                get_named_compile_time_arg_val("add_num_tiles"),
                get_named_compile_time_arg_val("down_proj_cb_out"),
                get_named_compile_time_arg_val("down_proj_per_core_n"),
                get_named_compile_time_arg_val("add_cb_in1_wait_tiles"),
                get_named_compile_time_arg_val("add_sender_index"),
                get_named_compile_time_arg_val("add_slice_size_bytes")>;

            using ResidualMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs residual_mcast_args{};

            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
                get_named_compile_time_arg_val("moe_rmsnorm_fp32_acc") == 1,
                get_named_compile_time_arg_val("moe_rmsnorm_num_tiles"),
                get_named_compile_time_arg_val("moe_rmsnorm_rsqrt_fast_approx") == 1,
                get_named_compile_time_arg_val("moe_rmsnorm_input_cb"),
                get_named_compile_time_arg_val("moe_rmsnorm_gamma_cb"),
                get_named_compile_time_arg_val("moe_rmsnorm_output_cb")>;
            deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
                get_common_arg_val<uint32_t>(
                    get_named_compile_time_arg_val("moe_rmsnorm_trisc_common_rt_arg_base") + 0),
                get_common_arg_val<float>(get_named_compile_time_arg_val("moe_rmsnorm_trisc_common_rt_arg_base") + 1),
            };

#ifdef ENABLE_REDUCE_TO_ONE
            using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ComputeCTArgs<
                get_named_compile_time_arg_val("reduce_device_role"),
                get_named_compile_time_arg_val("reduce_num_tiles"),
                get_named_compile_time_arg_val("reduce_local_cb"),
                get_named_compile_time_arg_val("reduce_received_cb_r1"),
                get_named_compile_time_arg_val("reduce_received_cb_r2"),
                get_named_compile_time_arg_val("reduce_received_cb_r3"),
                get_named_compile_time_arg_val("reduce_output_cb"),
                get_named_compile_time_arg_val("reduce_scratch_cb"),
                get_named_compile_time_arg_val("is_reduce_fabric_core")>;

            deepseek_b1_ops::ReduceToOneB1::ComputeArgs reduce_rt_args{};
#endif
        } routed;

        struct Shared {
            using GUMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ComputeCTArgs<>;
            deepseek_b1_ops::KNSlicedMatmul::ComputeArgs gu_matmul_args{
                get_named_compile_time_arg_val("shared_gu_act_cb"),
                get_named_compile_time_arg_val("shared_gu_weights_cb"),
                get_named_compile_time_arg_val("shared_gu_out_cb"),
                get_named_compile_time_arg_val("shared_gu_k_offset"),
                get_named_compile_time_arg_val("shared_gu_k_per_core"),
                get_named_compile_time_arg_val("shared_gu_act_total_tiles"),
            };

            deepseek_b1_ops::MoeGather::ComputeArgs ag_args{};
            deepseek_b1_ops::MoeGather::ComputeArgs bg_args{};

            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ComputeCTArgs<
                get_named_compile_time_arg_val("shared_gated_reduce_tiles_per_k"),
                get_named_compile_time_arg_val("shared_gated_reduce_k_num_tiles")>;
            deepseek_b1_ops::GatedReduce::ComputeArgs gated_reduce_args{
                get_named_compile_time_arg_val("shared_gated_reduce_group1_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_group2_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_intermed_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_mcast_src_cb"),
            };

            using DownMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs down_mcast_args{};

            using DownMatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val(
                "shared_down_matmul_out_w_per_core")>;
            deepseek_b1_ops::Matmul::ComputeArgs down_matmul_args{
                get_named_compile_time_arg_val("shared_down_matmul_in0"),
                get_named_compile_time_arg_val("shared_down_matmul_in1"),
                get_named_compile_time_arg_val("shared_down_matmul_out"),
                get_named_compile_time_arg_val("shared_down_matmul_k_num_tiles"),
            };

            using ResidualAddCTArgs = deepseek_b1_ops::ResidualAdd::ComputeCTArgs<get_named_compile_time_arg_val(
                "shared_residual_add_out_w")>;
            deepseek_b1_ops::ResidualAdd::ComputeArgs residual_add_args{
                get_named_compile_time_arg_val("shared_residual_add_in0"),
                get_named_compile_time_arg_val("shared_residual_add_in1"),
                get_named_compile_time_arg_val("shared_residual_add_out"),
                get_named_compile_time_arg_val("shared_residual_add_total_in1_tiles"),
                get_named_compile_time_arg_val("shared_residual_add_core_idx"),
            };

            deepseek_b1_ops::MoeGather::ComputeArgs og_args{};

            using OutputMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs output_mcast_args{};
        } shared;
    } moe;

    deepseek_compute_kernel_init();
#endif

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_input_core) {
        // Multi-device mode: NCRISC sets up gamma buffers while BRISC handles CCL
        // RMSNorm gamma buffer
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);

        // RMSNorm2 gamma buffer (3 tiles of 16x32)
        constexpr uint32_t rmsnorm2_gamma_cb = get_named_compile_time_arg_val("rmsnorm2_gamma_cb");
        constexpr uint32_t rmsnorm2_num_tiles = get_named_compile_time_arg_val("rmsnorm2_num_tiles");
        unified_kernels::setup_sharded_buffer(rmsnorm2_gamma_cb, rmsnorm2_num_tiles);
    }
    if constexpr (Core::is_qnope_core) {
        // Matmul3 CB indices and parameters from named compile-time args
        constexpr uint32_t matmul3_in1 = get_named_compile_time_arg_val("matmul3_in1");
        constexpr uint32_t matmul3_k_num_tiles = get_named_compile_time_arg_val("matmul3_k_num_tiles");
        constexpr uint32_t matmul3_out_w_per_core = get_named_compile_time_arg_val("matmul3_out_w_per_core");

        // Matmul3 weights (on Qnope cores, [128, 512] = 4 * 16 = 64 tiles per core)
        unified_kernels::setup_sharded_buffer(matmul3_in1, matmul3_k_num_tiles * matmul3_out_w_per_core);
    }

    if constexpr (Core::is_qrope_core) {
        constexpr uint32_t qrope_trans_mat_cb = get_named_compile_time_arg_val("qrope_trans_mat_cb");
        unified_kernels::setup_sharded_buffer(qrope_trans_mat_cb, 1);
    }

    if constexpr (Core::is_kv_rmsnorm_core) {
        // RMSNorm gamma (sharded weights)
        constexpr uint32_t kv_rmsnorm_gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
        constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_gamma_cb, kv_rmsnorm_num_tiles);
    }

    if constexpr (Core::is_krope_core) {
        constexpr uint32_t krope_trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb");
        unified_kernels::setup_sharded_buffer(krope_trans_mat_cb, 1);
    }

    if constexpr (Core::is_matmul4_core) {
        constexpr uint32_t matmul4_in1 = get_named_compile_time_arg_val("matmul4_in1");
        constexpr uint32_t matmul4_k_num_tiles = get_named_compile_time_arg_val("matmul4_k_num_tiles");
        constexpr uint32_t matmul4_out_w_per_core = get_named_compile_time_arg_val("matmul4_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul4_in1, matmul4_k_num_tiles * matmul4_out_w_per_core);
    }

    if constexpr (Core::is_matmul5_core) {
        constexpr uint32_t matmul5_in1 = get_named_compile_time_arg_val("matmul5_in1");
        constexpr uint32_t matmul5_k_num_tiles = get_named_compile_time_arg_val("matmul5_k_num_tiles");
        constexpr uint32_t matmul5_out_w_per_core = get_named_compile_time_arg_val("matmul5_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul5_in1, matmul5_k_num_tiles * matmul5_out_w_per_core);
    }
#endif
    DPRINT << " DONE ARGS" << ENDL();
#ifdef COMPILE_TEST_ONLY
    return;
#endif

    // ====================================================================
    // Mcast: Initialize persistent mcast
    // ====================================================================
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_input_core,
        Core::is_matmul2_core,
        Core::is_matmul_core || Core::is_dkv_matmul_core,
        true>
        mcast;
    {
        DeviceZoneScopedN("MCAST_INIT");
        mcast.init(mcast_args);
    }

    // ========================================================================
    // CCL Broadcast (optional, skip if single-device mode)
    // ========================================================================
    if constexpr (!Core::skip_ccl) {
        {
            DeviceZoneScopedN("CCL_BROADCAST");
            deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_input_core> bcast;
            bcast(bcast_args);
        }
    }

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_input_core) {
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

        cb_reserve_back(rmsnorm_input_cb, rmsnorm_num_tiles);
        cb_push_back(rmsnorm_input_cb, rmsnorm_num_tiles);
    }
#endif

    DPRINT << " DONE CCL BROADCAST" << ENDL();

    // SP position handling
#if defined(COMPILE_FOR_BRISC)
    uint32_t cur_pos_addr = get_common_arg_val<uint32_t>(1);
#elif defined(COMPILE_FOR_NCRISC)
    uint32_t cur_pos_addr = get_common_arg_val<uint32_t>(14);
#elif defined(COMPILE_FOR_TRISC)
    uint32_t cur_pos_addr = get_common_arg_val<uint32_t>(7);
#endif

    volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_addr);
    uint32_t cur_pos = pos_ptr[0];

    const bool skip_attention = false;
    const bool skip_kv_cache_update = false;
    const uint32_t local_cur_pos = cur_pos;

    if (!skip_attention) {
        DPRINT << " DOING ATTENTION" << ENDL();

        // ========================================================================
        // Input core: RMSNorm + Mcast send
        // ========================================================================
        {
            DeviceZoneScopedN("RMSNORM");
            deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core, true> rmsnorm;
            rmsnorm(rmsnorm_args);
        }

        {
            DeviceZoneScopedN("MCAST");
            mcast(mcast_args);
        }

        // ========================================================================
        // Matmul operation
        // ========================================================================
        {
            DeviceZoneScopedN("MATMUL");
            // pop_act = false (shared activation buffer), pop_weights = false (weights are persistent)
            deepseek_b1_ops::KNSlicedMatmul::Op<MatmulCTArgs, Core::is_matmul_core, false, false> matmul;
            matmul(matmul_args);
        }

        // ========================================================================
        // GatherReduce: matmul cores (senders) -> input core (receiver/reducer)
        // NCRISC sends from matmul cores, BRISC receives on input core, TRISC reduces CB7 += CB8
        // ========================================================================
        {
            DeviceZoneScopedN("GATHER");
            // pop_src = true (matmul output is consumed after gather)
            deepseek_b1_ops::GatherReduce::Op<Core::is_matmul_core, Core::is_input_core, Core::is_input_core, true>
                gather_reduce;
            gather_reduce(gather_reduce_args);
        }

        // ========================================================================
        // RMSNorm2: Apply RMSNorm to the gathered data (1536 elements = 3 tiles of 16x32)
        // Gather writes directly to rmsnorm2_input_cb (3 tiles of 16x32)
        // Uses SEPARATE CBs with exact sizes:
        //   - Input: rmsnorm2_input_cb (3 tiles from gather)
        //   - Output: rmsnorm2_output_cb (3 tiles)
        //   - Gamma: rmsnorm2_gamma_cb (3 tiles)
        // ========================================================================
        // pop_input = true (gathered data is consumed after RMSNorm2)
        {
            DeviceZoneScopedN("RMSNORM2");
            deepseek_b1_ops::RMSNorm::Op<RMSNorm2CTArgs, Core::is_input_core, true> rmsnorm2;
            rmsnorm2(rmsnorm2_args);
        }

        // ========================================================================
        // Mcast2: Broadcast rmsnorm2 output from input core to all matmul2 cores
        // Reads from rmsnorm2_output_cb, writes to matmul2_in0 with loopback
        // Uses same grid and semaphores as first mcast
        // ========================================================================
        // pop_src = true (rmsnorm2 output is consumed after mcast)
        {
            DeviceZoneScopedN("MCAST2");
            // Mcast2: NCRISC sends from input core, BRISC receives on matmul2 cores, TRISC no-op
            deepseek_b1_ops::Mcast::
                Op<McastCTArgs, Core::is_input_core, Core::is_matmul2_core, Core::is_matmul2_core, true>
                    mcast2;
            mcast2(mcast2_args);
        }

        // ========================================================================
        // Matmul2: matmul2_input[1, 1536] @ matmul2_weights[1536, N]
        // N = 12288 for P150 (96 cores * 4 tiles * 32) or 11264 for non-P150
        // Each core computes 1x4 output tiles (4 1x32 tiles)
        // ========================================================================
        {
            DeviceZoneScopedN("MATMUL2");
            // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
            // On Qnope cores: output stays in matmul2_output_cb for matmul3 input
            // On Qrope cores: output goes to matmul2_output_cb for RoPE input
            deepseek_b1_ops::Matmul::Op<Matmul2CTArgs, Core::is_matmul2_core, true, false> matmul2;
            matmul2(matmul2_args);
        }

        {
            DeviceZoneScopedN("Q_HEADS") static_assert(
                !(Core::is_qnope_core && Core::is_qrope_core), "Core cannot be both QNOPE and QROPE");

            // ========================================================================
            // Matmul3 (QNoPE): matmul3_input[64, 1, 128] @ matmul3_weights[64, 128, 512] -> matmul3_output[64, 1, 512]
            // 64 cores (8x8 grid) each compute 1x16 output tiles (16 1x32 tiles)
            // ========================================================================
            {
                DeviceZoneScopedN("QNOPE/MATMUL3");
                // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
                deepseek_b1_ops::Matmul::Op<Matmul3CTArgs, Core::is_qnope_core, true, false> matmul3;
                matmul3(matmul3_args);
            }

            // ========================================================================
            // RoPE (Qrope): Applies rotary position embedding to Qrope heads
            // Reads from matmul2_output_cb, writes to qrope_output_cb
            // ========================================================================
            {
                DeviceZoneScopedN("QROPE");
                deepseek_b1_ops::Rope::Op<QRopeCTArgs, Core::is_qrope_core> rope;
                rope(qrope_args);
            }

            // ========================================================================
            // CreateQHeads: 3-phase QNOPE/QROPE -> SDPA transfer with tilization
            // Phase 1: QNOPE first 256 elements → [8, 256] row-major → 8 tiles
            // Phase 2: QNOPE second 256 elements → [8, 256] row-major → 8 tiles
            // Phase 3: QROPE 64 elements per head → [8, 64] row-major → 2 tiles
            // Senders write to intermediate CB, TRISC tilizes to output CB
            // NCRISC sends from qnope/qrope cores, BRISC receives on sdpa input cores, TRISC no-op
            // ========================================================================
            {
                DeviceZoneScopedN("CREATE_Q_HEADS");
                // CreateQHeads Op configuration:
                // - IsSenderCore: is_qnope_core || is_qrope_core
                // - IsReceiverCore: is_sdpa_input_core
                // - pop_src: true (pop source CB after sending)
                constexpr bool is_create_q_heads_sender = Core::is_qnope_core || Core::is_qrope_core;
                deepseek_b1_ops::CreateQHeads::
                    Op<CreateQHeadsCTArgs, is_create_q_heads_sender, Core::is_sdpa_input_core, false, true>
                        create_q_heads;
                create_q_heads(create_q_heads_args);
            }
        }
        // ====================================================================
        // KV Cache Branch
        // Non-owning SP devices skip the entire branch and just signal the
        // KV-cache-ready semaphore so FlashMLA can proceed.
        // ====================================================================
        deepseek_b1_ops::KVCacheUpdate::Op<Core::is_kv_rmsnorm_core, Core::is_krope_core> kv_cache_update;
        kv_cache_update.set_local_cur_pos(kv_cache_update_args, local_cur_pos);
        if (!skip_kv_cache_update) {
            DeviceZoneScopedN("KV CACHE");
            // ================================================================
            // DKV Matmul: 9x2 grid, each core handles 1 head of 32 dim
            // ================================================================
            {
                DeviceZoneScopedN("DKV_MATMUL");
                deepseek_b1_ops::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, false, false> dkv_matmul;
                dkv_matmul(dkv_matmul_args);
            }

            // ================================================================
            // Gather: dkv matmul cores (senders) -> rmsnorm core (receiver)
            // ================================================================
            {
                DeviceZoneScopedN("DKV_GATHER");
                deepseek_b1_ops::Gather::Op<Core::is_knope_core, Core::is_kv_rmsnorm_core, true> dkv_gather;
                dkv_gather(dkv_gather_args);
            }

            // ================================================================
            // RMSNorm: Apply RMSNorm to the gathered data
            // ================================================================
            {
                DeviceZoneScopedN("KV_RMSNORM");
                deepseek_b1_ops::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
                kv_rmsnorm(kv_rmsnorm_args);
            }

            // ================================================================
            // RoPE
            // ================================================================
            {
                DeviceZoneScopedN("K_ROPE");
                deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> krope;
                krope(krope_args);
            }

            // ================================================================
            // KV Cache Update: Write results to DRAM interleaved tensor
            // ================================================================
            {
                DeviceZoneScopedN("KV_CACHE_UPDATE");
                kv_cache_update(kv_cache_update_args);
            }
        }
        {
            DeviceZoneScopedN("KV_CACHE_SIGNAL_READY");
            kv_cache_update.signal_cache_ready(kv_cache_update_args);
        }

        DPRINT << " DONE KV CACHE UPDATE" << ENDL();

        // ====================================================================
        // Flash MLA: Compute
        // ====================================================================
        {
            DeviceZoneScopedN("FLASH_MLA");
            deepseek_b1_ops::FlashMLADecode::
                Op<FlashMLACTArgs, Core::is_mla_core, Core::is_kv_rmsnorm_core || Core::is_krope_core>
                    flash_mla;
            flash_mla.set_local_cur_pos(flash_mla_args, local_cur_pos);
            flash_mla(flash_mla_args);
        }
    }
    DPRINT << " DONE FLASH MLA" << ENDL();
    {
        // ========================================================================
        // Post SDPA: Reduce-to-All + Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL All-Reduce
        // ========================================================================
        {
            DeviceZoneScopedN("POST_SDPA");
            if constexpr (Core::is_sdpa_worker_core) {
                deepseek_b1_ops::SdpaReduceWorker::Op<SdpaReduceWorkerCTArgs> sdpa_reduce_worker;
                sdpa_reduce_worker(sdpa_reduce_worker_args);
            }
            DPRINT << " DONE SDPA REDUCE WORKER" << ENDL();
            if constexpr (Core::is_sdpa_forwarder_core) {
                deepseek_b1_ops::SdpaReduceForwarder::Op<SdpaReduceForwarderCTArgs> sdpa_reduce_forwarder;
                sdpa_reduce_forwarder(sdpa_reduce_forwarder_args);
            }
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::is_matmul4_core) {
                constexpr uint32_t scatter_arrival_semaphore_id =
                    get_named_compile_time_arg_val("scatter_arrival_semaphore_id");
                volatile tt_l1_ptr uint32_t* scatter_arrival_sem_addr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scatter_arrival_semaphore_id));
                DPRINT << " wait for scatter arrival semaphore" << ENDL();
                noc_semaphore_wait(scatter_arrival_sem_addr, 1);
                noc_semaphore_set(scatter_arrival_sem_addr, 0);
                DPRINT << " scatter arrival semaphore waited" << ENDL();
                constexpr uint32_t matmul4_in0 = get_named_compile_time_arg_val("matmul4_in0");
                constexpr uint32_t matmul4_k_num_tiles = get_named_compile_time_arg_val("matmul4_k_num_tiles");
                unified_kernels::setup_sharded_buffer(matmul4_in0, matmul4_k_num_tiles);
            }
#endif
        }
        {
            DeviceZoneScopedN("MATMUL4");
            deepseek_b1_ops::Matmul::Op<Matmul4CTArgs, Core::is_matmul4_core, true, false> matmul4;
            matmul4(matmul4_args);
        }
        {
            DeviceZoneScopedN("GATHER2");
            deepseek_b1_ops::Gather::Op<Core::is_matmul4_core, Core::is_gather_receiver_core, true, true> gather2;
            gather2(gather2_args);
        }

        constexpr bool is_mcast3_grid_core = Core::is_mcast3_receiver_core && !Core::is_gather_receiver_core;
        deepseek_b1_ops::Mcast::
            Op<Mcast3CTArgs, Core::is_gather_receiver_core, is_mcast3_grid_core, Core::is_matmul5_core, true>
                mcast3;
        {
            DeviceZoneScopedN("MCAST3");
            mcast3(mcast3_args);
        }

        {
            DeviceZoneScopedN("MATMUL5");
            deepseek_b1_ops::Matmul::Op<Matmul5CTArgs, Core::is_matmul5_core, true, false> matmul5;
            matmul5(matmul5_args);
        }
        {
            DeviceZoneScopedN("GATHER3");
            deepseek_b1_ops::Gather::Op<Core::is_matmul5_core, Core::is_gather_receiver_core, true, true> gather3;
            gather3(gather3_args);
        }

#if defined(COMPILE_FOR_BRISC)
        // Signal CCL sender that gather3 is complete (gather receiver only)
        if constexpr (Core::is_gather_receiver_core && Core::is_ccl_receiver_core) {
            static_assert(noc_mode == DM_DYNAMIC_NOC, "CCL signal must be sent on dynamic NOC");
            constexpr uint8_t CCL_SIGNAL_NOC = 0;
            constexpr uint32_t gather3_completion_semaphore_id =
                get_named_compile_time_arg_val("gather3_completion_semaphore_id");
            constexpr uint32_t ccl_sender_noc_x = get_named_compile_time_arg_val("ccl_sender_noc_x");
            constexpr uint32_t ccl_sender_noc_y = get_named_compile_time_arg_val("ccl_sender_noc_y");
            uint64_t ccl_sender_semaphore_addr = get_noc_addr(
                ccl_sender_noc_x, ccl_sender_noc_y, get_semaphore(gather3_completion_semaphore_id), CCL_SIGNAL_NOC);
            noc_semaphore_inc(ccl_sender_semaphore_addr, 1, CCL_SIGNAL_NOC);
            noc_async_atomic_barrier(CCL_SIGNAL_NOC);
        }
#endif

#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_ccl_sender_core) {
            DeviceZoneScopedN("CCL_SENDER_READ");

            constexpr uint32_t gather3_completion_semaphore_id =
                get_named_compile_time_arg_val("ccl_sender_gather3_completion_semaphore_id");
            volatile tt_l1_ptr uint32_t* gather3_completion_semaphore_addr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather3_completion_semaphore_id));
            noc_semaphore_wait(gather3_completion_semaphore_addr, 1);
            noc_semaphore_set(gather3_completion_semaphore_addr, 0);

            deepseek_b1_ops::AllReduceSender::Op<CCLSenderReaderCTArgs, DummyWriterCTArgs> ccl_sender_reader;
            ccl_sender_reader(ccl_sender_args);
        }

        if constexpr (Core::is_ccl_receiver_core) {
            DeviceZoneScopedN("CCL_RECEIVER_WAIT");
            deepseek_b1_ops::AllReduceReceiver::Op<CCLReceiverReaderCTArgs, DummyComputeCTArgs> ccl_receiver_reader;
            ccl_receiver_reader(ccl_receiver_args);
        }

#elif defined(COMPILE_FOR_BRISC)
        if constexpr (Core::is_ccl_sender_core) {
            DeviceZoneScopedN("CCL_SENDER_SEND");

            deepseek_b1_ops::AllReduceSender::Op<DummyReaderCTArgs, CCLSenderWriterCTArgs> ccl_sender_writer;
            ccl_sender_writer(ccl_sender_args);
        }
        // CCL Receiver BRISC is no-op

#elif defined(COMPILE_FOR_TRISC)
        if constexpr (Core::is_ccl_receiver_core) {
            DeviceZoneScopedN("CCL_RECEIVER_COMPUTE");

            deepseek_b1_ops::AllReduceReceiver::Op<DummyReaderCTArgs, CCLReceiverComputeCTArgs> ccl_receiver_compute;
            ccl_receiver_compute(ccl_receiver_args);
        }
        // CCL Sender TRISC is no-op
#endif
    }

    DPRINT << " DONE POST_SDPA" << ENDL();

    // ========================================================================
    // Phase 2: CB Reconfiguration — attention_block layout → MOE layout
    // ========================================================================
#if !defined(UCK_CHLKC_MATH)
    {
        constexpr uint32_t cb_config_l1_addr = get_named_compile_time_arg_val("reconfig_cb_config_l1_addr");
        uint32_t tt_l1_ptr* cb_config = reinterpret_cast<uint32_t tt_l1_ptr*>(cb_config_l1_addr);
        unified_kernels::reconfig_cb_interfaces(cb_config);
    }
#if defined(COMPILE_FOR_NCRISC)
    setup_moe_sharded_buffers();
#endif
#endif

    DPRINT << " DONE CB_RECONFIG" << ENDL();

    // ========================================================================
    // Phase 3: MoE Operations
    // ========================================================================

    deepseek_b1_ops::Mcast::Op<
        Moe::Routed::ResidualMcastCTArgs,
        Core::is_sender_core,
        Core::is_mcast_grid_core,
        Core::Shared::is_mcast_receiver_core,
        false>
        residual_mcast;
    residual_mcast.init(moe.routed.residual_mcast_args);

    deepseek_b1_ops::Mcast::Op<
        Moe::Routed::McastCTArgs,
        Core::is_sender_core,
        Core::is_mcast_grid_core,
        Core::is_input_mcast_receiver,
        true>
        moe_mcast;

    {
        DeviceZoneScopedN("MOE_RESIDUAL_MCAST");
        residual_mcast(moe.routed.residual_mcast_args);
    }

    {
        DeviceZoneScopedN("MOE_RMSNORM");
        deepseek_b1_ops::RMSNorm::Op<Moe::Routed::RMSNormCTArgs, Core::is_sender_core, false> moe_rmsnorm;
        moe_rmsnorm(moe.routed.rmsnorm_args);
    }

    {
        DeviceZoneScopedN("MOE_INPUT_MCAST");
        moe_mcast(moe.routed.mcast_args);
    }

#ifdef ENABLE_ROUTING
    {
        DeviceZoneScopedN("MOE_GATE_MATMUL");
        deepseek_b1_ops::Matmul::Op<Moe::Routed::GateMMCTArgs, Core::Routed::is_gate_mm_core, false, false> gate_mm;
        gate_mm(moe.routed.gate_mm_args);
    }
    {
        DeviceZoneScopedN("MOE_GATE_GATHER");
        deepseek_b1_ops::MoeGather::Op<Core::Routed::is_gate_mm_core, Core::is_sender_core, true> moe_gather;
        moe_gather(moe.routed.gather_args);
    }
#endif

    {
        DeviceZoneScopedN("MOE_SHARED_GU_MATMUL");
        deepseek_b1_ops::KNSlicedMatmul::
            Op<Moe::Shared::GUMatmulCTArgs, Core::Shared::is_compute_core, !Core::Routed::is_gate_proj_core, false>
                shared_gu_matmul;
        shared_gu_matmul(moe.shared.gu_matmul_args);
    }

#ifdef ENABLE_ROUTING
    {
        DeviceZoneScopedN("MOE_GATE");
        deepseek_b1_ops::DeepseekMoeGate::Op<Moe::Routed::GateCTArgs, Core::is_sender_core> gate;
        gate();
    }
    {
        DeviceZoneScopedN("MOE_MCAST_INDEX");
        deepseek_b1_ops::Mcast::Op<
            Moe::Routed::McastCTArgs,
            Core::is_sender_core,
            Core::is_mcast_grid_core,
            Core::Routed::is_gate_proj_core,
            true>
            index_mcast;
        index_mcast(moe.routed.index_mcast_args);
    }
    {
        DeviceZoneScopedN("MOE_MCAST_EXPERT_SCALE");
        deepseek_b1_ops::Mcast::Op<
            Moe::Routed::McastCTArgs,
            Core::is_sender_core,
            Core::is_mcast_grid_core,
            Core::Routed::is_gate_proj_core,
            true>
            expert_scale_mcast;
        expert_scale_mcast(moe.routed.expert_scale_mcast_args);
    }
#endif

    {
        DeviceZoneScopedN("MOE_SHARED_GATE_GATHER");
        deepseek_b1_ops::MoeGather::
            Op<Core::Shared::is_gate_compute_core, Core::Shared::is_gated_reduce_core, true, true>
                shared_gate_gather;
        shared_gate_gather(moe.shared.ag_args);
    }
    {
        DeviceZoneScopedN("MOE_SHARED_UP_GATHER");
        deepseek_b1_ops::MoeGather::Op<Core::Shared::is_up_compute_core, Core::Shared::is_gated_reduce_core, true, true>
            shared_up_gather;
        shared_up_gather(moe.shared.bg_args);
    }
    {
        DeviceZoneScopedN("MOE_SHARED_GATED_REDUCE");
        deepseek_b1_ops::GatedReduce::Op<Moe::Shared::GatedReduceCTArgs, Core::Shared::is_gated_reduce_core>
            gated_reduce;
        gated_reduce(moe.shared.gated_reduce_args);
    }

    {
        DeviceZoneScopedN("MOE_GATE_PROJ");
        constexpr uint32_t gate_proj_cb_in1_addr = get_named_compile_time_arg_val("gate_proj_in1_buf_addr");
        deepseek_b1_ops::DRAMStreamingMatmul::
            Op<Moe::Routed::GateProjCTArgs, Core::Routed::is_gate_proj_core, false, true, gate_proj_cb_in1_addr>
                gate_proj_mm;
        gate_proj_mm();
    }
    {
        DeviceZoneScopedN("MOE_UP_PROJ");
        constexpr uint32_t cb_in1_addr = get_named_compile_time_arg_val("gate_proj_in1_buf_addr");
        deepseek_b1_ops::DRAMStreamingMatmul::
            Op<Moe::Routed::UpProjCTArgs, Core::Routed::is_gate_proj_core, true, true, cb_in1_addr, false, true>
                up_proj;
        up_proj();
    }
    {
        DeviceZoneScopedN("MOE_MUL");
        deepseek_b1_ops::EltwiseMul::Op<Moe::Routed::MulCTArgs, Core::Routed::is_gate_proj_core> mul_op;
        mul_op();
    }

    {
        DeviceZoneScopedN("MOE_DOWN_PROJ_GATHER");
        deepseek_b1_ops::MoeGather::Op<Core::Routed::is_gate_proj_core, Core::is_sender_core, true, true>
            down_proj_gather;
        down_proj_gather(moe.routed.down_proj_gather_args);
    }
    {
        DeviceZoneScopedN("MOE_DOWN_PROJ_MCAST");
        deepseek_b1_ops::Mcast::Op<
            Moe::Routed::McastCTArgs,
            Core::is_sender_core,
            Core::is_mcast_grid_core,
            Core::Routed::is_gate_proj_core,
            true>
            down_proj_mcast;
        down_proj_mcast(moe.routed.down_proj_mcast_args);
    }
    {
        DeviceZoneScopedN("MOE_DOWN_PROJ");
        constexpr uint32_t down_proj_cb_in1_addr = get_named_compile_time_arg_val("down_proj_in1_buf_addr");
        deepseek_b1_ops::DRAMStreamingMatmul::
            Op<Moe::Routed::DownProjCTArgs, Core::Routed::is_gate_proj_core, true, true, down_proj_cb_in1_addr, true>
                down_proj;
        down_proj();
    }

    {
        DeviceZoneScopedN("MOE_SHARED_DOWN_MCAST");
        deepseek_b1_ops::Mcast::Op<
            Moe::Shared::DownMcastCTArgs,
            Core::is_sender_core,
            Core::is_mcast_grid_core,
            Core::Shared::is_mcast_receiver_core,
            true>
            shared_down_mcast;
        shared_down_mcast(moe.shared.down_mcast_args);
    }
    {
        DeviceZoneScopedN("MOE_SHARED_DOWN_MATMUL");
        deepseek_b1_ops::Matmul::Op<Moe::Shared::DownMatmulCTArgs, Core::Shared::is_mcast_receiver_core, true, false>
            shared_down_matmul;
        shared_down_matmul(moe.shared.down_matmul_args);
    }
    {
        DeviceZoneScopedN("MOE_SHARED_RESIDUAL_ADD");
        deepseek_b1_ops::ResidualAdd::Op<Moe::Shared::ResidualAddCTArgs, Core::Shared::is_mcast_receiver_core>
            shared_residual_add;
        shared_residual_add(moe.shared.residual_add_args);
    }
    {
        DeviceZoneScopedN("MOE_SHARED_OUTPUT_GATHER");
        deepseek_b1_ops::MoeGather::Op<Core::Shared::is_mcast_receiver_core, Core::is_sender_core, true, true>
            shared_output_gather;
        shared_output_gather(moe.shared.og_args);
    }
    {
        DeviceZoneScopedN("MOE_SHARED_OUTPUT_MCAST");
        deepseek_b1_ops::Mcast::Op<
            Moe::Shared::OutputMcastCTArgs,
            Core::is_sender_core,
            Core::is_mcast_grid_core,
            Core::Routed::is_gate_proj_core,
            true>
            shared_output_mcast;
        shared_output_mcast(moe.shared.output_mcast_args);
    }

    {
        DeviceZoneScopedN("MOE_ELTWISE_ADD");
        constexpr bool add_pop_output =
#ifdef ENABLE_REDUCE_TO_ONE
            false;
#else
            true;
#endif
        deepseek_b1_ops::EltwiseAdd::Op<Moe::Routed::AddCTArgs, Core::Routed::is_gate_proj_core, true, add_pop_output>
            add_op;
        add_op();
    }

#ifdef ENABLE_REDUCE_TO_ONE
    {
        DeviceZoneScopedN("MOE_REDUCE_TO_ONE");
        constexpr bool is_reduce_core = Core::is_reduce_worker_core || Core::is_reduce_fabric_core;
        deepseek_b1_ops::ReduceToOneB1::Op<Moe::Routed::ReduceToOneCTArgs, is_reduce_core, true> reduce_op;
        reduce_op(moe.routed.reduce_rt_args);
    }

#if defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_reduce_fabric_core) {
        constexpr uint32_t sync_sem_addr = get_named_compile_time_arg_val("reduce_sync_sem_addr");
        constexpr uint32_t sync_noc_x = get_named_compile_time_arg_val("reduce_sync_noc_x");
        constexpr uint32_t sync_noc_y = get_named_compile_time_arg_val("reduce_sync_noc_y");
        uint64_t sync_sem_noc_addr = get_noc_addr(sync_noc_x, sync_noc_y, sync_sem_addr);
        noc_semaphore_inc(sync_sem_noc_addr, 1);
    }
#elif defined(COMPILE_FOR_NCRISC)
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t sync_sem_addr = get_named_compile_time_arg_val("reduce_sync_sem_addr");
        constexpr uint32_t num_fabric_cores = get_named_compile_time_arg_val("reduce_sync_num_fabric_cores");
        volatile tt_l1_ptr uint32_t* sync_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_sem_addr);
        noc_semaphore_wait(sync_sem_ptr, num_fabric_cores);
        noc_semaphore_set(sync_sem_ptr, 0);
    }
#endif
#endif

    // ====================================================================
    // Mcast: Teardown persistent mcast
    // ====================================================================
    {
        DeviceZoneScopedN("MCAST_TEARDOWN");
        mcast.teardown();
    }

    residual_mcast.teardown();

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
