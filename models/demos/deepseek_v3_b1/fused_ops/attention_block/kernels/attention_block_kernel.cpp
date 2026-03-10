// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Pre-SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: CCL Broadcast + RMSNorm + Mcast + Matmul + Gather + RMSNorm2 + Mcast2 + Matmul2 + Matmul3 + RoPE +
// CreateQHeads
// - NCRISC: CCL Broadcast Reader + RMSNorm reader + Mcast receiver (on matmul cores), Matmul reader + Gather sender (on
// matmul cores),
//           RMSNorm2 reader + Mcast2 receiver (on matmul2 cores), Matmul2 reader (on matmul2 cores),
//           Matmul3 reader (on qnope cores), RoPE reader (on qrope cores), CreateQHeads sender (on qnope/qrope cores)
// - BRISC: CCL Broadcast Writer + RMSNorm writer + Mcast sender (on input core), Matmul writer (on matmul cores),
// Gather receiver (on
//          input core), Mcast2 sender (on input core), Matmul2 writer (on matmul2 cores),
//          CreateQHeads receiver (on sdpa input cores) - matching gather pattern: NCRISC sender, BRISC receiver
// - TRISC: RMSNorm compute (on input core), Matmul compute (on matmul cores), RMSNorm2 compute (on input core),
//          Matmul2 compute (on matmul2 cores), Matmul3 compute (on qnope cores), RoPE compute (on qrope cores)
//
// Matmul2 output uses interleaved Qnope/Qrope layout (with shuffled weights):
// - Grid: 12 cols × 8 rows = 96 cores (P150)
// - Qnope cores (cols 0-7): 64 cores, 1 head × 128 elements per core
// - Qrope cores (cols 8-11): 32 cores, 2 heads × 64 elements per core
// - Each row: [Qnope heads 0-7 (1024)] [Qrope heads 0-7 (512)] = 1536 elements
// - Total: 8 rows × 1536 = 12288 elements

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

// Compile-time role flags for dead code elimination via if constexpr
// Defined at namespace scope (local classes cannot have static data members)
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
    static constexpr bool is_full_mcast_grid_core = get_named_compile_time_arg_val("is_full_mcast_grid_core") == 1;
    static constexpr bool is_matmul_core = get_named_compile_time_arg_val("is_matmul_core") == 1;
    static constexpr bool is_matmul2_core = get_named_compile_time_arg_val("is_matmul2_core") == 1;
    // Qnope/Qrope core differentiation for interleaved Q head layout after matmul2
    // Qnope cores: 64 cores (8x8 grid), each handles 1 head of 128 elements
    // Qrope cores: 32 cores (4x8 grid), each handles 2 heads of 64 elements
    static constexpr bool is_qnope_core = get_named_compile_time_arg_val("is_qnope_core") == 1;
    static constexpr bool is_qrope_core = get_named_compile_time_arg_val("is_qrope_core") == 1;
    // SDPA Input core: receives interleaved QNOPE/QROPE, runs create q heads (4×2 grid = 8 cores)
    static constexpr bool is_sdpa_input_core = get_named_compile_time_arg_val("is_sdpa_input_core") == 1;

    // DKV Matmul core: 9x2 grid, each core handles 1 head of 32 dim
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

    // Post SDPA
    // SDPA output cores 8 cores - run SDPA reduction and scatter
    static constexpr bool is_sdpa_worker_core = get_named_compile_time_arg_val("is_sdpa_worker_core") == 1;
    // SDPA forwarder cores (6,9), (7,9) = 2 cores - forward fabric packets for SDPA CCL
    static constexpr bool is_sdpa_forwarder_core = get_named_compile_time_arg_val("is_sdpa_forwarder_core") == 1;

    // First matmul on kv_b2 grid (5x8 + 12x2 = 64 cores) - receives scatter data from SDPA workers
    static constexpr bool is_matmul4_core = get_named_compile_time_arg_val("is_matmul4_core") == 1;
    // Gather core (12, 9) - receives gather2, sends mcast3, receives gather3, CCL receiver
    static constexpr bool is_gather_receiver_core = get_named_compile_time_arg_val("is_gather_receiver_core") == 1;
    // Active matmul5 cores (112 cores: o_proj grid 12x8 + 8x2)
    static constexpr bool is_matmul5_core = get_named_compile_time_arg_val("is_matmul5_core") == 1;
    // CCL sender core (11, 9) - reads from gather core, sends via fabric
    static constexpr bool is_ccl_sender_core = get_named_compile_time_arg_val("is_ccl_sender_core") == 1;
    // CCL receiver core = gather core (12, 9) - receives remote data, performs reduction
    static constexpr bool is_ccl_receiver_core = get_named_compile_time_arg_val("is_ccl_receiver_core") == 1;
};

void kernel_main() {
    // ============================================================================
    // NCRISC (Reader + Mcast Receiver) - ReaderConfigDescriptor compiles as NCRISC
    // Named compile-time args: rmsnorm reader, mcast receiver, matmul reader, gather sender
    // Runtime args: []
    // ============================================================================
    constexpr uint32_t num_iterations = get_named_compile_time_arg_val("num_iterations");
    constexpr uint32_t cb_config_l1_addr = get_named_compile_time_arg_val("mla_reconfig_cb_config_l1_addr");
    uint32_t tt_l1_ptr* cb_config = reinterpret_cast<uint32_t tt_l1_ptr*>(cb_config_l1_addr);
    // This is needed at the start because mcast is getting the cb ptrs for src/dst addresses
    unified_kernels::reconfig_cb_interfaces(cb_config);

    uint32_t per_core_rta_arg_idx = 0;
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type aliases (required for Op templates)
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
        get_named_compile_time_arg_val("mcast2_data_receiver_semaphore_addr"),
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
        .cos_sin_cb = get_named_compile_time_arg_val("qkv_rope_cos_sin_cb"),
        .cos_tensor_address = get_named_compile_time_arg_val("qrope_cos_tensor_address"),
        .sin_tensor_address = get_named_compile_time_arg_val("qrope_sin_tensor_address"),
        .position_ids_tensor_address = get_named_compile_time_arg_val("qrope_position_ids_tensor_address"),
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
        .cos_sin_cb = get_named_compile_time_arg_val("krope_cos_sin_cb"),
        .cos_tensor_address = get_named_compile_time_arg_val("krope_cos_tensor_address"),
        .sin_tensor_address = get_named_compile_time_arg_val("krope_sin_tensor_address"),
        .position_ids_tensor_address = get_named_compile_time_arg_val("krope_position_ids_tensor_address"),
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
        get_common_arg_val<uint32_t>(15),  // gather2_receiver_data_addr
        get_named_compile_time_arg_val("gather2_sender_idx"),
    };

    // Mcast3 receiver args
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
        get_common_arg_val<uint32_t>(16),  // gather3_receiver_data_addr
        get_named_compile_time_arg_val("gather3_sender_idx"),
    };

    // ========================================================================o
    // All CCLs are placed after other ops setup due to appended fabric rtas
    // ========================================================================
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
    }

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
    using DummyWriterCTArgs = deepseek_b1_ops::AllReduceSender::WriterCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>;
    // Dummy ComputeCTArgs - not used by NCRISC but needed for Op template
    using DummyComputeCTArgs = deepseek_b1_ops::AllReduceReceiver::ComputeCTArgs<0, 0, 0, 0, 0, 0, 0>;
    deepseek_b1_ops::AllReduceSender::RTArgs ccl_sender_args{};
    deepseek_b1_ops::AllReduceReceiver::RTArgs ccl_receiver_args{};

    if constexpr (Core::is_ccl_sender_core) {
        ccl_sender_args = {
            .tensor_address = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
    }

    if constexpr (Core::is_ccl_receiver_core) {
        ccl_receiver_args = {
            .sender_semaphore_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
    }
// ============================================================================
// BRISC (Writer + Mcast Sender) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: bcast writer + rmsnorm writer, mcast sender, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)

    // CTArgs type aliases (required for Op templates)
    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;  // BRISC is no-op
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_input_core && Core::is_full_mcast_grid_core>;  // loopback = false

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
        get_named_compile_time_arg_val("mcast2_data_receiver_semaphore_addr"),
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

    // Matmul4/2 CTArgs (BRISC is no-op for matmul)
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
    constexpr uint32_t mcast3_src_cb = get_named_compile_time_arg_val("mcast3_src_cb");
    constexpr uint32_t mcast3_dst_cb = get_named_compile_time_arg_val("mcast3_dst_cb");
    deepseek_b1_ops::Mcast::SenderArgs mcast3_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("mcast_data_sender_semaphore_addr"),
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

    // ========================================================================o
    // All CCLs are placed after other ops setup due to appended fabric rtas
    // ========================================================================
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

    using SdpaReduceWorkerCTArgs = deepseek_b1_ops::SdpaReduceWorker::WriterCTArgs<
        get_named_compile_time_arg_val("sdpa_cb_local_l"),
        get_named_compile_time_arg_val("sdpa_cb_local_ms"),
        get_named_compile_time_arg_val("sdpa_cb_r1_result_l"),
        get_named_compile_time_arg_val("sdpa_cb_r1_result_ms"),
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
            // scatter_arrival_enabled=1, so we need to pass the semaphore address
            .scatter_arrival_sem_addr = get_semaphore(get_named_compile_time_arg_val("scatter_arrival_semaphore_id")),
        };
        per_core_rta_arg_idx += SdpaReduceWorkerCTArgs::scatter_num_rows * 2;  // x, y value per dest
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
    }

    // CCL Sender BRISC CTArgs (sends via fabric)
    using CCLSenderWriterCTArgs = deepseek_b1_ops::AllReduceSender::WriterCTArgs<
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
        ccl_sender_args = {
            .receiver_base_address = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
            .receive_semaphore_addr = get_arg_val<uint32_t>(per_core_rta_arg_idx++),
        };
        uint32_t fabric_args_offset = get_arg_val<uint32_t>(per_core_rta_arg_idx++);
        ccl_sender_args.fabric_args_start_index = per_core_rta_arg_idx;
        per_core_rta_arg_idx += fabric_args_offset;
    }

// ============================================================================
// TRISC (Compute) - ComputeConfigDescriptor compiles as TRISC
// Named compile-time args: rmsnorm compute, matmul compute
// ============================================================================
#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type aliases (required for Op templates)

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
        get_common_arg_val<uint32_t>(0),   // epsilon
        get_common_arg_val<float>(1),      // scalar (1/sqrt(7168))
        get_common_arg_val<uint32_t>(15),  // rmsnorm2_gamma_addr
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
        get_common_arg_val<uint32_t>(0),   // epsilon (same as rmsnorm1)
        get_common_arg_val<float>(2),      // scalar (1/sqrt(1536))
        get_common_arg_val<uint32_t>(16),  // rmsnorm2_gamma_addr
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
        get_common_arg_val<uint32_t>(11),  // matmul3_weights_addr
    };

    // Qrope CTArgs type alias
    using QRopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("qrope_Wt"), get_named_compile_time_arg_val("qrope_Ht")>;

    // Qrope compute args (from compile-time args)
    deepseek_b1_ops::Rope::ComputeArgs qrope_args{
        get_named_compile_time_arg_val("qrope_in_cb"),  // Input from matmul2 output
        get_named_compile_time_arg_val("qkv_rope_cos_sin_cb"),
        get_named_compile_time_arg_val("qrope_trans_mat_cb"),
        get_named_compile_time_arg_val("qrope_rotated_in_interm_cb"),
        get_named_compile_time_arg_val("qrope_cos_sin_interm_cb"),
        get_named_compile_time_arg_val("qrope_output_cb"),
        get_common_arg_val<uint32_t>(14),  // qrope_trans_mat_addr
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

    // DKV Matmul compute args (from compile-time args, passed to op as runtime args)
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
        get_common_arg_val<uint32_t>(0),   // epsilon
        get_common_arg_val<float>(3),      // kv_scalar (1/sqrt(512))
        get_common_arg_val<uint32_t>(17),  // kv_rmsnorm_gamma_addr
    };

    using K_RopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("krope_Wt"), get_named_compile_time_arg_val("krope_Ht")>;

    // CB indices (passed as runtime args to ComputeArgs)
    constexpr uint32_t krope_input_cb = get_named_compile_time_arg_val("krope_in_cb");
    constexpr uint32_t krope_cos_sin_cb = get_named_compile_time_arg_val("krope_cos_sin_cb");
    constexpr uint32_t krope_trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb");
    constexpr uint32_t krope_rotated_in_interm_cb = get_named_compile_time_arg_val("krope_rotated_in_interm_cb");
    constexpr uint32_t krope_cos_sin_interm_cb = get_named_compile_time_arg_val("krope_cos_sin_interm_cb");
    constexpr uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");

    // Compute args: all CB indices
    deepseek_b1_ops::Rope::ComputeArgs krope_args{
        .in_cb = krope_input_cb,
        .cos_sin_cb = krope_cos_sin_cb,
        .trans_mat_cb = krope_trans_mat_cb,
        .rotated_in_interm_cb = krope_rotated_in_interm_cb,
        .cos_sin_interm_cb = krope_cos_sin_interm_cb,
        .out_cb = krope_output_cb,
        .trans_mat_address_override = get_common_arg_val<uint32_t>(14),
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
        get_common_arg_val<uint32_t>(12),  // matmul4_weights_addr
    };

    // Gather2 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather2_args{};

    // Mcast3 CTArgs (no-op)
    deepseek_b1_ops::Mcast::ComputeArgs mcast3_args{};

    // Matmul5 CTArgs
    using Matmul5CTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul5_out_w_per_core")>;
    deepseek_b1_ops::Matmul::ComputeArgs matmul5_args{
        get_named_compile_time_arg_val("matmul5_in0"),
        get_named_compile_time_arg_val("matmul5_in1"),
        get_named_compile_time_arg_val("matmul5_out"),
        get_named_compile_time_arg_val("matmul5_k_num_tiles"),
        get_common_arg_val<uint32_t>(13),  // matmul5_weights_addr
    };

    // Gather3 compute args (no-op)
    deepseek_b1_ops::Gather::ComputeArgs gather3_args{};

    // CCL Broadcast CTArgs (no-op for TRISC)
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ComputeCTArgs;
    deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};

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

    using DummyReaderCTArgs = deepseek_b1_ops::AllReduceReceiver::ReaderCTArgs<0, 0, 0, 0, 0, 0, 0, 0, 0>;
    // Dummy ReaderCTArgs - not used by TRISC but needed for Op template
    deepseek_b1_ops::AllReduceReceiver::RTArgs ccl_receiver_args{};

    deepseek_compute_kernel_init();
#endif

    // Setup all tensor-backed sharded buffers (marks pre-loaded tiles as ready)
    auto setup_all_sharded_buffers = [&]() __attribute__((always_inline)) {
#if defined(COMPILE_FOR_NCRISC)
    // if constexpr (Core::is_input_core) {
    //     // Multi-device mode: NCRISC sets up gamma buffers while BRISC handles CCL
    //     // RMSNorm gamma buffer
    //     constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
    //     constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
    //     constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
    //     unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);

    //     // RMSNorm2 gamma buffer (3 tiles of 16x32)
    //     constexpr uint32_t rmsnorm2_gamma_cb = get_named_compile_time_arg_val("rmsnorm2_gamma_cb");
    //     constexpr uint32_t rmsnorm2_num_tiles = get_named_compile_time_arg_val("rmsnorm2_num_tiles");
    //     unified_kernels::setup_sharded_buffer(rmsnorm2_gamma_cb, rmsnorm2_num_tiles);
    // }
    // if constexpr (Core::is_qnope_core) {
    //     // Matmul3 CB indices and parameters from named compile-time args
    //     constexpr uint32_t matmul3_in1 = get_named_compile_time_arg_val("matmul3_in1");
    //     constexpr uint32_t matmul3_k_num_tiles = get_named_compile_time_arg_val("matmul3_k_num_tiles");
    //     constexpr uint32_t matmul3_out_w_per_core = get_named_compile_time_arg_val("matmul3_out_w_per_core");

    //     // Matmul3 weights (on Qnope cores, [128, 512] = 4 * 16 = 64 tiles per core)
    //     unified_kernels::setup_sharded_buffer(matmul3_in1, matmul3_k_num_tiles * matmul3_out_w_per_core);
    // }

    // if constexpr (Core::is_qrope_core) {
    //     constexpr uint32_t qrope_trans_mat_cb = get_named_compile_time_arg_val("qrope_trans_mat_cb");
    //     unified_kernels::setup_sharded_buffer(qrope_trans_mat_cb, 1);
    // }

    // if constexpr (Core::is_kv_rmsnorm_core) {
    //     // RMSNorm gamma (sharded weights)
    //     constexpr uint32_t kv_rmsnorm_gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
    //     constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
    //     unified_kernels::setup_sharded_buffer(kv_rmsnorm_gamma_cb, kv_rmsnorm_num_tiles);
    // }

    // if constexpr (Core::is_krope_core) {
    //     constexpr uint32_t krope_trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb");
    //     unified_kernels::setup_sharded_buffer(krope_trans_mat_cb, 1);
    // }

    // if constexpr (Core::is_matmul4_core) {
    //     constexpr uint32_t matmul4_in1 = get_named_compile_time_arg_val("matmul4_in1");
    //     constexpr uint32_t matmul4_k_num_tiles = get_named_compile_time_arg_val("matmul4_k_num_tiles");
    //     constexpr uint32_t matmul4_out_w_per_core = get_named_compile_time_arg_val("matmul4_out_w_per_core");
    //     unified_kernels::setup_sharded_buffer(matmul4_in1, matmul4_k_num_tiles * matmul4_out_w_per_core);
    // }

    // if constexpr (Core::is_matmul5_core) {
    //     constexpr uint32_t matmul5_in1 = get_named_compile_time_arg_val("matmul5_in1");
    //     constexpr uint32_t matmul5_k_num_tiles = get_named_compile_time_arg_val("matmul5_k_num_tiles");
    //     constexpr uint32_t matmul5_out_w_per_core = get_named_compile_time_arg_val("matmul5_out_w_per_core");
    //     unified_kernels::setup_sharded_buffer(matmul5_in1, matmul5_k_num_tiles * matmul5_out_w_per_core);
    // }
#endif
    };

#if defined(COMPILE_FOR_BRISC)
    uint32_t cur_pos_addr = get_common_arg_val<uint32_t>(1);
#elif defined(COMPILE_FOR_NCRISC)
    uint32_t cur_pos_addr = get_common_arg_val<uint32_t>(14);
#elif defined(COMPILE_FOR_TRISC)
    uint32_t cur_pos_addr = get_common_arg_val<uint32_t>(7);
#endif

    // ====================================================================
    // Mcast: Initialize persistent mcast
    // ====================================================================
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_input_core,
        // Receive on the whole grid. This is used to block downstream ccls
        Core::is_full_mcast_grid_core,
        Core::is_matmul_core || Core::is_dkv_matmul_core,
        true>
        mcast;
    {
        DeviceZoneScopedN("MCAST_INIT");
        mcast.init(mcast_args);
    }

    auto mla_body = [&]() __attribute__((always_inline)) {
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
            // Gamma CBs are already set up by NCRISC via setup_sharded_buffer
            constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
            constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

            cb_reserve_back(rmsnorm_input_cb, rmsnorm_num_tiles);
            cb_push_back(rmsnorm_input_cb, rmsnorm_num_tiles);
        }
#endif

        // SP position handling.
        // Read the global position from L1 and decide whether this device has
        // work / owns the current KV-cache slot. The normalized (device-local)
        // local_cur_pos is used for kv cache update and flash mla.
        volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cur_pos_addr);
        uint32_t cur_pos = pos_ptr[0];

        const auto [skip_attention, skip_kv_cache_update, local_cur_pos] = get_device_mla_work_assignment(
            cur_pos, Core::kv_cache_sp_device_idx, Core::kv_cache_device_chunk_size, Core::kv_cache_num_sp_devices);

        using FlashMLAOp = deepseek_b1_ops::FlashMLADecode::Op<FlashMLACTArgs, Core::is_mla_core>;

        if (!skip_attention) {
            // ====================================================================
            // Input core: RMSNorm + Mcast send
            // ====================================================================
            {
                DeviceZoneScopedN("RMSNORM");
                deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core, true> rmsnorm;
                rmsnorm(rmsnorm_args);
            }

            {
                DeviceZoneScopedN("MCAST");
                mcast(mcast_args);
            }

            // ====================================================================
            // Matmul operation
            // ====================================================================
            {
                DeviceZoneScopedN("MATMUL");
                deepseek_b1_ops::KNSlicedMatmul::Op<MatmulCTArgs, Core::is_matmul_core, false, false> matmul;
                matmul(matmul_args);
            }

            // ====================================================================
            // GatherReduce: matmul cores (senders) -> input core (receiver/reducer)
            // ====================================================================
            {
                DeviceZoneScopedN("GATHER");
                deepseek_b1_ops::GatherReduce::Op<Core::is_matmul_core, Core::is_input_core, Core::is_input_core, true>
                    gather_reduce;
                gather_reduce(gather_reduce_args);
            }

            // ====================================================================
            // RMSNorm2
            // ====================================================================
            {
                DeviceZoneScopedN("RMSNORM2");
                deepseek_b1_ops::RMSNorm::Op<RMSNorm2CTArgs, Core::is_input_core, true> rmsnorm2;
                rmsnorm2(rmsnorm2_args);
            }

            // ====================================================================
            // Mcast2: Broadcast rmsnorm2 output to matmul2 cores
            // ====================================================================
            {
                DeviceZoneScopedN("MCAST2");
                deepseek_b1_ops::Mcast::
                    Op<McastCTArgs, Core::is_input_core, Core::is_matmul2_core, Core::is_matmul2_core, true>
                        mcast2;
                mcast2(mcast2_args);
            }

            // ====================================================================
            // Matmul2
            // ====================================================================
            {
                DeviceZoneScopedN("MATMUL2");
                deepseek_b1_ops::Matmul::Op<Matmul2CTArgs, Core::is_matmul2_core, true, false> matmul2;
                matmul2(matmul2_args);
            }

            {
                DeviceZoneScopedN("Q_HEADS") static_assert(
                    !(Core::is_qnope_core && Core::is_qrope_core), "Core cannot be both QNOPE and QROPE");

                // ================================================================
                // Matmul3 (QNoPE)
                // ================================================================
                {
                    DeviceZoneScopedN("QNOPE/MATMUL3");
                    deepseek_b1_ops::Matmul::Op<Matmul3CTArgs, Core::is_qnope_core, true, false> matmul3;
                    matmul3(matmul3_args);
                }

                // ================================================================
                // RoPE (Qrope)
                // ================================================================
                {
                    DeviceZoneScopedN("QROPE");
                    deepseek_b1_ops::Rope::Op<QRopeCTArgs, Core::is_qrope_core> rope;
                    rope(qrope_args);
                }

                // ================================================================
                // CreateQHeads
                // ================================================================
                {
                    DeviceZoneScopedN("CREATE_Q_HEADS");
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
                    // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
                    deepseek_b1_ops::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, false, false> dkv_matmul;
                    dkv_matmul(dkv_matmul_args);
                }

                // ================================================================
                // Gather: dkv matmul cores (senders) -> rmsnorm core (receiver)
                // NCRISC sends from knope grid, BRISC receives on rmsnorm grid
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
                // BRISC handles writing from output CBs to DRAM
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

            // ====================================================================
            // Flash MLA: Compute
            // ====================================================================
            {
                DeviceZoneScopedN("FLASH_MLA");
                FlashMLAOp flash_mla;
                flash_mla.set_local_cur_pos(flash_mla_args, local_cur_pos);
                flash_mla(flash_mla_args);
            }
        } else {
            // This device has no sequence data (e.g. SP2/SP3 with seq_len = 2047 and per_device_chunk_size = 1024).
            // Push dummy tiles into the hand-off CBs so SDPA reduce does not hang.
            // TODO: Fuse the final SP reduce into Flash MLA and handle this internally,
            // eliminating the need for this explicit dummy push.
            if constexpr (Core::is_sdpa_worker_core) {
                FlashMLAOp::push_dummy_sdpa_inputs();
            }
        }

        // ========================================================================
        // Post SDPA: Reduce-to-All + Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL All-Reduce
        // ========================================================================
        {
            DeviceZoneScopedN("POST_SDPA");
            if constexpr (Core::is_sdpa_worker_core) {
                deepseek_b1_ops::SdpaReduceWorker::Op<SdpaReduceWorkerCTArgs> sdpa_reduce_worker;
                sdpa_reduce_worker(sdpa_reduce_worker_args);
            }
            if constexpr (Core::is_sdpa_forwarder_core) {
                deepseek_b1_ops::SdpaReduceForwarder::Op<SdpaReduceForwarderCTArgs> sdpa_reduce_forwarder;
                // We need to make sure both riscs wait for the initial broadcast CCL to complete since
                // reduce forwarder uses both riscs to send over fabric
                // The first mcast syncs the ncrisc, so we need to also make sure brisc waits for the first mcast
#if defined(COMPILE_FOR_NCRISC)
                volatile tt_l1_ptr uint32_t* sync_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"));
                // Make sure the value is different from the mcasted value
                // The wait below is for safety if this runs on the same core as another doing the same sync
                // If that is the case we don't actually need to do another sync
                noc_semaphore_wait(sync_sem, 0);
                noc_semaphore_set(sync_sem, 2);
#elif defined(COMPILE_FOR_BRISC)
                volatile tt_l1_ptr uint32_t* sync_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"));
                noc_semaphore_wait(sync_sem, 2);
                noc_semaphore_set(sync_sem, 0);
#endif
                sdpa_reduce_forwarder(sdpa_reduce_forwarder_args);
            }
#if defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::is_matmul4_core) {
                constexpr uint32_t scatter_arrival_semaphore_id =
                    get_named_compile_time_arg_val("scatter_arrival_semaphore_id");
                volatile tt_l1_ptr uint32_t* scatter_arrival_sem_addr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(scatter_arrival_semaphore_id));
                noc_semaphore_wait(scatter_arrival_sem_addr, 1);
                noc_semaphore_set(scatter_arrival_sem_addr, 0);
                constexpr uint32_t matmul4_in0 = get_named_compile_time_arg_val("matmul4_in0");
                constexpr uint32_t matmul4_k_num_tiles = get_named_compile_time_arg_val("matmul4_k_num_tiles");
                unified_kernels::setup_sharded_buffer(matmul4_in0, matmul4_k_num_tiles);
            }
#endif
        }
        // ========================================================================
        // Matmul4: [1, 512] x [512, 128] -> [1, 128] per core (kv_b2 grid)
        // ========================================================================
        {
            DeviceZoneScopedN("MATMUL4");
            deepseek_b1_ops::Matmul::Op<Matmul4CTArgs, Core::is_matmul4_core, true, false> matmul4;
            matmul4(matmul4_args);
        }

        // ========================================================================
        // Gather2: matmul4 cores (kv_b2 grid) -> gather core (12, 9)
        // Collects [1, 128] * 64 = [1, 8192]
        // ========================================================================
        {
            DeviceZoneScopedN("GATHER2");
            deepseek_b1_ops::Gather::Op<Core::is_matmul4_core, Core::is_gather_receiver_core, true, true> gather2;
            gather2(gather2_args);
        }

        // ========================================================================
        // Mcast3: gather core -> 13x10 mcast3 grid (130 cores)
        // Broadcasts [1, 8192] to each core in mcast3 grid
        // Source: gather2_dst_cb (CB 3), Destination: mcast3_dst_cb = matmul5_in0 (CB 4)
        // Note: 18 inactive grid cores only do semaphore handshake; only matmul5 cores do full CB receive
        // ========================================================================
        deepseek_b1_ops::Mcast::
            Op<McastCTArgs, Core::is_gather_receiver_core, Core::is_matmul5_core, Core::is_matmul5_core, true>
                mcast3;
        {
            DeviceZoneScopedN("MCAST3");
            mcast3(mcast3_args);
        }

        // ========================================================================
        // Matmul5: [1, 8192] x [8192, 64] -> [1, 64] per core (112 active cores)
        // Input: mcast3_dst_cb (CB 4), Weights: matmul5_in1 (CB 5), Output: matmul5_out (CB 6)
        // Only runs on 112 active cores (is_matmul5_core=true), 18 inactive cores skip
        // ========================================================================
        {
            DeviceZoneScopedN("MATMUL5");
            // pop_in0 = true (mcast3 output consumed), pop_in1 = false (weights persistent)
            deepseek_b1_ops::Matmul::Op<Matmul5CTArgs, Core::is_matmul5_core, true, false> matmul5;
            matmul5(matmul5_args);
        }

        // ========================================================================
        // Gather3: 112 active matmul5 cores -> gather core (12, 9)
        // Collects [1, 64] * 112 = [1, 7168]
        // ========================================================================
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

        // ========================================================================
        // CCL All-Reduce: Exchange [1, 7168] between devices
        // - CCL Sender (11, 9): Reads gather3 output from gather core, sends via fabric
        // - CCL Receiver (12, 9): Receives remote data, performs reduction

        // Note: skip_local_push=1 is set for CCLReceiverReaderCTArgs because
        // gather3 already pushed to CB7 (gather3_dst_cb). The receiver just
        // needs to wait for remote data and perform the reduction.
        // ========================================================================
        if constexpr (Core::is_ccl_sender_core) {
            DeviceZoneScopedN("CCL_SENDER_SEND");
#if defined(COMPILE_FOR_NCRISC)
            // We need to make sure brisc waits for the initial broadcast CCL to complete before connecting
            // to fabric. Alternative is to move brisc connection logic after the cb wait
            // The first mcast syncs the ncrisc, so we need to also make sure brisc waits for the first mcast
            volatile tt_l1_ptr uint32_t* sync_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"));
            // Make sure the value is different from the mcasted value
            // The wait below is for safety if this runs on the same core as another doing the same sync
            // If that is the case we don't actually need to do another sync
            noc_semaphore_wait(sync_sem, 0);
            noc_semaphore_set(sync_sem, 2);

            // Wait for gather3 to complete before reading from gather core
            constexpr uint32_t gather3_completion_semaphore_id =
                get_named_compile_time_arg_val("ccl_sender_gather3_completion_semaphore_id");
            volatile tt_l1_ptr uint32_t* gather3_completion_semaphore_addr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather3_completion_semaphore_id));
            noc_semaphore_wait(gather3_completion_semaphore_addr, 1);
            noc_semaphore_set(gather3_completion_semaphore_addr, 0);

            deepseek_b1_ops::AllReduceSender::Op<CCLSenderReaderCTArgs, DummyWriterCTArgs> ccl_sender_reader;
            ccl_sender_reader(ccl_sender_args);
#elif defined(COMPILE_FOR_BRISC)
            volatile tt_l1_ptr uint32_t* sync_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                get_named_compile_time_arg_val("mcast_data_receiver_semaphore_addr"));
            noc_semaphore_wait(sync_sem, 2);
            noc_semaphore_set(sync_sem, 0);
            deepseek_b1_ops::AllReduceSender::Op<DummyReaderCTArgs, CCLSenderWriterCTArgs> ccl_sender_writer;
            ccl_sender_writer(ccl_sender_args);
#endif
        }

        if constexpr (Core::is_ccl_receiver_core) {
            DeviceZoneScopedN("CCL_RECEIVER");
#if defined(COMPILE_FOR_NCRISC)
            // TODO: We're popping the RMSNorm input and then re-pushing it here as the residual
            // Should avoid this and not pop in RMSNorm
            deepseek_b1_ops::AllReduceReceiver::Op<CCLReceiverReaderCTArgs, DummyComputeCTArgs> ccl_receiver_reader;
            ccl_receiver_reader(ccl_receiver_args);
#elif defined(COMPILE_FOR_TRISC)
            deepseek_b1_ops::AllReduceReceiver::Op<DummyReaderCTArgs, CCLReceiverComputeCTArgs> ccl_receiver_compute;
            ccl_receiver_compute(ccl_receiver_args);
#endif
        }
    };

    for (uint32_t i = 0; i < num_iterations; i++) {
        unified_kernels::reconfig_cb_interfaces(cb_config);
        setup_all_sharded_buffers();
        mla_body();
    }

    // ====================================================================
    // Mcast: Teardown persistent mcast
    // ====================================================================
    {
        DeviceZoneScopedN("MCAST_TEARDOWN");
        mcast.teardown();
    }
}
