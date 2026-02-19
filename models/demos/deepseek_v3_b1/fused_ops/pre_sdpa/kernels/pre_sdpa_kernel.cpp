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

// Compile-time role flags for dead code elimination via if constexpr
// Defined at namespace scope (local classes cannot have static data members)
struct Core {
    static constexpr bool is_input_core = get_named_compile_time_arg_val("is_input_core") == 1;
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
};

void kernel_main() {
// ============================================================================
// NCRISC (Reader + Mcast Receiver) - ReaderConfigDescriptor compiles as NCRISC
// Named compile-time args: rmsnorm reader, mcast receiver, matmul reader, gather sender
// Runtime args: []
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type aliases (required for Op templates)
    // CCL Broadcast CTArgs type alias
    using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_packet_size_in_pages"),
        get_named_compile_time_arg_val("bcast_tensor0_page_size"),
        get_named_compile_time_arg_val("bcast_is_sender"),
        get_named_compile_time_arg_val("bcast_core_noc_x"),
        get_named_compile_time_arg_val("bcast_core_noc_y"),
        get_named_compile_time_arg_val("bcast_is_secondary_sender")>;

    // CCL Broadcast reader runtime args (only populated when not skip_ccl)
    deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{};
    if constexpr (!Core::skip_ccl) {
        bcast_args = deepseek_b1_ops::Broadcast::ReaderArgs{
            get_common_arg_val<uint32_t>(0),  // tensor_address0
            get_common_arg_val<uint32_t>(1),  // tile_id_start
            get_common_arg_val<uint32_t>(2),  // tile_id_end
        };
    }

    using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;

    // RMSNorm reader runtime args
    deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

    // Mcast receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
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
        get_named_compile_time_arg_val("gather_reduce_receiver_semaphore_id"),
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
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("matmul2_in0"),
        get_named_compile_time_arg_val("mcast2_dst_num_pages"),
    };

    // Matmul3 reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul3_args{};

    // Qrope CTArgs type alias (NCRISC uses ReaderCTArgs)
    using QRopeCTArgs = deepseek_b1_ops::Rope::
        ReaderCTArgs<get_named_compile_time_arg_val("qrope_Wt"), get_named_compile_time_arg_val("qrope_Ht")>;

    // Qrope reader args (NCRISC is no-op)
    deepseek_b1_ops::Rope::ReaderArgs qrope_args{};

    // NCRISC: Sender args for QNOPE/QROPE cores
    // Senders write to intermediate CB, then compute tilizes to output CB
    // 3-phase synchronization: nope_phase1, nope_phase2, rope semaphores
    constexpr uint32_t cqh_receiver_in_cb = get_named_compile_time_arg_val("cqh_receiver_in_cb");
    deepseek_b1_ops::CreateQHeads::SenderArgs create_q_heads_args{
        0,  // sender_grid_start_x (logical 0)
        0,  // sender_grid_start_y (logical 0)
        get_named_compile_time_arg_val("cqh_qnope_data_size_bytes"),
        get_named_compile_time_arg_val("cqh_qrope_head_size_bytes"),
        get_named_compile_time_arg_val("cqh_head_stride_bytes"),
        get_named_compile_time_arg_val("cqh_qnope_cols"),
        get_named_compile_time_arg_val("cqh_qnope_src_cb"),
        get_named_compile_time_arg_val("cqh_qrope_src_cb"),
        Core::is_qnope_core ? get_named_compile_time_arg_val("cqh_qnope_src_num_pages")
                            : get_named_compile_time_arg_val("cqh_qrope_src_num_pages"),
        get_named_compile_time_arg_val("cqh_nope_phase1_semaphore_id"),
        get_named_compile_time_arg_val("cqh_nope_phase2_semaphore_id"),
        get_named_compile_time_arg_val("cqh_rope_semaphore_id"),
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
        get_named_compile_time_arg_val("dkv_gather_receiver_semaphore_id"),
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

    using K_RopeCTArgs = deepseek_b1_ops::Rope::
        ReaderCTArgs<get_named_compile_time_arg_val("krope_Wt"), get_named_compile_time_arg_val("krope_Ht")>;
    constexpr uint32_t krope_input_cb = get_named_compile_time_arg_val("krope_in_cb");
    constexpr uint32_t krope_cos_cb = get_named_compile_time_arg_val("krope_cos_cb");
    constexpr uint32_t krope_sin_cb = get_named_compile_time_arg_val("krope_sin_cb");
    constexpr uint32_t krope_trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb");

    // Reader args: CB indices for sharded input signaling
    deepseek_b1_ops::Rope::ReaderArgs krope_args{};

    deepseek_b1_ops::KVCacheUpdate::ReaderArgs kv_cache_update_args{};
// ============================================================================
// BRISC (Writer + Mcast Sender) - WriterConfigDescriptor compiles as BRISC
// Named compile-time args: bcast writer + rmsnorm writer, mcast sender, matmul writer, gather receiver
// ============================================================================
#elif defined(COMPILE_FOR_BRISC)

    // CCL Broadcast CTArgs type alias
    using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
        get_named_compile_time_arg_val("bcast_cb0_id"),
        get_named_compile_time_arg_val("bcast_packet_size_in_pages"),
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
            get_common_arg_val<uint32_t>(2),   // tile_id_start
            get_common_arg_val<uint32_t>(3),   // tile_id_end
            get_common_arg_val<uint32_t>(4),   // wait_output_semaphore
            get_common_arg_val<uint32_t>(5),   // reset_global_semaphore
            get_common_arg_val<uint32_t>(6),   // out_ready_sem_noc0_x
            get_common_arg_val<uint32_t>(7),   // out_ready_sem_noc0_y
            get_common_arg_val<uint32_t>(8),   // out_ready_sem_wait_value
            get_common_arg_val<uint32_t>(9),   // barrier_sem
            get_common_arg_val<uint32_t>(10),  // barrier_sem_noc0_x
            get_common_arg_val<uint32_t>(11),  // barrier_sem_noc0_y
            get_common_arg_val<uint32_t>(12),  // ring_index
            get_common_arg_val<uint32_t>(13),  // secondary_sync_sem
            get_common_arg_val<uint32_t>(14),  // num_connections (computed from len(dst_nodes))
        };
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
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
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
        get_named_compile_time_arg_val("gather_reduce_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_reduce_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_reduce_half0_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_half1_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_dst_num_tiles"),
    };

    // BRISC: Receiver args for SDPA input cores
    deepseek_b1_ops::CreateQHeads::ReceiverArgs create_q_heads_args{
        get_named_compile_time_arg_val("cqh_nope_phase1_semaphore_id"),
        get_named_compile_time_arg_val("cqh_nope_phase2_semaphore_id"),
        get_named_compile_time_arg_val("cqh_rope_semaphore_id"),
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
        get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
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
        get_named_compile_time_arg_val("dkv_gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("dkv_gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("dkv_gather_dst_cb"),
        get_named_compile_time_arg_val("dkv_gather_dst_num_pages"),
    };

    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
    deepseek_b1_ops::RMSNorm::WriterArgs kv_rmsnorm_args{};

    using K_RopeCTArgs = deepseek_b1_ops::Rope::WriterCTArgs;

    // Writer args (empty - no-op)
    deepseek_b1_ops::Rope::WriterArgs krope_args{};

    deepseek_b1_ops::KVCacheUpdate::WriterArgs kv_cache_update_args{
        .kv_cache_buffer_base_addr = get_common_arg_val<uint32_t>(15),
        .position_id = get_common_arg_val<uint32_t>(16),
        .kv_cache_input_cb = get_named_compile_time_arg_val("kv_cache_input_cb"),
        .kv_cache_intermed_cb = get_named_compile_time_arg_val("kv_cache_intermed_cb"),
        .kv_cache_output_cb = get_named_compile_time_arg_val("kv_cache_output_cb"),
        .kv_rmsnorm_output_cb = get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
        .krope_output_cb = get_named_compile_time_arg_val("krope_output_cb"),
        .grid_start_y = get_named_compile_time_arg_val("kv_cache_grid_start_y"),
    };
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
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;
    using RMSNorm2CTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("rmsnorm2_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
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
    };

    // Gather reduce compute args
    deepseek_b1_ops::GatherReduce::ComputeArgs gather_reduce_args{
        get_named_compile_time_arg_val("gather_reduce_half0_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_half1_dst_cb"),
        get_named_compile_time_arg_val("gather_reduce_dst_num_tiles"),
    };

    // RMSNorm2 compute args (separate CBs with exact sizes for testing)
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm2_args{
        get_named_compile_time_arg_val("rmsnorm2_input_cb"),   // separate input CB (3 tiles of 16x32)
        get_named_compile_time_arg_val("rmsnorm2_gamma_cb"),   // new gamma for 1536 elements
        get_named_compile_time_arg_val("rmsnorm2_output_cb"),  // separate output CB (3 tiles of 16x32)
        get_common_arg_val<uint32_t>(0),                       // epsilon (same as rmsnorm1)
        get_common_arg_val<float>(2),                          // scalar (1/sqrt(1536))
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
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs dkv_gather_args{};

    // CTArgs type aliases (required for Op templates)
    using KV_RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
        get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
        get_named_compile_time_arg_val("kv_rmsnorm_num_tiles"),
        get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1>;

    // RMSNorm compute runtime args
    deepseek_b1_ops::RMSNorm::ComputeArgs kv_rmsnorm_args{
        get_named_compile_time_arg_val("kv_rmsnorm_input_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("kv_rmsnorm_output_cb"),
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

    // Full init, CBs don't matter
    compute_kernel_hw_startup(0, 0, 0);
#endif

#if defined(COMPILE_FOR_NCRISC)
    // Setup sharded persistent buffers
    if constexpr (Core::is_input_core && !Core::skip_ccl) {
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
    if constexpr (Core::is_matmul_core) {
        // Matmul weights
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_per_core");
        constexpr uint32_t matmul_out_w_per_core = get_named_compile_time_arg_val("matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w_per_core);
    }
    if constexpr (Core::is_matmul2_core) {
        // Matmul2 CB indices and parameters from named compile-time args
        constexpr uint32_t matmul2_in1 = get_named_compile_time_arg_val("matmul2_in1");
        constexpr uint32_t matmul2_k_num_tiles = get_named_compile_time_arg_val("matmul2_k_num_tiles");
        constexpr uint32_t matmul2_out_w_per_core = get_named_compile_time_arg_val("matmul2_out_w_per_core");

        // Matmul2 weights (on all cores in main grid, 4 tiles per core)
        unified_kernels::setup_sharded_buffer(matmul2_in1, matmul2_k_num_tiles * matmul2_out_w_per_core);
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
        // Qrope CB indices and parameters from named compile-time args
        constexpr uint32_t qrope_cos_cb = get_named_compile_time_arg_val("qrope_cos_cb");
        constexpr uint32_t qrope_sin_cb = get_named_compile_time_arg_val("qrope_sin_cb");
        constexpr uint32_t qrope_trans_mat_cb = get_named_compile_time_arg_val("qrope_trans_mat_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("qrope_Wt");

        // NOTE: Do NOT setup qrope input CB (matmul2_output_cb) as sharded buffer!
        // The input to RoPE comes from matmul2 compute output, NOT from a sharded tensor.
        // Calling setup_sharded_buffer on it would fill the CB and block matmul2's cb_reserve_back.
        // Only setup the actual sharded tensor CBs (cos, sin, trans_mat).
        unified_kernels::setup_sharded_buffer(qrope_cos_cb, Wt);
        unified_kernels::setup_sharded_buffer(qrope_sin_cb, Wt);
        unified_kernels::setup_sharded_buffer(qrope_trans_mat_cb, 1);  // trans_mat is 1 tile (32x32)
    }

    if constexpr (Core::is_dkv_matmul_core) {
        // Matmul weights (in1)
        constexpr uint32_t dkv_matmul_in1 = get_named_compile_time_arg_val("dkv_matmul_in1");
        constexpr uint32_t dkv_matmul_out_w_per_core = get_named_compile_time_arg_val("dkv_matmul_out_w_per_core");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in1, dkv_matmul_k_num_tiles * dkv_matmul_out_w_per_core);
    }

    if constexpr (Core::is_kv_rmsnorm_core) {
        // RMSNorm gamma (sharded weights)
        constexpr uint32_t kv_rmsnorm_gamma_cb = get_named_compile_time_arg_val("kv_rmsnorm_gamma_cb");
        constexpr uint32_t kv_rmsnorm_num_tiles = get_named_compile_time_arg_val("kv_rmsnorm_num_tiles");
        unified_kernels::setup_sharded_buffer(kv_rmsnorm_gamma_cb, kv_rmsnorm_num_tiles);
    }

    if constexpr (Core::is_krope_core) {
        constexpr uint32_t krope_cos_cb = get_named_compile_time_arg_val("krope_cos_cb");
        constexpr uint32_t krope_sin_cb = get_named_compile_time_arg_val("krope_sin_cb");
        constexpr uint32_t krope_trans_mat_cb = get_named_compile_time_arg_val("krope_trans_mat_cb");
        constexpr uint32_t krope_Wt = get_named_compile_time_arg_val("krope_Wt");
        unified_kernels::setup_sharded_buffer(krope_cos_cb, krope_Wt);
        unified_kernels::setup_sharded_buffer(krope_sin_cb, krope_Wt);
        unified_kernels::setup_sharded_buffer(krope_trans_mat_cb, 1);
    }
#endif

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
    if constexpr (Core::is_input_core && Core::skip_ccl) {
        // Single-device mode: NCRISC sets up ALL sharded buffers (input + gamma + gamma2)
        // This matches the reference kernel behavior
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        constexpr uint32_t rmsnorm2_gamma_cb = get_named_compile_time_arg_val("rmsnorm2_gamma_cb");
        constexpr uint32_t rmsnorm2_num_tiles = get_named_compile_time_arg_val("rmsnorm2_num_tiles");

        unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
        unified_kernels::setup_sharded_buffer(rmsnorm2_gamma_cb, rmsnorm2_num_tiles);
    }
#endif

#if defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_input_core && !Core::skip_ccl) {
        // Multi-device mode only: BRISC sets up intermediate (broadcast output) buffer
        // Gamma CBs are already set up by NCRISC via setup_sharded_buffer
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

        cb_reserve_back(rmsnorm_input_cb, rmsnorm_num_tiles);
        cb_push_back(rmsnorm_input_cb, rmsnorm_num_tiles);
    }
#endif

    // ========================================================================
    // Input core: RMSNorm + Mcast send
    // ========================================================================
    {
        DeviceZoneScopedN("RMSNORM");
        // pop_input = true (input is consumed after RMSNorm)
        deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core, true> rmsnorm;
        rmsnorm(rmsnorm_args);
    }

    // pop_src = true (rmsnorm output is consumed after mcast)
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_input_core,
        Core::is_matmul2_core,
        Core::is_matmul_core || Core::is_dkv_matmul_core,
        true>
        mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        // Mcast: NCRISC sends from input core, BRISC receives on matmul cores, TRISC no-op
        // pop_src = true (input is consumed after mcast)
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
        deepseek_b1_ops::Mcast::Op<McastCTArgs, Core::is_input_core, Core::is_matmul2_core, Core::is_matmul2_core, true>
            mcast2;
        mcast2(mcast2_args);
    }
    mcast.teardown();

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
            deepseek_b1_ops::CreateQHeads::Op<is_create_q_heads_sender, Core::is_sdpa_input_core, false, true>
                create_q_heads;
            create_q_heads(create_q_heads_args);
        }
    }
    {
        // ========================================================================
        // KV Cache Branch - Matmul
        // DKV Matmul: 9x2 grid, each core handles 1 head of 32 dim
        // ========================================================================
        {
            DeviceZoneScopedN("DKV_MATMUL");
            // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)o
            deepseek_b1_ops::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, false, false> dkv_matmul;
            dkv_matmul(dkv_matmul_args);
        }

        // ========================================================================
        // KV Cache Branch: Gather: dkv matmul cores (senders) -> rmsnorm core (receiver)
        // NCRISC sends from knope grid of dkv matmul cores, BRISC receives on rmsnorm grid, TRISC no-op
        // ========================================================================
        {
            DeviceZoneScopedN("DKV_GATHER");
            deepseek_b1_ops::Gather::Op<Core::is_knope_core, Core::is_kv_rmsnorm_core, true> dkv_gather;
            dkv_gather(dkv_gather_args);
        }

        // ========================================================================
        // RMSNorm: Apply RMSNorm to the gathered data
        {
            DeviceZoneScopedN("KV_RMSNORM");
            deepseek_b1_ops::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
            kv_rmsnorm(kv_rmsnorm_args);
        }
        // ========================================================================
        // KV Cache Branch: RoPE
        // ========================================================================
        {
            DeviceZoneScopedN("K_ROPE");
            deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> krope;
            krope(krope_args);
        }
        // ========================================================================
        // KV Cache Update: Write results to DRAM interleaved tensor
        // BRISC handles writing from output CBs to DRAM
        // ========================================================================
        {
            DeviceZoneScopedN("KV_CACHE_UPDATE");
            deepseek_b1_ops::KVCacheUpdate::Op<Core::is_kv_rmsnorm_core, Core::is_krope_core> kv_cache_update;
            kv_cache_update(kv_cache_update_args);
        }
    }
}
