// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Pre-SDPA unified kernel
// Single kernel file, compiles correctly for all RISC cores
// Each RISC has its own CTArgs struct with different compile-time arg layout
//
// Implements: CCL Broadcast + RMSNorm + Mcast + Matmul + Gather + RMSNorm2 + Mcast2 + Matmul2 + Matmul3 + RoPE +
// GatherHeads
// - NCRISC: CCL Broadcast Reader + RMSNorm reader + Mcast receiver (on matmul cores), Matmul reader + Gather sender (on
// matmul cores),
//           RMSNorm2 reader + Mcast2 receiver (on matmul2 cores), Matmul2 reader (on matmul2 cores),
//           Matmul3 reader (on qnope cores), RoPE reader (on qrope cores), GatherHeads sender (on qnope/qrope cores)
// - BRISC: CCL Broadcast Writer + RMSNorm writer + Mcast sender (on input core), Matmul writer (on matmul cores),
// Gather receiver (on
//          input core), Mcast2 sender (on input core), Matmul2 writer (on matmul2 cores),
//          GatherHeads receiver (on sdpa input cores)
// - TRISC: RMSNorm compute (on input core), Matmul compute (on matmul cores), RMSNorm2 compute (on input core),
//          Matmul2 compute (on matmul2 cores), Matmul3 compute (on qnope cores), RoPE compute (on qrope cores)
//
// Matmul2 output uses interleaved Qnope/Qrope layout (with shuffled weights):
// - Grid: 12 cols × 8 rows = 96 cores (P150)
// - Qnope cores (cols 0-7): 64 cores, 1 head × 128 elements per core
// - Qrope cores (cols 8-11): 32 cores, 2 heads × 64 elements per core
// - Each row: [Qnope heads 0-7 (1024)] [Qrope heads 0-7 (512)] = 1536 elements
// - Total: 8 rows × 1536 = 12288 elements

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
// Wormhole-compatible replacements for Blackhole-only headers:
// - rmsnorm.hpp (uses experimental/mul_reduce_scalar, add_rsqrt, rmsnorm)
// - mcast.hpp (uses NOC_PCIE_MASK, NOC_BRCST_EXCLUDE)
// - matmul.hpp (uses custom_mm.h with Blackhole-only LLK)
#include "wormhole_rmsnorm.hpp"
#include "wormhole_mcast.hpp"
#include "wormhole_matmul.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/gather.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/gather_heads.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/rope.hpp"
#include "../../../../deepseek_v3_b1/unified_kernels/broadcast.hpp"

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
    // SDPA Input core: receives interleaved QNOPE/QROPE gather heads (4×2 grid = 8 cores)
    static constexpr bool is_sdpa_input_core = get_named_compile_time_arg_val("is_sdpa_input_core") == 1;

    // DKV Matmul core: 9x2 grid, each core handles 1 head of 32 dim
    static constexpr bool is_dkv_matmul_core = get_named_compile_time_arg_val("is_dkv_matmul_core") == 1;
    static constexpr bool is_kv_rmsnorm_core = get_named_compile_time_arg_val("is_kv_rmsnorm_core") == 1;
    static constexpr bool is_knope_core = get_named_compile_time_arg_val("is_knope_core") == 1;
    static constexpr bool is_krope_core = get_named_compile_time_arg_val("is_krope_core") == 1;
    static constexpr bool skip_ccl = get_named_compile_time_arg_val("skip_ccl") == 1;
    // Debug: early exit after stage N (99=run all, 0=setup only, 1=rmsnorm, 2=mcast, etc.)
    static constexpr uint32_t debug_stage = get_named_compile_time_arg_val("debug_stage");
};

// Wormhole RMSNorm scratch CB filling (NCRISC only)
// Fills a 1x32 tile in a CB with a uniform bf16 value in all 32 positions.
#if defined(COMPILE_FOR_NCRISC)
FORCE_INLINE void fill_rmsnorm_scratch_tile(uint32_t cb_id, uint32_t bf16_val_packed, uint32_t page_size_words) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    // Zero the tile using local write (NOT MEM_ZEROS_SIZE which is 512 bytes
    // and overflows the ~64-128 byte scratch tile slot, corrupting adjacent L1)
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(write_addr);
    for (uint32_t i = 0; i < page_size_words; i++) {
        dst[i] = 0;
    }
    // Fill first 32 bf16 positions with the value
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
    uint16_t val = static_cast<uint16_t>(bf16_val_packed);
    for (uint32_t j = 0; j < 32; ++j) {
        ptr[j] = val;
    }
    cb_push_back(cb_id, 1);
}
#endif

void kernel_main() {
    // Suppress unused-local-typedefs: CTArgs type aliases are defined for all cores
    // but only used inside if-constexpr blocks gated on core role flags.
    // Non-matching cores legitimately don't use them.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

    // DEBUG: at stage 0, do absolutely nothing — test if generic_op can launch empty kernel
    if constexpr (Core::debug_stage == 0) {
        return;
    }
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
        get_named_compile_time_arg_val("bcast_is_secondary_sender"),
        get_named_compile_time_arg_val("bcast_is_active_broadcaster")>;

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
    using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    using Matmul3CTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;

    // Matmul reader args (NCRISC is no-op)
    deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    // Gather sender args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::SenderArgs gather_args{
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
        get_write_ptr(get_named_compile_time_arg_val(
            "rmsnorm2_input_cb")),  // receiver_data_addr from CB write ptr (single-buffered)
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
    using QRopeCTArgs =
        deepseek_b1_ops::Rope::ReaderCTArgs<get_named_compile_time_arg_val("Wt"), get_named_compile_time_arg_val("Ht")>;

    // Qrope reader args (NCRISC is no-op)
    deepseek_b1_ops::Rope::ReaderArgs qrope_args{};

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
    deepseek_b1_ops::Rope::ReaderArgs krope_args{
        .in_cb = krope_input_cb,
        .cos_cb = krope_cos_cb,
        .sin_cb = krope_sin_cb,
        .trans_mat_cb = krope_trans_mat_cb,
    };

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
        get_named_compile_time_arg_val("bcast_has_reverse_secondary_connection"),
        get_named_compile_time_arg_val("bcast_start_distance_in_hops_forward"),
        get_named_compile_time_arg_val("bcast_range_hops_forward"),
        get_named_compile_time_arg_val("bcast_start_distance_in_hops_backward"),
        get_named_compile_time_arg_val("bcast_range_hops_backward"),
        get_named_compile_time_arg_val("bcast_using_persistent_buffers")>;

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
    using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul2CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    using Matmul3CTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;

    // Matmul writer args (BRISC is no-op)
    deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    // Gather receiver args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
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

    // BRISC: Sender args for QNOPE/QROPE cores
    // Senders write directly to output CB (allocated on sender+receiver cores)
    constexpr uint32_t receive_cb = get_named_compile_time_arg_val("receive_cb");
    deepseek_b1_ops::GatherHeads::SenderArgs gather_heads_args{
        0,  // sender_grid_start_x (logical 0)
        0,  // sender_grid_start_y (logical 0)
        get_named_compile_time_arg_val("qnope_data_size_bytes"),
        get_named_compile_time_arg_val("qrope_data_size_bytes"),
        get_named_compile_time_arg_val("head_stride_bytes"),
        get_named_compile_time_arg_val("qnope_grid_cols"),
        get_named_compile_time_arg_val("qnope_src_cb"),
        get_named_compile_time_arg_val("qrope_src_cb"),
        Core::is_qnope_core ? get_named_compile_time_arg_val("qnope_src_num_pages")
                            : get_named_compile_time_arg_val("qrope_src_num_pages"),
        get_named_compile_time_arg_val("receiver_semaphore_id"),
        {
            get_named_compile_time_arg_val("target_noc_coords_row0"),
            get_named_compile_time_arg_val("target_noc_coords_row1"),
            get_named_compile_time_arg_val("target_noc_coords_row2"),
            get_named_compile_time_arg_val("target_noc_coords_row3"),
            get_named_compile_time_arg_val("target_noc_coords_row4"),
            get_named_compile_time_arg_val("target_noc_coords_row5"),
            get_named_compile_time_arg_val("target_noc_coords_row6"),
            get_named_compile_time_arg_val("target_noc_coords_row7"),
        },
        get_write_ptr(receive_cb),  // Write directly to output CB
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
    constexpr uint32_t rmsnorm_scratch_cb = get_named_compile_time_arg_val("rmsnorm_scratch_cb");
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
        get_named_compile_time_arg_val("rmsnorm_input_cb"),
        get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
        get_named_compile_time_arg_val("rmsnorm_output_cb"),
        get_common_arg_val<uint32_t>(0),  // epsilon
        get_common_arg_val<float>(1),     // scalar (1/sqrt(hidden_size))
        rmsnorm_scratch_cb,               // Wormhole: intermediate CB for scaler/eps/rsqrt
    };

    // Mcast compute args (no-op for TRISC)
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // Matmul CTArgs type alias (out_w is compile-time for TRISC)
    using MatmulCTArgs =
        deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val("matmul_out_w_per_core")>;

    // Matmul compute args (from compile-time args, passed to op as runtime args)
    deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("matmul_in0"),
        get_named_compile_time_arg_val("matmul_in1"),
        get_named_compile_time_arg_val("matmul_out"),
        get_named_compile_time_arg_val("matmul_k_num_tiles"),
    };

    // Gather compute args (no-op for TRISC)
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    // RMSNorm2 compute args (separate CBs with exact sizes for testing)
    deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm2_args{
        get_named_compile_time_arg_val("rmsnorm2_input_cb"),   // separate input CB (3 tiles of 16x32)
        get_named_compile_time_arg_val("rmsnorm2_gamma_cb"),   // new gamma for 1536 elements
        get_named_compile_time_arg_val("rmsnorm2_output_cb"),  // separate output CB (3 tiles of 16x32)
        get_common_arg_val<uint32_t>(0),                       // epsilon (same as rmsnorm1)
        get_common_arg_val<float>(2),                          // scalar (1/sqrt(1536))
        rmsnorm_scratch_cb,                                    // Wormhole: shared intermediate CB
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
        ComputeCTArgs<get_named_compile_time_arg_val("Wt"), get_named_compile_time_arg_val("Ht")>;

    // Qrope compute args (from compile-time args)
    deepseek_b1_ops::Rope::ComputeArgs qrope_args{
        get_named_compile_time_arg_val("in_cb"),  // Input from matmul2 output
        get_named_compile_time_arg_val("cos_cb"),
        get_named_compile_time_arg_val("sin_cb"),
        get_named_compile_time_arg_val("trans_mat_cb"),
        get_named_compile_time_arg_val("rotated_in_interm_cb"),
        get_named_compile_time_arg_val("cos_interm_cb"),
        get_named_compile_time_arg_val("sin_interm_cb"),
        get_named_compile_time_arg_val("out_cb"),
    };

    // Gather heads compute args (no-op for TRISC)
    deepseek_b1_ops::GatherHeads::ComputeArgs gather_heads_args{};

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
        rmsnorm_scratch_cb,               // Wormhole: shared intermediate CB
    };

    using K_RopeCTArgs = deepseek_b1_ops::Rope::
        ComputeCTArgs<get_named_compile_time_arg_val("krope_Wt"), get_named_compile_time_arg_val("krope_Ht")>;

    // CB indices (passed as runtime args to ComputeArgs)
    constexpr uint32_t krope_input_cb = get_named_compile_time_arg_val("krope_in_cb");
    constexpr uint32_t krope_cos_cb = get_named_compile_time_arg_val("krope_cos_cb");
    constexpr uint32_t krope_sin_cb = get_named_compile_time_arg_val("krope_sin_cb");
    constexpr uint32_t trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
    constexpr uint32_t krope_rotated_in_interm_cb = get_named_compile_time_arg_val("krope_rotated_in_interm_cb");
    constexpr uint32_t krope_cos_interm_cb = get_named_compile_time_arg_val("krope_cos_interm_cb");
    constexpr uint32_t krope_sin_interm_cb = get_named_compile_time_arg_val("krope_sin_interm_cb");
    constexpr uint32_t krope_output_cb = get_named_compile_time_arg_val("krope_output_cb");

    // Compute args: all CB indices
    deepseek_b1_ops::Rope::ComputeArgs krope_args{
        .in_cb = krope_input_cb,
        .cos_cb = krope_cos_cb,
        .sin_cb = krope_sin_cb,
        .trans_mat_cb = trans_mat_cb,
        .rotated_in_interm_cb = krope_rotated_in_interm_cb,
        .cos_interm_cb = krope_cos_interm_cb,
        .sin_interm_cb = krope_sin_interm_cb,
        .out_cb = krope_output_cb,
    };
#endif

    // DEBUG: inner_debug_phase controls sub-stage isolation within stage 1
    // 0 = return immediately after arg construction (test: is arg construction OK?)
    // 1 = run NCRISC setup + CCL but skip RMSNorm
    // 2 = run NCRISC setup + CCL + scratch fill but skip TRISC compute
    // 99 = run everything (normal stage 1 behavior)
    constexpr uint32_t inner_debug_phase = 99;   // Run all sub-stages
    constexpr uint32_t inner_debug_phase_2 = 1;  // Stage 2 sub-phase: 0=before init, 1=after init, 99=full
    // Phase 0: return before any execution (test: is arg construction OK?)
    if constexpr (Core::debug_stage == 1 && inner_debug_phase == 0) {
        return;  // EARLY EXIT: skip ALL execution (setup, CCL, RMSNorm, etc.)
    }

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
        constexpr uint32_t matmul_in1 = get_named_compile_time_arg_val("matmul_in1");
        constexpr uint32_t matmul_k_num_tiles = get_named_compile_time_arg_val("matmul_k_num_tiles");
        constexpr uint32_t matmul_out_w_per_core = get_named_compile_time_arg_val("matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul_in1, matmul_k_num_tiles * matmul_out_w_per_core);
    }
    if constexpr (Core::is_matmul2_core) {
        constexpr uint32_t matmul2_in1 = get_named_compile_time_arg_val("matmul2_in1");
        constexpr uint32_t matmul2_k_num_tiles = get_named_compile_time_arg_val("matmul2_k_num_tiles");
        constexpr uint32_t matmul2_out_w_per_core = get_named_compile_time_arg_val("matmul2_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul2_in1, matmul2_k_num_tiles * matmul2_out_w_per_core);
    }
#if constexpr (Core::debug_stage >= 7)  // Stages 3-8 NCRISC setup: commented out for debug_stage<=2 testing (NOC reads
                                        // hang on invalid DRAM addrs)
    if constexpr (Core::is_qnope_core) {
        constexpr uint32_t matmul3_in1 = get_named_compile_time_arg_val("matmul3_in1");
        constexpr uint32_t matmul3_k_num_tiles = get_named_compile_time_arg_val("matmul3_k_num_tiles");
        constexpr uint32_t matmul3_out_w_per_core = get_named_compile_time_arg_val("matmul3_out_w_per_core");
        unified_kernels::setup_sharded_buffer(matmul3_in1, matmul3_k_num_tiles * matmul3_out_w_per_core);
    }
    if constexpr (Core::is_qrope_core) {
        constexpr uint32_t qrope_cos_cb = get_named_compile_time_arg_val("cos_cb");
        constexpr uint32_t qrope_sin_cb = get_named_compile_time_arg_val("sin_cb");
        constexpr uint32_t qrope_trans_mat_cb = get_named_compile_time_arg_val("trans_mat_cb");
        constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
        unified_kernels::setup_sharded_buffer(qrope_cos_cb, Wt);
        unified_kernels::setup_sharded_buffer(qrope_sin_cb, Wt);
        unified_kernels::setup_sharded_buffer(qrope_trans_mat_cb, 1);
    }
#endif  // Stages 3-8 qnope/qrope setup

    // NCRISC: Receiver args for SDPA input cores (struct init only, no NOC reads — safe to always run)
    deepseek_b1_ops::GatherHeads::ReceiverArgs gather_heads_args{
        get_named_compile_time_arg_val("receiver_semaphore_id"),
        get_named_compile_time_arg_val("num_senders"),
        get_named_compile_time_arg_val("receive_cb"),  // Output CB
        get_named_compile_time_arg_val("dst_num_pages"),
    };

#if constexpr (Core::debug_stage >= 8)  // Stages 3-8 NCRISC setup continued (NOC reads hang on invalid DRAM addrs)
    if constexpr (Core::is_dkv_matmul_core) {
        constexpr uint32_t dkv_matmul_in1 = get_named_compile_time_arg_val("dkv_matmul_in1");
        constexpr uint32_t dkv_matmul_out_w_per_core = get_named_compile_time_arg_val("dkv_matmul_out_w_per_core");
        constexpr uint32_t dkv_matmul_k_num_tiles = get_named_compile_time_arg_val("dkv_matmul_k_num_tiles");
        unified_kernels::setup_sharded_buffer(dkv_matmul_in1, dkv_matmul_k_num_tiles * dkv_matmul_out_w_per_core);
    }
    if constexpr (Core::is_kv_rmsnorm_core) {
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
#endif  // Stages 3-8 dkv/rmsnorm/krope setup
#endif

    // DEBUG: phase 1 exit — after NCRISC weight setup, before CCL+input setup+RMSNorm
    if constexpr (Core::debug_stage == 1 && inner_debug_phase == 1) {
        return;
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
        // Gamma weights are always local (not broadcast via CCL) — setup regardless of CCL mode
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");
        constexpr uint32_t rmsnorm2_gamma_cb = get_named_compile_time_arg_val("rmsnorm2_gamma_cb");
        constexpr uint32_t rmsnorm2_num_tiles = get_named_compile_time_arg_val("rmsnorm2_num_tiles");

        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_num_tiles);
        unified_kernels::setup_sharded_buffer(rmsnorm2_gamma_cb, rmsnorm2_num_tiles);
    }
    if constexpr (Core::is_input_core && Core::skip_ccl) {
        // Single-device only: input comes from local sharded buffer (not CCL broadcast)
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

        unified_kernels::setup_sharded_buffer(rmsnorm_input_cb, rmsnorm_num_tiles);
    }
#endif

#if defined(COMPILE_FOR_BRISC)
    if constexpr (Core::is_input_core && !Core::skip_ccl) {
        // Multi-device mode only: BRISC sets up intermediate (broadcast output) buffer
        constexpr uint32_t rmsnorm_input_cb = get_named_compile_time_arg_val("rmsnorm_input_cb");
        constexpr uint32_t rmsnorm_num_tiles = get_named_compile_time_arg_val("rmsnorm_num_tiles");

        cb_reserve_back(rmsnorm_input_cb, rmsnorm_num_tiles);
        cb_push_back(rmsnorm_input_cb, rmsnorm_num_tiles);
    }
#endif

    // DEBUG: phase 2 exit — after CCL+input setup, before RMSNorm scratch fill and compute
    if constexpr (Core::debug_stage == 1 && inner_debug_phase == 2) {
        return;
    }

    // ========================================================================
    // Stage 1: Input core RMSNorm + Mcast send
    // ========================================================================
    if constexpr (Core::debug_stage >= 1) {
#if defined(COMPILE_FOR_NCRISC)
        // Wormhole: fill scratch CB with scaler and eps for RMSNorm1
        if constexpr (Core::is_input_core) {
            constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("rmsnorm_scratch_cb");
            constexpr uint32_t scaler_bf16 = get_named_compile_time_arg_val("rmsnorm_scaler_bf16");
            constexpr uint32_t eps_bf16 = get_named_compile_time_arg_val("rmsnorm_eps_bf16");
            constexpr uint32_t scratch_page_words = get_named_compile_time_arg_val("rmsnorm_scratch_page_words");
            fill_rmsnorm_scratch_tile(scratch_cb, scaler_bf16, scratch_page_words);
            fill_rmsnorm_scratch_tile(scratch_cb, eps_bf16, scratch_page_words);
        }
#endif
        {
            DeviceZoneScopedN("RMSNORM");
            deepseek_b1_ops::RMSNorm::Op<RMSNormCTArgs, Core::is_input_core, true> rmsnorm;
            rmsnorm(rmsnorm_args);
        }
    }  // stage 1

    // pop_src = true (rmsnorm output is consumed after mcast)
    deepseek_b1_ops::Mcast::Op<
        McastCTArgs,
        Core::is_input_core,
        Core::is_matmul2_core,
        Core::is_matmul_core || Core::is_dkv_matmul_core,
        true>
        mcast;

    if constexpr (Core::debug_stage >= 2) {
        // Phase 0: return before mcast init (test: does pre-mcast code complete?)
        if constexpr (inner_debug_phase_2 == 0) {
            // Skip mcast entirely - just test that we get here OK
        } else {
            mcast.init(mcast_args);
            // Phase 1: return after init, before operator
            if constexpr (inner_debug_phase_2 >= 99) {
                DeviceZoneScopedN("MCAST");
                // Mcast: NCRISC sends from input core, BRISC receives on matmul cores, TRISC no-op
                // pop_src = true (input is consumed after mcast)
                mcast(mcast_args);
            }
        }
    }  // stage 2

    // ========================================================================
    // Stage 3: Matmul operation
    // ========================================================================
    if constexpr (Core::debug_stage >= 3) {
        {
            DeviceZoneScopedN("MATMUL");
            // pop_in0 = true (consumed), pop_in1 = false (weights are persistent)
            deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_matmul_core, false, false> matmul;
            matmul(matmul_args);
        }
    }  // stage 3

    // ========================================================================
    // Stage 4: Gather: matmul cores (senders) -> input core (receiver)
    // ========================================================================
    if constexpr (Core::debug_stage >= 4) {
        {
            DeviceZoneScopedN("GATHER");
            // pop_src = true (matmul output is consumed after gather)
            deepseek_b1_ops::Gather::Op<Core::is_matmul_core, Core::is_input_core, true> gather;
            gather(gather_args);
        }
    }  // stage 4

    // ========================================================================
    // Stage 5: RMSNorm2
    // ========================================================================
    if constexpr (Core::debug_stage >= 5) {
#if defined(COMPILE_FOR_NCRISC)
        // Wormhole: fill scratch CB with scaler and eps for RMSNorm2
        if constexpr (Core::is_input_core) {
            constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("rmsnorm_scratch_cb");
            constexpr uint32_t scaler2_bf16 = get_named_compile_time_arg_val("rmsnorm2_scaler_bf16");
            constexpr uint32_t eps_bf16 = get_named_compile_time_arg_val("rmsnorm_eps_bf16");
            constexpr uint32_t scratch_page_words = get_named_compile_time_arg_val("rmsnorm_scratch_page_words");
            fill_rmsnorm_scratch_tile(scratch_cb, scaler2_bf16, scratch_page_words);
            fill_rmsnorm_scratch_tile(scratch_cb, eps_bf16, scratch_page_words);
        }
#endif
        {
            DeviceZoneScopedN("RMSNORM2");
            deepseek_b1_ops::RMSNorm::Op<RMSNorm2CTArgs, Core::is_input_core, true> rmsnorm2;
            rmsnorm2(rmsnorm2_args);
        }
    }  // stage 5

    // ========================================================================
    // Stage 6: Mcast2 + Matmul2
    // ========================================================================
    if constexpr (Core::debug_stage >= 6) {
        {
            DeviceZoneScopedN("MCAST2");
            deepseek_b1_ops::Mcast::
                Op<McastCTArgs, Core::is_input_core, Core::is_matmul2_core, Core::is_matmul2_core, true>
                    mcast2;
            mcast2(mcast2_args);
        }
    }  // stage 6

    if constexpr (Core::debug_stage >= 2 && inner_debug_phase_2 >= 1) {
        mcast.teardown();
    }

    if constexpr (Core::debug_stage >= 7) {
        {
            DeviceZoneScopedN("MATMUL2");
            deepseek_b1_ops::Matmul::Op<Matmul2CTArgs, Core::is_matmul2_core, true, false> matmul2;
            matmul2(matmul2_args);
        }

        {
            DeviceZoneScopedN("Q_HEADS") static_assert(
                !(Core::is_qnope_core && Core::is_qrope_core), "Core cannot be both QNOPE and QROPE");

            {
                DeviceZoneScopedN("QNOPE/MATMUL3");
                deepseek_b1_ops::Matmul::Op<Matmul3CTArgs, Core::is_qnope_core, true, false> matmul3;
                matmul3(matmul3_args);
            }

            {
                DeviceZoneScopedN("QROPE");
                deepseek_b1_ops::Rope::Op<QRopeCTArgs, Core::is_qrope_core> rope;
                rope(qrope_args);
            }

            {
                DeviceZoneScopedN("GATHER_HEADS");
                constexpr bool is_gather_heads_sender = Core::is_qnope_core || Core::is_qrope_core;
                deepseek_b1_ops::GatherHeads::Op<is_gather_heads_sender, Core::is_sdpa_input_core, false, true, true>
                    gather_heads;
                gather_heads(gather_heads_args);
            }
        }
    }  // stage 7

    // ========================================================================
    // Stage 8: KV Cache Branch (DKV matmul + gather + rmsnorm + rope)
    // ========================================================================
    if constexpr (Core::debug_stage >= 8) {
        {
            {
                DeviceZoneScopedN("DKV_MATMUL");
                deepseek_b1_ops::Matmul::Op<DKV_MatmulCTArgs, Core::is_dkv_matmul_core, false, false> dkv_matmul;
                dkv_matmul(dkv_matmul_args);
            }

            {
                DeviceZoneScopedN("DKV_GATHER");
                deepseek_b1_ops::Gather::Op<Core::is_knope_core, Core::is_kv_rmsnorm_core, true> dkv_gather;
                dkv_gather(dkv_gather_args);
            }

#if defined(COMPILE_FOR_NCRISC)
            if constexpr (Core::is_kv_rmsnorm_core) {
                constexpr uint32_t scratch_cb = get_named_compile_time_arg_val("rmsnorm_scratch_cb");
                constexpr uint32_t kv_scaler_bf16 = get_named_compile_time_arg_val("kv_rmsnorm_scaler_bf16");
                constexpr uint32_t eps_bf16 = get_named_compile_time_arg_val("rmsnorm_eps_bf16");
                constexpr uint32_t scratch_page_words = get_named_compile_time_arg_val("rmsnorm_scratch_page_words");
                fill_rmsnorm_scratch_tile(scratch_cb, kv_scaler_bf16, scratch_page_words);
                fill_rmsnorm_scratch_tile(scratch_cb, eps_bf16, scratch_page_words);
            }
#endif
            {
                DeviceZoneScopedN("KV_RMSNORM");
                deepseek_b1_ops::RMSNorm::Op<KV_RMSNormCTArgs, Core::is_kv_rmsnorm_core, true> kv_rmsnorm;
                kv_rmsnorm(kv_rmsnorm_args);
            }
            {
                DeviceZoneScopedN("K_ROPE");
                deepseek_b1_ops::Rope::Op<K_RopeCTArgs, Core::is_krope_core> krope;
                krope(krope_args);
            }
        }
    }  // stage 8

#pragma GCC diagnostic pop
}
