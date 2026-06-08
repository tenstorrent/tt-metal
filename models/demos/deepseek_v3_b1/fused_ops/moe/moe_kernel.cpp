// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused MoE kernel: Routed Expert + Shared Expert
// Single kernel file, compiles correctly for all RISC cores.
// Three-level structure mirrors Python-side: MoeOp > MoeRoutedExpertOp / MoeSharedExpertOp
//   Core   → core role flags      (Core::Routed::*, Core::Shared::*)
//   Moe    → compile-time args    (Moe::Routed::*, Moe::Shared::*)
//
// Fused pipeline (step order):
//   0.  Residual Mcast (sender → 130 cores, pre-RMSNorm input)
//  0b.  RMSNorm (sender core: raw input → normalized input)
//   1.  RMSNorm Mcast (normalized input → routed + shared)
//   2.  Gate Matmul+Sigmoid (routed, 64 cores)
//   3.  Gate Gather (routed, 64 cores → sender)
//  3a.  Gate/Up KN-sliced matmul (shared, 128 cores — overlaps with step 2)
//   4.  Gate TopK (routed, sender core)
//  4a.  Gate Gather A (shared, 64 gate cores → sender)
//  4b.  Up Gather B (shared, 64 up cores → sender)
//   5.  Mcast Index (routed, sender → compute cores)
//  5b.  Mcast Expert Scale (routed, sender → gate_proj cores)
//  5c.  Gated Reduce (shared, sender core): SiLU(sum(gate)) * sum(up)
//   6.  gate_proj DRAM MM+SiLU (routed, 8 DRAM cores)
//   7.  up_proj DRAM MM (routed, 8 DRAM cores)
//   8.  Fused mul (routed, 8 DRAM cores)
//   9.  down_proj Gather (routed, gate_proj → sender)
//  10.  down_proj Mcast (routed, sender → gate_proj)
//  11.  down_proj DRAM MM (routed, 8 DRAM cores)
// 11b.  Down Mcast (shared, sender → 112 cores): gated reduce output [1, K_down]
// 11c.  Down Proj Matmul (shared, 112 cores): SRAM matmul [1,K_down] x [K_down,N_per_core]
// 11d.  Residual Add (shared, 112 cores): matmul_out + shard(residual)
// 11e.  Output Gather (shared, 112 cores → sender): collect residual add results
// 11f.  Output Mcast (shared, sender → 130 cores): DRAM cores receive into add_cb_in1
//  12.  Eltwise Add (routed, down_proj + shared_expert_output)
//
// Optimizations:
//   - gate_proj and up_proj share same CB (ResetCBIn1 between uses)
//   - Input mcast shared between routed and shared expert
//   - Residual mcast has no src CB — sender reads from tensor L1 address directly
//   - Shared gathers placed after TopK so sender BRISC doesn't block routed mcasts
//   - Gated Reduce placed after routed mcasts to overlap with DRAM matmuls

#include "../../unified_kernels/kernel_op_api.hpp"
#include "../../unified_kernels/kernel_utils.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/matmul.hpp"
#include "../../unified_kernels/moe_gather.hpp"
#include "../../unified_kernels/deepseek_moe_gate.hpp"
#if defined(COMPILE_FOR_TRISC)
#undef REDUCE_OP
#undef REDUCE_DIM
#endif
#include "../../unified_kernels/rmsnorm.hpp"
#include "../../unified_kernels/matmul_expert_compressed_dram.hpp"
#include "../../unified_kernels/eltwise_mul.hpp"
#include "../../unified_kernels/eltwise_add.hpp"
#include "../../unified_kernels/kn_sliced_matmul.hpp"
#include "../../unified_kernels/gated_reduce.hpp"
#include "../../unified_kernels/residual_add.hpp"
#include "../../unified_kernels/eltwise_add_or_copy.hpp"
#include "dram_helpers.hpp"
#include "sram_helpers.hpp"
#ifdef ENABLE_REDUCE_TO_ONE
#include "../../unified_kernels/reduce_to_one_b1.hpp"
#endif
#ifdef ENABLE_BCAST
#include "../../unified_kernels/broadcast.hpp"
#endif
// SRAM routed expert always available; gated at runtime via sram_*_active CT arg.
// Default (no SRAM experts) → sram_*_active=0 → Op call is `if constexpr false` no-op.
#include "../../unified_kernels/matmul_expert_compressed_sram.hpp"

// Compile-time role flags for dead code elimination via if constexpr.
// Mirrors Python-side MoeRoutedExpertOp / MoeSharedExpertOp split.
struct Core {
    // Shared between routed and shared expert — same physical core/grid
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;

    struct Routed {
        static constexpr bool is_gate_mm_core = get_named_compile_time_arg_val("is_gate_mm_core") == 1;
        static constexpr bool is_gate_proj_core = get_named_compile_time_arg_val("is_gate_proj_core") == 1;
        // 16-core gate_proj grid (8 primaries + 8 secondaries) when K-split is on for
        // gate_proj (cores_per_dram_bank=2, k_parallel_per_bank=2). Both senders (= K-slice 0)
        // and the K-reducer (= primary, K-slice 1) must be active so partials reduce on the
        // primary. Falls back to is_gate_proj_core (8 cores) when K-split is off.
        static constexpr bool is_gate_proj_streamer_core =
            get_named_compile_time_arg_val("is_gate_proj_streamer_core") == 1;
        // 16-core down_proj grid (8 primaries + 8 secondaries) when primary_at_last_offset=True;
        // 8 cores otherwise. Senders (post-swap = secondaries) MUST be active so they
        // NOC-write their accum onto the receiver/primary.
        static constexpr bool is_down_proj_streamer_core =
            get_named_compile_time_arg_val("is_down_proj_streamer_core") == 1;
    };
    struct Shared {
        static constexpr bool is_compute_core = get_named_compile_time_arg_val("is_shared_compute_core") == 1;
        static constexpr bool is_gate_compute_core = get_named_compile_time_arg_val("is_shared_gate_compute_core") == 1;
        static constexpr bool is_up_compute_core = get_named_compile_time_arg_val("is_shared_up_compute_core") == 1;
        static constexpr bool is_gated_reduce_core = get_named_compile_time_arg_val("is_shared_gated_reduce_core") == 1;
        static constexpr bool is_mcast_receiver_core =
            get_named_compile_time_arg_val("is_shared_mcast_receiver_core") == 1;
    };
    // Combined: cores that receive the input mcast. is_gate_proj_streamer_core is a
    // superset of is_gate_proj_core (16 cores vs 8 in K-split mode) so it must consume
    // the input mcast for matmul to have its activation.
    static constexpr bool is_input_mcast_receiver =
        Routed::is_gate_mm_core || Routed::is_gate_proj_streamer_core || Shared::is_compute_core;

    // Reduce-to-one core roles
    static constexpr bool is_reduce_worker_core = get_named_compile_time_arg_val("is_reduce_worker_core") == 1;
    static constexpr bool is_reduce_fabric_core = get_named_compile_time_arg_val("is_reduce_fabric_core") == 1;
};

// Count SRAM-flagged entries (bit-15) in the encoded TopK index array at
// `l1_addr`, after waiting for `cb_id` to be ready. On TRISC, only UNPACK
// scans (volatile L1 reads on MATH/PACK race against gate-kernel pushes
// even after cb_wait_front returns) and broadcasts the final count to
// MATH/PACK via the TRISC mailbox. On BRISC/NCRISC the scan is single-
// threaded.
// Multi-RISC scan over a 1-tile CB. Only BRISC does cb_wait_front; the other
// 4 RISCs reach the sync barrier and wait there. After sync_riscs_exit, all
// 5 RISCs are past the barrier and can volatile-read the indices L1 directly.
template <uint32_t cb_id, uint32_t l1_addr, uint32_t num_active, uint32_t sync_sem_addr>
FORCE_INLINE uint32_t scan_n_sram_active() {
    volatile uint32_t tt_l1_ptr* sync_sem = reinterpret_cast<volatile uint32_t tt_l1_ptr*>(sync_sem_addr);
#if defined(COMPILE_FOR_BRISC)
    cb_wait_front(cb_id, 1);
#endif
    unified_kernels::sync_riscs_enter<>(sync_sem);
    unified_kernels::sync_riscs_exit<>(sync_sem);

    uint32_t n = 0;
    volatile tt_l1_ptr uint16_t* idx = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t e = 0; e < num_active; e++) {
        if (deepseek_b1_ops::is_sram_expert(static_cast<uint32_t>(idx[e]))) {
            n++;
        }
    }
    return n;
}

void kernel_main() {
#if defined(RECONFIG_MOE_CBS)
    {
        constexpr uint32_t cb_config_l1_addr = get_named_compile_time_arg_val("reconfig_cb_config_l1_addr");
        uint32_t tt_l1_ptr* cb_config = reinterpret_cast<uint32_t tt_l1_ptr*>(cb_config_l1_addr);
        unified_kernels::reconfig_cb_interfaces(cb_config);
    }
#endif
// ============================================================================
// Compile-time args — grouped into Moe::Routed / Moe::Shared structs
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)

    struct Moe {
        struct Routed {
            // Mcast (receiver)
            using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::DMArgs mcast_args{
                .sender = {},
                .receiver = {
                    get_named_compile_time_arg_val("moe_mcast_data_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("moe_mcast_dst_cb"),
                    get_named_compile_time_arg_val("moe_mcast_dst_num_pages"),
                }};

#ifdef ENABLE_ROUTING
            // Gate Matmul (reader — no-op)
            using GateMMCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
            deepseek_b1_ops::Matmul::ReaderArgs gate_mm_args{};

            // Gather (receiver — MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs gather_args{
                get_named_compile_time_arg_val("gather_noc0_num_senders"),
                get_named_compile_time_arg_val("gather_noc1_num_senders"),
                get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("gather_dst_cb"),
                get_named_compile_time_arg_val("gather_dst_num_pages"),
            };

            // Gate (reader — no-op)
            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs;

            // Index Mcast (no-op on NCRISC — receiver on BRISC)
            deepseek_b1_ops::Mcast::DMArgs index_mcast_args{.sender = {}, .receiver = {}};

            // Expert Scale Mcast (no-op on NCRISC — receiver on BRISC)
            deepseek_b1_ops::Mcast::DMArgs expert_scale_mcast_args{.sender = {}, .receiver = {}};
#endif  // ENABLE_ROUTING

            // gate_proj DRAM Matmul Expert Compressed (reader) — pr42896 36-param shape
            using GateProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
                get_named_compile_time_arg_val("gate_proj_cb_in0"),
                get_named_compile_time_arg_val("gate_proj_cb_in1"),
                get_named_compile_time_arg_val("gate_proj_cb_out"),
                get_named_compile_time_arg_val("gate_proj_cb_index"),
                get_named_compile_time_arg_val("gate_proj_num_tiles_k"),
                get_named_compile_time_arg_val("gate_proj_subblock_k"),
                get_named_compile_time_arg_val("gate_proj_subblock_n"),
                get_named_compile_time_arg_val("gate_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("gate_proj_per_core_n"),
                get_named_compile_time_arg_val("gate_proj_bank_id"),
                get_named_compile_time_arg_val("gate_proj_vc"),
                get_named_compile_time_arg_val("gate_proj_expert_offsets_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_block_sizes_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_in1_size_bytes"),
                get_named_compile_time_arg_val("gate_proj_noc_max_page_size"),
                get_named_compile_time_arg_val("gate_proj_core_in_bank_idx"),
                get_named_compile_time_arg_val("gate_proj_pipeline_sem_addr"),
                get_named_compile_time_arg_val("gate_proj_next_core_noc_x"),
                get_named_compile_time_arg_val("gate_proj_next_core_noc_y"),
                get_named_compile_time_arg_val("gate_proj_cores_per_bank"),
                get_named_compile_time_arg_val("gate_proj_num_active_experts"),
                get_named_compile_time_arg_val("gate_proj_index_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_fmt"),
                get_named_compile_time_arg_val("gate_proj_fmt_dram_addr"),
                get_named_compile_time_arg_val("gate_proj_fmt_per_expert_bytes"),
                get_named_compile_time_arg_val("gate_proj_fmt_per_core_bytes"),
                get_named_compile_time_arg_val("gate_proj_fmt_cb_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_fmt_cb_page_size"),
                get_named_compile_time_arg_val("gate_proj_fmt_sem_addr_0"),
                get_named_compile_time_arg_val("gate_proj_fmt_sem_addr_1"),
                get_named_compile_time_arg_val("gate_proj_accum_experts"),
                get_named_compile_time_arg_val("gate_proj_index_offset"),
                get_named_compile_time_arg_val("gate_proj_k_parallel_per_bank"),
                get_named_compile_time_arg_val("gate_proj_k_slice_idx"),
                get_named_compile_time_arg_val("gate_proj_num_subblocks_k_local"),
                get_named_compile_time_arg_val("gate_proj_partial_sem_addr"),
                get_named_compile_time_arg_val("gate_proj_primary_at_last_offset"),
                get_named_compile_time_arg_val("gate_proj_gather_sync_sem_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_internal_acc"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("gate_proj_num_dram_experts_pre_selected")>;

            // up_proj DRAM Matmul Expert Compressed (reader) — shares weight CB with gate_proj
            using UpProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
                get_named_compile_time_arg_val("up_proj_cb_in0"),
                get_named_compile_time_arg_val("up_proj_cb_in1"),
                get_named_compile_time_arg_val("up_proj_cb_mm_out"),
                get_named_compile_time_arg_val("up_proj_cb_index"),
                get_named_compile_time_arg_val("up_proj_num_tiles_k"),
                get_named_compile_time_arg_val("up_proj_subblock_k"),
                get_named_compile_time_arg_val("up_proj_subblock_n"),
                get_named_compile_time_arg_val("up_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("up_proj_per_core_n"),
                get_named_compile_time_arg_val("up_proj_bank_id"),
                get_named_compile_time_arg_val("up_proj_vc"),
                get_named_compile_time_arg_val("up_proj_expert_offsets_l1_addr"),
                get_named_compile_time_arg_val("up_proj_block_sizes_l1_addr"),
                get_named_compile_time_arg_val("up_proj_cb_in1_size_bytes"),
                get_named_compile_time_arg_val("up_proj_noc_max_page_size"),
                get_named_compile_time_arg_val("up_proj_core_in_bank_idx"),
                get_named_compile_time_arg_val("up_proj_pipeline_sem_addr"),
                get_named_compile_time_arg_val("up_proj_next_core_noc_x"),
                get_named_compile_time_arg_val("up_proj_next_core_noc_y"),
                get_named_compile_time_arg_val("up_proj_cores_per_bank"),
                get_named_compile_time_arg_val("up_proj_num_active_experts"),
                get_named_compile_time_arg_val("up_proj_index_l1_addr"),
                get_named_compile_time_arg_val("up_proj_cb_fmt"),
                get_named_compile_time_arg_val("up_proj_fmt_dram_addr"),
                get_named_compile_time_arg_val("up_proj_fmt_per_expert_bytes"),
                get_named_compile_time_arg_val("up_proj_fmt_per_core_bytes"),
                get_named_compile_time_arg_val("up_proj_fmt_cb_l1_addr"),
                get_named_compile_time_arg_val("up_proj_fmt_cb_page_size"),
                get_named_compile_time_arg_val("up_proj_fmt_sem_addr_0"),
                get_named_compile_time_arg_val("up_proj_fmt_sem_addr_1"),
                get_named_compile_time_arg_val("up_proj_accum_experts"),
                get_named_compile_time_arg_val("up_proj_index_offset"),
                get_named_compile_time_arg_val("up_proj_k_parallel_per_bank"),
                get_named_compile_time_arg_val("up_proj_k_slice_idx"),
                get_named_compile_time_arg_val("up_proj_num_subblocks_k_local"),
                get_named_compile_time_arg_val("up_proj_partial_sem_addr"),
                get_named_compile_time_arg_val("up_proj_primary_at_last_offset"),
                get_named_compile_time_arg_val("up_proj_gather_sync_sem_addr"),
                get_named_compile_time_arg_val("up_proj_cb_internal_acc"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("up_proj_num_dram_experts_pre_selected")>;

            // Eltwise Mul (reader — no-op)
            using MulCTArgs = deepseek_b1_ops::EltwiseMul::ReaderCTArgs;

            // down_proj Gather (receiver — MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs down_proj_gather_args{
                get_named_compile_time_arg_val("down_proj_gather_noc0_num_senders"),
                get_named_compile_time_arg_val("down_proj_gather_noc1_num_senders"),
                get_named_compile_time_arg_val("down_proj_gather_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_gather_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("down_proj_gather_dst_cb"),
                get_named_compile_time_arg_val("down_proj_gather_dst_num_pages"),
            };

            // down_proj Mcast (no-op on NCRISC — receiver moved to BRISC)
            deepseek_b1_ops::Mcast::DMArgs down_proj_mcast_args{.sender = {}, .receiver = {}};
            // SRAM down Mcast (no-op on NCRISC — receiver on BRISC)
            deepseek_b1_ops::Mcast::DMArgs sram_down_mcast_args{.sender = {}, .receiver = {}};

            // down_proj DRAM Matmul Expert Compressed (reader) — pr42896 36-param shape
            using DownProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
                get_named_compile_time_arg_val("down_proj_cb_in0"),
                get_named_compile_time_arg_val("down_proj_cb_in1"),
                get_named_compile_time_arg_val("down_proj_cb_out"),
                get_named_compile_time_arg_val("down_proj_cb_index"),
                get_named_compile_time_arg_val("down_proj_num_tiles_k"),
                get_named_compile_time_arg_val("down_proj_subblock_k"),
                get_named_compile_time_arg_val("down_proj_subblock_n"),
                get_named_compile_time_arg_val("down_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("down_proj_per_core_n"),
                get_named_compile_time_arg_val("down_proj_bank_id"),
                get_named_compile_time_arg_val("down_proj_vc"),
                get_named_compile_time_arg_val("down_proj_expert_offsets_l1_addr"),
                get_named_compile_time_arg_val("down_proj_block_sizes_l1_addr"),
                get_named_compile_time_arg_val("down_proj_cb_in1_size_bytes"),
                get_named_compile_time_arg_val("down_proj_noc_max_page_size"),
                get_named_compile_time_arg_val("down_proj_core_in_bank_idx"),
                get_named_compile_time_arg_val("down_proj_pipeline_sem_addr"),
                get_named_compile_time_arg_val("down_proj_next_core_noc_x"),
                get_named_compile_time_arg_val("down_proj_next_core_noc_y"),
                get_named_compile_time_arg_val("down_proj_cores_per_bank"),
                get_named_compile_time_arg_val("down_proj_num_active_experts"),
                get_named_compile_time_arg_val("down_proj_index_l1_addr"),
                get_named_compile_time_arg_val("down_proj_cb_fmt"),
                get_named_compile_time_arg_val("down_proj_fmt_dram_addr"),
                get_named_compile_time_arg_val("down_proj_fmt_per_expert_bytes"),
                get_named_compile_time_arg_val("down_proj_fmt_per_core_bytes"),
                get_named_compile_time_arg_val("down_proj_fmt_cb_l1_addr"),
                get_named_compile_time_arg_val("down_proj_fmt_cb_page_size"),
                get_named_compile_time_arg_val("down_proj_fmt_sem_addr_0"),
                get_named_compile_time_arg_val("down_proj_fmt_sem_addr_1"),
                get_named_compile_time_arg_val("down_proj_accum_experts"),
                get_named_compile_time_arg_val("down_proj_index_offset"),
                get_named_compile_time_arg_val("down_proj_k_parallel_per_bank"),
                get_named_compile_time_arg_val("down_proj_k_slice_idx"),
                get_named_compile_time_arg_val("down_proj_num_subblocks_k_local"),
                get_named_compile_time_arg_val("down_proj_partial_sem_addr"),
                get_named_compile_time_arg_val("down_proj_primary_at_last_offset"),
                get_named_compile_time_arg_val("down_proj_gather_sync_sem_addr"),
                get_named_compile_time_arg_val("down_proj_cb_internal_acc"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("down_proj_num_dram_experts_pre_selected")>;

            // SRAM gate_proj Matmul Expert (reader, NCRISC).
            // Mirrors MatmulExpertCompressedSRAM::ReaderCTArgs (12 params).
            // Always defined; Op call gated by sram_gate_proj_active CT arg.
            using SramGateProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ReaderCTArgs<
                get_named_compile_time_arg_val("sram_gate_proj_cb_in0"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_in1"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_out"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_index"),
                get_named_compile_time_arg_val("sram_gate_proj_num_tiles_k"),
                get_named_compile_time_arg_val("sram_gate_proj_out_w"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_in0_num_pages"),
                get_named_compile_time_arg_val("sram_gate_proj_fmt_l1_addr"),
                get_named_compile_time_arg_val("sram_gate_proj_num_active_experts"),
                get_named_compile_time_arg_val("sram_gate_proj_index_l1_addr"),
                get_named_compile_time_arg_val("sram_gate_proj_k_per_core"),
                get_named_compile_time_arg_val("sram_gate_proj_k_offset")>;

            // SRAM up_proj Matmul Expert (reader, NCRISC) — mirror of gate_proj.
            using SramUpProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ReaderCTArgs<
                get_named_compile_time_arg_val("sram_up_proj_cb_in0"),
                get_named_compile_time_arg_val("sram_up_proj_cb_in1"),
                get_named_compile_time_arg_val("sram_up_proj_cb_out"),
                get_named_compile_time_arg_val("sram_up_proj_cb_index"),
                get_named_compile_time_arg_val("sram_up_proj_num_tiles_k"),
                get_named_compile_time_arg_val("sram_up_proj_out_w"),
                get_named_compile_time_arg_val("sram_up_proj_cb_in0_num_pages"),
                get_named_compile_time_arg_val("sram_up_proj_fmt_l1_addr"),
                get_named_compile_time_arg_val("sram_up_proj_num_active_experts"),
                get_named_compile_time_arg_val("sram_up_proj_index_l1_addr"),
                get_named_compile_time_arg_val("sram_up_proj_k_per_core"),
                get_named_compile_time_arg_val("sram_up_proj_k_offset")>;

            // SRAM down_proj Matmul Expert (reader, NCRISC) — runs on the 112
            // shared mcast receiver cores with accum_experts=1 + compact_in0=1.
            using SramDownProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ReaderCTArgs<
                get_named_compile_time_arg_val("sram_down_proj_cb_in0"),
                get_named_compile_time_arg_val("sram_down_proj_cb_in1"),
                get_named_compile_time_arg_val("sram_down_proj_cb_out"),
                get_named_compile_time_arg_val("sram_down_proj_cb_index"),
                get_named_compile_time_arg_val("sram_down_proj_num_tiles_k"),
                get_named_compile_time_arg_val("sram_down_proj_out_w"),
                get_named_compile_time_arg_val("sram_down_proj_cb_in0_num_pages"),
                get_named_compile_time_arg_val("sram_down_proj_fmt_l1_addr"),
                get_named_compile_time_arg_val("sram_down_proj_num_active_experts"),
                get_named_compile_time_arg_val("sram_down_proj_index_l1_addr"),
                get_named_compile_time_arg_val("sram_down_proj_k_per_core"),
                get_named_compile_time_arg_val("sram_down_proj_k_offset")>;

            // SRAM extended GatedReduce (NCRISC no-op).
            using SramGatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ReaderCTArgs;
            deepseek_b1_ops::GatedReduce::ReaderArgs sram_gated_reduce_args{};

            // SRAM down merge (NCRISC no-op).
            using SramDownMergeCTArgs = deepseek_b1_ops::EltwiseAddOrCopy::ReaderCTArgs;
            deepseek_b1_ops::EltwiseAddOrCopy::ReaderArgs sram_down_merge_args{};

            // SRAM gate/up gather receivers (NCRISC) — sender_core only.
            // dst_num_pages = num_active × pages_per_expert is a compile-time
            // constant: receivers always advance by full capacity to match
            // GR's padded out_cb push count downstream. Sender (BRISC) only
            // NOCs n_sram_active experts via sram_invoke_moe_gather.
            deepseek_b1_ops::MoeGather::ReceiverArgs sram_ag_args{
                get_named_compile_time_arg_val("sram_ag_noc0_num_senders"),
                0,  // noc1_num_senders
                get_named_compile_time_arg_val("sram_ag_noc0_receiver_semaphore_addr"),
                0,  // noc1_receiver_semaphore_addr
                get_named_compile_time_arg_val("sram_ag_dst_cb"),
                get_named_compile_time_arg_val("sram_gather_num_active_experts") *
                    get_named_compile_time_arg_val("sram_gather_pages_per_expert"),
            };
            deepseek_b1_ops::MoeGather::ReceiverArgs sram_bg_args{
                get_named_compile_time_arg_val("sram_bg_noc0_num_senders"),
                0,
                get_named_compile_time_arg_val("sram_bg_noc0_receiver_semaphore_addr"),
                0,
                get_named_compile_time_arg_val("sram_bg_dst_cb"),
                get_named_compile_time_arg_val("sram_gather_num_active_experts") *
                    get_named_compile_time_arg_val("sram_gather_pages_per_expert"),
            };

            // Eltwise Add (reader — no-op). Uses EltwiseAddOrCopy so the final
            // DRAM add can fall back to copy(shared_output) when n_dram_active==0.
            using AddCTArgs = deepseek_b1_ops::EltwiseAddOrCopy::ReaderCTArgs;
            deepseek_b1_ops::EltwiseAddOrCopy::ReaderArgs add_args{};

            // Residual Mcast — receiver (input from sender → residual CB)
            using ResidualMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::DMArgs residual_mcast_args{
                .sender = {},
                .receiver = {
                    get_named_compile_time_arg_val("shared_residual_mcast_data_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("shared_residual_cb"),
                    get_named_compile_time_arg_val("shared_residual_num_pages"),
                }};

            // RMSNorm (reader — no-op)
            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
            deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

#ifdef ENABLE_REDUCE_TO_ONE
            // ReduceToOneB1 (reader — receives data from fabric via semaphore waits)
            using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ReaderCTArgs<
                get_named_compile_time_arg_val("reduce_device_role"),
                get_named_compile_time_arg_val("reduce_num_tiles"),
                get_named_compile_time_arg_val("reduce_local_cb"),
                get_named_compile_time_arg_val("reduce_received_cb"),
                get_named_compile_time_arg_val("is_reduce_fabric_core")>;

            // Reader runtime args (common RT args at configurable base)
            deepseek_b1_ops::ReduceToOneB1::ReaderArgs reduce_rt_args{
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("reduce_ncrisc_common_rt_arg_base") + 0),
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("reduce_ncrisc_common_rt_arg_base") + 1),
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("reduce_ncrisc_common_rt_arg_base") + 2),
            };
#endif
        } routed;

        struct Shared {
            // KN-sliced matmul (reader — no-op for NCRISC)
            using GUMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ReaderCTArgs;
            deepseek_b1_ops::KNSlicedMatmul::ReaderArgs gu_matmul_args{};

            // Gate Gather (A) receiver (MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs ag_args{
                get_named_compile_time_arg_val("shared_ag_noc0_num_senders"),
                0,  // noc1_num_senders
                get_named_compile_time_arg_val("shared_ag_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_ag_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_ag_dst_cb"),
                get_named_compile_time_arg_val("shared_ag_dst_num_pages"),
            };

            // Up Gather (B) receiver (MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs bg_args{
                get_named_compile_time_arg_val("shared_bg_noc0_num_senders"),
                0,  // noc1_num_senders
                get_named_compile_time_arg_val("shared_bg_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_bg_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_bg_dst_cb"),
                get_named_compile_time_arg_val("shared_bg_dst_num_pages"),
            };

            // Gated Reduce (reader — no-op for NCRISC)
            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ReaderCTArgs;
            deepseek_b1_ops::GatedReduce::ReaderArgs gated_reduce_args{};

            // Down Mcast — receiver (gated reduce output → all 130 cores)
            using DownMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::DMArgs down_mcast_args{
                .sender = {},
                .receiver = {
                    get_named_compile_time_arg_val("shared_down_mcast_data_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("shared_down_mcast_dst_cb"),
                    get_named_compile_time_arg_val("shared_down_mcast_dst_num_pages"),
                }};

            // Down Proj Matmul — reader (no-op for NCRISC)
            using DownMatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
            deepseek_b1_ops::Matmul::ReaderArgs down_matmul_args{};

            // Residual Add — reader (no-op for NCRISC)
            using ResidualAddCTArgs = deepseek_b1_ops::ResidualAdd::ReaderCTArgs;
            deepseek_b1_ops::ResidualAdd::ReaderArgs residual_add_args{};

            // Output Gather — receiver (MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs og_args{
                get_named_compile_time_arg_val("shared_og_noc0_num_senders"),
                get_named_compile_time_arg_val("shared_og_noc1_num_senders"),
                get_named_compile_time_arg_val("shared_og_noc0_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_og_noc1_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_og_dst_cb"),
                get_named_compile_time_arg_val("shared_og_dst_num_pages"),
            };

            // Output Mcast — no-op on NCRISC (receiver moved to BRISC)
            using OutputMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::DMArgs output_mcast_args{.sender = {}, .receiver = {}};
        } shared;
    } moe;

    // Setup all tensor-backed sharded buffers (marks pre-loaded tiles as ready)
    auto setup_all_sharded_buffers = [&]() {
        if constexpr (Core::is_sender_core) {
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("moe_rmsnorm_gamma_cb"),
                get_named_compile_time_arg_val("moe_rmsnorm_gamma_num_pages"));
#ifdef ENABLE_ROUTING
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("gate_bias_cb"), 1);
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("gate_input_indices_cb"), 1);
#endif
#ifdef ENABLE_BCAST
            constexpr bool bcast_use_socket = get_named_compile_time_arg_val("bcast_use_socket") == 1;
            if constexpr (!bcast_use_socket) {
                constexpr uint32_t bcast_pkt_cb = get_named_compile_time_arg_val("bcast_pkt_cb");
                constexpr uint32_t bcast_num_pages = get_named_compile_time_arg_val("bcast_num_pages_to_read");
                unified_kernels::setup_sharded_buffer(bcast_pkt_cb, bcast_num_pages);
            }
#else
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("shared_residual_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages"));
#endif
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
            // mul_cb_in1 == gate_proj_cb_out is filled dynamically by gate_proj TRISC;
            // pre-loading it would fill CB capacity and deadlock gate_proj at expert 7.
            unified_kernels::setup_sharded_buffer(
                get_named_compile_time_arg_val("add_cb_in0"), get_named_compile_time_arg_val("add_cb_in0_wait_tiles"));
        }
        // SRAM gate_proj weights are L1-resident on the 64 shared gate cores.
        // Push 1 page once; kernel never pops cb_in1 (pop_in1=false), so the
        // page persists across iterations. Skipped when SRAM is disabled
        // (num_active_experts==0) — the cb_in1 has no host-side descriptor in
        // that case, so cb_reserve_back would hang.
        if constexpr (
            Core::Shared::is_gate_compute_core &&
            get_named_compile_time_arg_val("sram_gate_proj_num_active_experts") > 0) {
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("sram_gate_proj_cb_in1"), 1);
        }
        // SRAM up_proj weights are L1-resident on the 64 shared up cores.
        if constexpr (
            Core::Shared::is_up_compute_core && get_named_compile_time_arg_val("sram_up_proj_num_active_experts") > 0) {
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("sram_up_proj_cb_in1"), 1);
        }
        // SRAM down_proj weights are L1-resident on the 112 shared mcast receiver cores.
        if constexpr (
            Core::Shared::is_mcast_receiver_core &&
            get_named_compile_time_arg_val("sram_down_proj_num_active_experts") > 0) {
            unified_kernels::setup_sharded_buffer(get_named_compile_time_arg_val("sram_down_proj_cb_in1"), 1);
        }
    };
#ifndef RECONFIG_MOE_CBS
    setup_all_sharded_buffers();
#endif

#elif defined(COMPILE_FOR_BRISC)

    struct Moe {
        struct Routed {
            // Mcast (sender)
            using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
                get_named_compile_time_arg_val("moe_mcast_num_cores"),
                get_named_compile_time_arg_val("moe_mcast_is_part_of_receiver_grid"),
                Core::is_sender_core && Core::is_mcast_grid_core>;
            deepseek_b1_ops::Mcast::DMArgs mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {}};

#ifdef ENABLE_ROUTING
            // Gate Matmul (writer — no-op)
            using GateMMCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
            deepseek_b1_ops::Matmul::WriterArgs gate_mm_args{};

            // Gather (sender — MoeGather: sender on BRISC)
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
                0,  // sender_idx (unused when UsePerCoreSenderIdx=false)
                1,  // num_experts (single write)
                0,  // src_page_size (unused)
                0,  // expert_dst_stride (unused)
            };

            // Gate (writer)
            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::WriterCTArgs<
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("gate_output_indices_cb")>;

            // Index Mcast (sender + receiver on BRISC)
            deepseek_b1_ops::Mcast::DMArgs index_mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {
                    get_named_compile_time_arg_val("index_mcast_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("gate_proj_cb_index"),
                    get_named_compile_time_arg_val("index_mcast_num_pages"),
                }};

            // Expert Scale Mcast (sender + receiver on BRISC)
            deepseek_b1_ops::Mcast::DMArgs expert_scale_mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {
                    get_named_compile_time_arg_val("expert_scale_mcast_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("mul_cb_scalar_src"),
                    get_named_compile_time_arg_val("expert_scale_mcast_num_pages"),
                }};
#endif  // ENABLE_ROUTING

            // DRAM Matmul Expert Compressed (writer — empty, BRISC is no-op)
            using GateProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::WriterCTArgs;
            using UpProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::WriterCTArgs;

            // Eltwise Mul (writer)
            using MulCTArgs = deepseek_b1_ops::EltwiseMul::WriterCTArgs<
                get_named_compile_time_arg_val("mul_cb_out"),
                get_named_compile_time_arg_val("mul_num_tiles"),
                get_named_compile_time_arg_val("mul_cb_scalar"),
                get_named_compile_time_arg_val("mul_cb_scalar_src"),
                get_named_compile_time_arg_val("mul_scalar_index_offset"),
                get_named_compile_time_arg_val("enable_routing"),  // enable_scalar
                get_named_compile_time_arg_val("mul_num_experts")>;

            // down_proj Gather (receiver)
            // down_proj Gather (sender — MoeGather: sender on BRISC)
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
                get_named_compile_time_arg_val("down_proj_gather_num_experts"),
                get_named_compile_time_arg_val("down_proj_gather_src_page_size"),
                get_named_compile_time_arg_val("down_proj_gather_expert_dst_stride"),
            };

            // down_proj Mcast (sender + receiver on BRISC)
            deepseek_b1_ops::Mcast::DMArgs down_proj_mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {
                    get_named_compile_time_arg_val("down_proj_mcast_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("down_proj_mcast_dst_cb"),
                    get_named_compile_time_arg_val("down_proj_mcast_dst_num_pages"),
                }};

            // SRAM down Mcast (sender + receiver on BRISC). data_size_bytes /
            // src_num_pages / dst_num_pages get overridden at runtime to
            // n_sram_active * face_tile_size / n_sram_active in the body.
            deepseek_b1_ops::Mcast::DMArgs sram_down_mcast_args{
                .sender =
                    {
                        get_named_compile_time_arg_val("moe_mcast_dest_noc_start_x"),
                        get_named_compile_time_arg_val("moe_mcast_dest_noc_start_y"),
                        get_named_compile_time_arg_val("moe_mcast_dest_noc_end_x"),
                        get_named_compile_time_arg_val("moe_mcast_dest_noc_end_y"),
                        get_named_compile_time_arg_val("sram_down_mcast_sender_semaphore_addr"),
                        get_named_compile_time_arg_val("sram_down_mcast_receiver_semaphore_addr"),
                        get_named_compile_time_arg_val("sram_down_mcast_data_size_bytes"),
                        get_named_compile_time_arg_val("sram_down_mcast_src_cb"),
                        get_named_compile_time_arg_val("sram_down_mcast_src_num_pages"),
                        get_read_ptr(get_named_compile_time_arg_val("sram_down_mcast_src_cb")),
                        get_write_ptr(get_named_compile_time_arg_val("sram_down_mcast_dst_cb")),
                    },
                .receiver = {
                    get_named_compile_time_arg_val("sram_down_mcast_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("sram_down_mcast_dst_cb"),
                    get_named_compile_time_arg_val("sram_down_mcast_dst_num_pages"),
                }};

            // down_proj DRAM Matmul Expert Compressed (writer — empty, BRISC is no-op)
            using DownProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::WriterCTArgs;

            // SRAM gate_proj Matmul Expert (writer — empty, BRISC is no-op).
            using SramGateProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::WriterCTArgs;
            // SRAM up_proj Matmul Expert (writer — empty, BRISC is no-op).
            using SramUpProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::WriterCTArgs;
            // SRAM down_proj Matmul Expert (writer — empty, BRISC is no-op).
            using SramDownProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::WriterCTArgs;

            // SRAM extended GatedReduce (BRISC: per-expert scalar copy).
            using SramGatedReduceCTArgs = deepseek_b1_ops::GatedReduce::WriterCTArgs<
                get_named_compile_time_arg_val("sram_gr_scalar_cb"),
                get_named_compile_time_arg_val("sram_gr_scalar_src_l1_addr"),
                get_named_compile_time_arg_val("sram_gather_sender_index_l1_addr"),
                get_named_compile_time_arg_val("sram_gather_num_active_experts")>;
            deepseek_b1_ops::GatedReduce::WriterArgs sram_gated_reduce_args{};

            // SRAM down merge (BRISC no-op).
            using SramDownMergeCTArgs = deepseek_b1_ops::EltwiseAddOrCopy::WriterCTArgs;
            deepseek_b1_ops::EltwiseAddOrCopy::WriterArgs sram_down_merge_args{};

            // SRAM gate gather (A) sender (BRISC). num_experts init=0; set at
            // runtime to n_sram_active. src_page_size=64 (1 tile),
            // expert_dst_stride=4096 (64 tiles).
            // Field name `sram_ag_args` matches NCRISC ReceiverArgs so the Op
            // call selects the right type via SelectByRISCV.
            deepseek_b1_ops::MoeGather::SenderArgs sram_ag_args{
                get_named_compile_time_arg_val("sram_ag_dest_noc_x"),
                get_named_compile_time_arg_val("sram_ag_dest_noc_y"),
                get_named_compile_time_arg_val("sram_ag_data_size_bytes"),
                get_named_compile_time_arg_val("sram_ag_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("sram_ag_src_cb"),
                get_named_compile_time_arg_val("sram_ag_src_num_pages"),
                0,
                0,
                0,
                0,  // sender_grid (unused with UsePerCoreSenderIdx)
                0,  // row_major (unused)
                get_named_compile_time_arg_val("sram_ag_receiver_data_addr"),
                get_named_compile_time_arg_val("sram_ag_sender_idx"),
                0,  // num_experts — runtime
                get_named_compile_time_arg_val("sram_ag_src_page_size"),
                get_named_compile_time_arg_val("sram_ag_expert_dst_stride"),
            };
            // SRAM up gather (B) sender (BRISC).
            deepseek_b1_ops::MoeGather::SenderArgs sram_bg_args{
                get_named_compile_time_arg_val("sram_bg_dest_noc_x"),
                get_named_compile_time_arg_val("sram_bg_dest_noc_y"),
                get_named_compile_time_arg_val("sram_bg_data_size_bytes"),
                get_named_compile_time_arg_val("sram_bg_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("sram_bg_src_cb"),
                get_named_compile_time_arg_val("sram_bg_src_num_pages"),
                0,
                0,
                0,
                0,
                0,
                get_named_compile_time_arg_val("sram_bg_receiver_data_addr"),
                get_named_compile_time_arg_val("sram_bg_sender_idx"),
                0,  // num_experts — runtime
                get_named_compile_time_arg_val("sram_bg_src_page_size"),
                get_named_compile_time_arg_val("sram_bg_expert_dst_stride"),
            };

            // Eltwise Add (writer — no-op)
            using AddCTArgs = deepseek_b1_ops::EltwiseAddOrCopy::WriterCTArgs;
            deepseek_b1_ops::EltwiseAddOrCopy::WriterArgs add_args{};

            // Residual Mcast — sender (input from sender → residual CB, pop_src=false)
            using ResidualMcastCTArgs = McastCTArgs;
            deepseek_b1_ops::Mcast::DMArgs residual_mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {}};

            // RMSNorm (writer — no-op)
            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::WriterCTArgs;
            deepseek_b1_ops::RMSNorm::WriterArgs rmsnorm_args{};

#ifdef ENABLE_REDUCE_TO_ONE
            // ReduceToOneB1 (writer — sends data via fabric or NOC)
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
                get_named_compile_time_arg_val("reduce_enable_downstream_socket"),
                get_named_compile_time_arg_val("reduce_brisc_fabric_rt_arg_base"),
                get_named_compile_time_arg_val("reduce_total_num_workers"),
                get_named_compile_time_arg_val("reduce_agg_output_size_bytes"),
                get_named_compile_time_arg_val("reduce_persistent_fabric_rt_arg_base"),
                get_named_compile_time_arg_val("reduce_persistent_fabric_signal_enable"),
                get_named_compile_time_arg_val("reduce_forward_metadata_size_bytes")>;

            deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs reduce_rt_args{};
            // Populated below after struct initialization
#endif
        } routed;

        struct Shared {
            // KN-sliced matmul (writer — no-op for BRISC)
            using GUMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::WriterCTArgs;
            deepseek_b1_ops::KNSlicedMatmul::WriterArgs gu_matmul_args{};

            // Gate Gather (A) sender (MoeGather: sender on BRISC)
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
                0,  // sender_grid (unused with UsePerCoreSenderIdx)
                0,  // row_major (unused)
                get_named_compile_time_arg_val("shared_ag_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_ag_sender_idx"),
                1,  // num_experts (single write per sender)
                0,  // src_page_size (unused for num_experts=1)
                0,  // expert_dst_stride (unused for num_experts=1)
            };

            // Up Gather (B) sender (MoeGather: sender on BRISC)
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
                1,  // num_experts (single write per sender)
                0,  // src_page_size (unused for num_experts=1)
                0,  // expert_dst_stride (unused for num_experts=1)
            };

            // Gated Reduce (writer — no-op for BRISC; defaults to all-zero CT)
            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::WriterCTArgs<>;
            deepseek_b1_ops::GatedReduce::WriterArgs gated_reduce_args{};

            // Down Mcast — sender (reuse Routed::McastCTArgs: same grid, same persistent sender)
            using DownMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::DMArgs down_mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {}};

            // Down Proj Matmul — writer (no-op for BRISC)
            using DownMatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
            deepseek_b1_ops::Matmul::WriterArgs down_matmul_args{};

            // Residual Add — writer (no-op for BRISC)
            using ResidualAddCTArgs = deepseek_b1_ops::ResidualAdd::WriterCTArgs;
            deepseek_b1_ops::ResidualAdd::WriterArgs residual_add_args{};

            // Output Gather — sender (MoeGather: sender on BRISC)
            deepseek_b1_ops::MoeGather::SenderArgs og_args{
                get_named_compile_time_arg_val("shared_og_dest_noc_x"),
                get_named_compile_time_arg_val("shared_og_dest_noc_y"),
                get_named_compile_time_arg_val("shared_og_data_size_bytes"),
                get_named_compile_time_arg_val("shared_og_receiver_semaphore_addr"),
                get_named_compile_time_arg_val("shared_og_src_cb"),
                get_named_compile_time_arg_val("shared_og_src_num_pages"),
                0,  // sender_grid_start_x (unused with UsePerCoreSenderIdx)
                0,  // sender_grid_start_y
                0,  // sender_grid_end_x
                0,  // sender_grid_end_y
                0,  // row_major (unused)
                get_named_compile_time_arg_val("shared_og_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_residual_add_core_idx"),  // reuse matmul core index
                1,  // num_experts (single write per sender)
                0,  // src_page_size (unused for num_experts=1)
                0,  // expert_dst_stride (unused for num_experts=1)
            };

            // Output Mcast — sender + receiver on BRISC (sender core → 130 cores)
            using OutputMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::DMArgs output_mcast_args{
                .sender =
                    {
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
                    },
                .receiver = {
                    get_named_compile_time_arg_val("shared_output_mcast_data_receiver_semaphore_addr"),
                    get_named_compile_time_arg_val("add_cb_in1"),
                    get_named_compile_time_arg_val("shared_output_mcast_dst_num_pages"),
                }};
        } shared;
    } moe;

#ifdef ENABLE_REDUCE_TO_ONE
    // Populate BRISC reduce runtime args (must be outside struct initializer)
    constexpr size_t reduce_brisc_arg_start = get_named_compile_time_arg_val("reduce_brisc_rt_arg_base");
    if constexpr (Core::is_reduce_worker_core) {
        moe.routed.reduce_rt_args = deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs{
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 0),   // fabric_core_noc_x
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 1),   // fabric_core_noc_y
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 2),   // my_slot_idx
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 3),   // worker_sem_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 4),   // dst_l1_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 5),   // dst_sem_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 6),   // output_base_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 7),   // shard_idx
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 8),   // socket_config_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 9),   // metadata_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 10),  // agg_sem_l1_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 11),  // agg_core_noc_x
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 12),  // agg_core_noc_y
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 13),  // persistent_enable
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 14),  // persistent_dst_noc_x
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 15),  // persistent_dst_noc_y
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 16),  // persistent_dst_mesh_id
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 17),  // persistent_dst_chip_id
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 18),  // persistent_dst_sem_addr
        };
    }
#endif

#elif defined(COMPILE_FOR_TRISC)

    struct Moe {
        struct Routed {
            // Mcast (compute — no-op)
            using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

#ifdef ENABLE_ROUTING
            // Gate Matmul (compute)
            using GateMMCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<
                get_named_compile_time_arg_val("gate_mm_out_w"),
                false,  // transpose
                get_named_compile_time_arg_val("gate_mm_fused_activation")>;
            deepseek_b1_ops::Matmul::ComputeArgs gate_mm_args{
                get_named_compile_time_arg_val("gate_mm_in0"),
                get_named_compile_time_arg_val("gate_mm_in1"),
                get_named_compile_time_arg_val("gate_mm_out"),
                get_named_compile_time_arg_val("gate_mm_k_num_tiles"),
            };

            // Gather (compute — no-op)
            deepseek_b1_ops::MoeGather::ComputeArgs gather_args{};

            // Gate (compute)
            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ComputeCTArgs<
                get_named_compile_time_arg_val("gate_input_cb"),
                get_named_compile_time_arg_val("gate_bias_cb"),
                get_named_compile_time_arg_val("gate_input_indices_cb"),
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("gate_output_indices_cb"),
                get_named_compile_time_arg_val("gate_eps"),
                get_named_compile_time_arg_val("gate_scaling_factor"),
                get_named_compile_time_arg_val("gate_enable_sigmoid")>;

            // Index Mcast (compute — no-op)
            deepseek_b1_ops::Mcast::ComputeArgs index_mcast_args{};

            // Expert Scale Mcast (compute — no-op)
            deepseek_b1_ops::Mcast::ComputeArgs expert_scale_mcast_args{};
#endif  // ENABLE_ROUTING

            // gate_proj DRAM Matmul Expert Compressed (compute) — pr42896 28-param shape
            using GateProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
                get_named_compile_time_arg_val("gate_proj_cb_in0"),
                get_named_compile_time_arg_val("gate_proj_cb_in1"),
                get_named_compile_time_arg_val("gate_proj_cb_out"),
                get_named_compile_time_arg_val("gate_proj_cb_index"),
                get_named_compile_time_arg_val("gate_proj_num_tiles_k"),
                get_named_compile_time_arg_val("gate_proj_subblock_k"),
                get_named_compile_time_arg_val("gate_proj_subblock_n"),
                get_named_compile_time_arg_val("gate_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("gate_proj_per_core_n"),
                get_named_compile_time_arg_val("gate_proj_dram_fmt_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_num_active_experts"),
                get_named_compile_time_arg_val("gate_proj_index_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_fmt"),
                get_named_compile_time_arg_val("gate_proj_dram_meta_words_per_block"),
                get_named_compile_time_arg_val("gate_proj_in0_page_size"),
                get_named_compile_time_arg_val("gate_proj_fmt_cb_l1_addr"),
                get_named_compile_time_arg_val("gate_proj_fmt_cb_page_size"),
                get_named_compile_time_arg_val("gate_proj_fmt_sem_addr_0"),
                get_named_compile_time_arg_val("gate_proj_fmt_sem_addr_1"),
                get_named_compile_time_arg_val("gate_proj_accum_experts"),
                get_named_compile_time_arg_val("gate_proj_fuse_silu"),
                get_named_compile_time_arg_val("gate_proj_index_offset"),
                get_named_compile_time_arg_val("gate_proj_k_parallel_per_bank"),
                get_named_compile_time_arg_val("gate_proj_k_slice_idx"),
                get_named_compile_time_arg_val("gate_proj_num_subblocks_k_local"),
                get_named_compile_time_arg_val("gate_proj_partial_sem_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_out_silu"),
                get_named_compile_time_arg_val("gate_proj_silu_tile_h"),
                get_named_compile_time_arg_val("gate_proj_cores_per_bank"),
                get_named_compile_time_arg_val("gate_proj_core_in_bank_idx"),
                get_named_compile_time_arg_val("gate_proj_primary_at_last_offset"),
                get_named_compile_time_arg_val("gate_proj_gather_sync_sem_addr"),
                get_named_compile_time_arg_val("gate_proj_cb_internal_acc"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("gate_proj_num_dram_experts_pre_selected")>;

            // up_proj DRAM Matmul Expert Compressed (compute) — shares weight CB with gate_proj
            using UpProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
                get_named_compile_time_arg_val("up_proj_cb_in0"),
                get_named_compile_time_arg_val("up_proj_cb_in1"),
                get_named_compile_time_arg_val("up_proj_cb_mm_out"),
                get_named_compile_time_arg_val("up_proj_cb_index"),
                get_named_compile_time_arg_val("up_proj_num_tiles_k"),
                get_named_compile_time_arg_val("up_proj_subblock_k"),
                get_named_compile_time_arg_val("up_proj_subblock_n"),
                get_named_compile_time_arg_val("up_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("up_proj_per_core_n"),
                get_named_compile_time_arg_val("up_proj_dram_fmt_l1_addr"),
                get_named_compile_time_arg_val("up_proj_num_active_experts"),
                get_named_compile_time_arg_val("up_proj_index_l1_addr"),
                get_named_compile_time_arg_val("up_proj_cb_fmt"),
                get_named_compile_time_arg_val("up_proj_dram_meta_words_per_block"),
                get_named_compile_time_arg_val("up_proj_in0_page_size"),
                get_named_compile_time_arg_val("up_proj_fmt_cb_l1_addr"),
                get_named_compile_time_arg_val("up_proj_fmt_cb_page_size"),
                get_named_compile_time_arg_val("up_proj_fmt_sem_addr_0"),
                get_named_compile_time_arg_val("up_proj_fmt_sem_addr_1"),
                get_named_compile_time_arg_val("up_proj_accum_experts"),
                get_named_compile_time_arg_val("up_proj_fuse_silu"),
                get_named_compile_time_arg_val("up_proj_index_offset"),
                get_named_compile_time_arg_val("up_proj_k_parallel_per_bank"),
                get_named_compile_time_arg_val("up_proj_k_slice_idx"),
                get_named_compile_time_arg_val("up_proj_num_subblocks_k_local"),
                get_named_compile_time_arg_val("up_proj_partial_sem_addr"),
                get_named_compile_time_arg_val("up_proj_cb_out_silu"),
                get_named_compile_time_arg_val("up_proj_silu_tile_h"),
                get_named_compile_time_arg_val("up_proj_cores_per_bank"),
                get_named_compile_time_arg_val("up_proj_core_in_bank_idx"),
                get_named_compile_time_arg_val("up_proj_primary_at_last_offset"),
                get_named_compile_time_arg_val("up_proj_gather_sync_sem_addr"),
                get_named_compile_time_arg_val("up_proj_cb_internal_acc"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("up_proj_num_dram_experts_pre_selected")>;

            // Eltwise Mul (compute)
            using MulCTArgs = deepseek_b1_ops::EltwiseMul::ComputeCTArgs<
                get_named_compile_time_arg_val("mul_cb_in0"),
                get_named_compile_time_arg_val("mul_cb_in1"),
                get_named_compile_time_arg_val("mul_cb_out"),
                get_named_compile_time_arg_val("mul_num_tiles"),
                get_named_compile_time_arg_val("up_proj_cb_mm_out"),     // cb_in0_wait (actual producer)
                get_named_compile_time_arg_val("up_proj_per_core_n"),    // cb_in0_wait_tiles
                get_named_compile_time_arg_val("gate_proj_cb_out"),      // cb_in1_wait (actual producer)
                get_named_compile_time_arg_val("gate_proj_per_core_n"),  // cb_in1_wait_tiles
                get_named_compile_time_arg_val("mul_cb_scalar"),
                get_named_compile_time_arg_val("mul_fp32_dest_acc_en"),
                get_named_compile_time_arg_val("enable_routing"),  // enable_scalar
                get_named_compile_time_arg_val("mul_num_experts")>;

            // down_proj Gather (compute — no-op)
            deepseek_b1_ops::MoeGather::ComputeArgs down_proj_gather_args{};
            // SRAM gate/up gather (compute — no-op).
            deepseek_b1_ops::MoeGather::ComputeArgs sram_ag_args{};
            deepseek_b1_ops::MoeGather::ComputeArgs sram_bg_args{};

            // down_proj Mcast (compute — no-op)
            deepseek_b1_ops::Mcast::ComputeArgs down_proj_mcast_args{};
            deepseek_b1_ops::Mcast::ComputeArgs sram_down_mcast_args{};

            // down_proj DRAM Matmul Expert Compressed (compute) — pr42896 28-param shape
            using DownProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
                get_named_compile_time_arg_val("down_proj_cb_in0"),
                get_named_compile_time_arg_val("down_proj_cb_in1"),
                get_named_compile_time_arg_val("down_proj_cb_out"),
                get_named_compile_time_arg_val("down_proj_cb_index"),
                get_named_compile_time_arg_val("down_proj_num_tiles_k"),
                get_named_compile_time_arg_val("down_proj_subblock_k"),
                get_named_compile_time_arg_val("down_proj_subblock_n"),
                get_named_compile_time_arg_val("down_proj_num_subblocks_k"),
                get_named_compile_time_arg_val("down_proj_per_core_n"),
                get_named_compile_time_arg_val("down_proj_dram_fmt_l1_addr"),
                get_named_compile_time_arg_val("down_proj_num_active_experts"),
                get_named_compile_time_arg_val("down_proj_index_l1_addr"),
                get_named_compile_time_arg_val("down_proj_cb_fmt"),
                get_named_compile_time_arg_val("down_proj_dram_meta_words_per_block"),
                get_named_compile_time_arg_val("down_proj_in0_page_size"),
                get_named_compile_time_arg_val("down_proj_fmt_cb_l1_addr"),
                get_named_compile_time_arg_val("down_proj_fmt_cb_page_size"),
                get_named_compile_time_arg_val("down_proj_fmt_sem_addr_0"),
                get_named_compile_time_arg_val("down_proj_fmt_sem_addr_1"),
                get_named_compile_time_arg_val("down_proj_accum_experts"),
                get_named_compile_time_arg_val("down_proj_fuse_silu"),
                get_named_compile_time_arg_val("down_proj_index_offset"),
                get_named_compile_time_arg_val("down_proj_k_parallel_per_bank"),
                get_named_compile_time_arg_val("down_proj_k_slice_idx"),
                get_named_compile_time_arg_val("down_proj_num_subblocks_k_local"),
                get_named_compile_time_arg_val("down_proj_partial_sem_addr"),
                get_named_compile_time_arg_val("down_proj_cb_out_silu"),
                get_named_compile_time_arg_val("down_proj_silu_tile_h"),
                get_named_compile_time_arg_val("down_proj_cores_per_bank"),
                get_named_compile_time_arg_val("down_proj_core_in_bank_idx"),
                get_named_compile_time_arg_val("down_proj_primary_at_last_offset"),
                get_named_compile_time_arg_val("down_proj_gather_sync_sem_addr"),
                get_named_compile_time_arg_val("down_proj_cb_internal_acc"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("down_proj_num_dram_experts_pre_selected")>;

            // SRAM gate_proj Matmul Expert (compute, TRISC).
            // The trailing /*compact_in0=*/0 and enable_routing args map onto the
            // kernel's ComputeCTArgs trailing template params. enable_routing→
            // enable_indexing: 1 = read index_ptr (routed MoE); 0 = synthesize
            // raw_idx = EXPERT_SRAM_FLAG | exp_i (dense MLP, no mcast_index).
            using SramGateProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ComputeCTArgs<
                get_named_compile_time_arg_val("sram_gate_proj_cb_in0"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_in1"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_out"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_index"),
                get_named_compile_time_arg_val("sram_gate_proj_num_tiles_k"),
                get_named_compile_time_arg_val("sram_gate_proj_out_w"),
                get_named_compile_time_arg_val("sram_gate_proj_fmt_l1_addr"),
                get_named_compile_time_arg_val("sram_gate_proj_num_active_experts"),
                get_named_compile_time_arg_val("sram_gate_proj_index_l1_addr"),
                get_named_compile_time_arg_val("sram_gate_proj_base_addrs_l1_addr"),
                get_named_compile_time_arg_val("sram_gate_proj_meta_words_per_expert"),
                get_named_compile_time_arg_val("sram_gate_proj_in0_page_size"),
                get_named_compile_time_arg_val("sram_gate_proj_accum_experts"),
                get_named_compile_time_arg_val("sram_gate_proj_k_per_core"),
                get_named_compile_time_arg_val("sram_gate_proj_k_offset"),
                get_named_compile_time_arg_val("sram_gate_proj_cb_out_sram"),
                /*compact_in0=*/0,
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("sram_use_compression")>;

            // SRAM up_proj Matmul Expert (compute, TRISC) — mirror of gate_proj.
            using SramUpProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ComputeCTArgs<
                get_named_compile_time_arg_val("sram_up_proj_cb_in0"),
                get_named_compile_time_arg_val("sram_up_proj_cb_in1"),
                get_named_compile_time_arg_val("sram_up_proj_cb_out"),
                get_named_compile_time_arg_val("sram_up_proj_cb_index"),
                get_named_compile_time_arg_val("sram_up_proj_num_tiles_k"),
                get_named_compile_time_arg_val("sram_up_proj_out_w"),
                get_named_compile_time_arg_val("sram_up_proj_fmt_l1_addr"),
                get_named_compile_time_arg_val("sram_up_proj_num_active_experts"),
                get_named_compile_time_arg_val("sram_up_proj_index_l1_addr"),
                get_named_compile_time_arg_val("sram_up_proj_base_addrs_l1_addr"),
                get_named_compile_time_arg_val("sram_up_proj_meta_words_per_expert"),
                get_named_compile_time_arg_val("sram_up_proj_in0_page_size"),
                get_named_compile_time_arg_val("sram_up_proj_accum_experts"),
                get_named_compile_time_arg_val("sram_up_proj_k_per_core"),
                get_named_compile_time_arg_val("sram_up_proj_k_offset"),
                get_named_compile_time_arg_val("sram_up_proj_cb_out_sram"),
                /*compact_in0=*/0,
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("sram_use_compression")>;

            // SRAM down_proj Matmul Expert (compute, TRISC) — accum_experts=1 +
            // compact_in0=1 path. Reads compact mcast dst (n_sram_active face
            // tiles back-to-back), iterates SRAM-flagged TopK winners, accumulates
            // into a single per-core out tile (out_w=2 N-tiles).
            using SramDownProjCTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ComputeCTArgs<
                get_named_compile_time_arg_val("sram_down_proj_cb_in0"),
                get_named_compile_time_arg_val("sram_down_proj_cb_in1"),
                get_named_compile_time_arg_val("sram_down_proj_cb_out"),
                get_named_compile_time_arg_val("sram_down_proj_cb_index"),
                get_named_compile_time_arg_val("sram_down_proj_num_tiles_k"),
                get_named_compile_time_arg_val("sram_down_proj_out_w"),
                get_named_compile_time_arg_val("sram_down_proj_fmt_l1_addr"),
                get_named_compile_time_arg_val("sram_down_proj_num_active_experts"),
                get_named_compile_time_arg_val("sram_down_proj_index_l1_addr"),
                get_named_compile_time_arg_val("sram_down_proj_base_addrs_l1_addr"),
                get_named_compile_time_arg_val("sram_down_proj_meta_words_per_expert"),
                get_named_compile_time_arg_val("sram_down_proj_in0_page_size"),
                get_named_compile_time_arg_val("sram_down_proj_accum_experts"),
                get_named_compile_time_arg_val("sram_down_proj_k_per_core"),
                get_named_compile_time_arg_val("sram_down_proj_k_offset"),
                get_named_compile_time_arg_val("sram_down_proj_cb_out_sram"),
                get_named_compile_time_arg_val("sram_down_proj_compact_in0"),
                get_named_compile_time_arg_val("enable_routing"),  // → enable_indexing
                get_named_compile_time_arg_val("sram_use_compression")>;

            // SRAM extended GatedReduce (compute, sender_core only).
            // tiles_per_k = 8 K-partial faces per output face. k_num_tiles is
            // a runtime field, set to n_sram_active before the call.
            // enable_scalar tracks enable_routing:
            //   routing → 1: multiply by per-K scalar from scalar_cb (TopK score).
            //   dense   → 0: silu(g1) * g2 only (no scale; matches dense MLP math).
            //              BRISC's scalar copy is also CT-gated off via op.py
            //              setting sram_gr_scalar_cb=0 in dense mode.
            using SramGatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ComputeCTArgs<
                get_named_compile_time_arg_val("sram_gated_reduce_tiles_per_k"),
                get_named_compile_time_arg_val("enable_routing")>;  // → enable_scalar
            deepseek_b1_ops::GatedReduce::ComputeArgs sram_gated_reduce_args{
                get_named_compile_time_arg_val("sram_gated_reduce_group1_cb"),
                get_named_compile_time_arg_val("sram_gated_reduce_group2_cb"),
                get_named_compile_time_arg_val("sram_gated_reduce_intermed_cb"),
                get_named_compile_time_arg_val("sram_gated_reduce_out_cb"),
                get_named_compile_time_arg_val("sram_gated_reduce_scalar_cb"),
                /*k_num_tiles=*/0,  // runtime: set to n_sram_active before call
                /*out_cb_total_pushes=*/get_named_compile_time_arg_val("sram_gather_num_active_experts"),
            };

            // SRAM down merge (compute): eltwise_add when n_sram>0 else copy(in1→out).
            // do_add is set at runtime from n_sram_active.
            using SramDownMergeCTArgs = deepseek_b1_ops::EltwiseAddOrCopy::ComputeCTArgs<
                get_named_compile_time_arg_val("sram_down_merge_in0"),
                get_named_compile_time_arg_val("sram_down_merge_in1"),
                get_named_compile_time_arg_val("sram_down_merge_out"),
                get_named_compile_time_arg_val("sram_down_merge_num_tiles")>;
            deepseek_b1_ops::EltwiseAddOrCopy::ComputeArgs sram_down_merge_args{/*do_add=*/0};

            // Eltwise Add (compute) — switched to EltwiseAddOrCopy so the final
            // DRAM add can fall back to copy(shared_output) when all TopK
            // winners are SRAM-flagged (n_dram_active==0). The aliased
            // cb_in0_wait, asymmetric wait counts, and per-core cb_in1 slice
            // offset map directly onto EltwiseAddOrCopy's optional CT args.
            using AddCTArgs = deepseek_b1_ops::EltwiseAddOrCopy::ComputeCTArgs<
                get_named_compile_time_arg_val("add_cb_in0"),
                get_named_compile_time_arg_val("add_cb_in1"),
                get_named_compile_time_arg_val("add_cb_out"),
                get_named_compile_time_arg_val("add_num_tiles"),
                get_named_compile_time_arg_val("down_proj_cb_out"),  // cb_in0_wait (actual producer)
                // Full per-primary N after gather: per_core_n × cores_per_bank.
                // gather=False keeps cores_per_bank=1 → equals per_core_n (gate/up unchanged).
                get_named_compile_time_arg_val("down_proj_per_core_n") *
                    get_named_compile_time_arg_val("down_proj_cores_per_bank"),  // cb_in0_wait_tiles
                get_named_compile_time_arg_val("add_cb_in1_wait_tiles"),
                /*HasSliceOffset=*/true,
                get_named_compile_time_arg_val("add_sender_index"),
                get_named_compile_time_arg_val("add_slice_size_bytes")>;
            // do_add is patched at runtime from n_dram_active in the body.
            deepseek_b1_ops::EltwiseAddOrCopy::ComputeArgs add_args{/*do_add=*/0};

            // Residual Mcast (compute — no-op)
            using ResidualMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs residual_mcast_args{};

            // RMSNorm (compute — sender core only)
            // Input: residual_mcast_src_cb (raw activation), Output: rmsnorm_output_cb
            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
                get_named_compile_time_arg_val("moe_rmsnorm_fp32_acc") == 1,
                get_named_compile_time_arg_val("moe_rmsnorm_num_tiles"),
                get_named_compile_time_arg_val("moe_rmsnorm_rsqrt_fast_approx") == 1,
                get_named_compile_time_arg_val("moe_rmsnorm_input_cb"),  // residual_mcast_src_cb
                get_named_compile_time_arg_val("moe_rmsnorm_gamma_cb"),
                get_named_compile_time_arg_val("moe_rmsnorm_output_cb")>;  // rmsnorm_output_cb
            deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
                get_common_arg_val<uint32_t>(
                    get_named_compile_time_arg_val("moe_rmsnorm_trisc_common_rt_arg_base") + 0),
                get_common_arg_val<float>(get_named_compile_time_arg_val("moe_rmsnorm_trisc_common_rt_arg_base") + 1),
            };

#ifdef ENABLE_REDUCE_TO_ONE
            // ReduceToOneB1 (compute — performs reduction)
            using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ComputeCTArgs<
                get_named_compile_time_arg_val("reduce_device_role"),
                get_named_compile_time_arg_val("reduce_num_tiles"),
                get_named_compile_time_arg_val("reduce_local_cb"),
                get_named_compile_time_arg_val("reduce_received_cb"),
                get_named_compile_time_arg_val("reduce_output_cb"),
                get_named_compile_time_arg_val("reduce_scratch_cb"),
                get_named_compile_time_arg_val("is_reduce_fabric_core")>;

            // Compute has no runtime args
            deepseek_b1_ops::ReduceToOneB1::ComputeArgs reduce_rt_args{};
#endif
        } routed;

        struct Shared {
            // KN-sliced matmul (compute)
            using GUMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ComputeCTArgs<>;
            deepseek_b1_ops::KNSlicedMatmul::ComputeArgs gu_matmul_args{
                get_named_compile_time_arg_val("shared_gu_act_cb"),
                get_named_compile_time_arg_val("shared_gu_weights_cb"),
                get_named_compile_time_arg_val("shared_gu_out_cb"),
                get_named_compile_time_arg_val("shared_gu_k_offset"),
                get_named_compile_time_arg_val("shared_gu_k_per_core"),
                get_named_compile_time_arg_val("shared_gu_act_total_tiles"),
                get_named_compile_time_arg_val("shared_gu_weights_cb_addr"),
            };

            // Gather (compute — no-op for TRISC)
            deepseek_b1_ops::MoeGather::ComputeArgs ag_args{};
            deepseek_b1_ops::MoeGather::ComputeArgs bg_args{};

            // Gated Reduce (compute)
            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ComputeCTArgs<get_named_compile_time_arg_val(
                "shared_gated_reduce_tiles_per_k")>;
            deepseek_b1_ops::GatedReduce::ComputeArgs gated_reduce_args{
                get_named_compile_time_arg_val("shared_gated_reduce_group1_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_group2_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_intermed_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_mcast_src_cb"),
                /*scalar_cb=*/0,  // shared path: no per-expert scale
                get_named_compile_time_arg_val("shared_gated_reduce_k_num_tiles"),
            };

            // Down Mcast — compute no-op
            using DownMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs down_mcast_args{};

            // Down Proj Matmul (compute)
            using DownMatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<get_named_compile_time_arg_val(
                "shared_down_matmul_out_w_per_core")>;
            deepseek_b1_ops::Matmul::ComputeArgs down_matmul_args{
                get_named_compile_time_arg_val("shared_down_matmul_in0"),
                get_named_compile_time_arg_val("shared_down_matmul_in1"),
                get_named_compile_time_arg_val("shared_down_matmul_out"),
                get_named_compile_time_arg_val("shared_down_matmul_k_num_tiles"),
                get_named_compile_time_arg_val("shared_down_matmul_weights_cb_addr"),
            };

            // Residual Add (compute)
            using ResidualAddCTArgs = deepseek_b1_ops::ResidualAdd::ComputeCTArgs<get_named_compile_time_arg_val(
                "shared_residual_add_out_w")>;
            deepseek_b1_ops::ResidualAdd::ComputeArgs residual_add_args{
                get_named_compile_time_arg_val("shared_residual_add_in0"),
                get_named_compile_time_arg_val("shared_residual_add_in1"),
                get_named_compile_time_arg_val("shared_residual_add_out"),
                get_named_compile_time_arg_val("shared_residual_add_total_in1_tiles"),
                get_named_compile_time_arg_val("shared_residual_add_core_idx"),
            };

            // Output Gather (compute — no-op)
            deepseek_b1_ops::MoeGather::ComputeArgs og_args{};

            // Output Mcast (compute — no-op)
            using OutputMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs output_mcast_args{};
        } shared;
    } moe;

    deepseek_compute_kernel_init();
#endif

    // ============================================================================
    // Operation calls — Moe::Routed::* for types, moe.routed.* for args
    // ============================================================================

    // Init persistent mcast objects before the loop
    deepseek_b1_ops::Mcast::Op<
        Moe::Routed::ResidualMcastCTArgs,
        Core::is_sender_core,
        Core::is_mcast_grid_core,
        Core::Shared::is_mcast_receiver_core,
        false>  // pop_src=false: keep input for RMSNorm
        residual_mcast;
    residual_mcast.init(moe.routed.residual_mcast_args);

    // Mcast object reused for step 1 (RMSNorm mcast) and step 5 (index mcast)
    deepseek_b1_ops::Mcast::Op<
        Moe::Routed::McastCTArgs,
        Core::is_sender_core,
        Core::is_mcast_grid_core,
        Core::is_input_mcast_receiver,
        true>
        mcast;

    constexpr uint32_t num_iterations = get_named_compile_time_arg_val("num_iterations");
    constexpr uint32_t persistent_mode = get_named_compile_time_arg_val("persistent_mode");
    constexpr uint32_t persistent_next_iter_sem_addr = get_named_compile_time_arg_val("persistent_next_iter_sem_addr");

    uint32_t iteration = 0;

    auto moe_body = [&]() {
#if defined(COMPILE_FOR_BRISC) && defined(ENABLE_BCAST)
        if constexpr (persistent_mode != 0) {
            constexpr bool is_bcast_sender = get_named_compile_time_arg_val("bcast_is_sender") == 1;
            if constexpr (is_bcast_sender && Core::is_sender_core) {
                auto next_iter_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(persistent_next_iter_sem_addr);
                noc_semaphore_wait(next_iter_sem, 1);
                noc_semaphore_set(next_iter_sem, 0);
            }
        }
#endif

#if defined(RECONFIG_MOE_CBS)
        {
            constexpr uint32_t cb_config_l1_addr = get_named_compile_time_arg_val("reconfig_cb_config_l1_addr");
            uint32_t tt_l1_ptr* cb_config = reinterpret_cast<uint32_t tt_l1_ptr*>(cb_config_l1_addr);
            unified_kernels::reconfig_cb_interfaces(cb_config);
        }
#if defined(COMPILE_FOR_NCRISC)
        setup_all_sharded_buffers();
#endif
#endif  // RECONFIG_MOE_CBS

#ifdef ENABLE_BCAST
        // Step -1: CCL Broadcast — receive data from fabric into intermediate tensor
        {
            DeviceZoneScopedN("BCAST");
#if defined(COMPILE_FOR_NCRISC)
            using BcastCTArgs = deepseek_b1_ops::Broadcast::WriterCTArgs<
                get_named_compile_time_arg_val("bcast_pkt_cb"),
                get_named_compile_time_arg_val("bcast_num_pages_to_read"),
                get_named_compile_time_arg_val("bcast_tensor0_page_size"),
                get_named_compile_time_arg_val("bcast_num_neighbors"),
                get_named_compile_time_arg_val("bcast_num_links"),
                get_named_compile_time_arg_val("bcast_is_sender"),
                get_named_compile_time_arg_val("bcast_chunk_size_bytes"),
                get_named_compile_time_arg_val("bcast_last_chunk_size_bytes"),
                get_named_compile_time_arg_val("bcast_num_chunks")>;
            constexpr uint32_t bcast_ncrisc_base = get_named_compile_time_arg_val("bcast_ncrisc_common_rt_arg_base");
            uint32_t bcast_rta_offset = 0;
            uint32_t bcast_rta_num_args = 0;
            if constexpr (BcastCTArgs::num_neighbors > 0) {
                bcast_rta_num_args = get_arg_val<uint32_t>(0);
                bcast_rta_offset = 1;
            }
            deepseek_b1_ops::Broadcast::WriterArgs bcast_args{
                get_common_arg_val<uint32_t>(bcast_ncrisc_base + 0),
                get_common_arg_val<uint32_t>(bcast_ncrisc_base + 1),
                get_common_arg_val<uint32_t>(bcast_ncrisc_base + 2),
                {get_common_arg_val<uint32_t>(bcast_ncrisc_base + 3),
                 get_common_arg_val<uint32_t>(bcast_ncrisc_base + 4)},
                bcast_rta_offset,
                bcast_rta_num_args,
            };

            deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_sender_core> bcast_op;
            bcast_op(bcast_args);
#elif defined(COMPILE_FOR_BRISC)
            using BcastCTArgs = deepseek_b1_ops::Broadcast::ReaderCTArgs<
                get_named_compile_time_arg_val("bcast_pkt_cb"),
                get_named_compile_time_arg_val("bcast_num_pages_to_read"),
                get_named_compile_time_arg_val("bcast_is_sender"),
                get_named_compile_time_arg_val("bcast_use_socket")>;
            deepseek_b1_ops::Broadcast::ReaderArgs bcast_args{
                get_common_arg_val<uint32_t>(0),  // socket_config_addr
                get_common_arg_val<uint32_t>(1),  // socket_page_size
                get_common_arg_val<uint32_t>(2),  // socket_num_pages
            };
            deepseek_b1_ops::Broadcast::Op<BcastCTArgs, Core::is_sender_core> bcast_op;
            bcast_op(bcast_args);
#else
            // TRISC: broadcast is a no-op
            deepseek_b1_ops::Broadcast::Op<deepseek_b1_ops::Broadcast::ComputeCTArgs, Core::is_sender_core> bcast_op;
            deepseek_b1_ops::Broadcast::ComputeArgs bcast_args{};
            bcast_op(bcast_args);
#endif
        }
        // After broadcast, push CB 25 pages so residual mcast + RMSNorm can read
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_sender_core) {
            constexpr uint32_t bcast_residual_cb = get_named_compile_time_arg_val("shared_residual_mcast_src_cb");
            constexpr uint32_t bcast_residual_pages =
                get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages");
            unified_kernels::setup_sharded_buffer(bcast_residual_cb, bcast_residual_pages);
        }
#endif
#endif

        // 0. Residual Mcast: Broadcast input as residual to mcast receiver cores (pop_src=false)
        {
            DeviceZoneScopedN("RESIDUAL_MCAST");
            residual_mcast(moe.routed.residual_mcast_args);
        }

        // 0b. RMSNorm: normalize input on sender core (residual_mcast_src → rmsnorm_output)
        {
            DeviceZoneScopedN("RMSNORM");
            deepseek_b1_ops::RMSNorm::Op<
                Moe::Routed::RMSNormCTArgs,
                Core::is_sender_core,
                false>  // pop_input=false: tensor-backed CB, keep for next iteration
                rmsnorm;
            rmsnorm(moe.routed.rmsnorm_args);
        }

        // 1. RMSNorm Mcast: Broadcast normalized input from sender core to all receiver cores
        {
            DeviceZoneScopedN("MCAST");
            mcast(moe.routed.mcast_args);
        }

#ifdef ENABLE_BCAST
        // Pop CB 25 after consumers (residual mcast + RMSNorm) are done,
        // so next iteration's setup_sharded_buffer can push new data
#if defined(COMPILE_FOR_NCRISC)
        if constexpr (Core::is_sender_core) {
            constexpr uint32_t bcast_residual_cb = get_named_compile_time_arg_val("shared_residual_mcast_src_cb");
            constexpr uint32_t bcast_residual_pages =
                get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages");
            cb_pop_front(bcast_residual_cb, bcast_residual_pages);
        }
#endif
#endif

        // n_sram_active / n_dram_active gate every op in the SRAM and DRAM
        // pipelines respectively. Read by sram_invoke_* / dram_invoke_*
        // helpers below, plus SRAM_DOWN_MERGE (do_add = n_sram_active > 0)
        // and the final ELTWISE_ADD (do_add = n_dram_active > 0).
        //
        //   Routing mode:
        //     - Both default to 0 — non-scan cores (e.g., mcast_worker_grid
        //       extras that don't receive mcast_index) skip uniformly with
        //       scan-derived skips.
        //     - Scan cores overwrite below: n_sram_active from scan_n_sram_active,
        //       n_dram_active = num_active_experts - n_sram_active. Routing
        //       runs at most one TopK per iter so n_sram + n_dram = num_active.
        //
        //   Dense MLP (no routing): each leg's count comes from its own CT arg,
        //   so the two pipelines are independent — a leg runs iff its CT count
        //   is non-zero. The final ELTWISE_ADD uses do_add = (n_dram>0) so it
        //   adds when DRAM contributed, or copies shared_output through when
        //   it didn't. Configurations:
        //     - dram-only (today's test_mlp):      n_sram=0,    n_dram=N_dram
        //     - sram-only:                         n_sram=N_sram, n_dram=0
        //     - half-half (both contribute):       n_sram=N_sram, n_dram=N_dram
#ifdef ENABLE_ROUTING
        uint32_t n_sram_active = 0;
        uint32_t n_dram_active = 0;
#else
        // Dense MLP: DRAM list stays full (=gate_proj_num_active_experts, typically 8)
        // for matmul sizing/CB-descriptor probes, but SRAM-flagged slots are skipped
        // via raw_idx synthesis (num_dram_experts_pre_selected cutoff). So the runtime
        // DRAM count for dram_invoke_* helpers / ELTWISE_ADD do_add is the cutoff =
        // gate_proj_num_active_experts - sram_num_active_experts. When all chunks
        // are placed in SRAM, n_dram_active=0 → dram_invoke_* early-returns and
        // EltwiseAddOrCopy copies shared_output through (no cb_in0 wait).
        uint32_t n_sram_active = get_named_compile_time_arg_val("sram_gather_num_active_experts");
        uint32_t n_dram_active = get_named_compile_time_arg_val("gate_proj_num_active_experts") - n_sram_active;
#endif

#ifdef ENABLE_ROUTING
        // 2. Matmul + Activation: Routing matmul on gate_mm cores
        {
            DeviceZoneScopedN("MATMUL");
            deepseek_b1_ops::Matmul::Op<Moe::Routed::GateMMCTArgs, Core::Routed::is_gate_mm_core, false, false> gate_mm;
            gate_mm(moe.routed.gate_mm_args);
        }
#endif  // ENABLE_ROUTING

        // 3. Shared Expert: Gate/Up KN-sliced matmul on 128 compute cores
        //     CB 1 (act) is shared: on gate_proj_streamer cores (16 in K-split: primaries +
        //     K-senders) it is also consumed by gate_proj (step 6) and up_proj (step 7,
        //     which pops it on all streamers). So we only pop here on non-streamer cores.
        //     Using is_gate_proj_streamer_core (not is_gate_proj_core) — otherwise K-senders
        //     pop cb_in0 here, leaving gate_proj_mm's cb_wait_front waiting on empty CB.
        {
            DeviceZoneScopedN("SHARED_GU_MATMUL");
            deepseek_b1_ops::KNSlicedMatmul::Op<
                Moe::Shared::GUMatmulCTArgs,
                Core::Shared::is_compute_core,
                // pop_act: skip on shared gate AND up compute cores so SRAM
                // gate_proj/up_proj (step 5c) can read cb_in0 afterwards.
                !Core::Routed::is_gate_proj_streamer_core && !Core::Shared::is_gate_compute_core &&
                    !Core::Shared::is_up_compute_core,
                false>  // pop_weights
                shared_gu_matmul;
            shared_gu_matmul(moe.shared.gu_matmul_args);
        }

#ifdef ENABLE_ROUTING
        // 3c. Gather: Collect matmul outputs from compute cores to sender core
        {
            DeviceZoneScopedN("GATHER");
            deepseek_b1_ops::MoeGather::Op<Core::Routed::is_gate_mm_core, Core::is_sender_core, true> gather;
            gather(moe.routed.gather_args);
        }

        // 4. Gate: Top-K expert selection (on sender core only)
        {
            DeviceZoneScopedN("GATE");
            deepseek_b1_ops::DeepseekMoeGate::Op<Moe::Routed::GateCTArgs, Core::is_sender_core> gate;
            gate();
        }

        // 5. Mcast Index: Broadcast expert indices to all matmul-streamer cores.
        // IsReceiverCore=is_down_proj_streamer_core (16 cores in down gather mode, 8 otherwise).
        // Invariant: down gather and gate K-split are configured together (both use the same
        // primary_worker_cores + cores_per_dram_bank=2), so is_down_proj_streamer_core's
        // 16-core set IS the same set as is_gate_proj_streamer_core. Both gate/up matmul
        // (16 cores in K-split) and down (16 cores in gather) need the index for weight reads.
        // ─── Pre-mcast scan on sender_core ───────────────────────────────────
        // sender_core has gate_output_indices_cb populated by the gate kernel
        // before mcast_index pops it. Receivers don't have it yet — they
        // scan after the mcast (below). n_sram_active is declared at outer scope
        // (above the ENABLE_ROUTING ifdef) so SRAM_DOWN_MERGE can read it.
        if constexpr (Core::Shared::is_gated_reduce_core) {
            n_sram_active = scan_n_sram_active<
                get_named_compile_time_arg_val("gate_output_indices_cb"),
                get_named_compile_time_arg_val("sram_gather_sender_index_l1_addr"),
                get_named_compile_time_arg_val("sram_gather_num_active_experts"),
                get_named_compile_time_arg_val("scan_sync_sem_addr")>();
        }

        {
            DeviceZoneScopedN("MCAST_INDEX");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Routed::is_down_proj_streamer_core || Core::Shared::is_gate_compute_core ||
                    Core::Shared::is_up_compute_core || Core::Shared::is_mcast_receiver_core,
                true,
                /*ReceiverOnBrisc=*/true>
                index_mcast;
            index_mcast(moe.routed.index_mcast_args);
        }

        // ─── Post-mcast scan on receivers ────────────────────────────────────
        // a/b cores have gate_proj_cb_index populated by mcast_index — same
        // L1 addr SRAM matmul reads. Also scan on shared mcast receivers
        // (the 112 cores) so they know n_sram_active to wait/push the right
        // number of SRAM down mcast pages.
        // Extended to include is_gate_proj_streamer_core / is_down_proj_streamer_core:
        // the DRAM gate/up/down matmul cores derive n_dram_active = num_active -
        // n_sram_active to gate the DRAM chain (skip when all TopK winners are
        // SRAM-flagged). They already receive the index_mcast (line 1671 receiver
        // list), so cb_index / sram_gather_index_l1_addr are populated. In K-split
        // mode the two streamer sets coincide; in non-K-split they may differ
        // slightly so cover both.
        if constexpr (
            Core::Shared::is_gate_compute_core || Core::Shared::is_up_compute_core ||
            Core::Shared::is_mcast_receiver_core || Core::Routed::is_gate_proj_streamer_core ||
            Core::Routed::is_down_proj_streamer_core) {
            n_sram_active = scan_n_sram_active<
                get_named_compile_time_arg_val("sram_gather_cb_index"),
                get_named_compile_time_arg_val("sram_gather_index_l1_addr"),
                get_named_compile_time_arg_val("sram_gather_num_active_experts"),
                get_named_compile_time_arg_val("scan_sync_sem_addr")>();
        }

        // Derive DRAM-active count on cores that scanned (and therefore have a
        // valid n_sram_active). Sender_core scanned pre-mcast; streamers / a / b
        // / 112 scanned post-mcast. All other cores keep the routing-mode
        // default of 0 so they skip uniformly with the scan-derived skips.
        if constexpr (
            Core::Shared::is_gated_reduce_core ||  // sender_core (pre-mcast scan)
            Core::Shared::is_gate_compute_core || Core::Shared::is_up_compute_core ||
            Core::Shared::is_mcast_receiver_core || Core::Routed::is_gate_proj_streamer_core ||
            Core::Routed::is_down_proj_streamer_core) {
            // Base = total TopK / chunk count. Was previously sram_gather_num_active_experts,
            // but op.py emits that as 0 when SRAM isn't placed (it's tied to the SRAM gate
            // setup, not the TopK count). For routing-without-SRAM, n_sram_active=0 and
            // base=0 would give n_dram_active=0 → entire DRAM chain skips via
            // dram_invoke_* → output drops the routed contribution → PCC tanks.
            constexpr uint32_t num_active = get_named_compile_time_arg_val("gate_proj_num_active_experts");
            n_dram_active = num_active - n_sram_active;
        }

        // 5b. Mcast Expert Scale: Broadcast expert scale to gate_proj cores.
        //     Skipped when n_dram_active==0 because MUL (its only consumer) is
        //     skipped too. Unpopped gate_output_cb pages on sender are
        //     re-anchored by RECONFIG_MOE_CBS at next iter's top.
        {
            DeviceZoneScopedN("MCAST_EXPERT_SCALE");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Routed::is_gate_proj_core,
                true,
                /*ReceiverOnBrisc=*/true>
                expert_scale_mcast;
            deepseek_b1_ops::dram_invoke_mcast(expert_scale_mcast, moe.routed.expert_scale_mcast_args, n_dram_active);
        }

#endif  // ENABLE_ROUTING

        // 5b.1. Shared Expert: Gate Gather (A) + Up Gather (B) — paired and placed
        //       AFTER MCAST_INDEX + MCAST_EXPERT_SCALE so that BRISC on shared a/b cores
        //       receives the mcasts (populating cb_index) BEFORE doing the gather sends.
        //       That way TRISC's SRAM_GATE_PROJ / SRAM_UP_PROJ can start as soon as
        //       SHARED_GU_MATMUL output is ready — no waiting for BRISC's gather send
        //       to finish first. Both gathers fire back-to-back so SHARED_GATED_REDUCE
        //       (TRISC on sender_core) starts as early as possible → SHARED_DOWN_MCAST
        //       BRISC fires earlier → 112-core SHARED_DOWN_MATMUL TRISC starts earlier.
        //       MoeGather sender=BRISC; receiver=NCRISC on sender_core (= disjoint from
        //       sender_core BRISC's mcast queue).
        {
            DeviceZoneScopedN("SHARED_GATE_GATHER");
            deepseek_b1_ops::MoeGather::Op<
                Core::Shared::is_gate_compute_core,
                Core::Shared::is_gated_reduce_core,
                true,  // pop_src
                true>  // UsePerCoreSenderIdx
                shared_gate_gather;
            shared_gate_gather(moe.shared.ag_args);
        }

        {
            DeviceZoneScopedN("SHARED_UP_GATHER");
            deepseek_b1_ops::MoeGather::Op<
                Core::Shared::is_up_compute_core,
                Core::Shared::is_gated_reduce_core,
                true,  // pop_src
                true>  // UsePerCoreSenderIdx
                shared_up_gather;
            shared_up_gather(moe.shared.bg_args);
        }

        // SHARED Gated Reduce — moved here (right after SHARED_UP_GATHER) so sender_core
        // TRISC fires SHARED GR as soon as both SHARED gathers complete, freeing TRISC
        // to do other work later. Output (shared_mcast_src_cb) feeds SHARED_DOWN_MCAST
        // below — having SHARED_GR done early lets SHARED_DOWN_MCAST fire as the first
        // down-side mcast on sender BRISC.
        {
            DeviceZoneScopedN("SHARED_GATED_REDUCE");
            deepseek_b1_ops::GatedReduce::Op<Moe::Shared::GatedReduceCTArgs, Core::Shared::is_gated_reduce_core>
                shared_gated_reduce;
            shared_gated_reduce(moe.shared.gated_reduce_args);
        }

        // 5c. SRAM Routed Expert pipeline (matmul → gather → GatedReduce → mcast → matmul).
        //     Lifted OUTSIDE #ifdef ENABLE_ROUTING so the dense MLP path can run
        //     the same SRAM chain (single expert per device, enable_indexing=0
        //     on the SRAM matmul kernel — see matmul_expert_compressed_sram.hpp).
        //     Every helper early-returns when n_sram_active == 0:
        //       - Routing mode: sender pre-mcast scan + a/b/112/streamers post-mcast
        //         scan set n_sram_active; non-scan cores keep 0 (uniform skip).
        //       - Dense MLP (today): n_sram_active stays 0 → all helpers skip.
        //       - Dense+SRAM (future): op.py sets n_sram_active=num_active_experts
        //         and n_dram_active=0 (DRAM chain skipped via the dram_invoke_* path).
        //     All Moe::Routed::Sram* / McastCTArgs types and moe.routed.sram_*_args
        //     runtime args are defined unconditionally for both routing and dense
        //     compilations, so no struct-level changes are required.
        //
        // SRAM gate_proj on 64 shared gate compute cores. Iterates 8 TopK
        // entries, processes SRAM-flagged, skips DRAM-flagged.
        // pop_in0: TRUE on non-streamer gate_compute cores (no DRAM up_proj
        //   to drain cb_in0; without pop here, rmsnorm-mcast cb fills
        //   across iters → multi-iter hang).
        //   FALSE on streamer cores — DRAM up_proj pops cb_in0 there;
        //   double-popping would underflow.
        // pop_out=false: drained later by SRAM gate gather sender.
        constexpr bool sram_gp_pop_in0 =
            Core::Shared::is_gate_compute_core && !Core::Routed::is_gate_proj_streamer_core;
        {
            DeviceZoneScopedN("SRAM_GATE_PROJ");
            deepseek_b1_ops::MatmulExpertCompressedSRAM::Op<
                Moe::Routed::SramGateProjCTArgs,
                Core::Shared::is_gate_compute_core,
                /*pop_in0=*/sram_gp_pop_in0,
                /*pop_in1=*/false,
                /*pop_index=*/false,
                /*pop_out=*/false>
                sram_gate_proj;
            deepseek_b1_ops::sram_invoke_matmul(sram_gate_proj, n_sram_active);
        }

        // SRAM up_proj on 64 shared up compute cores (mirror of gate_proj).
        // pop_in0: TRUE on non-streamer up_compute cores (drain cb_in0 here
        //   since neither shared_gu_matmul nor SRAM up_proj otherwise pops).
        //   FALSE on streamer cores — DRAM up_proj pops there.
        constexpr bool sram_up_pop_in0 = Core::Shared::is_up_compute_core && !Core::Routed::is_gate_proj_streamer_core;
        {
            DeviceZoneScopedN("SRAM_UP_PROJ");
            deepseek_b1_ops::MatmulExpertCompressedSRAM::Op<
                Moe::Routed::SramUpProjCTArgs,
                Core::Shared::is_up_compute_core,
                /*pop_in0=*/sram_up_pop_in0,
                /*pop_in1=*/false,
                /*pop_index=*/false,
                /*pop_out=*/false>
                sram_up_proj;
            deepseek_b1_ops::sram_invoke_matmul(sram_up_proj, n_sram_active);
        }

        // 6/7/8. DRAM gate_proj / up_proj / mul — moved up here from after
        //        SHARED_GATED_REDUCE so NCRISC on streamer cores can prefetch
        //        DRAM weights into cb_in1 while TRISC is still busy with the
        //        SRAM down chain (which doesn't touch streamer cores). With
        //        SRAM matmul ahead of these blocks, streamer TRISC also gets
        //        its DRAM compute kicked off as soon as SRAM_UP_PROJ finishes,
        //        instead of waiting through the SRAM gather/GR/down phases.
        //        Skipped via dram_invoke_matmul when n_dram_active==0.
        {
            DeviceZoneScopedN("GATE_PROJ");
            constexpr uint32_t gp_cb_in1_addr = get_named_compile_time_arg_val("gate_proj_in1_buf_addr");
            // pop_out=true on secondary streamer cores only (= streamer && !primary).
            // Senders' cb_out is internal (their NCRISC NOC-wrote real bytes to the
            // primary; no downstream op reads sender's cb_out), so they must drain
            // for multi-iter. Primaries' cb_out is consumed by mul → don't pop.
            constexpr bool gp_secondary_pop =
                Core::Routed::is_gate_proj_streamer_core && !Core::Routed::is_gate_proj_core;
            deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<
                Moe::Routed::GateProjCTArgs,
                Core::Routed::is_gate_proj_streamer_core,
                /*pop_in0=*/false,
                /*pop_index=*/false,
                /*ResetCBIn1=*/true,
                gp_cb_in1_addr,
                /*pop_out=*/gp_secondary_pop>
                gate_proj_mm;
            deepseek_b1_ops::dram_invoke_matmul(gate_proj_mm, n_dram_active);
        }

        {
            DeviceZoneScopedN("UP_PROJ");
            constexpr uint32_t up_cb_in1_addr = get_named_compile_time_arg_val("gate_proj_in1_buf_addr");
            // pop_in0=true on streamers: this is the last cb_in0 consumer on those
            // cores (SRAM_GATE/UP_PROJ ran first with pop_in0=false on streamers).
            // pop_out: drain secondary streamer cores' cb_out.
            constexpr bool up_secondary_pop =
                Core::Routed::is_gate_proj_streamer_core && !Core::Routed::is_gate_proj_core;
            deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<
                Moe::Routed::UpProjCTArgs,
                Core::Routed::is_gate_proj_streamer_core,
                /*pop_in0=*/true,
                /*pop_index=*/false,
                /*ResetCBIn1=*/true,
                up_cb_in1_addr,
                /*pop_out=*/up_secondary_pop,
                /*SkipNocTridReset=*/true>
                up_proj;
            deepseek_b1_ops::dram_invoke_matmul(up_proj, n_dram_active);
        }

        {
            DeviceZoneScopedN("MUL");
            deepseek_b1_ops::EltwiseMul::Op<Moe::Routed::MulCTArgs, Core::Routed::is_gate_proj_core> mul_op;
            deepseek_b1_ops::dram_invoke_eltwise_mul(mul_op, n_dram_active);
        }

        // SRAM gathers + SRAM_GATED_REDUCE — placed before DRAM down gather/mcast
        // because SRAM output (from SRAM_GATE/UP_PROJ TRISC, ready before MUL completes)
        // arrives earlier than DRAM MUL output. Sender_core NCRISC processes the SRAM
        // gather receives immediately, then sender_core TRISC fires SRAM_GATED_REDUCE.
        //
        // SRAM_DOWN_MCAST and SRAM_DOWN_PROJ are deferred to AFTER DOWN_PROJ_GATHER +
        // DOWN_PROJ_MCAST so sender_core BRISC fires the DRAM down mcast first. That
        // gets DRAM DOWN_PROJ's cb_in0 onto the down_proj streamer cores as early as
        // possible, letting that 16-core TRISC chain (which is independent of the
        // 112-core SRAM/SHARED chain) start without waiting for SRAM_DOWN_MCAST's
        // 112-core mcast latency.
        // SRAM gather: BRISC's per-core write count = n_sram_active.
        // NCRISC dst_num_pages is set in the args initializer (compile-time
        // const = num_active × pages_per_expert) — receiver always advances
        // by full capacity to match GR's padded push count downstream.
        {
            DeviceZoneScopedN("SRAM_GATE_GATHER");
            deepseek_b1_ops::MoeGather::Op<
                Core::Shared::is_gate_compute_core,
                Core::Shared::is_gated_reduce_core,
                /*pop_src=*/true,
                /*UsePerCoreSenderIdx=*/true>
                sram_ag;
            deepseek_b1_ops::sram_invoke_moe_gather(sram_ag, moe.routed.sram_ag_args, n_sram_active);
        }
        {
            DeviceZoneScopedN("SRAM_UP_GATHER");
            deepseek_b1_ops::MoeGather::Op<
                Core::Shared::is_up_compute_core,
                Core::Shared::is_gated_reduce_core,
                /*pop_src=*/true,
                /*UsePerCoreSenderIdx=*/true>
                sram_bg;
            deepseek_b1_ops::sram_invoke_moe_gather(sram_bg, moe.routed.sram_bg_args, n_sram_active);
        }

        // SRAM extended GatedReduce on sender_core: silu(sum_K(gate)) * scale * sum_K(up)
        // per active SRAM expert. TRISC k_num_tiles = n_sram_active. BRISC
        // handles the per-expert scalar copy (independent of n_sram_active).
        {
            DeviceZoneScopedN("SRAM_GATED_REDUCE");
            deepseek_b1_ops::GatedReduce::Op<Moe::Routed::SramGatedReduceCTArgs, Core::Shared::is_gated_reduce_core>
                sram_gated_reduce;
            deepseek_b1_ops::sram_invoke_gated_reduce(
                sram_gated_reduce, moe.routed.sram_gated_reduce_args, n_sram_active);
        }

        // 10/11. DRAM down_proj Gather + Mcast — placed BEFORE SRAM_DOWN_MCAST/PROJ so
        //        sender_core BRISC fires DOWN_PROJ_MCAST before SRAM_DOWN_MCAST. That
        //        gets DRAM DOWN_PROJ's cb_in0 (the gathered+mcasted MUL output) onto
        //        the 16 down_proj streamer cores early, so the DRAM down matmul on
        //        those cores can start without waiting for SRAM_DOWN_MCAST's
        //        112-core mcast latency to drain on sender_core BRISC.
        //        Skipped via dram_invoke_* when n_dram_active==0.
        // DOWN_PROJ_GATHER: sender_core NCRISC gathers MUL output from gate_proj cores.
        // Fires right after MUL completes (gate_proj BRISC sends), independent of
        // sender BRISC's mcast queue.
        {
            DeviceZoneScopedN("DOWN_PROJ_GATHER");
            deepseek_b1_ops::MoeGather::Op<Core::Routed::is_gate_proj_core, Core::is_sender_core, true, true>
                down_proj_gather;
            deepseek_b1_ops::dram_invoke_moe_gather(down_proj_gather, moe.routed.down_proj_gather_args, n_dram_active);
        }

        // Down-chain mcasts on sender_core BRISC, ordered by data-readiness time:
        //   SHARED first (SHARED_GATED_REDUCE ran early, output ready earliest),
        //   SRAM next (SRAM_GATED_REDUCE ran above, output ready after SRAM gathers),
        //   DRAM (DOWN_PROJ_MCAST) last (waits for DOWN_PROJ_GATHER NCRISC which itself
        //   waits for MUL).
        // 112-core TRISC chain consumes them in the same order:
        //   SHARED_DOWN_MATMUL → SRAM_DOWN_PROJ → SRAM_DOWN_MERGE → SHARED_RESIDUAL_ADD.

        // 9. Shared: Down Mcast — broadcast SHARED_GATED_REDUCE output to 112 receivers.
        {
            DeviceZoneScopedN("SHARED_DOWN_MCAST");
            deepseek_b1_ops::Mcast::Op<
                Moe::Shared::DownMcastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Shared::is_mcast_receiver_core,
                true>
                shared_down_mcast;
            shared_down_mcast(moe.shared.down_mcast_args);
        }

        // SRAM down Mcast: SRAM_GATED_REDUCE output → 112 receivers.
        {
            DeviceZoneScopedN("SRAM_DOWN_MCAST");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Shared::is_mcast_receiver_core,
                /*pop_src=*/true,
                /*ReceiverOnBrisc=*/true>
                sram_down_mcast;
            deepseek_b1_ops::sram_invoke_down_mcast(sram_down_mcast, moe.routed.sram_down_mcast_args, n_sram_active);
        }

        // DRAM down Mcast: gathered MUL output → 16 down_proj streamer cores.
        {
            DeviceZoneScopedN("DOWN_PROJ_MCAST");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Routed::is_down_proj_streamer_core,
                true,
                /*ReceiverOnBrisc=*/true>
                down_proj_mcast;
            deepseek_b1_ops::dram_invoke_mcast(down_proj_mcast, moe.routed.down_proj_mcast_args, n_dram_active);
        }

        // 9b. Shared down matmul — 112-core TRISC, consumes SHARED_DOWN_MCAST data first.
        {
            DeviceZoneScopedN("SHARED_DOWN_MATMUL");
            deepseek_b1_ops::Matmul::Op<
                Moe::Shared::DownMatmulCTArgs,
                Core::Shared::is_mcast_receiver_core,
                /*pop_in0=*/true,
                /*pop_in1=*/false>
                shared_down_matmul;
            shared_down_matmul(moe.shared.down_matmul_args);
        }

        // SRAM down_proj on 112-core TRISC, consumes SRAM_DOWN_MCAST data after SHARED done.
        //   in0 = sram_down_mcast_dst_cb (n_sram_active face tiles compact).
        //   in1 = per-core SRAM weight slab (T expert (K=8 × N=2 bfp4 tiles)).
        //   out = sram_down_proj_out_cb (1×32 bf16, out_w=2 tiles per core).
        // accum_experts=1 + compact_in0=1: kernel sums across SRAM-flagged
        // TopK winners, indexing cb_in0 by sram_idx (compact) instead of
        // exp_i (expanded — would skip past nonexistent DRAM slots).
        // pop_in0=true: drain mcast dst CB so multi-iter starts clean.
        // pop_index=false: index reused later by DRAM down_proj.
        // pop_in1=false: cb_in1 is overlay-backed (per-core L1 slabs) — never popped.
        {
            DeviceZoneScopedN("SRAM_DOWN_PROJ");
            deepseek_b1_ops::MatmulExpertCompressedSRAM::Op<
                Moe::Routed::SramDownProjCTArgs,
                Core::Shared::is_mcast_receiver_core,
                /*pop_in0=*/true,
                /*pop_in1=*/false,
                /*pop_index=*/false,
                /*pop_out=*/false>
                sram_down_proj;
            deepseek_b1_ops::sram_invoke_matmul(sram_down_proj, n_sram_active);
        }

        // 9b'. SRAM_DOWN_MERGE — combine SRAM down output with shared down output.
        //   n_sram_active > 0 : merged = sram_down + shared_down (eltwise add)
        //   n_sram_active == 0: merged = shared_down (copy passthrough)
        // Runs on the 112 mcast receiver cores (same as shared_down_matmul). Output
        // CB merged_down_out_cb feeds residual_add (replaces shared_down_matmul_out
        // as the residual_add input). Always runs — copy path keeps the wiring
        // uniform for dense MLP / no-routing.
        {
            DeviceZoneScopedN("SRAM_DOWN_MERGE");
            deepseek_b1_ops::EltwiseAddOrCopy::
                Op<Moe::Routed::SramDownMergeCTArgs, Core::Shared::is_mcast_receiver_core>
                    sram_down_merge;
            deepseek_b1_ops::sram_invoke_eltwise_add_or_copy(
                sram_down_merge, moe.routed.sram_down_merge_args, n_sram_active);
        }

        // 9c. Shared: Residual Add — matmul_out + shard(residual) on 112 cores
        //      When multi-device reduce is enabled, only the ROOT1 device performs
        //      the actual add so the residual is counted exactly once after the
        //      cross-device sum.  Non-root devices pass matmul output through.
        {
            DeviceZoneScopedN("SHARED_RESIDUAL_ADD");
#ifdef ENABLE_REDUCE_TO_ONE
            constexpr bool skip_residual_add =
                get_named_compile_time_arg_val("reduce_device_role") != deepseek_b1_ops::MESH_ROOT1;
            deepseek_b1_ops::ResidualAdd::
                Op<Moe::Shared::ResidualAddCTArgs, Core::Shared::is_mcast_receiver_core, skip_residual_add>
                    shared_residual_add;
#else
            deepseek_b1_ops::ResidualAdd::Op<Moe::Shared::ResidualAddCTArgs, Core::Shared::is_mcast_receiver_core>
                shared_residual_add;
#endif
            shared_residual_add(moe.shared.residual_add_args);
        }

        // 9d. Shared: Output Gather — 112 matmul cores → sender core
        {
            DeviceZoneScopedN("SHARED_OUTPUT_GATHER");
            deepseek_b1_ops::MoeGather::Op<
                Core::Shared::is_mcast_receiver_core,  // IsSenderCore: 112 matmul cores
                Core::is_sender_core,                  // IsReceiverCore: sender core
                /*pop_src=*/true,
                /*UsePerCoreSenderIdx=*/true>
                shared_output_gather;
            shared_output_gather(moe.shared.og_args);
        }

        // 9e. Shared: Output Mcast — sender core → 130 cores (DRAM cores receive into add_cb_in1)
        {
            DeviceZoneScopedN("SHARED_OUTPUT_MCAST");
            deepseek_b1_ops::Mcast::Op<
                Moe::Shared::OutputMcastCTArgs,
                Core::is_sender_core,             // IsSenderCore
                Core::is_mcast_grid_core,         // IsMcastGridCore (all 130 cores for semaphore ack)
                Core::Routed::is_gate_proj_core,  // IsReceiverCore (8 DRAM cores receive into add_cb_in1)
                /*pop_src=*/true,
                /*ReceiverOnBrisc=*/true>
                shared_output_mcast;
            shared_output_mcast(moe.shared.output_mcast_args);
        }

        // 12. down_proj: DRAM Matmul Expert Compressed (PopIndex=true: last consumer of expert index)
        //     Active on `is_down_proj_streamer_core` (16 cores when primary_at_last_offset=True,
        //     8 cores otherwise) so senders (= secondaries post-swap) NOC-write their
        //     accum onto the receiver/primary.
        //     SkipNocTridReset=true: same rationale as up_proj — up's leftover NCRISC
        //     writes (incl. atomic sem_inc) carry trid from earlier reads; resetting
        //     trid counters here would wrap them when late acks arrive.
        //     ResetCBIn1=true: required for multi-iter — without it, kernel-managed
        //     l1_write_addr_in1 wraps at (drifted_wr_ptr + size) while framework wr_ptr
        //     wraps at the CB's physical end, so iter 2's writes land at a different
        //     L1 region than the framework's tracked rd_ptr → UNPACK reads stale bytes.
        //     When n_dram_active==0 the kernel is skipped via dram_invoke_matmul;
        //     down_proj's pop_index would have drained cb_index on streamers, but
        //     RECONFIG_MOE_CBS at next iter's top re-anchors the rd/wr ptrs.
        {
            DeviceZoneScopedN("DOWN_PROJ");
            constexpr uint32_t dp_cb_in1_addr = get_named_compile_time_arg_val("down_proj_in1_buf_addr");
            // pop_out=true on secondary streamer cores only. Senders' cb_out is
            // internal (their gather NOC write went to the primary's cb_out slot 0);
            // primaries' cb_out is consumed by eltwise_add → don't pop on primary.
            constexpr bool dp_secondary_pop =
                Core::Routed::is_down_proj_streamer_core && !Core::Routed::is_gate_proj_core;
            deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<
                Moe::Routed::DownProjCTArgs,
                Core::Routed::is_down_proj_streamer_core,
                /*pop_in0=*/true,
                /*pop_index=*/true,
                /*ResetCBIn1=*/true,
                dp_cb_in1_addr,
                /*pop_out=*/dp_secondary_pop,
                /*SkipNocTridReset=*/true>
                down_proj;
            deepseek_b1_ops::dram_invoke_matmul(down_proj, n_dram_active);
        }

        // 13. Eltwise Add: down_proj + shared_expert_output, or copy(shared_expert_output)
        //     when n_dram_active==0 (do_add patched at runtime).
        {
            DeviceZoneScopedN("ELTWISE_ADD");
            constexpr bool add_pop_output =
#ifdef ENABLE_REDUCE_TO_ONE
                false;  // reduce_local_cb aliases add_cb_out — reduce will consume it
#else
                true;  // pop for looping
#endif
            deepseek_b1_ops::EltwiseAddOrCopy::
                Op<Moe::Routed::AddCTArgs, Core::Routed::is_gate_proj_core, add_pop_output>
                    add_op;
            deepseek_b1_ops::dram_invoke_eltwise_add_or_copy(add_op, moe.routed.add_args, n_dram_active);
        }

        // 13. ReduceToOneB1: Multi-device reduce-to-one across 4x2 mesh
        //     Reduces final_output from all 8 devices to ROOT1 device
#ifdef ENABLE_REDUCE_TO_ONE
        {
            DeviceZoneScopedN("REDUCE_TO_ONE");

            // IsReduceCore includes both worker cores and fabric cores
            constexpr bool is_reduce_core = Core::is_reduce_worker_core || Core::is_reduce_fabric_core;
            deepseek_b1_ops::ReduceToOneB1::Op<Moe::Routed::ReduceToOneCTArgs, is_reduce_core, true> reduce_op;
            reduce_op(moe.routed.reduce_rt_args);
        }
#endif

        // Reduce fabric cores signal sender core that fabric sends are done.
        // Sender core NCRISC waits before starting next iteration.
#ifdef ENABLE_REDUCE_TO_ONE
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
            noc_semaphore_set(sync_sem_ptr, 0);  // reset for next iteration
        }
#endif
#endif
    };

    while (true) {
        iteration++;
        moe_body();

        if constexpr (persistent_mode == 0) {
            if (iteration >= num_iterations) {
                break;
            }
        }
    }

    // Teardown (one teardown since all mcasts reuse the same semaphores)
    residual_mcast.teardown(moe.routed.residual_mcast_args);

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
