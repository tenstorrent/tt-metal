// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
#include "../../unified_kernels/dram_streaming_matmul.hpp"
#include "../../unified_kernels/eltwise_mul.hpp"
#include "../../unified_kernels/eltwise_add.hpp"
#include "../../unified_kernels/kn_sliced_matmul.hpp"
#include "../../unified_kernels/gated_reduce.hpp"
#include "../../unified_kernels/residual_add.hpp"
#ifdef ENABLE_REDUCE_TO_ONE
#include "../../unified_kernels/reduce_to_one_b1.hpp"
#endif

// Compile-time role flags for dead code elimination via if constexpr.
// Mirrors Python-side MoeRoutedExpertOp / MoeSharedExpertOp split.
struct Core {
    // Shared between routed and shared expert — same physical core/grid
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
    // Combined: cores that receive the input mcast
    static constexpr bool is_input_mcast_receiver =
        Routed::is_gate_mm_core || Routed::is_gate_proj_core || Shared::is_compute_core;

    // Reduce-to-one core roles
    static constexpr bool is_reduce_worker_core = get_named_compile_time_arg_val("is_reduce_worker_core") == 1;
    static constexpr bool is_reduce_fabric_core = get_named_compile_time_arg_val("is_reduce_fabric_core") == 1;
};

void kernel_main() {
// ============================================================================
// Compile-time args — grouped into Moe::Routed / Moe::Shared structs
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)

    struct Moe {
        struct Routed {
            // Mcast (receiver)
            using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
                get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("mcast_dst_cb"),
                get_named_compile_time_arg_val("mcast_dst_num_pages"),
            };

#ifdef ENABLE_ROUTING
            // Gate Matmul (reader — no-op)
            using GateMMCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
            deepseek_b1_ops::Matmul::ReaderArgs gate_mm_args{};

            // Gather (receiver — MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs gather_args{
                get_named_compile_time_arg_val("gather_noc0_num_senders"),
                get_named_compile_time_arg_val("gather_noc1_num_senders"),
                get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
                get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
                get_named_compile_time_arg_val("gather_dst_cb"),
                get_named_compile_time_arg_val("gather_dst_num_pages"),
            };

            // Gate (reader — no-op)
            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs;

            // Index Mcast (receiver)
            deepseek_b1_ops::Mcast::ReceiverArgs index_mcast_args{
                get_named_compile_time_arg_val("index_mcast_receiver_semaphore"),
                get_named_compile_time_arg_val("gate_proj_cb_index"),
                get_named_compile_time_arg_val("index_mcast_num_pages"),
            };

            // Expert Scale Mcast (receiver)
            deepseek_b1_ops::Mcast::ReceiverArgs expert_scale_mcast_args{
                get_named_compile_time_arg_val("expert_scale_mcast_receiver_semaphore"),
                get_named_compile_time_arg_val("mul_cb_scalar_src"),
                get_named_compile_time_arg_val("expert_scale_mcast_num_pages"),
            };
#endif  // ENABLE_ROUTING

            // gate_proj DRAM Streaming Matmul (reader)
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
                get_named_compile_time_arg_val("enable_routing"),  // enable_indexing
                get_named_compile_time_arg_val("gate_proj_cb_index"),
                get_named_compile_time_arg_val("gate_proj_index_offset"),
                get_named_compile_time_arg_val("use_hardcoded_expert_index")>;

            // up_proj DRAM Streaming Matmul (reader) — shares CB with gate_proj
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
                get_named_compile_time_arg_val("enable_routing"),  // enable_indexing
                get_named_compile_time_arg_val("up_proj_cb_index"),
                get_named_compile_time_arg_val("up_proj_index_offset"),
                get_named_compile_time_arg_val("use_hardcoded_expert_index")>;

            // Eltwise Mul (reader — no-op)
            using MulCTArgs = deepseek_b1_ops::EltwiseMul::ReaderCTArgs;

            // down_proj Gather (receiver — MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs down_proj_gather_args{
                get_named_compile_time_arg_val("down_proj_gather_noc0_num_senders"),
                get_named_compile_time_arg_val("down_proj_gather_noc1_num_senders"),
                get_named_compile_time_arg_val("down_proj_gather_noc0_receiver_semaphore_id"),
                get_named_compile_time_arg_val("down_proj_gather_noc1_receiver_semaphore_id"),
                get_named_compile_time_arg_val("down_proj_gather_dst_cb"),
                get_named_compile_time_arg_val("down_proj_gather_dst_num_pages"),
            };

            // down_proj Mcast (receiver)
            deepseek_b1_ops::Mcast::ReceiverArgs down_proj_mcast_args{
                get_named_compile_time_arg_val("down_proj_mcast_receiver_semaphore"),
                get_named_compile_time_arg_val("down_proj_mcast_dst_cb"),
                get_named_compile_time_arg_val("down_proj_mcast_dst_num_pages"),
            };

            // down_proj DRAM Streaming Matmul (reader)
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
                get_named_compile_time_arg_val("enable_routing"),  // enable_indexing
                get_named_compile_time_arg_val("down_proj_cb_index"),
                get_named_compile_time_arg_val("down_proj_index_offset"),
                get_named_compile_time_arg_val("use_hardcoded_expert_index")>;

            // Eltwise Add (reader — no-op)
            using AddCTArgs = deepseek_b1_ops::EltwiseAdd::ReaderCTArgs;

            // Residual Mcast — receiver (input from sender → residual CB)
            using ResidualMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs residual_mcast_args{
                get_named_compile_time_arg_val("shared_residual_mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("shared_residual_cb"),
                get_named_compile_time_arg_val("shared_residual_num_pages"),
            };

            // RMSNorm (reader — no-op)
            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ReaderCTArgs;
            deepseek_b1_ops::RMSNorm::ReaderArgs rmsnorm_args{};

#ifdef ENABLE_REDUCE_TO_ONE
            // ReduceToOneB1 (reader — receives data from fabric via semaphore waits)
            using ReduceToOneCTArgs = deepseek_b1_ops::ReduceToOneB1::ReaderCTArgs<
                get_named_compile_time_arg_val("reduce_device_role"),
                get_named_compile_time_arg_val("reduce_num_tiles"),
                get_named_compile_time_arg_val("reduce_local_cb"),
                get_named_compile_time_arg_val("reduce_received_cb_r1"),
                get_named_compile_time_arg_val("reduce_received_cb_r2"),
                get_named_compile_time_arg_val("reduce_received_cb_r3"),
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
                get_named_compile_time_arg_val("shared_ag_noc0_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_ag_noc1_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_ag_dst_cb"),
                get_named_compile_time_arg_val("shared_ag_dst_num_pages"),
            };

            // Up Gather (B) receiver (MoeGather: receiver on NCRISC)
            deepseek_b1_ops::MoeGather::ReceiverArgs bg_args{
                get_named_compile_time_arg_val("shared_bg_noc0_num_senders"),
                0,  // noc1_num_senders
                get_named_compile_time_arg_val("shared_bg_noc0_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_bg_noc1_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_bg_dst_cb"),
                get_named_compile_time_arg_val("shared_bg_dst_num_pages"),
            };

            // Gated Reduce (reader — no-op for NCRISC)
            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ReaderCTArgs;
            deepseek_b1_ops::GatedReduce::ReaderArgs gated_reduce_args{};

            // Down Mcast — receiver (gated reduce output → all 130 cores)
            using DownMcastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs down_mcast_args{
                get_named_compile_time_arg_val("shared_down_mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("shared_down_mcast_dst_cb"),
                get_named_compile_time_arg_val("shared_down_mcast_dst_num_pages"),
            };

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
                get_named_compile_time_arg_val("shared_og_noc0_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_og_noc1_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_og_dst_cb"),
                get_named_compile_time_arg_val("shared_og_dst_num_pages"),
            };

            // Output Mcast — receiver (DRAM cores receive into add_cb_in1)
            using OutputMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::ReceiverArgs output_mcast_args{
                get_named_compile_time_arg_val("shared_output_mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("add_cb_in1"),
                get_named_compile_time_arg_val("shared_output_mcast_dst_num_pages"),
            };
        } shared;
    } moe;

    // Setup sharded persistent buffers (imperative — outside struct)
    if constexpr (Core::is_sender_core) {
        // RMSNorm gamma weights (tensor-backed)
        constexpr uint32_t rmsnorm_gamma_cb = get_named_compile_time_arg_val("rmsnorm_gamma_cb");
        constexpr uint32_t rmsnorm_gamma_num_pages = get_named_compile_time_arg_val("rmsnorm_gamma_num_pages");
        unified_kernels::setup_sharded_buffer(rmsnorm_gamma_cb, rmsnorm_gamma_num_pages);

#ifdef ENABLE_ROUTING
        constexpr uint32_t gate_bias_cb = get_named_compile_time_arg_val("gate_bias_cb");
        constexpr uint32_t gate_input_indices_cb = get_named_compile_time_arg_val("gate_input_indices_cb");
        unified_kernels::setup_sharded_buffer(gate_bias_cb, 1);
        unified_kernels::setup_sharded_buffer(gate_input_indices_cb, 1);
#endif  // ENABLE_ROUTING

        // Residual mcast source (pre-RMSNorm input on sender, tensor-backed)
        constexpr uint32_t shared_residual_mcast_src_cb =
            get_named_compile_time_arg_val("shared_residual_mcast_src_cb");
        constexpr uint32_t shared_residual_mcast_src_num_pages =
            get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(shared_residual_mcast_src_cb, shared_residual_mcast_src_num_pages);
    }
#ifdef ENABLE_ROUTING
    if constexpr (Core::Routed::is_gate_mm_core) {
        constexpr uint32_t gate_mm_in1 = get_named_compile_time_arg_val("gate_mm_in1");
        constexpr uint32_t gate_mm_k_num_tiles = get_named_compile_time_arg_val("gate_mm_k_num_tiles");
        constexpr uint32_t gate_mm_out_w = get_named_compile_time_arg_val("gate_mm_out_w");
        unified_kernels::setup_sharded_buffer(gate_mm_in1, gate_mm_k_num_tiles * gate_mm_out_w);
    }
#endif  // ENABLE_ROUTING
    if constexpr (Core::Routed::is_gate_proj_core) {
        constexpr uint32_t mul_cb_in1 = get_named_compile_time_arg_val("mul_cb_in1");
        constexpr uint32_t mul_num_tiles = get_named_compile_time_arg_val("mul_num_tiles");
        unified_kernels::setup_sharded_buffer(mul_cb_in1, mul_num_tiles);

        constexpr uint32_t add_cb_in0 = get_named_compile_time_arg_val("add_cb_in0");
        constexpr uint32_t add_cb_in0_wait_tiles = get_named_compile_time_arg_val("add_cb_in0_wait_tiles");
        unified_kernels::setup_sharded_buffer(add_cb_in0, add_cb_in0_wait_tiles);

        // NOTE: add_cb_in1 is NOT setup here — it is populated by the shared expert's Output Mcast
    }
    if constexpr (Core::Shared::is_compute_core) {
        constexpr uint32_t shared_gu_weights_cb = get_named_compile_time_arg_val("shared_gu_weights_cb");
        constexpr uint32_t shared_gu_weights_num_pages = get_named_compile_time_arg_val("shared_gu_weights_num_pages");
        unified_kernels::setup_sharded_buffer(shared_gu_weights_cb, shared_gu_weights_num_pages);
    }
    if constexpr (Core::Shared::is_mcast_receiver_core) {
        constexpr uint32_t shared_down_in1 = get_named_compile_time_arg_val("shared_down_matmul_in1");
        constexpr uint32_t shared_down_k = get_named_compile_time_arg_val("shared_down_matmul_k_num_tiles");
        constexpr uint32_t shared_down_w = get_named_compile_time_arg_val("shared_down_matmul_out_w_per_core");
        unified_kernels::setup_sharded_buffer(shared_down_in1, shared_down_k * shared_down_w);

        // NOTE: shared_residual_cb is no longer pre-loaded — it is populated by the Residual Mcast (step 0)
    }

#elif defined(COMPILE_FOR_BRISC)

    struct Moe {
        struct Routed {
            // Mcast (sender)
            using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
                get_named_compile_time_arg_val("mcast_num_cores"),
                get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
                Core::is_sender_core && Core::is_mcast_grid_core>;
            deepseek_b1_ops::Mcast::SenderArgs mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("mcast_data_sender_semaphore"),
                get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("mcast_data_size_bytes"),
                get_named_compile_time_arg_val("mcast_src_cb"),
                get_named_compile_time_arg_val("mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("mcast_dst_cb")),
            };

#ifdef ENABLE_ROUTING
            // Gate Matmul (writer — no-op)
            using GateMMCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
            deepseek_b1_ops::Matmul::WriterArgs gate_mm_args{};

            // Gather (sender — MoeGather: sender on BRISC)
            deepseek_b1_ops::MoeGather::SenderArgs gather_args{
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
                get_named_compile_time_arg_val("gather_receiver_data_addr"),
                0,  // sender_idx (unused when UsePerCoreSenderIdx=false)
            };

            // Gate (writer)
            using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::WriterCTArgs<
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("gate_output_indices_cb")>;

            // Index Mcast (sender)
            deepseek_b1_ops::Mcast::SenderArgs index_mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("index_mcast_sender_semaphore"),
                get_named_compile_time_arg_val("index_mcast_receiver_semaphore"),
                get_named_compile_time_arg_val("index_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("gate_output_indices_cb"),
                get_named_compile_time_arg_val("index_mcast_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("gate_output_indices_cb")),
                get_write_ptr(get_named_compile_time_arg_val("gate_proj_cb_index")),
            };

            // Expert Scale Mcast (sender)
            deepseek_b1_ops::Mcast::SenderArgs expert_scale_mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("expert_scale_mcast_sender_semaphore"),
                get_named_compile_time_arg_val("expert_scale_mcast_receiver_semaphore"),
                get_named_compile_time_arg_val("expert_scale_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("gate_output_cb"),
                get_named_compile_time_arg_val("expert_scale_mcast_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("gate_output_cb")),
                get_write_ptr(get_named_compile_time_arg_val("mul_cb_scalar_src")),
            };
#endif  // ENABLE_ROUTING

            // DRAM Streaming Matmul (writer — no-op for BRISC)
            using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;
            using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;

            // Eltwise Mul (writer)
            using MulCTArgs = deepseek_b1_ops::EltwiseMul::WriterCTArgs<
                get_named_compile_time_arg_val("mul_cb_out"),
                get_named_compile_time_arg_val("mul_num_tiles"),
                get_named_compile_time_arg_val("mul_cb_scalar"),
                get_named_compile_time_arg_val("mul_cb_scalar_src"),
                get_named_compile_time_arg_val("mul_scalar_index_offset"),
                get_named_compile_time_arg_val("enable_routing")>;  // enable_scalar

            // down_proj Gather (receiver)
            // down_proj Gather (sender — MoeGather: sender on BRISC)
            deepseek_b1_ops::MoeGather::SenderArgs down_proj_gather_args{
                get_named_compile_time_arg_val("down_proj_gather_dest_noc_x"),
                get_named_compile_time_arg_val("down_proj_gather_dest_noc_y"),
                get_named_compile_time_arg_val("down_proj_gather_data_size_bytes"),
                get_named_compile_time_arg_val("down_proj_gather_receiver_semaphore_id"),
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

            // down_proj Mcast (sender)
            deepseek_b1_ops::Mcast::SenderArgs down_proj_mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("down_proj_mcast_sender_semaphore"),
                get_named_compile_time_arg_val("down_proj_mcast_receiver_semaphore"),
                get_named_compile_time_arg_val("down_proj_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("down_proj_mcast_src_cb"),
                get_named_compile_time_arg_val("down_proj_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("down_proj_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("down_proj_mcast_dst_cb")),
            };

            // down_proj DRAM Matmul (writer — no-op for BRISC)
            using DownProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;

            // Eltwise Add (writer — no-op)
            using AddCTArgs = deepseek_b1_ops::EltwiseAdd::WriterCTArgs;

            // Residual Mcast — sender (input from sender → residual CB, pop_src=false)
            using ResidualMcastCTArgs = McastCTArgs;
            deepseek_b1_ops::Mcast::SenderArgs residual_mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("shared_residual_mcast_data_sender_semaphore"),
                get_named_compile_time_arg_val("shared_residual_mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("shared_residual_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("shared_residual_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_residual_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("shared_residual_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("shared_residual_mcast_dst_cb")),
            };

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
                get_named_compile_time_arg_val("shared_ag_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_ag_src_cb"),
                get_named_compile_time_arg_val("shared_ag_src_num_pages"),
                0,
                0,
                0,
                0,  // sender_grid (unused with UsePerCoreSenderIdx)
                0,  // row_major (unused)
                get_named_compile_time_arg_val("shared_ag_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_ag_sender_idx"),
            };

            // Up Gather (B) sender (MoeGather: sender on BRISC)
            deepseek_b1_ops::MoeGather::SenderArgs bg_args{
                get_named_compile_time_arg_val("shared_bg_dest_noc_x"),
                get_named_compile_time_arg_val("shared_bg_dest_noc_y"),
                get_named_compile_time_arg_val("shared_bg_data_size_bytes"),
                get_named_compile_time_arg_val("shared_bg_receiver_semaphore_id"),
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

            // Gated Reduce (writer — no-op for BRISC)
            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::WriterCTArgs;
            deepseek_b1_ops::GatedReduce::WriterArgs gated_reduce_args{};

            // Down Mcast — sender (reuse Routed::McastCTArgs: same grid, same persistent sender)
            using DownMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::SenderArgs down_mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("shared_down_mcast_data_sender_semaphore"),
                get_named_compile_time_arg_val("shared_down_mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("shared_down_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("shared_down_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_down_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("shared_down_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("shared_down_mcast_dst_cb")),
            };

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
                get_named_compile_time_arg_val("shared_og_receiver_semaphore_id"),
                get_named_compile_time_arg_val("shared_og_src_cb"),
                get_named_compile_time_arg_val("shared_og_src_num_pages"),
                0,  // sender_grid_start_x (unused with UsePerCoreSenderIdx)
                0,  // sender_grid_start_y
                0,  // sender_grid_end_x
                0,  // sender_grid_end_y
                0,  // row_major (unused)
                get_named_compile_time_arg_val("shared_og_receiver_data_addr"),
                get_named_compile_time_arg_val("shared_residual_add_core_idx"),  // reuse matmul core index
            };

            // Output Mcast — sender (sender core → 130 cores)
            using OutputMcastCTArgs = Routed::McastCTArgs;
            deepseek_b1_ops::Mcast::SenderArgs output_mcast_args{
                get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
                get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
                get_named_compile_time_arg_val("shared_output_mcast_data_sender_semaphore"),
                get_named_compile_time_arg_val("shared_output_mcast_data_receiver_semaphore"),
                get_named_compile_time_arg_val("shared_output_mcast_data_size_bytes"),
                get_named_compile_time_arg_val("shared_output_mcast_src_cb"),
                get_named_compile_time_arg_val("shared_output_mcast_src_num_pages"),
                get_read_ptr(get_named_compile_time_arg_val("shared_output_mcast_src_cb")),
                get_write_ptr(get_named_compile_time_arg_val("add_cb_in1")),
            };
        } shared;
    } moe;

#ifdef ENABLE_REDUCE_TO_ONE
    // Populate BRISC reduce runtime args (must be outside struct initializer)
    constexpr size_t reduce_brisc_arg_start = get_named_compile_time_arg_val("reduce_brisc_rt_arg_base");
    if constexpr (Core::is_reduce_worker_core) {
        moe.routed.reduce_rt_args = deepseek_b1_ops::ReduceToOneB1::WorkerWriterArgs{
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 0),  // fabric_core_noc_x
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 1),  // fabric_core_noc_y
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 2),  // my_slot_idx
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 3),  // worker_sem_id
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 4),  // dst_l1_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 5),  // dst_sem_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 6),  // output_base_addr
            get_arg_val<uint32_t>(reduce_brisc_arg_start + 7),  // shard_idx
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

            // gate_proj DRAM Streaming Matmul (compute)
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

            // up_proj DRAM Streaming Matmul (compute) — shares CB with gate_proj
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
                get_named_compile_time_arg_val("enable_routing")>;  // enable_scalar

            // down_proj Gather (compute — no-op)
            deepseek_b1_ops::MoeGather::ComputeArgs down_proj_gather_args{};

            // down_proj Mcast (compute — no-op)
            deepseek_b1_ops::Mcast::ComputeArgs down_proj_mcast_args{};

            // down_proj DRAM Streaming Matmul (compute)
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

            // Eltwise Add (compute)
            using AddCTArgs = deepseek_b1_ops::EltwiseAdd::ComputeCTArgs<
                get_named_compile_time_arg_val("add_cb_in0"),
                get_named_compile_time_arg_val("add_cb_in1"),
                get_named_compile_time_arg_val("add_cb_out"),
                get_named_compile_time_arg_val("add_num_tiles"),
                get_named_compile_time_arg_val("down_proj_cb_out"),      // cb_in0_wait (actual producer)
                get_named_compile_time_arg_val("down_proj_per_core_n"),  // cb_in0_wait_tiles
                get_named_compile_time_arg_val("add_cb_in1_wait_tiles"),
                get_named_compile_time_arg_val("add_sender_index"),
                get_named_compile_time_arg_val("add_slice_size_bytes")>;

            // Residual Mcast (compute — no-op)
            using ResidualMcastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
            deepseek_b1_ops::Mcast::ComputeArgs residual_mcast_args{};

            // RMSNorm (compute — sender core only)
            // Input: residual_mcast_src_cb (raw activation), Output: rmsnorm_output_cb
            using RMSNormCTArgs = deepseek_b1_ops::RMSNorm::ComputeCTArgs<
                get_named_compile_time_arg_val("rmsnorm_fp32_acc") == 1,
                get_named_compile_time_arg_val("rmsnorm_num_tiles"),
                get_named_compile_time_arg_val("rmsnorm_rsqrt_fast_approx") == 1,
                get_named_compile_time_arg_val("rmsnorm_input_cb"),  // residual_mcast_src_cb
                get_named_compile_time_arg_val("rmsnorm_gamma_cb"),
                get_named_compile_time_arg_val("rmsnorm_output_cb")>;  // rmsnorm_output_cb
            deepseek_b1_ops::RMSNorm::ComputeArgs rmsnorm_args{
                get_common_arg_val<uint32_t>(get_named_compile_time_arg_val("rmsnorm_trisc_common_rt_arg_base") + 0),
                get_common_arg_val<float>(get_named_compile_time_arg_val("rmsnorm_trisc_common_rt_arg_base") + 1),
            };

#ifdef ENABLE_REDUCE_TO_ONE
            // ReduceToOneB1 (compute — performs reduction)
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
            };

            // Gather (compute — no-op for TRISC)
            deepseek_b1_ops::MoeGather::ComputeArgs ag_args{};
            deepseek_b1_ops::MoeGather::ComputeArgs bg_args{};

            // Gated Reduce (compute)
            using GatedReduceCTArgs = deepseek_b1_ops::GatedReduce::ComputeCTArgs<
                get_named_compile_time_arg_val("shared_gated_reduce_tiles_per_k"),
                get_named_compile_time_arg_val("shared_gated_reduce_k_num_tiles")>;
            deepseek_b1_ops::GatedReduce::ComputeArgs gated_reduce_args{
                get_named_compile_time_arg_val("shared_gated_reduce_group1_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_group2_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_intermed_cb"),
                get_named_compile_time_arg_val("shared_gated_reduce_mcast_src_cb"),
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

    auto moe_body = [&]() {
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

#ifdef ENABLE_ROUTING
        // 2. Matmul + Activation: Routing matmul on gate_mm cores
        {
            DeviceZoneScopedN("MATMUL");
            deepseek_b1_ops::Matmul::Op<Moe::Routed::GateMMCTArgs, Core::Routed::is_gate_mm_core, false, false> gate_mm;
            gate_mm(moe.routed.gate_mm_args);
        }

        // 3. Gather: Collect matmul outputs from compute cores to sender core
        {
            DeviceZoneScopedN("GATHER");
            deepseek_b1_ops::MoeGather::Op<Core::Routed::is_gate_mm_core, Core::is_sender_core, true> gather;
            gather(moe.routed.gather_args);
        }
#endif  // ENABLE_ROUTING

        // 3a. Shared Expert: Gate/Up KN-sliced matmul on 128 compute cores
        //     CB 1 (act) is shared: on gate_proj cores it is also consumed by gate_proj (step 6)
        //     and up_proj (step 7, which pops it). So we only pop here on non-gate_proj cores.
        {
            DeviceZoneScopedN("SHARED_GU_MATMUL");
            deepseek_b1_ops::KNSlicedMatmul::Op<
                Moe::Shared::GUMatmulCTArgs,
                Core::Shared::is_compute_core,
                !Core::Routed::is_gate_proj_core,  // pop_act
                false>                             // pop_weights
                shared_gu_matmul;
            shared_gu_matmul(moe.shared.gu_matmul_args);
        }

#ifdef ENABLE_ROUTING
        // 4. Gate: Top-K expert selection (on sender core only)
        {
            DeviceZoneScopedN("GATE");
            deepseek_b1_ops::DeepseekMoeGate::Op<Moe::Routed::GateCTArgs, Core::is_sender_core> gate;
            gate();
        }

        // 5. Mcast Index: Broadcast expert indices to gate_proj cores only
        // Uses dedicated mcast with IsReceiverCore=is_gate_proj_core so only gate_proj
        // cores push to CB 10. Other grid cores just drain the semaphore (no CB ops).
        {
            DeviceZoneScopedN("MCAST_INDEX");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Routed::is_gate_proj_core,
                true>
                index_mcast;
            index_mcast(moe.routed.index_mcast_args);
        }

        // 5b. Mcast Expert Scale: Broadcast expert scale to gate_proj cores
        {
            DeviceZoneScopedN("MCAST_EXPERT_SCALE");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Routed::is_gate_proj_core,
                true>  // pop_src
                expert_scale_mcast;
            expert_scale_mcast(moe.routed.expert_scale_mcast_args);
        }
#endif  // ENABLE_ROUTING

        // 5c. Shared Expert: Gate Gather (A) — 64 gate cores send to sender core
        //     Uses MoeGather (sender=BRISC) to avoid NOC contention with DRAM matmul (NCRISC)
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

        // 5d. Shared Expert: Up Gather (B) — 64 up cores send to sender core
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

        // 5e. Shared Expert: Gated Reduce — SiLU(sum(gate)) * sum(up)
        //     Runs on sender core after receiving both gathers
        {
            DeviceZoneScopedN("SHARED_GATED_REDUCE");
            deepseek_b1_ops::GatedReduce::Op<Moe::Shared::GatedReduceCTArgs, Core::Shared::is_gated_reduce_core>
                gated_reduce;
            gated_reduce(moe.shared.gated_reduce_args);
        }

        // 6. gate_proj: DRAM Streaming Matmul + SiLU (PopIn0=false, keep input for up_proj)
        {
            DeviceZoneScopedN("GATE_PROJ");
            constexpr uint32_t gate_proj_cb_in1_addr = get_named_compile_time_arg_val("gate_proj_in1_buf_addr");
            deepseek_b1_ops::DRAMStreamingMatmul::
                Op<Moe::Routed::GateProjCTArgs, Core::Routed::is_gate_proj_core, false, true, gate_proj_cb_in1_addr>
                    gate_proj_mm;
            gate_proj_mm();
        }

        // 7. up_proj: DRAM Streaming Matmul (PopIn0=true, ResetCBIn1=true, WaitForOutput=true)
        {
            DeviceZoneScopedN("UP_PROJ");
            constexpr uint32_t cb_in1_addr = get_named_compile_time_arg_val("gate_proj_in1_buf_addr");
            deepseek_b1_ops::DRAMStreamingMatmul::
                Op<Moe::Routed::UpProjCTArgs, Core::Routed::is_gate_proj_core, true, true, cb_in1_addr, false, true>
                    up_proj;
            up_proj();
        }

        // 8. Mul: Element-wise multiply (up_proj * gate_proj * expert_scale)
        {
            DeviceZoneScopedN("MUL");
            deepseek_b1_ops::EltwiseMul::Op<Moe::Routed::MulCTArgs, Core::Routed::is_gate_proj_core> mul_op;
            mul_op();
        }

        // 9. down_proj Gather: Gather fused output from gate_proj cores to sender core
        {
            DeviceZoneScopedN("DOWN_PROJ_GATHER");
            deepseek_b1_ops::MoeGather::Op<Core::Routed::is_gate_proj_core, Core::is_sender_core, true, true>
                down_proj_gather;
            down_proj_gather(moe.routed.down_proj_gather_args);
        }

        // 10. down_proj Mcast: Broadcast gathered fused output to gate_proj cores
        {
            DeviceZoneScopedN("DOWN_PROJ_MCAST");
            deepseek_b1_ops::Mcast::Op<
                Moe::Routed::McastCTArgs,
                Core::is_sender_core,
                Core::is_mcast_grid_core,
                Core::Routed::is_gate_proj_core,
                true>  // pop_src
                down_proj_mcast;
            down_proj_mcast(moe.routed.down_proj_mcast_args);
        }

        // 11. down_proj: DRAM Streaming Matmul (PopIndex=true: last consumer of expert index CB)
        {
            DeviceZoneScopedN("DOWN_PROJ");
            constexpr uint32_t down_proj_cb_in1_addr = get_named_compile_time_arg_val("down_proj_in1_buf_addr");
            deepseek_b1_ops::DRAMStreamingMatmul::Op<
                Moe::Routed::DownProjCTArgs,
                Core::Routed::is_gate_proj_core,
                true,
                true,
                down_proj_cb_in1_addr,
                true>
                down_proj;
            down_proj();
        }

        // 11b. Shared: Down Mcast — broadcast gated reduce output [1, K_down] to all 130 cores
        //      Source is mcast_src_cb (CB 31) filled by gated reduce, pop_src=true
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

        // 11c. Shared: Down Proj Matmul — SRAM matmul [1, K_down] x [K_down, N_per_core] on 112 cores
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

        // 11d. Shared: Residual Add — matmul_out + shard(residual) on 112 cores
        {
            DeviceZoneScopedN("SHARED_RESIDUAL_ADD");
            deepseek_b1_ops::ResidualAdd::Op<Moe::Shared::ResidualAddCTArgs, Core::Shared::is_mcast_receiver_core>
                shared_residual_add;
            shared_residual_add(moe.shared.residual_add_args);
        }

        // 11e. Shared: Output Gather — 112 matmul cores → sender core
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

        // 11f. Shared: Output Mcast — sender core → 130 cores (DRAM cores receive into add_cb_in1)
        {
            DeviceZoneScopedN("SHARED_OUTPUT_MCAST");
            deepseek_b1_ops::Mcast::Op<
                Moe::Shared::OutputMcastCTArgs,
                Core::is_sender_core,             // IsSenderCore
                Core::is_mcast_grid_core,         // IsMcastGridCore (all 130 cores for semaphore ack)
                Core::Routed::is_gate_proj_core,  // IsReceiverCore (8 DRAM cores receive into add_cb_in1)
                /*pop_src=*/true>
                shared_output_mcast;
            shared_output_mcast(moe.shared.output_mcast_args);
        }

        // 12. Eltwise Add: down_proj + shared_expert_output
        {
            DeviceZoneScopedN("ELTWISE_ADD");
            constexpr bool add_pop_output =
#ifdef ENABLE_REDUCE_TO_ONE
                false;  // reduce_local_cb aliases add_cb_out — reduce will consume it
#else
                true;  // pop for looping
#endif
            deepseek_b1_ops::EltwiseAdd::Op<
                Moe::Routed::AddCTArgs,
                Core::Routed::is_gate_proj_core,
                true,            // PopInputs
                add_pop_output>  // PopOutput
                add_op;
            add_op();
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
    };

    for (uint32_t i = 0; i < num_iterations; i++) {
        moe_body();
    }

    // Teardown (one teardown since all mcasts reuse the same semaphores)
    residual_mcast.teardown();

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
