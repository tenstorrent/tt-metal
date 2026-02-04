// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// MoE Routed Expert fused kernel
// Single kernel file, compiles correctly for all RISC cores
//
// Implements:
//   1. Mcast Input: Broadcast input from sender → compute cores
//   2. Matmul+Sigmoid: Routing matmul on compute cores
//   3. Gather: Collect outputs from compute cores → sender core
//   4. Gate: Top-K expert selection on sender core (produces expert indices)
//   5. Mcast Index: Broadcast expert indices from sender → compute cores
//   6. DRAM Streaming Matmul+SiLU: Expert computation on compute cores
//
// Core roles:
// - Sender core: Mcast sender, Gather receiver, Gate compute, Index mcast sender
// - Compute cores: Mcast receiver, Routing matmul, Gather sender, DRAM matmul
// - Mcast grid: Rectangle encompassing sender and compute cores
//
// RISC assignments:
// - BRISC: Mcast sender (sender core), Gather receiver (sender core), Gate writer (sender core),
//          Index mcast sender (sender core), DRAM streaming reader (compute cores)
// - NCRISC: Mcast receiver, setup sharded buffers, Gather sender (compute cores),
//           Index mcast receiver (compute cores)
// - TRISC: Matmul+Activation compute (compute cores), Gate compute (sender core),
//          DRAM matmul+SiLU compute (compute cores)

#include "../../unified_kernels/kernel_op_api.hpp"
#include "../../unified_kernels/kernel_utils.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/matmul.hpp"
#include "../../unified_kernels/gather.hpp"
#include "../../unified_kernels/deepseek_moe_gate.hpp"
#include "../../unified_kernels/dram_streaming_matmul.hpp"
#include "../../unified_kernels/eltwise_mul.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_sender_core = get_named_compile_time_arg_val("is_sender_core") == 1;
    static constexpr bool is_mcast_grid_core = get_named_compile_time_arg_val("is_mcast_grid_core") == 1;
    static constexpr bool is_gate_mm_core = get_named_compile_time_arg_val("is_gate_mm_core") == 1;
    static constexpr bool is_gate_proj_core = get_named_compile_time_arg_val("is_gate_proj_core") == 1;
    // Cores that need to receive the input mcast (routing matmul OR dram matmul)
    static constexpr bool is_input_mcast_receiver = is_gate_mm_core || is_gate_proj_core;
};

void kernel_main() {
// ============================================================================
// Compile-time args and runtime args per RISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // ------------------------------------------------------------------------
    // Mcast (receiver)
    // ------------------------------------------------------------------------
    using McastCTArgs = deepseek_b1_ops::Mcast::ReceiverCTArgs;
    deepseek_b1_ops::Mcast::ReceiverArgs mcast_args{
        get_named_compile_time_arg_val("mcast_data_receiver_semaphore"),
        get_named_compile_time_arg_val("mcast_dst_cb"),
        get_named_compile_time_arg_val("mcast_dst_num_pages"),
    };

    // ------------------------------------------------------------------------
    // Matmul (reader)
    // ------------------------------------------------------------------------
    using GateMMCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
    deepseek_b1_ops::Matmul::ReaderArgs gate_mm_args{};

    // ------------------------------------------------------------------------
    // Gather (sender - compute cores send to sender core)
    // ------------------------------------------------------------------------
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
        get_named_compile_time_arg_val("gather_receiver_data_addr"),
    };

    // ------------------------------------------------------------------------
    // Gate (reader - empty, setup done below)
    // ------------------------------------------------------------------------
    using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ReaderCTArgs;

    // ------------------------------------------------------------------------
    // Index Mcast (receiver) - reuses same mcast Op, just different args
    // ------------------------------------------------------------------------
    deepseek_b1_ops::Mcast::ReceiverArgs index_mcast_args{
        get_named_compile_time_arg_val("index_mcast_receiver_semaphore"),
        get_named_compile_time_arg_val("gate_proj_cb_index"),
        get_named_compile_time_arg_val("index_mcast_num_pages"),
    };

    // ------------------------------------------------------------------------
    // DRAM Streaming Matmul (reader - no-op, cb_in0 reuses mcast_dst_cb)
    // ------------------------------------------------------------------------
    using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs;

    // ------------------------------------------------------------------------
    // up_proj Matmul (reader - no-op, cb_in0 reuses mcast_dst_cb)
    // ------------------------------------------------------------------------
    using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs;

    // ------------------------------------------------------------------------
    // Mul (reader - setup gate_proj_output as mul_in1 in 16x16 format)
    // ------------------------------------------------------------------------
    using MulCTArgs = deepseek_b1_ops::EltwiseMul::ReaderCTArgs;

    // ------------------------------------------------------------------------
    // Setup sharded persistent buffers
    // ------------------------------------------------------------------------
    if constexpr (Core::is_sender_core) {
        constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
        constexpr uint32_t mcast_src_num_pages = get_named_compile_time_arg_val("mcast_src_num_pages");
        unified_kernels::setup_sharded_buffer(mcast_src_cb, mcast_src_num_pages);

        // Gate tensor-backed CBs (bias and indices)
        // Note: gate_input_cb is NOT setup here - gather already pushes to it
        constexpr uint32_t gate_bias_cb = get_named_compile_time_arg_val("gate_bias_cb");
        constexpr uint32_t gate_input_indices_cb = get_named_compile_time_arg_val("gate_input_indices_cb");
        unified_kernels::setup_sharded_buffer(gate_bias_cb, 1);
        unified_kernels::setup_sharded_buffer(gate_input_indices_cb, 1);
    }
    if constexpr (Core::is_gate_mm_core) {
        constexpr uint32_t gate_mm_in1 = get_named_compile_time_arg_val("gate_mm_in1");
        constexpr uint32_t gate_mm_k_num_tiles = get_named_compile_time_arg_val("gate_mm_k_num_tiles");
        constexpr uint32_t gate_mm_out_w = get_named_compile_time_arg_val("gate_mm_out_w");
        unified_kernels::setup_sharded_buffer(gate_mm_in1, gate_mm_k_num_tiles * gate_mm_out_w);
    }
    if constexpr (Core::is_gate_proj_core) {
        // Setup mul_in1: gate_proj_output viewed as 16x16 tiles for element-wise multiply
        constexpr uint32_t mul_cb_in1 = get_named_compile_time_arg_val("mul_cb_in1");
        constexpr uint32_t mul_num_tiles = get_named_compile_time_arg_val("mul_num_tiles");
        unified_kernels::setup_sharded_buffer(mul_cb_in1, mul_num_tiles);
    }

#elif defined(COMPILE_FOR_BRISC)
    // ------------------------------------------------------------------------
    // Mcast (sender - loopback if sender is in the mcast grid)
    // ------------------------------------------------------------------------
    using McastCTArgs = deepseek_b1_ops::Mcast::SenderCTArgs<
        get_named_compile_time_arg_val("mcast_num_cores"),
        get_named_compile_time_arg_val("mcast_is_part_of_receiver_grid"),
        Core::is_sender_core && Core::is_mcast_grid_core>;

    constexpr uint32_t mcast_src_cb = get_named_compile_time_arg_val("mcast_src_cb");
    constexpr uint32_t mcast_dst_cb = get_named_compile_time_arg_val("mcast_dst_cb");
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

    // ------------------------------------------------------------------------
    // Matmul (writer)
    // ------------------------------------------------------------------------
    using GateMMCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
    deepseek_b1_ops::Matmul::WriterArgs gate_mm_args{};

    // ------------------------------------------------------------------------
    // Gather (receiver - sender core receives from compute cores)
    // ------------------------------------------------------------------------
    deepseek_b1_ops::Gather::ReceiverArgs gather_args{
        get_named_compile_time_arg_val("gather_noc0_num_senders"),
        get_named_compile_time_arg_val("gather_noc1_num_senders"),
        get_named_compile_time_arg_val("gather_noc0_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_noc1_receiver_semaphore_id"),
        get_named_compile_time_arg_val("gather_dst_cb"),
        get_named_compile_time_arg_val("gather_dst_num_pages"),
    };

    // ------------------------------------------------------------------------
    // Gate (writer)
    // ------------------------------------------------------------------------
    using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::WriterCTArgs<
        get_named_compile_time_arg_val("gate_output_cb"),
        get_named_compile_time_arg_val("gate_output_indices_cb")>;

    // ------------------------------------------------------------------------
    // Index Mcast (sender) - reuses same mcast Op, just different args
    // ------------------------------------------------------------------------
    constexpr uint32_t index_mcast_src_cb = get_named_compile_time_arg_val("gate_output_indices_cb");
    constexpr uint32_t index_mcast_dst_cb = get_named_compile_time_arg_val("gate_proj_cb_index");
    deepseek_b1_ops::Mcast::SenderArgs index_mcast_args{
        get_named_compile_time_arg_val("mcast_dest_noc_start_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_start_y"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_x"),
        get_named_compile_time_arg_val("mcast_dest_noc_end_y"),
        get_named_compile_time_arg_val("index_mcast_sender_semaphore"),
        get_named_compile_time_arg_val("index_mcast_receiver_semaphore"),
        get_named_compile_time_arg_val("index_mcast_data_size_bytes"),
        index_mcast_src_cb,
        get_named_compile_time_arg_val("index_mcast_num_pages"),
        get_read_ptr(index_mcast_src_cb),
        get_write_ptr(index_mcast_dst_cb),
    };

    // ------------------------------------------------------------------------
    // DRAM Streaming Matmul (writer)
    // ------------------------------------------------------------------------
    using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs<
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
        1,  // enable_indexing = true
        get_named_compile_time_arg_val("gate_proj_cb_index"),
        get_named_compile_time_arg_val("gate_proj_index_offset")>;

    // ------------------------------------------------------------------------
    // up_proj Matmul (writer) - writes to intermediate CB, not final output
    // ------------------------------------------------------------------------
    using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs<
        get_named_compile_time_arg_val("up_proj_cb_in1"),
        get_named_compile_time_arg_val("up_proj_cb_mm_out"),  // Intermediate output (before mul)
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
        1,  // enable_indexing = true
        get_named_compile_time_arg_val("up_proj_cb_index"),
        get_named_compile_time_arg_val("up_proj_index_offset")>;

    // ------------------------------------------------------------------------
    // Mul (writer) - waits for final output after mul
    // ------------------------------------------------------------------------
    using MulCTArgs = deepseek_b1_ops::EltwiseMul::
        WriterCTArgs<get_named_compile_time_arg_val("mul_cb_out"), get_named_compile_time_arg_val("mul_num_tiles")>;

#elif defined(COMPILE_FOR_TRISC)
    // ------------------------------------------------------------------------
    // Mcast (no-op for TRISC)
    // ------------------------------------------------------------------------
    using McastCTArgs = deepseek_b1_ops::Mcast::ComputeCTArgs;
    deepseek_b1_ops::Mcast::ComputeArgs mcast_args{};

    // ------------------------------------------------------------------------
    // Matmul (compute)
    // ------------------------------------------------------------------------
    using GateMMCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<
        get_named_compile_time_arg_val("gate_mm_out_w"),
        get_named_compile_time_arg_val("gate_mm_fused_activation")>;
    deepseek_b1_ops::Matmul::ComputeArgs gate_mm_args{
        get_named_compile_time_arg_val("gate_mm_in0"),
        get_named_compile_time_arg_val("gate_mm_in1"),
        get_named_compile_time_arg_val("gate_mm_out"),
        get_named_compile_time_arg_val("gate_mm_k_num_tiles"),
    };

    // ------------------------------------------------------------------------
    // Gather (no-op for TRISC)
    // ------------------------------------------------------------------------
    deepseek_b1_ops::Gather::ComputeArgs gather_args{};

    // ------------------------------------------------------------------------
    // Gate (compute)
    // ------------------------------------------------------------------------
    using GateCTArgs = deepseek_b1_ops::DeepseekMoeGate::ComputeCTArgs<
        get_named_compile_time_arg_val("gate_input_cb"),
        get_named_compile_time_arg_val("gate_bias_cb"),
        get_named_compile_time_arg_val("gate_input_indices_cb"),
        get_named_compile_time_arg_val("gate_output_cb"),
        get_named_compile_time_arg_val("gate_output_indices_cb"),
        get_named_compile_time_arg_val("gate_eps"),
        get_named_compile_time_arg_val("gate_scaling_factor"),
        get_named_compile_time_arg_val("gate_enable_sigmoid")>;

    // ------------------------------------------------------------------------
    // Index Mcast (no-op for TRISC) - reuses same mcast Op
    // ------------------------------------------------------------------------
    deepseek_b1_ops::Mcast::ComputeArgs index_mcast_args{};

    // ------------------------------------------------------------------------
    // DRAM Streaming Matmul (compute)
    // ------------------------------------------------------------------------
    using GateProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
        get_named_compile_time_arg_val("gate_proj_cb_in0"),
        get_named_compile_time_arg_val("gate_proj_cb_in1"),
        get_named_compile_time_arg_val("gate_proj_cb_out"),
        get_named_compile_time_arg_val("gate_proj_subblock_k"),
        get_named_compile_time_arg_val("gate_proj_per_core_n"),
        get_named_compile_time_arg_val("gate_proj_subblock_w"),
        get_named_compile_time_arg_val("gate_proj_num_subblocks_k"),
        get_named_compile_time_arg_val("gate_proj_tile_r_dim"),
        get_named_compile_time_arg_val("gate_proj_fuse_silu")>;

    // ------------------------------------------------------------------------
    // up_proj Matmul (compute) - writes to intermediate CB (before mul)
    // ------------------------------------------------------------------------
    using UpProjCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
        get_named_compile_time_arg_val("up_proj_cb_in0"),
        get_named_compile_time_arg_val("up_proj_cb_in1"),
        get_named_compile_time_arg_val("up_proj_cb_mm_out"),  // Intermediate output (before mul)
        get_named_compile_time_arg_val("up_proj_subblock_k"),
        get_named_compile_time_arg_val("up_proj_per_core_n"),
        get_named_compile_time_arg_val("up_proj_subblock_w"),
        get_named_compile_time_arg_val("up_proj_num_subblocks_k"),
        get_named_compile_time_arg_val("up_proj_tile_r_dim"),
        get_named_compile_time_arg_val("up_proj_fuse_silu")>;

    // ------------------------------------------------------------------------
    // Mul (compute): up_proj_mm_out (as 16x16) * gate_proj_out (as 16x16) -> final output
    // cb_in0_wait: wait on up_proj_mm_out (1x32 tiles) before reading aliased 16x16 CB
    // ------------------------------------------------------------------------
    using MulCTArgs = deepseek_b1_ops::EltwiseMul::ComputeCTArgs<
        get_named_compile_time_arg_val("mul_cb_in0"),           // up_proj output aliased as 16x16
        get_named_compile_time_arg_val("mul_cb_in1"),           // gate_proj output aliased as 16x16
        get_named_compile_time_arg_val("mul_cb_out"),           // final output (16x16)
        get_named_compile_time_arg_val("mul_num_tiles"),        // number of 16x16 tiles
        get_named_compile_time_arg_val("up_proj_cb_mm_out"),    // wait on this CB before reading mul_cb_in0
        get_named_compile_time_arg_val("up_proj_per_core_n")>;  // number of tiles in mm_out format
#endif

    // ============================================================================
    // Operation calls
    // ============================================================================

    // ========================================================================
    // 1. Mcast Input: Broadcast input from sender core to all cores that need input
    //    (routing matmul cores AND dram matmul cores)
    // ========================================================================
    deepseek_b1_ops::Mcast::
        Op<McastCTArgs, Core::is_sender_core, Core::is_mcast_grid_core, Core::is_input_mcast_receiver, true>
            mcast;
    mcast.init(mcast_args);
    {
        DeviceZoneScopedN("MCAST");
        mcast(mcast_args);
    }

    // ========================================================================
    // 2. Matmul + Activation: Routing matmul on all matmul cores
    // ========================================================================
    {
        DeviceZoneScopedN("MATMUL");
        // pop_in0 = false (kept for DRAM matmul), pop_in1 = false (weights are persistent)
        deepseek_b1_ops::Matmul::Op<GateMMCTArgs, Core::is_gate_mm_core, false, false> gate_mm;
        gate_mm(gate_mm_args);
    }

    // ========================================================================
    // 3. Gather: Collect matmul outputs from compute cores to sender core
    // ========================================================================
    {
        DeviceZoneScopedN("GATHER");
        // pop_src = true (matmul output consumed after gather)
        deepseek_b1_ops::Gather::Op<Core::is_gate_mm_core, Core::is_sender_core, true> gather;
        gather(gather_args);
    }

    // ========================================================================
    // 4. Gate: Top-K expert selection with normalized scores (on sender core only)
    // ========================================================================
    {
        DeviceZoneScopedN("GATE");
        deepseek_b1_ops::DeepseekMoeGate::Op<GateCTArgs, Core::is_sender_core> gate;
        gate();
    }

    // ========================================================================
    // 5. Mcast Index: Broadcast expert indices from sender core to matmul cores
    //    Reuse the same mcast Op (same grid, same semaphores)
    // ========================================================================
    {
        DeviceZoneScopedN("MCAST_INDEX");
        mcast(index_mcast_args);
    }

    // ========================================================================
    // 6. DRAM Streaming Matmul + SiLU (gate_proj): Expert computation on DRAM matmul cores
    //    PopIn0=false to keep input for up_proj
    // ========================================================================
    {
        DeviceZoneScopedN("GATE_PROJ");
        deepseek_b1_ops::DRAMStreamingMatmul::Op<GateProjCTArgs, Core::is_gate_proj_core, false> gate_proj_mm;
        gate_proj_mm();
    }

    // ========================================================================
    // 7. up_proj Matmul: Expert computation on DRAM matmul cores (no SiLU)
    //    PopIn0=true to release input after use
    //    Writes to intermediate CB (up_proj_cb_mm_out)
    // ========================================================================
    {
        DeviceZoneScopedN("UP_PROJ");
        deepseek_b1_ops::DRAMStreamingMatmul::Op<UpProjCTArgs, Core::is_gate_proj_core, true> up_proj;
        up_proj();
    }

    // ========================================================================
    // 8. Mul: Element-wise multiply up_proj_mm_out * gate_proj_out -> final output
    //    Both inputs viewed as 16x16 tiles (CB aliasing)
    // ========================================================================
    {
        DeviceZoneScopedN("MUL");
        deepseek_b1_ops::EltwiseMul::Op<MulCTArgs, Core::is_gate_proj_core> mul_op;
        mul_op();
    }

    // Only need one teardown since both mcasts reuse the same semaphores
    mcast.teardown();

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
