// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// DRAM Streaming Matmul unified kernel
// Single kernel file, compiles correctly for all RISC cores
//
// NCRISC: Signals tensor-backed CB is ready
// BRISC: Streams in1 from DRAM with transaction IDs
// TRISC: Performs matmul compute with optional SILU

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"

// Compile-time role flags for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define args per RISC (different compile-time arg layout per processor)
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // CTArgs type alias
    using DSMCTArgs =
        deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<get_named_compile_time_arg_val("dsm_num_tiles_k")>;

    // Named compile-time args
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("dsm_in0");

    // Reader args
    deepseek_b1_ops::DRAMStreamingMatmul::ReaderArgs dsm_args{
        .in0_cb = in0_cb,
    };

#elif defined(COMPILE_FOR_BRISC)
    // CTArgs type alias
    using DSMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs<
        get_named_compile_time_arg_val("dsm_in1_page_size"),
        get_named_compile_time_arg_val("dsm_in1_num_pages"),
        get_named_compile_time_arg_val("dsm_subblock_k"),
        get_named_compile_time_arg_val("dsm_per_core_N"),
        get_named_compile_time_arg_val("dsm_in1_block_size_bytes"),
        get_named_compile_time_arg_val("dsm_out_num_tiles"),
        get_named_compile_time_arg_val("dsm_num_subblocks_k")>;

    // Named compile-time args
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("dsm_in1");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("dsm_out");
    constexpr uint32_t in1_tensor_addr = get_named_compile_time_arg_val("dsm_in1_tensor_addr");

    // Per-core runtime args
    const uint32_t dram_bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);

    // Writer args
    deepseek_b1_ops::DRAMStreamingMatmul::WriterArgs dsm_args{
        .in1_cb = in1_cb,
        .out_cb = out_cb,
        .in1_tensor_addr = in1_tensor_addr,
        .dram_bank_id = dram_bank_id,
        .vc = vc,
    };

#elif defined(COMPILE_FOR_TRISC)
    // CTArgs type alias
    using DSMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
        get_named_compile_time_arg_val("dsm_subblock_k"),
        get_named_compile_time_arg_val("dsm_per_core_N"),
        get_named_compile_time_arg_val("dsm_subblock_w"),
        get_named_compile_time_arg_val("dsm_num_subblocks_k"),
        get_named_compile_time_arg_val("dsm_tile_r_dim")>;

    // Named compile-time args
    constexpr uint32_t in0_cb = get_named_compile_time_arg_val("dsm_in0");
    constexpr uint32_t in1_cb = get_named_compile_time_arg_val("dsm_in1");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("dsm_out");

    // Compute args
    deepseek_b1_ops::DRAMStreamingMatmul::ComputeArgs dsm_args{
        .in0_cb = in0_cb,
        .in1_cb = in1_cb,
        .out_cb = out_cb,
    };
#endif

    // ========================================================================
    // DRAMStreamingMatmul operation
    // ========================================================================
    deepseek_b1_ops::DRAMStreamingMatmul::Op<DSMCTArgs, Core::is_active_core> dsm;
    dsm(dsm_args);
}
