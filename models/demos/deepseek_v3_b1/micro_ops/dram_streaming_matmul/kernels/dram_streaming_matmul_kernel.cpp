// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// DRAM Streaming Matmul unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Sets up sharded CBs (in0, index tensor)
// BRISC: Streams in1 from DRAM with pipelining
// TRISC: Performs matmul compute with optional fused SiLU

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"

// Compile-time role flag for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("dram_mm_is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    using DRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs;

    // Named compile-time args
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("dram_mm_cb_in0");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("dram_mm_num_tiles_k");
    constexpr uint32_t enable_indexing = get_named_compile_time_arg_val("dram_mm_enable_indexing");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("dram_mm_cb_index");

    // Setup sharded persistent buffers
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
        if constexpr (enable_indexing == 1) {
            unified_kernels::setup_sharded_buffer(cb_index, 1);
        }
    }

#elif defined(COMPILE_FOR_BRISC)
    using DRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs<
        get_named_compile_time_arg_val("dram_mm_cb_in1"),
        get_named_compile_time_arg_val("dram_mm_cb_out"),
        get_named_compile_time_arg_val("dram_mm_in1_tensor_addr"),
        get_named_compile_time_arg_val("dram_mm_in1_page_size"),
        get_named_compile_time_arg_val("dram_mm_in1_num_pages"),
        get_named_compile_time_arg_val("dram_mm_subblock_k"),
        get_named_compile_time_arg_val("dram_mm_per_core_n"),
        get_named_compile_time_arg_val("dram_mm_in1_block_size_bytes"),
        get_named_compile_time_arg_val("dram_mm_out_num_tiles"),
        get_named_compile_time_arg_val("dram_mm_num_subblocks_k"),
        get_named_compile_time_arg_val("dram_mm_bank_id"),
        get_named_compile_time_arg_val("dram_mm_vc"),
        get_named_compile_time_arg_val("dram_mm_enable_indexing"),
        get_named_compile_time_arg_val("dram_mm_cb_index"),
        get_named_compile_time_arg_val("dram_mm_index_offset")>;

#elif defined(COMPILE_FOR_TRISC)
    using DRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
        get_named_compile_time_arg_val("dram_mm_cb_in0"),
        get_named_compile_time_arg_val("dram_mm_cb_in1"),
        get_named_compile_time_arg_val("dram_mm_cb_out"),
        get_named_compile_time_arg_val("dram_mm_subblock_k"),
        get_named_compile_time_arg_val("dram_mm_per_core_n"),
        get_named_compile_time_arg_val("dram_mm_subblock_w"),
        get_named_compile_time_arg_val("dram_mm_num_subblocks_k"),
        get_named_compile_time_arg_val("dram_mm_tile_r_dim"),
        get_named_compile_time_arg_val("dram_mm_fuse_silu")>;
#endif

    // ========================================================================
    // DRAM Streaming Matmul operation
    // ========================================================================
    deepseek_b1_ops::DRAMStreamingMatmul::Op<DRAMMMCTArgs, Core::is_active_core> dram_mm;
    dram_mm();
}
