// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// DRAM Streaming Matmul unified kernel
// Single kernel file, compiles for all RISC cores
//
// NCRISC: Streams in1 from DRAM with pipelining (uses NOC_0), sets up sharded CBs
// BRISC: No-op for DRAM streaming (handles mul scalar copy if enabled)
// TRISC: Performs matmul compute with optional fused SiLU and optional fused mul (16x16 tiles)

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/dram_streaming_matmul.hpp"
#include "../../../unified_kernels/eltwise_mul.hpp"

// Compile-time role flag for dead code elimination via if constexpr
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("dram_mm_is_active_core") == 1;
};

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // NCRISC: DRAM streaming (uses NOC_0 via ReaderConfigDescriptor)
    using DRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<
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

    // Named compile-time args for sharded buffer setup
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("dram_mm_cb_in0");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("dram_mm_num_tiles_k");
    constexpr uint32_t enable_indexing = get_named_compile_time_arg_val("dram_mm_enable_indexing");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("dram_mm_cb_index");
    constexpr uint32_t enable_mul = get_named_compile_time_arg_val("dram_mm_enable_mul");
    constexpr uint32_t cb_mul_in1 = get_named_compile_time_arg_val("dram_mm_cb_mul_in1");
    constexpr uint32_t mul_num_tiles = get_named_compile_time_arg_val("dram_mm_mul_num_tiles");
    constexpr uint32_t cb_scalar_src = get_named_compile_time_arg_val("dram_mm_cb_scalar_src");

    // Setup sharded persistent buffers
    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
        if constexpr (enable_indexing == 1) {
            unified_kernels::setup_sharded_buffer(cb_index, 1);
        }
        if constexpr (enable_mul == 1) {
            unified_kernels::setup_sharded_buffer(cb_mul_in1, mul_num_tiles);
            unified_kernels::setup_sharded_buffer(cb_scalar_src, 1);
        }
    }

    // MulCTArgs for NCRISC - no-op
    using MulCTArgs = deepseek_b1_ops::EltwiseMul::ReaderCTArgs;

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: No-op for DRAM streaming (handled by NCRISC)
    using DRAMMMCTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs;

    // Mul parameters for BRISC
    constexpr uint32_t enable_mul = get_named_compile_time_arg_val("dram_mm_enable_mul");

    // MulCTArgs: waits for final output (cb_out after mul writes to it)
    // Copies scalar from source CB to working CB
    using MulCTArgs = deepseek_b1_ops::EltwiseMul::WriterCTArgs<
        get_named_compile_time_arg_val("dram_mm_cb_final_out"),
        get_named_compile_time_arg_val("dram_mm_mul_num_tiles"),
        get_named_compile_time_arg_val("dram_mm_cb_scalar"),
        get_named_compile_time_arg_val("dram_mm_cb_scalar_src"),
        0>;  // scalar_index_offset

#elif defined(COMPILE_FOR_TRISC)
    // Matmul writes to dram_mm_cb_out (CB 4 when mul disabled, CB 8 when mul enabled)
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

    constexpr uint32_t enable_mul = get_named_compile_time_arg_val("dram_mm_enable_mul");

    // MulCTArgs: CB 7 * CB 6 -> CB 4 (all 16x16 tiles)
    // cb_in0_wait = dram_mm_cb_out (wait on matmul output before reading aliased CB 7)
    // Scalar multiply: result * cb_scalar -> result
    using MulCTArgs = deepseek_b1_ops::EltwiseMul::ComputeCTArgs<
        get_named_compile_time_arg_val("dram_mm_cb_mul_in0"),
        get_named_compile_time_arg_val("dram_mm_cb_mul_in1"),
        get_named_compile_time_arg_val("dram_mm_cb_mul_out"),
        get_named_compile_time_arg_val("dram_mm_mul_num_tiles"),
        get_named_compile_time_arg_val("dram_mm_cb_out"),         // cb_in0_wait
        get_named_compile_time_arg_val("dram_mm_per_core_n"),     // cb_in0_wait_tiles
        get_named_compile_time_arg_val("dram_mm_cb_mul_in1"),     // cb_in1_wait (same as cb_in1)
        get_named_compile_time_arg_val("dram_mm_mul_num_tiles"),  // cb_in1_wait_tiles (same as num_tiles)
        get_named_compile_time_arg_val("dram_mm_cb_scalar"),
        get_named_compile_time_arg_val("dram_mm_mul_fp32_dest_acc_en")>;
    deepseek_compute_kernel_init();
#endif

    // ========================================================================
    // DRAM Streaming Matmul operation (looped for testing CB-boundary wrapping)
    // ========================================================================
    constexpr uint32_t num_loop_iters = get_named_compile_time_arg_val("dram_mm_num_loop_iters");
    constexpr uint32_t cb_in1_buf_addr = get_named_compile_time_arg_val("dram_mm_cb_in1_buf_addr");

    for (uint32_t iter = 0; iter < num_loop_iters; ++iter) {
        deepseek_b1_ops::DRAMStreamingMatmul::Op<
            DRAMMMCTArgs,
            Core::is_active_core,
            true,                  // PopIn0 = true first, re-setup below
            (num_loop_iters > 1),  // ResetCBIn1 when looping
            cb_in1_buf_addr>
            dram_mm;
        dram_mm();

#if defined(COMPILE_FOR_NCRISC)
        // Between iterations: wait for compute to finish output, free cb_out, re-setup cb_in0
        if constexpr (Core::is_active_core && num_loop_iters > 1) {
            if (iter < num_loop_iters - 1) {
                // Wait for TRISC to finish pushing all output tiles, then pop to free space
                constexpr uint32_t out_num_tiles = get_named_compile_time_arg_val("dram_mm_out_num_tiles");
                constexpr uint32_t cb_out = get_named_compile_time_arg_val("dram_mm_cb_out");
                cb_wait_front(cb_out, out_num_tiles);
                cb_pop_front(cb_out, out_num_tiles);
                // Re-setup in0 sharded buffer (PopIn0 consumed it)
                unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
            }
        }
#endif
    }

    // ========================================================================
    // Optional fused element-wise multiply (16x16 tiles)
    // When enabled:
    //   - Matmul wrote to cb_out (CB 8, 8 tiles of 1x32)
    //   - mul_tensor is in cb_mul_in1 (CB 6, 1 tile of 16x16)
    //   - cb_mul_in0 (CB 7) views same memory as CB 8 but with 16x16 format
    //   - Mul waits on CB 8, then reads cb_mul_in0 * cb_mul_in1 -> cb_mul_out
    //   - If scalar_mul enabled, also multiplies by scalar from CB 9
    // ========================================================================
    if constexpr (enable_mul == 1) {
        deepseek_b1_ops::EltwiseMul::Op<MulCTArgs, Core::is_active_core> mul_op;
        mul_op();
    }
}
