// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused DRAM Streaming Matmul kernel
// Executes two DRAMStreamingMatmul ops sequentially using the unified kernel pattern

#include "../../unified_kernels/kernel_op_api.hpp"
#include "../../unified_kernels/dram_streaming_matmul.hpp"

// Compile-time role flags
struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

// Op enable flags (set to 1 when op is enabled, 0 otherwise)
constexpr bool sub0_enabled = get_named_compile_time_arg_val("sub0_enabled") == 1;
constexpr bool sub1_enabled = get_named_compile_time_arg_val("sub1_enabled") == 1;

void kernel_main() {
    // ============================================================================
    // Op 0: First DRAM Streaming Matmul
    // ============================================================================
    if constexpr (sub0_enabled) {
#if defined(COMPILE_FOR_NCRISC)
        using Sub0CTArgs =
            deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<get_named_compile_time_arg_val("sub0_num_tiles_k")>;

        constexpr uint32_t in0_cb = get_named_compile_time_arg_val("sub0_in0");
        deepseek_b1_ops::DRAMStreamingMatmul::ReaderArgs sub0_args{.in0_cb = in0_cb};

#elif defined(COMPILE_FOR_BRISC)
        using Sub0CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs<
            get_named_compile_time_arg_val("sub0_in1_page_size"),
            get_named_compile_time_arg_val("sub0_in1_num_pages"),
            get_named_compile_time_arg_val("sub0_subblock_k"),
            get_named_compile_time_arg_val("sub0_per_core_N"),
            get_named_compile_time_arg_val("sub0_in1_block_size_bytes"),
            get_named_compile_time_arg_val("sub0_out_num_tiles"),
            get_named_compile_time_arg_val("sub0_num_subblocks_k")>;

        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("sub0_in1");
        constexpr uint32_t out_cb = get_named_compile_time_arg_val("sub0_out");
        constexpr uint32_t in1_tensor_addr = get_named_compile_time_arg_val("sub0_in1_tensor_addr");
        const uint32_t dram_bank_id = get_arg_val<uint32_t>(0);
        const uint32_t vc = get_arg_val<uint32_t>(1);

        deepseek_b1_ops::DRAMStreamingMatmul::WriterArgs sub0_args{
            .in1_cb = in1_cb,
            .out_cb = out_cb,
            .in1_tensor_addr = in1_tensor_addr,
            .dram_bank_id = dram_bank_id,
            .vc = vc,
        };

#elif defined(COMPILE_FOR_TRISC)
        using Sub0CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
            get_named_compile_time_arg_val("sub0_subblock_k"),
            get_named_compile_time_arg_val("sub0_per_core_N"),
            get_named_compile_time_arg_val("sub0_subblock_w"),
            get_named_compile_time_arg_val("sub0_num_subblocks_k"),
            get_named_compile_time_arg_val("sub0_tile_r_dim")>;

        constexpr uint32_t in0_cb = get_named_compile_time_arg_val("sub0_in0");
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("sub0_in1");
        constexpr uint32_t out_cb = get_named_compile_time_arg_val("sub0_out");

        deepseek_b1_ops::DRAMStreamingMatmul::ComputeArgs sub0_args{
            .in0_cb = in0_cb,
            .in1_cb = in1_cb,
            .out_cb = out_cb,
        };
#endif

        deepseek_b1_ops::DRAMStreamingMatmul::Op<Sub0CTArgs, Core::is_active_core> sub0;
        sub0(sub0_args);
    }

    // ============================================================================
    // Op 1: Second DRAM Streaming Matmul
    // ============================================================================
    if constexpr (sub1_enabled) {
#if defined(COMPILE_FOR_NCRISC)
        using Sub1CTArgs =
            deepseek_b1_ops::DRAMStreamingMatmul::ReaderCTArgs<get_named_compile_time_arg_val("sub1_num_tiles_k")>;

        constexpr uint32_t in0_cb = get_named_compile_time_arg_val("sub1_in0");
        deepseek_b1_ops::DRAMStreamingMatmul::ReaderArgs sub1_args{.in0_cb = in0_cb};

#elif defined(COMPILE_FOR_BRISC)
        using Sub1CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::WriterCTArgs<
            get_named_compile_time_arg_val("sub1_in1_page_size"),
            get_named_compile_time_arg_val("sub1_in1_num_pages"),
            get_named_compile_time_arg_val("sub1_subblock_k"),
            get_named_compile_time_arg_val("sub1_per_core_N"),
            get_named_compile_time_arg_val("sub1_in1_block_size_bytes"),
            get_named_compile_time_arg_val("sub1_out_num_tiles"),
            get_named_compile_time_arg_val("sub1_num_subblocks_k")>;

        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("sub1_in1");
        constexpr uint32_t out_cb = get_named_compile_time_arg_val("sub1_out");
        constexpr uint32_t in1_tensor_addr = get_named_compile_time_arg_val("sub1_in1_tensor_addr");
        const uint32_t dram_bank_id = get_arg_val<uint32_t>(2);
        const uint32_t vc = get_arg_val<uint32_t>(3);

        deepseek_b1_ops::DRAMStreamingMatmul::WriterArgs sub1_args{
            .in1_cb = in1_cb,
            .out_cb = out_cb,
            .in1_tensor_addr = in1_tensor_addr,
            .dram_bank_id = dram_bank_id,
            .vc = vc,
        };

#elif defined(COMPILE_FOR_TRISC)
        using Sub1CTArgs = deepseek_b1_ops::DRAMStreamingMatmul::ComputeCTArgs<
            get_named_compile_time_arg_val("sub1_subblock_k"),
            get_named_compile_time_arg_val("sub1_per_core_N"),
            get_named_compile_time_arg_val("sub1_subblock_w"),
            get_named_compile_time_arg_val("sub1_num_subblocks_k"),
            get_named_compile_time_arg_val("sub1_tile_r_dim")>;

        constexpr uint32_t in0_cb = get_named_compile_time_arg_val("sub1_in0");
        constexpr uint32_t in1_cb = get_named_compile_time_arg_val("sub1_in1");
        constexpr uint32_t out_cb = get_named_compile_time_arg_val("sub1_out");

        deepseek_b1_ops::DRAMStreamingMatmul::ComputeArgs sub1_args{
            .in0_cb = in0_cb,
            .in1_cb = in1_cb,
            .out_cb = out_cb,
        };
#endif

        deepseek_b1_ops::DRAMStreamingMatmul::Op<Sub1CTArgs, Core::is_active_core> sub1;
        sub1(sub1_args);
    }
}
