// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Expert Matmul with Compressed Weights — Hybrid SRAM + DRAM kernel.
//
// Both sram_mm() and dram_mm() are called. Each loops over the index array,
// checks is_dram[expert_idx], and skips non-matching experts.
//
// SRAM path uses cb_in1 (CB 1) for sharded B data.
// DRAM path uses cb_in1_dram (CB 4) for streaming buffer.
// Sharded buffer setup for cb_in0, cb_in1, cb_index done here before Op calls.

#include "../../../unified_kernels/matmul_expert_compressed.hpp"

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC + shared NCRISC setup
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Shared setup: sharded buffers for A and index.
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");
    unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
    unified_kernels::setup_sharded_buffer(cb_index, 1);

    if constexpr (get_named_compile_time_arg_val("sram_active") != 0) {
        constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
        unified_kernels::setup_sharded_buffer(cb_in1, 1);
    }

    using SRAMArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("out_w"),
        get_named_compile_time_arg_val("cb_in0_num_pages"),
        get_named_compile_time_arg_val("sram_fmt_l1_addr"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("is_dram_l1_addr"),
        get_named_compile_time_arg_val("table_idx_l1_addr"),
        get_named_compile_time_arg_val("index_l1_addr")>;

    using DRAMArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1_dram"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("bank_id"),
        get_named_compile_time_arg_val("vc"),
        get_named_compile_time_arg_val("meta_l1_addr"),
        get_named_compile_time_arg_val("cb_in1_dram_size_bytes"),
        get_named_compile_time_arg_val("noc_max_page_size"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("pipeline_sem_id"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y"),
        get_named_compile_time_arg_val("cores_per_bank"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("is_dram_l1_addr"),
        get_named_compile_time_arg_val("table_idx_l1_addr"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("cb_fmt"),
        get_named_compile_time_arg_val("fmt_dram_addr"),
        get_named_compile_time_arg_val("fmt_per_expert_bytes"),
        get_named_compile_time_arg_val("fmt_per_core_bytes"),
        get_named_compile_time_arg_val("accum_experts")>;

#elif defined(COMPILE_FOR_BRISC)
    using SRAMArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::WriterCTArgs;
    using DRAMArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::WriterCTArgs;

#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();

    using SRAMArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("out_w"),
        get_named_compile_time_arg_val("sram_fmt_l1_addr"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("is_dram_l1_addr"),
        get_named_compile_time_arg_val("table_idx_l1_addr"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("accum_experts")>;

    using DRAMArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1_dram"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("dram_fmt_l1_addr"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("is_dram_l1_addr"),
        get_named_compile_time_arg_val("table_idx_l1_addr"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("cb_fmt"),
        get_named_compile_time_arg_val("accum_experts")>;
#endif

    constexpr bool sram_active = get_named_compile_time_arg_val("sram_active") != 0;
    constexpr bool dram_active = get_named_compile_time_arg_val("dram_active") != 0;
    deepseek_b1_ops::MatmulExpertCompressedSRAM::Op<SRAMArgs, sram_active> sram_mm;
    deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<DRAMArgs, dram_active> dram_mm;
    sram_mm();
    dram_mm();
}
