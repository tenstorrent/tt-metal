// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert Matmul with Compressed Weights — Hybrid SRAM + DRAM kernel.
//
// Both sram_mm() and dram_mm() are called. Each loops over the index array,
// checks bit 15 of the index value (1=SRAM, 0=DRAM), and skips non-matching experts.
//
// SRAM path uses cb_in1 (CB 1) for sharded B data.
// DRAM path uses cb_in1_dram (CB 4) for streaming buffer.
// Sharded buffer setup for cb_in0, cb_in1, cb_index done here before Op calls.

#include "../../../unified_kernels/matmul_expert_compressed_sram.hpp"
#include "../../../unified_kernels/matmul_expert_compressed_dram.hpp"

void kernel_main() {
// ============================================================================
// Define CTArgs per RISC + shared NCRISC setup
// ============================================================================
#if defined(COMPILE_FOR_NCRISC)
    // Shared setup: sharded buffers for A and index.
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");
    constexpr uint32_t accum_experts = get_named_compile_time_arg_val("accum_experts");
    constexpr uint32_t num_active_experts = get_named_compile_time_arg_val("num_active_experts");
    constexpr uint32_t cb_in0_pages = accum_experts ? num_tiles_k * num_active_experts : num_tiles_k;
    unified_kernels::setup_sharded_buffer(cb_in0, cb_in0_pages);
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
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("sram_k_per_core"),
        get_named_compile_time_arg_val("sram_k_offset")>;

    using DRAMArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1_dram"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("subblock_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("bank_id"),
        get_named_compile_time_arg_val("vc"),
        get_named_compile_time_arg_val("expert_offsets_l1_addr"),
        get_named_compile_time_arg_val("block_sizes_l1_addr"),
        get_named_compile_time_arg_val("cb_in1_dram_size_bytes"),
        get_named_compile_time_arg_val("noc_max_page_size"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("pipeline_sem_id"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y"),
        get_named_compile_time_arg_val("cores_per_dram_bank"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("cb_fmt_dram"),
        get_named_compile_time_arg_val("fmt_dram_addr"),
        get_named_compile_time_arg_val("fmt_per_expert_bytes"),
        get_named_compile_time_arg_val("fmt_per_core_bytes"),
        get_named_compile_time_arg_val("fmt_cb_l1_addr"),
        get_named_compile_time_arg_val("fmt_cb_page_size"),
        get_named_compile_time_arg_val("fmt_sem_addr_0"),
        get_named_compile_time_arg_val("fmt_sem_addr_1"),
        get_named_compile_time_arg_val("accum_experts"),
        get_named_compile_time_arg_val("index_offset"),
        get_named_compile_time_arg_val("k_parallel_per_bank"),
        get_named_compile_time_arg_val("k_slice_idx"),
        get_named_compile_time_arg_val("num_subblocks_k_local"),
        get_named_compile_time_arg_val("partial_sem_addr")>;

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
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("sram_base_addrs_l1_addr"),
        get_named_compile_time_arg_val("sram_meta_words_per_expert"),
        get_named_compile_time_arg_val("in0_page_size"),
        get_named_compile_time_arg_val("accum_experts"),
        get_named_compile_time_arg_val("sram_k_per_core"),
        get_named_compile_time_arg_val("sram_k_offset"),
        get_named_compile_time_arg_val("cb_out_sram")>;

    using DRAMArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1_dram"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("subblock_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("dram_fmt_l1_addr"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("cb_fmt_dram"),
        get_named_compile_time_arg_val("dram_meta_words_per_block"),
        get_named_compile_time_arg_val("in0_page_size"),
        get_named_compile_time_arg_val("fmt_cb_l1_addr"),
        get_named_compile_time_arg_val("fmt_cb_page_size"),
        get_named_compile_time_arg_val("fmt_sem_addr_0"),
        get_named_compile_time_arg_val("fmt_sem_addr_1"),
        get_named_compile_time_arg_val("accum_experts"),
        get_named_compile_time_arg_val("dram_fuse_silu"),
        get_named_compile_time_arg_val("index_offset"),
        get_named_compile_time_arg_val("k_parallel_per_bank"),
        get_named_compile_time_arg_val("k_slice_idx"),
        get_named_compile_time_arg_val("num_subblocks_k_local"),
        get_named_compile_time_arg_val("partial_sem_addr")>;
#endif

    constexpr bool sram_active = get_named_compile_time_arg_val("sram_active") != 0;
    constexpr bool dram_active = get_named_compile_time_arg_val("dram_active") != 0;
    // When both paths run on the same core, SRAM runs first and must NOT pop
    // cb_in0/cb_index — DRAM still needs them.
    constexpr bool sram_pop_in0 = sram_active && !dram_active;
    constexpr bool sram_pop_index = sram_active && !dram_active;
    deepseek_b1_ops::MatmulExpertCompressedSRAM::
        Op<SRAMArgs, sram_active, sram_pop_in0, /*pop_in1=*/true, sram_pop_index>
            sram_mm;
    deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<DRAMArgs, dram_active> dram_mm;
    sram_mm();
    dram_mm();
}
