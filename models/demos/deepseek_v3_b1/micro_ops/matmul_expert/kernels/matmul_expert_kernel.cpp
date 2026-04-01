// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Expert Matmul with Compressed Weights — SRAM and DRAM paths.
//
// Dispatched at compile time via the `is_dram` named arg (0 = SRAM, 1 = DRAM).
// Both paths share this kernel; the unused path is compiled-in but never
// executed due to if constexpr. All arg names for both paths must be present
// in the compile-time arg map (Python op passes 0 for unused args).
//
// See unified_kernels/matmul_expert_compressed.hpp for the kernel logic.

#include "../../../unified_kernels/matmul_expert_compressed.hpp"

void kernel_main() {
    constexpr bool is_dram = get_named_compile_time_arg_val("is_dram") == 1;

#if defined(COMPILE_FOR_NCRISC)
    if constexpr (!is_dram) {
        using CTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ReaderCTArgs<
            get_named_compile_time_arg_val("cb_in0"),
            get_named_compile_time_arg_val("cb_in1"),
            get_named_compile_time_arg_val("cb_out"),
            get_named_compile_time_arg_val("cb_index"),
            get_named_compile_time_arg_val("num_tiles_k"),
            get_named_compile_time_arg_val("out_w"),
            get_named_compile_time_arg_val("cb_in0_num_pages"),
            get_named_compile_time_arg_val("fmt_l1_addr")>;
        deepseek_b1_ops::MatmulExpertCompressedSRAM::reader<CTArgs>();
    } else {
        using CTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
            get_named_compile_time_arg_val("cb_in0"),
            get_named_compile_time_arg_val("cb_in1"),
            get_named_compile_time_arg_val("cb_out"),
            get_named_compile_time_arg_val("cb_index"),
            get_named_compile_time_arg_val("num_tiles_k"),
            get_named_compile_time_arg_val("subblock_k"),
            get_named_compile_time_arg_val("num_subblocks_k"),
            get_named_compile_time_arg_val("per_core_n"),
            get_named_compile_time_arg_val("bank_id"),
            get_named_compile_time_arg_val("vc"),
            get_named_compile_time_arg_val("meta_l1_addr"),
            get_named_compile_time_arg_val("cb_in1_size_bytes"),
            get_named_compile_time_arg_val("noc_max_page_size"),
            get_named_compile_time_arg_val("core_in_bank_idx"),
            get_named_compile_time_arg_val("pipeline_sem_id"),
            get_named_compile_time_arg_val("next_core_noc_x"),
            get_named_compile_time_arg_val("next_core_noc_y")>;
        deepseek_b1_ops::MatmulExpertCompressedDRAM::reader<CTArgs>();
    }

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: no-op

#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    if constexpr (!is_dram) {
        using CTArgs = deepseek_b1_ops::MatmulExpertCompressedSRAM::ComputeCTArgs<
            get_named_compile_time_arg_val("cb_in0"),
            get_named_compile_time_arg_val("cb_in1"),
            get_named_compile_time_arg_val("cb_out"),
            get_named_compile_time_arg_val("cb_index"),
            get_named_compile_time_arg_val("num_tiles_k"),
            get_named_compile_time_arg_val("out_w"),
            get_named_compile_time_arg_val("fmt_l1_addr")>;
        deepseek_b1_ops::MatmulExpertCompressedSRAM::compute<CTArgs>();
    } else {
        using CTArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
            get_named_compile_time_arg_val("cb_in0"),
            get_named_compile_time_arg_val("cb_in1"),
            get_named_compile_time_arg_val("cb_out"),
            get_named_compile_time_arg_val("cb_index"),
            get_named_compile_time_arg_val("num_tiles_k"),
            get_named_compile_time_arg_val("subblock_k"),
            get_named_compile_time_arg_val("num_subblocks_k"),
            get_named_compile_time_arg_val("per_core_n"),
            get_named_compile_time_arg_val("fmt_l1_addr")>;
        deepseek_b1_ops::MatmulExpertCompressedDRAM::compute<CTArgs>();
    }
#endif
}
