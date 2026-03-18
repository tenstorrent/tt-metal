// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// DRAM Streaming Matmul with Compressed Weights
//
// Combines DRAM streaming (variable-size reads) with per-tile BFP format reconfig.
// Supports pipelined multi-core-per-bank mode (cores_per_bank > 1):
//   Each core reads its own portion of N columns directly from DRAM.
//   Cores sharing a bank read sequentially with semaphore handoff:
//   core 0 reads first, signals core 1 after last request, core 1 waits then reads, etc.
//   BRISC: no-op. TRISC: compressed matmul with subblock-K partial accumulation.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/dram_streaming_matmul_compressed.hpp"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmulCompressed::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("in1_tensor_addr"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("out_num_tiles"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("bank_id"),
        get_named_compile_time_arg_val("vc"),
        get_named_compile_time_arg_val("meta_l1_addr"),
        get_named_compile_time_arg_val("cb_in1_size_bytes"),
        get_named_compile_time_arg_val("noc_max_page_size"),
        get_named_compile_time_arg_val("dram_start_offset"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("pipeline_sem_id"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y")>;

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");

    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
    }

#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmulCompressed::WriterCTArgs;

#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingMatmulCompressed::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("fmt_l1_addr")>;

    deepseek_compute_kernel_init();
#endif

    deepseek_b1_ops::DRAMStreamingMatmulCompressed::Op<CTArgs, Core::is_active_core> op;
    op();
}
