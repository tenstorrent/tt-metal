// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// DRAM Streaming Matmul with Compressed Weights — Multi-Expert
//
// Extends DRAMStreamingMatmulCompressed to process multiple experts per dispatch.
// Expert byte offsets are loaded from a pre-populated L1 table instead of using fixed stride.
// BRISC: no-op. TRISC: compressed matmul, all experts sequentially.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/dram_streaming_experts_matmul_compressed.hpp"

struct Core {
    static constexpr bool is_active_core = get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingExpertsMatmulCompressed::ReaderCTArgs<
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
        0,  // dram_start_offset — always 0 (cores_per_bank=1)
        0,  // pipeline_sem_id  — unused (cores_per_bank=1)
        0,  // next_core_noc_x  — unused
        0,  // next_core_noc_y  — unused
        get_named_compile_time_arg_val("selected_experts_k"),
        get_named_compile_time_arg_val("expert_offsets_l1_addr"),
        0,   // enable_indexing  — always 0 (experts in order 0..k-1)
        0,   // cb_index         — unused
        0>;  // index_offset     — unused

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");

    if constexpr (Core::is_active_core) {
        unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
    }

#elif defined(COMPILE_FOR_BRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingExpertsMatmulCompressed::WriterCTArgs;

#elif defined(COMPILE_FOR_TRISC)
    using CTArgs = deepseek_b1_ops::DRAMStreamingExpertsMatmulCompressed::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("selected_experts_k"),
        get_named_compile_time_arg_val("fmt_l1_addr")>;

    deepseek_compute_kernel_init();
#endif

    deepseek_b1_ops::DRAMStreamingExpertsMatmulCompressed::Op<CTArgs, Core::is_active_core> op;
    op();
}
