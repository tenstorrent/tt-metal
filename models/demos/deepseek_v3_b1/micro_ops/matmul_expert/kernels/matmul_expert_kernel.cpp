// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Expert SRAM Matmul with Compressed Weights
//
// Multi-expert variant: each core holds a [num_experts, num_tiles] fmt table in L1.
// Expert index is read at runtime from cb_index (HEIGHT_SHARDED uint16 tensor,
// same format as DRAMStreamingExpertsMatmul). Always uses the runtime impl.
//
// See unified_kernels/matmul_expert_compressed.hpp for the kernel logic.

#include "../../../unified_kernels/matmul_expert_compressed.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    using CTArgs = deepseek_b1_ops::MatmulExpertCompressed::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("out_w"),
        get_named_compile_time_arg_val("cb_in0_num_pages"),
        get_named_compile_time_arg_val("fmt_l1_addr")>;
    deepseek_b1_ops::MatmulExpertCompressed::reader<CTArgs>();

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: no-op

#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();
    using CTArgs = deepseek_b1_ops::MatmulExpertCompressed::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("out_w"),
        get_named_compile_time_arg_val("fmt_l1_addr")>;
    deepseek_b1_ops::MatmulExpertCompressed::compute<CTArgs>();
#endif
}
