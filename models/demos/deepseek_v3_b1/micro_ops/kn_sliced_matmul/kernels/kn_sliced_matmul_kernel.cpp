// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// KNSlicedMatmul standalone kernel
//
// Computes: output[1, out_w] = act[k_offset..k_offset+k_per_core] @ weights[k_per_core, out_w]
//
// NCRISC: setup_sharded_buffer for activation and weight CBs
// BRISC: No-op
// TRISC: KNSlicedMatmul compute

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/kn_sliced_matmul.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t act_cb = get_named_compile_time_arg_val("act_cb");
    constexpr uint32_t act_num_pages = get_named_compile_time_arg_val("act_num_pages");
    constexpr uint32_t weights_cb = get_named_compile_time_arg_val("weights_cb");
    constexpr uint32_t weights_num_pages = get_named_compile_time_arg_val("weights_num_pages");

    unified_kernels::setup_sharded_buffer(act_cb, act_num_pages);
    unified_kernels::setup_sharded_buffer(weights_cb, weights_num_pages);

    using KNSlicedMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::ReaderCTArgs;
    deepseek_b1_ops::KNSlicedMatmul::ReaderArgs matmul_args{};
    deepseek_b1_ops::KNSlicedMatmul::Op<KNSlicedMatmulCTArgs, true> matmul;

#elif defined(COMPILE_FOR_BRISC)
    using KNSlicedMatmulCTArgs = deepseek_b1_ops::KNSlicedMatmul::WriterCTArgs;
    deepseek_b1_ops::KNSlicedMatmul::WriterArgs matmul_args{};
    deepseek_b1_ops::KNSlicedMatmul::Op<KNSlicedMatmulCTArgs, true> matmul;

#elif defined(COMPILE_FOR_TRISC)
    using KNSlicedMatmulCTArgs =
        deepseek_b1_ops::KNSlicedMatmul::ComputeCTArgs<get_named_compile_time_arg_val("out_w")>;
    deepseek_b1_ops::KNSlicedMatmul::ComputeArgs matmul_args{
        get_named_compile_time_arg_val("act_cb"),
        get_named_compile_time_arg_val("weights_cb"),
        get_named_compile_time_arg_val("out_cb"),
        get_named_compile_time_arg_val("k_offset"),
        get_named_compile_time_arg_val("k_per_core"),
        get_named_compile_time_arg_val("act_total_tiles"),
    };
    deepseek_b1_ops::KNSlicedMatmul::Op<KNSlicedMatmulCTArgs, true, /*pop_act=*/true, /*pop_weights=*/false> matmul;
    deepseek_compute_kernel_init();
#endif
    matmul(matmul_args);
}
