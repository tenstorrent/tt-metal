// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of deepseek_v3_b1 EltwiseMul's compute with the expert scale (enable_scalar, the path moe_kernel.cpp,
// decoder_block_kernel.cpp and moe_routed_expert_kernel.cpp run): the Op of models/demos/deepseek_v3_b1/unified_kernels/
// eltwise_mul.hpp itself, after the fused kernels' deepseek_compute_kernel_init, run TWIN_ITERS times.
// Compile args: cb_in0, cb_in1, cb_out, cb_scalar, num_tiles, num_experts, fp32_dest_acc_en, iterations.
#include <cstdint>
#include "../../../../models/demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../models/demos/deepseek_v3_b1/unified_kernels/eltwise_mul.hpp"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scalar = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts = get_compile_time_arg_val(5);
    constexpr uint32_t fp32_dest_acc_en = get_compile_time_arg_val(6);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(7);
    using CT = deepseek_b1_ops::EltwiseMul::ComputeCTArgs<
        cb_in0,
        cb_in1,
        cb_out,
        num_tiles,
        cb_in0,
        num_tiles,
        cb_in1,
        num_tiles,
        cb_scalar,
        fp32_dest_acc_en,
        1,
        num_experts>;

    deepseek_compute_kernel_init();
    for (uint32_t it = 0; it < twin_iters; ++it) {
        deepseek_b1_ops::EltwiseMul::Op<CT, true, true> mul_op;
        mul_op();
    }
}
