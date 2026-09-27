// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/matmul.hpp"

#include <type_traits>

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t k_blocks = get_compile_time_arg_val(0);
    constexpr bool l1_acc = get_compile_time_arg_val(1);
    constexpr uint32_t post_op = get_compile_time_arg_val(2);  // See test_matmul_helpers.py for post-op cases.
    constexpr bool static_shape = get_compile_time_arg_val(3);
    constexpr uint32_t batches = get_compile_time_arg_val(4);
    constexpr bool same_cb = get_compile_time_arg_val(5);
    constexpr uint32_t output_blocks = get_compile_time_arg_val(6);
    constexpr bool activation_on_math = get_compile_time_arg_val(7);
    constexpr bool fast_approx = get_compile_time_arg_val(8);
    const auto shape = [] {
        if constexpr (static_shape) {
            return StaticMatmulShape<2, 2, 2, 2, 1, k_blocks, batches>{};
        } else {
            return MatmulShape::of(2, 2, 2, 2, 1, k_blocks, batches);
        }
    }();
    constexpr bool with_bias =
        post_op == 1 || post_op == 4 || post_op == 7 || post_op == 9 || post_op == 10 || post_op == 12;
    constexpr uint32_t bias_tiles = (post_op == 10 ? 16 : 4) * output_blocks;
    constexpr MatmulBiasMode bias_mode = post_op == 9    ? MatmulBiasMode::ColumnIndexed
                                         : post_op == 10 ? MatmulBiasMode::FullBlockElementwise
                                                         : MatmulBiasMode::RowBroadcast;
    constexpr uint32_t in0 = 0, in1 = 1, partials = 2, bias = 3, out = 16;
    DataflowBuffer a(in0), b(in1), bias_buf(bias);
    constexpr uint32_t partials_cb_id = same_cb ? out : partials;
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_block_init(in0, in1, false, 2, 2, 1);
    constexpr KernelActivation activation = post_op == 3 || post_op == 4     ? KernelActivation::RELU6
                                            : post_op == 8                   ? KernelActivation::GELU_TANH
                                            : post_op == 11 || post_op == 12 ? KernelActivation::MISH
                                            : post_op == 13                  ? KernelActivation::SQRT
                                            : post_op == 14                  ? KernelActivation::LEAKY_RELU
                                            : post_op == 15                  ? KernelActivation::ELU
                                            : post_op == 16                  ? KernelActivation::EXP
                                            : post_op == 17                  ? KernelActivation::RECIP
                                                                             : KernelActivation::NONE;
    static_assert(!MatmulActivation<>::enabled);
    constexpr bool pack_relu = post_op == 2 || post_op == 7;
    constexpr uint32_t param0 = post_op == 14 ? 0x3e000000u : post_op == 15 ? 0x3f800000u : uint32_t(fast_approx);
    using Activation = std::conditional_t<
        activation_on_math,
        MatmulActivation<activation, param0, 0, 0, pack_relu, ActivationThread::Math>,
        MatmulActivation<activation, param0, 0, 0, pack_relu>>;
    Activation::init();

    // Inputs reside in sharded L1 tensors, ordered by K block by the test.
    a.reserve_back(4 * k_blocks * batches * output_blocks);
    a.push_back(4 * k_blocks * batches * output_blocks);
    b.reserve_back(4 * k_blocks * batches * output_blocks);
    b.push_back(4 * k_blocks * batches * output_blocks);
    if constexpr (with_bias) {
        bias_buf.reserve_back(bias_tiles);
        bias_buf.push_back(bias_tiles);
    }
    for (uint32_t block = 0; block < output_blocks; ++block) {
        // Startup configures the first block. Repeated blocks exercise the result's
        // format restoration without matmul's entry reconfiguration masking it.
        constexpr auto reconfig = output_blocks > 1 ? matmul_config::DataFormatReconfig::None
                                                    : matmul_config::DataFormatReconfig::InputAndOutput;
        const auto result = compute_kernel_lib::matmul<
            false,
            l1_acc,
            matmul_config::InitMode::Initialize,
            matmul_config::InputPolicy::WaitAndPopPerKBlock,
            reconfig,
            Activation,
            with_bias,
            bias_mode>(in0, in1, out, partials_cb_id, shape, NoPreKBlock{}, MatmulBias{bias, bias_tiles, block * 4});
        if (block + 1 < output_blocks) {
            result.restore_input_formats();
        }
    }
    if constexpr (with_bias) {
        bias_buf.pop_front(bias_tiles);
    }
}
