// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Router top-k for one token, compute. The sigmoid + bias pass and the normalize / scale tail are the
// moe_grouped_topk blocks themselves (same CB ids, formats and compute config), so scores and weights are
// bit-equal to that op. The top-k between them runs on the writer core (scalar, router_topk_writer.cpp)
// instead of the tile sort network, which works on 32 token rows where decode has one.
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp"

void kernel_main() {
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t cb_in_scores = 0, cb_in_bias = 1, cb_out_weights = 2, cb_sigmoid = 4, cb_biased = 5;
    constexpr uint32_t cb_reduce_intermediate = 15, cb_ones = 17, cb_eps = 18, cb_scale = 19, cb_normalized = 20;
    constexpr uint32_t cb_recip = 21, cb_gathered = 22;

    compute_kernel_hw_startup(cb_in_scores, cb_in_bias, cb_biased);
    blocks::apply_score_func_add_bias<blocks::SCORE_FUNC_SIGMOID, false>(
        cb_in_scores, cb_in_bias, cb_sigmoid, cb_biased, width_tiles);
    blocks::normalize_scores(cb_gathered, cb_ones, cb_reduce_intermediate, cb_recip, cb_eps, cb_normalized);
    blocks::scale(cb_normalized, cb_scale, cb_out_weights);
}
