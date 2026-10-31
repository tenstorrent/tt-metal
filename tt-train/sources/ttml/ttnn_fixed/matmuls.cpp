// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmuls.hpp"

#include "core/compute_kernel_config.hpp"

namespace ttml::ttnn_fixed {
ttnn::Tensor matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    std::optional<ttnn::Tensor> output_tensor) {
    const auto grid_size = a.device()->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(grid_size.x, grid_size.y);

    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */
        ttml::core::ComputeKernelConfig::matmul(),
        /* core_grid */ core_grid,
        /* output_tile */ std::nullopt,
        /* optional_output_tensor */ std::move(output_tensor));
}

std::pair<std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> matmul_backward(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& out_grad,
    bool transpose_a,
    bool transpose_b,
    MatmulBackwardGrads grads) {
    // Flatten dL/dout by its own last dim: with transpose_a, a's rows are op(a)'s columns, not dL/dout's rows.
    const uint32_t grad_cols = out_grad.logical_shape()[-1];
    const auto flat_grad =
        ttnn::reshape(out_grad, ttnn::Shape({static_cast<uint32_t>(out_grad.logical_volume() / grad_cols), grad_cols}));

    std::optional<ttnn::Tensor> a_grad;
    if (grads.a) {
        // dA_eff = dL/dout @ op(b)^T. With transpose_a, a holds A^T, so dA = dA_eff^T = op(b) @ (dL/dout)^T.
        const auto flat_a_grad = transpose_a ? ttnn_fixed::matmul(b, flat_grad, transpose_b, true)
                                             : ttnn_fixed::matmul(flat_grad, b, false, !transpose_b);
        a_grad = ttnn::reshape(flat_a_grad, a.logical_shape());
    }

    std::optional<ttnn::Tensor> b_grad;
    if (grads.b) {
        const uint32_t a_cols = a.logical_shape()[-1];
        const auto flat_a = ttnn::reshape(a, ttnn::Shape({static_cast<uint32_t>(a.logical_volume() / a_cols), a_cols}));
        // dB_eff = op(a)^T @ dL/dout. With transpose_b, b holds B^T, so dB = dB_eff^T = (dL/dout)^T @ op(a).
        const auto flat_b_grad = transpose_b ? ttnn_fixed::matmul(flat_grad, flat_a, true, transpose_a)
                                             : ttnn_fixed::matmul(flat_a, flat_grad, !transpose_a, false);
        b_grad = ttnn::reshape(flat_b_grad, b.logical_shape());
    }
    return {a_grad, b_grad};
}
}  // namespace ttml::ttnn_fixed
