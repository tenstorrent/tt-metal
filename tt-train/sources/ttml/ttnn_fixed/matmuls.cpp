// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmuls.hpp"

#include "core/compute_kernel_config.hpp"

namespace ttml::ttnn_fixed {
tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b, bool transpose_a, bool transpose_b) {
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
        /* output_tile */ std::nullopt);
}

std::pair<tt::tt_metal::Tensor, tt::tt_metal::Tensor> matmul_backward(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    const tt::tt_metal::Tensor& out_grad,
    bool transpose_a,
    bool transpose_b) {
    auto a_shape = a.logical_shape();
    auto b_shape = b.logical_shape();
    auto grad_shape = out_grad.logical_shape();

    auto volume_without_features_a = a.logical_volume() / static_cast<uint64_t>(a.logical_shape()[-1]);
    auto reshaped_a =
        ttnn::reshape(a, ttnn::Shape({static_cast<uint32_t>(volume_without_features_a), a.logical_shape()[-1]}));
    auto reshaped_grad = ttnn::reshape(
        out_grad, ttnn::Shape({static_cast<uint32_t>(volume_without_features_a), out_grad.logical_shape()[-1]}));

    ttnn::Tensor reshaped_a_grad;
    ttnn::Tensor reshaped_b_grad;

    if (!transpose_a) {
        // A was used as is.
        // grad_A = reshaped_grad * ( (transpose_b ? B^T : B) )^T.
        // If transpose_b is false: (B)^T = B^T, if true: (B^T)^T = B.
        reshaped_a_grad = matmul(reshaped_grad, b, false, !transpose_b);
    } else {
        // A was transposed in the forward pass (i.e. we used A^T).
        // Compute dA_eff = reshaped_grad * ( (transpose_b ? B^T : B) )^T.
        // Then grad_A = (dA_eff)^T = ( (transpose_b ? B^T : B) ) * reshaped_grad^T.
        if (!transpose_b)
            reshaped_a_grad = matmul(b, reshaped_grad, false, true);  // B as is, reshaped_grad transposed.
        else
            reshaped_a_grad = matmul(b, reshaped_grad, true, true);  // B transposed, reshaped_grad transposed.
    }

    if (!transpose_b) {
        // B was used as is.
        // grad_B = ( (transpose_a ? A^T : A) )^T * d_out.
        // If transpose_a is false: (A)^T = A^T, if true: (A^T)^T = A.
        reshaped_b_grad = matmul(a, reshaped_grad, !transpose_a, false);
    } else {
        // B was transposed in the forward pass (i.e. we used B^T).
        // Compute dB_eff = ( (transpose_a ? A^T : A) )^T * d_out,
        // then grad_B = (dB_eff)^T = d_out^T * (transpose_a ? A^T : A).
        reshaped_b_grad = matmul(reshaped_grad, a, true, transpose_a);
    }
    auto a_grad = ttnn::reshape(reshaped_a_grad, a_shape);
    auto b_grad = ttnn::reshape(reshaped_b_grad, b_shape);
    return {a_grad, b_grad};
}
}  // namespace ttml::ttnn_fixed
