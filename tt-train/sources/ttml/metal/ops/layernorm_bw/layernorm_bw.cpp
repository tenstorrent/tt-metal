// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw.hpp"

#include <core/ttnn_all_includes.hpp>

#include "core/compute_kernel_config.hpp"
#include "device/layernorm_bw_device_operation.hpp"

namespace ttml::metal::ops::layernorm_bw {

std::vector<std::optional<ttnn::Tensor>> LayerNormBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& mean_tensor,
    const ttnn::Tensor& rstd_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    auto device_op = ttnn::prim::ttml_layernorm_bw;

    // Save original shape for reshaping outputs back
    const auto& original_shape = input_tensor.logical_shape();

    // Flatten all inputs to 2D: (batch*...*seq, hidden_size)
    // This makes the kernel dimension-agnostic
    uint32_t total_rows = 1;
    for (size_t i = 0; i < original_shape.rank() - 1; ++i) {
        total_rows *= original_shape[i];
    }
    uint32_t hidden_size = original_shape[-1];

    // Reshape to 2D
    auto input_2d = ttnn::reshape(input_tensor, ttnn::Shape({total_rows, hidden_size}));
    auto mean_2d = ttnn::reshape(mean_tensor, ttnn::Shape({total_rows, 1}));
    auto rstd_2d = ttnn::reshape(rstd_tensor, ttnn::Shape({total_rows, 1}));
    auto dy_2d = ttnn::reshape(dL_dout_tensor, ttnn::Shape({total_rows, hidden_size}));

    // Call the device operation with 2D tensors
    // Returns: [dx, dgamma_components, dbeta_components]
    auto result = device_op(input_2d, gamma_tensor, mean_2d, rstd_2d, dy_2d);

    // Reshape dx back to original shape
    auto dx = ttnn::reshape(result[0], original_shape);

    // Reshape gradient components back to original shape for reduction
    auto dgamma_components = ttnn::reshape(result[1], original_shape);
    auto dbeta_components = ttnn::reshape(result[2], original_shape);

    // dL_dgamma and dL_dbeta require sum over all batch dimensions
    // Sum over all dimensions except the last one
    ttnn::SmallVector<int> reduce_dims;
    for (int i = 0; i < static_cast<int>(original_shape.rank()) - 1; ++i) {
        reduce_dims.push_back(i);
    }

    return {
        dx,
        ttnn::sum(
            dgamma_components,
            reduce_dims,
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise()),
        ttnn::sum(
            dbeta_components,
            reduce_dims,
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /*compute_kernel_config */ core::ComputeKernelConfig::precise())};
}

}  // namespace ttml::metal::ops::layernorm_bw
