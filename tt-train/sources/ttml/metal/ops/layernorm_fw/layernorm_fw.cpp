// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fw.hpp"

#include <core/ttnn_all_includes.hpp>

#include "core/compute_kernel_config.hpp"
#include "device/layernorm_fw_device_operation.hpp"

namespace ttml::metal::ops::layernorm_fw {

std::vector<std::optional<ttnn::Tensor>> LayerNormForwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& beta_tensor,
    float epsilon,
    bool return_mean_rstd) {
    auto device_op = ttnn::prim::ttml_layernorm_fw;

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

    // Call the device operation with 2D tensors
    // Returns: [output, mean (optional), rstd (optional)]
    auto result = device_op(input_2d, gamma_tensor, beta_tensor, epsilon, return_mean_rstd);

    // Reshape output back to original shape
    auto output = ttnn::reshape(result[0].value(), original_shape);

    std::vector<std::optional<ttnn::Tensor>> return_tensors;
    return_tensors.push_back(output);

    // If mean and rstd were requested, reshape them as well
    if (return_mean_rstd) {
        // Mean and rstd have shape [total_rows, 1] - reshape to [B, 1, S, 1]
        ttnn::SmallVector<uint32_t> mean_rstd_shape;
        for (size_t i = 0; i < original_shape.rank() - 1; ++i) {
            mean_rstd_shape.push_back(original_shape[i]);
        }
        mean_rstd_shape.push_back(1);  // Last dimension is 1

        return_tensors.push_back(ttnn::reshape(result[1].value(), ttnn::Shape(mean_rstd_shape)));
        return_tensors.push_back(ttnn::reshape(result[2].value(), ttnn::Shape(mean_rstd_shape)));
    } else {
        return_tensors.push_back(std::nullopt);
        return_tensors.push_back(std::nullopt);
    }

    return return_tensors;
}

}  // namespace ttml::metal::ops::layernorm_fw
