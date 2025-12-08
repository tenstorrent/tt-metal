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
    // Returns: [output, mean (optional), rstd (optional)]
    return ttnn::prim::ttml_layernorm_fw(input_tensor, gamma_tensor, beta_tensor, epsilon, return_mean_rstd);
}

}  // namespace ttml::metal::ops::layernorm_fw
