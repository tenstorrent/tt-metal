// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw.hpp"

#include "device/rmsnorm_fw_device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_fw {

std::vector<std::optional<ttnn::Tensor>> RMSNormForwardOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& gamma_tensor, bool return_intermediates, float epsilon) {
    auto result = ttnn::prim::ttml_rmsnorm_fw(input_tensor, gamma_tensor, return_intermediates, epsilon);

    if (result.size() == 1U) {
        return {result[0], std::nullopt};
    }

    return {result[0], result[1]};
}

}  // namespace ttml::metal::ops::rmsnorm_fw
