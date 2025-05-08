// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_fw {

struct RMSNormForwardOperation {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& gamma_tensor,
        bool return_intermediates = true,
        float epsilon = 1e-6F);
};

}  // namespace ttml::metal::ops::rmsnorm_fw
