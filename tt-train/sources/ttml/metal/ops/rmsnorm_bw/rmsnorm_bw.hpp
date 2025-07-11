// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw {

struct RMSNormBackwardOperation {
    static std::vector<std::optional<ttnn::Tensor>> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& gamma_tensor,
        const ttnn::Tensor& rms_tensor,  // intermediate from fw
        const ttnn::Tensor& dL_dout_tensor);
};

}  // namespace ttml::metal::ops::rmsnorm_bw
