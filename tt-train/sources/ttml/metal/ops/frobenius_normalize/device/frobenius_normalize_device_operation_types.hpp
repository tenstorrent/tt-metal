// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct FrobeniusNormalizeAttributes {
    float epsilon{1e-7F};
};

struct FrobeniusNormalizeTensorArgs {
    const ttnn::Tensor& input;
    std::optional<ttnn::Tensor> preallocated_output;
};

using FrobeniusNormalizeTensorReturn = std::vector<ttnn::Tensor>;

using FrobeniusNormalizeSpecReturn = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::frobenius_normalize::device
