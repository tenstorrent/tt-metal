// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct FrobeniusNormalizeAttributes {
    float epsilon{1e-7F};

    static constexpr auto attribute_names = std::forward_as_tuple();
    auto attribute_values() const {
        return std::forward_as_tuple();
    }
};

struct FrobeniusNormalizeTensorArgs {
    const ttnn::Tensor& input;
    std::optional<ttnn::Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input_dtype", "input_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(input.dtype(), std::cref(input.logical_shape()));
    }
};

using FrobeniusNormalizeTensorReturn = std::vector<ttnn::Tensor>;

using FrobeniusNormalizeSpecReturn = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::frobenius_normalize::device
