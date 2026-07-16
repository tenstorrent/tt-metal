// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <tuple>

#include "metal/common/const_utils.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

struct KSplitGramMatmulParams {
    ttml::metal::OutputMode output_mode = ttml::metal::OutputMode::UpperTriangle;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mode", "math_fidelity");
    auto attribute_values() const {
        return std::forward_as_tuple(this->output_mode, this->math_fidelity);
    }
};

struct KSplitGramMatmulInputs {
    ttnn::Tensor input_tensor;
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple("input_logical_shape");
    auto attribute_values() const {
        return std::forward_as_tuple(this->input_tensor.logical_shape());
    }
};

using operation_attributes_t = KSplitGramMatmulParams;
using tensor_args_t = KSplitGramMatmulInputs;

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
