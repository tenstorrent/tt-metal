// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

// Attributes for the backward operation (add more if needed)
struct LayerNormBackwardParams {};

// Tensors required for backward
struct LayerNormBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;                                          // scale parameter (learnable weight)
    ttnn::Tensor mean;                                           // mean from forward pass
    ttnn::Tensor rstd;                                           // reciprocal std from forward pass
    ttnn::Tensor dL_dout;                                        // gradient from upstream
    std::optional<ttnn::Tensor> preallocated_dx = std::nullopt;  // dx (input gradient)
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;  // dgamma components
    std::optional<ttnn::Tensor> preallocated_dbeta_components = std::nullopt;   // dbeta components

    static constexpr auto attribute_names = std::forward_as_tuple("input_dtype", "input_logical_shape");
    auto attribute_values() const {
        return std::make_tuple(input.dtype(), std::cref(input.logical_shape()));
    }
};

using operation_attributes_t = LayerNormBackwardParams;
using tensor_args_t = LayerNormBackwardInputs;

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::layernorm_bw::device
