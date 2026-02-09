// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

// Attributes for the backward operation (add more if needed)
struct operation_attributes_t {
    static constexpr auto attribute_names = std::make_tuple();
    auto attribute_values() const {
        return std::make_tuple();
    }
};

// Tensors required for backward
struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor gamma;                                          // scale parameter (learnable weight)
    ttnn::Tensor mean;                                           // mean from forward pass
    ttnn::Tensor rstd;                                           // reciprocal std from forward pass
    ttnn::Tensor dL_dout;                                        // gradient from upstream
    std::optional<ttnn::Tensor> preallocated_dx = std::nullopt;  // dx (input gradient)
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;  // dgamma components
    std::optional<ttnn::Tensor> preallocated_dbeta_components = std::nullopt;   // dbeta components

    static constexpr auto attribute_names = std::forward_as_tuple(
        "input",
        "gamma",
        "mean",
        "rstd",
        "dL_dout",
        "preallocated_dx",
        "preallocated_dgamma_components",
        "preallocated_dbeta_components");
    auto attribute_values() const {
        return std::forward_as_tuple(
            input,
            gamma,
            mean,
            rstd,
            dL_dout,
            preallocated_dx,
            preallocated_dgamma_components,
            preallocated_dbeta_components);
    }
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::layernorm_bw::device
