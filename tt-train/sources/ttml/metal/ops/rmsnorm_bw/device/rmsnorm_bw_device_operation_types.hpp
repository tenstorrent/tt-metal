// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

// Attributes for the backward operation (add more if needed)
struct operation_attributes_t {
    float epsilon = 1e-6F;

    static constexpr auto attribute_names = std::forward_as_tuple("epsilon");
    auto attribute_values() const {
        return std::forward_as_tuple(epsilon);
    }
};

// Tensors required for backward
struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor rms;
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input", "gamma", "rms", "dL_dout", "preallocated_da", "preallocated_dgamma_components");
    auto attribute_values() const {
        return std::forward_as_tuple(input, gamma, rms, dL_dout, preallocated_da, preallocated_dgamma_components);
    }
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::rmsnorm_bw::device
