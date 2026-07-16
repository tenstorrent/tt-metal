// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>
#include <tuple>

#include "metal/common/const_utils.hpp"

namespace ttml::metal::optimizers::adamw::device {

struct operation_attributes_t {
    float lr{};
    float beta1{};
    float beta2{};
    float beta1_pow{};
    float beta2_pow{};
    float epsilon{};
    float weight_decay{};
    bool amsgrad{false};
    StochasticRounding stochastic_rounding{StochasticRounding::Disabled};

    static constexpr auto attribute_names = std::forward_as_tuple("amsgrad", "stochastic_rounding");
    auto attribute_values() const {
        return std::forward_as_tuple(amsgrad, stochastic_rounding);
    }
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;

    const ttnn::Tensor& exp_avg;
    const ttnn::Tensor& exp_avg_sq;
    std::optional<ttnn::Tensor> max_exp_avg_sq = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "max_exp_avg_sq_initialized",
        "param_dtype",
        "param_logical_shape",
        "param",
        "grad",
        "exp_avg",
        "exp_avg_sq",
        "max_exp_avg_sq");
    auto attribute_values() const {
        return std::make_tuple(
            max_exp_avg_sq.has_value(),
            param.dtype(),
            std::cref(param.logical_shape()),
            std::cref(param),
            std::cref(grad),
            std::cref(exp_avg),
            std::cref(exp_avg_sq),
            max_exp_avg_sq);
    }
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = ttnn::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw::device
