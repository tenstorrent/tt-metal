// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
    // Host-drawn entropy, spread over the cores by the program factory. Engaged iff SR is enabled.
    std::optional<uint32_t> stochastic_rounding_seed{std::nullopt};

    // Only these fields affect the compiled program. All optimizer scalars and the stochastic-rounding seed are
    // refreshed by override_runtime_arguments and must not fragment the program cache.
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

    static constexpr auto attribute_names =
        std::forward_as_tuple("param", "grad", "exp_avg", "exp_avg_sq", "max_exp_avg_sq");
    auto attribute_values() const {
        return std::forward_as_tuple(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq);
    }
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw::device
