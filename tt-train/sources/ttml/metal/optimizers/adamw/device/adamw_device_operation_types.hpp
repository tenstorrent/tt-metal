// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <ttnn/tensor/tensor.hpp>

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
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;

    const ttnn::Tensor& exp_avg;
    const ttnn::Tensor& exp_avg_sq;
    std::optional<ttnn::Tensor> max_exp_avg_sq = std::nullopt;

    // Bias-correction terms beta^t as single-element f32 tensors. When engaged, the
    // kernel derives `step_size` and `1 / bias_correction2` on device and the
    // `beta1_pow` / `beta2_pow` floats in operation_attributes_t are ignored. Both
    // must be supplied together. Lets a caller keep beta^t resident on device
    // instead of reading it back to host every optimizer step.
    std::optional<ttnn::Tensor> beta1_pow = std::nullopt;
    std::optional<ttnn::Tensor> beta2_pow = std::nullopt;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw::device
