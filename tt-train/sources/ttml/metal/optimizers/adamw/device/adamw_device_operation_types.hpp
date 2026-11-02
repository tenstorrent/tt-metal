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
    // Trace replay (iteration 45): a [1, 1, 32, 64] fp32 TILE tensor, tile 0 filled with (1 - beta1^t) / lr (the
    // inverse of the bias-corrected step size, applied to the denominator) and tile 1 with 1 / sqrt(1 - beta2^t),
    // computed on the device every step so a captured step replays with advancing powers; absent: the host scalars.
    std::optional<ttnn::Tensor> step_scalars = std::nullopt;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw::device
