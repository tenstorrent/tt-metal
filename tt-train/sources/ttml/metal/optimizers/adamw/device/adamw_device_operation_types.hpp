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

struct step_scalar_tensors_t {
    ttnn::Tensor step_size;
    ttnn::Tensor inv_sqrt_bc2;
    ttnn::Tensor decay_factor;
};

struct tensor_args_t {
    const ttnn::Tensor& param;
    const ttnn::Tensor& grad;

    const ttnn::Tensor& exp_avg;
    const ttnn::Tensor& exp_avg_sq;
    std::optional<ttnn::Tensor> max_exp_avg_sq = std::nullopt;

    std::optional<step_scalar_tensors_t> step_scalars = std::nullopt;
};

using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::optimizers::adamw::device
