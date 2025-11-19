// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_fused.hpp"

#include <stdexcept>

namespace ttml::metal {

ttnn::Tensor adamw_fused(
    [[maybe_unused]] const ttnn::Tensor& param_in,
    [[maybe_unused]] const ttnn::Tensor& grad,
    [[maybe_unused]] const ttnn::Tensor& exp_avg,
    [[maybe_unused]] const ttnn::Tensor& exp_avg_sq,
    [[maybe_unused]] const float lr,
    [[maybe_unused]] const float beta1,
    [[maybe_unused]] const float beta2,
    [[maybe_unused]] const float beta1_pow,
    [[maybe_unused]] const float beta2_pow,
    [[maybe_unused]] const float epsilon,
    [[maybe_unused]] const float weight_decay,
    [[maybe_unused]] const uint32_t step) {
    // TODO: Implement the fused AdamW kernel
    throw std::runtime_error("AdamW fused optimizer is not yet implemented");
}

}  // namespace ttml::metal
