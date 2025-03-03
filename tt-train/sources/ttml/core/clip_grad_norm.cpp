// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <core/ttnn_all_includes.hpp>

#include "core/compute_kernel_config.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters, float max_norm, float p_norm_type, bool error_if_nonfinite) {
    std::vector<tt::tt_metal::Tensor> grads;
    grads.reserve(parameters.size());
    for (const auto& [_, tensor] : parameters) {
        if (tensor->get_requires_grad() && tensor->is_grad_initialized()) {
            grads.push_back(tensor->get_grad());
        }
    }
    auto tt_result = ttnn::moreh_clip_grad_norm(
        grads,
        max_norm,
        p_norm_type,
        error_if_nonfinite,
        /* total_norm*/ std::nullopt,
        /* memory_config*/ std::nullopt,
        ttml::core::ComputeKernelConfig::precise());
    return autograd::create_tensor(tt_result);
}

}  // namespace ttml::core
