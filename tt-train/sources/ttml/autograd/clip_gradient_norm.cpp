// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/clip_gradient_norm.hpp"

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

void clip_tensor_norm_(tt::tt_metal::Tensor& tensor, float max_norm) {
    if (max_norm <= 0.F) {
        throw std::logic_error(fmt::format("max_norm should be positive, current max norm {}", max_norm));
    }

    auto squared = ttnn::multiply(tensor, tensor);
    auto shape = core::create_shape({1, 1, 1, 1});
    auto out = ttml::core::from_vector({0.F}, shape, &ttml::autograd::ctx().get_device());
    ttnn::moreh_sum(squared, std::nullopt, true, out, squared.memory_config(), std::nullopt);
    auto grad_norm_tensor = ttnn::sqrt(out);

    // this is workaround before ttnn::repeat is fixed
    auto grad_norm_tensor_float = ttml::core::to_vector(grad_norm_tensor)[0];
    if (grad_norm_tensor_float > max_norm) {
        auto scale = max_norm / grad_norm_tensor_float;
        tensor = ttnn::multiply(tensor, scale);
    }
}
}  // namespace ttml::autograd
