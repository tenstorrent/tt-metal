// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

void clip_tensor_norm_(const std::vector<tt::tt_metal::Tensor>& tensors, float max_norm, float norm_type);

template <typename Model>
void clip_gradient_norm_(Model& model, float max_norm, float norm_type = 2.F) {
    std::vector<tt::tt_metal::Tensor> tensors;
    tensors.reserve(model.parameters().size());
    for (auto& [name, param] : model.parameters()) {
        auto& grad = param->get_grad();
        if (core::is_tensor_initialized(grad)) {
            tensors.push_back(grad);
        }
    }

    clip_tensor_norm_(tensors, max_norm, norm_type);
};

}  // namespace ttml::autograd
