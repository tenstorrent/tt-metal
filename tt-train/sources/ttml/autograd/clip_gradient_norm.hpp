// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::autograd {

void clip_tensor_norm_(tt::tt_metal::Tensor& tensor, float max_norm);

template <typename Model>
void clip_gradient_norm_(Model& model, float max_norm) {
    for (auto& [name, param] : model.parameters()) {
        auto& grad = param->get_grad();
        if (core::is_tensor_initialized(grad)) {
            clip_tensor_norm_(grad, max_norm);
        }
    }
};

}  // namespace ttml::autograd
