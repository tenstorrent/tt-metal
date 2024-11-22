// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/clip_gradient_norm.hpp"

#include <core/ttnn_all_includes.hpp>

namespace ttml::autograd {

void clip_tensor_norm_(const std::vector<tt::tt_metal::Tensor>& tensors, float max_norm, float norm_type) {
    if (max_norm <= 0.F) {
        throw std::logic_error(fmt::format("max_norm should be positive, current max norm {}", max_norm));
    }

    ttnn::moreh_clip_grad_norm(
        tensors,
        max_norm,
        /* norm_type */ norm_type,
        /* error_if_nonfinite */ true,
        /* total_norm */ std::nullopt,
        /* memory_config */ std::nullopt,
        /* compute_kernel_config */ std::nullopt);
}

}  // namespace ttml::autograd
