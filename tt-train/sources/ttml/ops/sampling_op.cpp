// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sampling_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr sample_op(
    const autograd::TensorPtr& logits,
    float temperature,
    uint32_t seed,
    const autograd::TensorPtr& logits_padding_mask) {
    auto sampled_tensor = ttnn_fixed::sample(
        logits->get_value(),
        temperature,
        seed,
        logits_padding_mask == nullptr ? std::nullopt
                                       : std::optional<tt::tt_metal::Tensor>(logits_padding_mask->get_value()));

    auto out = autograd::create_tensor(sampled_tensor);

    autograd::GradFunction grad = []() {
        // Argmax in sampling is non-differentiable; no gradient to propagate.
        throw std::runtime_error("Sampling operation backward pass is not implemented.");
    };

    auto links = autograd::get_links(logits, logits_padding_mask);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
