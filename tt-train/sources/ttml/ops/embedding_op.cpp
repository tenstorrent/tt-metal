// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr embedding_op(const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight) {
    // prepare for embedding
    auto weight_tensor = weight->get_value();
    weight_tensor = ttnn::untilize(weight_tensor);

    auto embeddings =
        ttnn::embedding(tensor->get_value(), weight_tensor, /* pad_token */ std::nullopt, ttnn::Layout::TILE);
    auto embeddings_shape = embeddings.logical_shape();
    auto batch_size = embeddings_shape[0];
    auto sentence_size = embeddings_shape[1];
    auto embedding_dim = embeddings_shape[2];
    embeddings = ttnn::reshape(embeddings, ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}));
    auto out = autograd::create_tensor(embeddings);

    autograd::GradFunction grad = [tensor, weight, out]() {
        auto out_grad = out->get_grad();
        auto tensor_shape = tensor->get_value().logical_shape();
        out_grad = ttnn::reshape(
            out_grad, ttnn::Shape({1, 1, tensor_shape[0] * tensor_shape[-1], out_grad.logical_shape()[-1]}));
        auto weight_grad = ttnn::embedding_bw(tensor->get_value(), weight->get_value(), out_grad);
        weight->add_grad(weight_grad);
    };

    auto links = autograd::get_links(weight);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
