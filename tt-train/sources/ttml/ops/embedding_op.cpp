// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_op.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"

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
        auto indices = tensor->get_value();
        const auto batch_size = tensor_shape[0];
        const auto sentence_size = tensor_shape[-1];
        const auto padded_sentence_size = tt::round_up(sentence_size, tt::constants::TILE_WIDTH);
        if (sentence_size != padded_sentence_size) {
            const auto padding_size = padded_sentence_size - sentence_size;
            const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> indices_padding = {
                {0, 0}, {0, 0}, {0, 0}, {0, padding_size}};
            const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> grad_padding = {
                {0, 0}, {0, 0}, {0, padding_size}, {0, 0}};
            indices = ttnn::pad(indices, indices_padding, 0.0F);
            out_grad = ttnn::pad(out_grad, grad_padding, 0.0F);
        }
        out_grad = ttnn::reshape(
            out_grad, ttnn::Shape({1, 1, batch_size * padded_sentence_size, out_grad.logical_shape()[-1]}));
        auto weight_grad = ttnn::embedding_bw(indices, weight->get_value(), out_grad);
        weight->add_grad(weight_grad);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, weight, tensor));
    return out;
}

}  // namespace ttml::ops
