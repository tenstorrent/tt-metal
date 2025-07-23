// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_module.hpp"

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"
#include "ops/embedding_op.hpp"

namespace ttml::modules {

void Embedding::initialize_tensors(uint32_t num_embeddings, uint32_t embedding_dim) {
    auto* device = &autograd::ctx().get_device();
    m_weight = autograd::create_tensor();
    init::normal_init(m_weight, ttnn::Shape({1, 1, num_embeddings, embedding_dim}), /* normal params */ {0.F, 1.F});
}

Embedding::Embedding(uint32_t num_embeddings, uint32_t embedding_dim) {
    if (num_embeddings % TILE_HEIGHT != 0) {
        throw std::logic_error(
            fmt::format("num_embeddings must be a multiple of TILE_HEIGHT, current num_embeddings {}", num_embeddings));
    }
    if (embedding_dim % TILE_WIDTH != 0) {
        throw std::logic_error(
            fmt::format("embedding_dim must be a multiple of TILE_WIDTH, current embedding_dim {}", embedding_dim));
    }
    initialize_tensors(num_embeddings, embedding_dim);

    create_name("embedding");
    register_tensor(m_weight, "weight");
}

autograd::TensorPtr Embedding::operator()(const autograd::TensorPtr& tensor) {
    auto sentence_size = tensor->get_value().logical_shape()[-1];
    if (sentence_size % TILE_HEIGHT != 0 || sentence_size % TILE_WIDTH != 0) {
        throw std::logic_error(fmt::format(
            "sentence_size must be a multiple of TILE_HEIGHT and TILE_WIDTH, current sentence_size {}", sentence_size));
    }
    return ops::embedding_op(tensor, m_weight);
}

autograd::TensorPtr Embedding::get_weight() const {
    return m_weight;
}

Embedding::Embedding(const autograd::TensorPtr& weight) {
    m_weight = weight;
    create_name("embedding");
    register_tensor(m_weight, "weight");
}

}  // namespace ttml::modules
