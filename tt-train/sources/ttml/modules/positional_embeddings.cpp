// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/positional_embeddings.hpp"

#include <cmath>

#include "autograd/autocast_tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/dropout_module.hpp"
#include "ops/binary_ops.hpp"

namespace ttml::modules {

namespace {

autograd::AutocastTensor create_positional_embedding_tensor(uint32_t sequence_length, uint32_t embedding_dim) {
    std::vector<float> positional_embedding_data;
    positional_embedding_data.reserve(sequence_length * embedding_dim);

    const float div_const = 10000.F;
    for (uint32_t pos = 0; pos < sequence_length; ++pos) {
        for (uint32_t emb_idx = 0; emb_idx < embedding_dim; ++emb_idx) {
            float value = (emb_idx & 1)
                              ? std::cos(pos / std::pow(div_const, static_cast<float>(emb_idx - 1) / embedding_dim))
                              : std::sin(pos / std::pow(div_const, static_cast<float>(emb_idx) / embedding_dim));
            positional_embedding_data.push_back(value);
        }
    }

    auto shape = core::create_shape({1, 1, sequence_length, embedding_dim});
    auto* device = &autograd::ctx().get_device();
    auto tensor = core::from_vector(positional_embedding_data, shape, device);
    return autograd::AutocastTensor(tensor);
}

}  // namespace

PositionalEmbedding::PositionalEmbedding(const PositionalEmbeddingConfig& config) :
    m_sequence_length(config.sequence_length) {
    m_dropout = std::make_shared<DropoutLayer>(config.dropout_prob, config.use_dropout_seed_per_device);
    m_positional_embedding = create_positional_embedding_tensor(config.sequence_length, config.embedding_dim);

    create_name("positional_embedding");
    register_module(m_dropout, "dropout");
}

autograd::TensorPtr PositionalEmbedding::operator()(const autograd::TensorPtr& input) {
    auto input_tensor = input->get_value();
    auto input_shape = input_tensor.logical_shape();
    if (input_shape.rank() != 4) {
        throw std::runtime_error(
            "PositionalEmbedding: input tensor must have 4 dimensions. Got rank " + std::to_string(input_shape.rank()));
    }

    const uint32_t sequence_index = 2U;
    if (input_shape[sequence_index] != m_sequence_length) {
        throw std::runtime_error(fmt::format(
            "PositionalEmbedding: input tensor sequence length ({}) does not match the expected value ({})",
            input_shape[sequence_index],
            m_sequence_length));
    }

    auto x = ops::add(input, m_positional_embedding);
    x = (*m_dropout)(x);
    return x;
}

void TrainablePositionalEmbedding::initialize_tensors(uint32_t sequence_length, uint32_t embedding_dim) {
    auto* device = &autograd::ctx().get_device();
    m_weight = autograd::create_tensor();
    init::normal_init(
        m_weight, core::create_shape({1, 1, sequence_length, embedding_dim}), /* normal params */ {0.F, 1.F});
}

TrainablePositionalEmbedding::TrainablePositionalEmbedding(const PositionalEmbeddingConfig& config) :
    m_sequence_length(config.sequence_length) {
    m_dropout = std::make_shared<DropoutLayer>(config.dropout_prob, config.use_dropout_seed_per_device);
    initialize_tensors(config.sequence_length, config.embedding_dim);

    create_name("trainable_positional_embedding");
    register_module(m_dropout, "dropout");
    register_tensor(m_weight, "weight");
}

autograd::TensorPtr TrainablePositionalEmbedding::operator()(const autograd::TensorPtr& input) {
    auto input_tensor = input->get_value();
    auto input_shape = input_tensor.logical_shape();
    if (input_shape.rank() != 4) {
        throw std::runtime_error(
            "TrainablePositionalEmbedding: input tensor must have 4 dimensions. Got rank " +
            std::to_string(input_shape.rank()));
    }

    const uint32_t sequence_index = 2U;
    if (input_shape[sequence_index] != m_sequence_length) {
        throw std::runtime_error(fmt::format(
            "TrainablePositionalEmbedding: input tensor sequence length ({}) does not match the expected value ({})",
            input_shape[sequence_index],
            m_sequence_length));
    }

    auto x = ops::add(input, m_weight);
    x = (*m_dropout)(x);
    return x;
}

}  // namespace ttml::modules
