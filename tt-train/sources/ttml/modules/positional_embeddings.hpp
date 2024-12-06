// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/auto_context.hpp"
#include "autograd/autocast_tensor.hpp"
#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"

namespace ttml::modules {

class PositionalEmbeddingBase : public autograd::ModuleBase {
public:
    virtual autograd::TensorPtr operator()(const autograd::TensorPtr& input) = 0;
};

class PositionalEmbedding : public PositionalEmbeddingBase {
private:
    uint32_t m_sequence_length{};
    std::shared_ptr<DropoutLayer> m_dropout;
    autograd::AutocastTensor m_positional_embedding;

public:
    explicit PositionalEmbedding(uint32_t embedding_dim, float dropout_prob = 0.F, uint32_t sequence_length = 1024);
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;
};

class TrainablePositionalEmbedding : public PositionalEmbeddingBase {
    uint32_t m_sequence_length{};
    autograd::TensorPtr m_weight;
    std::shared_ptr<DropoutLayer> m_dropout;
    void initialize_tensors(uint32_t sequence_length, uint32_t embedding_dim);

public:
    explicit TrainablePositionalEmbedding(
        uint32_t embedding_dim, float dropout_prob = 0.F, uint32_t sequence_length = 1024);
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;
};

}  // namespace ttml::modules
