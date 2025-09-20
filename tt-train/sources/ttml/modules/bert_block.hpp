// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "modules/dropout_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"

namespace ttml::modules {

struct BertBlockConfig {
    uint32_t embedding_dim{};
    uint32_t intermediate_size{};
    uint32_t num_heads{};
    float dropout_prob{};
    float layer_norm_eps{1e-12F};
};

class BertMLP : public autograd::ModuleBase {
private:
    std::shared_ptr<LinearLayer> m_dense;
    std::shared_ptr<LinearLayer> m_output;
    std::shared_ptr<DropoutLayer> m_dropout;

public:
    BertMLP(uint32_t embedding_dim, uint32_t intermediate_size, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& input) override;
};

class BertAttention : public autograd::ModuleBase {
private:
    std::shared_ptr<MultiHeadAttention> m_self_attention;
    std::shared_ptr<LinearLayer> m_output_dense;
    std::shared_ptr<DropoutLayer> m_output_dropout;

public:
    BertAttention(uint32_t embedding_dim, uint32_t num_heads, float dropout_prob);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input, const autograd::TensorPtr& attention_mask) override;
};

class BertBlock : public autograd::ModuleBase {
private:
    std::shared_ptr<BertAttention> m_attention;
    std::shared_ptr<LayerNormLayer> m_attention_norm;
    std::shared_ptr<BertMLP> m_mlp;
    std::shared_ptr<LayerNormLayer> m_mlp_norm;

public:
    explicit BertBlock(const BertBlockConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input, const autograd::TensorPtr& attention_mask) override;
};

}  // namespace ttml::modules
