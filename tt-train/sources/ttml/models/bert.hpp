// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "models/base_transformer.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/bert_block.hpp"
#include "modules/dropout_module.hpp"
#include "modules/embedding_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/positional_embeddings.hpp"

namespace ttml::models::bert {

struct BertConfig {
    uint32_t vocab_size = 30522U;
    uint32_t max_sequence_length = 512U;
    uint32_t embedding_dim = 768U;
    uint32_t intermediate_size = 3072U;
    uint32_t num_heads = 12U;
    uint32_t num_blocks = 12U;
    float dropout_prob = 0.1F;
    float layer_norm_eps = 1e-12F;
    bool use_token_type_embeddings = true;
    uint32_t type_vocab_size = 2U;  // For sentence A/B distinction
    common::transformer::RunnerType runner_type = common::transformer::RunnerType::Default;
    bool use_pooler = false;  // For classification tasks
};

class Bert : public BaseTransformer {
private:
    std::shared_ptr<modules::Embedding> m_token_embeddings;
    std::shared_ptr<modules::TrainablePositionalEmbedding> m_position_embeddings;
    std::shared_ptr<modules::Embedding> m_token_type_embeddings;
    std::shared_ptr<modules::LayerNormLayer> m_embedding_norm;
    std::shared_ptr<modules::DropoutLayer> m_embedding_dropout;
    std::vector<std::shared_ptr<modules::BertBlock>> m_blocks;
    std::shared_ptr<modules::LinearLayer> m_pooler;  // Optional pooler for classification

    BertConfig m_config;
    common::transformer::RunnerType m_runner_type;

public:
    explicit Bert(const BertConfig& config);
    virtual ~Bert() = default;

    void load_from_safetensors(const std::filesystem::path& model_path) override;

    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& input_ids,
        const autograd::TensorPtr& attention_mask = nullptr,
        const autograd::TensorPtr& token_type_ids = nullptr);

private:
    [[nodiscard]] autograd::TensorPtr get_embeddings(
        const autograd::TensorPtr& input_ids, const autograd::TensorPtr& token_type_ids = nullptr);

    [[nodiscard]] autograd::TensorPtr process_attention_mask(const autograd::TensorPtr& attention_mask) const;
};

BertConfig read_config(const YAML::Node& config);
YAML::Node write_config(const BertConfig& bert_config);
std::shared_ptr<Bert> create(const BertConfig& config);
std::shared_ptr<Bert> create(const YAML::Node& config);

void load_model_from_safetensors(const std::filesystem::path& path, serialization::NamedParameters& parameters);

}  // namespace ttml::models::bert
