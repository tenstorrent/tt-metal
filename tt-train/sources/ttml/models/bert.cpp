// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bert.hpp"

#include <yaml-cpp/yaml.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "modules/bert_block.hpp"
#include "modules/dropout_module.hpp"
#include "modules/embedding_module.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/positional_embeddings.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/serializable.hpp"

namespace ttml::models::bert {

Bert::Bert(const BertConfig& config) : m_config(config), m_runner_type(config.runner_type) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t intermediate_size = config.intermediate_size;
    uint32_t num_heads = config.num_heads;
    uint32_t num_blocks = config.num_blocks;
    float dropout_prob = config.dropout_prob;
    float layer_norm_eps = config.layer_norm_eps;

    fmt::print("BERT configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Intermediate size: {}\n", intermediate_size);
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Layer norm epsilon: {}\n", layer_norm_eps);
    fmt::print("    Use token type embeddings: {}\n", config.use_token_type_embeddings ? "true" : "false");
    fmt::print(
        "    Runner type: {}\n",
        m_runner_type == common::transformer::RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Use pooler: {}\n", config.use_pooler ? "true" : "false");

    // Validation with detailed error messages
    if (max_sequence_length % 32 != 0) {
        throw std::logic_error(fmt::format(
            "Max sequence length must be divisible by 32 due to tensor limitations. "
            "max_sequence_length={}, required divisor=32, remainder={}",
            max_sequence_length,
            max_sequence_length % 32));
    }
    if (embedding_dim % 32 != 0) {
        throw std::logic_error(fmt::format(
            "Embedding dimension must be divisible by 32 due to tensor limitations. "
            "embedding_dim={}, required divisor=32, remainder={}",
            embedding_dim,
            embedding_dim % 32));
    }
    if (embedding_dim % num_heads != 0) {
        throw std::logic_error(fmt::format(
            "Embedding dimension must be divisible by number of heads. "
            "embedding_dim={}, num_heads={}, remainder={}",
            embedding_dim,
            num_heads,
            embedding_dim % num_heads));
    }

    // Create embeddings with proper alignment
    uint32_t vocab_size_aligned = ((vocab_size + 31) / 32) * 32;
    m_token_embeddings = std::make_shared<modules::Embedding>(vocab_size_aligned, embedding_dim);

    // Create positional embeddings
    modules::PositionalEmbeddingConfig pos_config{
        .embedding_dim = embedding_dim,
        .sequence_length = max_sequence_length,
        .dropout_prob = 0.0F,  // No dropout on position embeddings in BERT
        .use_dropout_seed_per_device = true};
    m_position_embeddings = std::make_shared<modules::TrainablePositionalEmbedding>(pos_config);

    // Create token type embeddings if enabled
    if (config.use_token_type_embeddings) {
        // Type vocab size is typically 2 (sentence A and sentence B)
        uint32_t type_vocab_aligned = ((config.type_vocab_size + 31) / 32) * 32;
        m_token_type_embeddings = std::make_shared<modules::Embedding>(type_vocab_aligned, embedding_dim);
    }

    // Embedding layer norm and dropout
    // Pass layer_norm_eps for consistent normalization across all layers
    m_embedding_norm = std::make_shared<modules::LayerNormLayer>(
        embedding_dim,
        layer_norm_eps,  // Use BERT's epsilon (typically 1e-12)
        false);
    m_embedding_dropout = std::make_shared<modules::DropoutLayer>(dropout_prob);

    // Create transformer blocks
    m_blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        modules::BertBlockConfig block_config{
            .embedding_dim = embedding_dim,
            .intermediate_size = intermediate_size,
            .num_heads = num_heads,
            .dropout_prob = dropout_prob,
            .layer_norm_eps = layer_norm_eps};
        m_blocks.push_back(std::make_shared<modules::BertBlock>(block_config));
    }

    // Optional pooler for classification
    if (config.use_pooler) {
        m_pooler = std::make_shared<modules::LinearLayer>(embedding_dim, embedding_dim);
    }

    // Register modules
    create_name("bert");
    register_module(m_token_embeddings, "token_embeddings");
    register_module(m_position_embeddings, "position_embeddings");
    if (m_token_type_embeddings) {
        register_module(m_token_type_embeddings, "token_type_embeddings");
    }
    register_module(m_embedding_norm, "embedding_norm");
    register_module(m_embedding_dropout, "embedding_dropout");

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(m_blocks[block_idx], fmt::format("bert_block_{}", block_idx));
    }

    if (m_pooler) {
        register_module(m_pooler, "pooler");
    }

    // Initialize weights with BERT-specific initialization
    // BERT uses truncated normal with std=0.02, but we'll use normal for now
    // TODO: Implement truncated normal initialization
    common::transformer::initialize_weights_gpt2(*this);
}

autograd::TensorPtr Bert::get_embeddings(
    const autograd::TensorPtr& input_ids, const autograd::TensorPtr& token_type_ids) {
    // Validate input shapes
    auto input_shape = input_ids->get_shape();

    // Token embeddings
    auto embeddings = (*m_token_embeddings)(input_ids);

    // Add positional embeddings using the operator() which adds positions and applies dropout (no-op since
    // dropout_prob=0)
    embeddings = (*m_position_embeddings)(embeddings);

    // Add token type embeddings if provided and enabled
    if (m_token_type_embeddings && token_type_ids) {
        // Validate token_type_ids shape matches input_ids
        auto type_shape = token_type_ids->get_shape();
        if (type_shape != input_shape) {
            throw std::logic_error(fmt::format(
                "token_type_ids shape must match input_ids shape. "
                "input_ids shape={}, token_type_ids shape={}",
                input_shape,
                type_shape));
        }
        auto type_embeddings = (*m_token_type_embeddings)(token_type_ids);
        embeddings = ops::add(embeddings, type_embeddings);
    }

    // Apply layer norm and dropout
    embeddings = (*m_embedding_norm)(embeddings);
    embeddings = (*m_embedding_dropout)(embeddings);

    return embeddings;
}

autograd::TensorPtr Bert::process_attention_mask(const autograd::TensorPtr& attention_mask) const {
    if (!attention_mask) {
        return nullptr;
    }

    // Convert attention mask from (1, 0) to (0, -10000) for additive attention
    // mask = (1 - attention_mask) * -10000
    // This makes padding tokens have very negative scores before softmax
    auto inverted_mask = ops::sub(
        autograd::create_tensor(core::ones(attention_mask->get_shape(), &autograd::ctx().get_device())),
        attention_mask);
    auto processed_mask = ops::mul(inverted_mask, -10000.0F);

    // Reshape mask for broadcasting with attention scores
    // From [batch, 1, 1, seq_len] to [batch, 1, seq_len, seq_len] for self-attention
    auto mask_shape = attention_mask->get_shape();
    auto batch_size = mask_shape[0];
    auto seq_len = mask_shape[3];

    // Expand mask to [batch, 1, seq_len, seq_len] for proper broadcasting
    auto expanded_mask = ttnn::reshape(processed_mask->get_value(), ttnn::Shape{batch_size, 1, 1, seq_len});
    expanded_mask = ttnn::repeat(expanded_mask, ttnn::Shape{1, 1, seq_len, 1});

    return autograd::create_tensor(expanded_mask);
}

// BaseTransformer interface implementation - required for polymorphism
autograd::TensorPtr Bert::operator()(const autograd::TensorPtr& x, const autograd::TensorPtr& mask) {
    // This satisfies the BaseTransformer virtual function requirement
    // Assumes x contains input_ids and mask is attention_mask
    // No token_type_ids in this interface - uses default behavior (all zeros)
    return forward(x, mask, nullptr);
}

// Primary BERT implementation with full functionality
autograd::TensorPtr Bert::forward(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    // Process attention mask for proper masking
    auto processed_mask = process_attention_mask(attention_mask);

    // Get input embeddings
    auto hidden_states = get_embeddings(input_ids, token_type_ids);

    // Pass through transformer blocks
    for (auto& block : m_blocks) {
        if (m_runner_type == common::transformer::RunnerType::MemoryEfficient) {
            hidden_states = common::transformer::memory_efficient_runner(*block, hidden_states, processed_mask);
        } else if (m_runner_type == common::transformer::RunnerType::Default) {
            hidden_states = (*block)(hidden_states, processed_mask);
        } else {
            throw std::runtime_error("Unknown runner type. Supported runner types ['default', 'memory_efficient']");
        }
    }

    // Optional pooling for classification tasks
    if (m_pooler) {
        // Extract [CLS] token representation (first token in sequence)
        auto hidden_shape = hidden_states->get_shape();
        auto batch_size = hidden_shape[0];
        auto embedding_dim = hidden_shape[3];

        // Slice to get only the [CLS] token: [batch, 1, 1, embedding_dim]
        // Note: hidden_shape[1] is 1 (channel dimension), not num_heads
        ttnn::SmallVector<uint32_t> start_indices = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end_indices = {batch_size, 1, 1, embedding_dim};
        ttnn::SmallVector<uint32_t> stride = {1, 1, 1, 1};

        auto cls_token = ttnn::slice(hidden_states->get_value(), start_indices, end_indices, stride);

        auto pooled_output = autograd::create_tensor(cls_token);
        pooled_output = (*m_pooler)(pooled_output);

        // BERT uses tanh activation for the pooler output
        pooled_output = ops::tanh(pooled_output);

        return pooled_output;
    }

    return hidden_states;
}

// Convenience operator() for backward compatibility
autograd::TensorPtr Bert::operator()(
    const autograd::TensorPtr& input_ids,
    const autograd::TensorPtr& attention_mask,
    const autograd::TensorPtr& token_type_ids) {
    return forward(input_ids, attention_mask, token_type_ids);
}

void Bert::load_from_safetensors(const std::filesystem::path& model_path) {
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            auto path = entry.path();
            fmt::print("Loading BERT model from: {}\n", path.string());
            auto parameters = this->parameters();
            load_model_from_safetensors(path, parameters);
        }
    }
}

void load_model_from_safetensors(const std::filesystem::path& path, serialization::NamedParameters& parameters) {
    fmt::print("Loading BERT weights from safetensors file\n");

    auto get_parameter = [&parameters](const std::string& name) -> ttml::autograd::TensorPtr {
        auto it = parameters.find(name);
        if (it == parameters.end()) {
            throw std::runtime_error(fmt::format("Parameter {} not found in the model", name));
        }
        return it->second;
    };

    // Helper to pad vocabulary embeddings if needed
    auto pad_vocab_embeddings = [](const std::vector<float>& flat, int64_t rows, int64_t cols, int64_t target_rows) {
        if (rows >= target_rows) {
            return flat;
        }
        std::vector<float> out(static_cast<size_t>(target_rows * cols), 0.0f);
        std::copy(flat.begin(), flat.end(), out.begin());
        return out;
    };

    serialization::SafetensorSerialization::TensorCallback loading_callback =
        [&parameters, &get_parameter, &pad_vocab_embeddings](
            const serialization::SafetensorSerialization::TensorInfo& info, std::span<const std::byte> bytes) {
            if (info.dtype != "F32") {
                throw std::runtime_error(fmt::format("Unsupported dtype: {}", info.dtype));
            }

            auto float_vec = serialization::SafetensorSerialization::bytes_to_floats_copy(bytes);

            // Word embeddings
            if (info.name == "bert.embeddings.word_embeddings.weight") {
                auto param = get_parameter("bert/token_embeddings/weight");
                auto padded = pad_vocab_embeddings(
                    float_vec, info.shape[0], info.shape[1], param->get_value().logical_shape()[-2]);
                param->set_value(core::from_vector(
                    padded, param->get_value().logical_shape(), param->get_value().device()));
                fmt::print("  Loaded word embeddings\n");
            }
            // Position embeddings
            else if (info.name == "bert.embeddings.position_embeddings.weight") {
                auto param = get_parameter("bert/position_embeddings/weight");
                param->set_value(core::from_vector(
                    float_vec, param->get_value().logical_shape(), param->get_value().device()));
                fmt::print("  Loaded position embeddings\n");
            }
            // Token type embeddings
            else if (info.name == "bert.embeddings.token_type_embeddings.weight") {
                if (parameters.find("bert/token_type_embeddings/weight") != parameters.end()) {
                    auto param = get_parameter("bert/token_type_embeddings/weight");
                    auto padded = pad_vocab_embeddings(
                        float_vec, info.shape[0], info.shape[1], param->get_value().logical_shape()[-2]);
                    param->set_value(core::from_vector(
                        padded, param->get_value().logical_shape(), param->get_value().device()));
                    fmt::print("  Loaded token type embeddings\n");
                }
            }
            // Embedding LayerNorm
            else if (info.name == "bert.embeddings.LayerNorm.weight") {
                auto param = get_parameter("bert/embedding_norm/gamma");
                param->set_value(core::from_vector(
                    float_vec, param->get_value().logical_shape(), param->get_value().device()));
            }
            else if (info.name == "bert.embeddings.LayerNorm.bias") {
                auto param = get_parameter("bert/embedding_norm/beta");
                param->set_value(core::from_vector(
                    float_vec, param->get_value().logical_shape(), param->get_value().device()));
            }
            // Pooler (if present)
            else if (info.name == "bert.pooler.dense.weight") {
                if (parameters.find("bert/pooler/weight") != parameters.end()) {
                    auto param = get_parameter("bert/pooler/weight");
                    // Note: HuggingFace stores as [out_features, in_features], we need [1, 1, out_features,
                    // in_features]
                    param->set_value(core::from_vector(
                        float_vec, param->get_value().logical_shape(), param->get_value().device()));
                    fmt::print("  Loaded pooler dense weight\n");
                }
            }
            else if (info.name == "bert.pooler.dense.bias") {
                if (parameters.find("bert/pooler/bias") != parameters.end()) {
                    auto param = get_parameter("bert/pooler/bias");
                    param->set_value(core::from_vector(
                        float_vec, param->get_value().logical_shape(), param->get_value().device()));
                    fmt::print("  Loaded pooler dense bias\n");
                }
            }

            // Process encoder layers
            // Check if this is an encoder layer parameter
            if (info.name.starts_with("bert.encoder.layer.")) {
                // Extract layer index
                size_t layer_start = std::string("bert.encoder.layer.").length();
                size_t layer_end = info.name.find('.', layer_start);
                std::string layer_idx_str = info.name.substr(layer_start, layer_end - layer_start);
                int layer_idx = std::stoi(layer_idx_str);

                std::string layer_suffix = info.name.substr(layer_end + 1);

                // Attention weights - BERT has separate Q, K, V, we need to combine them
                if (layer_suffix == "attention.self.query.weight" ||
                    layer_suffix == "attention.self.key.weight" ||
                    layer_suffix == "attention.self.value.weight") {
                    // TODO: We'll need to handle QKV combination in a separate pass
                    // For now, store them temporarily
                    fmt::print("  Skipping separate Q/K/V weight for layer {} (needs combination)\n", layer_idx);
                }
                // Attention output
                else if (layer_suffix == "attention.output.dense.weight") {
                    auto param_name =
                        fmt::format("bert/bert_block_{}/attention/self_attention/out_linear/weight", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                } else if (layer_suffix == "attention.output.dense.bias") {
                    auto param_name =
                        fmt::format("bert/bert_block_{}/attention/self_attention/out_linear/bias", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                }
                // Attention LayerNorm
                else if (layer_suffix == "attention.output.LayerNorm.weight") {
                    auto param_name = fmt::format("bert/bert_block_{}/attention_norm/gamma", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                } else if (layer_suffix == "attention.output.LayerNorm.bias") {
                    auto param_name = fmt::format("bert/bert_block_{}/attention_norm/beta", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                }
                // MLP intermediate (up projection)
                else if (layer_suffix == "intermediate.dense.weight") {
                    auto param_name = fmt::format("bert/bert_block_{}/mlp/dense/weight", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                } else if (layer_suffix == "intermediate.dense.bias") {
                    auto param_name = fmt::format("bert/bert_block_{}/mlp/dense/bias", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                }
                // MLP output (down projection)
                else if (layer_suffix == "output.dense.weight") {
                    auto param_name = fmt::format("bert/bert_block_{}/mlp/output/weight", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                } else if (layer_suffix == "output.dense.bias") {
                    auto param_name = fmt::format("bert/bert_block_{}/mlp/output/bias", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                }
                // Output LayerNorm
                else if (layer_suffix == "output.LayerNorm.weight") {
                    auto param_name = fmt::format("bert/bert_block_{}/mlp_norm/gamma", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                } else if (layer_suffix == "output.LayerNorm.bias") {
                    auto param_name = fmt::format("bert/bert_block_{}/mlp_norm/beta", layer_idx);
                    auto param = get_parameter(param_name);
                    param->set_value(
                        core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));
                }
            }

            return true;  // Continue processing
        };

    // First pass: load all weights
    serialization::SafetensorSerialization::visit_safetensors_file(path, loading_callback);

    fmt::print("BERT weights loaded successfully from safetensors\n");
}

BertConfig read_config(const YAML::Node& config) {
    BertConfig bert_config;

    // Set defaults similar to BERT-base
    bert_config.vocab_size = config["vocab_size"].as<uint32_t>(30522U);
    bert_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>(512U);
    bert_config.embedding_dim = config["embedding_dim"].as<uint32_t>(768U);
    bert_config.intermediate_size = config["intermediate_size"].as<uint32_t>(3072U);
    bert_config.num_heads = config["num_heads"].as<uint32_t>(12U);
    bert_config.num_blocks = config["num_blocks"].as<uint32_t>(12U);
    bert_config.dropout_prob = config["dropout_prob"].as<float>(0.1F);
    bert_config.layer_norm_eps = config["layer_norm_eps"].as<float>(1e-12F);
    bert_config.use_token_type_embeddings = config["use_token_type_embeddings"].as<bool>(true);
    bert_config.type_vocab_size = config["type_vocab_size"].as<uint32_t>(2U);
    bert_config.use_pooler = config["use_pooler"].as<bool>(false);

    bert_config.runner_type = common::transformer::read_runner_type(config);

    return bert_config;
}

YAML::Node write_config(const BertConfig& bert_config) {
    YAML::Node config;

    config["vocab_size"] = bert_config.vocab_size;
    config["max_sequence_length"] = bert_config.max_sequence_length;
    config["embedding_dim"] = bert_config.embedding_dim;
    config["intermediate_size"] = bert_config.intermediate_size;
    config["num_heads"] = bert_config.num_heads;
    config["num_blocks"] = bert_config.num_blocks;
    config["dropout_prob"] = bert_config.dropout_prob;
    config["layer_norm_eps"] = bert_config.layer_norm_eps;
    config["use_token_type_embeddings"] = bert_config.use_token_type_embeddings;
    config["type_vocab_size"] = bert_config.type_vocab_size;
    config["use_pooler"] = bert_config.use_pooler;

    return config;
}

std::shared_ptr<Bert> create(const BertConfig& config) {
    return std::make_shared<Bert>(config);
}

std::shared_ptr<Bert> create(const YAML::Node& config) {
    BertConfig bert_config = read_config(config);
    return std::make_shared<Bert>(bert_config);
}

}  // namespace ttml::models::bert
