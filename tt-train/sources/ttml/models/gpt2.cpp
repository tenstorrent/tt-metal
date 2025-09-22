// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gpt2.hpp"

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/embedding_module.hpp"
#include "modules/gpt_block.hpp"
#include "modules/layer_norm_module.hpp"
#include "modules/positional_embeddings.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/serializable.hpp"
namespace {

static std::vector<float> transpose_2d_flat(const std::vector<float> &flat, int64_t rows, int64_t cols) {
    fmt::print("Transposing!\n");
    assert(rows * cols == static_cast<int64_t>(flat.size()));
    std::vector<int> shape_vec = {static_cast<int>(rows), static_cast<int>(cols)};
    auto src = xt::adapt(flat, shape_vec);
    xt::xarray<float> t = xt::transpose(src);
    auto view = ttml::core::xtensor_to_span(t);
    return std::vector<float>(view.begin(), view.end());
}

static std::vector<float> pad_rows_flat(
    const std::vector<float> &flat, int64_t rows, int64_t cols, int64_t target_rows) {
    if (rows >= target_rows) {
        return flat;
    }
    std::vector<float> out(static_cast<size_t>(target_rows * cols), 0.0f);
    std::copy(flat.begin(), flat.end(), out.begin());
    return out;
}
}  // namespace

namespace ttml::models::gpt2 {

Transformer::Transformer(const TransformerConfig &config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t num_heads = config.num_heads;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    auto position_embedding_type = config.positional_embedding_type;
    auto use_composite_layernorm = config.experimental.use_composite_layernorm;
    runner_type = config.runner_type;

    fmt::print("Transformer configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print(
        "    Positional embedding type: {}\n",
        position_embedding_type == PositionalEmbeddingType::Trainable ? "Trainable" : "Fixed");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Composite layernorm: {}\n", use_composite_layernorm);
    fmt::print("    Weight tying: {}\n", config.weight_tying == WeightTyingType::Enabled ? "Enabled" : "Disabled");

    uint32_t vocab_size_divisible_by_32 = (vocab_size + 31) / 32 * 32;
    if (max_sequence_length % 32 != 0) {
        throw std::logic_error(fmt::format(
            "Max sequence length should be divisible by 32 due to current limitations in tensor. Provided "
            "max_sequence_length={}",
            max_sequence_length));
    }
    if (embedding_dim % 32 != 0) {
        throw std::logic_error(fmt::format(
            "Embedding size should be divisible by 32 due to current limitations in tensor. Provided "
            "embedding_dim={}",
            embedding_dim));
    }
    auto last_fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size, /* bias */ false);
    if (config.weight_tying == WeightTyingType::Enabled) {
        tok_emb = std::make_shared<ttml::modules::Embedding>(last_fc->get_weight());
    } else {
        tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);
    }

    auto create_positional_embedding = [position_embedding_type,
                                        max_sequence_length,
                                        embedding_dim,
                                        dropout_prob]() -> std::shared_ptr<autograd::ModuleBase> {
        if (position_embedding_type == PositionalEmbeddingType::Trainable) {
            return std::make_shared<ttml::modules::TrainablePositionalEmbedding>(
                ttml::modules::PositionalEmbeddingConfig{
                    .embedding_dim = embedding_dim,
                    .sequence_length = max_sequence_length,
                    .dropout_prob = dropout_prob,
                    .use_dropout_seed_per_device = false});
        } else {
            return std::make_shared<ttml::modules::PositionalEmbedding>(ttml::modules::PositionalEmbeddingConfig{
                .embedding_dim = embedding_dim,
                .sequence_length = max_sequence_length,
                .dropout_prob = dropout_prob,
                .use_dropout_seed_per_device = false});
        }
    };
    pos_emb = create_positional_embedding();
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(
            std::make_shared<ttml::modules::GPTBlock>(embedding_dim, num_heads, dropout_prob, use_composite_layernorm));
    }
    ln_fc = std::make_shared<ttml::modules::LayerNormLayer>(embedding_dim, use_composite_layernorm);
    fc = last_fc;

    create_name("transformer");
    register_module(tok_emb, "tok_emb");
    register_module(pos_emb, "pos_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("gpt_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    common::transformer::initialize_weights_gpt2(*this);
}

ttml::autograd::TensorPtr Transformer::operator()(
    const ttml::autograd::TensorPtr &x, const ttml::autograd::TensorPtr &mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = (*pos_emb)(tok_emb_out);
    for (auto &block : blocks) {
        if (runner_type == RunnerType::MemoryEfficient) {
            out = common::transformer::memory_efficient_runner(*block, out, mask);
        } else if (runner_type == RunnerType::Default) {
            out = (*block)(out, mask);
        } else {
            throw std::runtime_error("Unknown runner type. Supported runner types ['default', 'memory_efficient']");
        }
    }
    out = (*ln_fc)(out);
    auto logits = (*fc)(out);
    return logits;
}

PositionalEmbeddingType read_positional_embedding_type(const YAML::Node &config) {
    auto positional_embedding_str = config["positional_embedding_type"].as<std::string>("trainable");
    if (positional_embedding_str == "trainable") {
        return PositionalEmbeddingType::Trainable;
    } else if (positional_embedding_str == "fixed") {
        return PositionalEmbeddingType::Fixed;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown positional embedding type: {}. Supported positional embedding types [trainable, fixed]",
            positional_embedding_str));
    }
}

TransformerConfig read_config(const YAML::Node &config) {
    TransformerConfig transformer_config;
    transformer_config.num_heads = config["num_heads"].as<uint32_t>();
    transformer_config.embedding_dim = config["embedding_dim"].as<uint32_t>();
    transformer_config.dropout_prob = config["dropout_prob"].as<float>();
    transformer_config.num_blocks = config["num_blocks"].as<uint32_t>();
    transformer_config.vocab_size = config["vocab_size"].as<uint32_t>();
    transformer_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>();
    transformer_config.positional_embedding_type = read_positional_embedding_type(config);
    transformer_config.runner_type = common::transformer::read_runner_type(config);
    transformer_config.weight_tying = common::transformer::read_weight_tying_type(config);

    if (auto experimental_config = config["experimental"]) {
        transformer_config.experimental.use_composite_layernorm =
            experimental_config["use_composite_layernorm"].as<bool>();
    }
    return transformer_config;
}

YAML::Node write_config(const TransformerConfig &mlp_config) {
    YAML::Node config;
    config["num_heads"] = mlp_config.num_heads;
    config["embedding_dim"] = mlp_config.embedding_dim;
    config["dropout_prob"] = mlp_config.dropout_prob;
    config["num_blocks"] = mlp_config.num_blocks;
    config["vocab_size"] = mlp_config.vocab_size;
    config["max_sequence_length"] = mlp_config.max_sequence_length;
    return config;
}

std::shared_ptr<Transformer> create(const TransformerConfig &config) {
    return std::make_shared<Transformer>(config);
}

std::shared_ptr<Transformer> create(const YAML::Node &config) {
    TransformerConfig transformer_config = read_config(config);
    return std::make_shared<Transformer>(transformer_config);
}

void Transformer::load_from_safetensors(const std::filesystem::path &model_path) {
    for (const auto &entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            auto path = entry.path();
            fmt::print("Loading model from: {}\n", path.string());
            auto parameters = this->parameters();
            load_model_from_safetensors(path, parameters);
        }
    }
}

void load_model_from_safetensors(const std::filesystem::path &path, serialization::NamedParameters &parameters) {
    for (auto &[k, v] : parameters) {
        fmt::print("parameter name: {}\n", k);
    }
    auto get_parameter = [&parameters](const std::string &name) -> ttml::autograd::TensorPtr {
        auto it = parameters.find(name);
        if (it == parameters.end()) {
            throw std::runtime_error(fmt::format("Parameter {} not found in the model", name));
        }
        fmt::print(" Parameter {}, shape: {}\n", name, it->second->get_value().logical_shape());
        return it->second;
    };
    serialization::SafetensorSerialization::TensorCallback loading_callback =
        [&parameters, &get_parameter](
            const serialization::SafetensorSerialization::TensorInfo &info, std::span<const std::byte> bytes) {
            fmt::print("Loading tensor: {}, shape:{}, format: {}\n", info.name, info.shape, info.dtype);
            if (info.dtype != "F32") {
                throw std::runtime_error(fmt::format("Unsupported dtype: {}", info.dtype));
            }
            auto float_vec = serialization::SafetensorSerialization::bytes_to_floats_copy(bytes);
            if (info.name == "wte.weight") {
                auto out_tensor1 = get_parameter("transformer/fc/weight");
                fmt::print("Original shape {}, {}", info.shape[0], info.shape[1]);
                fmt::print(
                    "Transformed shape {}, {}\n",
                    out_tensor1->get_value().logical_shape()[-2],
                    out_tensor1->get_value().logical_shape()[-1]);
                auto padded_emb = pad_rows_flat(
                    float_vec, info.shape[0], info.shape[1], out_tensor1->get_value().logical_shape()[-2]);
                out_tensor1->set_value(core::from_vector(
                    padded_emb, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
            }

            //  pos_emb -> wpe.weight
            if (info.name == "wpe.weight") {
                auto out_tensor1 = get_parameter("transformer/pos_emb/weight");
                out_tensor1->set_value(core::from_vector(
                    float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
            }

            //  ln_f.{weight,bias}
            if (info.name == "ln_f.weight") {
                auto out_tensor1 = get_parameter("transformer/ln_fc/gamma");
                out_tensor1->set_value(core::from_vector(
                    float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
            }
            if (info.name == "ln_f.bias") {
                auto out_tensor1 = get_parameter("transformer/ln_fc/beta");
                out_tensor1->set_value(core::from_vector(
                    float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
            }

            // ---- Per-block mappings ----
            // Keys that require transpose in your Python:
            // {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
            for (int i = 0; i < 12; ++i) {
                const std::string pfx = "h." + std::to_string(i);

                // ln1
                if (info.name == pfx + ".ln_1.weight") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/ln1/gamma", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
                if (info.name == pfx + ".ln_1.bias") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/ln1/beta", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }

                // attention: qkv_linear <- attn.c_attn.{weight,bias}
                if (info.name == pfx + ".attn.c_attn.weight") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/attention/qkv_linear/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        transposed, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
                if (info.name == pfx + ".attn.c_attn.bias") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/attention/qkv_linear/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }

                // attention: out_linear <- attn.c_proj.{weight,bias}
                if (info.name == pfx + ".attn.c_proj.weight") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/attention/out_linear/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        transposed, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
                if (info.name == pfx + ".attn.c_proj.bias") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/attention/out_linear/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }

                // ln2
                if (info.name == pfx + ".ln_2.weight") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/ln2/gamma", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
                if (info.name == pfx + ".ln_2.bias") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/ln2/beta", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }

                // mlp: fc1 <- mlp.c_fc.{weight,bias}
                if (info.name == pfx + ".mlp.c_fc.weight") {
                    auto out_tensor1 = get_parameter("transformer/gpt_block_" + std::to_string(i) + "/mlp/fc1/weight");
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    out_tensor1->set_value(core::from_vector(
                        transposed, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
                if (info.name == pfx + ".mlp.c_fc.bias") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/mlp/fc1/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }

                // mlp: fc2 <- mlp.c_proj.{weight,bias}
                if (info.name == pfx + ".mlp.c_proj.weight") {
                    auto out_tensor1 = get_parameter("transformer/gpt_block_" + std::to_string(i) + "/mlp/fc2/weight");
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    out_tensor1->set_value(core::from_vector(
                        transposed, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
                if (info.name == pfx + ".mlp.c_proj.bias") {
                    auto block_name = fmt::format("transformer/gpt_block_{}/mlp/fc2/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    out_tensor1->set_value(core::from_vector(
                        float_vec, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                }
            }
            return true;
        };
    serialization::SafetensorSerialization::visit_safetensors_file(path, loading_callback);
}

}  // namespace ttml::models::gpt2
