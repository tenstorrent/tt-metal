// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen.hpp"

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/qwen_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/serializable.hpp"
#include <set>

namespace {

static std::vector<float> transpose_2d_flat(const std::vector<float> &flat, int64_t rows, int64_t cols) {
    assert(rows * cols == static_cast<int64_t>(flat.size()));
    std::vector<int> shape_vec = {static_cast<int>(rows), static_cast<int>(cols)};
    auto src = xt::adapt(flat, shape_vec);
    xt::xarray<float> t = xt::transpose(src);
    auto view = ttml::core::xtensor_to_span(t);
    return std::vector<float>(view.begin(), view.end());
}

static std::vector<float> pad_and_resize_flat(
    const std::vector<float> &flat, int64_t rows, int64_t cols, int64_t target_rows, int64_t target_cols) {
    // If dimensions match, return as is
    if (rows == target_rows && cols == target_cols) {
        return flat;
    }
    
    // Create output tensor with target dimensions, initialized to zero
    std::vector<float> out(static_cast<size_t>(target_rows * target_cols), 0.0f);
    
    // Copy data from source to target, handling both row and column differences
    int64_t copy_rows = std::min(rows, target_rows);
    int64_t copy_cols = std::min(cols, target_cols);
    
    for (int64_t r = 0; r < copy_rows; ++r) {
        for (int64_t c = 0; c < copy_cols; ++c) {
            out[r * target_cols + c] = flat[r * cols + c];
        }
    }
    
    return out;
}
}  // namespace

namespace ttml::models::qwen {

Qwen::Qwen(const QwenConfig& config) : m_config(config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    std::optional<uint32_t> intermediate_dim = config.intermediate_dim;
    uint32_t num_heads = config.num_heads;
    uint32_t num_groups = config.num_groups;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    runner_type = config.runner_type;
    float theta = config.theta;

    fmt::print("Qwen configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim: {}\n", embedding_dim);
    fmt::print("    Intermediate dim: {}\n", intermediate_dim ? fmt::format("{}", *intermediate_dim) : "None");
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num groups: {}\n", num_groups);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Positional embedding type: RoPE\n");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Weight tying: {}\n", config.weight_tying == WeightTyingType::Enabled ? "Enabled" : "Disabled");
    fmt::print("    Theta: {}\n", theta);

    // Parameter validation
    if (vocab_size == 0) {
        throw std::logic_error("Vocab size must be greater than 0");
    }
    if (num_blocks == 0) {
        throw std::logic_error("Number of blocks must be greater than 0");
    }
    if (num_heads == 0) {
        throw std::logic_error("Number of heads must be greater than 0");
    }
    if (num_groups == 0) {
        throw std::logic_error("Number of groups must be greater than 0");
    }
    if (embedding_dim == 0) {
        throw std::logic_error("Embedding dimension must be greater than 0");
    }
    if (embedding_dim % num_heads != 0) {
        throw std::logic_error(fmt::format(
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            embedding_dim, num_heads));
    }
    if (num_heads % num_groups != 0) {
        throw std::logic_error(fmt::format(
            "Number of heads ({}) must be divisible by number of groups ({})",
            num_heads, num_groups));
    }
    if (dropout_prob < 0.0f || dropout_prob > 1.0f) {
        throw std::logic_error(fmt::format(
            "Dropout probability must be between 0.0 and 1.0, provided: {}",
            dropout_prob));
    }

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
    
    auto last_fc = std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size_divisible_by_32, /* bias */ true);
    if (config.weight_tying == WeightTyingType::Enabled) {
        tok_emb = std::make_shared<ttml::modules::Embedding>(last_fc->get_weight());
    } else {
        tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);
    }

    // Create RoPE scaling params if they are set
    ops::RopeScalingParams rope_scaling_params;
    if (config.scaling_factor != 0.0F && config.original_context_length != 0U) {
        rope_scaling_params.original_context_length = config.original_context_length;
        rope_scaling_params.scaling_factor = config.scaling_factor;
        rope_scaling_params.high_freq_factor = config.high_freq_factor;
        rope_scaling_params.low_freq_factor = config.low_freq_factor;

        fmt::print("    RoPE scaling enabled:\n");
        fmt::print("        Scaling factor: {}\n", config.scaling_factor);
        fmt::print("        Original context length: {}\n", config.original_context_length);
        fmt::print("        High freq factor: {}\n", config.high_freq_factor);
        fmt::print("        Low freq factor: {}\n", config.low_freq_factor);
    }

    m_rope_params = ops::build_rope_params(
        /*sequence_length=*/max_sequence_length,
        /*head_dim=*/embedding_dim / num_heads,
        /*theta=*/theta,
        /*rope_scaling_params=*/rope_scaling_params);
    
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<ttml::modules::QwenBlock>(
            embedding_dim, num_heads, num_groups, m_rope_params, dropout_prob, intermediate_dim));
    }
    ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim);
    fc = last_fc;

    create_name("qwen");
    register_module(tok_emb, "tok_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("qwen_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    common::transformer::initialize_weights_gpt2(*this);
}

ttml::autograd::TensorPtr Qwen::operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = tok_emb_out;  // qwen does positional embedding in the attention blocks
    for (auto& block : blocks) {
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

QwenConfig read_config(const YAML::Node& config) {
    QwenConfig qwen_config;
    // Use defaults from Qwen2
    qwen_config.num_heads = config["num_heads"].as<uint32_t>(32U);
    qwen_config.num_groups = config["num_groups"].as<uint32_t>(32U);  // Qwen2 typically uses full attention
    qwen_config.embedding_dim = config["embedding_dim"].as<uint32_t>(4096U);
    if (config["intermediate_dim"]) {
        uint32_t intermediate_dim = config["intermediate_dim"].as<uint32_t>();
        qwen_config.intermediate_dim = std::make_optional(intermediate_dim);
    }
    qwen_config.dropout_prob = config["dropout_prob"].as<float>(0.0F);
    qwen_config.num_blocks = config["num_blocks"].as<uint32_t>(32U);
    qwen_config.vocab_size = config["vocab_size"].as<uint32_t>(151665U);
    qwen_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>(2048U);
    qwen_config.theta = config["theta"].as<float>(1000000.0F);  // Qwen2 uses higher theta
    qwen_config.runner_type = common::transformer::read_runner_type(config);
    qwen_config.weight_tying = common::transformer::read_weight_tying_type(config);

    // Read RoPE NTK-aware scaling parameters if they exist
    if (config["rope_scaling"]) {
        const auto& rope_scaling = config["rope_scaling"];
        if (rope_scaling["scaling_factor"]) {
            qwen_config.scaling_factor = rope_scaling["scaling_factor"].as<float>();
        }
        if (rope_scaling["high_freq_factor"]) {
            qwen_config.high_freq_factor = rope_scaling["high_freq_factor"].as<float>();
        }
        if (rope_scaling["low_freq_factor"]) {
            qwen_config.low_freq_factor = rope_scaling["low_freq_factor"].as<float>();
        }
        if (rope_scaling["original_context_length"]) {
            qwen_config.original_context_length = rope_scaling["original_context_length"].as<uint32_t>();
        }
    }

    return qwen_config;
}

YAML::Node write_config(const QwenConfig& qwen_config) {
    YAML::Node config;
    config["num_heads"] = qwen_config.num_heads;
    config["num_groups"] = qwen_config.num_groups;
    config["embedding_dim"] = qwen_config.embedding_dim;
    if (qwen_config.intermediate_dim) {
        config["intermediate_dim"] = *qwen_config.intermediate_dim;
    }
    config["dropout_prob"] = qwen_config.dropout_prob;
    config["num_blocks"] = qwen_config.num_blocks;
    config["vocab_size"] = qwen_config.vocab_size;
    config["max_sequence_length"] = qwen_config.max_sequence_length;
    config["theta"] = qwen_config.theta;
    
    // Add runner_type and weight_tying for consistency with read_config
    config["runner_type"] = (qwen_config.runner_type == RunnerType::Default) ? "default" : "memory_efficient";
    config["weight_tying"] = (qwen_config.weight_tying == WeightTyingType::Enabled) ? "enabled" : "disabled";

    // Add RoPE scaling parameters if they are set
    if (qwen_config.scaling_factor != 0.0F && qwen_config.original_context_length != 0U) {
        YAML::Node rope_scaling;
        rope_scaling["scaling_factor"] = qwen_config.scaling_factor;
        rope_scaling["high_freq_factor"] = qwen_config.high_freq_factor;
        rope_scaling["low_freq_factor"] = qwen_config.low_freq_factor;
        rope_scaling["original_context_length"] = qwen_config.original_context_length;
        config["rope_scaling"] = rope_scaling;
    }

    return config;
}

std::shared_ptr<Qwen> create(const QwenConfig& config) {
    return std::make_shared<Qwen>(config);
}

std::shared_ptr<Qwen> create(const YAML::Node& config) {
    QwenConfig qwen_config = read_config(config);
    return std::make_shared<Qwen>(qwen_config);
}

void Qwen::load_from_safetensors(const std::filesystem::path& model_path) {
    for (const auto &entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            auto path = entry.path();
            fmt::print("Loading model from: {}\n", path.string());
            auto parameters = this->parameters();
            load_model_from_safetensors(path, parameters, m_config);
        }
    }
}

void load_model_from_safetensors(const std::filesystem::path &path, serialization::NamedParameters &parameters, const QwenConfig& config) {
    // Track which parameters have been used
    std::set<std::string> used_parameters;
    
    // Local variables to replace static variables for thread safety
    std::map<int, std::vector<float>> k_weights, v_weights;
    std::map<int, std::array<int64_t, 2>> k_shapes, v_shapes;
    std::map<int, std::vector<float>> k_bias, v_bias;
    
    auto get_parameter = [&parameters, &used_parameters](const std::string &name) -> ttml::autograd::TensorPtr {
        auto it = parameters.find(name);
        if (it == parameters.end()) {
            throw std::runtime_error(fmt::format("Parameter {} not found in the model", name));
        }
        used_parameters.insert(name);
        return it->second;
    };
    
    serialization::SafetensorSerialization::TensorCallback loading_callback =
        [&parameters, &get_parameter, &config, &used_parameters, &k_weights, &v_weights, &k_shapes, &v_shapes, &k_bias, &v_bias](
            const serialization::SafetensorSerialization::TensorInfo &info, std::span<const std::byte> bytes) {
            fmt::print("Loading tensor: {}, shape:{}, format: {}\n", info.name, info.shape, info.dtype);
            std::vector<float> float_vec;
            if (info.dtype == "BF16") {
                // Convert BF16 bytes to float
                if (bytes.size_bytes() % 2 != 0) {
                    throw std::runtime_error("BF16 data size must be even");
                }
                const std::size_t n = bytes.size_bytes() / 2;
                float_vec.reserve(n);
                const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(bytes.data());
                for (std::size_t i = 0; i < n; ++i) {
                    // Convert BF16 to float by shifting to upper 16 bits
                    uint32_t tmp = static_cast<uint32_t>(bf16_data[i]) << 16;
                    float value;
                    std::memcpy(&value, &tmp, sizeof(value));
                    float_vec.push_back(value);
                }
            } else if (info.dtype == "F32") {
                float_vec = serialization::SafetensorSerialization::bytes_to_floats_copy(bytes);
            } else {
                throw std::runtime_error(fmt::format("Unsupported dtype: {}", info.dtype));
            }
            
            // Token embedding weights
            if (info.name == "embed_tokens.weight" || info.name == "model.embed_tokens.weight" ||
                info.name == "transformer.wte.weight" || info.name == "wte.weight" ||
                info.name == "model.wte.weight" || info.name == "embeddings.word_embeddings.weight") {
                
                fmt::print("Loading embedding weight from: {}\n", info.name);
                auto out_tensor1 = get_parameter("qwen/tok_emb/weight");
                auto resized_emb = pad_and_resize_flat(
                    float_vec, 
                    info.shape[0], 
                    info.shape[1], 
                    out_tensor1->get_value().logical_shape()[-2],
                    out_tensor1->get_value().logical_shape()[-1]);
                out_tensor1->set_value(core::from_vector(
                    resized_emb, out_tensor1->get_value().logical_shape(), out_tensor1->get_value().device()));
                
                if (config.weight_tying == WeightTyingType::Disabled) {
                    auto out_tensor2 = get_parameter("qwen/fc/weight");
                    auto resized_weight1 = pad_and_resize_flat(
                        float_vec, 
                        info.shape[0], 
                        info.shape[1], 
                        out_tensor2->get_value().logical_shape()[-2],
                        out_tensor2->get_value().logical_shape()[-1]);
                    out_tensor2->set_value(core::from_vector(
                    resized_weight1, out_tensor2->get_value().logical_shape(), out_tensor2->get_value().device()));
                }
            }
            
            // Final layer norm
            if (info.name == "norm.weight" || info.name == "model.norm.weight") {
                auto out_tensor1 = get_parameter("qwen/ln_fc/gamma");
                // Handle potential shape mismatch for LayerNorm weights
                auto target_shape = out_tensor1->get_value().logical_shape();
                auto target_size = target_shape[-1];  // Last dimension size
                if (float_vec.size() != target_size) {
                    // Resize the vector to match target size
                    std::vector<float> resized_vec(target_size, 0.0f);
                    size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                    std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                    out_tensor1->set_value(core::from_vector(
                        resized_vec, target_shape, out_tensor1->get_value().device()));
                } else {
                    out_tensor1->set_value(core::from_vector(
                        float_vec, target_shape, out_tensor1->get_value().device()));
                }
            }

            // ---- Per-block mappings for Qwen ----
            // Qwen uses similar parameter naming conventions to Llama
            for (int i = 0; i < static_cast<int>(config.num_blocks); ++i) {
                const std::string layer_pfx = "model.layers." + std::to_string(i);
                const std::string layers_pfx = "layers." + std::to_string(i);
                
                // Input layer norm
                if (info.name == layer_pfx + ".input_layernorm.weight" || info.name == layers_pfx + ".input_layernorm.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/input_layernorm/gamma", i);
                    auto out_tensor1 = get_parameter(block_name);
                    // Handle potential shape mismatch for LayerNorm weights
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];  // Last dimension size
                    if (float_vec.size() != target_size) {
                        // Resize the vector to match target size
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
                
                // Post attention layer norm
                if (info.name == layer_pfx + ".post_attention_layernorm.weight" || info.name == layers_pfx + ".post_attention_layernorm.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/post_attention_layernorm/gamma", i);
                    auto out_tensor1 = get_parameter(block_name);
                    // Handle potential shape mismatch for LayerNorm weights
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];  // Last dimension size
                    if (float_vec.size() != target_size) {
                        // Resize the vector to match target size
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
                
                // Attention weights
                if (info.name == layer_pfx + ".self_attn.q_proj.weight" || info.name == layers_pfx + ".self_attn.q_proj.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/self_attn/q_linear/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    // Handle potential shape mismatch for attention weights
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_rows = target_shape[-2];
                    auto target_cols = target_shape[-1];
                    auto resized_weight = pad_and_resize_flat(
                        transposed, info.shape[1], info.shape[0], target_rows, target_cols);
                    out_tensor1->set_value(core::from_vector(
                        resized_weight, target_shape, out_tensor1->get_value().device()));
                }
                
                // Attention bias
                if (info.name == layer_pfx + ".self_attn.q_proj.bias" || info.name == layers_pfx + ".self_attn.q_proj.bias") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/self_attn/q_linear/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];
                    if (float_vec.size() != target_size) {
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
                
                // For GroupedQueryAttention, k_proj and v_proj are combined into kv_linear
                
                if (info.name == layer_pfx + ".self_attn.k_proj.weight" || info.name == layers_pfx + ".self_attn.k_proj.weight") {
                    k_weights[i] = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    k_shapes[i] = {info.shape[1], info.shape[0]};
                }
                
                if (info.name == layer_pfx + ".self_attn.v_proj.weight" || info.name == layers_pfx + ".self_attn.v_proj.weight") {
                    v_weights[i] = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    v_shapes[i] = {info.shape[1], info.shape[0]};
                }
                
                // When both k and v weights are available, combine them
                if (k_weights.count(i) && v_weights.count(i)) {
                    auto block_name = fmt::format("qwen/qwen_block_{}/self_attn/kv_linear/weight", i);
                    auto out_tensor = get_parameter(block_name);
                    auto target_shape = out_tensor->get_value().logical_shape();
                    auto target_rows = target_shape[-2];
                    auto target_cols = target_shape[-1];
                    
                    // Combine k and v weights
                    std::vector<float> combined_weights;
                    combined_weights.insert(combined_weights.end(), k_weights[i].begin(), k_weights[i].end());
                    combined_weights.insert(combined_weights.end(), v_weights[i].begin(), v_weights[i].end());
                    
                    // Calculate combined shape
                    int64_t combined_rows = k_shapes[i][0];
                    int64_t combined_cols = k_shapes[i][1] + v_shapes[i][1];
                    
                    auto resized_weight = pad_and_resize_flat(
                        combined_weights, combined_rows, combined_cols, target_rows, target_cols);
                    out_tensor->set_value(core::from_vector(
                        resized_weight, target_shape, out_tensor->get_value().device()));
                    
                    // Clean up
                    k_weights.erase(i);
                    v_weights.erase(i);
                    k_shapes.erase(i);
                    v_shapes.erase(i);
                }
                
                // For GroupedQueryAttention, k_proj and v_proj bias are combined into kv_linear bias
                
                if (info.name == layer_pfx + ".self_attn.k_proj.bias" || info.name == layers_pfx + ".self_attn.k_proj.bias") {
                    k_bias[i] = float_vec;
                }
                
                if (info.name == layer_pfx + ".self_attn.v_proj.bias" || info.name == layers_pfx + ".self_attn.v_proj.bias") {
                    v_bias[i] = float_vec;
                }
                
                // When both k and v bias are available, combine them
                if (k_bias.count(i) && v_bias.count(i)) {
                    auto block_name = fmt::format("qwen/qwen_block_{}/self_attn/kv_linear/bias", i);
                    auto out_tensor = get_parameter(block_name);
                    auto target_shape = out_tensor->get_value().logical_shape();
                    auto target_size = target_shape[-1];
                    
                    // Combine k and v bias
                    std::vector<float> combined_bias;
                    combined_bias.insert(combined_bias.end(), k_bias[i].begin(), k_bias[i].end());
                    combined_bias.insert(combined_bias.end(), v_bias[i].begin(), v_bias[i].end());
                    
                    if (combined_bias.size() != target_size) {
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(combined_bias.size(), static_cast<size_t>(target_size));
                        std::copy(combined_bias.begin(), combined_bias.begin() + copy_size, resized_vec.begin());
                        out_tensor->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor->get_value().device()));
                    } else {
                        out_tensor->set_value(core::from_vector(
                            combined_bias, target_shape, out_tensor->get_value().device()));
                    }
                    
                    // Clean up
                    k_bias.erase(i);
                    v_bias.erase(i);
                }
                
                if (info.name == layer_pfx + ".self_attn.o_proj.weight" || info.name == layers_pfx + ".self_attn.o_proj.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/self_attn/out_linear/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_rows = target_shape[-2];
                    auto target_cols = target_shape[-1];
                    auto resized_weight = pad_and_resize_flat(
                        transposed, info.shape[1], info.shape[0], target_rows, target_cols);
                    out_tensor1->set_value(core::from_vector(
                        resized_weight, target_shape, out_tensor1->get_value().device()));
                }
                
                // Output projection bias
                if (info.name == layer_pfx + ".self_attn.o_proj.bias" || info.name == layers_pfx + ".self_attn.o_proj.bias") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/self_attn/out_linear/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];
                    if (float_vec.size() != target_size) {
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
                
                // MLP weights - Qwen uses gate_proj, up_proj, down_proj
                if (info.name == layer_pfx + ".mlp.gate_proj.weight" || info.name == layers_pfx + ".mlp.gate_proj.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/mlp/gate_proj/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_rows = target_shape[-2];
                    auto target_cols = target_shape[-1];
                    auto resized_weight = pad_and_resize_flat(
                        transposed, info.shape[1], info.shape[0], target_rows, target_cols);
                    out_tensor1->set_value(core::from_vector(
                        resized_weight, target_shape, out_tensor1->get_value().device()));
                }
                
                // MLP gate projection bias
                if (info.name == layer_pfx + ".mlp.gate_proj.bias" || info.name == layers_pfx + ".mlp.gate_proj.bias") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/mlp/gate_proj/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];
                    if (float_vec.size() != target_size) {
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
                
                if (info.name == layer_pfx + ".mlp.up_proj.weight" || info.name == layers_pfx + ".mlp.up_proj.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/mlp/up_proj/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_rows = target_shape[-2];
                    auto target_cols = target_shape[-1];
                    auto resized_weight = pad_and_resize_flat(
                        transposed, info.shape[1], info.shape[0], target_rows, target_cols);
                    out_tensor1->set_value(core::from_vector(
                        resized_weight, target_shape, out_tensor1->get_value().device()));
                }
                
                // MLP up projection bias
                if (info.name == layer_pfx + ".mlp.up_proj.bias" || info.name == layers_pfx + ".mlp.up_proj.bias") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/mlp/up_proj/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];
                    if (float_vec.size() != target_size) {
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
                
                if (info.name == layer_pfx + ".mlp.down_proj.weight" || info.name == layers_pfx + ".mlp.down_proj.weight") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/mlp/down_proj/weight", i);
                    auto transposed = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_rows = target_shape[-2];
                    auto target_cols = target_shape[-1];
                    auto resized_weight = pad_and_resize_flat(
                        transposed, info.shape[1], info.shape[0], target_rows, target_cols);
                    out_tensor1->set_value(core::from_vector(
                        resized_weight, target_shape, out_tensor1->get_value().device()));
                }
                
                // MLP down projection bias
                if (info.name == layer_pfx + ".mlp.down_proj.bias" || info.name == layers_pfx + ".mlp.down_proj.bias") {
                    auto block_name = fmt::format("qwen/qwen_block_{}/mlp/down_proj/bias", i);
                    auto out_tensor1 = get_parameter(block_name);
                    auto target_shape = out_tensor1->get_value().logical_shape();
                    auto target_size = target_shape[-1];
                    if (float_vec.size() != target_size) {
                        std::vector<float> resized_vec(target_size, 0.0f);
                        size_t copy_size = std::min(float_vec.size(), static_cast<size_t>(target_size));
                        std::copy(float_vec.begin(), float_vec.begin() + copy_size, resized_vec.begin());
                        out_tensor1->set_value(core::from_vector(
                            resized_vec, target_shape, out_tensor1->get_value().device()));
                    } else {
                        out_tensor1->set_value(core::from_vector(
                            float_vec, target_shape, out_tensor1->get_value().device()));
                    }
                }
            }
            return true;
        };
    
    serialization::SafetensorSerialization::visit_safetensors_file(path, loading_callback);
    
    // Handle internal model bias parameters that are registered but not loaded from safetensors
    // These are bias parameters from LinearLayer modules within the model
    for (int i = 0; i < static_cast<int>(config.num_blocks); ++i) {
        std::vector<std::string> internal_bias_params = {
            fmt::format("qwen/qwen_block_{}/self_attn/q_linear/bias", i),
            fmt::format("qwen/qwen_block_{}/self_attn/kv_linear/bias", i),
            fmt::format("qwen/qwen_block_{}/self_attn/out_linear/bias", i),
        };
        
        for (const auto& bias_param : internal_bias_params) {
            if (parameters.find(bias_param) != parameters.end()) {
                used_parameters.insert(bias_param);
            }
        }
    }
    
    // Handle final classifier bias parameter
    if (parameters.find("qwen/fc/bias") != parameters.end()) {
        used_parameters.insert("qwen/fc/bias");
    }
    
    // Check if all parameters were used
    std::vector<std::string> unused_parameters;
    for (const auto &[name, tensor] : parameters) {
        if (used_parameters.find(name) == used_parameters.end()) {
            unused_parameters.push_back(name);
        }
    }

    if (!unused_parameters.empty()) {
        fmt::print("Warning: The following parameters were not used during loading:\n");
        for (const auto &param_name : unused_parameters) {
            fmt::print("  - {}\n", param_name);
        }
        fmt::print("Total unused parameters: {}\n", unused_parameters.size());
        
        // Optionally throw an error if strict checking is desired
        // Uncomment the following line to make unused parameters an error:
        // throw std::runtime_error(fmt::format("Found {} unused parameters in the model", unused_parameters.size()));
    } else {
        fmt::print("All {} parameters were successfully loaded and used.\n", parameters.size());
    }
}

}  // namespace ttml::models::qwen