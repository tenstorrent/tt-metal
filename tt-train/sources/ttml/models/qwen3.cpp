// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "qwen3.hpp"

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/common/transformer_common.hpp"
#include "modules/embedding_module.hpp"
#include "modules/qwen3_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/weight_utils.hpp"

// Import shared weight loading utilities
using ttml::serialization::pad_and_resize_flat;
using ttml::serialization::strict_copy_linear;
using ttml::serialization::transpose_flat;
using ttml::serialization::unpermute_norm_weights;
using ttml::serialization::unpermute_proj_rows;

namespace ttml::models::qwen3 {

Qwen3::Qwen3(const Qwen3Config& config) : m_config(config) {
    uint32_t vocab_size = config.vocab_size;
    uint32_t max_sequence_length = config.max_sequence_length;
    uint32_t embedding_dim = config.embedding_dim;
    uint32_t head_dim = config.head_dim;
    std::optional<uint32_t> intermediate_dim = config.intermediate_dim;
    uint32_t num_heads = config.num_heads;
    uint32_t num_groups = config.num_groups;
    float dropout_prob = config.dropout_prob;
    uint32_t num_blocks = config.num_blocks;
    runner_type = config.runner_type;
    float theta = config.theta;

    fmt::print("Qwen3 configuration:\n");
    fmt::print("    Vocab size: {}\n", vocab_size);
    fmt::print("    Max sequence length: {}\n", max_sequence_length);
    fmt::print("    Embedding dim (hidden_size): {}\n", embedding_dim);
    fmt::print("    Head dim: {}\n", head_dim);
    fmt::print("    Attention output dim: {}\n", num_heads * head_dim);
    fmt::print("    Intermediate dim: {}\n", intermediate_dim ? fmt::format("{}", *intermediate_dim) : "None");
    fmt::print("    Num heads: {}\n", num_heads);
    fmt::print("    Num groups (KV heads): {}\n", num_groups);
    fmt::print("    Dropout probability: {}\n", dropout_prob);
    fmt::print("    Num blocks: {}\n", num_blocks);
    fmt::print("    Positional embedding type: RoPE\n");
    fmt::print("    Runner type: {}\n", runner_type == RunnerType::Default ? "Default" : "Memory efficient");
    fmt::print("    Weight tying: {}\n", config.weight_tying == WeightTyingType::Enabled ? "Enabled" : "Disabled");
    fmt::print("    Theta: {}\n", theta);
    fmt::print("    RMSNorm epsilon: {}\n", config.rms_norm_eps);

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
        /*head_dim=*/head_dim,  // Use explicit head_dim for Qwen3
        /*theta=*/theta,
        /*rope_scaling_params=*/rope_scaling_params);

    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<ttml::modules::Qwen3Block>(
            embedding_dim,
            num_heads,
            num_groups,
            head_dim,
            m_rope_params,
            dropout_prob,
            config.rms_norm_eps,
            intermediate_dim));
    }
    // Final layer norm also uses Qwen3's epsilon=1e-6
    ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim, config.rms_norm_eps);
    fc = last_fc;

    create_name("qwen3");
    register_module(tok_emb, "tok_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("qwen3_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");
}

ttml::autograd::TensorPtr Qwen3::operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    return (*this)(x, mask, nullptr, 0);
}

ttml::autograd::TensorPtr Qwen3::operator()(
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& mask,
    std::shared_ptr<common::transformer::KvCache> kv_cache,
    const uint32_t new_tokens) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = tok_emb_out;

    for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
        auto& block = blocks[block_idx];
        if (runner_type == RunnerType::MemoryEfficient) {
            // Memory efficient mode does not support KV cache
            // Note: capture block by value (copy shared_ptr), not by reference, because
            // this lambda is stored for backward and block is a loop-local reference.
            auto block_impl = [block, block_idx](
                                  const ttml::autograd::TensorPtr& input, const ttml::autograd::TensorPtr& mask_arg) {
                return (*block)(input, mask_arg, nullptr, static_cast<uint32_t>(block_idx), 0);
            };
            out = common::transformer::memory_efficient_runner(block_impl, out, mask);
        } else {
            out = (*block)(out, mask, kv_cache, static_cast<uint32_t>(block_idx), new_tokens);
        }
    }

    out = (*ln_fc)(out);
    auto logits = (*fc)(out);
    return logits;
}

Qwen3Config read_config(const YAML::Node& config) {
    Qwen3Config qwen3_config;
    qwen3_config.num_heads = config["num_heads"].as<uint32_t>(16U);
    qwen3_config.num_groups = config["num_groups"].as<uint32_t>(8U);
    qwen3_config.embedding_dim = config["embedding_dim"].as<uint32_t>(1024U);
    qwen3_config.head_dim = config["head_dim"].as<uint32_t>(128U);  // Explicit for Qwen3
    if (config["intermediate_dim"]) {
        uint32_t intermediate_dim = config["intermediate_dim"].as<uint32_t>();
        qwen3_config.intermediate_dim = std::make_optional(intermediate_dim);
    }
    qwen3_config.dropout_prob = config["dropout_prob"].as<float>(0.0F);
    qwen3_config.num_blocks = config["num_blocks"].as<uint32_t>(28U);
    qwen3_config.vocab_size = config["vocab_size"].as<uint32_t>(151936U);
    qwen3_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>(2048U);
    qwen3_config.theta = config["theta"].as<float>(1000000.0F);
    qwen3_config.rms_norm_eps = config["rms_norm_eps"].as<float>(1e-6F);  // Qwen3 default is 1e-6
    qwen3_config.runner_type = common::transformer::read_runner_type(config);
    qwen3_config.weight_tying = common::transformer::read_weight_tying_type(config);

    // Read RoPE NTK-aware scaling parameters if they exist
    if (config["rope_scaling"]) {
        const auto& rope_scaling = config["rope_scaling"];
        if (rope_scaling["scaling_factor"]) {
            qwen3_config.scaling_factor = rope_scaling["scaling_factor"].as<float>();
        }
        if (rope_scaling["high_freq_factor"]) {
            qwen3_config.high_freq_factor = rope_scaling["high_freq_factor"].as<float>();
        }
        if (rope_scaling["low_freq_factor"]) {
            qwen3_config.low_freq_factor = rope_scaling["low_freq_factor"].as<float>();
        }
        if (rope_scaling["original_context_length"]) {
            qwen3_config.original_context_length = rope_scaling["original_context_length"].as<uint32_t>();
        }
    }
    return qwen3_config;
}

YAML::Node write_config(const Qwen3Config& qwen3_config) {
    YAML::Node config;
    config["num_heads"] = qwen3_config.num_heads;
    config["num_groups"] = qwen3_config.num_groups;
    config["embedding_dim"] = qwen3_config.embedding_dim;
    config["head_dim"] = qwen3_config.head_dim;
    if (qwen3_config.intermediate_dim) {
        config["intermediate_dim"] = *qwen3_config.intermediate_dim;
    }
    config["dropout_prob"] = qwen3_config.dropout_prob;
    config["num_blocks"] = qwen3_config.num_blocks;
    config["vocab_size"] = qwen3_config.vocab_size;
    config["max_sequence_length"] = qwen3_config.max_sequence_length;
    config["theta"] = qwen3_config.theta;
    config["rms_norm_eps"] = qwen3_config.rms_norm_eps;

    // Add RoPE scaling parameters if they are set
    if (qwen3_config.scaling_factor != 0.0F && qwen3_config.original_context_length != 0U) {
        YAML::Node rope_scaling;
        rope_scaling["scaling_factor"] = qwen3_config.scaling_factor;
        rope_scaling["high_freq_factor"] = qwen3_config.high_freq_factor;
        rope_scaling["low_freq_factor"] = qwen3_config.low_freq_factor;
        rope_scaling["original_context_length"] = qwen3_config.original_context_length;
        config["rope_scaling"] = rope_scaling;
    }

    return config;
}

std::shared_ptr<Qwen3> create(const Qwen3Config& config) {
    return std::make_shared<Qwen3>(config);
}

std::shared_ptr<Qwen3> create(const YAML::Node& config) {
    Qwen3Config qwen3_config = read_config(config);
    return std::make_shared<Qwen3>(qwen3_config);
}

void Qwen3::load_from_safetensors(const std::filesystem::path& model_path) {
    bool verbose = false;
    std::vector<std::filesystem::path> safetensor_files;
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            safetensor_files.push_back(entry.path());
        }
    }
    std::sort(safetensor_files.begin(), safetensor_files.end());

    std::set<std::string> used_parameters_global;
    std::set<std::string> ignored_parameters_global;
    auto parameters = this->parameters();

    for (size_t i = 0; i < safetensor_files.size(); ++i) {
        const auto& path = safetensor_files[i];
        if (verbose)
            fmt::print("Loading model from: {} ({}/{})\n", path.string(), i + 1, safetensor_files.size());
        load_model_from_safetensors(
            path, parameters, m_config, used_parameters_global, ignored_parameters_global, verbose);
        if (verbose)
            fmt::print("Completed loading file {}/{}\n", i + 1, safetensor_files.size());
    }

    std::vector<std::string> unused_parameters;
    for (const auto& [param_name, _] : parameters) {
        if (used_parameters_global.find(param_name) == used_parameters_global.end())
            unused_parameters.push_back(param_name);
    }

    if (!unused_parameters.empty()) {
        fmt::print("\nWarning: The following parameters were not used during loading:\n");
        for (const auto& param_name : unused_parameters) fmt::print("  - {}\n", param_name);
        fmt::print("Total unused parameters: {}\n", unused_parameters.size());
    } else {
        fmt::print("\nAll {} parameters were successfully loaded and used.\n", parameters.size());
    }

    if (!ignored_parameters_global.empty()) {
        fmt::print("\nNote: The following parameters were ignored during loading:\n");
        for (const auto& param_name : ignored_parameters_global) fmt::print("  - {}\n", param_name);
        fmt::print("Total ignored parameters: {}\n", ignored_parameters_global.size());
    }
}

void load_model_from_safetensors(
    const std::filesystem::path& path,
    serialization::NamedParameters& parameters,
    const Qwen3Config& config,
    std::set<std::string>& used_parameters,
    std::set<std::string>& ignored_parameters,
    bool verbose) {
    const bool meta_style = false;

    auto get_parameter =
        [&parameters, &used_parameters, verbose](const std::string& name) -> ttml::autograd::TensorPtr {
        auto it = parameters.find(name);

        if (it == parameters.end()) {
            throw std::runtime_error(fmt::format("Parameter {} not found in the model", name));
        }
        if (verbose)
            fmt::print("Using parameter: {} with shape: {}\n", name, it->second->get_value().logical_shape());
        used_parameters.insert(name);
        return it->second;
    };

    serialization::SafetensorSerialization::TensorCallback loading_callback =
        [&](const serialization::SafetensorSerialization::TensorInfo& info, std::span<const std::byte> bytes) {
            if (verbose)
                fmt::print("Loading tensor: {}, shape:{}, dtype:{}\n", info.name, info.shape, info.dtype);

            std::vector<float> float_vec;
            std::string dtype = info.dtype;
            for (auto& c : dtype) c = std::toupper(static_cast<unsigned char>(c));

            float_vec = serialization::SafetensorSerialization::bytes_to_float_vec(bytes, dtype);

            // ---- Embeddings (allow pad/resize) ----
            if (info.name == "embed_tokens.weight" || info.name == "model.embed_tokens.weight" ||
                info.name == "transformer.wte.weight" || info.name == "wte.weight" || info.name == "model.wte.weight" ||
                info.name == "embeddings.word_embeddings.weight") {
                auto weight_tying = config.weight_tying;
                auto embedding_weights_name =
                    (weight_tying == WeightTyingType::Enabled) ? "qwen3/fc/weight" : "qwen3/tok_emb/weight";
                auto out_tensor1 = get_parameter(embedding_weights_name);

                auto tgt = out_tensor1->get_value().logical_shape();

                auto resized_emb = pad_and_resize_flat(float_vec, info.shape[0], info.shape[1], tgt[-2], tgt[-1]);
                out_tensor1->set_value(core::from_vector(resized_emb, tgt, out_tensor1->get_value().device()));

                return true;
            }
            if (info.name == "lm_head.weight") {
                if (config.weight_tying == WeightTyingType::Disabled) {
                    auto out_tensor2 = get_parameter("qwen3/fc/weight");
                    auto tgt2 = out_tensor2->get_value().logical_shape();

                    auto resized_emb2 =
                        pad_and_resize_flat(float_vec, info.shape[0], info.shape[1], tgt2[-2], tgt2[-1]);
                    out_tensor2->set_value(core::from_vector(resized_emb2, tgt2, out_tensor2->get_value().device()));
                }
                return true;
            }
            // ---- Final LayerNorm (vector) ----
            if (info.name == "norm.weight" || info.name == "model.norm.weight") {
                auto out = get_parameter("qwen3/ln_fc/gamma");
                auto tgt = out->get_value().logical_shape();
                const int64_t N = tgt[-1];

                if (static_cast<int64_t>(float_vec.size()) != N) {
                    throw std::runtime_error(
                        fmt::format("[final LN] length mismatch: src={}, tgt={}", float_vec.size(), N));
                }
                out->set_value(core::from_vector(float_vec, tgt, out->get_value().device()));
                return true;
            }

            // ---- Per-block ----
            for (int i = 0; i < static_cast<int>(config.num_blocks); ++i) {
                const std::string layer_pfx = "model.layers." + std::to_string(i);
                const std::string layers_pfx = "layers." + std::to_string(i);

                // input_layernorm
                if (info.name == layer_pfx + ".input_layernorm.weight" ||
                    info.name == layers_pfx + ".input_layernorm.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/input_layernorm/gamma", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t N = tgt[-1];

                    if (static_cast<int64_t>(float_vec.size()) != N) {
                        throw std::runtime_error(
                            fmt::format("[attn LN] layer {} length mismatch: src={}, tgt={}", i, float_vec.size(), N));
                    }
                    out->set_value(core::from_vector(float_vec, tgt, out->get_value().device()));
                    return true;
                }

                // post_attention_layernorm
                if (info.name == layer_pfx + ".post_attention_layernorm.weight" ||
                    info.name == layers_pfx + ".post_attention_layernorm.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/post_attention_layernorm/gamma", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t N = tgt[-1];

                    if (static_cast<int64_t>(float_vec.size()) != N) {
                        throw std::runtime_error(
                            fmt::format("[mlp LN] layer {} length mismatch: src={}, tgt={}", i, float_vec.size(), N));
                    }
                    out->set_value(core::from_vector(float_vec, tgt, out->get_value().device()));
                    return true;
                }

                // q_norm.weight - CRITICAL for Qwen3 numerical stability
                if (info.name == layer_pfx + ".self_attn.q_norm.weight" ||
                    info.name == layers_pfx + ".self_attn.q_norm.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/self_attn/q_norm/gamma", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t N = tgt[-1];

                    if (static_cast<int64_t>(float_vec.size()) != N) {
                        throw std::runtime_error(
                            fmt::format("[q_norm] layer {} length mismatch: src={}, tgt={}", i, float_vec.size(), N));
                    }

                    std::vector<float> src = float_vec;
                    if (!meta_style) {
                        // Apply unpermute for norm weights (converts from meta format)
                        src = unpermute_norm_weights(src);
                    }

                    out->set_value(core::from_vector(src, tgt, out->get_value().device()));
                    return true;
                }

                // k_norm.weight - CRITICAL for Qwen3 numerical stability
                if (info.name == layer_pfx + ".self_attn.k_norm.weight" ||
                    info.name == layers_pfx + ".self_attn.k_norm.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/self_attn/k_norm/gamma", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t N = tgt[-1];

                    if (static_cast<int64_t>(float_vec.size()) != N) {
                        throw std::runtime_error(
                            fmt::format("[k_norm] layer {} length mismatch: src={}, tgt={}", i, float_vec.size(), N));
                    }

                    std::vector<float> src = float_vec;
                    if (!meta_style) {
                        // Apply unpermute for norm weights (converts from meta format)
                        src = unpermute_norm_weights(src);
                    }

                    out->set_value(core::from_vector(src, tgt, out->get_value().device()));
                    return true;
                }

                // q_proj.weight — optional unpermute (only if meta_style==false), then try T if needed
                if (info.name == layer_pfx + ".self_attn.q_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.q_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/self_attn/q_linear/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    std::vector<float> src = float_vec;
                    if (!meta_style) {
                        const int64_t rows = info.shape[0];  // num_heads * head_dim
                        const int64_t cols = info.shape[1];  // hidden_size
                        const int64_t n_h = static_cast<int64_t>(config.num_heads);
                        src = unpermute_proj_rows(src, rows, cols, n_h);
                    }

                    auto exact = strict_copy_linear(
                        src, info.shape[0], info.shape[1], tr, tc, fmt::format("q_proj layer {}", i), verbose);

                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                // k_proj.weight — load directly (no combining for Qwen3, we have separate k_linear)
                if (info.name == layer_pfx + ".self_attn.k_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.k_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/self_attn/k_linear/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    std::vector<float> src = float_vec;
                    if (!meta_style) {
                        const int64_t rows = info.shape[0];  // num_kv_heads * head_dim
                        const int64_t cols = info.shape[1];  // hidden_size
                        const int64_t n_kv = static_cast<int64_t>(config.num_groups);
                        src = unpermute_proj_rows(src, rows, cols, n_kv);
                    }

                    auto exact = strict_copy_linear(
                        src, info.shape[0], info.shape[1], tr, tc, fmt::format("k_proj layer {}", i), verbose);
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                // v_proj.weight — load directly (no combining for Qwen3, we have separate v_linear)
                if (info.name == layer_pfx + ".self_attn.v_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.v_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/self_attn/v_linear/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    // V projection doesn't need unpermute
                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("v_proj layer {}", i), verbose);
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                // o_proj.weight — try transpose if needed
                if (info.name == layer_pfx + ".self_attn.o_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.o_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/self_attn/out_linear/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("o_proj layer {}", i), verbose);
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                // MLP: w1 (gate), w3 (up), w2 (down) — try transpose if needed
                if (info.name == layer_pfx + ".mlp.gate_proj.weight" ||
                    info.name == layers_pfx + ".mlp.gate_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/mlp/w1/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("mlp.w1 layer {}", i), verbose);
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                if (info.name == layer_pfx + ".mlp.up_proj.weight" || info.name == layers_pfx + ".mlp.up_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/mlp/w3/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("mlp.w3 layer {}", i), verbose);
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                if (info.name == layer_pfx + ".mlp.down_proj.weight" ||
                    info.name == layers_pfx + ".mlp.down_proj.weight") {
                    auto name = fmt::format("qwen3/qwen3_block_{}/mlp/w2/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("mlp.w2 layer {}", i), verbose);
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }
            }

            // Unhandled tensor: ignore (e.g., lm_head.weight in some exports)
            ignored_parameters.insert(info.name);
            return true;
        };

    serialization::SafetensorSerialization::visit_safetensors_file(path, loading_callback);
}

}  // namespace ttml::models::qwen3
