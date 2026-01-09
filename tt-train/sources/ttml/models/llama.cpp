// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama.hpp"

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/llama_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "serialization/safetensors.hpp"

namespace {

static std::vector<float> pad_and_resize_flat(
    const std::vector<float>& flat, int64_t rows, int64_t cols, int64_t target_rows, int64_t target_cols) {
    // If dimensions match, return as is
    if (rows == target_rows && cols == target_cols) {
        return flat;
    }

    // Create output tensor with target dimensions
    std::vector<float> out(static_cast<size_t>(target_rows * target_cols), 0.0f);

    // Copy data from source to target, handling both row and column differences
    int64_t copy_rows = std::min(rows, target_rows);
    int64_t copy_cols = std::min(cols, target_cols);

    for (int64_t r = 0; r < copy_rows; ++r) {
        for (int64_t c = 0; c < copy_cols; ++c) {
            out[r * target_cols + c] = flat[r * cols + c];
        }
    }

    // Initialize random number generator once if we need to fill additional space
    bool need_random_fill = (target_rows > rows) || (target_cols > cols);
    std::mt19937 gen;
    std::normal_distribution<float> dist(0.0f, 0.02f);  // Small random values

    if (need_random_fill) {
        std::random_device rd;
        gen.seed(rd());
    }

    // For additional rows (if target_rows > rows), use small random initialization
    // instead of zeros to avoid dead neurons
    if (target_rows > rows) {
        for (int64_t r = copy_rows; r < target_rows; ++r) {
            for (int64_t c = 0; c < target_cols; ++c) {
                out[r * target_cols + c] = dist(gen);
            }
        }
    }

    // For additional columns (if target_cols > cols), use small random initialization
    if (target_cols > cols) {
        for (int64_t r = 0; r < copy_rows; ++r) {
            for (int64_t c = copy_cols; c < target_cols; ++c) {
                out[r * target_cols + c] = dist(gen);
            }
        }
    }

    return out;
}

static std::vector<float> unpermute_proj_rows(
    const std::vector<float>& w, int64_t rows, int64_t cols, int64_t n_heads) {
    // Reorder rows within each head: [0..D/2-1, D/2..D-1] → interleave → [0, D/2, 1, D/2+1, ...]
    if (rows % n_heads != 0) {
        throw std::runtime_error(
            fmt::format("unpermute_proj_rows: rows {} not divisible by n_heads {}", rows, n_heads));
    }
    const int64_t D = rows / n_heads;  // rows per head
    if (D % 2 != 0) {
        throw std::runtime_error(fmt::format("unpermute_proj_rows: rows per head {} must be even", D));
    }

    std::vector<float> out(w.size());
    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t head_row0 = h * D;
        const int64_t half = D / 2;
        for (int64_t i = 0; i < half; ++i) {
            const int64_t src_even = head_row0 + i;
            const int64_t src_odd = head_row0 + half + i;
            const int64_t dst_even = head_row0 + (2 * i);
            const int64_t dst_odd = head_row0 + (2 * i + 1);

            std::memcpy(&out[dst_even * cols], &w[src_even * cols], sizeof(float) * cols);
            std::memcpy(&out[dst_odd * cols], &w[src_odd * cols], sizeof(float) * cols);
        }
    }
    return out;
}

}  // namespace

namespace ttml::models::llama {

Llama::Llama(const LlamaConfig& config) : m_config(config) {
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

    fmt::print("Llama configuration:\n");
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
        /*head_dim=*/embedding_dim / num_heads,
        /*theta=*/theta,
        /*rope_scaling_params=*/rope_scaling_params);
    blocks.reserve(num_blocks);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        blocks.push_back(std::make_shared<ttml::modules::LlamaBlock>(
            embedding_dim, num_heads, num_groups, m_rope_params, dropout_prob, intermediate_dim));
    }
    ln_fc = std::make_shared<ttml::modules::RMSNormLayer>(embedding_dim);
    fc = last_fc;

    create_name("llama");
    register_module(tok_emb, "tok_emb");
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        register_module(blocks[block_idx], fmt::format("llama_block_{}", block_idx));
    }
    register_module(ln_fc, "ln_fc");
    register_module(fc, "fc");

    common::transformer::initialize_weights_gpt2(*this);
}

ttml::autograd::TensorPtr Llama::operator()(
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& mask,
    std::shared_ptr<common::transformer::KvCache> kv_cache,
    const uint32_t new_tokens) {
    // Pad input tokens to nearest multiple of 32 before embedding
    constexpr uint32_t TILE_SIZE = 32;
    auto x_shape = x->get_value().logical_shape();
    uint32_t actual_seq_len = x_shape[-1];  // Last dimension is sequence length
    uint32_t padded_seq_len = ((actual_seq_len + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    autograd::TensorPtr x_padded = x;
    if (padded_seq_len != actual_seq_len) {
        // Pad the sequence dimension (last dimension) with zeros
        // Create a new tensor instead of modifying in-place
        ttnn::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
            {0, 0},                               // batch dimension
            {0, 0},                               // first spatial dimension
            {0, 0},                               // second spatial dimension
            {0, padded_seq_len - actual_seq_len}  // sequence dimension
        };
        auto x_padded_tensor = ttnn::pad(x->get_value(), padding, 0.0f, false, std::nullopt);
        x_padded = autograd::create_tensor(x_padded_tensor);
    }

    auto tok_emb_out = (*tok_emb)(x_padded);

    // Unpad after embedding to restore original sequence length
    autograd::TensorPtr out = tok_emb_out;
    if (padded_seq_len != actual_seq_len) {
        // Slice back to original sequence length (sequence dimension is now at index 2)
        // Create a new tensor instead of modifying in-place
        ttnn::SmallVector<uint32_t> slice_start = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> slice_end = {
            tok_emb_out->get_value().logical_shape()[0],
            tok_emb_out->get_value().logical_shape()[1],
            actual_seq_len,
            tok_emb_out->get_value().logical_shape()[3]};
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
        auto out_tensor = ttnn::slice(tok_emb_out->get_value(), slice_start, slice_end, step);
        out = autograd::create_tensor(out_tensor);
    }

    // llama does positional embedding in the attention blocks

    if (kv_cache) {
        // Inference mode with KV cache
        for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx) {
            auto& block = blocks[block_idx];
            // Cast block to LlamaBlock to access the cache-aware operator
            auto llama_block = std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(block);
            out = (*llama_block)(out, mask, kv_cache, static_cast<uint32_t>(block_idx), new_tokens);
        }
    } else {
        // Training mode or inference without cache
        for (auto& block : blocks) {
            if (runner_type == RunnerType::MemoryEfficient) {
                out = common::transformer::memory_efficient_runner(*block, out, mask);
            } else if (runner_type == RunnerType::Default) {
                out = (*block)(out, mask);
            } else {
                throw std::runtime_error("Unknown runner type. Supported runner types ['default', 'memory_efficient']");
            }
        }
    }

    out = (*ln_fc)(out);
    auto logits = (*fc)(out);
    return logits;
}

LlamaConfig read_config(const YAML::Node& config) {
    LlamaConfig llama_config;
    // Use defaults from nanollama3
    llama_config.num_heads = config["num_heads"].as<uint32_t>(6U);
    llama_config.num_groups = config["num_groups"].as<uint32_t>(3U);
    llama_config.embedding_dim = config["embedding_dim"].as<uint32_t>(384U);
    if (config["intermediate_dim"]) {
        uint32_t intermediate_dim = config["intermediate_dim"].as<uint32_t>();
        llama_config.intermediate_dim = std::make_optional(intermediate_dim);
    }
    llama_config.dropout_prob = config["dropout_prob"].as<float>(0.0F);
    llama_config.num_blocks = config["num_blocks"].as<uint32_t>(6U);
    llama_config.vocab_size = config["vocab_size"].as<uint32_t>(96U);
    llama_config.max_sequence_length = config["max_sequence_length"].as<uint32_t>(256U);
    llama_config.theta = config["theta"].as<float>(500000.0F);
    llama_config.runner_type = common::transformer::read_runner_type(config);
    llama_config.weight_tying = common::transformer::read_weight_tying_type(config);

    // Read RoPE NTK-aware scaling parameters if they exist
    if (config["rope_scaling"]) {
        const auto& rope_scaling = config["rope_scaling"];
        if (rope_scaling["scaling_factor"]) {
            llama_config.scaling_factor = rope_scaling["scaling_factor"].as<float>();
        }
        if (rope_scaling["high_freq_factor"]) {
            llama_config.high_freq_factor = rope_scaling["high_freq_factor"].as<float>();
        }
        if (rope_scaling["low_freq_factor"]) {
            llama_config.low_freq_factor = rope_scaling["low_freq_factor"].as<float>();
        }
        if (rope_scaling["original_context_length"]) {
            llama_config.original_context_length = rope_scaling["original_context_length"].as<uint32_t>();
        }
    }
    return llama_config;
}

YAML::Node write_config(const LlamaConfig& llama_config) {
    YAML::Node config;
    config["num_heads"] = llama_config.num_heads;
    config["num_groups"] = llama_config.num_groups;
    config["embedding_dim"] = llama_config.embedding_dim;
    if (llama_config.intermediate_dim) {
        config["intermediate_dim"] = *llama_config.intermediate_dim;
    }
    config["dropout_prob"] = llama_config.dropout_prob;
    config["num_blocks"] = llama_config.num_blocks;
    config["vocab_size"] = llama_config.vocab_size;
    config["max_sequence_length"] = llama_config.max_sequence_length;
    config["theta"] = llama_config.theta;

    // Add RoPE scaling parameters if they are set
    if (llama_config.scaling_factor != 0.0F && llama_config.original_context_length != 0U) {
        YAML::Node rope_scaling;
        rope_scaling["scaling_factor"] = llama_config.scaling_factor;
        rope_scaling["high_freq_factor"] = llama_config.high_freq_factor;
        rope_scaling["low_freq_factor"] = llama_config.low_freq_factor;
        rope_scaling["original_context_length"] = llama_config.original_context_length;
        config["rope_scaling"] = rope_scaling;
    }

    return config;
}

std::shared_ptr<Llama> create(const LlamaConfig& config) {
    return std::make_shared<Llama>(config);
}
std::shared_ptr<Llama> create(const YAML::Node& config) {
    LlamaConfig llama_config = read_config(config);
    return std::make_shared<Llama>(llama_config);
}

void Llama::load_from_safetensors(const std::filesystem::path& model_path) {
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() == ".safetensors") {
            auto path = entry.path();
            fmt::print("Loading model from: {}\n", path.string());
            auto parameters = this->parameters();
            load_model_from_safetensors(path, parameters, m_config);
        }
    }
}

void load_model_from_safetensors(
    const std::filesystem::path& path, serialization::NamedParameters& parameters, const LlamaConfig& config) {
    // Keep your working setting
    const bool meta_style = false;

    // --- helpers -------------------------------------------------------------
    auto transpose_flat_ = [](const std::vector<float>& x, int64_t r, int64_t c) -> std::vector<float> {
        std::vector<float> y(x.size());
        for (int64_t i = 0; i < r; ++i)
            for (int64_t j = 0; j < c; ++j) y[(size_t)j * r + i] = x[(size_t)i * c + j];
        return y;
    };

    auto strict_copy_linear = [&](const std::vector<float>& src,
                                  int64_t src_r,
                                  int64_t src_c,
                                  int64_t tgt_r,
                                  int64_t tgt_c,
                                  const std::string& dbg) -> std::vector<float> {
        if (src_r == tgt_r && src_c == tgt_c) {
            return src;  // as-is fits
        }
        if (src_c == tgt_r && src_r == tgt_c) {
            fmt::print("[{}] transposing weights\n", dbg);
            return transpose_flat_(src, src_r, src_c);  // transposed fits
        }
        throw std::runtime_error(fmt::format(
            "[{}] shape mismatch: src=({}x{}), src^T=({}x{}), tgt=({}x{})",
            dbg,
            src_r,
            src_c,
            src_c,
            src_r,
            tgt_r,
            tgt_c));
    };
    // ------------------------------------------------------------------------

    std::set<std::string> used_parameters;
    std::set<std::string> ignored_parameters;

    // Stage K/V for row-wise concat into kv_linear
    std::map<int, std::vector<float>> k_weights, v_weights;
    std::map<int, std::array<int64_t, 2>> k_shapes, v_shapes;

    auto get_parameter = [&parameters, &used_parameters](const std::string& name) -> ttml::autograd::TensorPtr {
        auto it = parameters.find(name);

        if (it == parameters.end()) {
            throw std::runtime_error(fmt::format("Parameter {} not found in the model", name));
        }
        fmt::print("Using parameter: {} with shape: {}\n", name, it->second->get_value().logical_shape());
        used_parameters.insert(name);
        return it->second;
    };

    auto try_combine_kv_weights = [&](int layer_idx) {
        if (k_weights.find(layer_idx) == k_weights.end() || v_weights.find(layer_idx) == v_weights.end())
            return;

        auto block_name = fmt::format("llama/llama_block_{}/attention/kv_linear/weight", layer_idx);
        auto out_tensor = get_parameter(block_name);

        const auto tgt = out_tensor->get_value().logical_shape();
        const int64_t tr = tgt[-2], tc = tgt[-1];

        auto K = k_weights[layer_idx];
        auto V = v_weights[layer_idx];
        auto Ks = k_shapes[layer_idx];  // {rows, cols}
        auto Vs = v_shapes[layer_idx];

        // Try all four (K,V) orientation combos to match target exactly.
        auto try_orient = [&](bool kT, bool vT) -> bool {
            std::vector<float> Kx = kT ? transpose_flat_(K, Ks[0], Ks[1]) : K;
            std::vector<float> Vx = vT ? transpose_flat_(V, Vs[0], Vs[1]) : V;
            int64_t Kr = kT ? Ks[1] : Ks[0], Kc = kT ? Ks[0] : Ks[1];
            int64_t Vr = vT ? Vs[1] : Vs[0], Vc = vT ? Vs[0] : Vs[1];
            if (Kc != Vc)
                return false;
            int64_t cr = Kr + Vr, cc = Kc;
            if (cr != tr || cc != tc)
                return false;

            std::vector<float> combined;
            combined.reserve(Kx.size() + Vx.size());
            combined.insert(combined.end(), Kx.begin(), Kx.end());
            combined.insert(combined.end(), Vx.begin(), Vx.end());
            out_tensor->set_value(core::from_vector(combined, tgt, out_tensor->get_value().device()));
            return true;
        };

        if (!(try_orient(false, false) || try_orient(true, false) || try_orient(false, true) ||
              try_orient(true, true))) {
            throw std::runtime_error(fmt::format(
                "KV concat: cannot align at layer {}. "
                "K=({}x{}), V=({}x{}), tgt=({}x{})",
                layer_idx,
                Ks[0],
                Ks[1],
                Vs[0],
                Vs[1],
                tr,
                tc));
        }

        // cleanup
        k_weights.erase(layer_idx);
        v_weights.erase(layer_idx);
        k_shapes.erase(layer_idx);
        v_shapes.erase(layer_idx);

        fmt::print("Combined k_proj + v_proj → kv_linear for layer {}\n", layer_idx);
    };

    serialization::SafetensorSerialization::TensorCallback loading_callback =
        [&](const serialization::SafetensorSerialization::TensorInfo& info, std::span<const std::byte> bytes) {
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
                    (weight_tying == WeightTyingType::Enabled) ? "llama/fc/weight" : "llama/tok_emb/weight";
                auto out_tensor1 = get_parameter(embedding_weights_name);

                auto tgt = out_tensor1->get_value().logical_shape();

                auto resized_emb = pad_and_resize_flat(float_vec, info.shape[0], info.shape[1], tgt[-2], tgt[-1]);
                out_tensor1->set_value(core::from_vector(resized_emb, tgt, out_tensor1->get_value().device()));

                return true;
            }
            if (info.name == "lm_head.weight") {
                if (config.weight_tying == WeightTyingType::Disabled) {
                    auto out_tensor2 = get_parameter("llama/fc/weight");
                    auto tgt2 = out_tensor2->get_value().logical_shape();

                    auto resized_emb2 =
                        pad_and_resize_flat(float_vec, info.shape[0], info.shape[1], tgt2[-2], tgt2[-1]);
                    out_tensor2->set_value(core::from_vector(resized_emb2, tgt2, out_tensor2->get_value().device()));
                }
                return true;
            }
            // ---- Final LayerNorm (vector) ----
            if (info.name == "norm.weight" || info.name == "model.norm.weight") {
                auto out = get_parameter("llama/ln_fc/gamma");
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
                    auto name = fmt::format("llama/llama_block_{}/attention_norm/gamma", i);
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
                    auto name = fmt::format("llama/llama_block_{}/mlp_norm/gamma", i);
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

                // q_proj.weight — optional unpermute (only if meta_style==false), then try T if needed
                if (info.name == layer_pfx + ".self_attn.q_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.q_proj.weight") {
                    auto name = fmt::format("llama/llama_block_{}/attention/q_linear/weight", i);
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
                        src, info.shape[0], info.shape[1], tr, tc, fmt::format("q_proj layer {}", i));

                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                // k_proj.weight — stage; optional unpermute (GQA: use num_groups as kv heads)
                if (info.name == layer_pfx + ".self_attn.k_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.k_proj.weight") {
                    std::vector<float> src = float_vec;
                    if (!meta_style) {
                        const int64_t rows = info.shape[0];  // num_kv_heads * head_dim
                        const int64_t cols = info.shape[1];  // hidden_size
                        const int64_t n_kv = static_cast<int64_t>(config.num_groups);
                        src = unpermute_proj_rows(src, rows, cols, n_kv);
                    }
                    k_weights[i] = std::move(src);
                    k_shapes[i] = {info.shape[0], info.shape[1]};
                    try_combine_kv_weights(i);
                    return true;
                }

                // v_proj.weight — stage (no unpermute)
                if (info.name == layer_pfx + ".self_attn.v_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.v_proj.weight") {
                    std::vector<float> src = float_vec;
                    v_weights[i] = std::move(src);
                    v_shapes[i] = {info.shape[0], info.shape[1]};
                    try_combine_kv_weights(i);
                    return true;
                }

                // o_proj.weight — try transpose if needed
                if (info.name == layer_pfx + ".self_attn.o_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.o_proj.weight") {
                    auto name = fmt::format("llama/llama_block_{}/attention/out_linear/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("o_proj layer {}", i));
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                // MLP: w1 (gate), w3 (up), w2 (down) — try transpose if needed
                if (info.name == layer_pfx + ".mlp.gate_proj.weight" ||
                    info.name == layers_pfx + ".mlp.gate_proj.weight") {
                    auto name = fmt::format("llama/llama_block_{}/mlp/w1/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("mlp.w1 layer {}", i));
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                if (info.name == layer_pfx + ".mlp.up_proj.weight" || info.name == layers_pfx + ".mlp.up_proj.weight") {
                    auto name = fmt::format("llama/llama_block_{}/mlp/w3/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("mlp.w3 layer {}", i));
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }

                if (info.name == layer_pfx + ".mlp.down_proj.weight" ||
                    info.name == layers_pfx + ".mlp.down_proj.weight") {
                    auto name = fmt::format("llama/llama_block_{}/mlp/w2/weight", i);
                    auto out = get_parameter(name);
                    auto tgt = out->get_value().logical_shape();
                    const int64_t tr = tgt[-2], tc = tgt[-1];

                    auto exact = strict_copy_linear(
                        float_vec, info.shape[0], info.shape[1], tr, tc, fmt::format("mlp.w2 layer {}", i));
                    out->set_value(core::from_vector(exact, tgt, out->get_value().device()));
                    return true;
                }
            }

            // Unhandled tensor: ignore (e.g., lm_head.weight in some exports)
            ignored_parameters.insert(info.name);
            return true;
        };

    serialization::SafetensorSerialization::visit_safetensors_file(path, loading_callback);

    // Report unused parameters
    std::vector<std::string> unused_parameters;
    for (const auto& [param_name, _] : parameters) {
        if (used_parameters.find(param_name) == used_parameters.end())
            unused_parameters.push_back(param_name);
    }

    if (!unused_parameters.empty()) {
        fmt::print("Warning: The following parameters were not used during loading:\n");
        for (const auto& param_name : unused_parameters) fmt::print("  - {}\n", param_name);
        fmt::print("Total unused parameters: {}\n", unused_parameters.size());
    } else {
        fmt::print("All {} parameters were successfully loaded and used.\n", parameters.size());
    }
    if (!ignored_parameters.empty()) {
        fmt::print("Note: The following parameters were ignored during loading:\n");
        for (const auto& param_name : ignored_parameters) fmt::print("  - {}\n", param_name);
        fmt::print("Total ignored parameters: {}\n", ignored_parameters.size());
    }
}
}  // namespace ttml::models::llama
