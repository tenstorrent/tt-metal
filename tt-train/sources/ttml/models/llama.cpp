// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/llama_block.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/serializable.hpp"

namespace {

static std::vector<float> transpose_2d_flat(const std::vector<float>& flat, int64_t rows, int64_t cols) {
    assert(rows * cols == static_cast<int64_t>(flat.size()));
    std::vector<int> shape_vec = {static_cast<int>(rows), static_cast<int>(cols)};
    auto src = xt::adapt(flat, shape_vec);
    xt::xarray<float> t = xt::transpose(src);
    auto view = ttml::core::xtensor_to_span(t);
    return std::vector<float>(view.begin(), view.end());
}

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

static void validate_weight_distribution(const std::vector<float>& weights, const std::string& weight_name) {
    if (weights.empty()) {
        fmt::print("[WARNING] Weight {} is empty\n", weight_name);
        return;
    }

    // Calculate basic statistics
    float min_val = *std::min_element(weights.begin(), weights.end());
    float max_val = *std::max_element(weights.begin(), weights.end());
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    float mean = sum / weights.size();

    // Calculate standard deviation
    float sq_sum = std::inner_product(weights.begin(), weights.end(), weights.begin(), 0.0f);
    float variance = sq_sum / weights.size() - mean * mean;
    float std_dev = std::sqrt(variance);

    // Count zeros and extreme values
    int zero_count = std::count(weights.begin(), weights.end(), 0.0f);
    int extreme_count = std::count_if(weights.begin(), weights.end(), [](float val) { return std::abs(val) > 10.0f; });

    fmt::print(
        "[WEIGHT_VALIDATION] {}: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}, zeros={}/{}, extreme_vals={}\n",
        weight_name,
        min_val,
        max_val,
        mean,
        std_dev,
        zero_count,
        weights.size(),
        extreme_count);

    // Check for potential issues
    if (zero_count > weights.size() * 0.5) {
        fmt::print("[WARNING] Weight {} has >50% zeros, this may cause issues\n", weight_name);
    }
    if (extreme_count > 0) {
        fmt::print(
            "[WARNING] Weight {} has {} extreme values (>10.0), this may cause issues\n", weight_name, extreme_count);
    }
    if (std::abs(mean) > 1.0f) {
        fmt::print("[WARNING] Weight {} has large mean ({:.6f}), this may cause issues\n", weight_name, mean);
    }
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

    // Safely calculate vocab_size divisible by 32, avoiding potential overflow
    if (vocab_size > UINT32_MAX - 31) {
        throw std::logic_error(
            fmt::format("Vocab size {} is too large and would cause overflow when rounding to 32", vocab_size));
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
    // Ensure consistent vocab size between embedding and output layers
    auto last_fc =
        std::make_shared<ttml::modules::LinearLayer>(embedding_dim, vocab_size_divisible_by_32, /* bias */ false);
    if (config.weight_tying == WeightTyingType::Enabled) {
        tok_emb = std::make_shared<ttml::modules::Embedding>(last_fc->get_weight());
    } else {
        tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_size_divisible_by_32, embedding_dim);
    }

    // Store the original vocab size for token validation
    m_original_vocab_size = vocab_size;

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

ttml::autograd::TensorPtr Llama::operator()(const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    auto tok_emb_out = (*tok_emb)(x);
    auto out = tok_emb_out;  // llama does positional embedding in the attention blocks
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

// --- helpers (leave your transpose_2d_flat, pad_and_resize_flat, unpermute_proj_rows, validate_* as-is) ---

// For LINEAR weights (everything except embeddings and tied-lm_head): require exact shape match.
static std::vector<float> strict_copy_linear(
    const std::vector<float>& flat,
    int64_t rows,
    int64_t cols,
    int64_t target_rows,
    int64_t target_cols,
    fmt::string_view debug_name) {
    if (rows != target_rows || cols != target_cols) {
        throw std::runtime_error(fmt::format(
            "[{}] Linear weight shape mismatch: src=({}x{}), tgt=({}x{})",
            debug_name,
            rows,
            cols,
            target_rows,
            target_cols));
    }
    return flat;  // exact fit
}

void load_model_from_safetensors(
    const std::filesystem::path& path, serialization::NamedParameters& parameters, const LlamaConfig& config) {
    // meta_style=true => NO Q/K unpermute (assumes interleaved layout already)
    const bool meta_style = false;

    std::set<std::string> used_parameters;

    // Stage K/V for row-wise concat into kv_linear
    std::map<int, std::vector<float>> k_weights, v_weights;
    std::map<int, std::array<int64_t, 2>> k_shapes, v_shapes;

    auto get_parameter = [&parameters, &used_parameters](const std::string& name) -> ttml::autograd::TensorPtr {
        auto it = parameters.find(name);
        if (it == parameters.end()) {
            throw std::runtime_error(fmt::format("Parameter {} not found in the model", name));
        }
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

        auto& K = k_weights[layer_idx];
        auto& V = v_weights[layer_idx];
        auto Ks = k_shapes[layer_idx];  // {rows, cols}
        auto Vs = v_shapes[layer_idx];

        if (Ks[1] != Vs[1]) {
            throw std::runtime_error(
                fmt::format("KV concat: k_cols != v_cols at layer {} ({} vs {})", layer_idx, Ks[1], Vs[1]));
        }

        // Concatenate rows: [K; V]  (NO transpose here)
        std::vector<float> combined;
        combined.reserve(K.size() + V.size());
        combined.insert(combined.end(), K.begin(), K.end());
        combined.insert(combined.end(), V.begin(), V.end());

        const int64_t cr = Ks[0] + Vs[0];
        const int64_t cc = Ks[1];

        // For kv_linear we ALSO enforce exact match; if target is different, it's a real mismatch.
        auto exact = strict_copy_linear(combined, cr, cc, tr, tc, fmt::format("kv_linear layer {}", layer_idx));

        out_tensor->set_value(core::from_vector(exact, tgt, out_tensor->get_value().device()));

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

            // --- dtype decode ---
            std::vector<float> float_vec;
            std::string dtype = info.dtype;
            for (auto& c : dtype) c = std::toupper(static_cast<unsigned char>(c));

            if (dtype == "BF16" || dtype == "BFLOAT16") {
                if (bytes.size_bytes() % 2 != 0)
                    throw std::runtime_error("BF16 data size must be even");
                const std::size_t n = bytes.size_bytes() / 2;
                float_vec.reserve(n);
                const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(bytes.data());
                for (std::size_t i = 0; i < n; ++i) {
                    uint32_t tmp = static_cast<uint32_t>(bf16_data[i]) << 16;
                    float value;
                    std::memcpy(&value, &tmp, sizeof(value));
                    float_vec.push_back(value);
                }
            } else if (dtype == "F32" || dtype == "FLOAT32") {
                float_vec = serialization::SafetensorSerialization::bytes_to_floats_copy(bytes);
            } else {
                throw std::runtime_error(fmt::format("Unsupported dtype: {}", info.dtype));
            }

            // ---- Embeddings (allow pad/resize) ----
            if (info.name == "embed_tokens.weight" || info.name == "model.embed_tokens.weight" ||
                info.name == "transformer.wte.weight" || info.name == "wte.weight" || info.name == "model.wte.weight" ||
                info.name == "embeddings.word_embeddings.weight") {
                validate_weight_distribution(float_vec, fmt::format("original_{}", info.name));

                auto out_tensor1 = get_parameter("llama/tok_emb/weight");
                auto tgt = out_tensor1->get_value().logical_shape();

                // auto resized_emb = pad_and_resize_flat(
                //     float_vec, info.shape[0], info.shape[1], tgt[-2], tgt[-1]);

                validate_weight_distribution(float_vec, "resized_embedding_weight");

                out_tensor1->set_value(core::from_vector(float_vec, tgt, out_tensor1->get_value().device()));

                if (config.weight_tying == WeightTyingType::Disabled) {
                    // lm_head = transpose(emb) with pad/resize
                    auto out_tensor2 = get_parameter("llama/fc/weight");
                    auto tgt2 = out_tensor2->get_value().logical_shape();

                    // auto transposed_emb = transpose_2d_flat(float_vec, info.shape[0], info.shape[1]);  // (emb,
                    // vocab)
                    //  auto resized_w2 = pad_and_resize_flat(
                    //      transposed_emb, info.shape[1], info.shape[0], tgt2[-2], tgt2[-1]);

                    out_tensor2->set_value(core::from_vector(float_vec, tgt2, out_tensor2->get_value().device()));
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

                // q_proj.weight — optional interleave/unpermute (only if meta_style==false)
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

                // v_proj.weight — stage (NO unpermute)
                if (info.name == layer_pfx + ".self_attn.v_proj.weight" ||
                    info.name == layers_pfx + ".self_attn.v_proj.weight") {
                    v_weights[i] = float_vec;
                    v_shapes[i] = {info.shape[0], info.shape[1]};
                    try_combine_kv_weights(i);
                    return true;
                }

                // o_proj.weight — strict shape
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

                // MLP: w1 (gate), w3 (up), w2 (down) — strict shapes
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

            // Unhandled tensor: just ignore.
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
}

}  // namespace ttml::models::llama
