// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_common.hpp"

#include <cstring>
#include <random>

#include "core/tt_tensor_utils.hpp"
#include "yaml-cpp/yaml.h"

namespace ttml::models::common::transformer {

void initialize_weights_gpt2(ttml::modules::ModuleBase& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            init::normal_init(tensor_ptr, tensor.logical_shape(), {0.F, 0.02F});
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.logical_shape(), 0.F);
        }
    }
}

void initialize_weights_he_kaiming_normal(modules::ModuleBase& model) {
    auto params = model.parameters();
    for (auto& [name, tensor_ptr] : params) {
        const auto& tensor = tensor_ptr->get_value();
        if (name.find("weight") != std::string::npos) {
            auto mean = 0.0F;
            // take penultimate dimension as the input dim.
            auto fan_in = tensor.logical_shape()[-2];
            auto stddev = std::sqrt(2.0F / fan_in);
            init::normal_init(tensor_ptr, tensor.logical_shape(), {mean, stddev});
        } else if (name.find("bias") != std::string::npos) {
            init::constant_init(tensor_ptr, tensor.logical_shape(), 0.F);
        }
    }
}

RunnerType read_runner_type(const YAML::Node& config) {
    auto runner_type_str = config["runner_type"].as<std::string>("default");
    if (runner_type_str == "default") {
        return RunnerType::Default;
    } else if (runner_type_str == "memory_efficient") {
        return RunnerType::MemoryEfficient;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown runner type: {}. Supported runner types [default, memory_efficient]", runner_type_str));
    }
}

WeightTyingType read_weight_tying_type(const YAML::Node& config) {
    auto weight_tying_str = config["weight_tying"].as<std::string>("disabled");
    if (weight_tying_str == "disabled") {
        return WeightTyingType::Disabled;
    } else if (weight_tying_str == "enabled") {
        return WeightTyingType::Enabled;
    } else {
        throw std::runtime_error(fmt::format(
            "Unknown weight tying type: {}. Supported weight tying types [disabled, enabled]", weight_tying_str));
    }
}

KvCache::KvCache(
    const uint32_t num_layers,
    const uint32_t batch_size,
    const uint32_t num_groups,
    const uint32_t max_seq_len,
    const uint32_t head_dim) {
    fmt::print("Initializing KV cache:\n");
    fmt::print("    Batch size: {}\n", batch_size);
    fmt::print("    Num layers: {}\n", num_layers);
    fmt::print("    Num groups: {}\n", num_groups);
    fmt::print("    Max sequence length: {}\n", max_seq_len);
    fmt::print("    Head dim: {}\n", head_dim);

    m_kv_cache.clear();
    m_kv_cache.reserve(num_layers);

    // Create cache tensors in DRAM
    // Shape: [batch_size, num_groups, max_seq_len, head_dim]
    const auto dram_memory_config = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};

    for (uint32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto kv_cache_shape = ttnn::Shape({batch_size, num_groups, max_seq_len, head_dim});

        auto k_cache = ttnn::zeros(
            kv_cache_shape,
            ttnn::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            std::ref(autograd::ctx().get_device()),
            dram_memory_config);
        auto v_cache = ttnn::zeros(
            kv_cache_shape,
            ttnn::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            std::ref(autograd::ctx().get_device()),
            dram_memory_config);

        m_kv_cache.emplace_back(k_cache, v_cache);
    }

    m_cache_position = 0U;
    fmt::print("KV cache initialized successfully\n");
}

KvCache::KvCache(const KvCacheConfig& config) :
    KvCache(config.num_layers, config.batch_size, config.num_groups, config.max_seq_len, config.head_dim) {
}

const uint32_t KvCache::update_prefill(
    const tt::tt_metal::Tensor& key_tensor,
    const tt::tt_metal::Tensor& value_tensor,
    tt::tt_metal::Tensor& k_cache,
    tt::tt_metal::Tensor& v_cache,
    const uint32_t new_tokens) {
    const auto kv_shape = key_tensor.logical_shape();
    TT_FATAL(
        new_tokens <= key_tensor.logical_shape()[-2], "New tokens must be less than or equal to the sequence length");
    const auto cache_shape = k_cache.logical_shape();

    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> token_start = {0, 0, 0, 0};
    const ttnn::SmallVector<uint32_t> kv_end = {kv_shape[0], kv_shape[1], new_tokens, kv_shape[3]};

    const tt::tt_metal::Tensor& new_key = ttnn::slice(key_tensor, token_start, kv_end, step);
    const tt::tt_metal::Tensor& new_value = ttnn::slice(value_tensor, token_start, kv_end, step);

    const ttnn::SmallVector<uint32_t> cache_start = {0, 0, 0, 0};
    const ttnn::SmallVector<uint32_t> cache_end = {cache_shape[0], cache_shape[1], new_tokens, cache_shape[3]};

    ttnn::experimental::slice_write(new_key, k_cache, cache_start, cache_end, step);
    ttnn::experimental::slice_write(new_value, v_cache, cache_start, cache_end, step);

    return new_tokens;
}

const uint32_t KvCache::update_decode(
    const tt::tt_metal::Tensor& key_tensor,
    const tt::tt_metal::Tensor& value_tensor,
    tt::tt_metal::Tensor& k_cache,
    tt::tt_metal::Tensor& v_cache,
    const uint32_t cache_position,
    const uint32_t new_tokens) {
    const auto cache_shape = k_cache.logical_shape();
    const auto kv_shape = key_tensor.logical_shape();
    TT_FATAL(new_tokens <= kv_shape[-2], "New tokens must be less than or equal to the sequence length");

    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> token_start = {0, 0, 0, 0};
    const ttnn::SmallVector<uint32_t> kv_end = {kv_shape[0], kv_shape[1], new_tokens, kv_shape[3]};

    const tt::tt_metal::Tensor& new_key = ttnn::slice(key_tensor, token_start, kv_end, step);
    const tt::tt_metal::Tensor& new_value = ttnn::slice(value_tensor, token_start, kv_end, step);

    const ttnn::SmallVector<uint32_t> cache_start = {0, 0, cache_position, 0};
    const ttnn::SmallVector<uint32_t> cache_end = {
        cache_shape[0], cache_shape[1], cache_position + new_tokens, cache_shape[3]};

    ttnn::experimental::slice_write(new_key, k_cache, cache_start, cache_end, step);
    ttnn::experimental::slice_write(new_value, v_cache, cache_start, cache_end, step);

    return cache_position + new_tokens;
}

const uint32_t KvCache::update(
    const uint32_t layer_idx,
    const tt::tt_metal::Tensor& key_states,
    const tt::tt_metal::Tensor& value_states,
    const uint32_t new_tokens) {
    TT_FATAL(layer_idx < m_kv_cache.size(), "Layer index out of range");

    auto& [k_cache, v_cache] = m_kv_cache[layer_idx];
    const auto kv_shape = key_states.logical_shape();

    // - If cache_position == 0: prefill mode (write starting at position 0)
    uint32_t new_position;
    if (m_cache_position == 0U) {
        // Prefill mode: write entire sequence starting at position 0
        new_position = update_prefill(key_states, value_states, k_cache, v_cache, new_tokens);
    } else {
        // Decode mode: write new tokens starting at current cache position
        new_position = update_decode(key_states, value_states, k_cache, v_cache, m_cache_position, new_tokens);
    }

    // Update cache position only on last layer to ensure consistency across layers
    if (layer_idx == m_kv_cache.size() - 1) {
        m_cache_position = new_position;
    }

    return new_position;
}

// ============================================================================
// Weight Loading Utilities Implementation
// ============================================================================

std::vector<float> pad_and_resize_flat(
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

std::vector<float> unpermute_proj_rows(const std::vector<float>& w, int64_t rows, int64_t cols, int64_t n_heads) {
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

std::vector<float> unpermute_norm_weights(const std::vector<float>& w) {
    // For RMSNorm weights: reshape to (2, head_dim/2), transpose, then flatten
    // This converts from non-meta format ([x1,x2,...,y1,y2...]) to the meta-style
    // format expected by TTML ([x1,y1,x2,y2,...])
    const int64_t total_size = w.size();
    const int64_t head_dim = total_size;

    if (head_dim % 2 != 0) {
        throw std::runtime_error(fmt::format("unpermute_norm_weights: head_dim {} must be even", head_dim));
    }

    const int64_t half = head_dim / 2;
    std::vector<float> out(total_size);

    // Reshape to (2, half), transpose to (half, 2), then flatten
    // Original layout: w[i*half + j] where i in [0,1], j in [0,half)
    // After transpose: out[j*2 + i] where j in [0,half), i in [0,1]
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < half; ++j) {
            out[j * 2 + i] = w[i * half + j];
        }
    }

    return out;
}

std::vector<float> transpose_flat(const std::vector<float>& x, int64_t r, int64_t c) {
    std::vector<float> y(x.size());
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < c; ++j) {
            y[(size_t)j * r + i] = x[(size_t)i * c + j];
        }
    }
    return y;
}

std::vector<float> strict_copy_linear(
    const std::vector<float>& src,
    int64_t src_r,
    int64_t src_c,
    int64_t tgt_r,
    int64_t tgt_c,
    const std::string& dbg,
    bool verbose) {
    if (src_r == tgt_r && src_c == tgt_c) {
        return src;
    }
    if (src_c == tgt_r && src_r == tgt_c) {
        if (verbose) {
            fmt::print("[{}] transposing weights\n", dbg);
        }
        return transpose_flat(src, src_r, src_c);
    }
    throw std::runtime_error(fmt::format(
        "[{}] shape mismatch: src=({}x{}), src^T=({}x{}), tgt=({}x{})", dbg, src_r, src_c, src_c, src_r, tgt_r, tgt_c));
}

}  // namespace ttml::models::common::transformer
