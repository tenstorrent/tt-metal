// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_common.hpp"

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

}  // namespace ttml::models::common::transformer
