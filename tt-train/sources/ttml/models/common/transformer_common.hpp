// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/scoped.hpp"
#include "init/tensor_initializers.hpp"
#include "modules/module_base.hpp"

namespace ttml::models::common::transformer {

enum class RunnerType {
    MemoryEfficient,
    Default,
};

enum class WeightTyingType {
    Disabled,
    Enabled,
};

autograd::TensorPtr memory_efficient_runner(
    auto&& forward_impl, const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED) {
        return forward_impl(input, mask);
    }

    // make a copy of a generator before running forward pass
    auto generator = autograd::ctx().get_generator();

    // running forward pass
    autograd::TensorPtr out;
    {
        auto scoped = ttml::core::Scoped(
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::DISABLED); },
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::ENABLED); });
        out = forward_impl(input, mask);
    }

    // define grad function and copy generator (in the state before forward pass)
    autograd::GradFunction grad = [input, mask, out, &forward_impl, generator]() {
        // detach input from existing graph
        auto input_detached = autograd::create_tensor(input->get_value());
        // run forward pass again
        autograd::TensorPtr output;
        {
            // set generator to the state before forward pass during construction
            // restore generator state after grad function is executed
            auto scoped = ttml::core::Scoped(
                [&generator]() { autograd::ctx().set_generator(generator); },
                [generator = autograd::ctx().get_generator()]() { autograd::ctx().set_generator(generator); });
            output = forward_impl(input_detached, mask);
        }
        // use gradients from new output
        output->set_grad(out->get_grad());
        output->backward();
        // reuse gradients from detached input
        input->add_grad(input_detached->get_grad());
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

void initialize_weights_gpt2(ttml::modules::ModuleBase& model);
void initialize_weights_he_kaiming_normal(ttml::modules::ModuleBase& model);

RunnerType read_runner_type(const YAML::Node& config);
WeightTyingType read_weight_tying_type(const YAML::Node& config);

/**
 * @brief Configuration structure for KV Cache
 */
struct KvCacheConfig {
    uint32_t num_layers;   ///< Number of transformer layers
    uint32_t batch_size;   ///< Batch size for the cache
    uint32_t num_groups;   ///< Number of KV groups (for GQA)
    uint32_t max_seq_len;  ///< Maximum sequence length
    uint32_t head_dim;     ///< Head dimension
    KvCacheConfig(
        uint32_t num_layers, uint32_t batch_size, uint32_t num_groups, uint32_t max_seq_len, uint32_t head_dim) :
        num_layers(num_layers),
        batch_size(batch_size),
        num_groups(num_groups),
        max_seq_len(max_seq_len),
        head_dim(head_dim) {
    }
};

/**
 * @brief KV Cache class for managing key-value cache storage and updates
 *
 * This class provides explicit external cache management (allocation/storage/updates)
 * that can be passed to models during inference. It handles both prefill (arbitrary
 * number of new tokens) and decode (single token) scenarios with a unified interface.
 *
 * The cache stores KV pairs per layer, and automatically handles cache position tracking.
 */
class KvCache {
public:
    explicit KvCache(const KvCacheConfig& config);

    KvCache(
        const uint32_t num_layers,
        const uint32_t batch_size,
        const uint32_t num_groups,
        const uint32_t max_seq_len,
        const uint32_t head_dim);

    const uint32_t update(
        const uint32_t layer_idx,
        const tt::tt_metal::Tensor& key_states,
        const tt::tt_metal::Tensor& value_states,
        const uint32_t new_tokens);

    /**
     * @brief Get the K cache tensor for a specific layer
     *
     * @param layer_idx Layer index (0-based)
     * @return const ttnn::Tensor& K cache tensor
     */
    [[nodiscard]] const ttnn::Tensor& get_k_cache(const uint32_t layer_idx) const {
        if (layer_idx >= m_kv_cache.size()) {
            throw std::runtime_error(
                fmt::format("Layer index {} out of range (max: {})", layer_idx, m_kv_cache.size() - 1));
        }
        return m_kv_cache[layer_idx].first;
    }

    /**
     * @brief Get the V cache tensor for a specific layer
     *
     * @param layer_idx Layer index (0-based)
     * @return const ttnn::Tensor& V cache tensor
     */
    [[nodiscard]] const ttnn::Tensor& get_v_cache(const uint32_t layer_idx) const {
        if (layer_idx >= m_kv_cache.size()) {
            throw std::runtime_error(
                fmt::format("Layer index {} out of range (max: {})", layer_idx, m_kv_cache.size() - 1));
        }
        return m_kv_cache[layer_idx].second;
    }

    /**
     * @brief Get the current cache position
     *
     * @return uint32_t Current position in cache
     */
    [[nodiscard]] const uint32_t get_cache_position() const {
        return m_cache_position;
    }

    /**
     * @brief Reset the cache position to start (for new sequence)
     */
    void reset() {
        m_cache_position = 0U;
    }

    /**
     * @brief Clear all cache data and reset position
     */
    void clear() {
        m_kv_cache.clear();
        m_cache_position = 0U;
    }

    /**
     * @brief Get the number of layers in the cache
     *
     * @return uint32_t Number of layers
     */
    [[nodiscard]] const uint32_t num_layers() const {
        return static_cast<uint32_t>(m_kv_cache.size());
    }

    /**
     * @brief Check if cache is empty
     *
     * @return true if cache is empty
     */
    [[nodiscard]] bool empty() const {
        return m_kv_cache.empty();
    }

private:
    // KV cache storage: [(k_cache, v_cache)] per layer
    std::vector<std::pair<ttnn::Tensor, ttnn::Tensor>> m_kv_cache;
    uint32_t m_cache_position = 0U;  // Current position in cache

    /**
     * @brief Update cache for prefill mode (writes entire sequence starting at position 0)
     */
    const uint32_t update_prefill(
        const tt::tt_metal::Tensor& key_tensor,
        const tt::tt_metal::Tensor& value_tensor,
        tt::tt_metal::Tensor& k_cache,
        tt::tt_metal::Tensor& v_cache,
        const uint32_t new_tokens);

    /**
     * @brief Update cache for decode mode (writes single token at cache_position)
     */
    const uint32_t update_decode(
        const tt::tt_metal::Tensor& key_tensor,
        const tt::tt_metal::Tensor& value_tensor,
        tt::tt_metal::Tensor& k_cache,
        tt::tt_metal::Tensor& v_cache,
        const uint32_t cache_position,
        const uint32_t new_tokens = 1);
};

}  // namespace ttml::models::common::transformer
