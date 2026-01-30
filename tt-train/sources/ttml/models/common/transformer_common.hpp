// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <string>
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

class KvCache;

autograd::TensorPtr memory_efficient_runner(
    auto&& forward_impl,
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& mask,
    std::shared_ptr<KvCache> kv_cache,
    const uint32_t layer_idx,
    const uint32_t new_tokens) {
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED) {
        return forward_impl(input, mask, kv_cache, layer_idx, new_tokens);
    }

    auto generator = autograd::ctx().get_generator();

    autograd::TensorPtr out;
    {
        auto scoped = ttml::core::Scoped(
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::DISABLED); },
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::ENABLED); });
        out = forward_impl(input, mask, kv_cache, layer_idx, new_tokens);
    }

    autograd::GradFunction grad = [input, mask, kv_cache, layer_idx, new_tokens, out, &forward_impl, generator]() {
        auto input_detached = autograd::create_tensor(input->get_value());
        autograd::TensorPtr output;
        {
            auto scoped = ttml::core::Scoped(
                [&generator]() { autograd::ctx().set_generator(generator); },
                [generator = autograd::ctx().get_generator()]() { autograd::ctx().set_generator(generator); });
            output = forward_impl(input_detached, mask, kv_cache, layer_idx, new_tokens);
        }
        output->set_grad(out->get_grad());
        output->backward();
        input->add_grad(input_detached->get_grad());
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

autograd::TensorPtr memory_efficient_runner(
    auto&& forward_impl, const autograd::TensorPtr& input, const autograd::TensorPtr& mask) {
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED) {
        return forward_impl(input, mask);
    }

    auto generator = autograd::ctx().get_generator();

    autograd::TensorPtr out;
    {
        auto scoped = ttml::core::Scoped(
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::DISABLED); },
            []() { autograd::ctx().set_gradient_mode(autograd::GradMode::ENABLED); });
        out = forward_impl(input, mask);
    }

    autograd::GradFunction grad = [input, mask, out, &forward_impl, generator]() {
        auto input_detached = autograd::create_tensor(input->get_value());
        autograd::TensorPtr output;
        {
            auto scoped = ttml::core::Scoped(
                [&generator]() { autograd::ctx().set_generator(generator); },
                [generator = autograd::ctx().get_generator()]() { autograd::ctx().set_generator(generator); });
            output = forward_impl(input_detached, mask);
        }
        output->set_grad(out->get_grad());
        output->backward();
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

inline std::pair<autograd::TensorPtr, autograd::TensorPtr> update_kv_cache_and_get_slices(
    std::shared_ptr<KvCache>& kv_cache,
    const uint32_t layer_idx,
    const autograd::TensorPtr& key_with_heads,
    const autograd::TensorPtr& value_with_heads,
    const autograd::TensorPtr& mask,
    const uint32_t new_tokens) {
    kv_cache->update(layer_idx, key_with_heads->get_value(), value_with_heads->get_value(), new_tokens);

    const auto& k_cache = kv_cache->get_k_cache(layer_idx);
    const auto& v_cache = kv_cache->get_v_cache(layer_idx);

    const ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::SmallVector<uint32_t> token_start = {0, 0, 0, 0};
    const auto cache_shape = k_cache.logical_shape();
    const ttnn::SmallVector<uint32_t> token_end = {
        cache_shape[0], cache_shape[1], mask->get_value().logical_shape()[-1], cache_shape[3]};

    const auto& k_cache_slice = ttnn::slice(k_cache, token_start, token_end, step);
    const auto& v_cache_slice = ttnn::slice(v_cache, token_start, token_end, step);

    return {autograd::create_tensor(k_cache_slice), autograd::create_tensor(v_cache_slice)};
}

// ============================================================================
// Weight Loading Utilities
// ============================================================================
// These utilities are used for loading transformer weights from safetensors,
// handling format conversions required for RoPE (rotary position embedding) and
// dimension adjustments.

/**
 * @brief Pad and resize a flat weight vector to target dimensions
 *
 * Copies data from source to target, padding with small random values for any
 * additional rows/columns to avoid dead neurons.
 *
 * @param flat Input flat vector of weights
 * @param rows Source number of rows
 * @param cols Source number of columns
 * @param target_rows Target number of rows
 * @param target_cols Target number of columns
 * @return std::vector<float> Resized weight vector
 */
std::vector<float> pad_and_resize_flat(
    const std::vector<float>& flat, int64_t rows, int64_t cols, int64_t target_rows, int64_t target_cols);

/**
 * @brief Unpermute projection weight rows for RoPE compatibility
 *
 * Reorders rows within each head from the non-meta format [0..D/2-1, D/2..D-1]
 * to interleaved format [0, D/2, 1, D/2+1, ...] expected by TTML RoPE.
 *
 * This is required for Q and K projection weights when loading from HuggingFace
 * format (non-meta style).
 *
 * @param w Input weight vector
 * @param rows Number of rows (num_heads * head_dim for Q, num_kv_heads * head_dim for K)
 * @param cols Number of columns (hidden_size)
 * @param n_heads Number of heads (num_heads for Q, num_kv_heads for K)
 * @return std::vector<float> Unpermuted weight vector
 */
std::vector<float> unpermute_proj_rows(const std::vector<float>& w, int64_t rows, int64_t cols, int64_t n_heads);

/**
 * @brief Unpermute RMSNorm weights for RoPE compatibility
 *
 * For RMSNorm weights (like Qwen3's q_norm/k_norm): reshape to (2, head_dim/2),
 * transpose, then flatten. This converts from non-meta format ([x1,x2,...,y1,y2...])
 * to the meta-style format expected by TTML ([x1,y1,x2,y2,...]).
 *
 * @param w Input weight vector
 * @return std::vector<float> Unpermuted weight vector
 */
std::vector<float> unpermute_norm_weights(const std::vector<float>& w);

/**
 * @brief Transpose a flat 2D weight matrix
 *
 * @param x Input flat vector
 * @param r Number of rows
 * @param c Number of columns
 * @return std::vector<float> Transposed flat vector
 */
std::vector<float> transpose_flat(const std::vector<float>& x, int64_t r, int64_t c);

/**
 * @brief Copy linear weights with optional transpose for shape matching
 *
 * Tries to match source weights to target dimensions, transposing if needed.
 * Throws if neither original nor transposed shape matches target.
 *
 * @param src Source weight vector
 * @param src_r Source rows
 * @param src_c Source columns
 * @param tgt_r Target rows
 * @param tgt_c Target columns
 * @param dbg Debug identifier for error messages
 * @param verbose If true, print when transposing
 * @return std::vector<float> Weights matching target dimensions
 */
std::vector<float> strict_copy_linear(
    const std::vector<float>& src,
    int64_t src_r,
    int64_t src_c,
    int64_t tgt_r,
    int64_t tgt_c,
    const std::string& dbg,
    bool verbose = false);

}  // namespace ttml::models::common::transformer
