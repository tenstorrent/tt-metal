// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_query_attention.hpp"

#include <core/ttnn_all_includes.hpp>

#include "dropout_module.hpp"
#include "linear_module.hpp"
#include "modules/rotary_embedding.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache.hpp"

namespace ttml::modules {

GroupedQueryAttention::GroupedQueryAttention(const GQAConfig& config) :
    m_embedding_dim(config.embedding_dim), m_num_heads(config.num_heads), m_num_groups(config.num_groups) {
    // create layers
    m_q_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim, config.bias_linears);
    auto concat_kv_dim = 2U * m_num_groups * (m_embedding_dim / m_num_heads);
    m_kv_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, concat_kv_dim, config.bias_linears);
    m_dropout = std::make_shared<ttml::modules::DropoutLayer>(config.dropout_prob);
    m_out_linear = std::make_shared<ttml::modules::LinearLayer>(m_embedding_dim, m_embedding_dim, config.bias_linears);
    m_embedding = std::make_shared<ttml::modules::RotaryEmbedding>(config.rope_params);

    // register modules
    create_name("grouped_query_attention");
    register_module(m_q_linear, "q_linear");
    register_module(m_kv_linear, "kv_linear");
    register_module(m_dropout, "dropout");
    register_module(m_out_linear, "out_linear");
    register_module(m_embedding, "embedding");
}

ttnn::MemoryConfig GroupedQueryAttention::create_sharded_memory_config(
    const ttnn::Tensor& tensor, ttnn::distributed::MeshDevice* device) const {
    auto kv_shape = tensor.logical_shape();
    uint32_t num_heads = kv_shape[1];
    uint32_t seq_len = kv_shape[2];
    uint32_t head_dim = kv_shape[3];

    // Get device compute grid
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t cores_x = compute_grid.x;
    uint32_t cores_y = compute_grid.y;
    uint32_t total_cores = cores_x * cores_y;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;

    // Calculate shard height and round up to tile boundary
    uint32_t shard_height = (num_heads * seq_len) / total_cores;
    shard_height = ((shard_height + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;

    // Ensure at least one tile in height
    if (shard_height == 0) {
        shard_height = TILE_HEIGHT;
    }

    // Round width to tile boundary as well
    uint32_t shard_width = ((head_dim + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;

    // Create core range covering the entire grid
    ttnn::CoreRangeSet core_ranges({ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(cores_x - 1, cores_y - 1))});

    // Create HEIGHT_SHARDED memory config
    return ttnn::MemoryConfig(
        ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
        ttnn::BufferType::L1,
        tt::tt_metal::ShardSpec(core_ranges, {shard_height, shard_width}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x, const ttml::autograd::TensorPtr& mask) {
    // Standard attention without KV cache
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads);
        key_with_heads = (*m_embedding)(key_with_heads);
    }

    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, key_with_heads, value_with_heads, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

ttml::autograd::TensorPtr GroupedQueryAttention::operator()(
    const ttml::autograd::TensorPtr& x,
    const ttml::autograd::TensorPtr& mask,
    const ttml::autograd::TensorPtr& k_cache,
    const ttml::autograd::TensorPtr& v_cache,
    uint32_t cache_position) {
    // Compute query, key, value projections
    auto q = (*m_q_linear)(x);
    auto kv = (*m_kv_linear)(x);

    auto [query_with_heads, key_with_heads, value_with_heads] =
        ops::grouped_heads_creation(q, kv, m_num_heads, m_num_groups);

    // Apply rotary positional embedding
    if (m_embedding) {
        query_with_heads = (*m_embedding)(query_with_heads);
        key_with_heads = (*m_embedding)(key_with_heads);
    }

    // Get underlying tensors
    auto key_tensor = key_with_heads->get_value();
    auto value_tensor = value_with_heads->get_value();
    auto device = key_tensor.device();

    if (cache_position == 0) {
        // PREFILL: Fill cache with entire prompt sequence
        // Convert K,V to sharded layout to support arbitrary sequence lengths (fill_cache requirement same as in python
        // llama)
        auto sharded_config = create_sharded_memory_config(key_tensor, device);
        auto key_sharded = ttnn::interleaved_to_sharded(key_tensor, sharded_config, std::nullopt);
        auto value_sharded = ttnn::interleaved_to_sharded(value_tensor, sharded_config, std::nullopt);

        // Fill cache tensors (operates in-place on DRAM cache)
        auto k_cache_tensor = k_cache->get_value();
        auto v_cache_tensor = v_cache->get_value();
        ttnn::fill_cache(k_cache_tensor, key_sharded, /*batch_index=*/0);
        ttnn::fill_cache(v_cache_tensor, value_sharded, /*batch_index=*/0);

        // Update autograd wrappers with filled cache
        k_cache->set_value(k_cache_tensor);
        v_cache->set_value(v_cache_tensor);

        ttnn::deallocate(key_sharded);
        ttnn::deallocate(value_sharded);
    } else {
        // DECODE: Update cache with single new token at cache_position
        auto kv_shape = key_tensor.logical_shape();

        // Extract single token from K,V (at position 0 in padded input)
        ttnn::SmallVector<uint32_t> start = {0, 0, 0, 0};
        ttnn::SmallVector<uint32_t> end = {kv_shape[0], kv_shape[1], 1, kv_shape[3]};
        ttnn::SmallVector<uint32_t> step = {1, 1, 1, 1};

        auto single_key = ttnn::reshape(
            ttnn::slice(key_tensor, start, end, step), ttnn::Shape({1, kv_shape[0], kv_shape[1], kv_shape[3]}));
        auto single_value = ttnn::reshape(
            ttnn::slice(value_tensor, start, end, step), ttnn::Shape({1, kv_shape[0], kv_shape[1], kv_shape[3]}));

        // Convert to sharded layout for paged update
        auto sharded_config = create_sharded_memory_config(single_key, device);
        auto single_key_sharded = ttnn::interleaved_to_sharded(single_key, sharded_config, std::nullopt);
        auto single_value_sharded = ttnn::interleaved_to_sharded(single_value, sharded_config, std::nullopt);

        // Create cache position tensor
        auto cache_position_tensor = ttnn::full(
            ttnn::Shape({1}), static_cast<int32_t>(cache_position), ttnn::DataType::INT32, std::nullopt, *device);

        // Update cache at specific position
        auto k_cache_tensor = k_cache->get_value();
        auto v_cache_tensor = v_cache->get_value();
        ttnn::experimental::paged_update_cache(
            k_cache_tensor,
            single_key_sharded,
            std::vector<uint32_t>{},
            cache_position_tensor,
            std::nullopt,
            std::nullopt,
            0,
            std::nullopt,
            std::nullopt);
        ttnn::experimental::paged_update_cache(
            v_cache_tensor,
            single_value_sharded,
            std::vector<uint32_t>{},
            cache_position_tensor,
            std::nullopt,
            std::nullopt,
            0,
            std::nullopt,
            std::nullopt);

        // Update autograd wrappers
        k_cache->set_value(k_cache_tensor);
        v_cache->set_value(v_cache_tensor);

        ttnn::deallocate(single_key_sharded);
        ttnn::deallocate(single_value_sharded);
        ttnn::deallocate(cache_position_tensor);
    }

    // Compute attention using cached K,V
    auto attention = ttml::ops::scaled_dot_product_attention(query_with_heads, k_cache, v_cache, mask);
    attention = ops::heads_fusion(attention);

    auto out = (*m_out_linear)(attention);
    out = (*m_dropout)(out);

    return out;
}

}  // namespace ttml::modules
