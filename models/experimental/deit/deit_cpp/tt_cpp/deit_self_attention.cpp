// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_self_attention.h"
#include "../helper_funcs.h"
#include <optional>
#include <stdexcept>
#include <cmath>
#include <torch/torch.h>


TtDeiTSelfAttention::TtDeiTSelfAttention(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device) {
    if (!device) {
        throw std::invalid_argument("Device cannot be null");
    }

    // Validate configuration
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::invalid_argument(
            "The hidden size (" + std::to_string(config.hidden_size) +
            ") is not a multiple of the number of attention heads (" +
            std::to_string(config.num_attention_heads) + ")."
        );
    }

    // Set attention parameters
    num_attention_heads = config.num_attention_heads;
    attention_head_size = config.hidden_size / config.num_attention_heads;
    all_head_size = num_attention_heads * attention_head_size;

    // Load weights and biases from state_dict
    std::string query_weight_key = base_address + "query.weight";
    std::string query_bias_key = base_address + "query.bias";
    std::string key_weight_key = base_address + "key.weight";
    std::string key_bias_key = base_address + "key.bias";
    std::string value_weight_key = base_address + "value.weight";
    std::string value_bias_key = base_address + "value.bias";

    auto query_weight_it = state_dict.find(query_weight_key);
    auto query_bias_it = state_dict.find(query_bias_key);
    auto key_weight_it = state_dict.find(key_weight_key);
    auto key_bias_it = state_dict.find(key_bias_key);
    auto value_weight_it = state_dict.find(value_weight_key);
    auto value_bias_it = state_dict.find(value_bias_key);

    if (query_weight_it == state_dict.end() || key_weight_it == state_dict.end() || value_weight_it == state_dict.end()) {
        throw std::runtime_error("Required weight parameters not found in state_dict for base_address: " + base_address);
    }

    query_weight = helper_funcs::torch_to_tt_tensor_tile(query_weight_it->second, device);
    query_bias = (query_bias_it != state_dict.end()) ? std::optional<ttnn::Tensor>(helper_funcs::torch_to_tt_tensor_tile(query_bias_it->second, device)) : std::nullopt;

    key_weight = helper_funcs::torch_to_tt_tensor_tile(key_weight_it->second, device);
    key_bias = (key_bias_it != state_dict.end()) ? std::optional<ttnn::Tensor>(helper_funcs::torch_to_tt_tensor_tile(key_bias_it->second, device)) : std::nullopt;

    value_weight = helper_funcs::torch_to_tt_tensor_tile(value_weight_it->second, device);
    value_bias = (value_bias_it != state_dict.end()) ? std::optional<ttnn::Tensor>(helper_funcs::torch_to_tt_tensor_tile(value_bias_it->second, device)) : std::nullopt;
}

ttnn::Tensor TtDeiTSelfAttention::transpose_for_scores(ttnn::Tensor& x) {
    // Convert to ROW_MAJOR layout for reshape (matching Python implementation)
    x = ttnn::to_layout(x, ttnn::ROW_MAJOR_LAYOUT);

    // Get current shape
    auto current_shape = x.logical_shape();

    // Create new shape following Python logic: list(x.shape)[1:-1] + [num_attention_heads, attention_head_size]
    // This means we take dimensions from index 1 to second-to-last, then append the attention dimensions
    // For input [1, 1, 198, 768], Python takes [1, 198] and adds [12, 64] -> [1, 198, 12, 64]
    std::vector<uint32_t> new_shape_vec;

    // Add middle dimensions (from index 1 to second-to-last)
    // For input [1, 1, 198, 768], this should take [1, 198] (indices 1 to 2)
    for (size_t i = 1; i < current_shape.size() - 1; ++i) {
        new_shape_vec.push_back(static_cast<uint32_t>(current_shape[i]));
    }

    // Add attention head dimensions
    new_shape_vec.push_back(static_cast<uint32_t>(this->num_attention_heads));
    new_shape_vec.push_back(static_cast<uint32_t>(this->attention_head_size));

    // Create ttnn::Shape from vector
    ttnn::Shape new_shape(new_shape_vec);


    // Use fallback reshape operation (matching Python implementation)
    // Since we don't have direct access to fallback_ops in C++, we'll use ttnn::reshape
    // but ensure the tensor is in the right layout
    auto reshaped = ttnn::reshape(x, new_shape);

    // Convert back to TILE layout
    reshaped = ttnn::to_layout(reshaped, ttnn::TILE_LAYOUT);

    // Debug: Print final tensor shape before permute
    auto final_shape = reshaped.logical_shape();

    // Transpose dimensions: should always be 4D following Python logic
    // Python uses: ttnn.permute(x, (0, 2, 1, 3))
    // [batch, middle_dim, num_heads, head_size] -> [batch, num_heads, middle_dim, head_size]
    ttnn::SmallVector<int64_t> permute_dims = {0, 2, 1, 3};

    auto transposed = ttnn::permute(reshaped, permute_dims);


    return transposed;
}

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> TtDeiTSelfAttention::forward(
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& head_mask,
    bool output_attentions
) {
    // Compute query, key, value projections using helper linear function
    auto query_layer = helper_funcs::linear_transform(hidden_states, query_weight, query_bias);
    auto key_layer = helper_funcs::linear_transform(hidden_states, key_weight, key_bias);
    auto value_layer = helper_funcs::linear_transform(hidden_states, value_weight, value_bias);

    // Transpose for attention computation
    query_layer = transpose_for_scores(query_layer);
    key_layer = transpose_for_scores(key_layer);
    value_layer = transpose_for_scores(value_layer);

    // Compute attention scores
    auto key_transposed = ttnn::transpose(key_layer, -2, -1);
    auto attention_scores = ttnn::matmul(query_layer, key_transposed);

    // Scale attention scores
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(attention_head_size));
    attention_scores = ttnn::multiply(attention_scores, scale_factor);

    // Apply softmax to get attention probabilities
    auto attention_probs = ttnn::softmax(attention_scores, -1);
    attention_probs = ttnn::to_layout(attention_probs, ttnn::TILE_LAYOUT);

    // Apply head mask if provided
    if (head_mask.has_value()) {
        attention_probs = ttnn::multiply(attention_probs, head_mask.value());
    }

    // Compute context layer
    auto context_layer = ttnn::matmul(attention_probs, value_layer);

    // Python uses: ttnn.permute(context_layer, (0, 2, 1, 3))
    // [batch, num_heads, middle_dim, head_size] -> [batch, middle_dim, num_heads, head_size]
    ttnn::SmallVector<int64_t> transpose_dims = {0, 2, 1, 3};

    context_layer = ttnn::permute(context_layer, transpose_dims);

    // Reshape to original hidden size
    auto padded_shape = context_layer.padded_shape();

    // Create new_context_layer_shape similar to Python version: (1,) + padded_shape[:-2] + (all_head_size,)
    // context_layer is [1, 198, 12, 64], we want [1, 198, 768]
    ttnn::Shape new_context_layer_shape({1, padded_shape[1], static_cast<uint32_t>(all_head_size)});
    auto final_context = ttnn::reshape(ttnn::DefaultQueueId, context_layer, new_context_layer_shape);
    final_context = ttnn::to_layout(final_context, ttnn::TILE_LAYOUT);

    // Return results
    if (output_attentions) {
        return std::make_tuple(final_context, attention_probs);
    } else {
        return std::make_tuple(final_context, std::nullopt);
    }
}
