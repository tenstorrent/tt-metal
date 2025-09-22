// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_pooler.h"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <stdexcept>

TtDeiTPooler::TtDeiTPooler(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config_(config), device_(device) {
    // Construct weight and bias keys
    std::string weight_key = base_address + "dense.weight";
    std::string bias_key = base_address + "dense.bias";
    
    // Check if weights exist in state_dict
    auto weight_it = state_dict.find(weight_key);
    auto bias_it = state_dict.find(bias_key);
    
    if (weight_it == state_dict.end()) {
        throw std::runtime_error("Weight not found in state_dict: " + weight_key);
    }
    if (bias_it == state_dict.end()) {
        throw std::runtime_error("Bias not found in state_dict: " + bias_key);
    }
    
    // Convert torch tensors to ttnn tensors with TILE layout
    dense_weight_ = helper_funcs::torch_to_tt_tensor_tile(weight_it->second, device_);
    dense_bias_ = helper_funcs::torch_to_tt_tensor_tile(bias_it->second, device_);
}

ttnn::Tensor TtDeiTPooler::forward(const ttnn::Tensor& hidden_states) {
    // Extract the first token (CLS token) from the sequence
    // hidden_states shape: [batch_size, seq_len, hidden_size]
    // We want to slice [:, 0, :] to get [batch_size, hidden_size]
    
    // Get tensor shape using correct API
    auto shape = hidden_states.logical_shape();
    uint32_t rank = shape.rank();
    
    // Create slice parameters for extracting the first token
    ttnn::SmallVector<uint32_t> slice_start(rank, 0);
    ttnn::SmallVector<uint32_t> slice_end(rank);
    ttnn::SmallVector<uint32_t> slice_step(rank, 1);
    
    // Set slice_end to full shape initially
    for (uint32_t i = 0; i < rank; ++i) {
        slice_end[i] = shape[i];
    }
    
    // Slice the sequence dimension (dimension 1) to get only the first token
    // Assuming shape is [batch_size, seq_len, hidden_size] or [1, batch_size, seq_len, hidden_size]
    if (rank >= 3) {
        // For 3D or 4D tensors, sequence dimension is typically the second-to-last or specific position
        uint32_t seq_dim = rank - 2; // Sequence dimension
        slice_end[seq_dim] = 1; // Take only the first token
    } else {
        throw std::runtime_error("Input tensor must have at least 3 dimensions for pooling operation");
    }
    
    // Perform the slice operation
    ttnn::Tensor first_token_tensor = ttnn::slice(hidden_states, slice_start, slice_end, slice_step);
    
    // Reshape to remove the sequence dimension (which is now 1)
    // Get the new shape after slicing
    auto current_shape = first_token_tensor.logical_shape();
    std::vector<uint32_t> new_shape_vec;
    
    // Copy dimensions, skipping the sequence dimension that has size 1
    for (uint32_t i = 0; i < current_shape.rank(); ++i) {
        if (current_shape[i] != 1 || i == 0 || i == current_shape.rank() - 1) {
            // Keep batch and hidden dimensions, skip sequence dimension with size 1
            new_shape_vec.push_back(current_shape[i]);
        }
    }
    
    // Ensure we have at least 4 dimensions for TILE layout
    while (new_shape_vec.size() < 4) {
        new_shape_vec.insert(new_shape_vec.begin(), 1);
    }
    
    ttnn::Shape reshape_shape(new_shape_vec);
    first_token_tensor = ttnn::reshape(first_token_tensor, reshape_shape);
    
    // Apply linear transformation: dense(first_token_tensor)
    ttnn::Tensor pooled_output = helper_funcs::linear_transform(
        first_token_tensor,
        dense_weight_,
        dense_bias_
    );
    
    // Apply tanh activation
    pooled_output = ttnn::tanh(pooled_output);
    
    return pooled_output;
}