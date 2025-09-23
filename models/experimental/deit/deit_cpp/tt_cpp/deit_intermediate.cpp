// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "deit_intermediate.h"
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <stdexcept>


TtDeiTIntermediate::TtDeiTIntermediate(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device), activation_type(config.hidden_act) {
    
    // Load dense layer weights and bias from state_dict
    std::string weight_key = base_address + "dense.weight";
    std::string bias_key = base_address + "dense.bias";
    
    auto weight_it = state_dict.find(weight_key);
    auto bias_it = state_dict.find(bias_key);
    
    if (weight_it == state_dict.end()) {
        throw std::runtime_error("Dense weight not found in state_dict: " + weight_key);
    }
    
    // Convert torch tensors to ttnn tensors
    dense_weight = helper_funcs::torch_to_tt_tensor_tile(weight_it->second, device);
    dense_bias = (bias_it != state_dict.end()) ? std::optional<ttnn::Tensor>(helper_funcs::torch_to_tt_tensor_tile(bias_it->second, device)) : std::nullopt;
}

ttnn::Tensor TtDeiTIntermediate::forward(const ttnn::Tensor& hidden_states) {
    // Apply dense linear transformation: weight * input + bias
    auto output = helper_funcs::linear_transform(
        hidden_states,
        dense_weight,
        dense_bias
    );
    
    // Apply activation function
    output = apply_activation(output);
    
    return output;
}

ttnn::Tensor TtDeiTIntermediate::apply_activation(const ttnn::Tensor& input) {
    if (activation_type == "gelu") {
        return ttnn::gelu(input);
    } else if (activation_type == "relu") {
        return ttnn::relu(input);
    } else if (activation_type == "silu" || activation_type == "swish") {
        return ttnn::silu(input);
    } else if (activation_type == "tanh") {
        return ttnn::tanh(input);
    } else {
        throw std::runtime_error("Unsupported activation function: " + activation_type);
    }
}
