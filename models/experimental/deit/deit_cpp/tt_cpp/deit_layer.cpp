// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "deit_layer.h"
#include "../helper_funcs.h"
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
#include <stdexcept>


TtDeiTLayer::TtDeiTLayer(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device) {

    // Initialize sub-modules
    attention = std::make_unique<TtDeiTAttention>(
        config, device, state_dict, base_address + "attention."
    );
    intermediate = std::make_unique<TtDeiTIntermediate>(
        config, device, state_dict, base_address + "intermediate."
    );
    output = std::make_unique<TtDeiTOutput>(
        config, device, state_dict, base_address + "output."
    );

    // Load layer normalization parameters
    std::string ln_before_weight_key = base_address + "layernorm_before.weight";
    std::string ln_before_bias_key = base_address + "layernorm_before.bias";
    std::string ln_after_weight_key = base_address + "layernorm_after.weight";
    std::string ln_after_bias_key = base_address + "layernorm_after.bias";

    auto ln_before_weight_it = state_dict.find(ln_before_weight_key);
    auto ln_before_bias_it = state_dict.find(ln_before_bias_key);
    auto ln_after_weight_it = state_dict.find(ln_after_weight_key);
    auto ln_after_bias_it = state_dict.find(ln_after_bias_key);

    if (ln_before_weight_it == state_dict.end() || ln_before_bias_it == state_dict.end() ||
        ln_after_weight_it == state_dict.end() || ln_after_bias_it == state_dict.end()) {
        throw std::runtime_error("Layer normalization parameters not found in state_dict");
    }

    // Convert torch tensors to ttnn tensors
    layernorm_before_weight = helper_funcs::torch_to_tt_tensor_tile(ln_before_weight_it->second, device);
    layernorm_before_bias = helper_funcs::torch_to_tt_tensor_tile(ln_before_bias_it->second, device);
    layernorm_after_weight = helper_funcs::torch_to_tt_tensor_tile(ln_after_weight_it->second, device);
    layernorm_after_bias = helper_funcs::torch_to_tt_tensor_tile(ln_after_bias_it->second, device);
}

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> TtDeiTLayer::forward(
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& head_mask,
    bool output_attentions
) {
    // Apply layer normalization before self-attention (pre-norm)
    auto normalized_hidden_states = helper_funcs::apply_layernorm(
        hidden_states,
        layernorm_before_weight,
        layernorm_before_bias,
        config.layer_norm_eps
    );


    // Self-attention
    auto attention_outputs = attention->forward(
        normalized_hidden_states,
        head_mask,
        output_attentions
    );

    auto attention_output = std::get<0>(attention_outputs);
    auto attention_weights = std::get<1>(attention_outputs);

    // First residual connection
    auto residual_output = ttnn::add(attention_output, hidden_states);

    // Apply layer normalization after self-attention
    auto layer_output = helper_funcs::apply_layernorm(
        residual_output,
        layernorm_after_weight,
        layernorm_after_bias,
        config.layer_norm_eps
    );


    // Intermediate layer (feed-forward)
    layer_output = intermediate->forward(layer_output);

    // Second residual connection (done in output layer)
    layer_output = output->forward(layer_output, residual_output);

    return std::make_tuple(layer_output, attention_weights);
}
