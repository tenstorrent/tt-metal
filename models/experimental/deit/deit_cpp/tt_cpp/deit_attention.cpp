// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_attention.h"
#include <stdexcept>

TtDeiTAttention::TtDeiTAttention(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device) {
    if (!device) {
        throw std::invalid_argument("Device cannot be null");
    }

    // Create attention component with appropriate base address
    std::string attention_address = base_address.empty() ? "attention" : base_address + ".attention";
    attention = std::make_unique<TtDeiTSelfAttention>(config, device, state_dict, attention_address);

    // Create output component with appropriate base address
    std::string output_address = base_address.empty() ? "output" : base_address + ".output";
    output = std::make_unique<TtDeiTSelfOutput>(config, device, state_dict, output_address);
}

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> TtDeiTAttention::forward(
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& head_mask,
    bool output_attentions
) {
    // Forward pass through self attention
    auto self_outputs = attention->forward(hidden_states, head_mask, output_attentions);
    
    // Extract attention output (first element of the tuple)
    ttnn::Tensor attention_hidden_states = std::get<0>(self_outputs);
    
    // Forward pass through output layer
    ttnn::Tensor attention_output = output->forward(attention_hidden_states, hidden_states);
    
    // Prepare return value
    if (output_attentions && std::get<1>(self_outputs).has_value()) {
        // Return both attention output and attention weights
        return std::make_tuple(attention_output, std::get<1>(self_outputs));
    } else {
        // Return only attention output
        return std::make_tuple(attention_output, std::nullopt);
    }
}