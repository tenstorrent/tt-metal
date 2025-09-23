// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "deit_encoder.h"
#include <stdexcept>

TtDeiTEncoder::TtDeiTEncoder(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device), gradient_checkpointing(false) {
    
    // Initialize all encoder layers
    layers.reserve(config.num_hidden_layers);
    
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        std::string layer_address = base_address + ".layer." + std::to_string(i) + ".";
        layers.push_back(
            std::make_unique<TtDeiTLayer>(config, device, state_dict, layer_address)
        );
    }
}

std::tuple<ttnn::Tensor, std::optional<std::vector<ttnn::Tensor>>, std::optional<std::vector<ttnn::Tensor>>> 
TtDeiTEncoder::forward(
    const ttnn::Tensor& hidden_states,
    const std::optional<std::vector<ttnn::Tensor>>& head_mask,
    bool output_attentions,
    bool output_hidden_states,
    bool return_dict
) {
    // Initialize output containers
    std::optional<std::vector<ttnn::Tensor>> all_hidden_states = std::nullopt;
    std::optional<std::vector<ttnn::Tensor>> all_self_attentions = std::nullopt;
    
    if (output_hidden_states) {
        all_hidden_states = std::vector<ttnn::Tensor>();
    }
    
    if (output_attentions) {
        all_self_attentions = std::vector<ttnn::Tensor>();
    }
    
    // Current hidden states
    ttnn::Tensor current_hidden_states = hidden_states;
    
    // Process through all layers
    for (size_t i = 0; i < layers.size(); ++i) {
        // Store hidden states if requested
        if (output_hidden_states) {
            all_hidden_states->push_back(current_hidden_states);
        }
        
        // Get layer head mask if provided
        std::optional<ttnn::Tensor> layer_head_mask = std::nullopt;
        if (head_mask.has_value() && i < head_mask->size()) {
            layer_head_mask = (*head_mask)[i];
        }
        
        // Forward pass through current layer
        if (gradient_checkpointing && false) { // Training not supported yet
            throw std::runtime_error("No support for training yet!");
        } else {
            auto layer_outputs = layers[i]->forward(
                current_hidden_states,
                layer_head_mask,
                output_attentions
            );
            
            // Extract outputs
            current_hidden_states = std::get<0>(layer_outputs);
            auto attention_weights = std::get<1>(layer_outputs);
            
            // Store attention weights if requested
            if (output_attentions && attention_weights.has_value()) {
                all_self_attentions->push_back(*attention_weights);
            }
        }
    }
    
    // Store final hidden states if requested
    if (output_hidden_states) {
        all_hidden_states->push_back(current_hidden_states);
    }
    
    return std::make_tuple(
        current_hidden_states,
        all_hidden_states,
        all_self_attentions
    );
}