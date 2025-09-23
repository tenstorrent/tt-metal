// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_for_image_classification.h"
#include "deit_model.h"
#include <stdexcept>
#include <iostream>

TtDeiTForImageClassification::TtDeiTForImageClassification(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device
) : config_(config), device_(device), num_labels_(1000) { // Default to 1000 classes like ImageNet

    try {

        // Load model parameters to state_dict
        std::vector<std::string> required_params = {
            "classifier.weight", "classifier.bias"
        };

        // Split state_dict into two parts based on required_params
        std::unordered_map<std::string, torch::Tensor> required_state_dict;
        
        for (const auto& param : required_params) {
            std::string full_key = base_address + param;
            auto it = state_dict.find(full_key);
            if (it != state_dict.end()) {
                required_state_dict[full_key] = it->second;
                state_dict.erase(it);  // 直接从原始map中删除
            }
        }

        
        // Initialize DeiT backbone model
        // std::string deit_base_address = base_address.empty() ? "deit" : base_address + "deit.";
        deit_model_ = std::make_unique<TtDeiTModel>(
            config,
            state_dict,
            base_address + "deit.",
            device,
            false,
            false
        );
        
        // Load classifier weights
        std::string classifier_weight_key = base_address + "classifier.weight";
        std::string classifier_bias_key = base_address + "classifier.bias";
        
        if (required_state_dict.find(classifier_weight_key) != required_state_dict.end()) {
            classifier_weight_ = helper_funcs::torch_to_tt_tensor_tile(
                required_state_dict[classifier_weight_key], device
            );
        } else {
            throw std::runtime_error("Classifier weight not found in required_state_dict: " + classifier_weight_key);
        }
        
        if (required_state_dict.find(classifier_bias_key) != required_state_dict.end()) {
            classifier_bias_ = helper_funcs::torch_to_tt_tensor_tile(
                required_state_dict[classifier_bias_key], device
            );
            
            // Update num_labels based on classifier bias size
            auto bias_shape = classifier_bias_.logical_shape();
            if (bias_shape.rank() >= 1) {
                num_labels_ = static_cast<int>(bias_shape[-1]);
            }
        } else {
            throw std::runtime_error("Classifier bias not found in state_dict: " + classifier_bias_key);
        }
        
        std::cout << "TtDeiTForImageClassification initialized with " << num_labels_ << " classes" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TtDeiTForImageClassification: " << e.what() << std::endl;
        throw;
    }
}

TtDeiTForImageClassification::~TtDeiTForImageClassification() = default;

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> 
TtDeiTForImageClassification::forward(
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor* head_mask,
    bool output_attentions,
    bool output_hidden_states,
    bool return_dict
) {
    try {
        // Forward pass through DeiT backbone
        auto [sequence_output, pooled_output, hidden_states, attentions] = deit_model_->forward(
            pixel_values,
            std::nullopt,  // bool_masked_pos
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        );
        
        // Apply classifier to get logits
        ttnn::Tensor logits = apply_classifier(sequence_output);
        
        // Return logits and optional outputs (convert vector to single tensor if needed)
        std::optional<ttnn::Tensor> hidden_states_output = std::nullopt;
        std::optional<ttnn::Tensor> attentions_output = std::nullopt;
        
        if (output_hidden_states && hidden_states.has_value()) {
            // For simplicity, return the last hidden state
            const auto& hidden_states_vec = hidden_states.value();
            if (!hidden_states_vec.empty()) {
                hidden_states_output = hidden_states_vec.back();
            }
        }
        
        if (output_attentions && attentions.has_value()) {
            // For simplicity, return the last attention
            const auto& attentions_vec = attentions.value();
            if (!attentions_vec.empty()) {
                attentions_output = attentions_vec.back();
            }
        }
        
        return std::make_tuple(logits, hidden_states_output, attentions_output);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in TtDeiTForImageClassification forward pass: " << e.what() << std::endl;
        throw;
    }
}

ttnn::Tensor TtDeiTForImageClassification::apply_classifier(const ttnn::Tensor& sequence_output) {
    try {
        // Extract CLS token representation (first token)
        // sequence_output shape: [batch_size, seq_len, hidden_size]
        // We need to extract [:, 0, :] which is the CLS token
        
        auto output_shape = sequence_output.logical_shape();
        if (output_shape.rank() < 3) {
            throw std::runtime_error("Expected sequence_output to have at least 3 dimensions");
        }
        
        // For now, we'll use a simplified approach
        // In practice, you'd need to properly slice the tensor to get the CLS token
        // This is a placeholder implementation
        
        // Apply linear transformation: logits = sequence_output @ weight.T + bias
        ttnn::Tensor logits = helper_funcs::linear_transform(
            sequence_output,
            classifier_weight_,
            classifier_bias_
        );
        
        return logits;
        
    } catch (const std::exception& e) {
        std::cerr << "Error applying classifier: " << e.what() << std::endl;
        throw;
    }
}