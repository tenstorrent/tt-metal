// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_for_image_classification.h"
#include <stdexcept>
#include <iostream>

// Forward declaration for TtDeiTModel
class TtDeiTModel {
public:
    TtDeiTModel(const DeiTConfig& config, std::unordered_map<std::string, torch::Tensor>& state_dict, 
                const std::string& base_address, std::shared_ptr<ttnn::MeshDevice> device) {}
    
    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> forward(
        const ttnn::Tensor& pixel_values,
        const ttnn::Tensor* head_mask = nullptr,
        bool output_attentions = false,
        bool output_hidden_states = false,
        bool return_dict = true
    ) {
        // Placeholder implementation
        return std::make_tuple(pixel_values, std::nullopt, std::nullopt);
    }
};

TtDeiTForImageClassification::TtDeiTForImageClassification(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device
) : config_(config), device_(device), num_labels_(1000) { // Default to 1000 classes like ImageNet
    
    try {
        // Initialize DeiT backbone model
        std::string deit_base_address = base_address.empty() ? "deit" : base_address + "deit";
        deit_model_ = new TtDeiTModel(
            config,
            state_dict,
            deit_base_address,
            device
        );
        
        // Load classifier weights
        std::string classifier_weight_key = base_address + "classifier.weight";
        std::string classifier_bias_key = base_address + "classifier.bias";
        
        if (state_dict.find(classifier_weight_key) != state_dict.end()) {
            classifier_weight_ = helper_funcs::torch_to_tt_tensor_tile(
                state_dict[classifier_weight_key], device
            );
        } else {
            throw std::runtime_error("Classifier weight not found in state_dict: " + classifier_weight_key);
        }
        
        if (state_dict.find(classifier_bias_key) != state_dict.end()) {
            classifier_bias_ = helper_funcs::torch_to_tt_tensor_tile(
                state_dict[classifier_bias_key], device
            );
            
            // Update num_labels based on classifier bias size
            auto bias_shape = classifier_bias_.get_logical_shape();
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
        auto [sequence_output, attentions, hidden_states] = deit_model_->forward(
            pixel_values,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        );
        
        // Apply classifier to get logits
        ttnn::Tensor logits = apply_classifier(sequence_output);
        
        // Return logits and optional outputs
        return std::make_tuple(
            logits,
            output_attentions ? attentions : std::nullopt,
            output_hidden_states ? hidden_states : std::nullopt
        );
        
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
        
        auto output_shape = sequence_output.get_logical_shape();
        if (output_shape.rank() < 3) {
            throw std::runtime_error("Expected sequence_output to have at least 3 dimensions");
        }
        
        uint32_t batch_size = output_shape[0];
        uint32_t hidden_size = output_shape[2];
        
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

std::shared_ptr<TtDeiTForImageClassification> create_deit_for_image_classification(
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::string& model_path
) {
    try {
        // Default configuration for DeiT-base
        DeiTConfig config;
        config.hidden_size = 768;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 12;
        config.intermediate_size = 3072;
        config.image_size = 224;
        config.patch_size = 16;
        config.num_channels = 3;
        
        // For now, create empty state_dict
        // In practice, you'd load this from a pretrained model file
        std::unordered_map<std::string, torch::Tensor> state_dict;
        
        // Create dummy weights for testing
        // classifier.weight: [num_labels, hidden_size]
        // classifier.bias: [num_labels]
        int num_labels = 1000; // ImageNet classes
        
        state_dict["classifier.weight"] = torch::randn({num_labels, config.hidden_size});
        state_dict["classifier.bias"] = torch::randn({num_labels});
        
        std::cout << "Creating TtDeiTForImageClassification with dummy weights" << std::endl;
        std::cout << "Note: Load actual pretrained weights for real usage" << std::endl;
        
        return std::make_shared<TtDeiTForImageClassification>(
            config,
            state_dict,
            "", // base_address
            device
        );
        
    } catch (const std::exception& e) {
        std::cerr << "Error creating TtDeiTForImageClassification: " << e.what() << std::endl;
        throw;
    }
}