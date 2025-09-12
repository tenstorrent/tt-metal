// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_for_image_classification_with_teacher.h"
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

TtDeiTForImageClassificationWithTeacher::TtDeiTForImageClassificationWithTeacher(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device
) : config_(config), device_(device), num_labels_(1000) {
    
    // Initialize DeiT backbone model
    // Note: TtDeiTModel should be implemented separately
    deit_model_ = new TtDeiTModel(config, state_dict, base_address + "deit.", device);
    
    // Load classification head weights
    std::string cls_weight_key = base_address + "cls_classifier.weight";
    std::string cls_bias_key = base_address + "cls_classifier.bias";
    std::string dist_weight_key = base_address + "distillation_classifier.weight";
    std::string dist_bias_key = base_address + "distillation_classifier.bias";
    
    if (state_dict.find(cls_weight_key) != state_dict.end()) {
        cls_classifier_weight_ = helper_funcs::torch_to_tt_tensor_tile(
            state_dict[cls_weight_key], device
        );
    } else {
        throw std::runtime_error("Classification head weight not found: " + cls_weight_key);
    }
    
    if (state_dict.find(cls_bias_key) != state_dict.end()) {
        cls_classifier_bias_ = helper_funcs::torch_to_tt_tensor_tile(
            state_dict[cls_bias_key], device
        );
    } else {
        throw std::runtime_error("Classification head bias not found: " + cls_bias_key);
    }
    
    if (state_dict.find(dist_weight_key) != state_dict.end()) {
        distillation_classifier_weight_ = helper_funcs::torch_to_tt_tensor_tile(
            state_dict[dist_weight_key], device
        );
    } else {
        throw std::runtime_error("Distillation head weight not found: " + dist_weight_key);
    }
    
    if (state_dict.find(dist_bias_key) != state_dict.end()) {
        distillation_classifier_bias_ = helper_funcs::torch_to_tt_tensor_tile(
            state_dict[dist_bias_key], device
        );
    } else {
        throw std::runtime_error("Distillation head bias not found: " + dist_bias_key);
    }
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> 
TtDeiTForImageClassificationWithTeacher::forward(
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor* head_mask,
    bool output_attentions,
    bool output_hidden_states,
    bool return_dict
) {
    // Forward pass through DeiT backbone
    auto deit_outputs = deit_model_->forward(
        pixel_values, head_mask, output_attentions, output_hidden_states, return_dict
    );
    
    // Extract sequence output (last hidden state)
    ttnn::Tensor sequence_output = std::get<0>(deit_outputs);
    
    // Extract CLS token (first token) for classification
    ttnn::Tensor cls_token = extract_cls_token(sequence_output);
    
    // Extract distillation token (second token) for distillation
    ttnn::Tensor distillation_token = extract_distillation_token(sequence_output);
    
    // Apply classification heads
    ttnn::Tensor cls_logits = apply_classifier(
        cls_token, cls_classifier_weight_, cls_classifier_bias_
    );
    
    ttnn::Tensor distillation_logits = apply_classifier(
        distillation_token, distillation_classifier_weight_, distillation_classifier_bias_
    );
    
    // Average the logits for final prediction
    ttnn::Tensor averaged_logits = average_logits(cls_logits, distillation_logits);
    
    // Extract optional outputs
    std::optional<ttnn::Tensor> attentions = std::nullopt;
    std::optional<ttnn::Tensor> hidden_states = std::nullopt;
    
    if (output_attentions && std::get<1>(deit_outputs).has_value()) {
        attentions = std::get<1>(deit_outputs);
    }
    
    if (output_hidden_states && std::get<2>(deit_outputs).has_value()) {
        hidden_states = std::get<2>(deit_outputs);
    }
    
    return std::make_tuple(averaged_logits, cls_logits, distillation_logits, attentions, hidden_states);
}

std::pair<ttnn::Tensor, ttnn::Tensor> TtDeiTForImageClassificationWithTeacher::get_separate_logits(
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor* head_mask
) {
    // Forward pass through DeiT backbone (no optional outputs needed)
    auto deit_outputs = deit_model_->forward(pixel_values, head_mask, false, false, true);
    
    // Extract sequence output
    ttnn::Tensor sequence_output = std::get<0>(deit_outputs);
    
    // Extract tokens
    ttnn::Tensor cls_token = extract_cls_token(sequence_output);
    ttnn::Tensor distillation_token = extract_distillation_token(sequence_output);
    
    // Apply classification heads
    ttnn::Tensor cls_logits = apply_classifier(
        cls_token, cls_classifier_weight_, cls_classifier_bias_
    );
    
    ttnn::Tensor distillation_logits = apply_classifier(
        distillation_token, distillation_classifier_weight_, distillation_classifier_bias_
    );
    
    return std::make_pair(cls_logits, distillation_logits);
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::apply_classifier(
    const ttnn::Tensor& sequence_output,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias
) {
    // Apply linear transformation: output = input @ weight.T + bias
    ttnn::Tensor logits = helper_funcs::linear_transform(sequence_output, weight, bias);
    return logits;
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::extract_cls_token(const ttnn::Tensor& sequence_output) {
    // Extract the first token (CLS token) from sequence output
    // sequence_output shape: [batch_size, seq_len, hidden_size]
    // cls_token shape: [batch_size, hidden_size]
    
    // Use indexing to extract first token
    auto shape = sequence_output.get_logical_shape();
    uint32_t batch_size = shape[0];
    uint32_t hidden_size = shape[2];
    
    // Create a simple slice by reusing the input tensor for now
    // This is a placeholder implementation
    ttnn::Tensor reshaped_cls = sequence_output;
    
    return reshaped_cls;
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::extract_distillation_token(const ttnn::Tensor& sequence_output) {
    // Extract the second token (distillation token) from sequence output
    // sequence_output shape: [batch_size, seq_len, hidden_size]
    // distillation_token shape: [batch_size, hidden_size]
    
    // Use indexing to extract second token
    auto shape = sequence_output.get_logical_shape();
    uint32_t batch_size = shape[0];
    uint32_t hidden_size = shape[2];
    
    // Create a simple slice by reusing the input tensor for now
    // This is a placeholder implementation
    ttnn::Tensor reshaped_dist = sequence_output;
    
    return reshaped_dist;
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::average_logits(
    const ttnn::Tensor& logits1, 
    const ttnn::Tensor& logits2
) {
    // Average two logit tensors: (logits1 + logits2) / 2
    ttnn::Tensor sum = ttnn::add(logits1, logits2);
    
    // Divide by 2 using scalar multiplication
    ttnn::Tensor averaged = ttnn::multiply(sum, 0.5f);
    return averaged;
}

std::shared_ptr<TtDeiTForImageClassificationWithTeacher> create_deit_for_image_classification_with_teacher(
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::string& model_path
) {
    // Load default configuration
    DeiTConfig config;
    
    // Load state dictionary from model path
    // This is a placeholder - actual implementation would load from file
    std::unordered_map<std::string, torch::Tensor> state_dict;
    
    if (!model_path.empty()) {
        // TODO: Implement actual model loading from file
        std::cout << "Loading model from: " << model_path << std::endl;
        // state_dict = load_state_dict(model_path);
    } else {
        // Use default pretrained model path
        std::cout << "Using default DeiT model configuration" << std::endl;
    }
    
    // Create and return the model instance
    return std::make_shared<TtDeiTForImageClassificationWithTeacher>(
        config, state_dict, "", device
    );
}