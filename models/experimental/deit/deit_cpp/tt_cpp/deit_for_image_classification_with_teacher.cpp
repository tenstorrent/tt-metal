// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_for_image_classification_with_teacher.h"
#include "deit_model.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <stdexcept>
#include <iostream>

TtDeiTForImageClassificationWithTeacher::TtDeiTForImageClassificationWithTeacher(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device
) : config_(config), device_(device), num_labels_(1000) { // Default to 1000 classes like ImageNet

    try {
        // Load model parameters to state_dict
        std::vector<std::string> required_params = {
            "cls_classifier.weight", "cls_classifier.bias",
            "distillation_classifier.weight", "distillation_classifier.bias"
        };

        // Split state_dict into two parts based on required_params
        std::unordered_map<std::string, torch::Tensor> required_state_dict;

        for (const auto& param : required_params) {
            std::string full_key = base_address + param;
            auto it = state_dict.find(full_key);
            if (it != state_dict.end()) {
                required_state_dict[full_key] = it->second;
                state_dict.erase(it);  // Remove from original map
            }
        }

        // Initialize DeiT backbone model
        deit_model_ = std::make_unique<TtDeiTModel>(
            config,
            state_dict,
            base_address + "deit.",
            device,
            false,  // add_pooling_layer
            false   // use_mask_token
        );

        // Load CLS classifier weights
        std::string cls_weight_key = base_address + "cls_classifier.weight";
        std::string cls_bias_key = base_address + "cls_classifier.bias";

        if (required_state_dict.find(cls_weight_key) != required_state_dict.end()) {
            cls_classifier_weight_ = helper_funcs::torch_to_tt_tensor_tile(
                required_state_dict[cls_weight_key], device
            );
        } else {
            throw std::runtime_error("CLS classifier weight not found in state_dict: " + cls_weight_key);
        }

        if (required_state_dict.find(cls_bias_key) != required_state_dict.end()) {
            cls_classifier_bias_ = helper_funcs::torch_to_tt_tensor_tile(
                required_state_dict[cls_bias_key], device
            );

            // Update num_labels based on classifier bias size
            auto bias_shape = cls_classifier_bias_.logical_shape();
            if (bias_shape.rank() >= 1) {
                num_labels_ = static_cast<int>(bias_shape[-1]);
            }
        } else {
            throw std::runtime_error("CLS classifier bias not found in state_dict: " + cls_bias_key);
        }

        // Load distillation classifier weights
        std::string distill_weight_key = base_address + "distillation_classifier.weight";
        std::string distill_bias_key = base_address + "distillation_classifier.bias";

        if (required_state_dict.find(distill_weight_key) != required_state_dict.end()) {
            distillation_classifier_weight_ = helper_funcs::torch_to_tt_tensor_tile(
                required_state_dict[distill_weight_key], device
            );
        } else {
            throw std::runtime_error("Distillation classifier weight not found in state_dict: " + distill_weight_key);
        }

        if (required_state_dict.find(distill_bias_key) != required_state_dict.end()) {
            distillation_classifier_bias_ = helper_funcs::torch_to_tt_tensor_tile(
                required_state_dict[distill_bias_key], device
            );
        } else {
            throw std::runtime_error("Distillation classifier bias not found in state_dict: " + distill_bias_key);
        }

        std::cout << "TtDeiTForImageClassificationWithTeacher initialized with " << num_labels_ << " classes" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing TtDeiTForImageClassificationWithTeacher: " << e.what() << std::endl;
        throw;
    }
}

TtDeiTForImageClassificationWithTeacher::~TtDeiTForImageClassificationWithTeacher() = default;

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>>
TtDeiTForImageClassificationWithTeacher::forward(
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

        // Average the logits for final prediction
        ttnn::Tensor averaged_logits = average_logits(cls_logits, distillation_logits);

        // Return outputs and optional additional outputs
        std::optional<ttnn::Tensor> hidden_states_output = std::nullopt;
        std::optional<ttnn::Tensor> attentions_output = std::nullopt;

        if (output_hidden_states && hidden_states.has_value()) {
            const auto& hidden_states_vec = hidden_states.value();
            if (!hidden_states_vec.empty()) {
                hidden_states_output = hidden_states_vec.back();
            }
        }

        if (output_attentions && attentions.has_value()) {
            const auto& attentions_vec = attentions.value();
            if (!attentions_vec.empty()) {
                attentions_output = attentions_vec.back();
            }
        }

        return std::make_tuple(averaged_logits, cls_logits, distillation_logits, attentions_output, hidden_states_output);

    } catch (const std::exception& e) {
        std::cerr << "Error in TtDeiTForImageClassificationWithTeacher forward pass: " << e.what() << std::endl;
        throw;
    }
}

std::pair<ttnn::Tensor, ttnn::Tensor> TtDeiTForImageClassificationWithTeacher::get_separate_logits(
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor* head_mask
) {
    // Forward pass through DeiT backbone (no optional outputs needed)
    auto deit_outputs = deit_model_->forward(pixel_values, std::nullopt, head_mask, false, false, true);

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

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::extract_cls_token(const ttnn::Tensor& sequence_output) {
    // Extract the first token (CLS token) from sequence output
    auto shape = sequence_output.logical_shape();
    uint32_t rank = shape.rank();

    if (rank == 3) {
        // sequence_output shape: [batch_size, seq_len, hidden_size]
        uint32_t batch_size = shape[0];
        uint32_t hidden_size = shape[2];

        ttnn::SmallVector<uint32_t> slice_start = {0, 0, 0};
        ttnn::SmallVector<uint32_t> slice_end = {batch_size, 1, hidden_size};
        ttnn::SmallVector<uint32_t> slice_step = {1, 1, 1};

        auto cls_token = ttnn::slice(sequence_output, slice_start, slice_end, slice_step);

        // Reshape to remove the sequence dimension: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        ttnn::SmallVector<uint32_t> new_shape = {batch_size, hidden_size};
        return ttnn::reshape(cls_token, ttnn::Shape(new_shape));
    } else if (rank == 4) {
        // sequence_output shape: [batch_size, 1, seq_len, hidden_size]
        uint32_t batch_size = shape[0];
        uint32_t dim1 = shape[1];
        uint32_t hidden_size = shape[3];

        std::array<uint32_t, 4> slice_start = {0, 0, 0, 0};
        std::array<uint32_t, 4> slice_end = {batch_size, dim1, 1, hidden_size};
        std::array<uint32_t, 4> slice_step = {1, 1, 1, 1};

        auto cls_token = ttnn::slice(sequence_output, slice_start, slice_end, slice_step);

        // Reshape to remove the sequence dimension: [batch_size, 1, 1, hidden_size] -> [batch_size, hidden_size]
        ttnn::SmallVector<uint32_t> new_shape = {batch_size, hidden_size};
        return ttnn::reshape(cls_token, ttnn::Shape(new_shape));
    } else {
        throw std::runtime_error("Unsupported sequence_output rank: " + std::to_string(rank));
    }
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::extract_distillation_token(const ttnn::Tensor& sequence_output) {
    // Extract the second token (distillation token) from sequence output
    auto shape = sequence_output.logical_shape();
    uint32_t rank = shape.rank();

    if (rank == 3) {
        // sequence_output shape: [batch_size, seq_len, hidden_size]
        uint32_t batch_size = shape[0];
        uint32_t hidden_size = shape[2];

        ttnn::SmallVector<uint32_t> slice_start = {0, 1, 0};
        ttnn::SmallVector<uint32_t> slice_end = {batch_size, 2, hidden_size};
        ttnn::SmallVector<uint32_t> slice_step = {1, 1, 1};

        auto distillation_token = ttnn::slice(sequence_output, slice_start, slice_end, slice_step);

        // Reshape to remove the sequence dimension: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        ttnn::SmallVector<uint32_t> new_shape = {batch_size, hidden_size};
        return ttnn::reshape(distillation_token, ttnn::Shape(new_shape));
    } else if (rank == 4) {
        // sequence_output shape: [batch_size, 1, seq_len, hidden_size]
        uint32_t batch_size = shape[0];
        uint32_t dim1 = shape[1];
        uint32_t hidden_size = shape[3];

        std::array<uint32_t, 4> slice_start = {0, 0, 1, 0};
        std::array<uint32_t, 4> slice_end = {batch_size, dim1, 2, hidden_size};
        std::array<uint32_t, 4> slice_step = {1, 1, 1, 1};

        auto distillation_token = ttnn::slice(sequence_output, slice_start, slice_end, slice_step);

        // Reshape to remove the sequence dimension: [batch_size, 1, 1, hidden_size] -> [batch_size, hidden_size]
        ttnn::SmallVector<uint32_t> new_shape = {batch_size, hidden_size};
        return ttnn::reshape(distillation_token, ttnn::Shape(new_shape));
    } else {
        throw std::runtime_error("Unsupported sequence_output rank: " + std::to_string(rank));
    }
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::apply_classifier(
    const ttnn::Tensor& token_output,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias
) {
    // Apply linear transformation: output = input @ weight.T + bias
    ttnn::Tensor logits = helper_funcs::linear_transform(token_output, weight, bias);
    return logits;
}

ttnn::Tensor TtDeiTForImageClassificationWithTeacher::average_logits(
    const ttnn::Tensor& logits1,
    const ttnn::Tensor& logits2
) {
    // Add the two logits and multiply by 0.5 to get average
    ttnn::Tensor sum_logits = ttnn::add(logits1, logits2);
    // Use scalar multiplication instead of creating a constant tensor
    return ttnn::multiply(sum_logits, 0.5f);
}

std::shared_ptr<TtDeiTForImageClassificationWithTeacher> create_deit_for_image_classification_with_teacher(
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::string& model_path
) {
    // This is a placeholder implementation
    // In practice, you would load the model from the specified path
    // For now, we'll create a default configuration
    DeiTConfig config;
    std::unordered_map<std::string, torch::Tensor> state_dict;

    // Load state dict from model_path if provided
    // This would require implementing model loading logic

    return std::make_shared<TtDeiTForImageClassificationWithTeacher>(
        config, state_dict, "", device
    );
}
