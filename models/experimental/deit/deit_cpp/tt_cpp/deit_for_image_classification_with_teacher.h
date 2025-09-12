// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_WITH_TEACHER_H
#define DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_WITH_TEACHER_H

#include "deit_config.h"
#include "helper_funcs.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <torch/torch.h>
#include <optional>
#include <tuple>

/**
 * TtDeiTForImageClassificationWithTeacher - C++ implementation of DeiT with knowledge distillation
 * Equivalent to the Python TtDeiTForImageClassificationWithTeacher class
 * 
 * This class wraps a DeiT model with dual classification heads for knowledge distillation:
 * - A DeiT backbone model (embeddings + encoder + layernorm + pooler)
 * - A classification head (cls_classifier)
 * - A distillation head (distillation_classifier)
 * 
 * During inference, it returns the average of both classifier predictions.
 */
class TtDeiTForImageClassificationWithTeacher {
public:
    /**
     * Constructor
     * 
     * @param config DeiT configuration
     * @param state_dict PyTorch state dictionary containing model weights
     * @param base_address Base address for weight loading
     * @param device TTNN device for tensor operations
     */
    TtDeiTForImageClassificationWithTeacher(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address,
        std::shared_ptr<ttnn::MeshDevice> device
    );

    /**
     * Forward pass for image classification with teacher
     * 
     * @param pixel_values Input image tensor [batch_size, channels, height, width]
     * @param head_mask Optional attention head mask
     * @param output_attentions Whether to output attention weights
     * @param output_hidden_states Whether to output hidden states
     * @param return_dict Whether to return structured output
     * @return Tuple containing (averaged_logits, cls_logits, distillation_logits, optional_attentions, optional_hidden_states)
     */
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> forward(
        const ttnn::Tensor& pixel_values,
        const ttnn::Tensor* head_mask = nullptr,
        bool output_attentions = false,
        bool output_hidden_states = false,
        bool return_dict = true
    );

    /**
     * Get the number of classification labels
     */
    int get_num_labels() const { return num_labels_; }

    /**
     * Get the hidden size
     */
    int get_hidden_size() const { return config_.hidden_size; }

    /**
     * Get separate classifier outputs (without averaging)
     * Useful for training with knowledge distillation loss
     */
    std::pair<ttnn::Tensor, ttnn::Tensor> get_separate_logits(
        const ttnn::Tensor& pixel_values,
        const ttnn::Tensor* head_mask = nullptr
    );

private:
    DeiTConfig config_;
    std::shared_ptr<ttnn::MeshDevice> device_;
    int num_labels_;
    
    // DeiT backbone components (forward declaration)
    class TtDeiTModel* deit_model_;
    
    // Dual classifier head weights
    ttnn::Tensor cls_classifier_weight_;
    ttnn::Tensor cls_classifier_bias_;
    ttnn::Tensor distillation_classifier_weight_;
    ttnn::Tensor distillation_classifier_bias_;
    
    /**
     * Apply classification layer
     * 
     * @param sequence_output Output from DeiT model [batch_size, seq_len, hidden_size]
     * @param weight Classifier weight tensor
     * @param bias Classifier bias tensor
     * @return Classification logits [batch_size, num_labels]
     */
    ttnn::Tensor apply_classifier(
        const ttnn::Tensor& sequence_output,
        const ttnn::Tensor& weight,
        const ttnn::Tensor& bias
    );
    
    /**
     * Extract CLS token (first token) from sequence output
     * 
     * @param sequence_output Full sequence output [batch_size, seq_len, hidden_size]
     * @return CLS token representation [batch_size, hidden_size]
     */
    ttnn::Tensor extract_cls_token(const ttnn::Tensor& sequence_output);
    
    /**
     * Extract distillation token (second token) from sequence output
     * 
     * @param sequence_output Full sequence output [batch_size, seq_len, hidden_size]
     * @return Distillation token representation [batch_size, hidden_size]
     */
    ttnn::Tensor extract_distillation_token(const ttnn::Tensor& sequence_output);
    
    /**
     * Average two logit tensors
     * 
     * @param logits1 First logit tensor
     * @param logits2 Second logit tensor
     * @return Averaged logits
     */
    ttnn::Tensor average_logits(const ttnn::Tensor& logits1, const ttnn::Tensor& logits2);
};

/**
 * Factory function to create TtDeiTForImageClassificationWithTeacher from pretrained model
 * 
 * @param device TTNN device
 * @param model_path Path to pretrained model (optional, uses default if empty)
 * @return Shared pointer to TtDeiTForImageClassificationWithTeacher instance
 */
std::shared_ptr<TtDeiTForImageClassificationWithTeacher> create_deit_for_image_classification_with_teacher(
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::string& model_path = ""
);

#endif // DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_WITH_TEACHER_H