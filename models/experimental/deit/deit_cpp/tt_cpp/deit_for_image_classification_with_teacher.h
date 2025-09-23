// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_WITH_TEACHER_H
#define DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_WITH_TEACHER_H

#include "deit_config.h"
#include "../helper_funcs.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <torch/torch.h>
#include <optional>

// Forward declaration
class TtDeiTModel;

/**
 * TtDeiTForImageClassificationWithTeacher - C++ implementation of DeiT for image classification with teacher
 * Equivalent to the Python TtDeiTForImageClassificationWithTeacher class
 *
 * This class wraps a DeiT model with dual classification heads for distillation learning.
 * It includes:
 * - A DeiT backbone model (embeddings + encoder + layernorm)
 * - A CLS classifier head for standard classification
 * - A distillation classifier head for teacher knowledge distillation
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
     * Destructor
     */
    ~TtDeiTForImageClassificationWithTeacher();

    /**
     * Forward pass for image classification with teacher
     * Returns averaged logits, cls logits, and distillation logits
     *
     * @param pixel_values Input image tensor [batch_size, channels, height, width]
     * @param head_mask Optional attention head mask
     * @param output_attentions Whether to output attention weights
     * @param output_hidden_states Whether to output hidden states
     * @param return_dict Whether to return structured output
     * @return Tuple of (averaged_logits, cls_logits, distillation_logits, attentions, hidden_states)
     */
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>>
    forward(
        const ttnn::Tensor& pixel_values,
        const ttnn::Tensor* head_mask = nullptr,
        bool output_attentions = false,
        bool output_hidden_states = false,
        bool return_dict = true
    );

    /**
     * Get separate logits from both classifiers without averaging
     * This is useful for testing and debugging the individual classifier outputs
     *
     * @param pixel_values Input image tensor
     * @param head_mask Optional attention head mask
     * @return Pair of (cls_logits, distillation_logits)
     */
    std::pair<ttnn::Tensor, ttnn::Tensor> get_separate_logits(
        const ttnn::Tensor& pixel_values,
        const ttnn::Tensor* head_mask = nullptr
    );

    /**
     * Get the number of classification labels
     */
    int get_num_labels() const { return num_labels_; }

    /**
     * Get the hidden size
     */
    int get_hidden_size() const { return config_.hidden_size; }

private:
    DeiTConfig config_;
    std::shared_ptr<ttnn::MeshDevice> device_;
    int num_labels_;

    // DeiT backbone model
    std::unique_ptr<TtDeiTModel> deit_model_;

    // Classifier head weights
    ttnn::Tensor cls_classifier_weight_;
    ttnn::Tensor cls_classifier_bias_;
    ttnn::Tensor distillation_classifier_weight_;
    ttnn::Tensor distillation_classifier_bias_;

    /**
     * Extract CLS token from sequence output
     *
     * @param sequence_output Output from DeiT model [batch_size, seq_len, hidden_size]
     * @return CLS token [batch_size, hidden_size]
     */
    ttnn::Tensor extract_cls_token(const ttnn::Tensor& sequence_output);

    /**
     * Extract distillation token from sequence output
     *
     * @param sequence_output Output from DeiT model [batch_size, seq_len, hidden_size]
     * @return Distillation token [batch_size, hidden_size]
     */
    ttnn::Tensor extract_distillation_token(const ttnn::Tensor& sequence_output);

    /**
     * Apply linear classification layer
     *
     * @param token_output Token representation [batch_size, hidden_size]
     * @param weight Classifier weight [num_labels, hidden_size]
     * @param bias Classifier bias [num_labels]
     * @return Classification logits [batch_size, num_labels]
     */
    ttnn::Tensor apply_classifier(
        const ttnn::Tensor& token_output,
        const ttnn::Tensor& weight,
        const ttnn::Tensor& bias
    );

    /**
     * Average two logits tensors
     *
     * @param logits1 First logits tensor
     * @param logits2 Second logits tensor
     * @return Averaged logits
     */
    ttnn::Tensor average_logits(
        const ttnn::Tensor& logits1,
        const ttnn::Tensor& logits2
    );
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
