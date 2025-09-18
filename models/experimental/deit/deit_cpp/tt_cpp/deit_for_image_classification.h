// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_H
#define DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_H

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
 * TtDeiTForImageClassification - C++ implementation of DeiT for image classification
 * Equivalent to the Python TtDeiTForImageClassification class
 * 
 * This class wraps a DeiT model with a classification head for image classification tasks.
 * It includes:
 * - A DeiT backbone model (embeddings + encoder + layernorm + pooler)
 * - A single linear classifier head
 */
class TtDeiTForImageClassification {
public:
    /**
     * Constructor
     * 
     * @param config DeiT configuration
     * @param state_dict PyTorch state dictionary containing model weights
     * @param base_address Base address for weight loading
     * @param device TTNN device for tensor operations
     */
    TtDeiTForImageClassification(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address,
        std::shared_ptr<ttnn::MeshDevice> device
    );

    /**
     * Destructor
     */
    ~TtDeiTForImageClassification();

    /**
     * Forward pass for image classification
     * 
     * @param pixel_values Input image tensor [batch_size, channels, height, width]
     * @param head_mask Optional attention head mask
     * @param output_attentions Whether to output attention weights
     * @param output_hidden_states Whether to output hidden states
     * @param return_dict Whether to return structured output
     * @return Classification logits and optional additional outputs
     */
    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> forward(
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

private:
    DeiTConfig config_;
    std::shared_ptr<ttnn::MeshDevice> device_;
    int num_labels_;
    
    // DeiT backbone components (we'll need to include the actual DeiT model class)
    // For now, we'll use forward declarations and implement in the cpp file
    std::unique_ptr<TtDeiTModel> deit_model_;
    
    // Classifier head weights
    ttnn::Tensor classifier_weight_;
    ttnn::Tensor classifier_bias_;
    
    /**
     * Apply linear classification layer
     * 
     * @param sequence_output Output from DeiT model [batch_size, seq_len, hidden_size]
     * @return Classification logits [batch_size, num_labels]
     */
    ttnn::Tensor apply_classifier(const ttnn::Tensor& sequence_output);
};

/**
 * Factory function to create TtDeiTForImageClassification from pretrained model
 * 
 * @param device TTNN device
 * @param model_path Path to pretrained model (optional, uses default if empty)
 * @return Shared pointer to TtDeiTForImageClassification instance
 */
std::shared_ptr<TtDeiTForImageClassification> create_deit_for_image_classification(
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::string& model_path = ""
);

#endif // DEIT_CPP_TT_CPP_DEIT_FOR_IMAGE_CLASSIFICATION_H