// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_MODEL_H
#define DEIT_CPP_TT_CPP_DEIT_MODEL_H

#include "deit_config.h"
#include "deit_embeddings.h"
#include "deit_encoder.h"
#include "deit_pooler.h"
#include "../helper_funcs.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <torch/torch.h>
#include <optional>
#include <tuple>
#include <vector>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>
#include <ttnn/device.hpp>

/**
 * TtDeiTModel class - C++ implementation of DeiT model
 * Equivalent to the Python TtDeiTModel class
 * 
 * This class implements the core DeiT transformer model with:
 * - Patch embeddings for converting images to token sequences
 * - Position embeddings and optional mask token
 * - Multi-layer transformer encoder
 * - Layer normalization
 * - Optional pooler for classification tasks
 */
class TtDeiTModel {
public:
    /**
     * Constructor
     * 
     * @param config DeiT configuration
     * @param state_dict PyTorch state dictionary containing model weights
     * @param base_address Base address for weight loading
     * @param device TTNN device for tensor operations
     * @param add_pooling_layer Whether to add pooling layer
     * @param use_mask_token Whether to use mask token in embeddings
     */
    TtDeiTModel(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address,
        std::shared_ptr<ttnn::MeshDevice> device,
        bool add_pooling_layer = true,
        bool use_mask_token = false
    );

    /**
     * Forward pass through the DeiT model
     * 
     * @param pixel_values Input image tensor [batch_size, channels, height, width]
     * @param bool_masked_pos Optional boolean mask for masked positions
     * @param head_mask Optional attention head mask
     * @param output_attentions Whether to output attention weights
     * @param output_hidden_states Whether to output hidden states from all layers
     * @param return_dict Whether to return structured output
     * @return Tuple containing sequence output, pooled output (optional), and additional outputs
     */
    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<std::vector<ttnn::Tensor>>, std::optional<std::vector<ttnn::Tensor>>> forward(
        const ttnn::Tensor& pixel_values,
        const std::optional<torch::Tensor>& bool_masked_pos = std::nullopt,
        const ttnn::Tensor* head_mask = nullptr,
        bool output_attentions = false,
        bool output_hidden_states = false,
        bool return_dict = true
    );

    /**
     * Get head mask for attention layers
     * 
     * @param head_mask Input head mask (can be nullptr)
     * @param num_hidden_layers Number of hidden layers
     * @return Processed head mask
     */
    std::vector<std::optional<ttnn::Tensor>> get_head_mask(
        const ttnn::Tensor* head_mask,
        int num_hidden_layers
    );

    /**
     * Destructor
     */
    ~TtDeiTModel();

private:
    // Configuration and device
    DeiTConfig config_;
    std::shared_ptr<ttnn::MeshDevice> device_;
    bool add_pooling_layer_;
    
    // Model components
    std::unique_ptr<TtDeiTEmbeddings> embeddings_;
    std::unique_ptr<TtDeiTEncoder> encoder_;
    std::unique_ptr<TtDeiTPooler> pooler_;
    
    // Layer normalization parameters
    ttnn::Tensor layernorm_weight_;
    ttnn::Tensor layernorm_bias_;
};

/**
 * Factory function to create TtDeiTModel from pretrained model
 * 
 * @param device TTNN device
 * @param model_path Path to pretrained model (optional, uses default if empty)
 * @param add_pooling_layer Whether to add pooling layer
 * @param use_mask_token Whether to use mask token
 * @return Shared pointer to TtDeiTModel instance
 */
std::shared_ptr<TtDeiTModel> create_deit_model(
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::string& model_path = "",
    bool add_pooling_layer = true,
    bool use_mask_token = false,
    const std::unordered_map<std::string, torch::Tensor>& state_dict = {}
);

#endif // DEIT_CPP_TT_CPP_DEIT_MODEL_H