// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <torch/torch.h>
#include "deit_config.h"
#include "deit_patch_embeddings.h"

class TtDeiTEmbeddings {
public:
    /**
     * Constructor for TtDeiTEmbeddings
     * @param config DeiT configuration
     * @param state_dict Model state dictionary containing weights and biases
     * @param base_address Base address for parameter lookup in state_dict
     * @param use_mask_token Whether to use mask token for masked image modeling
     */
    TtDeiTEmbeddings(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address,
        bool use_mask_token = false
    );

    /**
     * Forward pass for DeiT embeddings
     * Combines patch embeddings with cls and distillation tokens, applies position embeddings
     * @param pixel_values Input tensor with shape [batch_size, num_channels, height, width]
     * @param bool_masked_pos Optional boolean mask for masked image modeling
     * @return Embeddings tensor with shape [batch_size, seq_length, hidden_size]
     *         where seq_length = num_patches + 2 (cls + distillation tokens)
     */
    torch::Tensor forward(
        const torch::Tensor& pixel_values,
        const torch::Tensor* bool_masked_pos = nullptr
    );

    // Getters
    int get_num_patches() const { return patch_embeddings_->get_num_patches(); }
    int get_hidden_size() const { return hidden_size_; }
    bool has_mask_token() const { return use_mask_token_; }

private:
    // Configuration
    int hidden_size_;
    bool use_mask_token_;

    // Components
    std::unique_ptr<TtDeiTPatchEmbeddings> patch_embeddings_;

    // Torch tensors for tokens and embeddings
    torch::Tensor cls_token_;
    torch::Tensor distillation_token_;
    torch::Tensor mask_token_;  // Only used if use_mask_token_ is true
    torch::Tensor position_embeddings_;

    // Helper functions
    torch::Tensor expand_token(const torch::Tensor& token, int64_t batch_size, int64_t seq_length = 1) const;
    torch::Tensor apply_mask(const torch::Tensor& embeddings, const torch::Tensor& bool_masked_pos) const;
};
