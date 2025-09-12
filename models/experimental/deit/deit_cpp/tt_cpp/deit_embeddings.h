// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <torch/torch.h>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "deit_config.h"
#include "deit_patch_embeddings.h"
#include "helper_funcs.h"

class TtDeiTEmbeddings {
public:
    /**
     * Constructor for TtDeiTEmbeddings
     * @param config DeiT configuration
     * @param state_dict Model state dictionary containing weights and biases
     * @param base_address Base address for parameter lookup in state_dict
     * @param device TTNN device for tensor operations
     * @param use_mask_token Whether to use mask token for masked image modeling
     */
    TtDeiTEmbeddings(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address,
        std::shared_ptr<ttnn::MeshDevice> device,
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
    ttnn::Tensor forward(
        const ttnn::Tensor& pixel_values,
        const ttnn::Tensor* bool_masked_pos = nullptr
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
    
    // TTNN tensors for tokens and embeddings
    ttnn::Tensor cls_token_;
    ttnn::Tensor distillation_token_;
    ttnn::Tensor mask_token_;  // Only used if use_mask_token_ is true
    ttnn::Tensor position_embeddings_;
    
    // Device reference
    std::shared_ptr<ttnn::MeshDevice> device_;
    
    // Helper functions
    ttnn::Tensor expand_token(const ttnn::Tensor& token, uint32_t batch_size, int seq_length = 1) const;
    ttnn::Tensor apply_mask(const ttnn::Tensor& embeddings, const ttnn::Tensor& bool_masked_pos) const;
};