// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_embeddings.h"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/creation.hpp"
#include "helper_funcs.h"
#include <stdexcept>
#include <iostream>

TtDeiTEmbeddings::TtDeiTEmbeddings(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device,
    bool use_mask_token) :
    hidden_size_(config.hidden_size), use_mask_token_(use_mask_token), device_(device) {
    // Initialize patch embeddings
    patch_embeddings_ = std::make_unique<TtDeiTPatchEmbeddings>(
        config,
        state_dict,
        base_address + ".patch_embeddings.",
        device
    );

    // Load tokens and position embeddings from state_dict
    std::string cls_token_key = base_address + ".cls_token";
    std::string distillation_token_key = base_address + ".distillation_token";
    std::string position_embeddings_key = base_address + ".position_embeddings";

    if (state_dict.find(cls_token_key) == state_dict.end()) {
        throw std::runtime_error("Missing cls_token in state_dict: " + cls_token_key);
    }
    if (state_dict.find(distillation_token_key) == state_dict.end()) {
        throw std::runtime_error("Missing distillation_token in state_dict: " + distillation_token_key);
    }
    if (state_dict.find(position_embeddings_key) == state_dict.end()) {
        throw std::runtime_error("Missing position_embeddings in state_dict: " + position_embeddings_key);
    }

    // Convert to ttnn tensors with ROW_MAJOR layout for safe concatenation
    // cls_token: [1, 1, 1, hidden_size]
    auto cls_tensor = state_dict[cls_token_key].clone();
    cls_tensor = cls_tensor.reshape({1, 1, 1, hidden_size_});
    cls_token_ = helper_funcs::from_torch(cls_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
    cls_token_ = ttnn::to_device(cls_token_, device_.get(), ttnn::DRAM_MEMORY_CONFIG);

    // distillation_token: [1, 1, 1, hidden_size]
    auto dist_tensor = state_dict[distillation_token_key].clone();
    dist_tensor = dist_tensor.reshape({1, 1, 1, hidden_size_});
    distillation_token_ = helper_funcs::from_torch(dist_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
    distillation_token_ = ttnn::to_device(distillation_token_, device_.get(), ttnn::DRAM_MEMORY_CONFIG);

    // position_embeddings: [1, 1, seq_len, hidden_size]
    auto pos_tensor = state_dict[position_embeddings_key].clone();
    int seq_len = pos_tensor.size(1);
    pos_tensor = pos_tensor.reshape({1, 1, seq_len, hidden_size_});
    position_embeddings_ = helper_funcs::from_torch(pos_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
    position_embeddings_ = ttnn::to_device(position_embeddings_, device_.get(), ttnn::DRAM_MEMORY_CONFIG);

    // Load mask token if needed
    if (use_mask_token_) {
        std::string mask_token_key = base_address + "mask_token";
        if (state_dict.find(mask_token_key) == state_dict.end()) {
            throw std::runtime_error("Missing mask_token in state_dict: " + mask_token_key);
        }
        auto mask_tensor = state_dict[mask_token_key].clone();
        mask_tensor = mask_tensor.reshape({1, 1, 1, hidden_size_});
        mask_token_ = helper_funcs::from_torch(mask_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
        mask_token_ = ttnn::to_device(mask_token_, device_.get(), ttnn::DRAM_MEMORY_CONFIG);
    } else {
        // Create a dummy mask token (copy of cls_token)
        mask_token_ = cls_token_;
    }
}

ttnn::Tensor TtDeiTEmbeddings::forward(
    const ttnn::Tensor& pixel_values,
    const std::optional<ttnn::Tensor>& bool_masked_pos
) {
    // Get patch embeddings: [batch_size, 1, num_patches, hidden_size]
    // Output from conv2d is usually TILE layout
    auto embeddings = patch_embeddings_->forward(pixel_values);

    // Get batch size
    auto shape = embeddings.logical_shape();
    auto batch_size = shape[0];

    // Convert to ROW_MAJOR for masking and concatenation operations
    embeddings = ttnn::to_layout(embeddings, ttnn::Layout::ROW_MAJOR);

    // Apply mask if provided
    if (bool_masked_pos.has_value() && use_mask_token_) {
        embeddings = apply_mask(embeddings, bool_masked_pos.value());
    }

    // Expand cls and distillation tokens for the batch
    auto cls_tokens = expand_token(cls_token_, batch_size);
    auto distillation_tokens = expand_token(distillation_token_, batch_size);

    // Concatenate tokens: [cls_token, distillation_token, patch_embeddings]
    // All are ROW_MAJOR now.
    std::vector<ttnn::Tensor> tensors_to_concat = {cls_tokens, distillation_tokens, embeddings};
    auto concatenated = ttnn::concat(tensors_to_concat, 2);

    // Add position embeddings
    auto final_embeddings = ttnn::add(concatenated, position_embeddings_);

    // Convert back to TILE layout for subsequent transformer layers
    final_embeddings = ttnn::to_layout(final_embeddings, ttnn::Layout::TILE);

    return final_embeddings;
}

ttnn::Tensor TtDeiTEmbeddings::expand_token(const ttnn::Tensor& token, int64_t batch_size) const {
    // Token shape is [1, 1, 1, hidden_size]
    // We need to expand it to [batch_size, 1, 1, hidden_size]
    ttnn::Shape repeat_shape({static_cast<uint32_t>(batch_size), 1, 1, 1});
    return ttnn::repeat(token, repeat_shape);
}

ttnn::Tensor TtDeiTEmbeddings::apply_mask(const ttnn::Tensor& embeddings, const ttnn::Tensor& bool_masked_pos) const {
    // Input embeddings are ROW_MAJOR
    auto shape = embeddings.logical_shape();
    auto batch_size = shape[0];
    auto num_patches = shape[2];

    // Expand mask token to match embeddings shape
    ttnn::Shape repeat_shape({static_cast<uint32_t>(batch_size), 1, static_cast<uint32_t>(num_patches), 1});
    auto mask_tokens = ttnn::repeat(mask_token_, repeat_shape);

    // Use provided mask directly (expected to be float tensor on device)
    // Shape: [batch_size, 1, num_patches, 1]
    auto float_mask = bool_masked_pos;

    // Apply mask: embeddings * (1 - mask) + mask_tokens * mask
    auto neg_mask = ttnn::neg(float_mask);
    auto inv_mask = ttnn::add(neg_mask, 1.0f);

    auto masked_embeddings = ttnn::multiply(embeddings, inv_mask);
    auto masked_tokens_part = ttnn::multiply(mask_tokens, float_mask);

    return ttnn::add(masked_embeddings, masked_tokens_part);
}
