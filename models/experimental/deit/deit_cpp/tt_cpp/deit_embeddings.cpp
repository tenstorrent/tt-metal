// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include "deit_embeddings.h"
#include <stdexcept>
#include <iostream>

TtDeiTEmbeddings::TtDeiTEmbeddings(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device,
    bool use_mask_token
) : device_(device), use_mask_token_(use_mask_token), hidden_size_(config.hidden_size) {
    
    // Initialize patch embeddings
    patch_embeddings_ = std::make_unique<TtDeiTPatchEmbeddings>(
        config,
        state_dict,
        base_address + ".patch_embeddings",
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
    
    // Convert torch tensors to ttnn tensors
    cls_token_ = helper_funcs::torch_to_tt_tensor_tile(state_dict[cls_token_key], device_);
    distillation_token_ = helper_funcs::torch_to_tt_tensor_tile(state_dict[distillation_token_key], device_);
    position_embeddings_ = helper_funcs::torch_to_tt_tensor_tile(state_dict[position_embeddings_key], device_);
    
    // Load mask token if needed
    if (use_mask_token_) {
        std::string mask_token_key = base_address + ".mask_token";
        if (state_dict.find(mask_token_key) == state_dict.end()) {
            throw std::runtime_error("Missing mask_token in state_dict: " + mask_token_key);
        }
        mask_token_ = helper_funcs::torch_to_tt_tensor_tile(state_dict[mask_token_key], device_);
    } else {
        // Create a dummy mask token (won't be used)
        // Use a simple tensor creation approach
        mask_token_ = cls_token_;  // Reuse cls_token as dummy
    }
    
    std::cout << "TtDeiTEmbeddings initialized with:" << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
    std::cout << "  Use mask token: " << (use_mask_token_ ? "true" : "false") << std::endl;
    std::cout << "  Num patches: " << get_num_patches() << std::endl;
}

ttnn::Tensor TtDeiTEmbeddings::forward(
    const ttnn::Tensor& pixel_values,
    const ttnn::Tensor* bool_masked_pos
) {
    // Get patch embeddings
    auto embeddings = patch_embeddings_->forward(pixel_values);
    
    // Get batch size and sequence length
    auto shape = embeddings.get_logical_shape();
    auto batch_size = shape[0];
    auto seq_length = shape[1];
    
    // Apply mask if provided
    if (bool_masked_pos != nullptr && use_mask_token_) {
        embeddings = apply_mask(embeddings, *bool_masked_pos);
    }
    
    // Expand cls and distillation tokens for the batch
    auto cls_tokens = expand_token(cls_token_, batch_size);
    auto distillation_tokens = expand_token(distillation_token_, batch_size);
    
    // Concatenate tokens: [cls_token, distillation_token, patch_embeddings]
    std::vector<ttnn::Tensor> tokens_to_concat = {cls_tokens, distillation_tokens, embeddings};
    auto concatenated = ttnn::concat(tokens_to_concat, 1);  // Concatenate along sequence dimension
    
    // Add position embeddings
    auto final_embeddings = ttnn::add(concatenated, position_embeddings_);
    
    return final_embeddings;
}

ttnn::Tensor TtDeiTEmbeddings::expand_token(const ttnn::Tensor& token, uint32_t batch_size, int seq_length) const {
    // Token shape is typically [1, 1, hidden_size]
    // We need to expand it to [batch_size, seq_length, hidden_size]
    
    // For now, return the token as-is and handle expansion in the calling code
    // TODO: Implement proper token expansion when TTNN API is stable
    return token;
}

ttnn::Tensor TtDeiTEmbeddings::apply_mask(const ttnn::Tensor& embeddings, const ttnn::Tensor& bool_masked_pos) const {
    // Get dimensions
    auto shape = embeddings.get_logical_shape();
    auto batch_size = shape[0];
    auto seq_length = shape[1];
    
    // Expand mask token to match embeddings shape
    auto mask_tokens = expand_token(mask_token_, batch_size, seq_length);
    
    // Convert boolean mask to float mask
    // bool_masked_pos should have shape [batch_size, seq_length]
    // We need to expand it to [batch_size, seq_length, 1] for broadcasting
    auto mask_shape = bool_masked_pos.get_logical_shape();
    ttnn::Tensor float_mask;
    
    if (mask_shape.rank() == 2) {
        // For now, use the mask as-is and handle reshaping in a simpler way
        // TODO: Implement proper mask reshaping when TTNN API is stable
        float_mask = bool_masked_pos;
    } else {
        throw std::invalid_argument("bool_masked_pos must have 2 dimensions");
    }
    
    // Apply mask: embeddings * (1 - mask) + mask_tokens * mask
    auto ones_tensor = ttnn::ones_like(float_mask);
    auto inv_mask = ttnn::subtract(ones_tensor, float_mask);
    
    auto masked_embeddings = ttnn::multiply(embeddings, inv_mask);
    auto masked_tokens = ttnn::multiply(mask_tokens, float_mask);
    
    return ttnn::add(masked_embeddings, masked_tokens);
}