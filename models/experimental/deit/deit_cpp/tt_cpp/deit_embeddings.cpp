// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include "deit_embeddings.h"
#include <stdexcept>
#include <iostream>

TtDeiTEmbeddings::TtDeiTEmbeddings(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    bool use_mask_token
) : use_mask_token_(use_mask_token), hidden_size_(config.hidden_size) {

    // Initialize patch embeddings
    patch_embeddings_ = std::make_unique<TtDeiTPatchEmbeddings>(
        config,
        state_dict,
        base_address + ".patch_embeddings."
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

    // Store torch tensors directly
    cls_token_ = state_dict[cls_token_key].clone();
    distillation_token_ = state_dict[distillation_token_key].clone();
    position_embeddings_ = state_dict[position_embeddings_key].clone();

    // Load mask token if needed
    if (use_mask_token_) {
        std::string mask_token_key = base_address + "mask_token";
        if (state_dict.find(mask_token_key) == state_dict.end()) {
            throw std::runtime_error("Missing mask_token in state_dict: " + mask_token_key);
        }
        mask_token_ = state_dict[mask_token_key].clone();
    } else {
        // Create a dummy mask token (won't be used)
        mask_token_ = cls_token_.clone();
    }

    std::cout << "TtDeiTEmbeddings initialized with:" << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
    std::cout << "  Use mask token: " << (use_mask_token_ ? "true" : "false") << std::endl;
    std::cout << "  Num patches: " << get_num_patches() << std::endl;
}

torch::Tensor TtDeiTEmbeddings::forward(
    const torch::Tensor& pixel_values,
    const torch::Tensor* bool_masked_pos
) {
    // Get patch embeddings
    auto embeddings = patch_embeddings_->forward(pixel_values);

    // Get batch size
    auto batch_size = embeddings.size(0);

    // Apply mask if provided
    if (bool_masked_pos != nullptr && use_mask_token_) {
        embeddings = apply_mask(embeddings, *bool_masked_pos);
    }

    // Expand cls and distillation tokens for the batch
    auto cls_tokens = expand_token(cls_token_, batch_size);
    auto distillation_tokens = expand_token(distillation_token_, batch_size);

    // Concatenate tokens: [cls_token, distillation_token, patch_embeddings]
    auto concatenated = torch::cat({cls_tokens, distillation_tokens, embeddings}, 1);  // Concatenate along sequence dimension

    // Add position embeddings
    auto final_embeddings = concatenated + position_embeddings_;

    return final_embeddings;
}

torch::Tensor TtDeiTEmbeddings::expand_token(const torch::Tensor& token, int64_t batch_size, int64_t seq_length) const {
    // Token shape is typically [1, 1, hidden_size]
    // We need to expand it to [batch_size, seq_length, hidden_size]
    return token.expand({batch_size, seq_length, -1});
}

torch::Tensor TtDeiTEmbeddings::apply_mask(const torch::Tensor& embeddings, const torch::Tensor& bool_masked_pos) const {
    // Get dimensions
    auto batch_size = embeddings.size(0);
    auto seq_length = embeddings.size(1);

    // Expand mask token to match embeddings shape
    auto mask_tokens = expand_token(mask_token_, batch_size, seq_length);

    // Convert boolean mask to float mask
    // bool_masked_pos should have shape [batch_size, seq_length]
    // We need to expand it to [batch_size, seq_length, 1] for broadcasting
    torch::Tensor float_mask;

    if (bool_masked_pos.dim() == 2) {
        // Expand mask to [batch_size, seq_length, 1] for broadcasting
        float_mask = bool_masked_pos.unsqueeze(-1).to(torch::kFloat32);
    } else {
        throw std::invalid_argument("bool_masked_pos must have 2 dimensions");
    }

    // Apply mask: embeddings * (1 - mask) + mask_tokens * mask
    auto inv_mask = 1.0f - float_mask;

    auto masked_embeddings = embeddings * inv_mask;
    auto masked_tokens = mask_tokens * float_mask;

    return masked_embeddings + masked_tokens;
}
