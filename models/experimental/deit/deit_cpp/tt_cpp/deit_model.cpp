// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_model.h"
#include "../helper_funcs.h"
#include <stdexcept>
#include <iostream>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>

TtDeiTModel::TtDeiTModel(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device,
    bool add_pooling_layer,
    bool use_mask_token
) : config_(config), device_(device), add_pooling_layer_(add_pooling_layer) {
    
    try {
        // Initialize embeddings
        std::string embeddings_address = base_address.empty() ? "embeddings" : base_address + "embeddings";
        embeddings_ = std::make_unique<TtDeiTEmbeddings>(
            config,
            state_dict,
            embeddings_address,
            use_mask_token
        );
        
        // Initialize encoder
        std::string encoder_address = base_address.empty() ? "encoder" : base_address + "encoder";
        encoder_ = std::make_unique<TtDeiTEncoder>(
            config,
            device,
            state_dict,
            encoder_address
        );
        
        // Load layer normalization parameters
        std::string ln_weight_key = base_address.empty() ? "layernorm.weight" : base_address + "layernorm.weight";
        std::string ln_bias_key = base_address.empty() ? "layernorm.bias" : base_address + "layernorm.bias";
        
        if (state_dict.find(ln_weight_key) != state_dict.end()) {
            layernorm_weight_ = helper_funcs::torch_to_tt_tensor_tile(
                state_dict[ln_weight_key], device
            );
        } else {
            throw std::runtime_error("LayerNorm weight not found: " + ln_weight_key);
        }
        
        if (state_dict.find(ln_bias_key) != state_dict.end()) {
            layernorm_bias_ = helper_funcs::torch_to_tt_tensor_tile(
                state_dict[ln_bias_key], device
            );
        } else {
            throw std::runtime_error("LayerNorm bias not found: " + ln_bias_key);
        }
        
        // Initialize pooler if needed
        if (add_pooling_layer) {
            std::string pooler_address = base_address.empty() ? "pooler" : base_address + "pooler.";
            pooler_ = std::make_unique<TtDeiTPooler>(
                config,
                device,
                state_dict,
                pooler_address
            );
        }
        
        std::cout << "TtDeiTModel initialized successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TtDeiTModel: " << e.what() << std::endl;
        throw;
    }
}

std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>, std::optional<std::vector<ttnn::Tensor>>, std::optional<std::vector<ttnn::Tensor>>> 
TtDeiTModel::forward(
    const ttnn::Tensor& pixel_values,
    const std::optional<torch::Tensor>& bool_masked_pos,
    const ttnn::Tensor* head_mask,
    bool output_attentions,
    bool output_hidden_states,
    bool return_dict
) {
    try {
        // Set default values based on config if not provided
        bool actual_output_attentions = output_attentions;
        bool actual_output_hidden_states = output_hidden_states;
        bool actual_return_dict = return_dict;
        
        // Validate input
        if (pixel_values.get_logical_shape().rank() == 0) {
            throw std::runtime_error("You have to specify pixel_values");
        }
        
        // Get head mask
        auto processed_head_mask = get_head_mask(head_mask, config_.num_hidden_layers);

        // Convert pixel_values to torch tensor for embeddings processing
        ttnn::Tensor pixel_values_host = ttnn::from_device(pixel_values);
        auto pixel_values_torch = helper_funcs::to_torch(pixel_values_host);
        
        // Check and convert dtype if needed
        // Note: This is a simplified version - in practice you'd need to check the embeddings dtype
        if (pixel_values_torch.dtype() != torch::kFloat32) {
            pixel_values_torch = pixel_values_torch.to(torch::kFloat32);
        }
        
        // Apply embeddings
        const torch::Tensor* bool_masked_pos_ptr = bool_masked_pos.has_value() ? &bool_masked_pos.value() : nullptr;
        auto embedding_output_torch = embeddings_->forward(pixel_values_torch, bool_masked_pos_ptr);
        auto embedding_output = helper_funcs::torch_to_tt_tensor_tile(embedding_output_torch, device_);
        
        // Apply encoder
        std::optional<std::vector<ttnn::Tensor>> head_mask_opt = std::nullopt;
        auto encoder_outputs = encoder_->forward(
            embedding_output,
            head_mask_opt,
            actual_output_attentions,
            actual_output_hidden_states,
            actual_return_dict
        );
        
        // Extract sequence output and apply layer normalization
        ttnn::Tensor sequence_output = std::get<0>(encoder_outputs);
        sequence_output = helper_funcs::apply_layernorm(
            sequence_output,
            layernorm_weight_,
            layernorm_bias_,
            config_.layer_norm_eps);
        
        // Ensure proper layout for subsequent operations
        // sequence_output = ttnn::to_layout(sequence_output, ttnn::TILE_LAYOUT);
        
        // Apply pooler if available
        std::optional<ttnn::Tensor> pooled_output = std::nullopt;
        if (pooler_) {
            pooled_output = pooler_->forward(sequence_output);
        }
        
        // Extract optional outputs from encoder
        std::optional<std::vector<ttnn::Tensor>> all_hidden_states = std::nullopt;
        std::optional<std::vector<ttnn::Tensor>> all_attentions = std::nullopt;
        
        if (actual_output_hidden_states && std::get<1>(encoder_outputs).has_value()) {
            all_hidden_states = std::get<1>(encoder_outputs);
        }
        
        if (actual_output_attentions && std::get<2>(encoder_outputs).has_value()) {
            all_attentions = std::get<2>(encoder_outputs);
        }
        
        return std::make_tuple(
            sequence_output,
            pooled_output,
            all_hidden_states,
            all_attentions
        );
        
    } catch (const std::exception& e) {
        std::cerr << "Error in TtDeiTModel forward pass: " << e.what() << std::endl;
        throw;
    }
}

std::vector<std::optional<ttnn::Tensor>> TtDeiTModel::get_head_mask(
    const ttnn::Tensor* head_mask,
    int num_hidden_layers
) {
    std::vector<std::optional<ttnn::Tensor>> processed_mask(num_hidden_layers);
    
    if (head_mask == nullptr) {
        // Return vector of nullopt if no head mask provided
        for (int i = 0; i < num_hidden_layers; ++i) {
            processed_mask[i] = std::nullopt;
        }
    } else {
        // Process the provided head mask
        // This is a simplified implementation - in practice you'd need to
        // handle different head mask shapes and broadcast appropriately
        for (int i = 0; i < num_hidden_layers; ++i) {
            processed_mask[i] = *head_mask;
        }
    }
    
    return processed_mask;
}


TtDeiTModel::~TtDeiTModel() {
    // Unique pointers will automatically clean up
}
