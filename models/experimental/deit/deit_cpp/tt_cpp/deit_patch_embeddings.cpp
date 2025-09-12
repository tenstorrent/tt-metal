// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include "deit_patch_embeddings.h"
#include <stdexcept>
#include <iostream>

TtDeiTPatchEmbeddings::TtDeiTPatchEmbeddings(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address,
    std::shared_ptr<ttnn::MeshDevice> device
) : device_(device) {
    // Extract configuration parameters
    auto image_size = config.image_size;
    auto patch_size = config.patch_size;
    num_channels_ = config.num_channels;
    hidden_size_ = config.hidden_size;
    
    // Handle image_size and patch_size (they are int values)
    image_size_ = {image_size, image_size};
    patch_size_ = {patch_size, patch_size};
    
    // Calculate number of patches
    num_patches_ = (image_size_.first / patch_size_.first) * (image_size_.second / patch_size_.second);
    
    // Load projection weights and bias from state_dict
    std::string weight_key = base_address + ".projection.weight";
    std::string bias_key = base_address + ".projection.bias";
    
    if (state_dict.find(weight_key) == state_dict.end()) {
        throw std::runtime_error("Missing projection weight in state_dict: " + weight_key);
    }
    if (state_dict.find(bias_key) == state_dict.end()) {
        throw std::runtime_error("Missing projection bias in state_dict: " + bias_key);
    }
    
    // Convert torch tensors to ttnn tensors
    auto weight_torch = state_dict[weight_key];
    auto bias_torch = state_dict[bias_key];
    
    // Convert to TTNN tensors and move to device
    projection_weight_ = helper_funcs::torch_to_tt_tensor_tile(weight_torch, device_);
    projection_bias_ = helper_funcs::torch_to_tt_tensor_tile(bias_torch, device_);
    
    std::cout << "TtDeiTPatchEmbeddings initialized with:" << std::endl;
    std::cout << "  Image size: (" << image_size_.first << ", " << image_size_.second << ")" << std::endl;
    std::cout << "  Patch size: (" << patch_size_.first << ", " << patch_size_.second << ")" << std::endl;
    std::cout << "  Num patches: " << num_patches_ << std::endl;
    std::cout << "  Num channels: " << num_channels_ << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
}

ttnn::Tensor TtDeiTPatchEmbeddings::forward(const ttnn::Tensor& pixel_values) {
    // Validate input dimensions
    validate_input(pixel_values);
    
    // Apply 2D convolution to extract patches
    // For now, use a simplified approach with matrix operations
    // TODO: Implement proper conv2d when available
    
    // Get input shape
    auto shape = pixel_values.get_logical_shape();
    auto batch_size = shape[0];
    auto channels = shape[1];
    auto height = shape[2];
    auto width = shape[3];
    
    // For simplicity, reshape input to [batch_size, num_patches, patch_size*patch_size*channels]
    // and use matrix multiplication with reshaped weights
    int patch_area = patch_size_.first * patch_size_.second * num_channels_;
    
    // Reshape pixel_values for patch extraction
    // This is a simplified version - in practice, you'd need proper patch extraction
    ttnn::Shape input_shape({batch_size, static_cast<uint32_t>(num_patches_), static_cast<uint32_t>(patch_area)});
    auto reshaped_input = ttnn::reshape(ttnn::DefaultQueueId, pixel_values, input_shape);
    
    // Reshape projection weight for matrix multiplication
    auto weight_shape = projection_weight_.get_logical_shape();
    ttnn::Shape weight_reshape({static_cast<uint32_t>(patch_area), static_cast<uint32_t>(hidden_size_)});
    auto reshaped_weight = ttnn::reshape(ttnn::DefaultQueueId, projection_weight_, weight_reshape);
    
    // Perform matrix multiplication: [batch_size, num_patches, patch_area] x [patch_area, hidden_size]
    auto conv_output = ttnn::matmul(reshaped_input, reshaped_weight);
    
    // Add bias if available
    if (projection_bias_.get_logical_shape().volume() > 0) {
        conv_output = ttnn::add(conv_output, projection_bias_);
    }
    
    auto transposed = conv_output;
    
    return transposed;
}

void TtDeiTPatchEmbeddings::validate_input(const ttnn::Tensor& pixel_values) const {
    auto shape = pixel_values.get_logical_shape();
    
    if (shape.rank() != 4) {
        throw std::invalid_argument("Input tensor must be 4D: [batch_size, num_channels, height, width]");
    }
    
    auto batch_size = shape[0];
    auto num_channels = shape[1];
    auto height = shape[2];
    auto width = shape[3];
    
    if (static_cast<int>(num_channels) != num_channels_) {
        throw std::invalid_argument(
            "Channel dimension mismatch. Expected: " + std::to_string(num_channels_) + 
            ", Got: " + std::to_string(num_channels)
        );
    }
    
    if (static_cast<int>(height) != image_size_.first || static_cast<int>(width) != image_size_.second) {
        throw std::invalid_argument(
            "Input image size (" + std::to_string(height) + "*" + std::to_string(width) + 
            ") doesn't match model (" + std::to_string(image_size_.first) + "*" + 
            std::to_string(image_size_.second) + ")"
        );
    }
}