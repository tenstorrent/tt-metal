// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include "deit_patch_embeddings.h"
#include <stdexcept>
#include <iostream>

TtDeiTPatchEmbeddings::TtDeiTPatchEmbeddings(
    const DeiTConfig& config,
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : projection_(torch::nn::Conv2dOptions(config.num_channels, config.hidden_size, config.patch_size)
                .stride(config.patch_size)
                .padding(0)) {
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
    std::string weight_key = base_address + "projection.weight";
    std::string bias_key = base_address + "projection.bias";

    if (state_dict.find(weight_key) == state_dict.end()) {
        throw std::runtime_error("Missing projection weight in state_dict: " + weight_key);
    }
    if (state_dict.find(bias_key) == state_dict.end()) {
        throw std::runtime_error("Missing projection bias in state_dict: " + bias_key);
    }

    // Load weights and bias into the Conv2d module
    auto weight_torch = state_dict[weight_key];
    auto bias_torch = state_dict[bias_key];

    // Set the weights and bias for the Conv2d module
    projection_->weight.data() = weight_torch;
    projection_->bias.data() = bias_torch;

    std::cout << "TtDeiTPatchEmbeddings initialized with:" << std::endl;
    std::cout << "  Image size: (" << image_size_.first << ", " << image_size_.second << ")" << std::endl;
    std::cout << "  Patch size: (" << patch_size_.first << ", " << patch_size_.second << ")" << std::endl;
    std::cout << "  Num patches: " << num_patches_ << std::endl;
    std::cout << "  Num channels: " << num_channels_ << std::endl;
    std::cout << "  Hidden size: " << hidden_size_ << std::endl;
}

torch::Tensor TtDeiTPatchEmbeddings::forward(const torch::Tensor& pixel_values) {
    // Validate input dimensions
    auto input_shape = pixel_values.sizes();
    int num_channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];

    // Check channel dimension
    if (num_channels != num_channels_) {
        throw std::invalid_argument(
            "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
        );
    }

    // Check image size
    if (height != image_size_.first || width != image_size_.second) {
        throw std::invalid_argument(
            "Input image size (" + std::to_string(height) + "*" + std::to_string(width) +
            ") doesn't match model (" + std::to_string(image_size_.first) + "*" +
            std::to_string(image_size_.second) + ")."
        );
    }

    // Perform 2D convolution: x = self.projection(pixel_values)
    auto x = projection_->forward(pixel_values);

    // Flatten and transpose: .flatten(2).transpose(1, 2)
    // flatten(2) flattens from dimension 2 onwards
    x = x.flatten(2);
    // transpose(1, 2) swaps dimensions 1 and 2
    x = x.transpose(1, 2);

    return x;
}
