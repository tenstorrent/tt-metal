// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include "deit_config.h"

class TtDeiTPatchEmbeddings {
public:
    /**
     * Constructor for TtDeiTPatchEmbeddings
     * @param config DeiT configuration
     * @param state_dict Model state dictionary containing weights and biases
     * @param base_address Base address for parameter lookup in state_dict
     */
    TtDeiTPatchEmbeddings(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address
    );

    /**
     * Forward pass for patch embeddings
     * Converts input pixel values to patch embeddings using 2D convolution
     * @param pixel_values Input tensor with shape [batch_size, num_channels, height, width]
     * @return Patch embeddings tensor with shape [batch_size, num_patches, hidden_size]
     */
    torch::Tensor forward(const torch::Tensor& pixel_values);

    // Getters
    int get_num_patches() const { return num_patches_; }
    std::pair<int, int> get_image_size() const { return image_size_; }
    std::pair<int, int> get_patch_size() const { return patch_size_; }
    int get_num_channels() const { return num_channels_; }

private:
    // Configuration parameters
    std::pair<int, int> image_size_;  // (height, width)
    std::pair<int, int> patch_size_;  // (patch_height, patch_width)
    int num_channels_;
    int num_patches_;
    int hidden_size_;
    
    // Model parameters - using libtorch Conv2d
    torch::nn::Conv2d projection_;
    

};