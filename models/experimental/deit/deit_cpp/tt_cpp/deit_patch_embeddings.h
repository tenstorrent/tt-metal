// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "deit_config.h"
#include "helper_funcs.h"

class TtDeiTPatchEmbeddings {
public:
    /**
     * Constructor for TtDeiTPatchEmbeddings
     * @param config DeiT configuration
     * @param state_dict Model state dictionary containing weights and biases
     * @param base_address Base address for parameter lookup in state_dict
     * @param device TTNN device for tensor operations
     */
    TtDeiTPatchEmbeddings(
        const DeiTConfig& config,
        std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address,
        std::shared_ptr<ttnn::MeshDevice> device
    );

    /**
     * Forward pass for patch embeddings
     * Converts input pixel values to patch embeddings using 2D convolution
     * @param pixel_values Input tensor with shape [batch_size, num_channels, height, width]
     * @return Patch embeddings tensor with shape [batch_size, num_patches, hidden_size]
     */
    ttnn::Tensor forward(const ttnn::Tensor& pixel_values);

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
    
    // TTNN tensors for convolution weights and bias
    ttnn::Tensor projection_weight_;
    ttnn::Tensor projection_bias_;
    
    // Device reference
    std::shared_ptr<ttnn::MeshDevice> device_;
    
    // Helper function to validate input dimensions
    void validate_input(const ttnn::Tensor& pixel_values) const;
};