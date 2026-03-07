// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include "deit_patch_embeddings.h"
#include "../helper_funcs.h"
#include <stdexcept>
#include <iostream>
#include <variant>
#include <tuple>
#include <array>
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/host_api.hpp>

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

    // Initialize padded_num_channels_
    padded_num_channels_ = num_channels_;
    if (padded_num_channels_ < 16) {
        padded_num_channels_ = 16;
    }

    // Load projection weights and bias from state_dict
    std::string weight_key = base_address + "projection.weight";
    std::string bias_key = base_address + "projection.bias";

    if (state_dict.find(weight_key) == state_dict.end()) {
        throw std::runtime_error("Missing projection weight in state_dict: " + weight_key);
    }
    if (state_dict.find(bias_key) == state_dict.end()) {
        throw std::runtime_error("Missing projection bias in state_dict: " + bias_key);
    }

    // Load weights and bias
    auto weight_torch = state_dict[weight_key];
    auto bias_torch = state_dict[bias_key];

    // Pad weights if needed
    if (padded_num_channels_ > num_channels_) {
        auto options = weight_torch.options();
        auto padded_weight_torch = torch::zeros({weight_torch.size(0), padded_num_channels_, weight_torch.size(2), weight_torch.size(3)}, options);
        using namespace torch::indexing;
        padded_weight_torch.index_put_({Slice(), Slice(0, num_channels_), Slice(), Slice()}, weight_torch);
        weight_torch = padded_weight_torch;
    }

    // Convert to TTNN tensors
    // weight shape: (out_channels, in_channels, kernel_h, kernel_w)
    // ttnn::conv2d requires weights on host in ROW_MAJOR layout
    weight_ = helper_funcs::from_torch(weight_torch, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    // Bias must be 4D [1, 1, 1, out_channels]
    auto bias_torch_reshaped = bias_torch.reshape({1, 1, 1, -1});
    bias_ = helper_funcs::from_torch(bias_torch_reshaped, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    // Initialize configs
    conv_config_.weights_dtype = ttnn::DataType::BFLOAT16;

    compute_config_ = ttnn::init_device_compute_kernel_config(
        device_->arch(),
        std::nullopt,
        MathFidelity::HiFi4
    );
}

ttnn::Tensor TtDeiTPatchEmbeddings::forward(const ttnn::Tensor& pixel_values) {
    // Validate input dimensions (expect NHWC)
    auto input_shape = pixel_values.logical_shape();
    int batch_size = input_shape[0];
    int height = input_shape[1];
    int width = input_shape[2];
    int num_channels = input_shape[3];

    // Check channel dimension
    if (num_channels != num_channels_ && num_channels != padded_num_channels_) {
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

    // Perform 2D convolution
    // output shape: (batch_size, height/patch_size, width/patch_size, hidden_size)
    auto result = ttnn::conv2d(
        pixel_values,
        weight_,
        device_.get(),
        padded_num_channels_,
        hidden_size_,
        batch_size,
        height,
        width,
        std::array<uint32_t, 2>{static_cast<uint32_t>(patch_size_.first), static_cast<uint32_t>(patch_size_.second)}, // kernel_size
        std::array<uint32_t, 2>{static_cast<uint32_t>(patch_size_.first), static_cast<uint32_t>(patch_size_.second)}, // stride
        std::array<uint32_t, 2>{0, 0}, // padding
        std::array<uint32_t, 2>{1, 1}, // dilation
        1,      // groups
        ttnn::DataType::BFLOAT16, // dtype
        bias_,
        conv_config_,
        compute_config_,
        std::nullopt,
        std::nullopt,
        true,         // return_output_dim
        true          // return_weights_and_bias
    );

    // Extract results from variant
    // We expect the 4th alternative: tuple<Tensor, tuple<H,W>, tuple<Tensor, opt<Tensor>>>
    auto& result_tuple = std::get<3>(result);
    auto output_tensor = std::get<0>(result_tuple);
    auto dims = std::get<1>(result_tuple);
    auto output_height = std::get<0>(dims);
    auto output_width = std::get<1>(dims);
    auto weights_and_bias = std::get<2>(result_tuple);
    auto weight_tensor = std::get<0>(weights_and_bias);
    auto bias_tensor = std::get<1>(weights_and_bias);

    // Update weight and bias (in case they were modified/moved by conv2d)
    weight_ = weight_tensor;
    if (bias_tensor.has_value()) {
        bias_ = bias_tensor.value();
    }

    // Reshape to (batch_size, 1, num_patches, hidden_size) to match ttnn 4D requirement
    // num_patches = output_height * output_width
    auto output_reshaped = ttnn::reshape(
        output_tensor,
        ttnn::Shape{
            static_cast<uint32_t>(batch_size),
            1,
            static_cast<uint32_t>(output_height * output_width),
            static_cast<uint32_t>(hidden_size_)});

    return output_reshaped;
}
