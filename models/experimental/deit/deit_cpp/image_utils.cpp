// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "image_utils.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <stdexcept>

namespace image_utils {

/**
 * Load and preprocess image for DeiT model inference
 * Mimics the functionality of AutoImageProcessor from transformers
 *
 * @param image_path Path to the input image file
 * @return Preprocessed image tensor [1, 3, 224, 224] ready for DeiT inference
 */
torch::Tensor load_and_preprocess_image(
    const std::string& image_path
) {
    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image from path: " + image_path);
    }

    // Convert BGR to RGB (OpenCV loads as BGR by default)
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Resize to 224x224 (DeiT input size)
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0/255.0);

    // Create torch tensor from OpenCV Mat
    // Shape: [H, W, C] -> [C, H, W]
    torch::Tensor tensor = torch::from_blob(
        float_image.data,
        {224, 224, 3},
        torch::kFloat32
    ).clone();

    // Permute dimensions from HWC to CHW
    tensor = tensor.permute({2, 0, 1});

    // Apply ImageNet normalization
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
    tensor = (tensor - mean) / std;

    // Add batch dimension: [C, H, W] -> [1, C, H, W]
    tensor = tensor.unsqueeze(0);

    // Return torch tensor directly
    return tensor;
}

} // namespace image_utils
