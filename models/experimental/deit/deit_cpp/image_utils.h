// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_IMAGE_UTILS_H
#define DEIT_CPP_IMAGE_UTILS_H

#include <string>
#include <torch/torch.h>

namespace image_utils {

/**
 * Load and preprocess an image for DeiT inference
 * Mimics the functionality of AutoImageProcessor from transformers
 * @param image_path Path to the image file
 * @return Preprocessed image tensor [1, 3, 224, 224] ready for DeiT inference
 */
torch::Tensor load_and_preprocess_image(
    const std::string& image_path
);

} // namespace image_utils

#endif // DEIT_CPP_IMAGE_UTILS_H
