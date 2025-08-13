// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <torch/torch.h>

#include "../tt_cpp/deit_config.h"
#include "../helper_funcs.h"
#include "../tt_cpp/deit_patch_embeddings.h"

namespace {

/**
 * Test DeiT Patch Embeddings inference
 * Matches the Python version test logic
 */
void test_deit_patch_embeddings_inference(double pcc_threshold = 0.99) {
    std::cout << "Setting up PyTorch model..." << std::endl;

    // Setup base address (matching Python version)
    std::string base_address = "model.embeddings.patch_embeddings.";

    // Create mock state_dict with random weights (simulating pretrained model)
    // In real scenario, this would be loaded from a pretrained DeiT model
    std::unordered_map<std::string, torch::Tensor> state_dict;

    // Create DeiT config
    DeiTConfig config;

    // Generate mock weights matching DeiT architecture
    // Conv2d weight shape: [out_channels, in_channels, kernel_height, kernel_width]
    auto projection_weight = torch::randn({config.hidden_size, config.num_channels, config.patch_size, config.patch_size});
    auto projection_bias = torch::randn({config.hidden_size});

    state_dict[base_address + "projection.weight"] = projection_weight;
    state_dict[base_address + "projection.bias"] = projection_bias;

    std::cout << "Created state_dict with " << state_dict.size() << " parameters" << std::endl;

    // Create PyTorch reference model (simulating torch_patch_embeddings)
    auto torch_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(config.num_channels, config.hidden_size, config.patch_size)
                                        .stride(config.patch_size)
                                        .padding(0));

    // Load weights into PyTorch model
    torch_conv->weight.data() = projection_weight;
    torch_conv->bias.data() = projection_bias;
    torch_conv->eval();

    // Synthesize input (matching Python version)
    torch::Tensor pixel_values = torch::randn({1, 3, 224, 224}, torch::kFloat32);

    std::cout << "Input shape: " << pixel_values.sizes() << std::endl;

    // Get PyTorch reference output
    auto torch_output = torch_conv->forward(pixel_values).flatten(2).transpose(1, 2);

    std::cout << "Setting up TT model..." << std::endl;

    // Setup TT model (now pure libtorch)
    TtDeiTPatchEmbeddings tt_patch_embeddings(config, state_dict, base_address);

    // Run TT model inference (now using torch tensors directly)
    auto tt_output = tt_patch_embeddings.forward(pixel_values);

    // Compute PCC between PyTorch and TT outputs (matching Python version logic)
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output);

    // Log results (similar to Python version)
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output.sizes() << std::endl;
    std::cout << "Output PCC: " << pcc << std::endl;

    // Check PCC passing condition (matching Python version)
    bool pcc_passing = (pcc >= pcc_threshold);

    if (pcc_passing) {
        std::cout << "PASSED: DeiT Patch Embeddings test with PCC = " << pcc << std::endl;
    } else {
        std::cerr << "FAILED: Low PCC: " << pcc << " (threshold: " << pcc_threshold << ")" << std::endl;
        throw std::runtime_error("Failed! Low pcc: " + std::to_string(pcc));
    }

    // No device cleanup needed for pure libtorch implementation
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Patch Embeddings test..." << std::endl;

    try {
        // Run test with default PCC threshold of 0.99 (no device needed for pure libtorch)
        test_deit_patch_embeddings_inference(0.99);

    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
