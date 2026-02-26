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
#include <optional>

#include <torch/torch.h>
#include <torch/script.h>

#include "../tt_cpp/deit_config.h"
#include "../helper_funcs.h"
#include "../tt_cpp/deit_embeddings.h"
#include "../helper_funcs.h"

namespace {

/**
 * Test DeiT Embeddings inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_embeddings_inference(const std::string& model_path) {
    const double pcc_threshold = 0.99;

    // Setup base address
    std::string base_address = "model.embeddings";
    bool use_mask_token = false;

    // Load state dict and model
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;

    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();

        std::cout << "Successfully loaded model from: " << model_path << std::endl;

        // Load model parameters to state_dict
        std::vector<std::string> required_params = {
            "patch_embeddings.projection.weight", "patch_embeddings.projection.bias",
            "cls_token", "distillation_token", "position_embeddings"
        };

        // Add mask_token if needed
        if (use_mask_token) {
            required_params.push_back("mask_token");
        }

        // Use named_parameters() method to get parameters directly
        auto named_params = model.named_parameters();
        std::unordered_map<std::string, at::Tensor> param_map;
        for (const auto& pair : named_params) {
            param_map[pair.name] = pair.value;
        }

        for (const auto& required : required_params) {
            std::string full_key = base_address+ "." + required;
            if (param_map.find(full_key) != param_map.end()) {
                state_dict[full_key] = param_map[full_key];
            } else {
                std::cerr << "Warning: Required parameter not found: " << full_key << std::endl;
            }
        }

        std::cout << "Loaded " << state_dict.size() << " embeddings parameters" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }

    // Get the embeddings module from PyTorch model
    torch::jit::script::Module embeddings_module;
    try {
        auto model_to_use = model.attr("model").toModule();
        embeddings_module = model_to_use.attr("embeddings").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get embeddings module: " << e.what() << std::endl;
        return;
    }

    // Create input tensor: [batch_size=1, channels=3, height=224, width=224]
    torch::Tensor input_tensor = torch::randn({1, 3, 224, 224}, torch::kFloat32);

    // Call embeddings module forward
    // For traced model, bool_masked_pos is typically None/nullptr
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    // Note: bool_masked_pos parameter might be optimized away in traced model

    auto output = embeddings_module.forward(inputs);
    auto torch_output = output.toTensor();

    // Create DeiT config
    DeiTConfig config;

    // Initialize device
    int device_id = 0;
    auto device = ttnn::MeshDevice::create(ttnn::MeshDeviceConfig(ttnn::MeshShape(1, 1), std::nullopt, {device_id}));

    // Setup TT model
    TtDeiTEmbeddings tt_embeddings(config, state_dict, base_address, device, use_mask_token);

    // Run TT model inference
    // Convert input to ttnn tensor (NHWC)
    auto permuted_input = input_tensor.permute({0, 2, 3, 1}); // NCHW -> NHWC
    auto ttnn_input = helper_funcs::from_torch(permuted_input, std::nullopt, ttnn::Layout::ROW_MAJOR);
    ttnn_input = ttnn::to_device(ttnn_input, device.get(), ttnn::DRAM_MEMORY_CONFIG);
    
    // bool_masked_pos is std::nullopt (equivalent to None in Python)
    auto tt_output_tensor = tt_embeddings.forward(ttnn_input, std::nullopt);

    // Convert output back to torch for comparison
    auto tt_output_host = ttnn::from_device(tt_output_tensor);
    // Output from embeddings is usually TILE layout, convert to torch
    auto tt_output_torch = helper_funcs::to_torch(tt_output_host);

    // Compute PCC between PyTorch and TT outputs
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output_torch);

    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;

    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT Embeddings test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
        throw std::runtime_error("Test failed: PCC below threshold");
    }
}

void test_deit_embeddings_inference_mock(double pcc_threshold = 0.99) {
    std::cout << "Running mock test with random weights..." << std::endl;
    
    // Config
    DeiTConfig config;
    std::string base_address = "model.embeddings";
    bool use_mask_token = false;
    
    // Generate random weights
    std::unordered_map<std::string, torch::Tensor> state_dict;
    
    auto projection_weight = torch::randn({config.hidden_size, config.num_channels, config.patch_size, config.patch_size});
    auto projection_bias = torch::randn({config.hidden_size});
    auto cls_token = torch::randn({1, 1, config.hidden_size});
    auto distillation_token = torch::randn({1, 1, config.hidden_size});
    // num_patches + 2 (cls, dist)
    int num_patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
    auto position_embeddings = torch::randn({1, num_patches + 2, config.hidden_size});
    
    state_dict[base_address + ".patch_embeddings.projection.weight"] = projection_weight;
    state_dict[base_address + ".patch_embeddings.projection.bias"] = projection_bias;
    state_dict[base_address + ".cls_token"] = cls_token;
    state_dict[base_address + ".distillation_token"] = distillation_token;
    state_dict[base_address + ".position_embeddings"] = position_embeddings;
    
    if (use_mask_token) {
        state_dict[base_address + ".mask_token"] = torch::randn({1, 1, config.hidden_size});
    }
    
    // Input
    auto pixel_values = torch::randn({1, 3, config.image_size, config.image_size});
    
    // Expected output (PyTorch implementation)
    // 1. Patch Embeddings
    auto torch_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(config.num_channels, config.hidden_size, config.patch_size)
                                        .stride(config.patch_size).padding(0));
    torch_conv->weight.data() = projection_weight;
    torch_conv->bias.data() = projection_bias;
    
    auto embeddings = torch_conv->forward(pixel_values).flatten(2).transpose(1, 2); // [B, N, D]
    
    // 2. Tokens
    auto batch_size = pixel_values.size(0);
    auto cls_token_expanded = cls_token.expand({batch_size, -1, -1});
    auto dist_token_expanded = distillation_token.expand({batch_size, -1, -1});
    
    // 3. Concat
    embeddings = torch::cat({cls_token_expanded, dist_token_expanded, embeddings}, 1);
    
    // 4. Pos Embs
    embeddings = embeddings + position_embeddings;
    
    auto torch_output = embeddings;
    
    // TTNN implementation
    int device_id = 0;
    auto device = ttnn::MeshDevice::create(ttnn::MeshDeviceConfig(ttnn::MeshShape(1, 1), std::nullopt, {device_id}));
    
    TtDeiTEmbeddings tt_embeddings(config, state_dict, base_address, device, use_mask_token);
    
    auto permuted_input = pixel_values.permute({0, 2, 3, 1}); // NCHW -> NHWC
    auto ttnn_input = helper_funcs::from_torch(permuted_input, std::nullopt, ttnn::Layout::ROW_MAJOR);
    ttnn_input = ttnn::to_device(ttnn_input, device.get(), ttnn::DRAM_MEMORY_CONFIG);
    
    auto tt_output_tensor = tt_embeddings.forward(ttnn_input, std::nullopt);
    auto tt_output_host = ttnn::from_device(tt_output_tensor);
    auto tt_output_torch = helper_funcs::to_torch(tt_output_host);
    
    // PCC
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output_torch);
    std::cout << "Mock Test PCC: " << pcc << std::endl;
    
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: Mock Test" << std::endl;
    } else {
        throw std::runtime_error("FAILED PCC: " + std::to_string(pcc));
    }
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Embeddings test..." << std::endl;

    if (argc < 2) {
        std::cout << "No model path provided. Running mock test..." << std::endl;
        try {
            test_deit_embeddings_inference_mock();
        } catch (const std::exception& e) {
            std::cerr << "Error during mock test: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

    // Default model path (relative path)
    /*if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return -1;
    }*/
    std::string model_path = argv[1];

    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "Using model path: " << model_path << std::endl;

    try {
        test_deit_embeddings_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
