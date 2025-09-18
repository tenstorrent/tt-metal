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
#include <torch/script.h>

#include "../tt_cpp/deit_config.h"
#include "../tt_cpp/deit_embeddings.h"

namespace {

/**
 * Compute Pearson Correlation Coefficient (PCC) between two tensors
 * @param tensor1 First tensor
 * @param tensor2 Second tensor
 * @return PCC value
 */
double compute_pcc(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
    auto flat1 = tensor1.flatten().to(torch::kFloat32);
    auto flat2 = tensor2.flatten().to(torch::kFloat32);
    
    auto mean1 = flat1.mean();
    auto mean2 = flat2.mean();
    
    auto centered1 = flat1 - mean1;
    auto centered2 = flat2 - mean2;
    
    auto numerator = (centered1 * centered2).sum();
    auto denominator = torch::sqrt((centered1 * centered1).sum() * (centered2 * centered2).sum());
    
    return numerator.item<double>() / denominator.item<double>();
}

/**
 * Test DeiT Embeddings inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_embeddings_inference(const std::string& model_path) {
    const double pcc_threshold = 0.99;
    
    // Setup base address
    std::string base_address = "model.embeddings.";
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
            std::string full_key = base_address + required;
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
    
    // Setup TT model
    TtDeiTEmbeddings tt_embeddings(config, state_dict, base_address, use_mask_token);
    
    // Run TT model inference
    // bool_masked_pos is nullptr (equivalent to None in Python)
    auto tt_output_torch = tt_embeddings.forward(input_tensor, nullptr);
    
    // Compute PCC between PyTorch and TT outputs
    double pcc = compute_pcc(torch_output, tt_output_torch);
    
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

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Embeddings test..." << std::endl;
    
    // Default model path (relative path)
    std::string model_path = "../deit_model/deit_traced_model.pt";
    
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