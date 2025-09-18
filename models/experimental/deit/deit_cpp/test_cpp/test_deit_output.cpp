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
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#include "../tt_cpp/deit_config.h"
#include "../tt_cpp/deit_output.h"
#include "../helper_funcs.h"

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
 * Test DeiT Output inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_output_inference(const std::string& model_path) {
    const double pcc_threshold = 0.99;
    
    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0, 
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    
    // Setup base address
    int layer_index = 0;  // Default to layer 0
    std::string base_address = "model.encoder.layer." + std::to_string(layer_index) + ".output.";
    
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
            "dense.weight", "dense.bias"
        };
        
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
            }
        }
        
        std::cout << "Loaded " << state_dict.size() << " output parameters for layer " << layer_index << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
    
    // Get the output module from the model
    torch::jit::script::Module output_module;
    try {
        auto model_to_use = model.attr("model").toModule();
        auto encoder = model_to_use.attr("encoder").toModule();
        auto layer = encoder.attr("layer").toModule();
        auto layer_0 = layer.attr("0").toModule();  // Access layer 0
        output_module = layer_0.attr("output").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get output module: " << e.what() << std::endl;
        return;
    }

    // Create input tensors based on Python test
    // hidden_state_shape = [1, 198, 3072] - intermediate output
    torch::Tensor hidden_state = torch::randn({1, 198, 3072}, torch::kFloat32);
    
    // input_tensor_shape = [1, 198, 768] - residual connection input
    torch::Tensor input_tensor = torch::randn({1, 198, 768}, torch::kFloat32);
    
    // Call output module forward
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(hidden_state);
    inputs.push_back(input_tensor);
    
    auto output = output_module.forward(inputs);
    auto torch_output = output.toTensor();

    // Create DeiT config
    DeiTConfig config;
    
    // Setup TT model
    TtDeiTOutput tt_output(config, device, state_dict, base_address);
    
    // Convert inputs to TT tensors
    auto tt_hidden_state = helper_funcs::torch_to_tt_tensor_tile(hidden_state, device);
    auto tt_input_tensor = helper_funcs::torch_to_tt_tensor_tile(input_tensor, device);
    
    // Run TT model inference
    auto tt_out = tt_output.forward(tt_hidden_state, tt_input_tensor);
    
    // Convert TT output back to torch tensor
    auto tt_out_host = ttnn::from_device(tt_out);
    auto tt_output_torch = helper_funcs::to_torch(tt_out_host);
    
    // Compute PCC between PyTorch and TT outputs
    double pcc = compute_pcc(torch_output, tt_output_torch);
    
    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;
    
    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT Output test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }
    
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Output test..." << std::endl;
    
    // Default model path (relative path)
    std::string model_path = "models/experimental/deit/deit_cpp/deit_model/deit_encoder_model.pt";
    
    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Using model path: " << model_path << std::endl;
    
    try {
        test_deit_output_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}