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
#include "../tt_cpp/deit_self_output.h"
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
 * Test DeiT Self Output inference
 */
void test_deit_self_output_inference() {
    const double pcc_threshold = 0.99;
    
    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0, 
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    
    // Setup base address and model path
    int layer_index = 0;  // Default to layer 0
    std::string base_address = "model.encoder.layer." + std::to_string(layer_index) + ".attention.output.";
    std::string model_path = "/home/openkylin/like/github/tt/like/tt-metal/models/experimental/deit/deit_cpp/model/deit_traced_model.pt";
    
    // Load state dict and model (inlined from load_deit_state_dict_and_model)
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;
    
    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();
        
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
            state_dict[full_key] = param_map[full_key];
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
    
    // Get the self-output module: model.encoder.layer.0.attention.output
    torch::jit::script::Module self_output_module;
    try {
        auto model_to_use = model.attr("model").toModule();
        auto encoder = model_to_use.attr("encoder").toModule();
        
        // Access encoder.layer.0
        auto layer = encoder.attr("layer").toModule();
        auto layer_0 = layer.attr("0").toModule();  // Access layer 0
        auto attention = layer_0.attr("attention").toModule();
        self_output_module = attention.attr("output").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get self-output module: " << e.what() << std::endl;
        return;
    }

    // Create input tensors: [1, 198, 768] for hidden_states
    // DeiT uses 196 patches + 1 CLS token + 1 distillation token = 198 tokens
    torch::Tensor hidden_states = torch::randn({1, 198, 768}, torch::kFloat32);

    
    // Call self-output module forward
    // The PyTorch forward method only takes hidden_states (self is implicit)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(hidden_states);
    
    auto output = self_output_module.forward(inputs);
    auto torch_output = output.toTensor();
    
    // Create DeiT config
    DeiTConfig config;
    // Setup TT model
    TtDeiTSelfOutput tt_self_output(config, device, state_dict, base_address);
    
    // Convert inputs to TT tensors
    auto tt_hidden_states = helper_funcs::torch_to_tt_tensor_tile(hidden_states, device);

    
    // Run TT model inference
    auto tt_out = tt_self_output.forward(tt_hidden_states);
    
    // Convert TT output back to torch tensor
    // First move tensor to host if it's on device
    auto tt_out_host = ttnn::from_device(tt_out);
    auto tt_output_torch = helper_funcs::to_torch(tt_out_host);
    tt_output_torch = tt_output_torch.squeeze(0); // Remove batch dimension
    
    // Compute PCC between PyTorch and TT outputs
    double pcc = compute_pcc(torch_output, tt_output_torch);
    
    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "Test PASSED! PCC: " << pcc << " (threshold: " << pcc_threshold << ")" << std::endl;
    } else {
        std::cout << "Test FAILED! PCC: " << pcc << " (threshold: " << pcc_threshold << ")" << std::endl;
    }

    // Device cleanup is handled automatically by MeshDevice
    device->close();
}

} // anonymous namespace

int main(int argc, char** argv) {
    
    try {
        test_deit_self_output_inference();
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}