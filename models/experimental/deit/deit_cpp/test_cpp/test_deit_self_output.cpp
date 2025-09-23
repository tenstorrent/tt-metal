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
#include <ttnn/distributed/api.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#include "../tt_cpp/deit_config.h"
#include "../tt_cpp/deit_self_output.h"
#include "../helper_funcs.h"

namespace {

/**
 * Test DeiT Self Output inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_self_output_inference(const std::string& model_path) {
    const double pcc_threshold = 0.99;

    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0,
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));

    // Setup base address
    int layer_index = 0;  // Default to layer 0
    std::string base_address = "model.encoder.layer." + std::to_string(layer_index) + ".attention.output.";

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
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output_torch);

    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "Test PASSED! PCC: " << pcc << " (threshold: " << pcc_threshold << ")" << std::endl;
    } else {
        std::cout << "Test FAILED! PCC: " << pcc << " (threshold: " << pcc_threshold << ")" << std::endl;
    }

    // Clean up device resources
    ttnn::distributed::close_mesh_device(device);
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Self Output test..." << std::endl;

    // Default model path (relative path)
    std::string model_path = "models/experimental/deit/deit_cpp/deit_model/deit_encoder_model.pt";

    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "Using model path: " << model_path << std::endl;

    try {
        test_deit_self_output_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
