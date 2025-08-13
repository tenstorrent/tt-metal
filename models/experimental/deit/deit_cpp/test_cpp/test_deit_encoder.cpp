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
#include "../tt_cpp/deit_encoder.h"
#include "../helper_funcs.h"

namespace {

/**
 * Test DeiT Encoder inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_encoder_inference(const std::string& model_path) {
    const double pcc_threshold = 0.97;

    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0,
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));

    // Setup base address
    std::string base_address = "model.encoder";

    // Load state dict and model
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;

    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();

        std::cout << "Successfully loaded model from: " << model_path << std::endl;

        // Load all encoder parameters to state_dict
        // For encoder, we need all layer parameters
        std::vector<std::string> layer_param_suffixes = {
            "attention.attention.query.weight", "attention.attention.query.bias",
            "attention.attention.key.weight", "attention.attention.key.bias",
            "attention.attention.value.weight", "attention.attention.value.bias",
            "attention.output.dense.weight", "attention.output.dense.bias",
            "layernorm_before.weight", "layernorm_before.bias",
            "intermediate.dense.weight", "intermediate.dense.bias",
            "output.dense.weight", "output.dense.bias",
            "layernorm_after.weight", "layernorm_after.bias"
        };

        // Use named_parameters() method to get parameters directly
        auto named_params = model.named_parameters();
        std::unordered_map<std::string, at::Tensor> param_map;
        for (const auto& pair : named_params) {
            param_map[pair.name] = pair.value;
        }

        // Load parameters for all 12 layers (DeiT-base has 12 layers)
        const int num_layers = 12;
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            for (const auto& suffix : layer_param_suffixes) {
                std::string full_key = base_address + ".layer." + std::to_string(layer_idx) + "." + suffix;
                if (param_map.find(full_key) != param_map.end()) {
                    state_dict[full_key] = param_map[full_key];
                }
            }
        }

        std::cout << "Loaded " << state_dict.size() << " encoder parameters for " << num_layers << " layers" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }

    // Get the encoder module: model.encoder
    torch::jit::script::Module encoder_module;
    try {
        auto model_to_use = model.attr("model").toModule();
        encoder_module = model_to_use.attr("encoder").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get encoder module: " << e.what() << std::endl;
        return;
    }

    // Create input tensor: [batch_size=1, seq_len=224, hidden_size=768]
    // Note: test_deit_encoder.py uses input_shape = torch.Size([1, 1, 224, 768])
    // This is different from layer test which uses 198 tokens
    torch::Tensor input_tensor = torch::randn({1, 1, 224, 768}, torch::kFloat32);

    // Prepare input for encoder module
    auto hidden_states = input_tensor.squeeze(0); // Remove batch dimension: [224, 768]
    torch::Tensor torch_head_mask; // None equivalent

    // Call encoder module forward
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(hidden_states);

    auto output = encoder_module.forward(inputs);
    // Handle output - it might be a tensor or tuple
    torch::Tensor torch_output;
    if (output.isTuple()) {
        torch_output = output.toTuple()->elements()[0].toTensor();
    } else {
        torch_output = output.toTensor();
    }

    // Create DeiT config
    DeiTConfig config;

    // Setup TT model
    TtDeiTEncoder tt_encoder(config, device, state_dict, base_address);

    // Convert input to TT tensor
    // Note: input_tensor shape is [1, 1, 224, 768], but we need [1, 224, 768] for encoder
    // Reshape to remove the extra dimension
    auto reshaped_input = input_tensor.squeeze(1); // Remove dimension 1: [1, 224, 768]
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(reshaped_input, device);

    // Run TT model inference
    std::optional<std::vector<ttnn::Tensor>> head_mask = std::nullopt;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;

    auto [tt_out, all_hidden_states, all_attentions] = tt_encoder.forward(
        tt_input, head_mask, output_attentions, output_hidden_states, return_dict
    );

    // Convert TT output back to torch tensor
    // First move tensor to host if it's on device
    auto tt_out_host = ttnn::from_device(tt_out);
    auto tt_output_torch = helper_funcs::to_torch(tt_out_host);
    tt_output_torch = tt_output_torch.squeeze(0); // Remove batch dimension

    // Compute PCC between PyTorch and TT outputs
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output_torch);

    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;

    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT Encoder test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Clean up device resources
    device->close();
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Encoder test..." << std::endl;

    // Default model path (relative path)
    std::string model_path = "models/experimental/deit/deit_cpp/deit_model/deit_encoder_model.pt";

    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "Using model path: " << model_path << std::endl;

    try {
        test_deit_encoder_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
