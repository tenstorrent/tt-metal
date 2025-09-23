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
#include "../tt_cpp/deit_for_image_classification.h"
#include "../helper_funcs.h"

namespace {

/**
 * Load DeiT state dict and model for image classification
 * @param model_path Path to the traced model
 * @return Pair of state_dict and loaded model
 */
std::pair<std::unordered_map<std::string, torch::Tensor>, torch::jit::script::Module>
load_deit_image_classification_model(const std::string& model_path) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;

    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();

        std::cout << "Successfully loaded model from: " << model_path << std::endl;

        // Load all model parameters to state_dict
        auto named_params = model.named_parameters();
        for (const auto& pair : named_params) {
            state_dict[pair.name] = pair.value;
        }

        std::cout << "Loaded " << state_dict.size() << " parameters from model" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }

    return std::make_pair(state_dict, model);
}

/**
 * Test DeiT for image classification inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_for_image_classification_inference(const std::string& model_path) {
    const double pcc_threshold = 0.95;

    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0,
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));

    // Setup base address
    std::string base_address = "model.";

    // Load state dict and model
    auto [state_dict, model] = load_deit_image_classification_model(model_path);

    // Use a sample image path for testing (you can replace this with any valid image file)
    std::string test_image_path = "/home/openkylin/like/github/tt/like/tt-metal/models/experimental/deit/deit_cpp/deit_model/input_image.jpg";

    torch::Tensor pixel_values = helper_funcs::load_and_preprocess_image(test_image_path);

    std::cout << "Using load_and_preprocess_image function for input preprocessing" << std::endl;

    // Get PyTorch model output for reference
    torch::Tensor torch_output;
    try {
        // Call the model with pixel_values
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pixel_values);

        auto output = model.forward(inputs);

        // Extract logits from the output
        // The output structure depends on the traced model format
        if (output.isTuple()) {
            auto output_tuple = output.toTuple();
            torch_output = output_tuple->elements()[0].toTensor(); // First element should be logits
        } else {
            torch_output = output.toTensor();
        }

        std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to get PyTorch model output: " << e.what() << std::endl;
        return;
    }

    // Create DeiT config
    DeiTConfig config;

    // Setup TT model
    TtDeiTForImageClassification tt_model(config, state_dict, base_address, device);

    // Convert input to TT tensor
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(pixel_values, device);

    // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;

    auto [tt_logits, attention_weights, hidden_states] = tt_model.forward(
        tt_input,
        head_mask.has_value() ? &head_mask.value() : nullptr,
        output_attentions,
        output_hidden_states,
        return_dict
    );

    // Convert TT output back to torch tensor
    auto tt_logits_host = ttnn::from_device(tt_logits);
    auto tt_output_torch = helper_funcs::to_torch(tt_logits_host);

    // Ensure output shapes match for comparison
    if (tt_output_torch.dim() > torch_output.dim()) {
        tt_output_torch = tt_output_torch.squeeze(0); // Remove batch dimension if needed
    }

    // Extract the classification logits (typically [:, 0, :] for DeiT)
    if (tt_output_torch.dim() == 3 && tt_output_torch.size(1) > 1) {
        tt_output_torch = tt_output_torch.select(1, 0); // Select first token (CLS token)
    }

    // Compute PCC between PyTorch and TT outputs
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output_torch);

    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;

    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT For Image Classification test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Clean up device resources
    ttnn::distributed::close_mesh_device(device);
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::cout << "Starting DeiT for Image Classification test..." << std::endl;

        // Default model path (relative path)
        std::string model_path = "models/experimental/deit/deit_cpp/deit_model/deit_classifier_model.pt";

        // Check if model path is provided as command line argument
        if (argc > 1) {
            model_path = argv[1];
        }

        std::cout << "Using model path: " << model_path << std::endl;

        // Test DeiT inference with load_and_preprocess_image function
        test_deit_for_image_classification_inference(model_path);

        std::cout << "DeiT for Image Classification test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
