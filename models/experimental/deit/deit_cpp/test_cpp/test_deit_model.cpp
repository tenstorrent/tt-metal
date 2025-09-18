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
#include "../tt_cpp/deit_model.h"
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
 * Load DeiT state dict and model from traced model file
 * @param model_path Path to the traced model file
 * @param base_address Base address for parameter extraction
 * @return Pair of state_dict and loaded model
 */
std::pair<std::unordered_map<std::string, torch::Tensor>, torch::jit::script::Module>
load_deit_state_dict_and_model(const std::string& model_path, const std::string& base_address) {
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
 * Create a synthetic image input tensor similar to what would be produced by image processor
 * @return Processed image tensor with shape [1, 3, 224, 224]
 */
torch::Tensor create_synthetic_image_input() {
    // Create a synthetic image tensor with shape [1, 3, 224, 224]
    // This simulates the output of AutoImageProcessor
    auto image_tensor = torch::randn({1, 3, 224, 224}, torch::kFloat32);
    
    // Normalize the image (typical ImageNet normalization)
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
    auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
    
    image_tensor = (image_tensor - mean) / std;
    
    return image_tensor;
}

/**
 * Test DeiT Model inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_model_inference(const std::string& model_path) {
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
    auto [state_dict, model] = load_deit_state_dict_and_model(model_path, base_address);
    
    // Create synthetic image input
    auto input_image = create_synthetic_image_input();
    
    // Get the DeiT model from the loaded model
    torch::jit::script::Module deit_model;
    try {
        // Access the deit submodule from the loaded model
        deit_model = model.attr("model").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get deit module: " << e.what() << std::endl;
        return;
    }
    
    // Prepare inputs for PyTorch model
    torch::Tensor bool_masked_pos; // None equivalent
    torch::Tensor head_mask; // None equivalent
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;
    
    // Call PyTorch DeiT model forward
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_image);
    // For traced model, optional parameters are typically baked in
    
    auto output = deit_model.forward(inputs);
    torch::Tensor torch_output;
    
    // Handle different output formats
    if (output.isTuple()) {
        auto output_tuple = output.toTuple();
        torch_output = output_tuple->elements()[0].toTensor();
    } else {
        torch_output = output.toTensor();
    }
    
    // Create DeiT config
    DeiTConfig config;
    
    // Setup TT model
    auto tt_model = std::make_shared<TtDeiTModel>(config, state_dict, base_address, device, false, false);
    
    // Convert input to TT tensor
    auto tt_image = helper_funcs::torch_to_tt_tensor_tile(input_image, device);
    
    // Run TT model inference
    std::optional<torch::Tensor> bool_masked_pos_opt = std::nullopt;
    const ttnn::Tensor* tt_head_mask = nullptr;
    auto [tt_out, pooler_output, hidden_states, attentions] = tt_model->forward(
        tt_image,
        bool_masked_pos_opt,
        tt_head_mask,
        output_attentions,
        output_hidden_states,
        return_dict
    );
    
    // Convert TT output back to torch tensor
    auto tt_out_host = ttnn::from_device(tt_out);
    auto tt_output_torch = helper_funcs::to_torch(tt_out_host);
    tt_output_torch = tt_output_torch.squeeze(0); // Remove batch dimension
    
    // Compute PCC between PyTorch and TT outputs
    double pcc = compute_pcc(torch_output, tt_output_torch);
    
    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;
    
    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT Model test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Device cleanup is handled automatically by MeshDevice
    device->close();
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Model test..." << std::endl;
    
    // Default model path (relative path)
    std::string model_path = "../deit_model/deit_encoder_model.pt";
    
    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Using model path: " << model_path << std::endl;
    
    try {
        test_deit_model_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}