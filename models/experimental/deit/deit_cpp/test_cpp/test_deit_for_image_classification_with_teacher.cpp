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
#include "../tt_cpp/deit_for_image_classification_with_teacher.h"
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
 * Create sample image input tensor
 * Simulates the output of AutoImageProcessor for DeiT
 * @return Sample image tensor [1, 3, 224, 224]
 */
torch::Tensor create_sample_image_input() {
    // Create a sample image tensor with shape [1, 3, 224, 224]
    // This simulates preprocessed image data from AutoImageProcessor
    return torch::randn({1, 3, 224, 224}, torch::kFloat32);
}

/**
 * Load DeiT state dict and model for image classification with teacher
 * @param model_path Path to the traced model
 * @return Pair of state_dict and loaded model
 */
std::pair<std::unordered_map<std::string, torch::Tensor>, torch::jit::script::Module>
load_deit_image_classification_with_teacher_model(const std::string& model_path) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;
    
    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();
        
        std::cout << "Successfully loaded model from: " << model_path << std::endl;
        
        // Load model parameters to state_dict
        std::vector<std::string> required_params = {
            "cls_classifier.weight", "cls_classifier.bias",
            "distillation_classifier.weight", "distillation_classifier.bias"
        };
        
        // Use named_parameters() method to get parameters
        auto named_params = model.named_parameters();
        std::unordered_map<std::string, at::Tensor> param_map;
        for (const auto& pair : named_params) {
            param_map[pair.name] = pair.value;
        }

        for (const auto& required : required_params) {
            if (param_map.find(required) != param_map.end()) {
                state_dict[required] = param_map[required];
            }
        }
        
        std::cout << "Loaded " << state_dict.size() << " image classification with teacher parameters" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
    
    return std::make_pair(state_dict, model);
}

/**
 * Test DeiT For Image Classification With Teacher inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_for_image_classification_with_teacher_inference(const std::string& model_path) {
    const double pcc_threshold = 0.95;
    
    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0, 
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    

    
    // Load state dict and model
    auto [state_dict, model] = load_deit_image_classification_with_teacher_model(model_path);
    
    // Create sample image input
    torch::Tensor pixel_values = create_sample_image_input();
    
    // Get PyTorch model output for reference
    torch::Tensor torch_output;
    try {
        // Call the model with pixel_values
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pixel_values);
        
        auto output = model.forward(inputs);
        
        // Extract logits from the output
        // For DeiTForImageClassificationWithTeacher, the output should be the averaged logits
        if (output.isTuple()) {
            auto output_tuple = output.toTuple();
            torch_output = output_tuple->elements()[0].toTensor(); // First element should be averaged logits
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
    TtDeiTForImageClassificationWithTeacher tt_model(config, state_dict, "", device);
    
    // Convert input to TT tensor
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(pixel_values, device);
    
    // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;
    
    auto [averaged_logits, cls_logits, distillation_logits, attention_weights, hidden_states] = tt_model.forward(
        tt_input, 
        head_mask.has_value() ? &head_mask.value() : nullptr,
        output_attentions, 
        output_hidden_states, 
        return_dict
    );
    
    // Convert TT output back to torch tensor
    auto tt_averaged_logits_host = ttnn::from_device(averaged_logits);
    auto tt_output_torch = helper_funcs::to_torch(tt_averaged_logits_host);
    
    // Ensure output shapes match for comparison
    if (tt_output_torch.dim() > torch_output.dim()) {
        tt_output_torch = tt_output_torch.squeeze(0); // Remove batch dimension if needed
    }
    
    // Extract the classification logits (typically [:, 0, :] for DeiT)
    if (tt_output_torch.dim() == 3 && tt_output_torch.size(1) > 1) {
        tt_output_torch = tt_output_torch.select(1, 0); // Select first token (CLS token)
    }
    
    // Compute PCC between PyTorch and TT outputs
    double pcc = compute_pcc(torch_output, tt_output_torch);
    
    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;
    
    // Also test separate logits functionality
    std::cout << "\nTesting separate logits functionality..." << std::endl;
    auto [separate_cls_logits, separate_dist_logits] = tt_model.get_separate_logits(
        tt_input, 
        head_mask.has_value() ? &head_mask.value() : nullptr
    );
    
    // Convert separate logits to torch tensors for inspection
    auto cls_logits_host = ttnn::from_device(separate_cls_logits);
    auto dist_logits_host = ttnn::from_device(separate_dist_logits);
    auto cls_logits_torch = helper_funcs::to_torch(cls_logits_host);
    auto dist_logits_torch = helper_funcs::to_torch(dist_logits_host);
    
    std::cout << "CLS logits shape: " << cls_logits_torch.sizes() << std::endl;
    std::cout << "Distillation logits shape: " << dist_logits_torch.sizes() << std::endl;
    
    // Verify that averaged logits are approximately the average of separate logits
    auto manual_average = (cls_logits_torch + dist_logits_torch) / 2.0;
    double average_pcc = compute_pcc(tt_output_torch, manual_average.squeeze());
    std::cout << "PCC between averaged logits and manual average: " << average_pcc << std::endl;
    
    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT For Image Classification With Teacher test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }
    
    // Additional check for averaging functionality
    if (average_pcc >= 0.99) {
        std::cout << "PASSED: Averaging functionality works correctly" << std::endl;
    } else {
        std::cout << "WARNING: Averaging functionality may have issues (PCC = " << average_pcc << ")" << std::endl;
    }

    // Device cleanup
    device->close();
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT For Image Classification With Teacher test..." << std::endl;
    
    // Default model path (relative path)
    std::string model_path = "../deit_model/deit_teacher_model.pt";
    
    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Using model path: " << model_path << std::endl;
    
    try {
        test_deit_for_image_classification_with_teacher_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}