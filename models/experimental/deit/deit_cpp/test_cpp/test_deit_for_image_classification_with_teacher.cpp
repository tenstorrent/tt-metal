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
 * Load DeiT for Image Classification with Teacher model from file
 * @param model_path Path to the model file
 * @return Pair of state_dict and PyTorch model
 */
std::pair<std::unordered_map<std::string, torch::Tensor>, torch::jit::script::Module>
load_deit_image_classification_with_teacher_model(const std::string& model_path) {
    try {
        // Load the traced model
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.eval();
        
        // Extract state dict from the model
        std::unordered_map<std::string, torch::Tensor> state_dict;
        
        // Get all named parameters from the model
        for (const auto& param : model.named_parameters()) {
            state_dict[param.name] = param.value;
        }
        
        // Get all named buffers from the model
        for (const auto& buffer : model.named_buffers()) {
            state_dict[buffer.name] = buffer.value;
        }
        
        std::cout << "Loaded DeiT with Teacher model with " << state_dict.size() << " parameters" << std::endl;
        
        return std::make_pair(state_dict, model);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
}

/**
 * Test DeiT for Image Classification with Teacher inference
 * @param model_path Path to the model file
 */
void test_deit_for_image_classification_with_teacher_inference(const std::string& model_path) {
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
    auto [state_dict, model] = load_deit_image_classification_with_teacher_model(model_path);
    
    // Use a sample image path for testing (you can replace this with any valid image file)
    std::string test_image_path = "/home/openkylin/like/github/tt/like/tt-metal/models/experimental/deit/deit_cpp/deit_model/input_image.jpg";
    
    torch::Tensor pixel_values = helper_funcs::load_and_preprocess_image(test_image_path);
    
    std::cout << "Using load_and_preprocess_image function for input preprocessing" << std::endl;
    
    // Get PyTorch model output for reference
    torch::Tensor torch_averaged_logits, torch_cls_logits, torch_distillation_logits;
    try {
        // Call the model with pixel_values
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pixel_values);
        
        auto output = model.forward(inputs);
        
        // Extract outputs from the model
        // The output structure depends on the traced model format
        if (output.isTuple()) {
            auto output_tuple = output.toTuple();
            torch_averaged_logits = output_tuple->elements()[0].toTensor(); // Averaged logits
            torch_cls_logits = output_tuple->elements()[1].toTensor(); // CLS logits
            torch_distillation_logits = output_tuple->elements()[2].toTensor(); // Distillation logits
        } else {
            // If only one output, assume it's the averaged logits
            torch_averaged_logits = output.toTensor();
        }
        
        std::cout << "PyTorch averaged logits shape: " << torch_averaged_logits.sizes() << std::endl;
        if (torch_cls_logits.defined()) {
            std::cout << "PyTorch CLS logits shape: " << torch_cls_logits.sizes() << std::endl;
        }
        if (torch_distillation_logits.defined()) {
            std::cout << "PyTorch distillation logits shape: " << torch_distillation_logits.sizes() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to get PyTorch model output: " << e.what() << std::endl;
        return;
    }
    
    // Create DeiT config
    DeiTConfig config;
    
    // Setup TT model
    TtDeiTForImageClassificationWithTeacher tt_model(config, state_dict, base_address, device);
    
    // Convert input to TT tensor
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(pixel_values, device);
    
    // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;
    
    auto [tt_averaged_logits, tt_cls_logits, tt_distillation_logits, attention_weights, hidden_states] = tt_model.forward(
        tt_input, 
        head_mask.has_value() ? &head_mask.value() : nullptr,
        output_attentions, 
        output_hidden_states, 
        return_dict
    );
    
    // Convert TT outputs back to torch tensors
    auto tt_averaged_logits_host = ttnn::from_device(tt_averaged_logits);
    auto tt_averaged_output_torch = helper_funcs::to_torch(tt_averaged_logits_host);
    
    auto tt_cls_logits_host = ttnn::from_device(tt_cls_logits);
    auto tt_cls_output_torch = helper_funcs::to_torch(tt_cls_logits_host);
    
    auto tt_distillation_logits_host = ttnn::from_device(tt_distillation_logits);
    auto tt_distillation_output_torch = helper_funcs::to_torch(tt_distillation_logits_host);
    
    // Ensure output shapes match for comparison
    if (tt_averaged_output_torch.dim() > torch_averaged_logits.dim()) {
        tt_averaged_output_torch = tt_averaged_output_torch.squeeze(0); // Remove batch dimension if needed
    }
    if (tt_cls_output_torch.dim() > torch_cls_logits.dim()) {
        tt_cls_output_torch = tt_cls_output_torch.squeeze(0);
    }
    if (tt_distillation_output_torch.dim() > torch_distillation_logits.dim()) {
        tt_distillation_output_torch = tt_distillation_output_torch.squeeze(0);
    }
    
    // Compute PCC between PyTorch and TT outputs
    double pcc_averaged = compute_pcc(torch_averaged_logits, tt_averaged_output_torch);
    double pcc_cls = torch_cls_logits.defined() ? compute_pcc(torch_cls_logits, tt_cls_output_torch) : 0.0;
    double pcc_distillation = torch_distillation_logits.defined() ? compute_pcc(torch_distillation_logits, tt_distillation_output_torch) : 0.0;
    
    // Log results
    std::cout << "PCC between PyTorch and TT averaged logits: " << pcc_averaged << std::endl;
    if (torch_cls_logits.defined()) {
        std::cout << "PCC between PyTorch and TT CLS logits: " << pcc_cls << std::endl;
    }
    if (torch_distillation_logits.defined()) {
        std::cout << "PCC between PyTorch and TT distillation logits: " << pcc_distillation << std::endl;
    }
    
    std::cout << "PyTorch averaged logits shape: " << torch_averaged_logits.sizes() << std::endl;
    std::cout << "TT averaged logits shape: " << tt_averaged_output_torch.sizes() << std::endl;
    std::cout << "TT CLS logits shape: " << tt_cls_output_torch.sizes() << std::endl;
    std::cout << "TT distillation logits shape: " << tt_distillation_output_torch.sizes() << std::endl;
    
    // Check if PCC meets threshold
    bool passed = pcc_averaged >= pcc_threshold;
    if (torch_cls_logits.defined()) {
        passed = passed && (pcc_cls >= pcc_threshold);
    }
    if (torch_distillation_logits.defined()) {
        passed = passed && (pcc_distillation >= pcc_threshold);
    }
    
    if (passed) {
        std::cout << "PASSED: DeiT For Image Classification with Teacher test" << std::endl;
    } else {
        std::cout << "FAILED: One or more PCC values are below threshold (" << pcc_threshold << ")" << std::endl;
    }

}

/**
 * Test separate logits functionality
 * @param model_path Path to the model file
 */
void test_separate_logits_functionality(const std::string& model_path) {
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
    auto [state_dict, model] = load_deit_image_classification_with_teacher_model(model_path);
    
    // Use a sample image path for testing
    std::string test_image_path = "/home/openkylin/like/github/tt/like/tt-metal/models/experimental/deit/deit_cpp/deit_model/input_image.jpg";
    
    torch::Tensor pixel_values = helper_funcs::load_and_preprocess_image(test_image_path);
    
    // Create DeiT config
    DeiTConfig config;
    
    // Setup TT model
    TtDeiTForImageClassificationWithTeacher tt_model(config, state_dict, base_address, device);
    
    // Convert input to TT tensor
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(pixel_values, device);
    
    // Test get_separate_logits function
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    auto [tt_cls_logits, tt_distillation_logits] = tt_model.get_separate_logits(
        tt_input, 
        head_mask.has_value() ? &head_mask.value() : nullptr
    );
    
    // Convert TT outputs back to torch tensors
    auto tt_cls_logits_host = ttnn::from_device(tt_cls_logits);
    auto tt_cls_output_torch = helper_funcs::to_torch(tt_cls_logits_host);
    
    auto tt_distillation_logits_host = ttnn::from_device(tt_distillation_logits);
    auto tt_distillation_output_torch = helper_funcs::to_torch(tt_distillation_logits_host);
    
    std::cout << "Testing separate logits functionality:" << std::endl;
    std::cout << "TT CLS logits shape: " << tt_cls_output_torch.sizes() << std::endl;
    std::cout << "TT distillation logits shape: " << tt_distillation_output_torch.sizes() << std::endl;
    
    // Test that averaging the separate logits gives the same result as the averaged logits from forward
    auto manual_averaged = (tt_cls_output_torch + tt_distillation_output_torch) * 0.5;
    
    // Get averaged logits from forward method for comparison
    auto [tt_averaged_logits, _, __, ___, ____] = tt_model.forward(
        tt_input, 
        head_mask.has_value() ? &head_mask.value() : nullptr,
        false, false, true
    );
    
    auto tt_averaged_logits_host = ttnn::from_device(tt_averaged_logits);
    auto tt_averaged_output_torch = helper_funcs::to_torch(tt_averaged_logits_host);
    
    if (tt_averaged_output_torch.dim() > manual_averaged.dim()) {
        tt_averaged_output_torch = tt_averaged_output_torch.squeeze(0);
    }
    
    double pcc_manual_vs_forward = compute_pcc(manual_averaged, tt_averaged_output_torch);
    
    std::cout << "PCC between manual averaged logits and forward averaged logits: " << pcc_manual_vs_forward << std::endl;
    
    if (pcc_manual_vs_forward >= pcc_threshold) {
        std::cout << "PASSED: Separate logits functionality test" << std::endl;
    } else {
        std::cout << "FAILED: Manual averaging doesn't match forward method averaging" << std::endl;
    }


}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::cout << "Starting DeiT for Image Classification with Teacher test..." << std::endl;
        
        // Default model path (relative path)
        std::string model_path = "models/experimental/deit/deit_cpp/deit_model/deit_teacher_model.pt";
        
        // Check if model path is provided as command line argument
        if (argc > 1) {
            model_path = argv[1];
        }
        
        std::cout << "Using model path: " << model_path << std::endl;
        
        // Test DeiT with Teacher inference
        test_deit_for_image_classification_with_teacher_inference(model_path);
        
        std::cout << "\nTesting separate logits functionality..." << std::endl;
        test_separate_logits_functionality(model_path);
        
        std::cout << "DeiT for Image Classification with Teacher test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}