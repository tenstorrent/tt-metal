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
#include "../tt_cpp/deit_attention.h"
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
 * Test DeiT Attention inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_attention_inference(const std::string& model_path) {
    const double pcc_threshold = 0.99;
    
    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0, 
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    
    // Setup base address
    int layer_index = 0;  // Default to layer 0
    std::string base_address = "model.encoder.layer." + std::to_string(layer_index) + ".attention.";
    
    // Load state dict and model (inlined from load_deit_state_dict_and_model)
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;
    
    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();
        
        std::cout << "Successfully loaded model from: " << model_path << std::endl;
        
        // 加载模型参数到state_dict
        std::vector<std::string> required_params = {
            "attention.query.weight", "attention.query.bias",
            "attention.key.weight", "attention.key.bias", 
            "attention.value.weight", "attention.value.bias",
            "output.dense.weight", "output.dense.bias"
        };
        
        // 使用 named_parameters() 方法直接获取参数
        auto named_params = model.named_parameters();
        std::unordered_map<std::string, at::Tensor> param_map;
        for (const auto& pair : named_params) {
            param_map[pair.name] = pair.value;
        }

        for (const auto& required : required_params) {
            std::string full_key = base_address + required;
            state_dict[full_key] = param_map[full_key];
        }
        
        std::cout << "Loaded " << state_dict.size() << " attention parameters for layer " << layer_index << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
    
    // Use PyTorch model's attention module directly for reference
    // Model is already loaded from load_deit_state_dict_and_model function
    
    // Get the attention module: model.encoder.layer.0.attention
    torch::jit::script::Module attention_module;
    try {
        auto model_to_use = model.attr("model").toModule();
        auto encoder = model_to_use.attr("encoder").toModule();
        
        // 从输出可以看到layer不是列表，而是直接的模块结构
        // 尝试直接访问 encoder.layer.0
        auto layer = encoder.attr("layer").toModule();
        auto layer_0 = layer.attr("0").toModule();  // 访问第0层
        attention_module = layer_0.attr("attention").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get attention module: " << e.what() << std::endl;
        return;
    }


    // Create input tensor: [batch_size=1, seq_len=198, hidden_size=768]
    // DeiT uses 196 patches + 1 CLS token + 1 distillation token = 198 tokens
    torch::Tensor input_tensor = torch::randn({1, 1, 198, 768}, torch::kFloat32);
    // Prepare input for attention module
    auto hidden_states = input_tensor.squeeze(0); // Remove batch dimension: [198, 768]
    torch::Tensor torch_head_mask; // None equivalent
    bool torch_output_attentions = false;
    
    // Call attention module forward
    // Traced模型的forward方法只接受2个参数：self和hidden_states
    // 可选参数（head_mask, output_attentions）在traced时被优化掉了
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(hidden_states);
    
    auto output = attention_module.forward(inputs);
    // Traced模型的forward方法直接返回Tensor，而不是Tuple
    auto torch_output = output.toTensor();

    
    // Create DeiT config
    DeiTConfig config;
    // Setup TT model
    TtDeiTAttention tt_attention(config, device, state_dict, base_address);
    
    // Convert input to TT tensor
    // Note: input_tensor shape is [1, 1, 198, 768], but we need [1, 198, 768] for attention
    // Reshape to remove the extra dimension
    auto reshaped_input = input_tensor.squeeze(1); // Remove dimension 1: [1, 198, 768]
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(reshaped_input, device);
    
    // Input tensor shape after reshape logged
    
    // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    auto [tt_out, attention_weights] = tt_attention.forward(tt_input, head_mask, output_attentions);
    
    // Convert TT output back to torch tensor
    // First move tensor to host if it's on device
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
        std::cout << "PASSED: DeiT Attention test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Device cleanup is handled automatically by MeshDevice
    device->close();
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Attention test..." << std::endl;
    
    // Default model path (relative path)
    std::string model_path = "../deit_model/deit_traced_model.pt";
    
    // Check if model path is provided as command line argument
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Using model path: " << model_path << std::endl;
    
    try {
        test_deit_attention_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}