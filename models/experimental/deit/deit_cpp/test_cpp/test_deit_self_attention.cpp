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
#include <ttnn/distributed/api.hpp>

#include "../tt_cpp/deit_config.h"
#include "../tt_cpp/deit_self_attention.h"
#include "../helper_funcs.h"

namespace {

/**
 * Test DeiT Self Attention inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_self_attention_inference(const std::string& model_path) {
    const double pcc_threshold = 0.99;
    
    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0, 
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    
    // Setup base address
    int layer_index = 0;  // Default to layer 0
    std::string base_address = "model.encoder.layer." + std::to_string(layer_index) + ".attention.attention.";
    
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
            "query.weight", "query.bias",
            "key.weight", "key.bias", 
            "value.weight", "value.bias"
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
    
    // Use PyTorch model's self-attention module directly for reference
    // Model is already loaded from load_deit_state_dict_and_model function
    
    // Get the self-attention module: model.encoder.layer.0.attention.attention
    torch::jit::script::Module self_attention_module;
    try {
        auto model_to_use = model.attr("model").toModule();
        auto encoder = model_to_use.attr("encoder").toModule();
        
        // 从输出可以看到layer不是列表，而是直接的模块结构
        // 尝试直接访问 encoder.layer.0
        auto layer = encoder.attr("layer").toModule();
        auto layer_0 = layer.attr("0").toModule();  // 访问第0层
        auto attention = layer_0.attr("attention").toModule();
        self_attention_module = attention.attr("attention").toModule();
    } catch (const std::exception& e) {
        std::cerr << "Failed to get self-attention module: " << e.what() << std::endl;
        return;
    }

    // Create input tensor: [batch_size=1, seq_len=198, hidden_size=768]
    // DeiT uses 196 patches + 1 CLS token + 1 distillation token = 198 tokens
    torch::Tensor input_tensor = torch::randn({1, 1, 198, 768}, torch::kFloat32);
    // Prepare input for self-attention module
    auto hidden_states = input_tensor.squeeze(0); // Remove batch dimension: [198, 768]
    torch::Tensor torch_head_mask; // None equivalent
    
    // Call self-attention module forward
    // Traced模型的forward方法只接受2个参数：self和hidden_states
    // 可选参数（head_mask, output_attentions）在traced时被优化掉了
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(hidden_states);
    
    auto output = self_attention_module.forward(inputs);
    // Traced模型的forward方法直接返回Tensor，而不是Tuple
    auto torch_output = output.toTensor();

    // Create DeiT config
    DeiTConfig config;
    // Setup TT model
    TtDeiTSelfAttention tt_self_attention(config, device, state_dict, base_address);
    
    // Convert input to TT tensor
    // Note: input_tensor shape is [1, 1, 198, 768], but we need [1, 198, 768] for self-attention
    // Reshape to remove the extra dimension
    auto reshaped_input = input_tensor.squeeze(1); // Remove dimension 1: [1, 198, 768]
    auto tt_input = helper_funcs::torch_to_tt_tensor_tile(reshaped_input, device);
    
    // Input tensor shape after reshape logged
    
    // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    auto [tt_out, attention_weights] = tt_self_attention.forward(tt_input, head_mask, output_attentions);
    
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
        std::cout << "PASSED: DeiT Self Attention test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Clean up device resources
    ttnn::distributed::close_mesh_device(device);
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "Starting DeiT Self Attention test..." << std::endl;
    
    // Default model path
    std::string model_path = "models/experimental/deit/deit_cpp/deit_model/deit_encoder_model.pt";
    
    // Parse command line arguments
    if (argc > 1) {
        model_path = argv[1];
        std::cout << "Using model path from command line: " << model_path << std::endl;
    } else {
        std::cout << "Using default model path: " << model_path << std::endl;
        std::cout << "Usage: " << argv[0] << " [model_path]" << std::endl;
    }
    
    try {
        test_deit_self_attention_inference(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error during test execution: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}