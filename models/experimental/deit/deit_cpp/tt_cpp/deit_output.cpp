#include "deit_output.h"
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <stdexcept>

namespace deit_cpp {

TtDeiTOutput::TtDeiTOutput(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device) {
    
    // Load dense layer weights and bias from state_dict
    std::string weight_key = base_address + ".dense.weight";
    std::string bias_key = base_address + ".dense.bias";
    
    auto weight_it = state_dict.find(weight_key);
    auto bias_it = state_dict.find(bias_key);
    
    if (weight_it == state_dict.end()) {
        throw std::runtime_error("Dense weight not found in state_dict: " + weight_key);
    }
    
    // Convert torch tensors to ttnn tensors
    dense_weight = helper_funcs::torch_to_tt_tensor_tile(weight_it->second, device);
    dense_bias = (bias_it != state_dict.end()) ? std::optional<ttnn::Tensor>(helper_funcs::torch_to_tt_tensor_tile(bias_it->second, device)) : std::nullopt;
}

ttnn::Tensor TtDeiTOutput::forward(const ttnn::Tensor& hidden_states, const ttnn::Tensor& input_tensor) {
    // Apply dense linear transformation: weight * input + bias
    auto output = helper_funcs::linear_transform(
        hidden_states,
        dense_weight,
        dense_bias
    );
    
    // Add residual connection: output + input_tensor
    output = ttnn::add(output, input_tensor);
    
    return output;
}

} // namespace deit_cpp