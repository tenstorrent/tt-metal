// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_self_output.h"
#include "helper_funcs.h"
#include <stdexcept>
#include <optional>
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

TtDeiTSelfOutput::TtDeiTSelfOutput(
    const DeiTConfig& config,
    std::shared_ptr<ttnn::MeshDevice> device,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& base_address
) : config(config), device(device) {
    if (!device) {
        throw std::invalid_argument("Device cannot be null");
    }

    // Load dense layer weights and bias from state_dict
    std::string dense_weight_key = base_address + "dense.weight";
    std::string dense_bias_key = base_address + "dense.bias";

    auto dense_weight_it = state_dict.find(dense_weight_key);
    auto dense_bias_it = state_dict.find(dense_bias_key);

    if (dense_weight_it == state_dict.end()) {
        throw std::runtime_error("Required dense weight parameter not found in state_dict for base_address: " + base_address);
    }

    dense_weight = helper_funcs::torch_to_tt_tensor_tile(dense_weight_it->second, device);
    dense_bias = (dense_bias_it != state_dict.end()) ? std::optional<ttnn::Tensor>(helper_funcs::torch_to_tt_tensor_tile(dense_bias_it->second, device)) : std::nullopt;
}

auto TtDeiTSelfOutput::forward(
    const ttnn::Tensor& hidden_states,
    const std::optional<ttnn::Tensor>& input_tensor
) -> ttnn::Tensor {
    // Apply dense linear transformation using matmul and add
    auto output = helper_funcs::linear_transform(hidden_states, dense_weight, dense_bias);
    // output = ttnn::add(output, input_tensor);
    
    // Note: In the original Python implementation, the residual connection
    // is handled in the DeiTLayer, not here. So we just return the dense output.
    // The input_tensor parameter is kept for compatibility but not used here.
    
    return output;
}