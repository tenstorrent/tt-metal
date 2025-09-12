// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_SELF_OUTPUT_H
#define DEIT_CPP_TT_CPP_DEIT_SELF_OUTPUT_H

#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "deit_config.h"

/**
 * TtDeiTSelfOutput class - C++ implementation of DeiT self-attention output layer
 * Equivalent to the Python TtDeiTSelfOutput class
 * 
 * Note: The residual connection is defined in DeiTLayer instead of here 
 * (as is the case with other models), due to the layernorm applied before each block.
 */
class TtDeiTSelfOutput {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTSelfOutput(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass
     * @param hidden_states Input hidden states tensor from attention
     * @param input_tensor Original input tensor (for potential residual connection)
     * @return Output tensor after dense transformation
     */
    ttnn::Tensor forward(
        const ttnn::Tensor& hidden_states,
        const ttnn::Tensor& input_tensor
    );

    /**
     * Destructor
     */
    ~TtDeiTSelfOutput() = default;

private:
    // Configuration and device
    DeiTConfig config;
    std::shared_ptr<ttnn::MeshDevice> device;
    
    // Dense layer weights and bias
    ttnn::Tensor dense_weight;
    std::optional<ttnn::Tensor> dense_bias;
};

#endif // DEIT_CPP_TT_CPP_DEIT_SELF_OUTPUT_H