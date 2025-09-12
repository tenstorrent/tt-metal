// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_POOLER_H
#define DEIT_CPP_TT_CPP_DEIT_POOLER_H

#include "deit_config.h"
#include "helper_funcs.h"
#include <memory>
#include <torch/torch.h>
#include <unordered_map>
#include <string>

/**
 * TtDeiTPooler class - C++ implementation of DeiT pooler
 * Equivalent to the Python TtDeiTPooler class
 * 
 * This class performs pooling operation on the hidden states by:
 * 1. Taking the first token (CLS token) from the sequence
 * 2. Applying a linear transformation
 * 3. Applying tanh activation
 */
class TtDeiTPooler {
public:
    /**
     * Constructor for TtDeiTPooler
     * 
     * @param config DeiT configuration containing model parameters
     * @param device Mesh device for tensor operations
     * @param state_dict State dictionary containing pre-trained weights
     * @param base_address Base address for accessing weights in state_dict
     */
    TtDeiTPooler(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass of the pooler
     * 
     * @param hidden_states Input hidden states tensor [batch_size, seq_len, hidden_size]
     * @return Pooled output tensor [batch_size, hidden_size]
     */
    ttnn::Tensor forward(const ttnn::Tensor& hidden_states);

private:
    // Configuration
    DeiTConfig config_;
    std::shared_ptr<ttnn::MeshDevice> device_;
    
    // Linear layer weights and bias
    ttnn::Tensor dense_weight_;
    ttnn::Tensor dense_bias_;
};

#endif // DEIT_CPP_TT_CPP_DEIT_POOLER_H