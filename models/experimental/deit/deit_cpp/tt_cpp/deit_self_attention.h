// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_SELF_ATTENTION_H
#define DEIT_CPP_TT_CPP_DEIT_SELF_ATTENTION_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "deit_config.h"

/**
 * TtDeiTSelfAttention class - C++ implementation of DeiT self-attention mechanism
 * Equivalent to the Python TtDeiTSelfAttention class
 */
class TtDeiTSelfAttention {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTSelfAttention(
        const DeiTConfig& config,
        std::shared_ptr<ttnn::MeshDevice> device,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& base_address = ""
    );

    /**
     * Forward pass
     * @param hidden_states Input hidden states tensor
     * @param head_mask Optional attention head mask
     * @param output_attentions Whether to output attention weights
     * @return Tuple containing context layer output and optionally attention probabilities
     */
    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> forward(
        const ttnn::Tensor& hidden_states,
        const std::optional<ttnn::Tensor>& head_mask = std::nullopt,
        bool output_attentions = false
    );

    /**
     * Destructor
     */
    ~TtDeiTSelfAttention() = default;

private:
    /**
     * Transpose tensor for attention score computation
     * @param x Input tensor
     * @return Transposed tensor
     */
    ttnn::Tensor transpose_for_scores(ttnn::Tensor& x);

    // Configuration and device
    DeiTConfig config;
    std::shared_ptr<ttnn::MeshDevice> device;
    
    // Attention parameters
    int num_attention_heads;
    int attention_head_size;
    int all_head_size;
    
    // Linear layer weights and biases
    ttnn::Tensor query_weight;
    std::optional<ttnn::Tensor> query_bias;
    ttnn::Tensor key_weight;
    std::optional<ttnn::Tensor> key_bias;
    ttnn::Tensor value_weight;
    std::optional<ttnn::Tensor> value_bias;
};

#endif // DEIT_CPP_TT_CPP_DEIT_SELF_ATTENTION_H