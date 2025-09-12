// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_ATTENTION_H
#define DEIT_CPP_TT_CPP_DEIT_ATTENTION_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "deit_config.h"
#include "deit_self_attention.h"
#include "deit_self_output.h"

/**
 * TtDeiTAttention class - C++ implementation of DeiT attention mechanism
 * Equivalent to the Python TtDeiTAttention class
 */
class TtDeiTAttention {
public:
    /**
     * Constructor
     * @param config DeiT configuration
     * @param device TTNN device pointer
     * @param state_dict Model parameters dictionary
     * @param base_address Base address for parameter loading
     */
    TtDeiTAttention(
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
     * @return Tuple containing attention output and optionally attention weights
     */
    std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> forward(
        const ttnn::Tensor& hidden_states,
        const std::optional<ttnn::Tensor>& head_mask = std::nullopt,
        bool output_attentions = false
    );

    /**
     * Destructor
     */
    ~TtDeiTAttention() = default;

private:
    std::unique_ptr<TtDeiTSelfAttention> attention;
    std::unique_ptr<TtDeiTSelfOutput> output;
    std::shared_ptr<ttnn::MeshDevice> device;
    DeiTConfig config;
};

#endif // DEIT_CPP_TT_CPP_DEIT_ATTENTION_H