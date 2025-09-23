// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef DEIT_CPP_TT_CPP_DEIT_CONFIG_H
#define DEIT_CPP_TT_CPP_DEIT_CONFIG_H

#include <string>
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
// #include "ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include <torch/torch.h>

/**
 * DeiTConfig struct - C++ implementation of DeiT configuration
 * Equivalent to the Python DeiTConfig class
 */
struct DeiTConfig {
    // Model architecture parameters
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int intermediate_size = 3072;
    std::string hidden_act = "gelu";

    // Dropout parameters
    float hidden_dropout_prob = 0.0f;
    float attention_probs_dropout_prob = 0.0f;

    // Initialization parameters
    float initializer_range = 0.02f;
    float layer_norm_eps = 1e-12f;

    // Image processing parameters
    int image_size = 224;
    int patch_size = 16;
    int num_channels = 3;

    // Attention parameters
    bool qkv_bias = true;

    // Encoder parameters
    int encoder_stride = 16;

    /**
     * Default constructor
     */
    DeiTConfig() = default;

    /**
     * Parameterized constructor
     */
    DeiTConfig(
        int hidden_size,
        int num_hidden_layers,
        int num_attention_heads,
        int intermediate_size,
        const std::string& hidden_act = "gelu",
        float hidden_dropout_prob = 0.0f,
        float attention_probs_dropout_prob = 0.0f,
        float initializer_range = 0.02f,
        float layer_norm_eps = 1e-12f,
        int image_size = 224,
        int patch_size = 16,
        int num_channels = 3,
        bool qkv_bias = true,
        int encoder_stride = 16
    ) : hidden_size(hidden_size),
        num_hidden_layers(num_hidden_layers),
        num_attention_heads(num_attention_heads),
        intermediate_size(intermediate_size),
        hidden_act(hidden_act),
        hidden_dropout_prob(hidden_dropout_prob),
        attention_probs_dropout_prob(attention_probs_dropout_prob),
        initializer_range(initializer_range),
        layer_norm_eps(layer_norm_eps),
        image_size(image_size),
        patch_size(patch_size),
        num_channels(num_channels),
        qkv_bias(qkv_bias),
        encoder_stride(encoder_stride) {}
};

#endif // DEIT_CPP_TT_CPP_DEIT_CONFIG_H
