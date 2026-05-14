// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include "ops/matmul.hpp"
#include "ops/eltwise_add.hpp"
#include "ops/gelu.hpp"

namespace vit {

// MLP for ViT Tiny:
//   FC1: [224, 192] x [192, 768] -> [224, 768]
//   GELU activation
//   FC2: [224, 768] x [768, 192] -> [224, 192]

struct MLPWeights {
    std::shared_ptr<distributed::MeshBuffer> fc1_weight;  // [192, 768] tilized
    std::shared_ptr<distributed::MeshBuffer> fc1_bias;    // [224, 768] broadcast-row tilized
    std::shared_ptr<distributed::MeshBuffer> fc2_weight;  // [768, 192] tilized
    std::shared_ptr<distributed::MeshBuffer> fc2_bias;    // [224, 192] broadcast-row tilized
};

inline std::shared_ptr<distributed::MeshBuffer> mlp_forward(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& input,  // [224, 192] tilized
    const MLPWeights& weights) {

    constexpr uint32_t seq_padded = 224;
    constexpr uint32_t embed_dim = 192;
    constexpr uint32_t mlp_dim = 768;

    constexpr uint32_t Mt = seq_padded / TILE_H;   // 7
    constexpr uint32_t Et = embed_dim / TILE_W;    // 6
    constexpr uint32_t MLPt = mlp_dim / TILE_W;    // 24

    // FC1: [224, 192] x [192, 768] -> [224, 768]
    auto fc1_out = create_dram_buffer(ctx, Mt * MLPt * SINGLE_TILE_SIZE);
    matmul_op(ctx, input, weights.fc1_weight, fc1_out, Mt, Et, MLPt);

    // Add FC1 bias
    auto fc1_biased = create_dram_buffer(ctx, Mt * MLPt * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, fc1_out, weights.fc1_bias, fc1_biased, Mt * MLPt);

    // GELU
    auto gelu_out = create_dram_buffer(ctx, Mt * MLPt * SINGLE_TILE_SIZE);
    gelu_op(ctx, fc1_biased, gelu_out, Mt * MLPt);

    // FC2: [224, 768] x [768, 192] -> [224, 192]
    auto fc2_out = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    matmul_op(ctx, gelu_out, weights.fc2_weight, fc2_out, Mt, MLPt, Et);

    // Add FC2 bias
    auto output = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, fc2_out, weights.fc2_bias, output, Mt * Et);

    return output;
}

}  // namespace vit
