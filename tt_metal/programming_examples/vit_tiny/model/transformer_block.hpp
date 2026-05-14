// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include "ops/layernorm.hpp"
#include "ops/eltwise_add.hpp"
#include "model/attention.hpp"
#include "model/mlp.hpp"

namespace vit {

// Transformer Block:
//   x_norm = layernorm(x)
//   attn_out = self_attention(x_norm)
//   x = x + attn_out              (residual)
//   x_norm = layernorm(x)
//   mlp_out = mlp(x_norm)
//   x = x + mlp_out               (residual)

struct TransformerBlockWeights {
    // LayerNorm 1
    std::shared_ptr<distributed::MeshBuffer> ln1_gamma;  // [Wt tiles] broadcast-row
    std::shared_ptr<distributed::MeshBuffer> ln1_beta;   // [Wt tiles] broadcast-row

    // Attention
    AttentionWeights attn;

    // LayerNorm 2
    std::shared_ptr<distributed::MeshBuffer> ln2_gamma;
    std::shared_ptr<distributed::MeshBuffer> ln2_beta;

    // MLP
    MLPWeights mlp;
};

inline std::shared_ptr<distributed::MeshBuffer> transformer_block_forward(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& input,  // [224, 192] tilized
    const TransformerBlockWeights& weights,
    const std::shared_ptr<distributed::MeshBuffer>& scale_buf) {

    constexpr uint32_t seq_padded = 224;
    constexpr uint32_t embed_dim = 192;
    constexpr uint32_t Mt = seq_padded / TILE_H;   // 7
    constexpr uint32_t Et = embed_dim / TILE_W;    // 6
    constexpr uint32_t n_tiles = Mt * Et;          // 42
    constexpr float eps = 1e-6f;

    // 1. LayerNorm 1
    auto ln1_out = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    layernorm_op(ctx, input, weights.ln1_gamma, weights.ln1_beta, ln1_out, Mt, Et, eps);

    // 2. Self-Attention
    auto attn_out = attention_forward(ctx, ln1_out, weights.attn, scale_buf);

    // 3. Residual: x = x + attn_out
    auto residual1 = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, input, attn_out, residual1, n_tiles);

    // 4. LayerNorm 2
    auto ln2_out = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    layernorm_op(ctx, residual1, weights.ln2_gamma, weights.ln2_beta, ln2_out, Mt, Et, eps);

    // 5. MLP
    auto mlp_out = mlp_forward(ctx, ln2_out, weights.mlp);

    // 6. Residual: x = x + mlp_out
    auto output = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, residual1, mlp_out, output, n_tiles);

    return output;
}

}  // namespace vit
