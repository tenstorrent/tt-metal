// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include "ops/matmul.hpp"
#include "ops/eltwise_add.hpp"
#include "ops/eltwise_mul.hpp"
#include "ops/softmax.hpp"
#include "ops/column_slice.hpp"
#include "ops/column_write.hpp"
#include "ops/transpose.hpp"

namespace vit {

// Multi-Head Self-Attention for ViT Tiny (all on device, no host round-trips):
//   embed_dim=192, num_heads=3, head_dim=64
//   1. QKV projection: matmul + bias add
//   2. Column-slice Q, K, V from QKV
//   3. Per-head: slice Q_h, K_h, V_h -> transpose K_h -> matmul -> scale -> softmax -> matmul
//   4. Column-write to concat heads
//   5. Output projection: matmul + bias add

struct AttentionWeights {
    std::shared_ptr<distributed::MeshBuffer> qkv_weight;   // [192, 576] tilized
    std::shared_ptr<distributed::MeshBuffer> qkv_bias;     // [224, 576] broadcast-row tilized
    std::shared_ptr<distributed::MeshBuffer> proj_weight;   // [192, 192] tilized
    std::shared_ptr<distributed::MeshBuffer> proj_bias;     // [224, 192] broadcast-row tilized
};

inline std::shared_ptr<distributed::MeshBuffer> attention_forward(
    MeshContext& ctx,
    const std::shared_ptr<distributed::MeshBuffer>& input,      // [224, 192] tilized
    const AttentionWeights& weights,
    const std::shared_ptr<distributed::MeshBuffer>& scale_buf)   // [224, 224] filled with 0.125
{
    constexpr uint32_t seq_padded = 224;
    constexpr uint32_t embed_dim = 192;
    constexpr uint32_t qkv_dim = 576;  // 3 * 192
    constexpr uint32_t head_dim = 64;
    constexpr uint32_t num_heads = 3;

    constexpr uint32_t Mt = seq_padded / TILE_H;   // 7
    constexpr uint32_t Et = embed_dim / TILE_W;    // 6
    constexpr uint32_t QKVt = qkv_dim / TILE_W;   // 18
    constexpr uint32_t Ht = head_dim / TILE_W;     // 2
    constexpr uint32_t St = seq_padded / TILE_W;   // 7

    // 1. QKV projection: [224, 192] x [192, 576] -> [224, 576]
    auto qkv_buf = create_dram_buffer(ctx, Mt * QKVt * SINGLE_TILE_SIZE);
    matmul_op(ctx, input, weights.qkv_weight, qkv_buf, Mt, Et, QKVt);

    // Add QKV bias
    auto qkv_biased = create_dram_buffer(ctx, Mt * QKVt * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, qkv_buf, weights.qkv_bias, qkv_biased, Mt * QKVt);

    // 2. Split Q, K, V via column slicing (all on device)
    auto q_buf = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    auto k_buf = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    auto v_buf = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    column_slice_op(ctx, qkv_biased, q_buf, Mt, QKVt, 0, Et);       // Q: cols 0..5
    column_slice_op(ctx, qkv_biased, k_buf, Mt, QKVt, Et, Et);      // K: cols 6..11
    column_slice_op(ctx, qkv_biased, v_buf, Mt, QKVt, 2 * Et, Et);  // V: cols 12..17

    // 3. Per-head attention (all on device)
    auto concat_buf = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);

    for (uint32_t h = 0; h < num_heads; h++) {
        // Extract per-head Q_h, K_h, V_h via column slicing
        auto q_h = create_dram_buffer(ctx, Mt * Ht * SINGLE_TILE_SIZE);
        auto k_h = create_dram_buffer(ctx, Mt * Ht * SINGLE_TILE_SIZE);
        auto v_h = create_dram_buffer(ctx, Mt * Ht * SINGLE_TILE_SIZE);
        column_slice_op(ctx, q_buf, q_h, Mt, Et, h * Ht, Ht);
        column_slice_op(ctx, k_buf, k_h, Mt, Et, h * Ht, Ht);
        column_slice_op(ctx, v_buf, v_h, Mt, Et, h * Ht, Ht);

        // Transpose K_h: [Mt=7, Ht=2] -> [Ht=2, Mt=7]
        auto k_h_t = create_dram_buffer(ctx, Ht * Mt * SINGLE_TILE_SIZE);
        transpose_2d_op(ctx, k_h, k_h_t, Mt, Ht);

        // Q_h @ K_h^T: [224, 64] x [64, 224] -> [224, 224]
        auto scores = create_dram_buffer(ctx, Mt * St * SINGLE_TILE_SIZE);
        matmul_op(ctx, q_h, k_h_t, scores, Mt, Ht, St);

        // Scale by 1/sqrt(head_dim) = 0.125 (eltwise mul on device)
        auto scores_scaled = create_dram_buffer(ctx, Mt * St * SINGLE_TILE_SIZE);
        eltwise_mul_op(ctx, scores, scale_buf, scores_scaled, Mt * St);

        // Softmax
        auto attn_weights = create_dram_buffer(ctx, Mt * St * SINGLE_TILE_SIZE);
        softmax_op(ctx, scores_scaled, attn_weights, Mt, St);

        // Attention @ V_h: [224, 224] x [224, 64] -> [224, 64]
        auto head_out = create_dram_buffer(ctx, Mt * Ht * SINGLE_TILE_SIZE);
        matmul_op(ctx, attn_weights, v_h, head_out, Mt, St, Ht);

        // Write head output to correct columns in concat buffer
        column_write_op(ctx, head_out, concat_buf, Mt, Et, h * Ht, Ht);
    }

    // 4. Output projection: [224, 192] x [192, 192] -> [224, 192]
    auto proj_out = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    matmul_op(ctx, concat_buf, weights.proj_weight, proj_out, Mt, Et, Et);

    // Add projection bias
    auto output = create_dram_buffer(ctx, Mt * Et * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, proj_out, weights.proj_bias, output, Mt * Et);

    return output;
}

}  // namespace vit
