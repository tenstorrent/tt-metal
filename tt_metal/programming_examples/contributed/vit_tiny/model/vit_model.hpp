// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include "ops/matmul.hpp"
#include "ops/eltwise_add.hpp"
#include "ops/layernorm.hpp"
#include "model/patch_embed.hpp"
#include "model/transformer_block.hpp"

namespace vit {

// Full ViT Tiny model:
//   1. Patch embedding (unfold + CLS + projection + pos_embed)
//   2. 12x Transformer blocks
//   3. Final LayerNorm
//   4. CLS token extraction (row 0) — only host round-trip
//   5. Classification head: [1, 192] x [192, 1024] -> [1, 1024]

struct ViTWeights {
    PatchEmbedWeights patch_embed;
    std::vector<TransformerBlockWeights> blocks;  // 12 blocks
    std::shared_ptr<distributed::MeshBuffer> final_ln_gamma;
    std::shared_ptr<distributed::MeshBuffer> final_ln_beta;
    std::shared_ptr<distributed::MeshBuffer> head_weight;  // [192, 1024] tilized
    std::shared_ptr<distributed::MeshBuffer> head_bias;    // [32, 1024] broadcast-row tilized
};

inline std::vector<float> vit_forward(
    MeshContext& ctx,
    const std::vector<float>& image,  // [3, 224, 224] CHW normalized
    const ViTWeights& weights) {

    constexpr uint32_t seq_padded = 224;
    constexpr uint32_t embed_dim = 192;
    constexpr uint32_t num_classes_padded = 1024;
    constexpr uint32_t Mt = seq_padded / TILE_H;
    constexpr uint32_t Et = embed_dim / TILE_W;
    constexpr uint32_t St = seq_padded / TILE_W;  // 7
    constexpr uint32_t Ct = num_classes_padded / TILE_W;  // 32
    constexpr uint32_t n_tiles = Mt * Et;
    constexpr float eps = 1e-6f;

    // Create scale buffer for attention: [224, 224] = Mt*St = 49 tiles, filled with 0.125
    auto scale_buf = create_dram_buffer(ctx, Mt * St * SINGLE_TILE_SIZE);
    {
        std::vector<bfloat16> scale_data(seq_padded * seq_padded, bfloat16(0.125f));
        auto scale_tiled = tilize_nfaces(scale_data, seq_padded, seq_padded);
        write_to_device(ctx, scale_buf, scale_tiled);
    }

    // 1. Patch embedding
    fmt::print("  Patch embedding...\n");
    auto x = patch_embed_forward(ctx, image, weights.patch_embed);

    // 2. Transformer blocks
    for (uint32_t i = 0; i < 12; i++) {
        fmt::print("  Transformer block {}...\n", i);
        x = transformer_block_forward(ctx, x, weights.blocks[i], scale_buf);
    }

    // 3. Final LayerNorm
    fmt::print("  Final LayerNorm...\n");
    auto ln_out = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    layernorm_op(ctx, x, weights.final_ln_gamma, weights.final_ln_beta, ln_out, Mt, Et, eps);

    // 4. Extract CLS token (row 0) -> [1, 192]
    // This is the only host round-trip (final output extraction)
    std::vector<bfloat16> final_data(seq_padded * embed_dim);
    read_from_device(ctx, final_data, ln_out);
    final_data = untilize_nfaces(final_data, seq_padded, embed_dim);

    std::vector<bfloat16> cls_feat(TILE_H * embed_dim, bfloat16(0.0f));
    for (uint32_t j = 0; j < embed_dim; j++) {
        cls_feat[j] = final_data[j];
    }
    auto cls_tiled = tilize_nfaces(cls_feat, TILE_H, embed_dim);
    auto cls_buf = create_dram_buffer(ctx, 1 * Et * SINGLE_TILE_SIZE);
    write_to_device(ctx, cls_buf, cls_tiled);

    // 5. Classification head: [32, 192] x [192, 1024] -> [32, 1024]
    fmt::print("  Classification head...\n");
    auto logits_buf = create_dram_buffer(ctx, 1 * Ct * SINGLE_TILE_SIZE);
    matmul_op(ctx, cls_buf, weights.head_weight, logits_buf, 1, Et, Ct);

    // Add head bias
    auto logits_biased = create_dram_buffer(ctx, 1 * Ct * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, logits_buf, weights.head_bias, logits_biased, 1 * Ct);

    // Read logits
    std::vector<bfloat16> logits_data(TILE_H * num_classes_padded);
    read_from_device(ctx, logits_data, logits_biased);
    logits_data = untilize_nfaces(logits_data, TILE_H, num_classes_padded);

    std::vector<float> logits(1000);
    for (uint32_t j = 0; j < 1000; j++) {
        logits[j] = static_cast<float>(logits_data[j]);
    }

    return logits;
}

}  // namespace vit
