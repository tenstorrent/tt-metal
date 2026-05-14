// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include "ops/matmul.hpp"
#include "ops/eltwise_add.hpp"

namespace vit {

// ViT Patch Embedding:
//   1. Unfold image [3, 224, 224] into patches [196, 768] at rows 1-196 (host-side)
//   2. Linear projection: matmul([224, 768], [768, 192]) -> [224, 192]
//      Row 0 projects to zero since it was zero-padded
//   3. Add projection bias
//   4. Add CLS token at row 0 via eltwise_add with pre-computed CLS buffer
//   5. Add position embedding

struct PatchEmbedWeights {
    std::shared_ptr<distributed::MeshBuffer> proj_weight;   // [768, 192] tilized
    std::shared_ptr<distributed::MeshBuffer> proj_bias;     // [224, 192] broadcast-row tilized
    std::shared_ptr<distributed::MeshBuffer> pos_embed;     // [224, 192] tilized (197 valid + padding)
    std::shared_ptr<distributed::MeshBuffer> cls_buf;       // [224, 192] tilized: CLS at row 0, zeros elsewhere
};

// Unfold image into patch tokens at rows 1-196, row 0 left as zeros for CLS.
// Output: row-major bfloat16 [224, 768]
inline std::vector<bfloat16> unfold_image(
    const std::vector<float>& image,
    uint32_t channels = 3,
    uint32_t img_size = 224,
    uint32_t patch_size = 16) {

    uint32_t num_patches_per_dim = img_size / patch_size;  // 14
    uint32_t patch_dim = channels * patch_size * patch_size;  // 768
    uint32_t seq_padded = round_up_to_tile(num_patches_per_dim * num_patches_per_dim + 1);  // 224
    uint32_t patch_dim_padded = round_up_to_tile(patch_dim);  // 768

    std::vector<bfloat16> patches(seq_padded * patch_dim_padded, bfloat16(0.0f));

    // Row 0: left as zeros (CLS placeholder)
    // Rows 1-196: image patches
    for (uint32_t py = 0; py < num_patches_per_dim; py++) {
        for (uint32_t px = 0; px < num_patches_per_dim; px++) {
            uint32_t patch_idx = py * num_patches_per_dim + px;
            uint32_t row = patch_idx + 1;  // offset by 1 for CLS
            for (uint32_t c = 0; c < channels; c++) {
                for (uint32_t dy = 0; dy < patch_size; dy++) {
                    for (uint32_t dx = 0; dx < patch_size; dx++) {
                        uint32_t img_y = py * patch_size + dy;
                        uint32_t img_x = px * patch_size + dx;
                        uint32_t feat_idx = c * patch_size * patch_size + dy * patch_size + dx;
                        patches[row * patch_dim_padded + feat_idx] =
                            bfloat16(image[c * img_size * img_size + img_y * img_size + img_x]);
                    }
                }
            }
        }
    }
    return patches;
}

// Run patch embedding: unfold + linear projection + bias + CLS add + pos_embed
// No device-to-host round-trips.
inline std::shared_ptr<distributed::MeshBuffer> patch_embed_forward(
    MeshContext& ctx,
    const std::vector<float>& image,
    const PatchEmbedWeights& weights) {

    constexpr uint32_t embed_dim = 192;
    constexpr uint32_t patch_dim = 768;
    constexpr uint32_t seq_padded = 224;
    constexpr uint32_t Mt_seq = seq_padded / TILE_H;     // 7
    constexpr uint32_t Kt_patch = patch_dim / TILE_W;    // 24
    constexpr uint32_t Nt_embed = embed_dim / TILE_W;    // 6
    constexpr uint32_t n_tiles_seq = Mt_seq * Nt_embed;  // 42

    // 1. Unfold image to patches [224, 768] with row 0 = zeros (CLS placeholder)
    auto patches = unfold_image(image);
    auto patches_tiled = tilize_nfaces(patches, seq_padded, patch_dim);

    auto patches_buf = create_dram_buffer(ctx, Mt_seq * Kt_patch * SINGLE_TILE_SIZE);
    write_to_device(ctx, patches_buf, patches_tiled);

    // 2. Linear projection: [224, 768] x [768, 192] -> [224, 192]
    // Row 0 will be all zeros after projection
    auto proj_out_buf = create_dram_buffer(ctx, n_tiles_seq * SINGLE_TILE_SIZE);
    matmul_op(ctx, patches_buf, weights.proj_weight, proj_out_buf, Mt_seq, Kt_patch, Nt_embed);

    // 3. Add projection bias
    auto proj_biased_buf = create_dram_buffer(ctx, n_tiles_seq * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, proj_out_buf, weights.proj_bias, proj_biased_buf, n_tiles_seq);

    // 4. Add CLS token at row 0 via eltwise_add
    // cls_buf has CLS values at row 0, zeros elsewhere.
    // After add: row 0 = 0 + bias + CLS = bias + CLS, rows 1-196 = projected + bias
    auto with_cls_buf = create_dram_buffer(ctx, n_tiles_seq * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, proj_biased_buf, weights.cls_buf, with_cls_buf, n_tiles_seq);

    // 5. Add position embedding
    auto output_buf = create_dram_buffer(ctx, n_tiles_seq * SINGLE_TILE_SIZE);
    eltwise_add_op(ctx, with_cls_buf, weights.pos_embed, output_buf, n_tiles_seq);

    return output_buf;
}

}  // namespace vit
