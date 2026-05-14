// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ops/common.hpp"
#include "model/vit_model.hpp"

#include <fstream>
#include <filesystem>

namespace vit {

// Load a raw binary file of float32 values
inline std::vector<float> load_binary_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        TT_THROW("Failed to open weight file: {}", path);
    }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// Convert float32 vector to bfloat16
inline std::vector<bfloat16> to_bf16(const std::vector<float>& data) {
    std::vector<bfloat16> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i] = bfloat16(data[i]);
    }
    return result;
}

// Load a 2D weight matrix [rows, cols] and upload to DRAM as tilized bfloat16
// Pads to tile-aligned dimensions
inline std::shared_ptr<distributed::MeshBuffer> load_weight_2d(
    MeshContext& ctx, const std::string& path,
    uint32_t rows, uint32_t cols) {

    auto data_f32 = load_binary_f32(path);
    uint32_t rows_padded = round_up_to_tile(rows);
    uint32_t cols_padded = round_up_to_tile(cols);

    std::vector<bfloat16> padded(rows_padded * cols_padded, bfloat16(0.0f));
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            padded[i * cols_padded + j] = bfloat16(data_f32[i * cols + j]);
        }
    }

    auto tiled = tilize_nfaces(padded, rows_padded, cols_padded);
    uint32_t n_tiles = num_tiles(rows) * num_tiles(cols);
    auto buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    write_to_device(ctx, buf, tiled);
    return buf;
}

// Load a 1D bias vector [cols] and create broadcast-row tiles [rows_padded, cols_padded]
// Each tile row has the same values (broadcast-row format).
inline std::shared_ptr<distributed::MeshBuffer> load_bias_broadcast_row(
    MeshContext& ctx, const std::string& path,
    uint32_t cols, uint32_t rows_padded) {

    auto data_f32 = load_binary_f32(path);
    uint32_t cols_padded = round_up_to_tile(cols);

    std::vector<bfloat16> padded(rows_padded * cols_padded, bfloat16(0.0f));
    for (uint32_t i = 0; i < rows_padded; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            padded[i * cols_padded + j] = bfloat16(data_f32[j]);
        }
    }

    auto tiled = tilize_nfaces(padded, rows_padded, cols_padded);
    uint32_t n_tiles = num_tiles(rows_padded) * num_tiles(cols);
    auto buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    write_to_device(ctx, buf, tiled);
    return buf;
}

// Load a 1D vector [cols] as a single tile-row (broadcast-row format for LayerNorm gamma/beta)
// Only Wt tiles are stored, not the full [Mt, Wt] grid.
inline std::shared_ptr<distributed::MeshBuffer> load_ln_param(
    MeshContext& ctx, const std::string& path, uint32_t cols) {

    auto data_f32 = load_binary_f32(path);
    uint32_t cols_padded = round_up_to_tile(cols);
    uint32_t Wt = num_tiles(cols);

    // Create broadcast-row tile(s): [32, cols_padded]
    std::vector<bfloat16> padded(TILE_H * cols_padded, bfloat16(0.0f));
    for (uint32_t r = 0; r < TILE_H; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            padded[r * cols_padded + c] = bfloat16(data_f32[c]);
        }
    }

    auto tiled = tilize_nfaces(padded, TILE_H, cols_padded);
    auto buf = create_dram_buffer(ctx, Wt * SINGLE_TILE_SIZE);
    write_to_device(ctx, buf, tiled);
    return buf;
}

// Load all ViT Tiny weights from a directory
inline ViTWeights load_vit_weights(MeshContext& ctx, const std::string& weight_dir) {
    ViTWeights weights;
    std::string d = weight_dir;

    fmt::print("Loading weights from {}...\n", d);

    // Patch embedding
    weights.patch_embed.proj_weight = load_weight_2d(ctx, d + "/patch_embed.proj.weight.bin", 768, 192);
    weights.patch_embed.proj_bias = load_bias_broadcast_row(ctx, d + "/patch_embed.proj.bias.bin", 192, 224);
    weights.patch_embed.pos_embed = load_weight_2d(ctx, d + "/pos_embed.bin", 224, 192);
    // CLS token: create [224, 192] buffer with (CLS - proj_bias) at row 0, zeros elsewhere.
    // After matmul (row 0 = 0) + bias_add (row 0 = bias) + cls_add (row 0 = bias + CLS - bias = CLS).
    {
        auto cls_f32 = load_binary_f32(d + "/cls_token.bin");
        auto bias_f32 = load_binary_f32(d + "/patch_embed.proj.bias.bin");
        constexpr uint32_t seq_padded = 224, embed_dim = 192;
        std::vector<bfloat16> cls_data(seq_padded * embed_dim, bfloat16(0.0f));
        for (uint32_t j = 0; j < std::min<uint32_t>(cls_f32.size(), embed_dim); j++) {
            float bias_val = (j < bias_f32.size()) ? bias_f32[j] : 0.0f;
            cls_data[j] = bfloat16(cls_f32[j] - bias_val);
        }
        auto cls_tiled = tilize_nfaces(cls_data, seq_padded, embed_dim);
        uint32_t n_tiles = num_tiles(seq_padded) * num_tiles(embed_dim);
        weights.patch_embed.cls_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
        write_to_device(ctx, weights.patch_embed.cls_buf, cls_tiled);
    }

    // Transformer blocks
    weights.blocks.resize(12);
    for (uint32_t i = 0; i < 12; i++) {
        std::string prefix = d + "/blocks." + std::to_string(i) + ".";

        // LayerNorm 1
        weights.blocks[i].ln1_gamma = load_ln_param(ctx, prefix + "norm1.weight.bin", 192);
        weights.blocks[i].ln1_beta = load_ln_param(ctx, prefix + "norm1.bias.bin", 192);

        // Attention QKV
        weights.blocks[i].attn.qkv_weight = load_weight_2d(ctx, prefix + "attn.qkv.weight.bin", 192, 576);
        weights.blocks[i].attn.qkv_bias = load_bias_broadcast_row(ctx, prefix + "attn.qkv.bias.bin", 576, 224);

        // Attention output projection
        weights.blocks[i].attn.proj_weight = load_weight_2d(ctx, prefix + "attn.proj.weight.bin", 192, 192);
        weights.blocks[i].attn.proj_bias = load_bias_broadcast_row(ctx, prefix + "attn.proj.bias.bin", 192, 224);

        // LayerNorm 2
        weights.blocks[i].ln2_gamma = load_ln_param(ctx, prefix + "norm2.weight.bin", 192);
        weights.blocks[i].ln2_beta = load_ln_param(ctx, prefix + "norm2.bias.bin", 192);

        // MLP
        weights.blocks[i].mlp.fc1_weight = load_weight_2d(ctx, prefix + "mlp.fc1.weight.bin", 192, 768);
        weights.blocks[i].mlp.fc1_bias = load_bias_broadcast_row(ctx, prefix + "mlp.fc1.bias.bin", 768, 224);
        weights.blocks[i].mlp.fc2_weight = load_weight_2d(ctx, prefix + "mlp.fc2.weight.bin", 768, 192);
        weights.blocks[i].mlp.fc2_bias = load_bias_broadcast_row(ctx, prefix + "mlp.fc2.bias.bin", 192, 224);

        fmt::print("  Block {} loaded\n", i);
    }

    // Final LayerNorm
    weights.final_ln_gamma = load_ln_param(ctx, d + "/norm.weight.bin", 192);
    weights.final_ln_beta = load_ln_param(ctx, d + "/norm.bias.bin", 192);

    // Classification head
    weights.head_weight = load_weight_2d(ctx, d + "/head.weight.bin", 192, 1000);
    weights.head_bias = load_bias_broadcast_row(ctx, d + "/head.bias.bin", 1000, 32);

    fmt::print("All weights loaded.\n");
    return weights;
}

}  // namespace vit
