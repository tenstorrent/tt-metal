// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <string_view>
#include <tt_stl/assert.hpp>

#include "core/ttnn_all_includes.hpp"
#include "metal/common/program_utils.hpp"

// Shared by swiglu_packed_fw and swiglu_packed_bw: the checks a packed [gate | up] tensor and
// its companions must pass, and the block geometry both program factories split work by.
namespace ttml::metal::ops::swiglu_packed {

// packed's shape with the two halves collapsed into one. packed must already be known 4D.
inline ttnn::Shape halve_last_dim(const ttnn::Shape& shape) {
    return ttnn::Shape({shape[0], shape[1], shape[2], shape[-1] / 2U});
}

// Every tensor the packed ops touch: on device, TILE, bf16, interleaved.
inline void check_tensor(const ttnn::Tensor& tensor, std::string_view name, std::string_view op) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "{}: {} must be on device, got storage type {}",
        op,
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "{}: {} buffer is null", op, name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "{}: {} must be TILE, got layout {}",
        op,
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "{}: {} must be BFLOAT16, got {}",
        op,
        name,
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
        "{}: {} must be INTERLEAVED, got {}",
        op,
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
}

// Both tensors must already have passed check_tensor. The kernels take every buffer address as
// local to packed's device, so a tensor elsewhere would be read through the wrong device.
inline void check_same_device(
    const ttnn::Tensor& tensor, const ttnn::Tensor& reference, std::string_view name, std::string_view op) {
    TT_FATAL(tensor.device() == reference.device(), "{}: {} is not on packed's device", op, name);
}

// 4D, with both the padded and the logical last dim splitting into two tile-aligned halves.
inline void validate_packed(const ttnn::Tensor& packed, std::string_view op) {
    const auto& padded = packed.padded_shape();
    TT_FATAL(padded.rank() == 4U, "{}: packed must be 4D, got rank {}", op, padded.rank());
    const uint32_t two_tiles_w = 2U * tt::constants::TILE_WIDTH;
    TT_FATAL(
        padded[-1] % two_tiles_w == 0U,
        "{}: packed last padded dim {} must be a multiple of {} so each half is tile-aligned",
        op,
        padded[-1],
        two_tiles_w);
    TT_FATAL(
        packed.logical_shape()[-1] % two_tiles_w == 0U,
        "{}: packed last logical dim {} must be a multiple of {} so the gate|up split lands on the "
        "tile boundary where the kernels split the padded row",
        op,
        packed.logical_shape()[-1],
        two_tiles_w);
}

// Padded and logical shape both equal to packed's with the last dim halved: the forward's output
// and the backward's upstream gradient. A different rank fails the same comparison.
inline void validate_half_of_packed(
    const ttnn::Tensor& tensor, const ttnn::Tensor& packed, std::string_view name, std::string_view op) {
    TT_FATAL(
        tensor.padded_shape() == halve_last_dim(packed.padded_shape()),
        "{}: {} padded shape {} must be packed {} with the last dim halved",
        op,
        name,
        tensor.padded_shape(),
        packed.padded_shape());
    TT_FATAL(
        tensor.logical_shape() == halve_last_dim(packed.logical_shape()),
        "{}: {} logical shape {} must be packed {} with the last dim halved",
        op,
        name,
        tensor.logical_shape(),
        packed.logical_shape());
}

// Padded and logical shape both equal to packed's: the backward's output.
inline void validate_same_as_packed(
    const ttnn::Tensor& tensor, const ttnn::Tensor& packed, std::string_view name, std::string_view op) {
    TT_FATAL(
        tensor.padded_shape() == packed.padded_shape(),
        "{}: {} padded shape {} must match packed {}",
        op,
        name,
        tensor.padded_shape(),
        packed.padded_shape());
    TT_FATAL(
        tensor.logical_shape() == packed.logical_shape(),
        "{}: {} logical shape {} must match packed {}",
        op,
        name,
        tensor.logical_shape(),
        packed.logical_shape());
}

// How both program factories cut a packed tensor into blocks of tiles.
struct BlockGeometry {
    uint32_t Wt;          // width of one half (gate or up) in tiles; a packed row is 2*Wt tiles
    uint32_t total_rows;  // tile-rows over all leading dims
    uint32_t block_size;  // largest divisor of Wt up to 4; real MLP widths (I/32 a multiple of 8) get 4
    uint32_t blocks_per_row;
    uint32_t total_blocks;
};

inline BlockGeometry block_geometry(const ttnn::Shape& packed_padded_shape) {
    BlockGeometry g{};
    g.Wt = packed_padded_shape[-1] / tt::constants::TILE_WIDTH / 2U;
    g.total_rows =
        packed_padded_shape[0] * packed_padded_shape[1] * (packed_padded_shape[-2] / tt::constants::TILE_HEIGHT);
    g.block_size = get_block_size(g.Wt, 4U);
    g.blocks_per_row = g.Wt / g.block_size;
    g.total_blocks = g.total_rows * g.blocks_per_row;
    return g;
}

}  // namespace ttml::metal::ops::swiglu_packed
