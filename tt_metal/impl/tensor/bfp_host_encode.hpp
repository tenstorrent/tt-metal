// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include "../data_format/bfp_tile_encode.hpp"

#include <algorithm>
#include <bit>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>

namespace tt::tt_metal::detail {

// Encode face rows directly from the logical matrix. The ordinary path first
// creates a full-size tile-layout buffer, only to read it once for BFP packing.
// A row stride also allows mesh column slices to bypass a contiguous copy.
// Keep the ordinary path for padding, nonstandard tiles and other source types.
template <typename T>
std::optional<HostTensor> try_encode_bfp_matrix(
    std::span<const T> buffer, const TensorSpec& spec, size_t row_stride = 0) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16>) {
        const auto dtype = spec.data_type();
        const auto& tile = spec.tile();
        if ((dtype != DataType::BFLOAT4_B && dtype != DataType::BFLOAT8_B) || spec.layout() != Layout::TILE ||
            spec.logical_2d_shape() != spec.physical_shape() || tile.get_height() != 32 || tile.get_width() != 32 ||
            tile.get_face_shape()[0] != 16 || tile.get_face_shape()[1] != 16 || tile.get_transpose_within_face() ||
            tile.get_transpose_of_faces() || std::endian::native != std::endian::little) {
            return std::nullopt;
        }
        const size_t width = spec.physical_shape().width();
        const size_t height = spec.physical_shape().height();
        if (width % 32 != 0 || height % 32 != 0) {
            return std::nullopt;
        }
        row_stride = row_stride == 0 ? width : row_stride;
        if (row_stride < width || (height != 0 && width != 0 &&
                                   (buffer.size() < width || height - 1 > (buffer.size() - width) / row_stride))) {
            return std::nullopt;
        }
        auto encode = [&]<int bits>() {
            const BfpTileGeometry geometry(tile);
            const size_t tile_count = height * width / (geometry.height * geometry.width);
            const size_t bytes = tile_count * geometry.packed_bytes<bits>();
            TT_FATAL(
                bytes == spec.compute_packed_buffer_size_bytes(),
                "BFP encoder buffer size {} differs from tensor specification {}",
                bytes,
                spec.compute_packed_buffer_size_bytes());
            const size_t words = bytes / sizeof(uint32_t);
            // Workers initialize their own output pages. The array owner stays
            // attached to HostBuffer through every host-tensor copy and upload.
            auto packed = std::make_shared_for_overwrite<uint32_t[]>(words);
            encode_bfp_tiles<bits>(
                buffer.data(),
                reinterpret_cast<uint8_t*>(packed.get()),
                tile_count,
                geometry,
                true,
                row_stride,
                width / geometry.width);
            HostBuffer output(ttsl::Span<uint32_t>(packed.get(), words), MemoryPin(packed));
            return HostTensor::from_buffer(std::move(output), spec);
        };
        return dtype == DataType::BFLOAT8_B ? encode.template operator()<7>() : encode.template operator()<3>();
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::detail
