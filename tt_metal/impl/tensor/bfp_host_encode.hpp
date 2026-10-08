// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include "../data_format/bfp_simd.hpp"
#include "common/executor.hpp"

#include <algorithm>
#include <bit>
#include <future>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

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
            constexpr size_t tile_bytes = 64 + 1024 * (bits + 1) / 8;
            const size_t tile_count = height * width / 1024;
            const size_t tiles_wide = width / 32;
            std::vector<uint32_t> packed(tile_count * tile_bytes / sizeof(uint32_t));
            const auto pack_row = bfp_simd::select_row_packer<bits, T>();
            auto process = [&](size_t begin, size_t end) {
                for (size_t tile_id = begin; tile_id < end; ++tile_id) {
                    const auto* source =
                        buffer.data() + (tile_id / tiles_wide) * 32 * row_stride + (tile_id % tiles_wide) * 32;
                    auto* dest = reinterpret_cast<uint8_t*>(packed.data()) + tile_id * tile_bytes;
                    for (size_t face = 0; face < 4; ++face) {
                        for (size_t row = 0; row < 16; ++row) {
                            const size_t packed_row = face * 16 + row;
                            pack_row(
                                source + (face / 2 * 16 + row) * row_stride + face % 2 * 16,
                                dest + packed_row,
                                dest + 64 + packed_row * 2 * (bits + 1));
                        }
                    }
                }
            };
            if (tile_count < 256) {
                process(0, tile_count);
            } else {
                const size_t chunks = std::max<size_t>(1, std::min(GetExecutor().num_workers(), tile_count / 128));
                std::vector<std::shared_future<void>> pending;
                pending.reserve(chunks);
                for (size_t chunk = 0; chunk < chunks; ++chunk) {
                    const size_t begin = tile_count * chunk / chunks;
                    const size_t end = tile_count * (chunk + 1) / chunks;
                    pending.emplace_back(async([&, begin, end] { process(begin, end); }));
                }
                for (auto& future : pending) {
                    future.get();
                }
            }
            return HostTensor::from_buffer(HostBuffer(std::move(packed)), spec);
        };
        return dtype == DataType::BFLOAT8_B ? encode.template operator()<7>() : encode.template operator()<3>();
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::detail
