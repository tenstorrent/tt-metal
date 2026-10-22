// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tt-metalium/tile.hpp>
#include "bfp_simd.hpp"
#include "bfp_tasks.hpp"
#include "common/executor.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::detail {

// The legacy vector API and direct host-tensor factory use the same tile layout
// and traversal. A nonzero matrix_row_stride reads tiles from a logical matrix;
// otherwise the input contains consecutive row-major or face-major tiles.
struct BfpTileGeometry {
    size_t height;
    size_t width;
    size_t face_height;
    size_t face_width;
    size_t exponent_bytes;

    explicit BfpTileGeometry(const Tile& tile) :
        height(tile.get_height()),
        width(tile.get_width()),
        face_height(tile.get_face_shape()[0]),
        face_width(tile.get_face_shape()[1]) {
        const size_t rows = height * width / face_width;
        const size_t alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
        exponent_bytes = ((rows + alignment - 1) / alignment) * alignment;
    }

    template <int bits>
    size_t packed_bytes() const {
        return exponent_bytes + height * width * (bits + 1) / 8;
    }
};

template <typename Process>
void run_bfp_tile_ranges(size_t tile_count, const Process& process) {
    // A nested conversion must never wait for work queued to its own pool.
    if (tile_count < 256 || GetExecutor().this_worker_id() != -1) {
        process(0, tile_count);
        return;
    }
    const size_t chunks = std::max<size_t>(1, std::min(GetExecutor().num_workers(), tile_count / 128));
    if (chunks == 1) {
        process(0, tile_count);
        return;
    }
    run_bfp_tasks(chunks, [&](size_t chunk) {
        const size_t begin = tile_count * chunk / chunks;
        const size_t end = tile_count * (chunk + 1) / chunks;
        return async([&, begin, end] { process(begin, end); });
    });
}

template <int bits, typename T>
void encode_bfp_tiles(
    const T* input,
    uint8_t* output,
    size_t tile_count,
    const BfpTileGeometry& tile,
    bool row_major_input,
    size_t matrix_row_stride = 0,
    size_t tiles_per_matrix_row = 0) {
    const size_t elements = tile.height * tile.width;
    const size_t rows = elements / tile.face_width;
    const size_t tile_bytes = tile.packed_bytes<bits>();
    const auto pack_row = bfp_simd::select_row_packer<bits, T>();
    run_bfp_tile_ranges(tile_count, [&](size_t begin, size_t end) {
        for (size_t t = begin; t < end; ++t) {
            const T* source =
                input + (matrix_row_stride ? (t / tiles_per_matrix_row) * tile.height * matrix_row_stride +
                                                 (t % tiles_per_matrix_row) * tile.width
                                           : t * elements);
            auto* dest = output + t * tile_bytes;
            // Every encoded byte is written below. Only exponent padding needs
            // initialization, on the same worker that first touches the tile.
            std::memset(dest + rows, 0, tile.exponent_bytes - rows);
            size_t row = 0;
            for (size_t tr = 0; tr < tile.height / tile.face_height; ++tr) {
                for (size_t tc = 0; tc < tile.width / tile.face_width; ++tc) {
                    for (size_t r = 0; r < tile.face_height; ++r, ++row) {
                        const size_t offset =
                            row_major_input
                                ? (tr * tile.face_height + r) * (matrix_row_stride ? matrix_row_stride : tile.width) +
                                      tc * tile.face_width
                                : row * tile.face_width;
                        pack_row(source + offset, dest + row, dest + tile.exponent_bytes + row * 2 * (bits + 1));
                    }
                }
            }
        }
    });
}

}  // namespace tt::tt_metal::detail
