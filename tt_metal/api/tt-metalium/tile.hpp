// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <array>
#include <optional>
#include <tuple>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {
enum class DataFormat : uint8_t;
}  // namespace tt

namespace tt::tt_metal {

enum class DataType;

struct Tile {
    // The shape of the tile in elements (H, W)
    using TileShape = std::array<uint32_t, 2>;

    // The shape of the face in elements (H, W)
    using FaceShape = std::array<uint32_t, 2>;

    /**
     * Construct a Tile with a given tile shape (H, W).
     *
     * The tile face shape is derived automatically from the tile shape.
     */
    Tile(TileShape tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH}, bool transpose_tile = false);

    /**
     * Construct a Tile with a given tile shape (H, W) and custom face shape (H, W).
     */
    Tile(TileShape tile_shape, FaceShape face_shape, bool transpose_tile = false);

    /**
     * Construct a Tile from a grid of faces.
     *
     * num_faces_height and num_faces_width are the number of faces along each axis.
     * If a face shape is not provided, the default face shape (16x16) is used.
     * The tile shape in elements is face_shape scaled by those counts.
     *
     * Tile::from_face_grid(2, 2, {8, 16});
     * //                   │  │  └─ each face: 8 rows × 16 columns of elements
     * //                   │  └─ tile: 2 columns of faces
     * //                   └─ tile: 2 rows of faces
     *
     * That produces a 16 × 32-element tile, containing four faces:
     *
     *          16 elements   16 elements
     *         ┌─────────────┬─────────────┐
     *  8 rows │   face 0    │   face 1    │
     *         ├─────────────┼─────────────┤
     *  8 rows │   face 2    │   face 3    │
     *         └─────────────┴─────────────┘
     */
    static Tile from_face_grid(
        uint32_t num_faces_height,
        uint32_t num_faces_width,
        FaceShape face_shape = {constants::FACE_HEIGHT, constants::FACE_WIDTH}) {
        return Tile({face_shape[0] * num_faces_height, face_shape[1] * num_faces_width}, face_shape);
    }

    // Getter methods
    uint32_t get_height() const { return tile_shape[0]; }
    uint32_t get_width() const { return tile_shape[1]; }
    uint32_t get_num_faces() const { return num_faces; }
    uint32_t get_tile_hw() const { return tile_hw; }
    uint32_t get_face_hw() const { return face_hw; }
    uint32_t get_partial_face() const { return partial_face; }
    uint32_t get_narrow_tile() const { return narrow_tile; }
    TileShape get_tile_shape() const { return tile_shape; }
    FaceShape get_face_shape() const { return face_shape; }
    bool get_transpose_within_face() const { return transpose_within_face; }
    bool get_transpose_of_faces() const { return transpose_of_faces; }

    uint32_t get_tile_size(const DataFormat& format) const;
    // Uses datatype_to_dataformat_converter(data_type).
    uint32_t get_tile_size(DataType data_type) const;

    // operators
    bool operator==(const Tile& other) const;

    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }

private:
    std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH};
    std::array<uint32_t, 2> face_shape = {constants::FACE_HEIGHT, constants::FACE_WIDTH};
    uint32_t tile_hw = constants::TILE_HW;
    uint32_t face_hw = constants::FACE_HW;
    uint32_t num_faces = constants::TILE_HW / constants::FACE_HW;
    uint32_t partial_face = 0;
    uint32_t narrow_tile = 0;
    bool transpose_within_face = false;  // transpose datums within each face
    bool transpose_of_faces = false;     // transpose the face order
};

std::ostream& operator<<(std::ostream& os, const Tile& tile);

}  // namespace tt::tt_metal
