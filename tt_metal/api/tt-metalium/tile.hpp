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

struct Tile {
    Tile(
        std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH},
        bool transpose_tile = false);

    // Explicit face layout.
    //
    // The single-argument constructor derives `face_shape` from `tile_shape` via a fixed table, which
    // only ever yields the "tallest" face layout for a given tile. Use this overload when an operand
    // packs its data into shorter, more numerous faces than that default: e.g. a 2x32 tile holding
    // four 1x16 faces (a 2x2 face grid) rather than the default two 2x16 faces.
    //
    // `num_faces` stays derived (`tile_hw / face_hw`), so the face grid is always exactly
    // `tile_shape / face_shape` and cannot be set inconsistently.
    Tile(std::array<uint32_t, 2> tile_shape, std::array<uint32_t, 2> face_shape, bool transpose_tile = false);

    // Getter methods
    uint32_t get_height() const { return tile_shape[0]; }
    uint32_t get_width() const { return tile_shape[1]; }
    uint32_t get_num_faces() const { return num_faces; }
    uint32_t get_tile_hw() const { return tile_hw; }
    uint32_t get_face_hw() const { return face_hw; }
    uint32_t get_partial_face() const { return partial_face; }
    uint32_t get_narrow_tile() const { return narrow_tile; }
    std::array<uint32_t, 2> get_tile_shape() const { return tile_shape; }
    std::array<uint32_t, 2> get_face_shape() const { return face_shape; }
    bool get_transpose_within_face() const { return transpose_within_face; }
    bool get_transpose_of_faces() const { return transpose_of_faces; }

    uint32_t get_tile_size(const DataFormat& format) const;

    // operators
    bool operator==(const Tile& other) const;

    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }

private:
    // Fills in everything derived from `tile_shape` / `face_shape`. Shared by both constructors.
    void init_derived_dims(bool transpose_tile);

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
