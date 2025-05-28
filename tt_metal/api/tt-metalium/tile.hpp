// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

constexpr std::array<std::array<std::array<uint32_t, 2>, 2>, 12> TILE_FACE_HW_CHOICES = {{
    // TODO: add other tile shapes once llk supported it
    {{ {32, 32}, {16, 16} }},
    {{ {16, 32}, {16, 16} }},
    {{ {32, 16}, {16, 16} }},
    {{ {16, 16}, {16, 16} }},
    // this shapes are not supported yet on llk, just for host loopback
    {{ {8, 32}, {8, 16} }},
    {{ {4, 32}, {4, 16} }},
    {{ {2, 32}, {2, 16} }},
    {{ {1, 32}, {1, 16} }},
    // this shapes are not supported yet on llk, just for host loopback
    {{ {8, 16}, {8, 16} }},
    {{ {4, 16}, {4, 16} }},
    {{ {2, 16}, {2, 16} }},
    {{ {1, 16}, {1, 16} }}
}};

struct Tile {
    std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH};
    std::array<uint32_t, 2> face_shape = {constants::FACE_HEIGHT, constants::FACE_WIDTH};
    uint32_t tile_hw = constants::TILE_HW;
    uint32_t face_hw = constants::FACE_HW;
    uint32_t num_faces = constants::TILE_HW / constants::FACE_HW;
    uint32_t partial_face = 0;
    uint32_t narrow_tile = 0;
    bool transpose_within_face = false; // tranpose datums within each face
    bool transpose_of_faces = false; // transpose the face order

    Tile(
        std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH},
        bool transpose_tile = false);

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
};

}  // namespace tt::tt_metal
