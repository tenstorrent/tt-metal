// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "common/bfloat16.hpp"
#include "common/tt_backend_api_types.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"

namespace tt {

namespace tt_metal {

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

    Tile(const std::array<uint32_t, 2>& tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH}) : tile_shape(tile_shape) {
        auto it = std::find_if(TILE_FACE_HW_CHOICES.begin(), TILE_FACE_HW_CHOICES.end(),
                           [this, &tile_shape](const auto& pair) {
                               if (pair[0] == tile_shape) {
                                   this->face_shape = pair[1];
                                   return true;
                               }
                               return false;
                           });
        if (it == TILE_FACE_HW_CHOICES.end()) {
            TT_THROW("Tile size is not valid for our hardware");
        }

        tile_hw = tile_shape[0] * tile_shape[1];
        face_hw = face_shape[0] * face_shape[1];
        num_faces = tile_hw / face_hw;
        partial_face = (uint32_t)(tile_shape[0] < constants::TILE_HEIGHT);
        narrow_tile = (uint32_t)(tile_shape[1] < constants::TILE_WIDTH);
    }

    // Getter methods
    const uint32_t get_num_faces() const { return num_faces; }
    const uint32_t get_tile_hw() const { return tile_hw; }
    const uint32_t get_face_hw() const { return face_hw; }
    const uint32_t get_partial_face() const { return partial_face; }
    const uint32_t get_narrow_tile() const { return narrow_tile; }
    const std::array<uint32_t, 2> get_tile_shape() const { return tile_shape; }
    const std::array<uint32_t, 2> get_face_shape() const { return face_shape; }

    const uint32_t get_tile_size(const DataFormat& format) const {
        switch (format) {
            case DataFormat::Bfp2:
            case DataFormat::Bfp2_b: return (tile_hw / 4) + (16 * num_faces);
            case DataFormat::Bfp4:
            case DataFormat::Bfp4_b: return (tile_hw / 2) + (16 * num_faces);
            case DataFormat::Bfp8:
            case DataFormat::Bfp8_b: return tile_hw + (16 * num_faces);
            case DataFormat::Float16:
            case DataFormat::Float16_b: return (tile_hw * 2);
            case DataFormat::Float32: return (tile_hw * 4);
            case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported atm");
            case DataFormat::Int8: return tile_hw;
            case DataFormat::Lf8: return tile_hw;
            case DataFormat::UInt8: return tile_hw;
            case DataFormat::UInt16: return (tile_hw * 2);
            case DataFormat::UInt32: return (tile_hw * 4);
            case DataFormat::RawUInt8: return tile_hw;
            case DataFormat::RawUInt16: return (tile_hw * 2);
            case DataFormat::Int32: return (tile_hw * 4);
            case DataFormat::RawUInt32: return (tile_hw * 4);
            case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
            default: throw std::invalid_argument("Unknown format");
        }
    }

    // operators
    bool operator==(const Tile& other) const {
        return tile_shape == other.tile_shape && face_shape == other.face_shape;
    }

    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    const auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }
};

}  // namespace tt_metal

}  // namespace tt
