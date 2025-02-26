// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "bfloat16.hpp"
#include "tt_backend_api_types.hpp"
#include "constants.hpp"
#include "math.hpp"
#include "hal.hpp"

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
    bool transpose_within_face = false; // tranpose datums within each face
    bool transpose_of_faces = false; // transpose the face order

    Tile(std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH}, bool transpose_tile = false) :
        tile_shape(tile_shape) {

        auto it = std::find_if(TILE_FACE_HW_CHOICES.begin(), TILE_FACE_HW_CHOICES.end(),
                            [this](const auto& pair) {
                                if (pair[0] == this->tile_shape) {
                                    this->face_shape = pair[1];
                                    return true;
                                }
                                return false;
                            });

        if (it == TILE_FACE_HW_CHOICES.end()) {
            TT_THROW("Tile size is not valid for our hardware");
        }

        if (transpose_tile) {
            transpose_within_face = true;
            transpose_of_faces = true;
        } else {
            transpose_within_face = false;
            transpose_of_faces = false;
        }

        if (transpose_tile) {
            TT_FATAL((this->tile_shape[0] == constants::FACE_HEIGHT || this->tile_shape[0] == constants::TILE_HEIGHT),
                    "Tile height must equal 16 or 32 in transpose mode");
        }

        tile_hw = this->tile_shape[0] * this->tile_shape[1];
        face_hw = face_shape[0] * face_shape[1];
        num_faces = tile_hw / face_hw;
        partial_face = static_cast<uint32_t>(this->tile_shape[0] < constants::TILE_HEIGHT);
        narrow_tile = static_cast<uint32_t>(this->tile_shape[1] < constants::TILE_WIDTH);
    }

    // Getter methods
    const uint32_t get_height() const { return tile_shape[0]; }
    const uint32_t get_width() const { return tile_shape[1]; }
    const uint32_t get_num_faces() const { return num_faces; }
    const uint32_t get_tile_hw() const { return tile_hw; }
    const uint32_t get_face_hw() const { return face_hw; }
    const uint32_t get_partial_face() const { return partial_face; }
    const uint32_t get_narrow_tile() const { return narrow_tile; }
    const std::array<uint32_t, 2> get_tile_shape() const { return tile_shape; }
    const std::array<uint32_t, 2> get_face_shape() const { return face_shape; }
    const bool get_transpose_within_face() const { return transpose_within_face; }
    const bool get_transpose_of_faces() const { return transpose_of_faces; }

    const uint32_t get_tile_size(const DataFormat& format) const {
        uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
        uint32_t aligned_exp_size = tt::round_up(face_shape[0] * num_faces, l1_alignment);
        switch (format) {
            case DataFormat::Bfp2:
            case DataFormat::Bfp2_b: return (tile_hw / 4) + aligned_exp_size;
            case DataFormat::Bfp4:
            case DataFormat::Bfp4_b: return (tile_hw / 2) + aligned_exp_size;
            case DataFormat::Bfp8:
            case DataFormat::Bfp8_b: return tile_hw + aligned_exp_size;
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
