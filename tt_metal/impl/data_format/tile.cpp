// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tile.hpp>
#include <algorithm>
#include <stdexcept>

#include "assert.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "math.hpp"
#include "tt_backend_api_types.hpp"

namespace tt::tt_metal {

Tile::Tile(std::array<uint32_t, 2> tile_shape, bool transpose_tile) : tile_shape(tile_shape) {
    auto it = std::find_if(TILE_FACE_HW_CHOICES.begin(), TILE_FACE_HW_CHOICES.end(), [this](const auto& pair) {
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
        TT_FATAL(
            (this->tile_shape[0] == constants::FACE_HEIGHT || this->tile_shape[0] == constants::TILE_HEIGHT),
            "Tile height must equal 16 or 32 in transpose mode");
    }

    tile_hw = this->tile_shape[0] * this->tile_shape[1];
    face_hw = face_shape[0] * face_shape[1];
    num_faces = tile_hw / face_hw;
    partial_face = static_cast<uint32_t>(this->tile_shape[0] < constants::TILE_HEIGHT);
    narrow_tile = static_cast<uint32_t>(this->tile_shape[1] < constants::TILE_WIDTH);
}

uint32_t Tile::get_tile_size(const DataFormat& format) const {
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
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

bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}

}  // namespace tt::tt_metal
