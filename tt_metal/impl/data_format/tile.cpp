// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tile.hpp>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include <tt_stl/assert.hpp>
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "math.hpp"
#include "tt_backend_api_types.hpp"
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal {

constexpr std::array<std::array<std::array<uint32_t, 2>, 2>, 12> TILE_FACE_HW_CHOICES = {
    {// TODO: add other tile shapes once llk supported it
     {{{32, 32}, {16, 16}}},
     {{{16, 32}, {16, 16}}},
     {{{32, 16}, {16, 16}}},
     {{{16, 16}, {16, 16}}},
     // these shapes are not supported yet on llk, just for host loopback
     {{{8, 32}, {8, 16}}},
     {{{4, 32}, {4, 16}}},
     {{{2, 32}, {2, 16}}},
     {{{1, 32}, {1, 16}}},
     // these shapes are not supported yet on llk, just for host loopback
     {{{8, 16}, {8, 16}}},
     {{{4, 16}, {4, 16}}},
     {{{2, 16}, {2, 16}}},
     {{{1, 16}, {1, 16}}}}};

std::array<uint32_t, 2> get_default_face_shape(std::array<uint32_t, 2> tile_shape) {
    const auto* it =
        std::find_if(TILE_FACE_HW_CHOICES.begin(), TILE_FACE_HW_CHOICES.end(), [tile_shape](const auto& pair) {
            return pair[0] == tile_shape;
        });
    TT_FATAL(it != TILE_FACE_HW_CHOICES.end(), "Tile size is not valid for our hardware");
    return (*it)[1];
}

Tile::Tile(std::array<uint32_t, 2> tile_shape, bool transpose_tile) :
    tile_shape(tile_shape),
    face_shape(get_default_face_shape(tile_shape)),
    tile_hw(tile_shape[0] * tile_shape[1]),
    face_hw(face_shape[0] * face_shape[1]),
    num_faces(tile_hw / face_hw),
    partial_face(static_cast<uint32_t>(tile_shape[0] < constants::TILE_HEIGHT)),
    narrow_tile(static_cast<uint32_t>(tile_shape[1] < constants::TILE_WIDTH)),
    transpose_within_face(transpose_tile),
    transpose_of_faces(transpose_tile) {
    if (transpose_tile) {
        TT_FATAL(
            (this->tile_shape[0] == constants::FACE_HEIGHT || this->tile_shape[0] == constants::TILE_HEIGHT),
            "Tile height must equal 16 or 32 in transpose mode");
    }
}

Tile::Tile(std::array<uint32_t, 2> tile_shape, std::array<uint32_t, 2> face_shape, bool transpose_tile) :
    tile_shape(tile_shape),
    face_shape(face_shape),
    tile_hw(tile_shape[0] * tile_shape[1]),
    face_hw(face_shape[0] * face_shape[1]),
    num_faces(tile_hw / face_hw),
    partial_face(static_cast<uint32_t>(tile_shape[0] < constants::TILE_HEIGHT)),
    narrow_tile(static_cast<uint32_t>(tile_shape[1] < constants::TILE_WIDTH)),
    transpose_within_face(transpose_tile),
    transpose_of_faces(transpose_tile) {
    const bool known_tile_shape =
        std::any_of(TILE_FACE_HW_CHOICES.begin(), TILE_FACE_HW_CHOICES.end(), [&tile_shape](const auto& pair) {
            return pair[0] == tile_shape;
        });

    TT_FATAL(known_tile_shape, "Tile size is not valid for our hardware");

    // Face shape must fit evenly into tile shape
    TT_FATAL(
        (tile_shape[0] % face_shape[0]) == 0 && (tile_shape[1] % face_shape[1]) == 0,
        "tile shape ({}x{}) must tile evenly into face shape ({}x{})",
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]);

    // These constants are derived from device side constants at (tt-llk/common/tensor_shape.h)
    // Maintainers: Please keep these in sync with device side constants.
    constexpr std::uint8_t MAX_FACE_R_DIM = 16;
    constexpr std::uint8_t MAX_FACE_C_DIM = 16;
    constexpr std::uint8_t MAX_TILE_R_DIM = 32;
    constexpr std::uint8_t MAX_TILE_C_DIM = 32;
    constexpr std::uint8_t MAX_NUM_FACES_R_DIM = 2;
    constexpr std::uint8_t MAX_NUM_FACES_C_DIM = 2;
    constexpr std::uint8_t MAX_NUM_FACES = MAX_NUM_FACES_R_DIM * MAX_NUM_FACES_C_DIM;

    TT_FATAL(
        face_shape[0] <= MAX_FACE_R_DIM && face_shape[1] <= MAX_FACE_C_DIM,
        "face shape ({}x{}) exceeds the maximum supported face shape ({}x{})",
        face_shape[0],
        face_shape[1],
        MAX_FACE_R_DIM,
        MAX_FACE_C_DIM);
    TT_FATAL(
        tile_shape[0] <= MAX_TILE_R_DIM && tile_shape[1] <= MAX_TILE_C_DIM,
        "tile shape ({}x{}) exceeds the maximum supported tile shape ({}x{})",
        tile_shape[0],
        tile_shape[1],
        MAX_TILE_R_DIM,
        MAX_TILE_C_DIM);

    auto num_faces_r = tile_shape[0] / face_shape[0];
    auto num_faces_c = tile_shape[1] / face_shape[1];
    TT_FATAL(
        num_faces_r <= MAX_NUM_FACES_R_DIM && num_faces_c <= MAX_NUM_FACES_C_DIM,
        "num_faces ({}x{}) exceeds the maximum supported num_faces ({}x{})",
        num_faces_r,
        num_faces_c,
        MAX_NUM_FACES_R_DIM,
        MAX_NUM_FACES_C_DIM);

    TT_FATAL(
        num_faces <= MAX_NUM_FACES,
        "num_faces ({}) exceeds the maximum supported num_faces ({})",
        num_faces,
        MAX_NUM_FACES);
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
        case DataFormat::MxFp4:
        case DataFormat::MxFp6P:
        case DataFormat::MxFp6R:
        case DataFormat::MxFp8R:
        case DataFormat::MxFp8P:
        case DataFormat::MxInt8:
        case DataFormat::MxInt4:
        case DataFormat::MxInt2: {
            // All MX formats share a [block scales | packed elements] tile layout:
            // one E8M0 scale byte per 32-element block (padded to L1 alignment),
            // followed by the elements packed at the format's storage width.
            constexpr uint32_t kMxBlockSize = 32;
            TT_ASSERT(tile_hw % kMxBlockSize == 0, "MX tile size must be a multiple of 32 elements");
            const uint32_t exp_bytes = tt::round_up(tile_hw / kMxBlockSize, l1_alignment);
            uint32_t elem_bytes = tile_hw;  // 8-bit storage: MxFp6 / MxFp8 / MxInt8
            if (format == DataFormat::MxFp4 || format == DataFormat::MxInt4) {
                elem_bytes = tile_hw / 2;  // 4-bit elements, 2 per byte
            } else if (format == DataFormat::MxInt2) {
                elem_bytes = tile_hw / 4;  // 2-bit elements, 4 per byte
            }
            return exp_bytes + elem_bytes;
        }
        case DataFormat::Float16:
        case DataFormat::Float16_b: return (tile_hw * 2);
        case DataFormat::Float32: return (tile_hw * 4);
        case DataFormat::Fp8_e4m3:
        case DataFormat::Int8:
        case DataFormat::Lf8:
        case DataFormat::UInt8:
        case DataFormat::RawUInt8: return tile_hw;
        case DataFormat::UInt16:
        case DataFormat::Int16:
        case DataFormat::RawUInt16: return (tile_hw * 2);
        case DataFormat::UInt32:
        case DataFormat::Int32:
        case DataFormat::RawUInt32: return (tile_hw * 4);
        case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported atm");
        case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
        default: throw std::invalid_argument("Unknown format");
    }
}

bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Tile& tile) {
    ttsl::reflection::operator<<(os, tile);
    return os;
}

}  // namespace tt::tt_metal
