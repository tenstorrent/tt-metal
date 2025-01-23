// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_types_generated.h"

namespace tt::tt_metal {
inline namespace v0 {

inline BufferType FromFlatbuffer(tt::tt_metal::flatbuffer::BufferType type) {
    switch (type) {
        case tt::tt_metal::flatbuffer::BufferType::DRAM: return BufferType::DRAM;
        case tt::tt_metal::flatbuffer::BufferType::L1: return BufferType::L1;
        case tt::tt_metal::flatbuffer::BufferType::SystemMemory: return BufferType::SYSTEM_MEMORY;
        case tt::tt_metal::flatbuffer::BufferType::L1Small: return BufferType::L1_SMALL;
        case tt::tt_metal::flatbuffer::BufferType::Trace: return BufferType::TRACE;
        default: throw std::invalid_argument("Unknown BufferType value in FromFlatbuffer()");
    }
}

inline tt::tt_metal::DataMovementProcessor FromFlatbuffer(tt::tt_metal::flatbuffer::DataMovementProcessor in) {
    switch (in) {
        case tt::tt_metal::flatbuffer::DataMovementProcessor::RISCV_0:
            return tt::tt_metal::DataMovementProcessor::RISCV_0;
        case tt::tt_metal::flatbuffer::DataMovementProcessor::RISCV_1:
            return tt::tt_metal::DataMovementProcessor::RISCV_1;
        default: throw std::invalid_argument("Unknown DataMovementProcessor value in FromFlatbuffer()");
    }
}

inline tt::tt_metal::NOC FromFlatbuffer(tt::tt_metal::flatbuffer::NOC in) {
    switch (in) {
        case tt::tt_metal::flatbuffer::NOC::NOC_0: return tt::tt_metal::NOC::NOC_0;
        case tt::tt_metal::flatbuffer::NOC::NOC_1: return tt::tt_metal::NOC::NOC_1;
        default: throw std::invalid_argument("Invalid NOC value passed to FromFlatbuffer");
    }
}

inline tt::tt_metal::NOC_MODE FromFlatbuffer(tt::tt_metal::flatbuffer::NOC_MODE in) {
    switch (in) {
        case tt::tt_metal::flatbuffer::NOC_MODE::DM_DEDICATED_NOC: return tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
        case tt::tt_metal::flatbuffer::NOC_MODE::DM_DYNAMIC_NOC: return tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC;
        default: throw std::invalid_argument("Unknown NOC_MODE value in FromFlatbuffer()");
    }
}

inline tt::tt_metal::Eth FromFlatbuffer(tt::tt_metal::flatbuffer::Eth in) {
    switch (in) {
        case tt::tt_metal::flatbuffer::Eth::SENDER: return tt::tt_metal::Eth::SENDER;
        case tt::tt_metal::flatbuffer::Eth::RECEIVER: return tt::tt_metal::Eth::RECEIVER;
        case tt::tt_metal::flatbuffer::Eth::IDLE: return tt::tt_metal::Eth::IDLE;
        default: throw std::invalid_argument("Unknown Eth value in FromFlatbuffer()");
    }
}

inline MathFidelity FromFlatbuffer(tt::tt_metal::flatbuffer::MathFidelity input) {
    switch (input) {
        case tt::tt_metal::flatbuffer::MathFidelity::LoFi: return MathFidelity::LoFi;
        case tt::tt_metal::flatbuffer::MathFidelity::HiFi2: return MathFidelity::HiFi2;
        case tt::tt_metal::flatbuffer::MathFidelity::HiFi3: return MathFidelity::HiFi3;
        case tt::tt_metal::flatbuffer::MathFidelity::HiFi4: return MathFidelity::HiFi4;
        case tt::tt_metal::flatbuffer::MathFidelity::Invalid: return MathFidelity::Invalid;
        default: throw std::invalid_argument("Unknown MathFidelity value in FromFlatbuffer()");
    }
}

inline UnpackToDestMode FromFlatbuffer(tt::tt_metal::flatbuffer::UnpackToDestMode input) {
    switch (input) {
        case tt::tt_metal::flatbuffer::UnpackToDestMode::UnpackToDestFp32: return UnpackToDestMode::UnpackToDestFp32;
        case tt::tt_metal::flatbuffer::UnpackToDestMode::Default: return UnpackToDestMode::Default;
        default: throw std::invalid_argument("Invalid UnpackToDestMode value passed to FromFlatbuffer");
    }
}

inline tt::DataFormat FromFlatbuffer(tt::tt_metal::flatbuffer::DataFormat input) {
    switch (input) {
        case tt::tt_metal::flatbuffer::DataFormat::Float32: return tt::DataFormat::Float32;
        case tt::tt_metal::flatbuffer::DataFormat::Float16: return tt::DataFormat::Float16;
        case tt::tt_metal::flatbuffer::DataFormat::Bfp8: return tt::DataFormat::Bfp8;
        case tt::tt_metal::flatbuffer::DataFormat::Bfp4: return tt::DataFormat::Bfp4;
        case tt::tt_metal::flatbuffer::DataFormat::Bfp2: return tt::DataFormat::Bfp2;
        case tt::tt_metal::flatbuffer::DataFormat::Float16_b: return tt::DataFormat::Float16_b;
        case tt::tt_metal::flatbuffer::DataFormat::Bfp8_b: return tt::DataFormat::Bfp8_b;
        case tt::tt_metal::flatbuffer::DataFormat::Bfp4_b: return tt::DataFormat::Bfp4_b;
        case tt::tt_metal::flatbuffer::DataFormat::Bfp2_b: return tt::DataFormat::Bfp2_b;
        case tt::tt_metal::flatbuffer::DataFormat::Lf8: return tt::DataFormat::Lf8;
        case tt::tt_metal::flatbuffer::DataFormat::Fp8_e4m3: return tt::DataFormat::Fp8_e4m3;
        case tt::tt_metal::flatbuffer::DataFormat::Int8: return tt::DataFormat::Int8;
        case tt::tt_metal::flatbuffer::DataFormat::Tf32: return tt::DataFormat::Tf32;
        case tt::tt_metal::flatbuffer::DataFormat::UInt8: return tt::DataFormat::UInt8;
        case tt::tt_metal::flatbuffer::DataFormat::UInt16: return tt::DataFormat::UInt16;
        case tt::tt_metal::flatbuffer::DataFormat::Int32: return tt::DataFormat::Int32;
        case tt::tt_metal::flatbuffer::DataFormat::UInt32: return tt::DataFormat::UInt32;
        case tt::tt_metal::flatbuffer::DataFormat::RawUInt8: return tt::DataFormat::RawUInt8;
        case tt::tt_metal::flatbuffer::DataFormat::RawUInt16: return tt::DataFormat::RawUInt16;
        case tt::tt_metal::flatbuffer::DataFormat::RawUInt32: return tt::DataFormat::RawUInt32;
        case tt::tt_metal::flatbuffer::DataFormat::Invalid: return tt::DataFormat::Invalid;
        default: throw std::invalid_argument("Unknown DataFormat value in FromFlatbuffer()");
    }
}

inline Tile FromFlatbuffer(const tt::tt_metal::flatbuffer::Tile* tile_fb) {
    if (!tile_fb) {
        throw std::runtime_error("Invalid Tile FlatBuffer object");
    }

    // Convert FlatBuffer vectors to std::array
    std::array<uint32_t, 2> tile_shape = {tile_fb->tile_shape()->Get(0), tile_fb->tile_shape()->Get(1)};
    std::array<uint32_t, 2> face_shape = {tile_fb->face_shape()->Get(0), tile_fb->face_shape()->Get(1)};

    // Create and return the Tile object, explicitly initializing the members
    Tile tile;
    tile.tile_shape = tile_shape;
    tile.face_shape = face_shape;
    tile.tile_hw = tile_fb->tile_hw();
    tile.face_hw = tile_fb->face_hw();
    tile.num_faces = tile_fb->num_faces();
    tile.partial_face = tile_fb->partial_face();
    tile.narrow_tile = tile_fb->narrow_tile();
    tile.transpose_within_face = tile_fb->transpose_within_face();
    tile.transpose_of_faces = tile_fb->transpose_of_faces();

    return tile;
}

inline std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> FromFlatbuffer(
    const flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>>* tiles_fb) {
    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles = {};
    if (tiles_fb) {
        for (size_t i = 0; i < tiles_fb->size() && i < NUM_CIRCULAR_BUFFERS; ++i) {
            tiles[i] = FromFlatbuffer(tiles_fb->Get(i));
        }
    }
    return tiles;
}

}  // namespace v0
}  // namespace tt::tt_metal
