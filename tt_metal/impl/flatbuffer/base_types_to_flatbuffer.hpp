// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_types_generated.h"

namespace tt::tt_metal {

// Original types defined in buffer_constants.hpp
inline tt::tt_metal::flatbuffer::BufferType ToFlatbuffer(BufferType type) {
    switch (type) {
        case BufferType::DRAM: return tt::tt_metal::flatbuffer::BufferType::DRAM;
        case BufferType::L1: return tt::tt_metal::flatbuffer::BufferType::L1;
        case BufferType::SYSTEM_MEMORY: return tt::tt_metal::flatbuffer::BufferType::SystemMemory;
        case BufferType::L1_SMALL: return tt::tt_metal::flatbuffer::BufferType::L1Small;
        case BufferType::TRACE: return tt::tt_metal::flatbuffer::BufferType::Trace;
        default: throw std::invalid_argument("Unknown BufferType value in ToFlatbuffer()");
    }
}

// Original types defined in buffer_constants.hpp
inline tt::tt_metal::flatbuffer::TensorMemoryLayout ToFlatbuffer(TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::Interleaved;
        case TensorMemoryLayout::SINGLE_BANK: return tt::tt_metal::flatbuffer::TensorMemoryLayout::SingleBank;
        case TensorMemoryLayout::HEIGHT_SHARDED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::HeightSharded;
        case TensorMemoryLayout::WIDTH_SHARDED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::WidthSharded;
        case TensorMemoryLayout::BLOCK_SHARDED: return tt::tt_metal::flatbuffer::TensorMemoryLayout::BlockSharded;
        default: throw std::invalid_argument("Unknown TensorMemoryLayout value in ToFlatbuffer()");
    }
}

// Original types defined in data_types.hpp
inline tt::tt_metal::flatbuffer::DataMovementProcessor ToFlatbuffer(tt::tt_metal::DataMovementProcessor in) {
    switch (in) {
        case tt::tt_metal::DataMovementProcessor::RISCV_0:
            return tt::tt_metal::flatbuffer::DataMovementProcessor::RISCV_0;
        case tt::tt_metal::DataMovementProcessor::RISCV_1:
            return tt::tt_metal::flatbuffer::DataMovementProcessor::RISCV_1;
        default: throw std::invalid_argument("Unknown DataMovementProcessor value in ToFlatbuffer()");
    }
}

inline tt::tt_metal::flatbuffer::NOC ToFlatbuffer(tt::tt_metal::NOC in) {
    switch (in) {
        case tt::tt_metal::NOC::NOC_0: return tt::tt_metal::flatbuffer::NOC::NOC_0;
        case tt::tt_metal::NOC::NOC_1: return tt::tt_metal::flatbuffer::NOC::NOC_1;
        default: throw std::invalid_argument("Invalid NOC value passed to ToFlatbuffer");
    }
}

inline tt::tt_metal::flatbuffer::NOC_MODE ToFlatbuffer(tt::tt_metal::NOC_MODE in) {
    switch (in) {
        case tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC: return tt::tt_metal::flatbuffer::NOC_MODE::DM_DEDICATED_NOC;
        case tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC: return tt::tt_metal::flatbuffer::NOC_MODE::DM_DYNAMIC_NOC;
        default: throw std::invalid_argument("Unknown NOC_MODE value in ToFlatbuffer()");
    }
}

inline tt::tt_metal::flatbuffer::Eth ToFlatbuffer(tt::tt_metal::Eth in) {
    switch (in) {
        case tt::tt_metal::Eth::SENDER: return tt::tt_metal::flatbuffer::Eth::SENDER;
        case tt::tt_metal::Eth::RECEIVER: return tt::tt_metal::flatbuffer::Eth::RECEIVER;
        case tt::tt_metal::Eth::IDLE: return tt::tt_metal::flatbuffer::Eth::IDLE;
        default: throw std::invalid_argument("Unknown Eth value in ToFlatbuffer()");
    }
}

// Original types defined in base_types.hpp
inline tt::tt_metal::flatbuffer::MathFidelity ToFlatbuffer(MathFidelity input) {
    switch (input) {
        case MathFidelity::LoFi: return tt::tt_metal::flatbuffer::MathFidelity::LoFi;
        case MathFidelity::HiFi2: return tt::tt_metal::flatbuffer::MathFidelity::HiFi2;
        case MathFidelity::HiFi3: return tt::tt_metal::flatbuffer::MathFidelity::HiFi3;
        case MathFidelity::HiFi4: return tt::tt_metal::flatbuffer::MathFidelity::HiFi4;
        case MathFidelity::Invalid: return tt::tt_metal::flatbuffer::MathFidelity::Invalid;
        default: throw std::invalid_argument("Unknown MathFidelity value in ToFlatbuffer()");
    }
}

inline tt::tt_metal::flatbuffer::UnpackToDestMode ToFlatbuffer(UnpackToDestMode input) {
    switch (input) {
        case UnpackToDestMode::UnpackToDestFp32: return tt::tt_metal::flatbuffer::UnpackToDestMode::UnpackToDestFp32;
        case UnpackToDestMode::Default: return tt::tt_metal::flatbuffer::UnpackToDestMode::Default;
        default: throw std::invalid_argument("Invalid UnpackToDestMode value passed to ToFlatbuffer");
    }
}

// Original types defined in tt_backend_api_types.hpp
inline tt::tt_metal::flatbuffer::DataFormat ToFlatbuffer(tt::DataFormat input) {
    switch (input) {
        case tt::DataFormat::Float32: return tt::tt_metal::flatbuffer::DataFormat::Float32;
        case tt::DataFormat::Float16: return tt::tt_metal::flatbuffer::DataFormat::Float16;
        case tt::DataFormat::Bfp8: return tt::tt_metal::flatbuffer::DataFormat::Bfp8;
        case tt::DataFormat::Bfp4: return tt::tt_metal::flatbuffer::DataFormat::Bfp4;
        case tt::DataFormat::Bfp2: return tt::tt_metal::flatbuffer::DataFormat::Bfp2;
        case tt::DataFormat::Float16_b: return tt::tt_metal::flatbuffer::DataFormat::Float16_b;
        case tt::DataFormat::Bfp8_b: return tt::tt_metal::flatbuffer::DataFormat::Bfp8_b;
        case tt::DataFormat::Bfp4_b: return tt::tt_metal::flatbuffer::DataFormat::Bfp4_b;
        case tt::DataFormat::Bfp2_b: return tt::tt_metal::flatbuffer::DataFormat::Bfp2_b;
        case tt::DataFormat::Lf8: return tt::tt_metal::flatbuffer::DataFormat::Lf8;
        case tt::DataFormat::Fp8_e4m3: return tt::tt_metal::flatbuffer::DataFormat::Fp8_e4m3;
        case tt::DataFormat::Int8: return tt::tt_metal::flatbuffer::DataFormat::Int8;
        case tt::DataFormat::Tf32: return tt::tt_metal::flatbuffer::DataFormat::Tf32;
        case tt::DataFormat::UInt8: return tt::tt_metal::flatbuffer::DataFormat::UInt8;
        case tt::DataFormat::UInt16: return tt::tt_metal::flatbuffer::DataFormat::UInt16;
        case tt::DataFormat::Int32: return tt::tt_metal::flatbuffer::DataFormat::Int32;
        case tt::DataFormat::UInt32: return tt::tt_metal::flatbuffer::DataFormat::UInt32;
        case tt::DataFormat::RawUInt8: return tt::tt_metal::flatbuffer::DataFormat::RawUInt8;
        case tt::DataFormat::RawUInt16: return tt::tt_metal::flatbuffer::DataFormat::RawUInt16;
        case tt::DataFormat::RawUInt32: return tt::tt_metal::flatbuffer::DataFormat::RawUInt32;
        case tt::DataFormat::Invalid: return tt::tt_metal::flatbuffer::DataFormat::Invalid;
        default: throw std::invalid_argument("Unknown DataFormat value in ToFlatbuffer()");
    }
}

inline flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile> ToFlatbuffer(
    const Tile& tile, flatbuffers::FlatBufferBuilder& builder) {
    auto tile_shape_fb = builder.CreateVector(tile.get_tile_shape().data(), tile.get_tile_shape().size());
    auto face_shape_fb = builder.CreateVector(tile.get_face_shape().data(), tile.get_face_shape().size());

    return tt::tt_metal::flatbuffer::CreateTile(
        builder,
        tile_shape_fb,
        face_shape_fb,
        tile.get_tile_hw(),
        tile.get_face_hw(),
        tile.get_num_faces(),
        tile.get_partial_face(),
        tile.get_narrow_tile(),
        tile.get_transpose_within_face(),
        tile.get_transpose_of_faces());
}

inline flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>>> ToFlatbuffer(
    const std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>& tiles, flatbuffers::FlatBufferBuilder& builder) {
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Tile>> tiles_fb;
    for (const auto& tile_opt : tiles) {
        if (tile_opt) {
            tiles_fb.push_back(ToFlatbuffer(*tile_opt, builder));
        }
    }

    return builder.CreateVector(tiles_fb);
}

}  // namespace tt::tt_metal
