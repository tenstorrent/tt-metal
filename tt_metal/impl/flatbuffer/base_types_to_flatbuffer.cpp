// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/base_types_to_flatbuffer.hpp"

namespace tt::tt_metal {

// Original types defined in data_types.hpp
flatbuffer::DataMovementProcessor to_flatbuffer(DataMovementProcessor in) {
    switch (in) {
        case DataMovementProcessor::RISCV_0: return flatbuffer::DataMovementProcessor::RISCV_0;
        case DataMovementProcessor::RISCV_1: return flatbuffer::DataMovementProcessor::RISCV_1;
    }
    TT_THROW("Unsupported DataMovementProcessor to flatbuffer.");
}

flatbuffer::NOC to_flatbuffer(NOC in) {
    switch (in) {
        case NOC::NOC_0: return flatbuffer::NOC::NOC_0;
        case NOC::NOC_1: return flatbuffer::NOC::NOC_1;
    }
    TT_THROW("Unsupported NOC to flatbuffer.");
}

flatbuffer::NOC_MODE to_flatbuffer(NOC_MODE in) {
    switch (in) {
        case NOC_MODE::DM_DEDICATED_NOC: return flatbuffer::NOC_MODE::DM_DEDICATED_NOC;
        case NOC_MODE::DM_DYNAMIC_NOC: return flatbuffer::NOC_MODE::DM_DYNAMIC_NOC;
    }
    TT_THROW("Unsupported NOC_MODE to flatbuffer.");
}

flatbuffer::EthMode to_flatbuffer(Eth in) {
    switch (in) {
        case Eth::SENDER: return flatbuffer::EthMode::SENDER;
        case Eth::RECEIVER: return flatbuffer::EthMode::RECEIVER;
        case Eth::IDLE: return flatbuffer::EthMode::IDLE;
    }
    TT_THROW("Unsupported Eth to flatbuffer.");
}

// Original types defined in base_types.hpp
flatbuffer::MathFidelity to_flatbuffer(MathFidelity input) {
    switch (input) {
        case MathFidelity::LoFi: return flatbuffer::MathFidelity::LoFi;
        case MathFidelity::HiFi2: return flatbuffer::MathFidelity::HiFi2;
        case MathFidelity::HiFi3: return flatbuffer::MathFidelity::HiFi3;
        case MathFidelity::HiFi4: return flatbuffer::MathFidelity::HiFi4;
        case MathFidelity::Invalid: return flatbuffer::MathFidelity::Invalid;
    }
    TT_THROW("Unsupported MathFidelity to flatbuffer.");
}

flatbuffer::UnpackToDestMode to_flatbuffer(UnpackToDestMode input) {
    switch (input) {
        case UnpackToDestMode::UnpackToDestFp32: return flatbuffer::UnpackToDestMode::UnpackToDestFp32;
        case UnpackToDestMode::Default: return flatbuffer::UnpackToDestMode::Default;
    }
    TT_THROW("Unsupported UnpackToDestMode to flatbuffer.");
}

// Original types defined in tt_backend_api_types.hpp
flatbuffer::DataFormat to_flatbuffer(tt::DataFormat input) {
    switch (input) {
        case tt::DataFormat::Float32: return flatbuffer::DataFormat::Float32;
        case tt::DataFormat::Float16: return flatbuffer::DataFormat::Float16;
        case tt::DataFormat::Bfp8: return flatbuffer::DataFormat::Bfp8;
        case tt::DataFormat::Bfp4: return flatbuffer::DataFormat::Bfp4;
        case tt::DataFormat::Bfp2: return flatbuffer::DataFormat::Bfp2;
        case tt::DataFormat::Float16_b: return flatbuffer::DataFormat::Float16_b;
        case tt::DataFormat::Bfp8_b: return flatbuffer::DataFormat::Bfp8_b;
        case tt::DataFormat::Bfp4_b: return flatbuffer::DataFormat::Bfp4_b;
        case tt::DataFormat::Bfp2_b: return flatbuffer::DataFormat::Bfp2_b;
        case tt::DataFormat::Lf8: return flatbuffer::DataFormat::Lf8;
        case tt::DataFormat::Fp8_e4m3: return flatbuffer::DataFormat::Fp8_e4m3;
        case tt::DataFormat::Int8: return flatbuffer::DataFormat::Int8;
        case tt::DataFormat::Tf32: return flatbuffer::DataFormat::Tf32;
        case tt::DataFormat::UInt8: return flatbuffer::DataFormat::UInt8;
        case tt::DataFormat::UInt16: return flatbuffer::DataFormat::UInt16;
        case tt::DataFormat::Int32: return flatbuffer::DataFormat::Int32;
        case tt::DataFormat::UInt32: return flatbuffer::DataFormat::UInt32;
        case tt::DataFormat::RawUInt8: return flatbuffer::DataFormat::RawUInt8;
        case tt::DataFormat::RawUInt16: return flatbuffer::DataFormat::RawUInt16;
        case tt::DataFormat::RawUInt32: return flatbuffer::DataFormat::RawUInt32;
        case tt::DataFormat::Invalid: return flatbuffer::DataFormat::Invalid;
    }
    TT_THROW("Unsupported DataFormat to flatbuffer.");
}

flatbuffer::Tile to_flatbuffer(const Tile& tile) {
    TT_FATAL(tile.get_tile_shape().size() == 2, "Conversion to Flatbuffer expecting 2D Tile Shapes.");
    std::array<uint32_t, 2> shape = {tile.get_tile_shape()[0], tile.get_tile_shape()[1]};

    return flatbuffer::Tile(
        flatbuffers::span<const uint32_t, 2>(shape), tile.get_transpose_within_face() && tile.get_transpose_of_faces());
}

}  // namespace tt::tt_metal
