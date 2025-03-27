// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer/base_types_from_flatbuffer.hpp"
#include "assert.hpp"

namespace tt::tt_metal {

DataMovementProcessor from_flatbuffer(flatbuffer::DataMovementProcessor in) {
    switch (in) {
        case flatbuffer::DataMovementProcessor::RISCV_0: return DataMovementProcessor::RISCV_0;
        case flatbuffer::DataMovementProcessor::RISCV_1: return DataMovementProcessor::RISCV_1;
    }
    TT_THROW("Unsupported DataMovementProcessor from flatbuffer.");
}

NOC from_flatbuffer(flatbuffer::NOC in) {
    switch (in) {
        case flatbuffer::NOC::NOC_0: return NOC::NOC_0;
        case flatbuffer::NOC::NOC_1: return NOC::NOC_1;
    }
    TT_THROW("Unsupported NOC from flatbuffer.");
}

NOC_MODE from_flatbuffer(flatbuffer::NOC_MODE in) {
    switch (in) {
        case flatbuffer::NOC_MODE::DM_DEDICATED_NOC: return NOC_MODE::DM_DEDICATED_NOC;
        case flatbuffer::NOC_MODE::DM_DYNAMIC_NOC: return NOC_MODE::DM_DYNAMIC_NOC;
    }
    TT_THROW("Unsupported NOC_MODE from flatbuffer.");
}

Eth from_flatbuffer(flatbuffer::EthMode in) {
    switch (in) {
        case flatbuffer::EthMode::SENDER: return Eth::SENDER;
        case flatbuffer::EthMode::RECEIVER: return Eth::RECEIVER;
        case flatbuffer::EthMode::IDLE: return Eth::IDLE;
    }
    TT_THROW("Unsupported EthMode from flatbuffer.");
}

MathFidelity from_flatbuffer(flatbuffer::MathFidelity input) {
    switch (input) {
        case flatbuffer::MathFidelity::LoFi: return MathFidelity::LoFi;
        case flatbuffer::MathFidelity::HiFi2: return MathFidelity::HiFi2;
        case flatbuffer::MathFidelity::HiFi3: return MathFidelity::HiFi3;
        case flatbuffer::MathFidelity::HiFi4: return MathFidelity::HiFi4;
        case flatbuffer::MathFidelity::Invalid: return MathFidelity::Invalid;
    }
    TT_THROW("Unsupported MathFidelity from flatbuffer.");
}

UnpackToDestMode from_flatbuffer(flatbuffer::UnpackToDestMode input) {
    switch (input) {
        case flatbuffer::UnpackToDestMode::UnpackToDestFp32: return UnpackToDestMode::UnpackToDestFp32;
        case flatbuffer::UnpackToDestMode::Default: return UnpackToDestMode::Default;
    }
    TT_THROW("Unsupported UnpackToDestMode from flatbuffer.");
}

tt::DataFormat from_flatbuffer(flatbuffer::DataFormat input) {
    switch (input) {
        case flatbuffer::DataFormat::Float32: return tt::DataFormat::Float32;
        case flatbuffer::DataFormat::Float16: return tt::DataFormat::Float16;
        case flatbuffer::DataFormat::Bfp8: return tt::DataFormat::Bfp8;
        case flatbuffer::DataFormat::Bfp4: return tt::DataFormat::Bfp4;
        case flatbuffer::DataFormat::Bfp2: return tt::DataFormat::Bfp2;
        case flatbuffer::DataFormat::Float16_b: return tt::DataFormat::Float16_b;
        case flatbuffer::DataFormat::Bfp8_b: return tt::DataFormat::Bfp8_b;
        case flatbuffer::DataFormat::Bfp4_b: return tt::DataFormat::Bfp4_b;
        case flatbuffer::DataFormat::Bfp2_b: return tt::DataFormat::Bfp2_b;
        case flatbuffer::DataFormat::Lf8: return tt::DataFormat::Lf8;
        case flatbuffer::DataFormat::Fp8_e4m3: return tt::DataFormat::Fp8_e4m3;
        case flatbuffer::DataFormat::Int8: return tt::DataFormat::Int8;
        case flatbuffer::DataFormat::Tf32: return tt::DataFormat::Tf32;
        case flatbuffer::DataFormat::UInt8: return tt::DataFormat::UInt8;
        case flatbuffer::DataFormat::UInt16: return tt::DataFormat::UInt16;
        case flatbuffer::DataFormat::Int32: return tt::DataFormat::Int32;
        case flatbuffer::DataFormat::UInt32: return tt::DataFormat::UInt32;
        case flatbuffer::DataFormat::RawUInt8: return tt::DataFormat::RawUInt8;
        case flatbuffer::DataFormat::RawUInt16: return tt::DataFormat::RawUInt16;
        case flatbuffer::DataFormat::RawUInt32: return tt::DataFormat::RawUInt32;
        case flatbuffer::DataFormat::Invalid: return tt::DataFormat::Invalid;
    }
    TT_THROW("Unsupported DataFormat from flatbuffer.");
}

Tile from_flatbuffer(const flatbuffer::Tile& tile_fb) {
    const auto& shape = *tile_fb.tile_shape();
    // Tile shape is already 2D in flatbuffer schema.
    std::array<uint32_t, 2> tile_shape = {shape[0], shape[1]};
    bool transpose_tile = tile_fb.transpose_tile();
    return Tile(tile_shape, transpose_tile);
}

}  // namespace tt::tt_metal
