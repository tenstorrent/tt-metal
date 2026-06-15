// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt_stl/span.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

namespace tt::tt_metal::test::dprint {

inline std::vector<uint32_t> GenerateInputTile(tt::DataFormat data_format) {
    constexpr uint32_t elements_in_tile = 32 * 32;

    std::vector<uint32_t> u32_vec;
    if (data_format == tt::DataFormat::Float32) {
        u32_vec.resize(elements_in_tile);
        for (int i = 0; i < u32_vec.size(); i++) {
            float val = -12.3345 + static_cast<float>(i);  // Rebias to force some negative #s to be printed
            u32_vec.at(i) = *reinterpret_cast<uint32_t*>(&val);
        }
    } else if (data_format == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> fp16b_vec(elements_in_tile);
        for (int i = 0; i < fp16b_vec.size(); i++) {
            uint16_t val = 0x3dfb + i;  // Start at some known value (~0.1226) and increment for new numbers
            fp16b_vec[i] = bfloat16(val);
        }
        u32_vec = pack_bfloat16_vec_into_uint32_vec(fp16b_vec);
    } else if (data_format == tt::DataFormat::Bfp8_b) {
        std::vector<float> float_vec(elements_in_tile);
        for (int i = 0; i < float_vec.size(); i++) {
            float_vec[i] = 0.012345 * i * (i % 32 == 0 ? -1 : 1);  // Small increments and force negatives for testing
        }
        u32_vec = pack_as_bfp8_tiles(ttsl::make_const_span(float_vec), true, false);
    } else if (data_format == tt::DataFormat::Bfp4_b) {
        std::vector<float> float_vec(elements_in_tile);
        for (int i = 0; i < float_vec.size(); i++) {
            float_vec[i] = 0.012345 * i * (i % 16 == 0 ? -1 : 1);  // Small increments and force negatives for testing
        }
        u32_vec = pack_as_bfp4_tiles(ttsl::make_const_span(float_vec), true, false);
    } else if (data_format == tt::DataFormat::Int8) {
        std::vector<int8_t> int8_vec(elements_in_tile);
        for (int i = 0; i < int8_vec.size(); i++) {
            int8_vec[i] = ((i / 2) % 256) - 128;  // Force prints to be different (/2), within the int8 range (%256),
                                                  // and include negatives (-128) for testing purposes.
        }
        uint32_t datums_per_32 = sizeof(uint32_t) / tt::datum_size(data_format);
        u32_vec.resize(elements_in_tile / datums_per_32);
        std::memcpy(u32_vec.data(), int8_vec.data(), elements_in_tile * sizeof(int8_t));
    } else if (data_format == tt::DataFormat::UInt8) {
        std::vector<uint8_t> uint8_vec(elements_in_tile);
        for (int i = 0; i < uint8_vec.size(); i++) {
            uint8_vec[i] = ((i / 2) % 256);  // Same as int8, just no negatives
        }
        uint32_t datums_per_32 = sizeof(uint32_t) / tt::datum_size(data_format);
        u32_vec.resize(elements_in_tile / datums_per_32);
        std::memcpy(u32_vec.data(), uint8_vec.data(), elements_in_tile * sizeof(uint8_t));
    } else if (data_format == tt::DataFormat::UInt16) {
        std::vector<uint16_t> uint16_vec(elements_in_tile);
        for (int i = 0; i < uint16_vec.size(); i++) {
            uint16_vec[i] = (i % 0x10000);  // Force to within uint16 range
        }
        uint32_t datums_per_32 = sizeof(uint32_t) / tt::datum_size(data_format);
        u32_vec.resize(elements_in_tile / datums_per_32);
        std::memcpy(u32_vec.data(), uint16_vec.data(), elements_in_tile * sizeof(uint16_t));
    } else if (data_format == tt::DataFormat::Int32) {
        u32_vec.resize(elements_in_tile);
        for (int i = 0; i < u32_vec.size(); i++) {
            u32_vec[i] = (i % 2) ? i : i * -1;  // Make every other number negative for printing purposes
        }
    } else if (data_format == tt::DataFormat::UInt32) {
        u32_vec.resize(elements_in_tile);
        for (int i = 0; i < u32_vec.size(); i++) {
            u32_vec[i] = i;
        }
    }
    return u32_vec;
}

inline std::vector<uint32_t> GenerateInputTileWithOffset(tt::DataFormat data_format, uint32_t offset) {
    std::vector<uint32_t> u32_vec = GenerateInputTile(data_format);
    for (unsigned int& i : u32_vec) {
        i += offset;
    }
    return u32_vec;
}

template <typename T>
inline std::string FormatTileData(const T* data) {
    std::string out;
    for (uint32_t col = 0; col < 32; col += 8) {
        out += fmt::format(
            "\n{} {} {} {}", data[(col * 32) + 0], data[(col * 32) + 8], data[(col * 32) + 16], data[(col * 32) + 24]);
    }
    return out;
}

inline std::string GenerateExpectedData(tt::DataFormat data_format, std::vector<uint32_t>& input_tile) {
    switch (data_format) {
        case tt::DataFormat::Float32: return FormatTileData(reinterpret_cast<const float*>(input_tile.data()));
        case tt::DataFormat::Float16_b: {
            std::vector<bfloat16> fp16b_vec = unpack_uint32_vec_into_bfloat16_vec(input_tile);
            std::vector<float> float_vec(fp16b_vec.size());
            for (size_t i = 0; i < fp16b_vec.size(); ++i) {
                float_vec[i] = static_cast<float>(fp16b_vec[i]);
            }
            return FormatTileData(float_vec.data());
        }
        case tt::DataFormat::Bfp8_b: {
            std::vector<float> float_vec = unpack_bfp8_tiles_into_float_vec(input_tile, true, false);
            return FormatTileData(float_vec.data());
        }
        case tt::DataFormat::Bfp4_b: {
            std::vector<float> float_vec = unpack_bfp4_tiles_into_float_vec(input_tile, true, false);
            return FormatTileData(float_vec.data());
        }
        case tt::DataFormat::Int8: return FormatTileData(reinterpret_cast<const int8_t*>(input_tile.data()));
        case tt::DataFormat::UInt8: return FormatTileData(reinterpret_cast<const uint8_t*>(input_tile.data()));
        case tt::DataFormat::UInt16: return FormatTileData(reinterpret_cast<const uint16_t*>(input_tile.data()));
        case tt::DataFormat::Int32: return FormatTileData(reinterpret_cast<const int32_t*>(input_tile.data()));
        case tt::DataFormat::UInt32: return FormatTileData(reinterpret_cast<const uint32_t*>(input_tile.data()));
        default: return {};
    }
}

}  // namespace tt::tt_metal::test::dprint
