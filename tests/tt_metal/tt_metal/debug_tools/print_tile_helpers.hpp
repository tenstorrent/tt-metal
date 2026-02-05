// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/data_types.hpp>
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

inline std::string GenerateExpectedData(tt::DataFormat data_format, std::vector<uint32_t>& input_tile) {
    std::string data;
    if (data_format == tt::DataFormat::Float32) {
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                *reinterpret_cast<float*>(&input_tile[(col * 32) + 0]),
                *reinterpret_cast<float*>(&input_tile[(col * 32) + 8]),
                *reinterpret_cast<float*>(&input_tile[(col * 32) + 16]),
                *reinterpret_cast<float*>(&input_tile[(col * 32) + 24]));
        }
    } else if (data_format == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> fp16b_vec = unpack_uint32_vec_into_bfloat16_vec(input_tile);
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                static_cast<float>(fp16b_vec[(col * 32) + 0]),
                static_cast<float>(fp16b_vec[(col * 32) + 8]),
                static_cast<float>(fp16b_vec[(col * 32) + 16]),
                static_cast<float>(fp16b_vec[(col * 32) + 24]));
        }
    } else if (data_format == tt::DataFormat::Bfp8_b) {
        std::vector<float> float_vec = unpack_bfp8_tiles_into_float_vec(input_tile, true, false);
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 0]),
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 8]),
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 16]),
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 24]));
        }
    } else if (data_format == tt::DataFormat::Bfp4_b) {
        std::vector<float> float_vec = unpack_bfp4_tiles_into_float_vec(input_tile, true, false);
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 0]),
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 8]),
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 16]),
                *reinterpret_cast<float*>(&float_vec[(col * 32) + 24]));
        }
    } else if (data_format == tt::DataFormat::Int8) {
        int8_t* int8_ptr = reinterpret_cast<int8_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                int8_ptr[(col * 32) + 0],
                int8_ptr[(col * 32) + 8],
                int8_ptr[(col * 32) + 16],
                int8_ptr[(col * 32) + 24]);
        }
    } else if (data_format == tt::DataFormat::UInt8) {
        uint8_t* uint8_ptr = reinterpret_cast<uint8_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                uint8_ptr[(col * 32) + 0],
                uint8_ptr[(col * 32) + 8],
                uint8_ptr[(col * 32) + 16],
                uint8_ptr[(col * 32) + 24]);
        }
    } else if (data_format == tt::DataFormat::UInt16) {
        uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                uint16_ptr[(col * 32) + 0],
                uint16_ptr[(col * 32) + 8],
                uint16_ptr[(col * 32) + 16],
                uint16_ptr[(col * 32) + 24]);
        }
    } else if (data_format == tt::DataFormat::Int32) {
        int32_t* int32_ptr = reinterpret_cast<int32_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                int32_ptr[(col * 32) + 0],
                int32_ptr[(col * 32) + 8],
                int32_ptr[(col * 32) + 16],
                int32_ptr[(col * 32) + 24]);
        }
    } else if (data_format == tt::DataFormat::UInt32) {
        uint32_t* uint32_ptr = reinterpret_cast<uint32_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                uint32_ptr[(col * 32) + 0],
                uint32_ptr[(col * 32) + 8],
                uint32_ptr[(col * 32) + 16],
                uint32_ptr[(col * 32) + 24]);
        }
    }
    return data;
}

}  // namespace tt::tt_metal::test::dprint
