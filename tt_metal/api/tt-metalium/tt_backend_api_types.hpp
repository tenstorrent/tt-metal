// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>

#include <fmt/base.h>

namespace tt {

/**
 * @brief Tensix data format enum, used across the entire SW+HW stack.
 * This enum is also replicated in tensix_types.h (LLK), but we don't want to include it here since the other data types
 * are not relevant.
 */
enum class DataFormat : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Bfp8 = 2,
    Bfp4 = 3,
    Bfp2 = 11,
    Float16_b = 5,
    Bfp8_b = 6,
    Bfp4_b = 7,
    Bfp2_b = 15,
    Lf8 = 10,
    Fp8_e4m3 = 0x1A,
    Int8 = 14,
    Tf32 = 4,
    UInt8 = 30,
    UInt16 = 9,
    Int32 = 8,
    UInt32 = 24,
    RawUInt8 = 0xf0,
    RawUInt16 = 0xf1,
    RawUInt32 = 0xf2,
    Invalid = 0xff
};

/**
 * @brief Evaluates to true if the data format is an integer type.
 */
bool is_integer_format(DataFormat format);

std::ostream& operator<<(std::ostream& os, const DataFormat& format);

constexpr uint32_t datum_size(const DataFormat& format) {
    switch (format) {
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b: throw std::invalid_argument("datum for bfp2, bfp4, bfp8 is invalid");
        case DataFormat::Float16:
        case DataFormat::Float16_b: return 2;
        case DataFormat::Float32: return 4;
        case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported atm");
        case DataFormat::Int8:
        case DataFormat::Lf8:
        case DataFormat::UInt8:
        case DataFormat::RawUInt8: return 1;
        case DataFormat::UInt16:
        case DataFormat::RawUInt16: return 2;
        case DataFormat::UInt32:
        case DataFormat::Int32:
        case DataFormat::RawUInt32: return 4;
        case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
        default: throw std::invalid_argument("Unknown format");
    }
}

/**
 * Returns tile size of given data format in bytes
 *
 * Return value: uint32_t
 *
 * | Argument    | Description    | Type                | Valid Range | Required |
 * |-------------|----------------|---------------------|-------------|----------|
 * | format      | Format of data | tt::DataFormat enum |             | Yes      |
 */
constexpr static uint32_t tile_size(const DataFormat& format) {
    switch (format) {
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b: return (64 * 4) + (16 * 4);
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b: return (128 * 4) + (16 * 4);
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b: return (256 * 4) + (16 * 4);
        case DataFormat::Float16:
        case DataFormat::Float16_b: return (1024 * 2);
        case DataFormat::Float32: return (1024 * 4);
        case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported atm");
        case DataFormat::Int8:
        case DataFormat::Lf8:
        case DataFormat::UInt8:
        case DataFormat::RawUInt8: return 1024;
        case DataFormat::UInt16:
        case DataFormat::RawUInt16: return (1024 * 2);
        case DataFormat::UInt32:
        case DataFormat::Int32:
        case DataFormat::RawUInt32: return (1024 * 4);
        case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
        default: throw std::invalid_argument("Unknown format");
    }
}

}  // namespace tt

template <>
struct std::hash<tt::DataFormat> {
    std::size_t operator()(tt::DataFormat const& obj) const noexcept { return static_cast<std::size_t>(obj); }
};

template <>
struct fmt::formatter<tt::DataFormat> : formatter<string_view> {
    auto format(tt::DataFormat df, format_context& ctx) const -> format_context::iterator;
};
