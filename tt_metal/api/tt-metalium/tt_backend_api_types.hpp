// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <ostream>
#include <stdexcept>

#include <fmt/base.h>

#include <umd/device/types/arch.hpp>

namespace tt {

/**
 * @brief Tensix data format enum.
 * @details This enum contains the union of all data formats supported by Tensix hardware of all generations.
 * Not all data formats are supported by all hardware generations; compatibility is legality checked at runtime.
 * Architecture-specific data format enums are replicated in tensix_types.h (LLK).
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
    Fp8_e4m3 = 26,
    MxFp4 = 22,
    Int8 = 14,
    Tf32 = 4,
    UInt8 = 30,
    UInt16 = 9,
    Int16 = 13,
    Int32 = 8,
    UInt32 = 24,
    RawUInt8 = 240,
    RawUInt16 = 241,
    RawUInt32 = 242,
    Invalid = 255
};

/**
 * @brief Evaluates to true if the data format is an integer type.
 */
bool is_integer_format(DataFormat format);

/**
 * @brief Whether the data format is supported by the Tensix compute engine of a given architecture.
 */
bool is_data_format_supported(DataFormat format, ARCH arch);

std::ostream& operator<<(std::ostream& os, const DataFormat& format);

constexpr static uint32_t datum_size(const DataFormat& format) {
    switch (format) {
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b: throw std::invalid_argument("datum for bfp2, bfp4, bfp8 is invalid");
        case DataFormat::MxFp4: throw std::invalid_argument("datum for mxfp4 is invalid");
        case DataFormat::Float16:
        case DataFormat::Float16_b: return 2;
        case DataFormat::Float32: return 4;
        case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported atm");
        case DataFormat::Fp8_e4m3:
        case DataFormat::Int8:
        case DataFormat::Lf8:
        case DataFormat::UInt8:
        case DataFormat::RawUInt8: return 1;
        case DataFormat::UInt16:
        case DataFormat::Int16:
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
        case DataFormat::MxFp4: return (1024 / 2) + 32;  // 544 bytes: 32 scales (1 per 32-elem block) + 512 data
        case DataFormat::Float16:
        case DataFormat::Float16_b: return (1024 * 2);
        case DataFormat::Float32: return (1024 * 4);
        case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported atm");
        case DataFormat::Fp8_e4m3:
        case DataFormat::Int8:
        case DataFormat::Lf8:
        case DataFormat::UInt8:
        case DataFormat::RawUInt8: return 1024;
        case DataFormat::UInt16:
        case DataFormat::Int16:
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
