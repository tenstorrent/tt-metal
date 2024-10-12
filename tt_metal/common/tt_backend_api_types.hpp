// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "tt_metal/third_party/umd/device/tt_arch_types.h"

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
inline bool is_integer_format(DataFormat format) {
    return (
        (format == DataFormat::UInt32) || (format == DataFormat::Int8) || (format == DataFormat::UInt16) ||
        (format == DataFormat::UInt8) || (format == DataFormat::Int32) || (format == DataFormat::RawUInt32) ||
        (format == DataFormat::RawUInt16) || (format == DataFormat::RawUInt8));
}

inline std::ostream& operator<<(std::ostream& os, const DataFormat& format) {
    switch (format) {
        case DataFormat::Bfp2: os << "Bfp2"; break;
        case DataFormat::Bfp2_b: os << "Bfp2_b"; break;
        case DataFormat::Bfp4: os << "Bfp4"; break;
        case DataFormat::Bfp4_b: os << "Bfp4_b"; break;
        case DataFormat::Bfp8: os << "Bfp8"; break;
        case DataFormat::Bfp8_b: os << "Bfp8_b"; break;
        case DataFormat::Float16: os << "Float16"; break;
        case DataFormat::Float16_b: os << "Float16_b"; break;
        case DataFormat::Float32: os << "Float32"; break;
        case DataFormat::Tf32: os << "Tf32"; break;
        case DataFormat::Int8: os << "Int8"; break;
        case DataFormat::UInt8: os << "UInt8"; break;
        case DataFormat::Lf8: os << "Lf8"; break;
        case DataFormat::UInt16: os << "UInt16"; break;
        case DataFormat::UInt32: os << "UInt32"; break;
        case DataFormat::Int32: os << "Int32"; break;
        case DataFormat::RawUInt8: os << "RawUInt8"; break;
        case DataFormat::RawUInt16: os << "RawUInt16"; break;
        case DataFormat::RawUInt32: os << "RawUInt32"; break;
        case DataFormat::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
}

// Size of datum in bytes
inline constexpr static uint32_t datum_size(const DataFormat& format) {
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
        case DataFormat::Int8: return 1;
        case DataFormat::Lf8: return 1;
        case DataFormat::UInt8: return 1;
        case DataFormat::UInt16: return 2;
        case DataFormat::UInt32: return 4;
        case DataFormat::RawUInt8: return 1;
        case DataFormat::RawUInt16: return 2;
        case DataFormat::Int32: return 4;
        case DataFormat::RawUInt32: return 4;
        case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
        default: throw std::invalid_argument("Unknown format");
    }
}

// Size of tile in bytes
inline constexpr static uint32_t tile_size(const DataFormat& format) {
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
        case DataFormat::Int8: return 1024;
        case DataFormat::Lf8: return 1024;
        case DataFormat::UInt8: return 1024;
        case DataFormat::UInt16: return (1024 * 2);
        case DataFormat::UInt32: return (1024 * 4);
        case DataFormat::RawUInt8: return 1024;
        case DataFormat::RawUInt16: return (1024 * 2);
        case DataFormat::Int32: return (1024 * 4);
        case DataFormat::RawUInt32: return (1024 * 4);
        case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
        default: throw std::invalid_argument("Unknown format");
    }
}

std::string get_string(ARCH arch);
std::string get_string_lowercase(ARCH arch);
std::string get_alias(ARCH arch);
ARCH get_arch_from_string(const std::string& arch_str);

enum RISCV : uint8_t {
    BRISC = 0,
    NCRISC = 1,
    TRISC0 = 2,
    TRISC1 = 3,
    TRISC2 = 4,
    ERISC = 5,
    COMPUTE = 6,  // Encompasses TRISC0, TRISC1, and TRISC2
    MAX = 7,
};

inline std::ostream& operator<<(std::ostream& os, const RISCV& riscv) {
    switch (riscv) {
        case RISCV::BRISC: os << "BRISC"; break;
        case RISCV::NCRISC: os << "NCRISC"; break;
        case RISCV::TRISC0: os << "TRISC0"; break;
        case RISCV::TRISC1: os << "TRISC1"; break;
        case RISCV::TRISC2: os << "TRISC2"; break;
        case RISCV::ERISC: os << "ERISC"; break;
        case RISCV::COMPUTE: os << "COMPUTE"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
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
