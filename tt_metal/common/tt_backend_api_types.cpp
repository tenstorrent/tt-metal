// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tt_backend_api_types.hpp>

#include <fmt/base.h>
#include <enchantum/enchantum.hpp>
#include <string_view>

bool tt::is_integer_format(DataFormat format) {
    return (
        (format == DataFormat::UInt32) || (format == DataFormat::Int8) || (format == DataFormat::UInt16) ||
        (format == DataFormat::UInt8) || (format == DataFormat::Int32) || (format == DataFormat::RawUInt32) ||
        (format == DataFormat::RawUInt16) || (format == DataFormat::RawUInt8));
}

std::ostream& tt::operator<<(std::ostream& os, const DataFormat& format) {
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
        case DataFormat::Fp8_e4m3: os << "Fp8_e4m3"; break;
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

auto fmt::formatter<tt::DataFormat>::format(tt::DataFormat df, format_context& ctx) const -> format_context::iterator {
    const auto name = enchantum::to_string(df);

    if (name.empty()) {
        throw std::invalid_argument("Unknown format");
    }
    return formatter<string_view>::format(name, ctx);
}
