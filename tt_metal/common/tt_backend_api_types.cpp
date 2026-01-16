// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/tt_backend_api_types.hpp"

#include <fmt/base.h>
#include <enchantum/enchantum.hpp>
#include <string_view>

std::string tt::get_string(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "GRAYSKULL"; break;
        case tt::ARCH::WORMHOLE_B0: return "WORMHOLE_B0"; break;
        case tt::ARCH::BLACKHOLE: return "BLACKHOLE"; break;
        case tt::ARCH::Invalid:
        default: return "Invalid"; break;
    }
}

std::string tt::get_string_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole_b0"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        case tt::ARCH::Invalid:
        default: return "invalid"; break;
    }
}

std::string tt::get_alias(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        default: return "invalid"; break;
    }
}

tt::ARCH tt::get_arch_from_string(const std::string& arch_str) {
    tt::ARCH arch;
    if ((arch_str == "grayskull") || (arch_str == "GRAYSKULL")) {
        arch = tt::ARCH::GRAYSKULL;
    } else if ((arch_str == "wormhole_b0") || (arch_str == "WORMHOLE_B0")) {
        arch = tt::ARCH::WORMHOLE_B0;
    } else if ((arch_str == "blackhole") || (arch_str == "BLACKHOLE")) {
        arch = tt::ARCH::BLACKHOLE;
    } else if ((arch_str == "quasar") || (arch_str == "QUASAR")) {
        arch = tt::ARCH::QUASAR;
    } else if ((arch_str == "Invalid") || (arch_str == "INVALID")) {
        arch = tt::ARCH::Invalid;
    } else {
        throw std::runtime_error(arch_str + " is not recognized as tt::ARCH.");
    }

    return arch;
}

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
