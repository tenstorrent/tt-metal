// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_backend_api_types.hpp"

std::string tt::get_string(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::JAWBRIDGE: return "JAWBRIDGE"; break;
        case tt::ARCH::GRAYSKULL: return "GRAYSKULL"; break;
        case tt::ARCH::WORMHOLE: return "WORMHOLE"; break;
        case tt::ARCH::WORMHOLE_B0: return "WORMHOLE_B0"; break;
        case tt::ARCH::BLACKHOLE: return "BLACKHOLE"; break;
        case tt::ARCH::Invalid: return "Invalid"; break;
        default: return "Invalid"; break;
    }
}

std::string tt::get_string_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::JAWBRIDGE: return "jawbridge"; break;
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE: return "wormhole"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole_b0"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        case tt::ARCH::Invalid: return "invalid"; break;
        default: return "invalid"; break;
    }
}

tt::ARCH tt::get_arch_from_string(const std::string &arch_str) {
    tt::ARCH arch;

    if ((arch_str == "jawbridge") || (arch_str == "JAWBRIDGE")) {
        arch = tt::ARCH::JAWBRIDGE;
    } else if ((arch_str == "grayskull") || (arch_str == "GRAYSKULL")) {
        arch = tt::ARCH::GRAYSKULL;
    } else if ((arch_str == "wormhole") || (arch_str == "WORMHOLE")) {
        arch = tt::ARCH::WORMHOLE;
    } else if ((arch_str == "wormhole_b0") || (arch_str == "WORMHOLE_B0")) {
        arch = tt::ARCH::WORMHOLE_B0;
    } else if ((arch_str == "blackhole") || (arch_str == "BLACKHOLE")) {
        arch = tt::ARCH::BLACKHOLE;
    } else if ((arch_str == "Invalid") || (arch_str == "INVALID")) {
        arch = tt::ARCH::Invalid;
    } else {
        throw std::runtime_error(arch_str + " is not recognized as tt::ARCH.");
    }

    return arch;
}
