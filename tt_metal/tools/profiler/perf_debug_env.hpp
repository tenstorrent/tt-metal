// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared truthiness/parsing for the TT_METAL_PERF_DEBUG_* knobs. Numbers decide by value
// ("0"/"0x0" off, "01" on) and falsy words are honoured -- the old `*s != '0'` idiom read
// "=false" and "=off" as enabled.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace tt::tt_metal::perf_debug {

inline bool env_flag(const char* name, bool default_value = false) {
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') {
        return default_value;
    }
    std::string v(s);
    for (char& c : v) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (v == "false" || v == "off" || v == "no" || v == "n" || v == "disable" || v == "disabled") {
        return false;
    }
    char* end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 0);
    if (end != nullptr && *end == '\0') {
        return n != 0;
    }
    return true;
}

inline uint64_t env_u64(const char* name, uint64_t default_value) {
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') {
        return default_value;
    }
    return std::strtoull(s, nullptr, 0);
}

inline uint32_t env_u32(const char* name, uint32_t default_value) {
    return static_cast<uint32_t>(env_u64(name, default_value));
}

}  // namespace tt::tt_metal::perf_debug
