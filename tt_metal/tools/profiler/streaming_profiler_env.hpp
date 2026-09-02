// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared parsing for the numeric TT_METAL_STREAMING_PROFILER_* knobs. The base is auto-detected, so a
// value may be written decimal, octal or 0x-hexadecimal.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace tt::tt_metal::streaming_profiler {

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

}  // namespace tt::tt_metal::streaming_profiler
