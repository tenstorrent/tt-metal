// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace dprint_detail {

namespace structures {

struct DPrintStringInfo {
    const char* format_string_ptr;
    const char* file;
    std::uint32_t line;
};

struct DPrintHeader {
    static constexpr uint16_t max_info_id_value = 1023;
    union {
        struct {
            uint8_t is_kernel : 1;  // 0 = firmware, 1 = kernel
            uint8_t risc_id : 5;    // 0-31 risc id (supporting quasar)
            uint16_t info_id : 10;  // Index into .dprint_strings_info (max 1024 entries)
        };
        uint16_t value;
    };
};

static_assert(sizeof(DPrintHeader) == sizeof(uint16_t));

}  // namespace structures

}  // namespace dprint_detail
