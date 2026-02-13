// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace device_print_detail {

namespace structures {

struct DevicePrintStringInfo {
    const char* format_string_ptr;
    const char* file;
    std::uint32_t line;
} __attribute__((packed));

struct DevicePrintHeader {
    static constexpr uint16_t max_info_id_value = 65535;
    static constexpr uint16_t max_message_payload_size = 1023;  // 10 bits for message_payload
    union {
        struct {
            uint8_t is_kernel : 1;          // 0 = firmware, 1 = kernel
            uint8_t risc_id : 5;            // 0-31 risc id (supporting quasar)
            uint16_t message_payload : 10;  // Message payload size (<1024 bytes)
            uint16_t info_id : 16;          // Index into .device_print_strings_info (max 65536 entries)
        } __attribute__((packed));
        uint32_t value;
    };
} __attribute__((packed));

static_assert(sizeof(DevicePrintHeader) == sizeof(uint32_t));

}  // namespace structures

}  // namespace device_print_detail
