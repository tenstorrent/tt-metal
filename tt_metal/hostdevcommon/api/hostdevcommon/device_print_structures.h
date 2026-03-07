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
    std::size_t line;
};

static_assert(sizeof(DevicePrintStringInfo) == 3 * sizeof(std::size_t), "DevicePrintStringInfo size must be correct");

// Parsed format string info for 32-bit ELF.
struct DevicePrintStringInfo32 {
    uint32_t format_string_ptr;
    uint32_t file;
    uint32_t line;
};

static_assert(sizeof(DevicePrintStringInfo32) == 3 * sizeof(uint32_t), "DevicePrintStringInfo32 size must be correct");
static_assert(
    sizeof(const char*) != sizeof(uint32_t) || sizeof(DevicePrintStringInfo) == sizeof(DevicePrintStringInfo32),
    "DevicePrintStringInfo size must match pointer size");

// Parsed format string info for 64-bit ELF.
struct DevicePrintStringInfo64 {
    uint64_t format_string_ptr;
    uint64_t file;
    uint64_t line;
};

static_assert(sizeof(DevicePrintStringInfo64) == 3 * sizeof(uint64_t), "DevicePrintStringInfo64 size must be correct");
static_assert(
    sizeof(const char*) != sizeof(uint64_t) || sizeof(DevicePrintStringInfo) == sizeof(DevicePrintStringInfo64),
    "DevicePrintStringInfo size must match pointer size");

struct DevicePrintHeader {
    static constexpr uint32_t max_info_id_value = 65535;
    static constexpr uint32_t max_message_payload_size = 1023;  // 10 bits for message_payload
    union {
        struct {
            uint32_t is_kernel : 1;         // 0 = firmware, 1 = kernel
            uint32_t risc_id : 5;           // 0-31 risc id (supporting quasar)
            uint32_t message_payload : 10;  // Message payload size (<1024 bytes)
            uint32_t info_id : 16;          // Index into .device_print_strings_info (max 65536 entries)
        };
        uint32_t value;
    };
};

static_assert(sizeof(DevicePrintHeader) == sizeof(uint32_t));

}  // namespace structures

}  // namespace device_print_detail
