// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace device_print_detail::structures {

struct DevicePrintStringInfo {
    const char* format_string_ptr;
    const char* file;
    std::size_t line;
    std::size_t padding;  // Padding to ensure the structure size is a power of two for easier division on device.
};

static_assert(sizeof(DevicePrintStringInfo) == 4 * sizeof(std::size_t), "DevicePrintStringInfo size must be correct");

// Parsed format string info for 32-bit ELF.
struct DevicePrintStringInfo32 {
    uint32_t format_string_ptr;
    uint32_t file;
    uint32_t line;
    uint32_t padding;  // Padding to ensure the structure size is a power of two for easier division on device.
};

static_assert(sizeof(DevicePrintStringInfo32) == 4 * sizeof(uint32_t), "DevicePrintStringInfo32 size must be correct");
static_assert(
    sizeof(const char*) != sizeof(uint32_t) || sizeof(DevicePrintStringInfo) == sizeof(DevicePrintStringInfo32),
    "DevicePrintStringInfo size must match pointer size");

// Parsed format string info for 64-bit ELF.
struct DevicePrintStringInfo64 {
    uint64_t format_string_ptr;
    uint64_t file;
    uint64_t line;
    uint64_t padding;  // Padding to ensure the structure size is a power of two for easier division on device.
};

static_assert(sizeof(DevicePrintStringInfo64) == 4 * sizeof(uint64_t), "DevicePrintStringInfo64 size must be correct");
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

}  // namespace device_print_detail::structures

namespace device_print_dispatch {

struct NocLocationInputInfo {
    uint16_t x : 6;
    uint16_t y : 6;
    uint64_t rw_ptr_addr : 52;
    uint16_t buf_offset;
    uint16_t buf_size;
} __attribute__((packed, aligned(4)));

static_assert(sizeof(NocLocationInputInfo) == 12, "NocLocationInputInfo must be 12 bytes");
static_assert(sizeof(NocLocationInputInfo) % 4 == 0, "NocLocationInputInfo must be 4-byte aligned");

struct DramStreamMessageHeader {
    uint16_t x : 6;
    uint16_t y : 6;
    uint16_t align : 6;
    uint16_t buffer_wrapped : 1;
    uint16_t length : 13;
} __attribute__((packed, aligned(4)));

static_assert(sizeof(DramStreamMessageHeader) == 4, "DramStreamMessageHeader must be 4 bytes");

}  // namespace device_print_dispatch
