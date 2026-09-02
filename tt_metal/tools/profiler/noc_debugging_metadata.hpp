// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(DEVICE_DEBUG_DUMP)
#include "api/debug/assert.h"
#endif

struct alignas(uint64_t) NocDebuggingEventMetadata {
    enum class NocDebugEventType : unsigned char {
        CB_LOCK = 0,
        CB_UNLOCK = 1,
        MEM_LOCK = 2,
        MEM_UNLOCK = 3,
        DFB_LOCK = 4,
        DFB_UNLOCK = 5,
        DFB_REGION_START = 6,
        DFB_REGION_CLEAR = 7,
    };

    union {
        uint64_t raw;
        struct {
            uint64_t event_type : 8;
            uint64_t locked_addr : 24;
            uint64_t locked_size : 24;
            uint64_t reserved : 8;
        };
    };

    NocDebuggingEventMetadata() : raw(0) {}

    explicit NocDebuggingEventMetadata(const uint64_t raw_data) : raw(raw_data) {}

    void setEventType(NocDebugEventType type) { event_type = static_cast<uint64_t>(type); }

    void setLockedRegion(uint32_t locked_address_base, uint32_t num_bytes) {
        constexpr uint32_t max_field_value = 0xFFFFFF;
#if defined(DEVICE_DEBUG_DUMP)
        ASSERT(locked_address_base <= max_field_value);
        ASSERT(num_bytes <= max_field_value);
#endif
        locked_addr = locked_address_base > max_field_value ? max_field_value : locked_address_base;
        locked_size = num_bytes > max_field_value ? max_field_value : num_bytes;
    }

    uint32_t getLockedAddressBase() const { return static_cast<uint32_t>(locked_addr); }
    uint32_t getNumBytes() const { return static_cast<uint32_t>(locked_size); }

    uint64_t asU64() const { return raw; }
};
static_assert(sizeof(NocDebuggingEventMetadata) == sizeof(uint64_t));
