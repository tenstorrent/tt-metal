// SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

struct alignas(uint64_t) NocDebuggingEventMetadata {
    enum class NocDebugEventType : unsigned char {
        CB_LOCK = 0,
        CB_UNLOCK = 1,
        MEM_LOCK = 2,
        MEM_UNLOCK = 3,
    };

    union {
        uint64_t raw;
        struct {
            uint64_t event_type : 8;
            uint64_t locked_addr_16b : 16;
            uint64_t size_16b : 16;
            uint64_t reserved : 24;
        };
    };

    NocDebuggingEventMetadata() : raw(0) {}

    explicit NocDebuggingEventMetadata(const uint64_t raw_data) : raw(raw_data) {}

    void setEventType(NocDebugEventType type) { event_type = static_cast<uint64_t>(type); }

    void setLockedRegion(uint32_t locked_address_base, uint32_t num_bytes) {
        locked_addr_16b = locked_address_base >> 4;
        size_16b = num_bytes >> 4;
    }

    uint32_t getLockedAddressBase() const { return static_cast<uint32_t>(locked_addr_16b) << 4; }
    uint32_t getNumBytes() const { return static_cast<uint32_t>(size_16b) << 4; }

    uint64_t asU64() const { return raw; }
};
static_assert(sizeof(NocDebuggingEventMetadata) == sizeof(uint64_t));
