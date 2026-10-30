// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal::mimir_test {

// instance is the GDDR channel, tile * 4 + slice.
struct MimirChannelLocation {
    uint32_t tile = 0;
    uint32_t slice = 0;
    uint32_t instance = 0;
};

// GDDR channel of the 128-byte-aligned block that holds a Mimir chiplet offset.
inline MimirChannelLocation mimir_channel_location(uint64_t chiplet_offset) {
    constexpr uint64_t kAddressStart = 0x8'0000'0000ULL;
    constexpr uint64_t kAddressSpan = 0x8'0000'0000ULL;  // 2^35
    constexpr uint64_t kHalfBytes = 0x1'0000'0000ULL;    // 4 GiB
    constexpr uint64_t kTileBit = 1ULL << 34;

    const uint64_t address = kAddressStart + chiplet_offset;
    uint64_t realigned = address;
    if (address >= kAddressStart + kHalfBytes) {
        const uint64_t low_bits = ((address & (kAddressSpan - 1)) - kHalfBytes) & (kTileBit - 1);
        realigned = (address & ~(kAddressSpan - 1)) | kTileBit | low_bits;
    }

    MimirChannelLocation location;
    location.tile = static_cast<uint32_t>((realigned >> 34) & 1);
    location.slice = static_cast<uint32_t>((realigned >> 7) & 3);
    location.instance = location.tile * 4 + location.slice;
    return location;
}

}  // namespace tt::tt_metal::mimir_test
