// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-device live-buffer registries consumed by the emule sanitizers.
// snapshot() returns packed (low << 32) | high pairs — flat uint64_t scan
// in JIT translation units, no std::vector layout dependency. See SANITIZER_CHECKS.md.

#include <cstdint>
#include <vector>

namespace tt::tt_metal::emule {

class LiveL1Ranges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static void remove(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

class LiveDramRanges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static void remove(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

// Raw L1 regions the host designates via WriteToDeviceL1 / ReadFromDeviceL1 —
// valid data living outside the Buffer allocator, so absent from LiveL1Ranges.
// Add-only (deduped); consumed only by the OOB check, never by Object Intent.
class LiveL1HostPokeRanges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static std::vector<uint64_t> snapshot(int device_id);
};

// Padding regions [logical_end, physical_end) declared via emule::register_logical_size.
class LiveL1PaddingRanges {
public:
    static void set(int device_id, uint32_t start, uint32_t logical_end, uint32_t physical_end);
    static void clear(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

}  // namespace tt::tt_metal::emule
