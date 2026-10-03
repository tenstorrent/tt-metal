// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kProcessorCount = static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC1) + 1;

// 64x64 covers every supported grid; a coordinate outside it is unknown.
struct CoreTable {
    static constexpr uint16_t kNone = 0xFFFF;
    static constexpr uint32_t kCoordBits = 6;
    static constexpr uint32_t kCoordMask = (1u << kCoordBits) - 1;
    static constexpr size_t kSlots = size_t{1} << (2 * kCoordBits);
    static constexpr uint32_t kOutsideGrid = ~((kCoordMask << 16) | kCoordMask);
    std::vector<uint16_t> slot = std::vector<uint16_t>(kSlots, kNone);
    static uint32_t index(uint32_t xy) { return (((xy >> 16) & kCoordMask) << kCoordBits) | (xy & kCoordMask); }
    uint16_t& operator[](uint32_t xy) { return slot[index(xy)]; }
    uint32_t find(uint32_t xy) const { return (xy & kOutsideGrid) != 0 ? kNone : slot[index(xy)]; }
};

// Immutable once the receiver starts.
struct CaptureContext {
    // The device HostSync ties to the host over PCIe; every chip's clock maps onto its refclk.
    static constexpr uint32_t kRootDevice = 0;
    struct Device {
        struct Tile {
            uint32_t xy = 0;  // kernel_profiler::NocXy
            // The tracker's wall tick minus this core's. Every tile keeps its own wall clock on the one AICLK, so it's
            // one integer for the capture.
            int64_t clock_offset = 0;
        };
        // profiler::kSpscNRiscDecode lanes per tile, in tile order and RISC order; an eth tile fills its unused lanes
        // with ERISC1.
        std::vector<experimental::streaming_profiler::Core> lanes;
        std::vector<Tile> tiles;  // by core index
        CoreTable core_of_xy;     // a tile's xy to its core index
        uint32_t chip_id = 0;
        // The tracker's wall tick minus the sync check ruler's, or 0 without a ruler.
        int64_t ruler_offset = 0;
    };
    std::vector<Device> devices;
    struct Link {
        uint32_t dev_a = 0, dev_b = 0;
        uint32_t core_a = 0, core_b = 0;
        CoreCoord eth_a, eth_b;
    };
    std::vector<Link> links;
    bool sync_check = false;
};

template <std::predicate<size_t> Usable>
std::vector<bool> reached_from_root(std::span<const CaptureContext::Link> links, size_t devices, Usable usable) {
    std::vector<bool> reached(devices, false);
    reached[CaptureContext::kRootDevice] = true;
    for (bool grew = true; grew;) {
        grew = false;
        for (size_t link_index = 0; link_index < links.size(); link_index++) {
            const CaptureContext::Link& link = links[link_index];
            if (reached[link.dev_a] != reached[link.dev_b] && usable(link_index)) {
                reached[link.dev_a] = reached[link.dev_b] = true;
                grew = true;
            }
        }
    }
    return reached;
}

}  // namespace tt::tt_metal::streaming_profiler
