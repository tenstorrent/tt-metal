// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kProcessorCount = static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC1) + 1;
inline constexpr std::array<const char*, kProcessorCount> kProcessorNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC_0", "ERISC_1"};

constexpr bool is_ethernet(experimental::streaming_profiler::Processor processor) {
    return processor == experimental::streaming_profiler::Processor::ERISC0 ||
           processor == experimental::streaming_profiler::Processor::ERISC1;
}

// Maps NoC coordinates to core indices. 64x64 covers every supported grid, and find() returns kNone outside it.
struct CoreTable {
    static constexpr uint16_t kNone = 0xFFFF;
    static constexpr uint32_t kCoordBits = 6;
    static constexpr uint32_t kCoordMask = (1u << kCoordBits) - 1;
    static constexpr size_t kSlots = size_t{1} << (2 * kCoordBits);
    std::vector<uint16_t> slot = std::vector<uint16_t>(kSlots, kNone);
    static uint32_t index(kernel_profiler::NocXy c) { return ((c.y & kCoordMask) << kCoordBits) | (c.x & kCoordMask); }
    uint16_t& operator[](uint32_t xy) { return slot[index(kernel_profiler::word_as<kernel_profiler::NocXy>(xy))]; }
    uint32_t find(uint32_t xy) const {
        const auto c = kernel_profiler::word_as<kernel_profiler::NocXy>(xy);
        return (c.x | c.y) > kCoordMask ? kNone : slot[index(c)];
    }
};

struct FileClose {
    void operator()(FILE* file) const { std::fclose(file); }
};

// A capture is one mesh device profiled from the moment it opens until it closes. This holds the capture's chips and
// the eth links between them that the clock sync measures, and it is immutable once the receiver starts.
struct CaptureContext {
    // The device whose refclk the host reads over PCIe. Every chip's clock is mapped onto this device's refclk.
    static constexpr uint32_t kRootDevice = 0;
    struct Device {
        // Holds kernel_profiler::PROFILER_SPSC_TENSIX_RISC lanes for each tile, ordered by tile and then by RISC. An
        // eth tile has two RISCs and fills its other lanes with ERISC1.
        std::vector<experimental::streaming_profiler::Core> lanes;
        // Each tile's clock offset, by core index. It is the wall clock of the chip's wall-clock core minus the tile's
        // at the same instant, so adding it puts the tile's timestamps on the wall-clock core's clock, which the clock
        // sync maps to the root chip's refclk. Every tile's wall clock counts the same AICLK, so the offset stays fixed
        // for the whole capture.
        std::vector<int64_t> clock_offsets;
        CoreTable core_of_xy;  // a tile's xy to its core index
        uint32_t chip_id = 0;
        // The wall-clock core's wall clock minus the check core's, or 0 when the sync check is off.
        int64_t check_offset = 0;
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
