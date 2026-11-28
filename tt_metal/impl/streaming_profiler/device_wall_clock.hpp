// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>

#include <tt-metalium/vocabulary/core_coord.hpp>

namespace tt {
class Cluster;
}
namespace tt::umd {
class IoWindow;
}

namespace tt::tt_metal {

// Returns the host address of the word at `offset` in `window`'s mapping. Device clocks are read with plain loads from
// it, because UMD's read32 times every access and, past its budget, runs a NoC hang check behind a lock that all slow
// accesses share, which can hold a timed read for milliseconds.
const volatile uint32_t* mapped_word(tt::umd::IoWindow& window, uint64_t offset);

// One Tensix core's wall clock, read from the host through an uncached PCIe window onto the core's debug registers.
// Reading the low word latches the core's high word, so the core should be one whose own RISCs rarely read their clock.
class DeviceWallClock {
public:
    DeviceWallClock(tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& virtual_core);
    ~DeviceWallClock();

    DeviceWallClock(const DeviceWallClock&) = delete;
    DeviceWallClock& operator=(const DeviceWallClock&) = delete;

    struct Reading {
        int64_t host = 0;   // the host clock at the midpoint of the read
        uint64_t wall = 0;  // the core's wall clock
    };

    // Reads the clock `reads` times and returns the read with the shortest round trip. `host_clock` returns the host
    // time and is called just before and just after each read.
    template <typename HostClock>
    Reading read(uint32_t reads, HostClock&& host_clock) const {
        int64_t best_round_trip = std::numeric_limits<int64_t>::max();
        Reading best;
        for (uint32_t i = 0; i < reads; i++) {
            const int64_t before = host_clock();
            const uint32_t lo = *lo_;
            const int64_t after = host_clock();
            if (after - before < best_round_trip) {
                best_round_trip = after - before;
                best = {.host = std::midpoint(before, after), .wall = (static_cast<uint64_t>(*hi_) << 32) | lo};
            }
        }
        return best;
    }

private:
    std::unique_ptr<tt::umd::IoWindow> window_;
    const volatile uint32_t* lo_;
    const volatile uint32_t* hi_;
};

}  // namespace tt::tt_metal
