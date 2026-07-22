// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Enriched worker-core kernel-zone packet for the perf-debug (X280) profiler. Host-built and fully
// resolved (NOC0 coords translated, zone name deciphered, start/end split), so PerfDebugTracyHandler
// just pushes it. Mirrors realtime_profiler's WorkerZonePacket (which the clean cut reverted away).
#pragma once

#include <cstdint>
#include <string_view>

namespace tt::tt_metal::perf_debug {

struct WorkerZonePacket {
    uint32_t chip_id = 0;
    uint32_t core_virtual_x = 0;  // as relayed by the X280 (its NoC view)
    uint32_t core_virtual_y = 0;
    uint32_t core_noc0_x = 0;  // translated -> matches the standard DeviceProfiler / DRAM view
    uint32_t core_noc0_y = 0;
    uint32_t risc = 0;       // 0=BRISC 1=NCRISC 2/3/4=TRISC_0/1/2
    uint32_t timer_id = 0;   // 16-bit zone-name hash
    std::string_view name;   // deciphered zone name; stable for the profiler session
    uint64_t timestamp = 0;  // full device ticks (59-bit, reconstructed from STICKY_TIMER)
    bool is_start = false;   // true = ZONE_START, false = ZONE_END
};

}  // namespace tt::tt_metal::perf_debug
