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
    bool is_x280 = false;    // true = an X280 L2CPU-hart zone (own context, distinct color) not a RISC
    uint32_t color = 0;      // explicit Tracy zone color (0 = auto by name); set for X280 zones
    std::string_view ctx_name;  // X280 only: overrides the context row name (e.g. "X280 rd0") so the row is
                                // labeled by hart -- the per-lane header is GUI-derived from risc bits and
                                // cannot be set client-side, so we make each hart its own named context row
};

// A point-in-time worker-core event: the unified PP_DATA packet (an "event" is just size 0). Resolved
// host-side exactly like a zone, but it has no duration, so it lands on the lane's marker row rather than
// in the zone nesting. `values` is a NON-OWNING view into the consumer's reassembly scratch -- valid only
// for the duration of the HandleWorkerEvent call, same contract as `name`.
struct WorkerEventPacket {
    uint32_t chip_id = 0;
    uint32_t core_virtual_x = 0;
    uint32_t core_virtual_y = 0;
    uint32_t core_noc0_x = 0;
    uint32_t core_noc0_y = 0;
    uint32_t risc = 0;
    uint32_t id = 0;                   // 20-bit wire id: a compile-time name hash, or a runtime value
    bool runtime_id = false;           // true = the id is a runtime value (DeviceRuntimeEvent): NOT nameable
    std::string_view name;             // resolved from the hash map; always empty when runtime_id
    uint64_t timestamp = 0;            // full device ticks, already rebased like a zone's
    uint32_t runtime_host_id = 0;      // STICKY_PROG value in effect
    const uint64_t* values = nullptr;  // payload, one entry per uint64 the kernel passed
    uint32_t num_values = 0;
};

}  // namespace tt::tt_metal::perf_debug
