// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Enriched worker-core kernel-zone packet for the perf-debug (drainer) profiler. Host-built and fully
// resolved (NOC0 coords translated, zone name deciphered, start/end split), so PerfDebugTracyHandler
// just pushes it. Mirrors realtime_profiler's WorkerZonePacket (which the clean cut reverted away).
#pragma once

#include <cstdint>
#include <string_view>

namespace tt::tt_metal::perf_debug {

struct WorkerZonePacket {
    uint32_t chip_id = 0;
    uint32_t core_virtual_x = 0;  // as relayed by the drainer (its NoC view)
    uint32_t core_virtual_y = 0;
    uint32_t core_noc0_x = 0;  // translated -> matches the standard DeviceProfiler / DRAM view
    uint32_t core_noc0_y = 0;
    uint32_t risc = 0;      // 0=BRISC 1=NCRISC 2/3/4=TRISC_0/1/2
    uint32_t timer_id = 0;  // the 27-bit structural zone id (hostdevcommon/profiler_zone_id.h)
    std::string_view name;  // zone name, resolved from the kernel's own ELF; stable for the profiler session
    // One COMPLETE zone, both endpoints in full device ticks. The wire ships zones whole at close, and
    // Tracy now takes them whole (QueueGpuZone), so there is no begin/end split anywhere on this path.
    // Per (core, risc) packets must arrive in zone-COMPLETION order (non-decreasing `end`) -- that is the
    // paired stream's arrival order, and it is what the Tracy server rebuilds nesting from.
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t color = 0;  // explicit Tracy zone color (0 = auto by name)
};

// A point-in-time worker-core event: a PP_DATA packet (payload) or a PP_EVENT flag (none). Resolved
// host-side exactly like a zone -- both types carry a compile-time 27-bit structural id -- but it has no
// duration, so it lands on the lane's marker row rather than in the zone nesting. `values` is a NON-OWNING
// view into the consumer's reassembly scratch -- valid only for the duration of the HandleWorkerEvent
// call, same contract as `name`.
struct WorkerEventPacket {
    uint32_t chip_id = 0;
    uint32_t core_virtual_x = 0;
    uint32_t core_virtual_y = 0;
    uint32_t core_noc0_x = 0;
    uint32_t core_noc0_y = 0;
    uint32_t risc = 0;
    uint32_t id = 0;                   // the 27-bit structural zone id, compile-time like a zone's
    std::string_view name;             // resolved from the emitting kernel's ELF, like a zone's
    uint64_t timestamp = 0;            // full device ticks, already rebased like a zone's
    uint32_t runtime_host_id = 0;      // STICKY_PROG value in effect
    const uint64_t* values = nullptr;  // payload, one entry per uint64 the kernel passed
    uint32_t num_values = 0;
};

}  // namespace tt::tt_metal::perf_debug
