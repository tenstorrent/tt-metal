// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Worker-core packets handed to StreamingProfilerTracyHandler, resolved host-side: NOC0 coords
// translated and zone name looked up, so the handler only has to push them.
#pragma once

#include <cstdint>
#include <string_view>

namespace tt::tt_metal::streaming_profiler {

struct WorkerZonePacket {
    uint32_t chip_id = 0;
    uint32_t core_virtual_x = 0;  // as relayed by the relay (its NoC view)
    uint32_t core_virtual_y = 0;
    uint32_t core_noc0_x = 0;  // translated -> matches the standard DeviceProfiler / DRAM view
    uint32_t core_noc0_y = 0;
    uint32_t risc = 0;      // 0=BRISC 1=NCRISC 2/3/4=TRISC_0/1/2
    uint32_t timer_id = 0;  // the 27-bit structural zone id (hostdevcommon/profiler_zone_id.h)
    std::string_view name;  // resolved from the kernel's own ELF; stable for the profiler session
    // One complete zone, both endpoints in full device ticks. Per (core, risc), packets must arrive in
    // zone-completion order (non-decreasing `end`): that is what the Tracy server rebuilds nesting from.
    uint64_t start = 0;
    uint64_t end = 0;
    uint32_t color = 0;  // explicit Tracy zone color (0 = auto by name)
};

// A point-in-time worker-core event: a PP_DATA packet (with payload) or a PP_EVENT flag (without). It
// has no duration, so it lands on the lane's marker row rather than in the zone nesting. `values` and
// `name` are non-owning views, valid only for the duration of the HandleWorkerEvent call.
struct WorkerEventPacket {
    uint32_t chip_id = 0;
    uint32_t core_virtual_x = 0;
    uint32_t core_virtual_y = 0;
    uint32_t core_noc0_x = 0;
    uint32_t core_noc0_y = 0;
    uint32_t risc = 0;
    uint32_t id = 0;                   // the 27-bit structural zone id
    std::string_view name;             // resolved from the emitting kernel's ELF
    uint64_t timestamp = 0;            // full device ticks, already rebased
    uint32_t runtime_host_id = 0;      // STICKY_PROG value in effect
    const uint64_t* values = nullptr;  // payload, one entry per uint64 the kernel passed
    uint32_t num_values = 0;
};

}  // namespace tt::tt_metal::streaming_profiler
