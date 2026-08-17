// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Enriched worker-core event packet for the perf-debug (drainer) profiler. Host-built and fully
// resolved (NOC0 coords translated, name deciphered), so PerfDebugTracyHandler just pushes it.
// Zones no longer travel as packets: they are complete {start, end} pairs by the time they leave the
// decode workers and go through the handler's lock-free ZoneSink path instead.
#pragma once

#include <cstdint>
#include <string_view>

namespace tt::tt_metal::perf_debug {

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
