// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler record contract, the internal side. The decode kernels write the public records
// (experimental::streaming_profiler::Zone / Event / TimestampedData, 48-byte values) straight into per-kind buffers
// that a batch's spans cover, and payload elements into a per-batch arena:
//  - A zone arrives as one record with start and duration; consumers never see an unpaired half.
//  - Zones are emitted at close, so per lane they arrive in end order: a nested child precedes its parent and
//    start is not monotonic. One exception: a zone spanning a low-word wrap reads its end before reserving ring
//    space, so a stall zone raised by that reservation precedes it with a later end.
//  - Cross-lane and cross-socket interleaving is arbitrary; the record's meta carries lane and device.
//  - Every id is the 27-bit structural zone id (hostdevcommon/profiler_zone_id.h) and resolves to a name through
//    the zone-meta registry (llrt/zone_meta.hpp); an unnamed id is a bug.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"

namespace tt::tt_metal::streaming_profiler {

using ConsumerHandle = uint64_t;
// Indexed by Core::risc; the order is tracy::RiscType's.
inline constexpr std::array<const char*, 5> kRiscNames = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};

// Immutable once the receiver starts. Zone names are not here: they arrive per ELF as binaries JIT-load, so
// the process-wide site table publishes them as ELFs load (init_site_registry).
// The host<->device clock relation of one chip as the device layer measured it.
struct DeviceClock {
    uint32_t chip_id = 0;
    double frequency_ghz = 0.0;  // device ticks per nanosecond
    uint64_t anchor_ticks = 0;
    int64_t anchor_host_ns = 0;  // std::chrono::steady_clock at `anchor_ticks`, in nanoseconds since its epoch
};

struct CaptureContext {
    struct Device {
        std::vector<experimental::streaming_profiler::Core> lanes;  // index by the record's lane
        std::vector<uint32_t> core_xy;  // core index -> packed NoC (y << 16) | x, the identity a frame carries
        uint32_t chip_id = 0;
        DeviceClock clock;  // the baked anchor the records carry; the d2d sync composes its term with it
        DeviceClock eth_clock;     // the idle-eth core's own wall-clock anchor, for placing PP_CLOCK plot samples
        uint32_t n_eth_cores = 0;  // trailing cores in `lanes` that are eth (idle + active); they use eth_clock
    };
    std::vector<Device> devices;
    // A boot-time eth link sync: the sender on device index dev_a at logical eth core eth_a, the receiver on dev_b
    // at eth_b. The d2d-sync consumer pairs the two ends' PP_CLOCK(LINK) samples by round.
    struct Link {
        uint32_t dev_a = 0, dev_b = 0;
        uint32_t chip_a = 0, chip_b = 0;
        CoreCoord eth_a, eth_b;
    };
    std::vector<Link> links;
};

// What the decoder writes into every record of a lane besides the packet's own words (Record's coordinate, chip,
// RISC, frequency and offset fields), in the record's byte layout.
inline profiler::SpscRecConsts record_consts(const experimental::streaming_profiler::Core& core, const DeviceClock& k) {
    const auto hz = static_cast<uint32_t>(
        std::clamp<int64_t>(std::llround(k.frequency_ghz * 1e9), 1, std::numeric_limits<uint32_t>::max()));
    // The offset is taken against the frequency as the record rounds it, so the anchor itself converts exactly.
    const int64_t offset =
        std::llround(static_cast<double>(k.anchor_host_ns) * (hz * 1e-9)) - static_cast<int64_t>(k.anchor_ticks);
    return profiler::SpscRecConsts{
        .coords =
            {static_cast<uint32_t>(core.logical.x & 0xFFFFu) | (static_cast<uint32_t>(core.logical.y & 0xFFFFu) << 16),
             static_cast<uint32_t>(core.physical.x & 0xFFFFu) |
                 (static_cast<uint32_t>(core.physical.y & 0xFFFFu) << 16)},
        .tail = {
            (core.chip_id & 0xFFFFu) | (static_cast<uint32_t>(core.risc) << 16),
            hz,
            static_cast<uint32_t>(static_cast<uint64_t>(offset)),
            static_cast<uint32_t>(static_cast<uint64_t>(offset) >> 32)}};
}

// Publishes the zone-name registry to the record accessors (api::detail::site_of); idempotent.
void init_site_registry();


struct StreamStats {
    uint64_t records = 0, zones = 0, order_regressions = 0, epoch_fixes = 0;
    uint64_t clock_samples = 0;  // PP_CLOCK samples decoded (idle-eth clock trackers); the drain's proof of life
};

}  // namespace tt::tt_metal::streaming_profiler
