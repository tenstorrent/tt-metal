// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf-debug profiler record contract: the 16 B record the receiver publishes, the lane
// table consumers resolve identity against, and the batch-callback types.
//
// Stream contracts:
//  - Per lane, records appear in emission order with monotonic timestamps. Cross-lane and
//    cross-socket interleaving is arbitrary; demux by meta.lane / meta.dev.
//  - A Data/Event head is followed immediately by one Ext record (ts = id20<<32 | payload
//    word count) and then Cont records (one payload uint64 each, hi word first), with no
//    other records interleaved.
//  - ZoneTotal carries an accumulated duration sum in ts, not a timestamp.
//  - Event ids are runtime values and must never be name-resolved; Data ids may be.
//
// The record is in-process only (never serialized), so bit-field layout portability is
// not a concern; the static_asserts pin the size.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal::perf_debug {

enum class PerfDebugRecType : uint32_t {
    ZoneStart = 1,
    ZoneEnd = 2,
    ZoneTotal = 3,
    Data = 4,
    Event = 5,
    Ext = 6,
    Cont = 7,
};

struct PerfDebugRecMeta {
    uint32_t id : 16;  // zone srcloc hash / data-event id16
    uint32_t lane : 10;
    uint32_t dev : 3;
    PerfDebugRecType type : 3;
};
static_assert(sizeof(PerfDebugRecMeta) == 4);

struct PerfDebugRec {
    uint64_t ts;
    PerfDebugRecMeta meta;
    uint32_t prog;  // runtime host-id in force on this lane (0 when never set); exact per lane
};
static_assert(sizeof(PerfDebugRec) == 16);
static_assert(std::is_trivially_copyable_v<PerfDebugRec>);

inline constexpr uint32_t kPerfDebugMaxLanes = 1u << 10;
inline constexpr uint32_t kPerfDebugMaxDevices = 1u << 3;

enum class PerfDebugLaneRole : uint8_t { Worker = 0, Filler = 1, Mover = 2, Full = 3 };

struct PerfDebugLaneInfo {
    uint32_t chip_id = 0;
    uint16_t virtual_x = 0, virtual_y = 0;
    uint16_t noc0_x = 0, noc0_y = 0;
    uint8_t risc = 0;
    PerfDebugLaneRole role = PerfDebugLaneRole::Worker;
};

// Immutable once the receiver starts, except zone_names, which the receiver fills exactly
// once before the first batch is delivered to any consumer.
struct PerfDebugCaptureContext {
    struct Device {
        uint32_t chip_id = 0;
        bool clock_synced = false;             // true -> record timestamps are raw device time with a
        double frequency_ghz = 0.0;            //         registered host anchor; false -> rebase yourself
        std::vector<PerfDebugLaneInfo> lanes;  // index by PerfDebugRecMeta::lane
    };
    std::vector<Device> devices;
    std::unordered_map<uint16_t, std::string> zone_names;

    std::string_view zone_name(uint16_t hash) const {
        auto it = zone_names.find(hash);
        return it == zone_names.end() ? std::string_view{} : std::string_view{it->second};
    }
};

struct PerfDebugRecordBatch {
    std::span<const PerfDebugRec> records;  // oldest first; valid only for the duration of the call
    uint64_t dropped_delta = 0;             // records THIS consumer lost to ring lag since its last batch
    const PerfDebugCaptureContext* context = nullptr;
};

using PerfDebugRecordCallback = std::function<void(const PerfDebugRecordBatch&)>;
using PerfDebugConsumerHandle = uint64_t;

// ---- Consumer registration (internal) ---------------------------------------------------------
//
// Registers with a process-wide registry rather than a live receiver, so it works at any time:
// before the profiler boots, mid-capture, between captures. The profiler attaches every registered
// consumer at capture start, a mid-capture registration attaches immediately, and registrations
// persist across captures until unregistered. Each attached consumer gets its own delivery thread;
// a slow consumer drops only its own records, reported per batch via dropped_delta. Must not be
// called from inside a consumer callback.
PerfDebugConsumerHandle register_consumer(std::string name, PerfDebugRecordCallback cb);
void unregister_consumer(PerfDebugConsumerHandle handle);

class PerfDebugReceiver;
// Capture-lifetime glue for the profiler control plane only: attach binds every registered
// consumer to the receiver and routes later registrations to it; detach must precede the
// receiver's shutdown so a concurrent registration cannot attach to a dying receiver.
void attach_registered_consumers(PerfDebugReceiver& receiver);
void detach_registered_consumers();

}  // namespace tt::tt_metal::perf_debug
