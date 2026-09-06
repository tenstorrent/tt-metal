// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler record contract: the record the receiver delivers, the lane table identity resolves
// against, and the batch-callback types.
//  - A zone arrives as one `Zone` record with start and duration; consumers never see an unpaired half.
//  - Zones are emitted at close, so per lane they arrive in end order: a nested child precedes its parent and
//    start is not monotonic.
//  - Cross-lane and cross-socket interleaving is arbitrary; demux by meta.lane / meta.dev.
//  - A Data head is followed immediately by one Ext record (id = payload word count, data.ext = payload words
//    1-2 as (hi << 32) | lo) and then Cont records for words 3 and up (one uint64 each, hi word first), with
//    nothing interleaved. An Event is complete in itself.
//  - Every id is the 27-bit structural zone id (hostdevcommon/profiler_zone_id.h) and resolves to a name via
//    ZoneNameMirror; an unnamed id is a bug.
// In-process only, never serialized; the static_asserts pin the size.
#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

enum class RecType : uint32_t {
    Zone = 1,   // a complete zone: data.zone = {start, duration}
    Data = 2,   // point marker with payload: data.ts; payload follows via Ext (+ Cont)
    Event = 3,  // point marker, no payload: data.ts; complete in itself
    Ext = 4,    // Data continuation: id = payload word count, data.ext = payload words 1-2
    Cont = 5,   // one uint64 of Data payload (words 3 and up): data.payload
};

struct RecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;  // which (core, RISC) stream: lane = core_index * 5 + risc
    uint32_t dev : 3;    // device index into the capture context
    RecType type : 3;
};
static_assert(sizeof(RecMeta) == 4);

struct Rec {
    // Active member is decided by meta.type.
    union DataField {
        uint64_t ts;       // Data / Event: head timestamp
        uint64_t ext;      // Ext: payload words 1-2, (hi << 32) | lo, zero-filled
        uint64_t payload;  // Cont: one payload uint64 (hi word first on the wire)
        struct {
            uint64_t start;     // device timestamp of the zone open
            uint64_t duration;  // device cycles
        } zone;                 // Zone
    } data;
    uint32_t id;  // full 27-bit structural zone id (tu_id << TT_ZONE_LOCAL_BITS | local)
    RecMeta meta;
    uint32_t prog;  // runtime host-id in force on this lane; 0 when never set
};
static_assert(sizeof(Rec) == 32);
static_assert(std::is_trivially_copyable_v<Rec>);

inline constexpr uint32_t kStreamingProfilerMaxLanes = 1u << 10;
inline constexpr uint32_t kStreamingProfilerMaxDevices = 1u << 3;

struct LaneInfo {
    uint32_t chip_id = 0;
    uint16_t logical_x = 0, logical_y = 0;
    uint8_t risc = 0;
};

// Immutable once the receiver starts. Zone names are not here: they arrive per ELF as binaries JIT-load, so
// each consumer keeps its own ZoneNameMirror.
struct CaptureContext {
    struct Device {
        uint32_t chip_id = 0;
        std::vector<LaneInfo> lanes;                   // index by RecMeta::lane
        std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y << 16) | x -> dense core index
    };
    std::vector<Device> devices;
};

// Per-consumer mirror of the process-wide per-ELF zone-name registry; each consumer runs on its own thread, so
// lookups need no lock. refresh() once per batch, lookup() per record. Names register at load, strictly before a
// binary can emit, so a miss is a binary without .tt_zone_meta or a tu_id collision.
class ZoneNameMirror {
public:
    struct Site {
        std::string name;
        std::string file;
        uint32_t line = 0;
    };

    void refresh();
    // nullptr for an unnamed id.
    const Site* lookup_site(uint32_t id) const {
        const auto it = sites_.find(id);
        return it != sites_.end() ? &it->second : nullptr;
    }
    std::string_view lookup(uint32_t id) const {
        const Site* s = lookup_site(id);
        return s != nullptr ? std::string_view(s->name) : std::string_view{};
    }

private:
    std::unordered_map<uint32_t, Site> sites_;
    uint32_t cursor_ = 0;
};

struct RecordBatch {
    std::span<const Rec> records;                   // oldest first; valid only for the duration of the call
    uint64_t dropped_delta = 0;                     // records this consumer lost to ring lag since its last batch
    const CaptureContext* context = nullptr;
    std::span<const experimental::streaming_profiler::Clock> clocks;  // per device index, as of this batch
    // PROFILER-STALL zones in this batch, each one time a producer RISC blocked on a full ring.
    uint64_t stall_delta = 0;
};

using RecordCallback = std::function<void(const RecordBatch&)>;
using ConsumerHandle = uint64_t;

}  // namespace tt::tt_metal::streaming_profiler
