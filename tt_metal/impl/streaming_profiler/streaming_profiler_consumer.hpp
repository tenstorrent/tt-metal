// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler record contract: the record the receiver delivers, the lane table identity resolves
// against, and the batch-callback types.
//  - A zone arrives as one `Zone` record with start and duration; consumers never see an unpaired half.
//  - Zones are emitted at close, so per lane they arrive in end order: a nested child precedes its parent and
//    start is not monotonic. One exception: a zone spanning a low-word wrap reads its end before reserving ring
//    space, so a stall zone raised by that reservation precedes it with a later end.
//  - Cross-lane and cross-socket interleaving is arbitrary; demux by meta.lane / meta.dev.
//  - A Data head is followed immediately by one Ext record (id = payload word count, data.ext = payload words
//    1-2 as (hi << 32) | lo) and then Cont records for words 3 and up (one uint64 each, hi word first), with
//    nothing interleaved. An Event is complete in itself.
//  - Every id is the 27-bit structural zone id (hostdevcommon/profiler_zone_id.h) and resolves to a name through
//    the zone-meta registry (llrt/zone_meta.hpp); an unnamed id is a bug.
// In-process only, never serialized; the static_asserts pin the size.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>
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
// Indexed by LaneInfo::risc; the order is tracy::RiscType's.
inline constexpr std::array<const char*, 5> kRiscNames = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};

struct LaneInfo {
    uint32_t chip_id = 0;
    uint16_t logical_x = 0, logical_y = 0;
    uint16_t physical_x = 0, physical_y = 0;  // NoC 0
    uint8_t risc = 0;
};

// Immutable once the receiver starts. Zone names are not here: they arrive per ELF as binaries JIT-load, so
// each subscription mirrors the registry for itself (streaming_profiler_api.cpp).
struct CaptureContext {
    struct Device {
        std::vector<LaneInfo> lanes;    // index by RecMeta::lane
        std::vector<uint32_t> core_xy;  // core index -> packed NoC (y << 16) | x, the identity a frame carries
    };
    std::vector<Device> devices;
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

// One decoder's wire-integrity totals for one stream and, once it detaches, each lane's words-consumed mirror (0 for
// a lane it never saw) and the ring lines its reader lost.
struct StreamStats {
    uint64_t records = 0, zones = 0, order_regressions = 0, bad_frames = 0, epoch_fixes = 0;
    uint64_t resync_words = 0, anomalies = 0, unknown_core_frames = 0;
    uint64_t dropped = 0;
    std::vector<uint32_t> heads;
};

}  // namespace tt::tt_metal::streaming_profiler
