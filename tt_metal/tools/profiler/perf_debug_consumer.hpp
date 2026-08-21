// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf-debug profiler record contract: the record the receiver delivers to consumers, the lane
// table consumers resolve identity against, and the batch-callback types.
//
// Stream contracts:
//  - A zone arrives as ONE record: `Zone`, carrying its start timestamp and duration. Worker
//    zones ship from the device already whole (one ZONE_ATOMIC packet at scope close); the few
//    remaining start/end pairs (stall zone, >3.2s fallback) are paired by the
//    receiver BEFORE delivery -- consumers never see an unpaired half.
//  - A Zone is emitted when it CLOSES, so per lane, Zones arrive in END order: with nested
//    zones the child precedes its parent and data.zone.start is NOT monotonic. start+duration
//    is complete information; sort on start if your analysis needs open order.
//  - Cross-lane and cross-socket interleaving is arbitrary; demux by meta.lane / meta.dev.
//  - A Data/Event head is followed immediately by one Ext record (data.ext = id<<32 | payload
//    word count) and then Cont records (one payload uint64 each, hi word first), with no
//    other records interleaved.
//  - Every id is the FULL 27-bit structural zone id (hostdevcommon/profiler_zone_id.h) and
//    resolves to a name from the emitting binary's own ELF via ZoneNameMirror below --
//    zones, Data and Event alike. (Event used to carry a runtime value here; that value now
//    rides Data payload, so an unnamed id is a bug, not a category.)
//
// The record is in-process only (never serialized), so bit-field and union layout portability
// is not a concern; the static_asserts pin the size.
#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal::perf_debug {

enum class PerfDebugRecType : uint32_t {
    Zone = 1,  // a complete zone: data.zone = {start, duration}
    // 2 retired (was ZoneTotal -- the SUM/accumulate zone, feature removed)
    Data = 3,   // point marker with payload: data.ts; payload follows via Ext + Cont
    Event = 4,  // point marker, no payload: data.ts
    Ext = 5,    // Data/Event continuation header: data.ext = (id << 32) | payload word count
    Cont = 6,   // one uint64 of Data payload: data.payload
};

struct PerfDebugRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;  // which (core, RISC) stream: lane = core_index * 5 + risc
    uint32_t dev : 3;    // device index into the capture context
    PerfDebugRecType type : 3;
};
static_assert(sizeof(PerfDebugRecMeta) == 4);

struct PerfDebugRec {
    // The active member is decided by meta.type -- see the enum above.
    union DataField {
        uint64_t ts;       // Data / Event: head timestamp
        uint64_t ext;      // Ext: (id << 32) | payload word count
        uint64_t payload;  // Cont: one payload uint64 (hi word first on the wire)
        struct {
            uint64_t start;     // device timestamp of the zone open
            uint64_t duration;  // device cycles
        } zone;                 // Zone
    } data;
    uint32_t id;  // full 27-bit structural zone id (tu_id << TT_ZONE_LOCAL_BITS | local)
    PerfDebugRecMeta meta;
    uint32_t prog;  // runtime host-id in force on this lane (0 when never set); exact per lane
};
static_assert(sizeof(PerfDebugRec) == 32);
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

// Immutable once the receiver starts. Zone NAMES are deliberately not here: they arrive per-ELF as
// binaries JIT-load (llrt::ZoneMetaRegistry), so the table GROWS throughout a model run and each
// consumer keeps its own lazily-refreshed ZoneNameMirror instead of sharing a fill-once snapshot.
struct PerfDebugCaptureContext {
    struct Device {
        uint32_t chip_id = 0;
        bool clock_synced = false;             // true -> record timestamps are raw device time with a
        double frequency_ghz = 0.0;            //         registered host anchor; false -> rebase yourself
        std::vector<PerfDebugLaneInfo> lanes;  // index by PerfDebugRecMeta::lane
    };
    std::vector<Device> devices;
};

// Per-consumer mirror of the process-wide per-ELF zone-name registry (llrt::ZoneMetaRegistry).
// Each consumer runs on its own delivery thread, so a member mirror needs no lock on the per-record
// lookup; refresh() copies only the registry's append-only delta and is cheap (an empty delta) except
// when an ELF just loaded. Call refresh() once per batch, lookup() per record.
//
// `unnamed` MUST end at 0: a binary's names are registered when it is LOADED, strictly before it can
// emit, so a miss means a binary without .tt_zone_meta or a tu_id collision -- exactly the failure the
// per-ELF registry replaced, counted so it cannot come back silently. The distinct offending ids are
// kept (capped) so the count decomposes into tu/local and says WHICH.
class ZoneNameMirror {
public:
    static constexpr size_t kMaxUnnamedIds = 16;

    void refresh();
    std::string_view lookup(uint32_t id) {
        if (auto it = names_.find(id); it != names_.end()) {
            return it->second;
        }
        unnamed_++;
        if (unnamed_ids_.size() < kMaxUnnamedIds) {
            unnamed_ids_.insert(id);
        }
        return {};
    }
    uint64_t unnamed() const { return unnamed_; }
    const std::set<uint32_t>& unnamed_ids() const { return unnamed_ids_; }

private:
    std::unordered_map<uint32_t, std::string> names_;
    uint32_t cursor_ = 0;
    uint64_t unnamed_ = 0;
    std::set<uint32_t> unnamed_ids_;
};

// Log a consumer's unnamed-id tally (no-op when it is 0, the invariant). Call at consumer teardown.
void log_unnamed_ids(std::string_view consumer_name, const ZoneNameMirror& mirror);

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
