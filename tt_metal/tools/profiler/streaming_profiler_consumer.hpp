// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler record contract: the record the receiver delivers to consumers, the lane
// table consumers resolve identity against, and the batch-callback types.
//
// Stream contracts:
//  - A zone arrives as one record, `Zone`, carrying its start timestamp and duration; the receiver
//    pairs the device's raw start/end markers per lane before delivery, so consumers never see an
//    unpaired half.
//  - A Zone is emitted when it closes, so per lane Zones arrive in end order: with nested zones the
//    child precedes its parent and data.zone.start is not monotonic. Sort on start if your analysis
//    needs open order.
//  - Cross-lane and cross-socket interleaving is arbitrary; demux by meta.lane / meta.dev.
//  - A Data head is followed immediately by one Ext record (id = payload word count, data.ext =
//    payload words 1-2 as (hi << 32) | lo, zero-filled) and then Cont records for words 3 and up
//    (one payload uint64 each, hi word first), with no other records interleaved. An Event is
//    payload-less and complete in itself.
//  - ZoneTotal carries an accumulated duration sum (data.sum), not a timestamp.
//  - Every id is the full 27-bit structural zone id (hostdevcommon/profiler_zone_id.h) and resolves
//    to a name from the emitting binary's own ELF via ZoneNameMirror below, zones and markers alike;
//    an unnamed id is a bug.
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

namespace tt::tt_metal::streaming_profiler {

enum class StreamingProfilerRecType : uint32_t {
    Zone = 1,       // a complete zone: data.zone = {start, duration}
    ZoneTotal = 2,  // accumulated-duration zone: data.sum
    Data = 3,       // point marker with payload: data.ts; payload follows via Ext (+ Cont)
    Event = 4,      // point marker, no payload: data.ts; complete in itself
    Ext = 5,        // Data continuation: id = payload word count, data.ext = payload words 1-2
    Cont = 6,       // one uint64 of Data payload (words 3 and up): data.payload
};

struct StreamingProfilerRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;  // which (core, RISC) stream: lane = core_index * 5 + risc
    uint32_t dev : 3;    // device index into the capture context
    StreamingProfilerRecType type : 3;
};
static_assert(sizeof(StreamingProfilerRecMeta) == 4);

struct StreamingProfilerRec {
    // Active member is decided by meta.type.
    union DataField {
        uint64_t ts;       // Data / Event: head timestamp
        uint64_t sum;      // ZoneTotal: accumulated duration
        uint64_t ext;      // Ext: payload words 1-2, (hi << 32) | lo, zero-filled
        uint64_t payload;  // Cont: one payload uint64 (hi word first on the wire)
        struct {
            uint64_t start;     // device timestamp of the zone open
            uint64_t duration;  // device cycles
        } zone;                 // Zone
    } data;
    uint32_t id;  // full 27-bit structural zone id (tu_id << TT_ZONE_LOCAL_BITS | local)
    StreamingProfilerRecMeta meta;
    uint32_t prog;  // runtime host-id in force on this lane; 0 when never set
};
static_assert(sizeof(StreamingProfilerRec) == 32);
static_assert(std::is_trivially_copyable_v<StreamingProfilerRec>);

inline constexpr uint32_t kStreamingProfilerMaxLanes = 1u << 10;
inline constexpr uint32_t kStreamingProfilerMaxDevices = 1u << 3;

enum class StreamingProfilerLaneRole : uint8_t { Worker = 0, Relay = 1 };

struct StreamingProfilerLaneInfo {
    uint32_t chip_id = 0;
    uint16_t virtual_x = 0, virtual_y = 0;
    uint16_t noc0_x = 0, noc0_y = 0;
    uint8_t risc = 0;
    StreamingProfilerLaneRole role = StreamingProfilerLaneRole::Worker;
};

// Immutable once the receiver starts. Zone names are not here: they arrive per-ELF as binaries
// JIT-load, so the table grows throughout a run and each consumer keeps its own lazily-refreshed
// ZoneNameMirror below.
struct StreamingProfilerCaptureContext {
    struct Device {
        uint32_t chip_id = 0;
        bool clock_synced = false;                     // true -> record timestamps are raw device time with a
        double frequency_ghz = 0.0;                    //         registered host anchor; false -> rebase yourself
        std::vector<StreamingProfilerLaneInfo> lanes;  // index by StreamingProfilerRecMeta::lane
    };
    std::vector<Device> devices;
};

// Per-consumer mirror of the process-wide per-ELF zone-name registry (llrt::ZoneMetaRegistry). Each
// consumer runs on its own delivery thread, so a member mirror needs no lock on the per-record lookup.
// Call refresh() once per batch, lookup() per record.
//
// `unnamed` must end at 0: a binary's names are registered when it is loaded, strictly before it can
// emit, so a miss means a binary without .tt_zone_meta or a tu_id collision. The distinct offending ids
// are kept (capped) so the count says which.
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

// Log a consumer's unnamed-id tally; no-op when it is 0. Call at consumer teardown.
void log_unnamed_ids(std::string_view consumer_name, const ZoneNameMirror& mirror);

struct StreamingProfilerRecordBatch {
    std::span<const StreamingProfilerRec> records;  // oldest first; valid only for the duration of the call
    uint64_t dropped_delta = 0;                     // records this consumer lost to ring lag since its last batch
    const StreamingProfilerCaptureContext* context = nullptr;
    // PRODUCER-STALL zones in this batch (matched by ELF-resolved name): each is one time a producer
    // RISC blocked on a full L1 ring. Counted among the records handed over here, so it carries the
    // same delivery lag as they do.
    uint64_t stall_delta = 0;
};

using StreamingProfilerRecordCallback = std::function<void(const StreamingProfilerRecordBatch&)>;
using StreamingProfilerConsumerHandle = uint64_t;

// Registers with a process-wide registry rather than a live receiver, so it works at any time: before
// the profiler boots, mid-capture, between captures. The profiler attaches every registered consumer at
// capture start, a mid-capture registration attaches immediately, and registrations persist across
// captures until unregistered. Each attached consumer gets its own delivery thread; a slow consumer
// drops only its own records, reported per batch via dropped_delta. Must not be called from inside a
// consumer callback.
StreamingProfilerConsumerHandle register_consumer(std::string name, StreamingProfilerRecordCallback cb);
void unregister_consumer(StreamingProfilerConsumerHandle handle);

class StreamingProfilerReceiver;
// Capture-lifetime glue for the profiler control plane: attach binds every registered consumer to the
// receiver and routes later registrations to it; detach must precede the receiver's shutdown so a
// concurrent registration cannot attach to a dying receiver.
void attach_registered_consumers(StreamingProfilerReceiver& receiver);
void detach_registered_consumers();

void register_file_consumer_impl(
    std::string name,
    std::string (*path)(),
    StreamingProfilerRecordCallback on_batch,
    std::function<void(const std::string&)> write);

// Declares a consumer that accumulates over the whole process and writes one file at exit, enabled by
// `path` returning a non-empty string. `Consumer` needs operator()(const StreamingProfilerRecordBatch&)
// and write_csv(const std::string&).
//
// Meant to be called from a static initializer, so `path` is not invoked here: it reads rtoptions, which
// only exists once MetalContext is constructed. It runs when a capture attaches, and the consumer is
// registered only then and only if the path is non-empty.
template <typename Consumer>
void register_file_consumer(std::string name, std::string (*path)()) {
    // Leaked deliberately: an exit-time destructor would be ordered against other statics.
    auto* c = new Consumer();
    register_file_consumer_impl(
        std::move(name),
        path,
        [c](const StreamingProfilerRecordBatch& b) { (*c)(b); },
        [c](const std::string& p) { c->write_csv(p); });
}

}  // namespace tt::tt_metal::streaming_profiler
