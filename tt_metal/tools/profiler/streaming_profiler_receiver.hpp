// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler host receiver: D2H socket streams -> frames mirrored verbatim into per-stream
// BroadcastRings -> per-consumer decode + pairing -> registered consumers.
//
// Ingest threads fuse poll + frame + copy + ack: whole frames are NT-streamed from the pinned FIFO into
// ring lines and acked as soon as they land, so credit return is copy-paced and the ring -- host RAM,
// sized by env -- is the capture's elastic buffer, not the FIFO. One ring per socket stream keeps each
// ring single-writer.
//
// All decoding happens on the consumer side: each consumer thread owns a reader and a decode state per
// ring, walks the frames itself (the wire's BULK_SPAN prefix does the framing; every frame starts on a
// ring line), and pairs start/end markers into the public 32 B StreamingProfilerRec before its callback.
// A lagging consumer drops its own oldest lines (counted, per consumer) and recovers the way decode
// recovers from a device-side drop: head adoption plus resync counters, timestamps re-anchoring at the
// next absolute zone. An internal audit consumer (record composition compiled out) decodes every stream
// to produce the per-stream wire-integrity report that ingest, a pure copier, does not see.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tools/profiler/streaming_profiler_consumer.hpp"
#include "tools/profiler/spsc_marker_decode.hpp"

namespace tt::tt_metal {

namespace distributed {
class D2HSocket;
}
template <typename T>
class BroadcastRing;

namespace streaming_profiler {

// Receiver-internal record, not the consumer contract: the record consumer-side decode composes into
// its scratch, holding the device's own unpaired start/end markers in 24 B. Its field and bit layout
// (lane at 16, dev at 26, type at 29; ZoneStart=1 / ZoneEnd=2) is pinned by the vectorized packer --
// widening or reordering it is a multi-x decode regression, which is why the public
// StreamingProfilerRec is a separate type built on top. Reaches a consumer only through
// add_raw_consumer below.
enum class StreamingProfilerRawRecType : uint32_t {
    // A complete zone from the device's atomic-zone path (PP_ZONE_ATOMIC): ts is the END and `duration`
    // is set, so start = ts - duration and no pairing is required. Value 0 because the 3-bit type field
    // has 1..7 spoken for.
    Zone = 0,
    ZoneStart = 1,
    ZoneEnd = 2,
    ZoneTotal = 3,
    Data = 4,
    Event = 5,
    Ext = 6,
    Cont = 7,
};

struct StreamingProfilerRawRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;
    uint32_t dev : 3;
    StreamingProfilerRawRecType type : 3;
};
static_assert(sizeof(StreamingProfilerRawRecMeta) == 4);

struct StreamingProfilerRawRec {
    uint64_t ts;
    uint32_t id;  // full 27-bit structural zone id
    StreamingProfilerRawRecMeta meta;
    uint32_t prog;
    uint32_t duration;  // type == Zone only: cycles, with ts being the END. 0 otherwise.
};
// duration lands in tail padding the 8-byte alignment already required, so it costs no wire bytes.
static_assert(sizeof(StreamingProfilerRawRec) == 24);
static_assert(std::is_trivially_copyable_v<StreamingProfilerRawRec>);

// Ring transport unit: one 64 B cache line of wire words. The per-stream ring mirrors the D2H FIFO's
// frames verbatim -- frames are page(=line)-multiples stored back to back, so every frame starts on a
// line and the wire's own BULK_SPAN prefix does the framing.
struct alignas(64) StreamingProfilerRingLine {
    uint32_t w[16];
};
static_assert(sizeof(StreamingProfilerRingLine) == 64);

struct StreamingProfilerRawRecordBatch {
    std::span<const StreamingProfilerRawRec> records;  // oldest first; valid only for the duration of the call
    uint64_t dropped_delta = 0;
    const StreamingProfilerCaptureContext* context = nullptr;
    // PRODUCER-STALL zone opens among the records delivered here (matched by ELF-resolved name): how
    // many times a producer RISC blocked on a full ring. Same delivery lag as the records.
    uint64_t stall_delta = 0;
};

using StreamingProfilerRawRecordCallback = std::function<void(const StreamingProfilerRawRecordBatch&)>;

struct ReceiverDeviceConfig {
    uint32_t chip_id = 0;
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
    uint32_t num_cores = 0;
    std::vector<StreamingProfilerLaneInfo> lane_table;  // size num_cores * 5
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // incl. DRISC self-zone cores
    bool clock_synced = false;
    double frequency_ghz = 0.0;
    int numa_node = -1;  // host node closest to this device; -1 leaves ring and thread unbound
};

// Mirror of the ELF-resolved PRODUCER-STALL zone ids -- one per kernel TU, because that header declares
// the zone at namespace scope. Refreshed by cursor once per decoded frame.
//
// The membership test runs per marker on the decode walk, so it is an open-addressed table: one L1 load
// on the overwhelming miss path. A [min,max] pre-screen would not work -- stall ids carry tu_id in their
// high bits like every other zone id, so their range spans the whole id space.
struct StallIdMirror {
    uint32_t cursor = 0;
    std::vector<uint32_t> ids;    // insertion order, source for rebuilds
    std::vector<uint32_t> table;  // open addressing, linear probe; 0xFFFFFFFF = empty
    uint32_t mask = 0;
    void refresh();
    bool contains(uint32_t id) const {
        if (table.empty()) {
            return false;
        }
        uint32_t slot = (id * 0x9E3779B9u) & mask;
        while (true) {
            const uint32_t v = table[slot];
            if (v == id) {
                return true;
            }
            if (v == 0xFFFFFFFFu) {
                return false;
            }
            slot = (slot + 1) & mask;
        }
    }
};

class StreamingProfilerReceiver {
public:
    explicit StreamingProfilerReceiver(std::vector<ReceiverDeviceConfig> devices);
    ~StreamingProfilerReceiver();

    StreamingProfilerReceiver(const StreamingProfilerReceiver&) = delete;
    StreamingProfilerReceiver& operator=(const StreamingProfilerReceiver&) = delete;

    void start();

    StreamingProfilerConsumerHandle add_consumer(std::string name, StreamingProfilerRecordCallback cb);
    // Internal: subscribe to the raw ring stream (unpaired start/end markers), bypassing the pairing
    // stage. Not part of the public consumer contract; everything else registers through
    // register_consumer.
    StreamingProfilerConsumerHandle add_raw_consumer(std::string name, StreamingProfilerRawRecordCallback cb);
    void remove_consumer(StreamingProfilerConsumerHandle handle);

    // Every relay owning (device, socket) has published done, which implies the device saw all its
    // bytes acked, so the stream retires itself after one final empty check.
    void notify_producers_done(uint32_t device_index, uint32_t socket_index);

    void shutdown();

    const StreamingProfilerCaptureContext& capture_context() const { return ctx_; }
    // Final per-lane words-consumed mirrors; valid after shutdown(). Feeds the control
    // plane's completeness check against the workers' own tails.
    std::vector<uint32_t> final_lane_heads(uint32_t device_index) const;
    void log_report() const;

private:
    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        int ring_node = -1;  // node this stream's ring is bound to, and that its ingest thread runs on
        std::unique_ptr<BroadcastRing<StreamingProfilerRingLine>> ring;
        // Decode-quality accounting, written by the audit consumer and never by the ingest thread:
        // the per-stream decode state plus the counters the report lines quote.
        profiler::SpanDecodeState decode;
        std::vector<uint64_t> last_zone_ts;  // per lane, order invariant (must never regress)
        std::atomic<bool> producers_done{false};
        bool retired = false;

        uint64_t passes = 0, frames = 0, pages = 0, records = 0, zone_markers = 0, stall_zones = 0;
        uint64_t decode_ticks = 0;
        // Split of decode_ticks: ring-copy stores vs pop/ack MMIO writes; the remainder is pass setup.
        uint64_t frame_ticks = 0, ack_ticks = 0;
        uint64_t wpos = 0;  // ring lines written, at or ahead of the writer's committed position
        uint64_t order_regressions = 0, bad_frames = 0;
        uint64_t first_data_tsc = 0, last_commit_tsc = 0;
        uint64_t min_zone_ts = 0, max_zone_ts = 0;
        bool desync_warned = false;
        bool watchdog_fired = false;
        std::chrono::steady_clock::time_point last_progress;
    };

    struct Consumer {
        std::string name;
        StreamingProfilerRecordCallback cb;         // paired (public) path; empty for raw consumers
        StreamingProfilerRawRecordCallback raw_cb;  // raw path (Tracy sink only); empty for public consumers
        StreamingProfilerConsumerHandle handle = 0;
        // The internal wire auditor: decodes every stream (record composition compiled out) and owns
        // the per-stream decode-quality fields on Stream. No callback, not in the public consumer set.
        bool audit = false;
        std::atomic<int> mode{0};  // 0 = run, 1 = drain-then-stop, 2 = stop-now
        uint64_t delivered = 0;
        uint64_t dropped = 0;
        uint64_t busy_ticks = 0;  // time spent decoding/pairing/delivering (excludes idle polling)
        std::thread thread;
    };

    void prefault_rings();
    void decode_thread(std::vector<Stream*> streams);
    // One poll+decode+ack pass over a stream. Returns true if it moved data; sets
    // s.retired when the stream is finished.
    bool ingest_pass(Stream& s);
    void consumer_thread(Consumer& c);

    std::vector<ReceiverDeviceConfig> devices_;
    StreamingProfilerCaptureContext ctx_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<std::thread> decode_threads_;

    std::mutex consumers_mu_;
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<std::unique_ptr<Consumer>> consumers_report_;
    uint64_t next_handle_ = 1;

    std::chrono::seconds watchdog_{120};

    std::atomic<bool> stop_{false};
    std::atomic<bool> shutdown_done_{false};
    bool started_ = false;
    uint32_t nthreads_ = 0;  // decode threads; stream i belongs to thread i % nthreads_
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
