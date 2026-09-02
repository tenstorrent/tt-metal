// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler host receiver: D2H socket streams are mirrored verbatim into per-stream BroadcastRings,
// and every consumer thread decodes and pairs for itself before its callback. Ingest fuses poll + frame + copy
// + ack: frames are NT-streamed from the pinned FIFO into ring lines and acked as they land, so the ring (host
// RAM) is the capture's elastic buffer, not the FIFO; one ring per socket stream keeps each ring
// single-writer. A lagging consumer drops its own oldest lines (counted per consumer) and recovers as decode
// recovers from a device-side drop: head adoption, resync counters, timestamps re-anchoring at the next
// absolute zone. An internal audit consumer decodes every stream for the wire-integrity report that ingest, a
// pure copier, cannot produce.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
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

// Receiver-internal record, not the consumer contract: the device's own unpaired start/end markers in 24 B.
// The bit layout (lane at 16, dev at 26, type at 29; ZoneStart=1 / ZoneEnd=2) is pinned by the vectorized
// packer, which is why the public StreamingProfilerRec is a separate type.
enum class StreamingProfilerRawRecType : uint32_t {
    // A complete zone from the atomic wire path: ts is the END and `duration` is set, so no pairing. 0 because the
    // 3-bit type field has 1..7 spoken for.
    Zone = 0,
    ZoneStart = 1,
    ZoneEnd = 2,
    ZoneTotal = 3,
    Data = 4,
    Event = 5,
    Ext = 6,
    Cont = 7,
};
// The vectorized zone packer derives the type from ZoneStart plus the wire's END bit, by addition.
static_assert(
    static_cast<uint32_t>(StreamingProfilerRawRecType::ZoneEnd) ==
    static_cast<uint32_t>(StreamingProfilerRawRecType::ZoneStart) + 1);

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

// One 64 B line of wire words. Frames are line-multiples stored back to back, so every frame starts on a line
// and the BULK_SPAN prefix does the framing.
struct alignas(64) StreamingProfilerRingLine {
    uint32_t w[16];
};
static_assert(sizeof(StreamingProfilerRingLine) == 64);

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

// Mirror of the ELF-resolved PRODUCER-STALL zone ids, one per kernel TU. Open-addressed because the membership
// test runs per marker on the decode walk; a [min,max] pre-screen cannot work since stall ids carry tu_id in
// their high bits.
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
    void remove_consumer(StreamingProfilerConsumerHandle handle);

    // Every relay owning (device, socket) has published done, so the device saw all its bytes acked and the
    // stream retires after one final empty check.
    void notify_producers_done(uint32_t device_index, uint32_t socket_index);

    void shutdown();

    const StreamingProfilerCaptureContext& capture_context() const { return ctx_; }
    // Final per-lane words-consumed mirrors, valid after shutdown(); the completeness check compares them
    // against the workers' own tails.
    std::vector<uint32_t> final_lane_heads(uint32_t device_index) const;
    void log_report() const;

private:
    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        int ring_node = -1;  // node this stream's ring is bound to, and that its ingest thread runs on
        std::unique_ptr<BroadcastRing<StreamingProfilerRingLine>> ring;
        // Written by the audit consumer, never by ingest.
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
        StreamingProfilerRecordCallback cb;
        StreamingProfilerConsumerHandle handle = 0;
        // The internal wire auditor: decodes every stream, owns the decode-quality fields on Stream, has no callback.
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
