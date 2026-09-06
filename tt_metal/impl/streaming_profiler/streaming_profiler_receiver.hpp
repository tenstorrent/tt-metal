// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler host receiver: D2H socket streams are mirrored verbatim into per-stream BroadcastRings,
// which it exposes to the service as a producer; every consumer thread decodes and pairs for itself. Ingest fuses poll
// + frame + copy
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
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"

namespace tt::tt_metal {

namespace distributed {
class D2HSocket;
class MeshDevice;
}
template <typename T>
class BroadcastRing;

namespace streaming_profiler {

// Receiver-internal record, not the consumer contract: the device's own unpaired start/end markers in 24 B.
// The bit layout (lane at 16, dev at 26, type at 29; ZoneStart=1 / ZoneEnd=2) is pinned by the vectorized

// One 64 B line of wire words. Frames are line-multiples stored back to back, so every frame starts on a line
// and the BULK_SPAN prefix does the framing.
struct alignas(64) RingLine {
    uint32_t w[16];
};
static_assert(sizeof(RingLine) == 64);

struct ReceiverDeviceConfig {
    uint32_t chip_id = 0;
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
    uint32_t num_cores = 0;
    std::vector<LaneInfo> lane_table;                   // size num_cores * 5
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // incl. DRISC self-zone cores
    experimental::streaming_profiler::Clock clock;
    int numa_node = -1;  // host node closest to this device; -1 leaves the ring unbound
};

// Mirror of the ELF-resolved PROFILER-STALL zone ids, one per kernel TU. Open-addressed because the membership
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

class Devices;

class Receiver : public Producer {
public:
    // The capture for a MeshDevice: boots the relays on its local devices, attaches to the service and
    // starts ingest. nullptr when no device came up (logged). Destruction ends the capture: relays quiesce, ingest
    // drains, the service's consumers finish the rings, completeness is verified and reported.
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver() override;

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    // Every relay owning (device, socket) has published done, so the device saw all its bytes acked and the
    // stream retires after one final empty check.
    void notify_producers_done(uint32_t device_index, uint32_t socket_index);
    // Final per-lane words-consumed mirrors, valid once ingest has retired; the completeness check compares them
    // against the workers' own tails.
    std::vector<uint32_t> final_lane_heads(uint32_t device_index) const;

    std::span<const ProducerStream> streams() const override { return streams_view_; }
    const CaptureContext& capture_context() const override { return ctx_; }
    std::vector<experimental::streaming_profiler::Clock> clocks() const override;

private:
    explicit Receiver(std::vector<ReceiverDeviceConfig> devices);
    void start();
    void stop();
    // Joins ingest once every stream has retired, then the audit.
    void shutdown();
    void log_report() const;
    struct MappedRegion {
        void* base = nullptr;
        size_t bytes = 0;
        MappedRegion() = default;
        MappedRegion(const MappedRegion&) = delete;
        MappedRegion& operator=(const MappedRegion&) = delete;
        ~MappedRegion();
    };

    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        int ring_node = -1;         // node this stream's ring is bound to
        MappedRegion ring_storage;  // declared before `ring`, which must be destroyed first
        std::unique_ptr<BroadcastRing<RingLine>> ring;
        // Written by the audit consumer, never by ingest.
        profiler::SpanDecodeState decode;
        std::vector<uint64_t> last_zone_ts;  // per lane, order invariant (must never regress)
        std::atomic<bool> producers_done{false};
        bool retired = false;

        uint64_t frames = 0, pages = 0, records = 0, zones = 0;
        uint64_t decode_ticks = 0;
        uint64_t wpos = 0;       // ring lines written
        uint64_t frame_end = 0;  // where the frame under construction ends; == wpos when none is pending
        uint64_t order_regressions = 0, bad_frames = 0, epoch_fixes = 0;
        bool desync_warned = false;
    };

    void create_rings(uint64_t ring_lines);
    void decode_thread(std::vector<Stream*> streams);
    // One poll+decode+ack pass over a stream. Returns true if it moved data; sets
    // s.retired when the stream is finished.
    bool ingest_pass(Stream& s);
    // The wire audit: decodes every stream with a null sink and is the only writer of the decode-quality fields
    // on Stream.
    void audit_thread();

    std::vector<ReceiverDeviceConfig> devices_;
    std::unique_ptr<Devices> relays_;
    bool attached_ = false;  // a producer of the service
    CaptureContext ctx_;
    mutable std::mutex clocks_mu_;
    std::vector<experimental::streaming_profiler::Clock> clocks_;  // indexed like ctx_.devices
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ProducerStream> streams_view_;
    std::vector<std::thread> decode_threads_;
    std::thread audit_thread_;
    std::atomic<bool> audit_drain_{false};
    uint64_t audit_dropped_ = 0;

    std::atomic<bool> stop_{false};
    std::atomic<bool> shutdown_done_{false};
    uint32_t nthreads_ = 0;  // decode threads; stream i belongs to thread i % nthreads_
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
