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

    // The relay owning (device, socket) has pushed its last page and waits on its socket barrier: ingest returns
    // every credit from here on instead of holding back the readers' lag budget.
    void notify_producers_drained(uint32_t device_index, uint32_t socket_index);
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

    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        // The socket's FIFO adopted as the ring: the device writes it, readers decode it in place, and ingest_pass
        // publishes the device's progress and returns credits behind the readers' lag budget.
        std::unique_ptr<BroadcastRing<RingLine>> ring;
        uint64_t capacity = 0;  // pages
        uint64_t keep = 0;      // pages a reader may lag before the horizon passes it: capacity minus the runway
        // Written by the audit consumer, never by ingest.
        profiler::SpanDecodeState decode;
        std::vector<uint64_t> last_zone_ts;  // per lane, order invariant (must never regress)
        std::atomic<bool> drained{false};
        std::atomic<bool> producers_done{false};
        bool retired = false;

        uint64_t pages = 0, records = 0, zones = 0;
        uint64_t arrived = 0, acked = 0;  // absolute pages: published to readers, credited back to the device
        uint64_t claim = 0;               // the ring's published overwrite horizon plus capacity
        uint64_t order_regressions = 0, bad_frames = 0, epoch_fixes = 0;
    };

    void decode_thread(std::vector<Stream*> streams);
    // One poll+publish+ack pass over a stream. Returns true if pages arrived; sets s.retired when the stream is
    // finished.
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
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
