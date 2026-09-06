// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler host receiver: each D2H socket's FIFO is adopted as a BroadcastRing the device writes in
// place, exposed to the service as a producer; every subscriber decodes for itself. Ingest publishes the device's
// progress and returns credits behind the readers' lag budget, so the FIFO is the capture's elastic buffer. A
// lagging subscriber drops its own oldest lines (counted per subscriber) and recovers as decode recovers from a
// device-side drop: head adoption, resync counters, timestamps re-anchoring at the next absolute zone. When the
// capture detaches, each subscriber reports its final per-stream totals; the fullest one is the wire-integrity
// report and the completeness check's view of what was consumed.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_device.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"

namespace tt::tt_metal {

namespace distributed {
class D2HSocket;
class MeshDevice;
}  // namespace distributed

namespace streaming_profiler {

class Receiver : public Producer {
public:
    // The capture for a MeshDevice: boots the relays on its local devices, attaches to the service and
    // starts ingest. nullptr when no device came up (logged). Destruction ends the capture: relays quiesce, ingest
    // drains, the service's consumers finish the rings, completeness is verified and reported.
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver() override;

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    std::span<const ProducerStream> streams() const override { return streams_view_; }
    const CaptureContext& capture_context() const override { return ctx_; }
    std::span<const experimental::streaming_profiler::Clock> clocks() const override { return clocks_; }
    void finish_stream(uint32_t stream, const StreamStats& stats) override;

private:
    Receiver(std::unique_ptr<Devices> relays, std::vector<CapturedDevice> devices);

    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        // The socket's FIFO adopted as the ring: the device writes it, readers decode it in place, and ingest_pass
        // publishes the device's progress and returns credits behind the readers' lag budget.
        std::unique_ptr<BroadcastRing<RingLine>> ring;
        uint64_t capacity = 0;  // pages
        uint64_t keep = 0;      // pages a reader may lag before the horizon passes it: capacity minus the runway
        std::atomic<bool> drained{false};
        std::atomic<bool> producers_done{false};
        bool retired = false;

        uint64_t arrived = 0, acked = 0;  // absolute pages: published to readers, credited back to the device
        uint64_t claim = 0;               // the ring's published overwrite horizon plus capacity
        StreamStats stats;                // the fullest subscriber's final view
    };

    void ingest_thread(std::vector<Stream*> streams);
    // One poll+publish+ack pass over a stream. Returns true if pages arrived; sets s.retired when the stream is
    // finished.
    bool ingest_pass(Stream& s);
    Stream& stream(uint32_t device_index, uint32_t socket_index);
    // The fullest subscriber's final per-lane words-consumed mirrors; empty when no subscriber decoded the device.
    std::vector<uint32_t> final_lane_heads(uint32_t device_index) const;
    void log_report() const;

    std::unique_ptr<Devices> relays_;
    std::vector<CapturedDevice> devices_;
    CaptureContext ctx_;
    std::vector<experimental::streaming_profiler::Clock> clocks_;  // indexed like ctx_.devices
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ProducerStream> streams_view_;
    std::vector<std::thread> ingest_threads_;
    std::mutex stats_mu_;

    std::atomic<bool> stop_{false};
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
