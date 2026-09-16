// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
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
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver() override;

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    std::span<const ProducerStream> streams() const override { return streams_view_; }
    const CaptureContext& capture_context() const override { return ctx_; }
    const DeviceClock& clock(uint32_t dev) const override { return devices_[dev].clock; }
    uint64_t live_head(uint32_t stream) const override;
    void finish_stream(uint32_t stream, uint64_t dropped_bytes, const StreamStats& stats) override;

private:
    Receiver(std::unique_ptr<Devices> relays, std::vector<CapturedDevice> devices);

    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        std::span<const std::byte> fifo;  // the socket's host FIFO: `capacity` pages of SPSC_SPAN_PAGE_WORDS
        uint64_t capacity = 0;            // a power of two
        std::atomic<RelayState> relay{RelayState::Running};
        bool retired = false;

        uint64_t arrived = 0, consumed = 0, acked = 0;  // absolute pages: landed, walked, credited back
        uint64_t fullest = 0;                    // most pages ever awaiting credit; capacity means the device waited
        std::atomic<uint64_t> arrived_bytes{0};  // `arrived`, for consumers widening the device's head
        std::atomic<uint64_t> walked_bytes{0};   // `consumed`: the consumers' walk limit and resync point
        uint64_t frames = 0;
        std::vector<std::atomic<uint64_t>> marks;  // first frame boundary per kMarkBytes; UINT64_MAX until written
        uint64_t mark_block = UINT64_MAX;
        uint64_t consumer_dropped = 0;  // frame bytes, the most any consumer missed of this stream
        StreamStats consumer_stats;     // per field, the most any consumer counted

        const std::byte* page(uint64_t p) const {
            return fifo.data() + (p & (capacity - 1)) * kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
        }
    };

    void ingest_thread(std::vector<Stream*> streams);
    bool poll(Stream& s);
    bool walk_frame(Stream& s);
    bool publish(Stream& s);
    bool settle(Stream& s);
    Stream& stream(uint32_t device_index, uint32_t socket_index);
    void log_report() const;

    std::unique_ptr<Devices> relays_;
    std::vector<CapturedDevice> devices_;
    CaptureContext ctx_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ProducerStream> streams_view_;
    std::vector<std::thread> ingest_threads_;
    std::mutex stats_mu_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
