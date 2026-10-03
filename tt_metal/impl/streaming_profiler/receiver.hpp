// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/engine.hpp"
#include "impl/streaming_profiler/sync/host_sync.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace streaming_profiler {

class Receiver {
public:
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver();

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    std::span<const ReceiverStream> streams() const { return streams_view_; }
    const CaptureContext& capture_context() const { return programs_->capture_context(); }
    // Callable from any consumer thread.
    uint64_t live_head(uint32_t stream) const;
    const ClockMap& clock_map() const { return map_; }
    ClockMap& clock_map() { return map_; }
    SyncEngine& sync() { return sync_; }
    HostSync& host_sync() { return host_sync_; }

private:
    Receiver(tt::Cluster& cluster, std::unique_ptr<DevicePrograms> programs);

    struct Stream {
        CapturedSocket captured;
        std::span<const std::byte> fifo;
        uint64_t capacity_pages = 0;  // a power of two
        std::atomic<RelayState> relay{RelayState::Running};
        bool retired = false;

        uint64_t arrived_pages = 0, walked_pages = 0, acked_pages = 0;
        uint64_t fullest_pages = 0;
        std::atomic<uint64_t> arrived_bytes{0};
        std::atomic<uint64_t> walked_bytes{0};
        uint64_t frames = 0;
        std::vector<std::atomic<uint64_t>> marks;  // UINT64_MAX until written
        uint64_t mark_block = UINT64_MAX;

        const std::byte* page(uint64_t p) const {
            return fifo.data() + (p & (capacity_pages - 1)) * kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
        }
    };

    void ingest_thread(std::vector<Stream*> streams);
    bool poll(Stream& s);
    bool walk_frame(Stream& s);
    bool settle(Stream& s);
    Stream& stream(uint32_t device_index, uint32_t socket_index);
    void log_report() const;

    std::unique_ptr<DevicePrograms> programs_;
    HostSync host_sync_;
    ClockMap map_;
    SyncEngine sync_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ReceiverStream> streams_view_;
    std::vector<std::thread> ingest_threads_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
