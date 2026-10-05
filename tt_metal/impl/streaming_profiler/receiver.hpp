// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/clock_solver.hpp"

namespace tt {
class Cluster;
namespace umd {
class IoWindow;
}
}  // namespace tt

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace streaming_profiler {

// Reads the root chip's refclk from the host through the PCIe tile's count-from-reset timer. That timer counts the same
// distributed clock as the eth tiles, sits one NoC hop from the PCIe entry, and nothing else reads it, so its high word
// holds between the two halves of a read. It must be destroyed before its device is torn down.
class RootRefclkWindow {
public:
    RootRefclkWindow(tt::Cluster& cluster, uint32_t chip_id);
    ~RootRefclkWindow();

    // The refclk and the host TSC when the window opened.
    const ClockBases& bases() const { return bases_; }
    // The NUMA node of the root chip's PCIe link, or -1 if it has none.
    int numa_node() const { return numa_node_; }
    // Reads the refclk kRefclkBurstReads times back to back.
    RefclkBurst read_burst();

private:
    const int numa_node_;
    std::unique_ptr<tt::umd::IoWindow> window_;
    ClockBases bases_;
};

class Receiver {
public:
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver();

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    // The streams that carry profiler records.
    std::span<const ReceiverStream> streams() const { return streams_view_; }
    const CaptureContext& capture_context() const { return programs_->capture_context(); }
    // The absolute byte position up to which the device has written stream `stream`'s FIFO. Callable from any thread.
    uint64_t live_head(uint32_t stream) const;
    const ClockMap& clock_map() const { return clock_solver_.map(); }

private:
    Receiver(tt::Cluster& cluster, std::unique_ptr<DevicePrograms> programs);

    struct Stream {
        CapturedSocket captured;
        std::span<const std::byte> fifo;  // the socket's host FIFO: `capacity` pages of SPSC_SPAN_PAGE_WORDS
        uint64_t capacity = 0;            // a power of two
        std::atomic<RelayState> relay{RelayState::Running};

        uint64_t arrived = 0, consumed = 0, acked = 0;  // absolute pages: landed, walked, credited back
        uint64_t fullest = 0;                    // most pages ever awaiting credit; capacity means the device waited
        std::atomic<uint64_t> arrived_bytes{0};  // `arrived`, for consumers widening the device's head
        std::atomic<uint64_t> walked_bytes{0};   // `consumed`: the consumers' walk limit and resync point
        uint64_t frames = 0;
        std::vector<std::atomic<uint64_t>> marks;  // first frame boundary per kMarkBytes; UINT64_MAX until written
        uint64_t mark_block = UINT64_MAX;

        const std::byte* page(uint64_t p) const {
            return fifo.data() + (p & (capacity - 1)) * kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
        }
    };

    void ingest_thread(std::vector<Stream*> streams);
    // Feeds the clock solver the clock sync records that the eth relays send on their sync sockets, plus a burst of
    // host reads of the root chip's refclk every kRefclkBurstPeriod. Once stopped, drains the sync streams and finishes
    // the clock map.
    void sync_thread(const std::stop_token& stop);
    // Wakes the sync thread, which sleeps until a sync stream publishes or it is stopped.
    void wake_sync_thread();
    bool poll(Stream& s);
    bool walk_frame(Stream& s);
    // Publishes the stream's walk position, credits walked pages back once the relay has stopped, and returns whether
    // the relay is done.
    bool settle(Stream& s);
    Stream& stream(uint32_t device_index, uint32_t socket_index);
    void log_report() const;

    std::unique_ptr<DevicePrograms> programs_;
    RootRefclkWindow root_refclk_;
    ClockSolver clock_solver_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ReceiverStream> streams_view_;
    std::vector<std::thread> ingest_threads_;
    std::atomic<uint32_t> sync_wake_{0};
    std::jthread sync_thread_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
