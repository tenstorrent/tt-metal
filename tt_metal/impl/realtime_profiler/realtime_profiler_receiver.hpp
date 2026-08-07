// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "context/context_types.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_clock_sync.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_service.hpp"

namespace tt::tt_metal {

class IDevice;
class Program;
class DataCollector;

namespace distributed {
class D2HSocket;
class MeshDevice;
}  // namespace distributed

using experimental::ProgramRealtimeRecord;

// L1 carve-out addresses (ring buffer + D2H socket config) for the reserved RT-profiler tensix, anchored past
// UNRESERVED to bypass the user-space allocator.
struct RealtimeProfilerCoreL1Addrs {
    uint32_t ring_buffer = 0;
    uint32_t socket_config = 0;
};

// The record producer for one MeshDevice: owns the per-device sockets, the record ring, and the receiver thread behind
// them, and publishes what it drains to whatever consumers the service has attached.
class RealtimeProfilerReceiver : public ProgramRecordProducer {
public:
    // Null when no local device passed the eligibility gate; a constructed receiver always has devices to drain.
    static std::unique_ptr<RealtimeProfilerReceiver> create(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~RealtimeProfilerReceiver();

    RealtimeProfilerReceiver(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver& operator=(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver(RealtimeProfilerReceiver&&) = delete;
    RealtimeProfilerReceiver& operator=(RealtimeProfilerReceiver&&) = delete;

    // Idempotent.
    void shutdown();

    size_t max_batch_records() const override;
    RealtimeProfilerRecordRing::Reader make_reader() override;
    void wait_until_no_readers() override;

    // All-time, so it answers "did the FIFO ever reach capacity" and nothing else. A run that spikes once reads the
    // spike forever after, which cannot distinguish a device that recovered from one that never did.
    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    // High-water since the previous call, which is what a periodic report wants: reading it clears it, so exactly one
    // caller can use it.
    uint32_t take_peak_fifo_pages() { return peak_fifo_pages_since_report_.exchange(0, std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const override { return num_published_records_.load(std::memory_order_relaxed); }
    uint64_t num_published_batches() const { return num_published_batches_.load(std::memory_order_relaxed); }
    // Records rejected at decode as corrupt: an end timestamp before its start, or timestamps predating every
    // retained clock probe
    uint64_t num_malformed_records() const { return num_malformed_records_.load(std::memory_order_relaxed); }
    // Blocking device L1 read across every chip; not for a latency-sensitive path.
    uint32_t read_ring_full_wait_count();
    size_t num_active_devices() const { return devices_.size(); }

private:
    struct DeviceState {
        IDevice* device = nullptr;
        uint32_t chip_id = 0;
        CoreCoord realtime_profiler_core;
        std::unique_ptr<distributed::D2HSocket> socket;
        // Keeps the BRISC+NCRISC kernels (and their tt-inspector metadata) alive for the receiver's lifetime.
        std::unique_ptr<Program> realtime_profiler_program;
        RealtimeProfilerCoreL1Addrs core_l1;
        bool fifo_reached_capacity = false;
        // Held by pointer so moving DeviceState does not copy the sync object's 128KB probe ring.
        std::unique_ptr<RealtimeProfilerClockSync> clock_sync;

        DeviceState();
        ~DeviceState();
        DeviceState(DeviceState&& o) noexcept;
        DeviceState& operator=(DeviceState&&) = delete;
        DeviceState(const DeviceState&) = delete;
        DeviceState& operator=(const DeviceState&) = delete;
    };

    RealtimeProfilerReceiver(ContextId context_id, std::vector<DeviceState> devices);

    // Devices failing the eligibility gate or socket creation are skipped, so the result may be empty.
    static std::vector<DeviceState> initialize_devices(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id);

    void report_sync_cost(std::chrono::steady_clock::time_point now);
    void note_fifo_depth(uint32_t available);
    // Decodes `pages` and publishes them, placed against the probe history. True when anything was published, so the
    // caller knows to wake consumers. `batch` is the caller's scratch, reused so this never allocates.
    bool publish_pages(
        DeviceState& dev_state,
        std::chrono::steady_clock::time_point now,
        std::span<const uint32_t> pages,
        std::vector<ProgramRealtimeRecord>& batch);

    struct DrainResult {
        uint32_t pages = 0;
        bool published = false;
    };

    // Receiver thread body.
    void run();
    uint64_t run_loop(std::vector<uint32_t>& page_buf);
    uint64_t drain_on_shutdown(std::vector<uint32_t>& page_buf);
    // `now` is re-read as devices are drained, so a device late in a long pass isn't gated on a stale timestamp.
    uint32_t drain_all_devices(std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf);
    // Reads, probes, then publishes -- in that order, which is what makes the batch's bracketing pair exist.
    DrainResult drain_device_pages(
        DeviceState& dev_state, std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf);

    // Owning MeshDevice's ContextId; all MetalContext access must go through instance(context_id_) so a non-default
    // context doesn't leak to silicon DEFAULT_CONTEXT_ID. See #38445 / #39849.
    ContextId context_id_;
    const DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* realtime_profiler_service_ = nullptr;

    std::vector<DeviceState> devices_;
    RealtimeProfilerRecordRing ring_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};

    std::atomic<uint32_t> peak_fifo_pages_{0};               // all-time peak D2H FIFO usage
    std::atomic<uint32_t> peak_fifo_pages_since_report_{0};  // peak since take_peak_fifo_pages()
    uint32_t fifo_pages_window_max_ = 0;         // peak since the last Tracy plot sample; that plot is its only reader
    std::chrono::nanoseconds pass_sync_busy_{};  // clock-read time in the pass just finished
    std::chrono::steady_clock::time_point last_drain_gap_warn_{};
    std::atomic<uint64_t> num_published_records_{0};  // records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // batches published to the ring
    std::atomic<uint64_t> num_malformed_records_{0};  // rejected at decode as unmappable

    // One pass's decoded records, published as a batch. Owned here rather than per device because only one device is
    // being drained at a time, and preallocated because the drain thread must never touch the allocator.
    std::vector<ProgramRealtimeRecord> publish_batch_;

    std::chrono::steady_clock::time_point last_malformed_warn_{};
    std::chrono::steady_clock::time_point last_sync_cost_report_{};
    RealtimeProfilerClockSync::Cost sync_cost_at_last_report_;
};

}  // namespace tt::tt_metal
