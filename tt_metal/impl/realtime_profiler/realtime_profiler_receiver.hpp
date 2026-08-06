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
#include <tt-metalium/mesh_coord.hpp>

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

    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const override { return num_published_records_.load(std::memory_order_relaxed); }
    uint64_t num_published_batches() const { return num_published_batches_.load(std::memory_order_relaxed); }
    // Records dropped before publication because their end timestamp preceded their start
    uint64_t num_malformed_records() const { return num_malformed_records_.load(std::memory_order_relaxed); }
    // Blocking device L1 read across every chip; not for a latency-sensitive path.
    uint32_t read_ring_full_wait_count();
    size_t num_active_devices() const { return devices_.size(); }

private:
    struct DeviceState {
        IDevice* device = nullptr;
        uint32_t chip_id = 0;
        distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0);
        CoreCoord realtime_profiler_core;
        std::unique_ptr<distributed::D2HSocket> socket;
        // Keeps the BRISC+NCRISC kernels (and their tt-inspector metadata) alive for the receiver's lifetime.
        std::unique_ptr<Program> realtime_profiler_program;
        RealtimeProfilerCoreL1Addrs core_l1;
        bool fifo_reached_capacity = false;
        uint32_t consecutive_resync_failures = 0;
        std::chrono::steady_clock::time_point next_poll_at{};
        // Held by pointer so DeviceState stays movable: the sync object carries atomics and cannot be.
        std::unique_ptr<RealtimeProfilerClockSync> clock_sync;

        // Records decoded but not yet published, waiting for the anchor that closes the interval they ran in. Their
        // host-facing fields are unset until then.
        std::vector<ProgramRealtimeRecord> staged;

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
    void bring_up_device_clocks();

    // Called on the drain loop immediately before draining dev_state, so a device's sync interval is bounded by its
    // own drain rather than by every other device's. Returns the time the clock read blocked for, zero when none was
    // due.
    std::chrono::nanoseconds sync_device(DeviceState& dev_state, std::chrono::steady_clock::time_point now);
    // Called from report_sync_cost, not from the drain loop: a device that has stopped answering is reported at
    // most once per warning interval, so scanning every device for it on every pass is work the drain never needs.
    void report_stalled_syncs(std::chrono::steady_clock::time_point now);
    void report_sync_cost(std::chrono::steady_clock::time_point now);
    void stagger_sync_phases();
    void stage_pages(
        DeviceState& dev_state, std::chrono::steady_clock::time_point now, std::span<const uint32_t> pages);
    // True when records were published, so the caller knows to wake consumers.
    bool close_staging(DeviceState& dev_state);

    // Receiver thread body.
    void run();
    uint64_t run_loop(std::vector<uint32_t>& page_buf);
    uint64_t drain_on_shutdown(std::vector<uint32_t>& page_buf);
    // `now` is re-read as devices are drained, so a device late in a long pass isn't gated on a stale timestamp.
    uint32_t drain_all_devices(std::chrono::steady_clock::time_point now, std::vector<uint32_t>& page_buf);
    uint32_t drain_device_pages(
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

    std::atomic<uint32_t> peak_fifo_pages_{0};  // all-time peak D2H FIFO usage
    uint32_t fifo_pages_window_max_ = 0;        // peak since the last Tracy plot sample
    std::chrono::nanoseconds pass_sync_busy_{};  // clock-read time in the pass just finished
    std::chrono::steady_clock::time_point last_drain_gap_warn_{};
    std::atomic<uint64_t> num_published_records_{0};  // records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // batches published to the ring
    std::atomic<uint64_t> num_malformed_records_{0};  // dropped at decode for having end < start

    std::chrono::steady_clock::time_point last_malformed_warn_{};
    std::chrono::steady_clock::time_point last_staging_full_warn_{};
    std::chrono::steady_clock::time_point last_probe_timeout_warn_{};
    std::chrono::steady_clock::time_point last_sync_cost_report_{};
    RealtimeProfilerClockSync::Cost sync_cost_at_last_report_;
};

}  // namespace tt::tt_metal
