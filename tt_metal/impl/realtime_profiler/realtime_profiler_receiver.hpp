// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
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

// Owns the RT-profiler producers for one MeshDevice: the per-device sockets, the record ring, and the receiver
// thread behind them. Bring-up fits a device-cycle<->host-time line per device; the receiver then drains the sockets,
// stamps every record it publishes with the mapping current at that moment, and between drains probes each device's
// clock, re-anchoring the ones whose mapping has visibly moved.
class RealtimeProfilerReceiver {
public:
    // Null when no local device passed the eligibility gate; a constructed receiver always has devices to drain.
    static std::unique_ptr<RealtimeProfilerReceiver> create(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~RealtimeProfilerReceiver();

    RealtimeProfilerReceiver(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver& operator=(const RealtimeProfilerReceiver&) = delete;
    RealtimeProfilerReceiver(RealtimeProfilerReceiver&&) = delete;
    RealtimeProfilerReceiver& operator=(RealtimeProfilerReceiver&&) = delete;

    // Idempotent: writes terminate flags, joins the receiver, and drains/detaches the record ring.
    void shutdown();

    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const { return num_published_records_.load(std::memory_order_relaxed); }
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
        // Owns the BRISC+NCRISC program to keep its kernels (and their metadata for tt-inspector) alive for the
        // receiver's lifetime.
        std::unique_ptr<Program> realtime_profiler_program;
        RealtimeProfilerCoreL1Addrs core_l1;
        bool fifo_reached_capacity = false;
        uint32_t consecutive_resync_failures = 0;
        // Held by pointer so DeviceState stays movable: the sync object carries atomics and cannot be.
        std::unique_ptr<RealtimeProfilerClockSync> clock_sync;

        DeviceState();
        ~DeviceState();
        DeviceState(DeviceState&& o) noexcept;
        DeviceState& operator=(DeviceState&&) = delete;
        DeviceState(const DeviceState&) = delete;
        DeviceState& operator=(const DeviceState&) = delete;
    };

    RealtimeProfilerReceiver(ContextId context_id, std::vector<DeviceState> devices);

    // Sets up the D2H socket and launches the BRISC/NCRISC kernels on each eligible local device. Devices failing
    // the eligibility gate or socket creation are skipped, so the result may be empty.
    static std::vector<DeviceState> initialize_devices(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id);
    void bring_up_device_clocks();

    // Runs on the drain loop: one probe per device, offered straight to its clock model. Costs the drain a read per
    // device per interval, far under the drain-gap threshold, and far less than a second thread costs in wakeups.
    void sync_devices(std::chrono::steady_clock::time_point now);

    // Receiver thread body.
    void run();
    uint64_t run_loop(std::vector<uint32_t>& page_buf, std::vector<ProgramRealtimeRecord>& record_buf);
    uint64_t drain_on_shutdown(std::vector<uint32_t>& page_buf, std::vector<ProgramRealtimeRecord>& record_buf);
    // `now` is the instant the pass started, and is what each published record's clock mapping is evaluated against.
    uint32_t drain_all_devices(
        std::chrono::steady_clock::time_point now,
        std::vector<uint32_t>& page_buf,
        std::vector<ProgramRealtimeRecord>& record_buf);
    uint32_t drain_device_pages(
        DeviceState& dev_state,
        std::chrono::steady_clock::time_point now,
        std::vector<uint32_t>& page_buf,
        std::vector<ProgramRealtimeRecord>& record_buf);
    // Records that cannot form a valid duration are dropped rather than delivered.
    void publish_pages(
        const DeviceState& dev_state,
        std::chrono::steady_clock::time_point now,
        std::span<const uint32_t> pages,
        std::vector<ProgramRealtimeRecord>& records);

    // Owning MeshDevice's ContextId; all MetalContext access must go through instance(context_id_) so a non-default
    // context doesn't leak to silicon DEFAULT_CONTEXT_ID. See #38445 / #39849.
    ContextId context_id_;
    const DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* realtime_profiler_service_ = nullptr;

    std::vector<DeviceState> devices_;
    RealtimeProfilerRecordRing ring_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    std::chrono::steady_clock::time_point next_poll_at_{};
    std::chrono::steady_clock::time_point last_excursion_at_{};

    std::atomic<uint32_t> peak_fifo_pages_{0};  // all-time peak D2H FIFO usage
    uint32_t fifo_pages_window_max_ = 0;        // peak since the last Tracy plot sample
    std::chrono::steady_clock::time_point last_drain_gap_warn_{};
    std::atomic<uint64_t> num_published_records_{0};  // records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // batches published to the ring
    std::atomic<uint64_t> num_malformed_records_{0};  // dropped at decode for having end < start

    std::chrono::steady_clock::time_point last_malformed_warn_{};
    std::chrono::steady_clock::time_point last_probe_timeout_warn_{};
};

}  // namespace tt::tt_metal
