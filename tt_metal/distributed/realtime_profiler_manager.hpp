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
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/realtime_profiler/realtime_profiler_service.hpp"

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal::experimental {
class PinnedMemory;
}

namespace tt::tt_metal {

class IDevice;
class Program;
class DataCollector;

namespace distributed {

class D2HSocket;
class MeshDevice;

// L1 carve-out addresses (ring buffer + D2H socket config) for the reserved RT-profiler tensix, anchored past
// UNRESERVED to bypass the user-space allocator.
struct RealtimeProfilerCoreL1Addrs {
    uint32_t base = 0;
    uint32_t ring_buffer = 0;
    uint32_t socket_config = 0;
};

// Owns the RT-profiler producers for one MeshDevice: device sockets/state, the receiver thread, the record ring, and
// the host-device sync handshake. Init calibration fits a device-cycle<->host-time line; the receiver then runs a
// free-running servo (kServoInterval) that re-anchors the offset off a one-shot device sync to track clock drift.
class RealtimeProfilerManager {
public:
    explicit RealtimeProfilerManager(const std::shared_ptr<MeshDevice>& mesh_device);
    ~RealtimeProfilerManager();

    RealtimeProfilerManager(const RealtimeProfilerManager&) = delete;
    RealtimeProfilerManager& operator=(const RealtimeProfilerManager&) = delete;
    RealtimeProfilerManager(RealtimeProfilerManager&&) = delete;
    RealtimeProfilerManager& operator=(RealtimeProfilerManager&&) = delete;

    // Idempotent: writes terminate flags, joins the receiver, and drains/detaches the record ring.
    void shutdown();

    // RT-profiler diagnostics
    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const { return num_published_records_.load(std::memory_order_relaxed); }
    uint64_t num_published_batches() const { return num_published_batches_.load(std::memory_order_relaxed); }
    uint32_t ring_full_wait_count() const;  // reads device L1
    size_t num_active_devices() const { return devices_.size(); }

private:
    struct DeviceState {
        IDevice* device = nullptr;
        uint32_t chip_id = 0;
        MeshCoordinate mesh_coord = MeshCoordinate(0);
        CoreCoord realtime_profiler_core;
        std::unique_ptr<D2HSocket> socket;
        // Owns the BRISC+NCRISC program to keep its kernels (and their metadata for tt-inspector) alive for the
        // manager's lifetime.
        std::unique_ptr<Program> realtime_profiler_program;
        RealtimeProfilerCoreL1Addrs core_l1;
        bool fifo_reached_capacity = false;
        uint64_t first_timestamp = 0;
        int64_t sync_host_start = 0;
        double device_cycles_per_host_tick = 0.0;
        double sync_frequency = 0.0;
        // Device cycle at host time 0: device_cycle = sync_frequency * host_ns + device_cycle_offset. Set from the fit
        // in run_sync and re-anchored (slope held fixed) by the servo every kServoInterval so its motion tracks drift.
        int64_t device_cycle_offset = 0;
        // Drift since the last re-anchor (host_real - host_predicted), ns; the drift term of clock_sync.sync_error_ns.
        int64_t sync_tracking_error_ns = 0;
        // Round trip of the most recent handshake, host ticks; the re-anchor places device_time at the midpoint
        // (+RTT/2).
        int64_t sync_rtt_ticks = 0;
        uint32_t sync_host_ts_addr = 0;
        // L1 address of the device's published WALL_CLOCK [lo, hi]; the host reads it on the fast ACK path to
        // re-anchor.
        uint32_t sync_device_time_addr = 0;
        uint32_t sync_seq = 0;  // monotonic per-handshake token (never 0); distinct so a stale ACK can't false-match
        // Pinned host word the device NOC-writes the token into (device->host, bypassing the record FIFO); the host
        // polls it locally to time the handshake round trip. Backing + PinnedMemory held for the manager's lifetime.
        std::shared_ptr<uint32_t[]> ack_host_backing;
        std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> ack_pinned;
        volatile uint32_t* ack_host_ptr = nullptr;
        int64_t sync_host_time_before = 0;
        // Cached UMD TLB window to this device's profiler core, resolved once at init on architectures that map L1
        // statically (Blackhole). When set, the sync timestamp is written with a single MMIO store instead of
        // WriteToDeviceL1; null elsewhere (see write_sync_timestamp). Owned by UMD's TLBManager, not us.
        tt::umd::TlbWindow* sync_tlb = nullptr;
        // Updated after each successful re-anchor or init SYNC_CHECK handshake; used to pace the servo to at most one
        // re-anchor per kServoInterval per device.
        std::optional<std::chrono::steady_clock::time_point> last_finish_sync_at;
        enum class FinishSyncPhase : uint8_t { Idle, AwaitingResponse };
        FinishSyncPhase finish_sync_phase = FinishSyncPhase::Idle;
        std::chrono::steady_clock::time_point finish_sync_deadline;

        DeviceState();
        ~DeviceState();
        DeviceState(DeviceState&& o) noexcept;
        DeviceState& operator=(DeviceState&&) = delete;
        DeviceState(const DeviceState&) = delete;
        DeviceState& operator=(const DeviceState&) = delete;
    };

    // Set up the D2H socket and launch the BRISC/NCRISC kernels on each eligible local device. Devices failing the
    // eligibility gate or socket creation are skipped.
    void initialize_devices(const std::shared_ptr<MeshDevice>& mesh_device);
    void run_sync(DeviceState& dev_state, uint32_t num_samples);
    void run_init_sync();

    // Re-anchor dev_state.device_cycle_offset from a fresh (host_anchor, device_anchor) sync point, holding the fitted
    // slope fixed. Sets sync_tracking_error_ns to the pre-update residual (host_real - host_predicted). Run every servo
    // tick; a plain offset re-anchor beat a 2-state Kalman on the p99 tail in ablation (the filter rang on AICLK
    // excursions), so the mapping tracks drift purely by re-anchoring often.
    void reanchor_device_cycle_offset(DeviceState& dev_state, int64_t host_anchor, uint64_t device_anchor);

    // Round trip of a sync handshake: after write_sync_timestamp(host_time_id), busy-poll the pinned host word the
    // device NOC-writes the token into (bypassing the record FIFO) until it reads host_time_id, timed from
    // host_before. Upper-bounds the one-way host->device latency by causality. Returns 0 if no host ACK word.
    int64_t measure_sync_rtt_ticks(const DeviceState& dev_state, int64_t host_before, uint32_t host_time_id);

    // Publish dev_state's current calibration to the process-wide per-chip cache so a rapid MeshDevice reopen can
    // reuse it (see kRtProfilerMinSyncInterval) instead of re-running the host-device sync.
    void cache_calibration(const DeviceState& dev_state);

    // Receiver thread entry point: drain every device socket, advance the finish-sync handshake, and publish decoded
    // records to the context-wide service's ring readers.
    void run_receiver();
    uint64_t run_receiver_loop();
    uint64_t drain_receiver_on_shutdown();
    // One drain pass over every device; wakes readers if any pages were read. Returns the number of pages read.
    uint32_t drain_all_devices(
        bool scan_sync_marker, std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf);
    // Pages read and records published while draining one device.
    struct DrainCounts {
        uint32_t pages = 0;
        size_t records = 0;
    };
    DrainCounts drain_device_pages(
        DeviceState& dev_state,
        bool scan_sync_marker,
        std::vector<uint32_t>& page_buf,
        std::vector<tt::ProgramRealtimeRecord>& record_buf);
    // Decode program records from drained pages and publish them to the broadcast ring; returns the count published.
    size_t publish_pages(
        const DeviceState& dev_state,
        const uint32_t* page_buf,
        uint32_t num_pages,
        std::vector<tt::ProgramRealtimeRecord>& records);
    // Writes the 32-bit host timestamp to the profiler core for a sync handshake, via the cached TLB window when
    // available (one MMIO store) or WriteToDeviceL1 otherwise. The host->device latency of this write is the
    // sync-error floor, so the fast path measurably tightens it (~3x lower jitter on Blackhole).
    static void write_sync_timestamp(DeviceState& dev_state, uint32_t value);
    [[nodiscard]] bool has_active_finish_sync() const;
    void start_finish_syncs(std::chrono::steady_clock::time_point now);
    void advance_finish_sync(DeviceState& dev_state, std::chrono::steady_clock::time_point now);
    void service_finish_sync(std::chrono::steady_clock::time_point now, bool allow_start);

    // Owning MeshDevice's ContextId; all MetalContext access must go through instance(context_id_) so a non-default
    // context doesn't leak to silicon DEFAULT_CONTEXT_ID. See #38445 / #39849.
    ContextId context_id_;
    const DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* realtime_profiler_service_ = nullptr;
    std::vector<DeviceState> devices_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> finish_sync_busy_{false};

    // Receiver diagnostics
    std::atomic<uint32_t> peak_fifo_pages_{0};  // all-time peak D2H FIFO usage (diagnostics getter)
    // Max D2H FIFO occupancy observed since the last diagnostics-plot sample; reset each plot tick so the plot shows
    // per-window backlog peaks instead of a monotonic high-water mark. Receiver-thread only (updated on drain,
    // read+reset in the plot block), so no atomic needed.
    uint32_t fifo_pages_window_max_ = 0;
    std::atomic<uint64_t> num_published_records_{0};  // count of records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // count of batches published to the ring

    static constexpr size_t kMaxConsumerBatchPerDevice = 1u << 15;  // max batch size per device
    static constexpr size_t kMaxConsumerBatchCap = 1u
                                                   << 20;  // hard cap on total batch size (corresponds to 32 devices)
    static constexpr size_t kRingHeadroomBatches = 4;
    static constexpr size_t kMaxRingCapacity = 1u << 22;

    std::optional<RealtimeProfilerRecordRing> ring_;
    bool ring_attached_ = false;
};

}  // namespace distributed
}  // namespace tt::tt_metal
