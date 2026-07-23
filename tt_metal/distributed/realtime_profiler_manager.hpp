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
#include "tt_metal/distributed/realtime_profiler_clock_sync.hpp"

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
        RealtimeProfilerClockSync clock_sync;

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
    void run_init_sync();

    // Receiver thread entry point: drain every device socket, run the sync servo, and publish decoded records to the
    // context-wide service's ring readers.
    void run_receiver();
    uint64_t run_receiver_loop();
    uint64_t drain_receiver_on_shutdown();
    // One drain pass over every device; wakes readers if any pages were read. Returns the number of pages read.
    uint32_t drain_all_devices(std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf);
    // Pages read and records published while draining one device.
    struct DrainCounts {
        uint32_t pages = 0;
        size_t records = 0;
    };
    DrainCounts drain_device_pages(
        DeviceState& dev_state, std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf);
    // Decode program records from drained pages and publish them to the broadcast ring; returns the count published.
    size_t publish_pages(
        const DeviceState& dev_state,
        const uint32_t* page_buf,
        uint32_t num_pages,
        std::vector<tt::ProgramRealtimeRecord>& records);
    void service_offset_servo(std::chrono::steady_clock::time_point now);

    // Owning MeshDevice's ContextId; all MetalContext access must go through instance(context_id_) so a non-default
    // context doesn't leak to silicon DEFAULT_CONTEXT_ID. See #38445 / #39849.
    ContextId context_id_;
    // Host without IOMMU + 64-bit PCIe: the D2H socket and the sync-ACK word both fall back to a CQ-sysmem slot whose
    // device PCIe writes may be non-snooped, so reads must evict the cache line first. Arch+host-wide (not per device),
    // set once at construction. Mirrors d2h_uses_hugepage_fallback().
    bool d2h_hugepage_fallback_ = false;
    const DataCollector* data_collector_ = nullptr;
    RealtimeProfilerService* realtime_profiler_service_ = nullptr;
    std::vector<DeviceState> devices_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};

    // Receiver diagnostics
    std::atomic<uint32_t> peak_fifo_pages_{0};  // all-time peak D2H FIFO usage
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
