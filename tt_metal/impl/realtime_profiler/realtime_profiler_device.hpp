// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "context/context_types.hpp"
#include "tt_metal/common/ring_buffer.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/realtime_profiler/device_clock_sync.hpp"

namespace tt::tt_metal {

class IDevice;
class Program;

namespace distributed {
class D2HSocket;
class MeshDevice;
}  // namespace distributed

// L1 carve-out addresses (ring buffer + D2H socket config) for the reserved RT-profiler tensix, anchored past
// UNRESERVED to bypass the user-space allocator.
struct RealtimeProfilerCoreL1Addrs {
    uint32_t ring_buffer = 0;
    uint32_t socket_config = 0;
};

// Real-time profiler runtime constants. On-device L1 layout sizes are reused from
// realtime_profiler_ring_buffer.hpp so host and device share a single source of truth.
struct RealtimeProfilerRuntimeSizes {
    static constexpr uint32_t fifo_pages = 32768;                  // host D2H FIFO depth, in pages
    static constexpr uint32_t page_size = RT_PROFILER_ENTRY_SIZE;  // host page size == ring entry size
    static constexpr uint32_t page_words = page_size / sizeof(uint32_t);
    static constexpr uint32_t fifo_size = fifo_pages * page_size;  // pinned-host FIFO, in bytes (2 MiB)
    static constexpr uint32_t core_l1_size = sizeof(RealtimeProfilerCoreL1);
};

// A decoded record waiting for the chord around its end timestamp to be finalized (one more probe).
struct PendingRealtimeRecord {
    uint64_t start_timestamp = 0;
    uint64_t end_timestamp = 0;
    uint32_t runtime_id = 0;
};

// One profiler-enabled chip, fully brought up: eligibility passed, clock register mapped, D2H socket created,
// and the BRISC+NCRISC kernels launched on the reserved tensix.
struct RealtimeProfilerDevice {
    IDevice* device = nullptr;
    uint32_t chip_id = 0;
    CoreCoord realtime_profiler_core;
    std::unique_ptr<distributed::D2HSocket> socket;
    std::unique_ptr<Program> realtime_profiler_program;
    RealtimeProfilerCoreL1Addrs core_l1;
    std::unique_ptr<DeviceClockSync> clock_sync;
    // The receiver's consumer-side handle onto clock_sync (declared after it: destroyed first).
    std::unique_ptr<DeviceClockSync::View> clock_view;
    // Sized to a full FIFO of records: overflow then means holdback exceeded an entire FIFO of
    // past-watermark records — a probe outage longer than a FIFO fill — where the evicted records
    // earn fallback pricing anyway. Ends are monotone (dispatch_s stamps serially).
    RingBuffer<PendingRealtimeRecord> pending_records{RealtimeProfilerRuntimeSizes::fifo_pages};
    // Pages consumed but not yet acked to the device (see the receiver's kAckBatchPages).
    uint32_t unacked_pages = 0;
    bool fifo_capacity_warned = false;

    RealtimeProfilerDevice();
    ~RealtimeProfilerDevice();
    RealtimeProfilerDevice(RealtimeProfilerDevice&& o) noexcept;
    RealtimeProfilerDevice& operator=(RealtimeProfilerDevice&&) = delete;
    RealtimeProfilerDevice(const RealtimeProfilerDevice&) = delete;
    RealtimeProfilerDevice& operator=(const RealtimeProfilerDevice&) = delete;
};

// Devices failing the eligibility gate or socket creation are skipped, so the result may be
// empty. The caller registers each device's clock sync with its ProbeScheduler; nothing here
// retains a pointer into the result, so a mid-loop throw unwinds cleanly.
std::vector<RealtimeProfilerDevice> initialize_realtime_profiler_devices(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id);

}  // namespace tt::tt_metal
