// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>

#include "context/context_types.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"

namespace tt::tt_metal {

class DeviceWallClock;

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

// One profiler-enabled chip, fully brought up: eligibility passed, D2H socket created, and the BRISC+NCRISC kernels
// launched on the reserved tensix.
struct RealtimeProfilerDevice {
    IDevice* device = nullptr;
    uint32_t chip_id = 0;
    distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0);
    CoreCoord realtime_profiler_core;
    // dispatch_s core feeding this device's profiler, and the L1 address of its record_full_wait_count.
    std::optional<CoreCoord> dispatch_s_core;
    uint32_t record_full_wait_count_addr = 0;
    // Dispatch stalls reported for this device (receiver thread only until shutdown joins it).
    uint64_t dispatch_stall_events = 0;
    uint64_t dispatch_stall_cycles = 0;
    std::unique_ptr<distributed::D2HSocket> socket;
    // Owns the BRISC+NCRISC program to keep its kernels (and their metadata for tt-inspector) alive for the
    // receiver's lifetime.
    std::unique_ptr<Program> realtime_profiler_program;
    RealtimeProfilerCoreL1Addrs core_l1;
    bool fifo_reached_capacity = false;
    // Pages read from the FIFO that the device has not been told about yet.
    uint32_t unacked_pages = 0;

    // Host-device clock mapping, kept by the receiver: device time anchor_device (cycles) was host time
    // anchor_host (ticks), and the device clock runs at `frequency` cycles per ns.
    uint64_t anchor_device = 0;
    int64_t anchor_host = 0;
    double frequency = 0.0;
    // The profiler core's wall clock as the host reads it.
    std::unique_ptr<DeviceWallClock> wall_clock;

    RealtimeProfilerDevice();
    ~RealtimeProfilerDevice();
    RealtimeProfilerDevice(RealtimeProfilerDevice&&) noexcept;
};

// Sets up the D2H socket and launches the BRISC/NCRISC kernels on each eligible local device. Devices failing the
// eligibility gate or socket creation are skipped, so the result may be empty. Marks per-device profiler init
// complete either way, so nothing waits on a device that bowed out.
std::vector<RealtimeProfilerDevice> initialize_realtime_profiler_devices(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, ContextId context_id);

}  // namespace tt::tt_metal
