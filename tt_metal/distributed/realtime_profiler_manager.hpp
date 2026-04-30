// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {

class IDevice;
class Program;
class RealtimeProfilerTracyHandler;

namespace distributed {

class D2HSocket;
class MeshDevice;

// L1 carve-out addresses for the reserved RT-profiler tensix core. The ring buffer
// (BRISC->NCRISC handoff) and the D2H socket sender config sit in a single carve-out
// anchored past dispatch_mem_map's UNRESERVED, bypassing the user-space allocator.
struct RealtimeProfilerCoreL1Addrs {
    uint32_t base = 0;
    uint32_t ring_buffer = 0;
    uint32_t socket_config = 0;
};

// Owns the full RT-profiler subsystem for a single MeshDevice: per-device state, the
// background receiver thread, the host-side Tracy handler, and the host-device sync
// handshake.
//
// Lifecycle:
//   * Constructor runs the eligibility gate, brings up D2H sockets and BRISC/NCRISC
//     kernels on each eligible device, and starts the receiver thread.
//   * shutdown() (or destruction) signals receiver termination, joins the thread, and
//     drops the Tracy handler. Idempotent.
//   * trigger_sync_check() pauses the receiver, runs a sync handshake on each device,
//     and resumes the receiver. Called from the FD command queue's finish path.
class RealtimeProfilerManager {
public:
    explicit RealtimeProfilerManager(const std::shared_ptr<MeshDevice>& mesh_device);
    ~RealtimeProfilerManager();

    RealtimeProfilerManager(const RealtimeProfilerManager&) = delete;
    RealtimeProfilerManager& operator=(const RealtimeProfilerManager&) = delete;
    RealtimeProfilerManager(RealtimeProfilerManager&&) = delete;
    RealtimeProfilerManager& operator=(RealtimeProfilerManager&&) = delete;

    // Idempotent: writes terminate flag, joins receiver thread, releases Tracy handler,
    // and notifies deactivation. Safe to call multiple times.
    void shutdown();

    // Pauses receiver, performs host-device sync handshake, resumes receiver. No-op
    // when no devices are active or the Tracy handler has been released.
    void trigger_sync_check();

    // First active device's D2H socket, or nullptr if no device is active.
    D2HSocket* get_socket() const;

private:
    struct DeviceState {
        IDevice* device = nullptr;
        uint32_t chip_id = 0;
        MeshCoordinate mesh_coord = MeshCoordinate(0);
        CoreCoord realtime_profiler_core;
        std::unique_ptr<D2HSocket> socket;
        // Owns the BRISC+NCRISC realtime-profiler program so its kernels remain alive for
        // the lifetime of the manager. If this goes out of scope while the kernels are
        // still running, downstream tooling (e.g. tt-inspector) loses the kernel metadata.
        std::unique_ptr<Program> realtime_profiler_program;
        RealtimeProfilerCoreL1Addrs core_l1;
        uint64_t first_timestamp = 0;
        int64_t sync_host_start = 0;
        double sync_frequency = 0.0;
        uint32_t realtime_profiler_base_addr = 0;
        uint32_t sync_request_addr = 0;
        uint32_t sync_host_ts_addr = 0;
        std::atomic<bool> sync_response_received{true};
        int64_t sync_host_time_before = 0;

        DeviceState();
        ~DeviceState();
        DeviceState(DeviceState&& o) noexcept;
        DeviceState& operator=(DeviceState&&) = delete;
        DeviceState(const DeviceState&) = delete;
        DeviceState& operator=(const DeviceState&) = delete;
    };

    void run_sync(DeviceState& dev_state, uint32_t num_samples);

    std::vector<DeviceState> devices_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> paused_{false};
    std::unique_ptr<RealtimeProfilerTracyHandler> tracy_handler_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
