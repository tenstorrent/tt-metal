// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "context/context_types.hpp"

namespace tt::tt_metal {

class IDevice;
class Program;
class RealtimeProfilerTracyHandler;

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

// Owns the RT-profiler subsystem for one MeshDevice: per-device state, the receiver thread, the Tracy handler, and the
// host-device sync handshake (sharded across a small worker pool, with a 60s per-chip throttle).
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

    // Runs the sync handshake only on devices due for one (last sync >60s ago, or first finish-path sync after init);
    // pauses the receiver while doing so.
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
        // Owns the BRISC+NCRISC program to keep its kernels (and their metadata for tt-inspector) alive for the
        // manager's lifetime.
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
        // Updated after a successful finish-path or init SYNC_CHECK handshake; used to
        // throttle redundant finish syncs (minimum 60s between attempts per device).
        std::optional<std::chrono::steady_clock::time_point> last_finish_sync_at;
        // First finish-path sync bypasses the throttle so short runs still get a FINISH_SYNC pair; cleared after that
        // handshake.
        bool pending_first_unthrottled_finish_sync = false;

        DeviceState();
        ~DeviceState();
        DeviceState(DeviceState&& o) noexcept;
        DeviceState& operator=(DeviceState&&) = delete;
        DeviceState(const DeviceState&) = delete;
        DeviceState& operator=(const DeviceState&) = delete;
    };

    void run_sync(DeviceState& dev_state, uint32_t num_samples);

    // Mirror the rt-profiler Tracy calibration onto the device profiler's sync anchor so worker zones host-align to the
    // same line as rt records.
    void publish_device_profiler_sync_anchor(
        uint32_t chip_id, double host_anchor, double device_anchor, double frequency, const std::string& core_label);

    // Owning MeshDevice's ContextId; all MetalContext access must go through instance(context_id_) so a non-default
    // context doesn't leak to silicon DEFAULT_CONTEXT_ID. See #38445 / #39849.
    ContextId context_id_;
    std::vector<DeviceState> devices_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> paused_{false};
    std::unique_ptr<RealtimeProfilerTracyHandler> tracy_handler_;
    // Finish-path sync may drain profiler pages from parallel workers; serialize callbacks.
    std::mutex parallel_finish_sync_callback_mu_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
