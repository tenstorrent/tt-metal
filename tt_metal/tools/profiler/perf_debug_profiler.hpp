// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf-debug profiler: the host-side home for the X280 (Blackhole L2CPU) device-zone capture path.
//
// This is a clean module rather than a graft onto RealtimeProfilerManager: the X280 path needs none of the
// manager's legacy baggage (the program-record D2H socket, the reserved-tensix core + dispatch handshake,
// the host<->device sync machinery, the stale 4-word/44-bit-graft decode). It drains the worker per-RISC
// SPSC profiler rings DIRECTLY via the resident X280 `profzone` firmware and streams device zones to Tracy.
//
// Engine (proven in test_x280_realprof, silicon-verified):
//   boot_profzone (idle-once + active-FW JUMP handoff)  ->  2 D2HSockets (dual relay, 4 MiB each, multi-window)
//   ->  N continuous drain threads (pages -> profzone_decode -> WorkerZonePacket)  ->  RealtimeProfilerTracyHandler.
// Reuses the shared contracts x280_profzone_boot.hpp + x280_profzone_decode.hpp so it can never drift from
// the firmware. Booted once at MeshDevice bring-up (resident); P_STOP at teardown -- the X280 reset is
// released exactly once and never re-asserted (re-asserting reset on a live L2CPU is the reservation-churn
// trigger; only the active FW is (re)loaded via the JUMP handoff).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class D2HSocket;
}  // namespace distributed
class PerfDebugTracyHandler;

namespace profiler {
class X280Driver;
struct ProfzoneDecodeState;
}  // namespace profiler

// One PerfDebugProfiler per MeshDevice. Constructing it boots the X280 drainer on every eligible local
// Blackhole device and starts the drain threads; destroying it (or calling stop()) signals P_STOP, joins
// the threads, and leaves the resident idle FW alone (no reset).
class PerfDebugProfiler {
public:
    explicit PerfDebugProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~PerfDebugProfiler();

    PerfDebugProfiler(const PerfDebugProfiler&) = delete;
    PerfDebugProfiler& operator=(const PerfDebugProfiler&) = delete;

    // Stop draining: set P_STOP on every device, wait for the drainers to quiesce, join the host threads.
    // Idempotent. The idle FW stays resident (no reset).
    void stop();

private:
    // Fixed drain config -- the silicon-validated defaults (see test_x280_realprof / the knee + Tracy sweeps):
    // 2 reader harts + 2 relay harts (dual relay), one 4 MiB D2HSocket FIFO per relay, adaptive per-core drain.
    static constexpr uint32_t kNRead = 2;
    static constexpr uint32_t kNRelay = 2;            // dual relay
    static constexpr uint32_t kNSockets = 2;          // one D2H FIFO per relay
    static constexpr uint32_t kHRingWords = 1048576;  // 4 MiB / socket (multi-window; see the 4 MiB knee finding)
    static constexpr uint32_t kPageSize = 64;
    static constexpr uint32_t kNRisc = 5;

    struct DeviceCtx {
        uint32_t chip_id = 0;
        std::unique_ptr<profiler::X280Driver> driver;
        std::unique_ptr<distributed::D2HSocket> sockets[kNSockets];
        uint64_t params_addr = 0;  // profzone MBOX_PARAMS (P_STOP at teardown)
        uint32_t nl = 0;           // lanes = num_cores * NRISC
        // core_index -> virtual (x,y) [what the SRC lane resolves to], and virtual -> NOC0 (x,y) [Tracy view].
        std::vector<std::pair<uint32_t, uint32_t>> core_virt;
        std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> virt_to_noc0;
        std::unique_ptr<profiler::ProfzoneDecodeState> decode[kNSockets];
        std::thread drain[kNSockets];
        bool active = false;

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };

    void start(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    bool boot_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    void drain_loop(DeviceCtx& ctx, uint32_t sock_idx);

    std::vector<DeviceCtx> devices_;
    std::unique_ptr<PerfDebugTracyHandler> tracy_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stopped_{false};
    std::unordered_map<uint16_t, std::string> zone_names_;  // srcloc hash -> zone name (Tracy)
    std::once_flag names_once_;  // zone names are loaded LAZILY on first drain (after kernels JIT-compile,
                                 // so the zone-source-location log exists) -- not at start()/bring-up.
};

}  // namespace tt::tt_metal
