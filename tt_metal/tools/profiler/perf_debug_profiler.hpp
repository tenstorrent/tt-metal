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
    void push_hart_zones();  // map X280 drain-hart spans to Tracy lanes (see .cpp); called from stop()

private:
    // Fixed drain config -- the silicon-validated defaults (see test_x280_realprof / the knee + Tracy sweeps):
    // 2 reader harts + 2 relay harts (dual relay), one 12 MiB D2HSocket FIFO per relay, adaptive per-core drain.
    static constexpr uint32_t kNRead = 2;
    static constexpr uint32_t kNRelay = 2;   // dual relay
    static constexpr uint32_t kNSockets = 2; // one D2H FIFO per relay
    // 12 MiB / socket. RAISED from 4 MiB (1048576 words), which was sized from a "4 MiB knee" measurement
    // that later proved to be 4 MiB's OWN floor, not the hardware's. On a host-bound box 4 MiB pins the FIFO
    // at 100%: relay hostfull 415k -> reader spsc-wait 155M -> producers stall, reader copy% 0.8 (spinning),
    // X280 wall 1921M cyc. At 12 MiB: hostfull 0, spsc-wait 0, copy% 52, wall 30M cyc. Matches the
    // test_x280_realprof --hring default. This is also the CEILING: NOC_2M_WINDOW_COUNT=224 with
    // SOCKET_WIN_BASE=208 leaves 16 TLB windows and the FW maps nwin=ceil((in_off+bytes+64)/2MiB) consecutive
    // windows per socket, so kNSockets * nwin <= 16 (12 MiB -> nwin=7 -> 14). Raising further needs
    // SOCKET_WIN_BASE moved too.
    static constexpr uint32_t kHRingWords = 3145728;
    // Per-read page cap, matching test_x280_realprof's --maxpages default. Without it one read can pull the
    // WHOLE FIFO (196607 pages at 12 MiB) into a single decode pass before the sender is acked, which stalls
    // the relay behind one long host turn; capping keeps read/ack cadence tight.
    static constexpr uint32_t kMaxPagesPerRead = 1024;
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
        // --hartzones equivalent (TT_METAL_PERF_DEBUG_HART_ZONES=1): the X280 drain harts inject their own
        // busy/idle spans IN-BAND. {rdcycle, meta} pairs per hart (START,END alternating); each hart is written
        // by exactly one drain thread, so no lock is needed. Mapped to Tracy at stop() using the per-cluster
        // rdcycle->Tensix calibration that hart0 writes at boot when this mode is on.
        struct HZMark {
            uint64_t rdc;
            uint32_t meta;
        };
        std::vector<std::vector<HZMark>> hz_raw;  // sized kNRead + kNRelay when enabled
        uint64_t nharts = 0;
        // Marker rebase origin (first worker-kernel device ts seen by ANY socket), published so the hart-zone
        // push can share it. The hart spans MUST use the same origin as the markers: the X280 starts draining
        // at MeshDevice bring-up, seconds before the model runs, so rebasing the harts on their own first span
        // shifts their whole lane left of the kernels.
        // Plain (not atomic): DeviceCtx must stay movable, and the only contention is two drain threads both
        // wanting to record "the first marker ts". Whichever wins differs by microseconds -- irrelevant for an
        // origin, so the benign race is preferable to making the struct unmovable.
        uint64_t marker_ts_base = 0;

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
