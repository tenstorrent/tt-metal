// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf-debug profiler: the host-side home for the Blackhole device-zone capture path.
//
// This is a clean module rather than a graft onto RealtimeProfilerManager: the drain path needs none of the
// manager's legacy baggage (the program-record D2H socket, the reserved-tensix core + dispatch handshake,
// the host<->device sync machinery, the stale 4-word/44-bit-graft decode). It drains the worker per-RISC
// SPSC profiler rings DIRECTLY via the resident drainer firmware and streams device zones to Tracy.
//
// This class is the CONTROL PLANE: drainer bring-up (resident kernels, NIU stream-mode flips, D2H socket
// construction, static TLBs), host<->device clock sync and Tracy anchoring, the quiesce, the device
// results reporting, and the teardown completeness check. Everything from the socket reads to the
// record consumers lives in PerfDebugReceiver; the Tracy sink is one more paired-record consumer.
// Booted once at MeshDevice bring-up (resident); at teardown the host writes 1 to each drainer's stop word
// to quiesce it, then 2 to release its NIU (no reset).
#pragma once

#include <cstdint>
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hostdevcommon/profiler_common.h"

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class MeshCoordinate;
class D2HSocket;
}  // namespace distributed
namespace perf_debug {
class PerfDebugReceiver;
class PerfDebugTracyConsumer;
}  // namespace perf_debug
class PerfDebugTracyHandler;
class Program;
class IDevice;

// One PerfDebugProfiler per MeshDevice. Constructing it boots the drainers on every eligible local
// Blackhole device and starts the receiver; destroying it (or calling stop()) quiesces the drainers,
// drains and shuts down the receiver, and leaves the resident idle FW alone (no reset).
class PerfDebugProfiler {
public:
    explicit PerfDebugProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~PerfDebugProfiler();

    PerfDebugProfiler(const PerfDebugProfiler&) = delete;
    PerfDebugProfiler& operator=(const PerfDebugProfiler&) = delete;

    // Quiesce the drainers, drain the receiver to completion, verify capture completeness, report.
    // Idempotent. The idle FW stays resident (no reset).
    void stop();

private:
    // EIGHT DRISC FILLERS, each sweeping an eighth of the worker grid and pushing frames straight into its
    // own D2H socket. The knee is the filler's scan over its slice (FINDINGS N+28/N+34/N+40), so fillers are
    // the thing to multiply, and every DRAM view with a spare NoC port can host one -- as of the 2026-08
    // UMD/soc-descriptor state all eight views pick distinct spare ports (the view-7-collides-with-view-0
    // and view-2-bringup failures that capped this at six no longer reproduce). Host-facing duty from
    // NoC rows y != 0 rides each filler's own static TLB window (configured at bring-up; the socket asks
    // UMD for it in init_sender_tlb), the same path the two y == 0 movers used before direct push.
    static constexpr uint32_t kNFillers = 8;
    static constexpr uint32_t kNSockets = kNFillers;
    // Host FIFO per socket: TT_METAL_PERF_DEBUG_FIFO_MB, default 64 MiB (host_fifo_bytes() in the .cpp).
    // This FIFO is the pipeline's ONLY elasticity now that the device-DRAM ring is gone; the default
    // matches the per-filler ring the direct push deleted, and growing it to hold a whole capture makes a
    // filler structurally immune to host-drain lag (credit-wait 0 by construction). It is plain mmap +
    // IOMMU host RAM (D2HSocket::init_host_buffer) reached by a full 64-bit NoC/PCIe address -- no
    // TLB-window budget and no channel cap; those belong to the Wormhole hugepage fallback, which
    // Blackhole never takes.
    static constexpr uint32_t kPageSize = 64;
    static constexpr uint32_t kNRisc = 5;

    struct DeviceCtx {
        uint32_t chip_id = 0;
        std::unique_ptr<distributed::D2HSocket> sockets[kNSockets];  // moved into the receiver at start()
        uint32_t nl = 0;  // lanes = num_cores * NRISC (+ n_drisc * NRISC with self-profiling on)
        // Worker cores only, i.e. where the DRISC self-profiling lane block starts inside core_virt. 0 when
        // self-profiling is off, in which case core_virt holds nothing but workers.
        uint32_t n_worker_cores = 0;
        // ---- DRISC drainer ----
        // The program stays alive for the life of the profiler: its kernel is still running. It was
        // launched OUTSIDE the command queue (detail::LaunchProgram with force_slow_dispatch), which is
        // what makes a resident drainer possible at all -- a DRAM-only program touches none of the fast
        // dispatch worker grid or dispatch column, so it can sit there across every user workload. Going
        // through the CQ instead would deadlock the first Finish().
        std::unique_ptr<Program> drain_program[kNFillers];
        IDevice* device = nullptr;
        // Per-DRISC. Each drainer owns a disjoint slice of the worker grid, its own socket and its own L1
        // window -- nothing is shared between them on the device side.
        CoreCoord drisc_logical[kNFillers];
        CoreCoord drisc_virtual[kNFillers];
        uint64_t drisc_l1_noc[kNFillers] = {};  // NoC-addressable base of each DRISC L1 window
        uint32_t drisc_l1_base[kNFillers] = {};
        uint32_t stop_addr[kNFillers] = {};  // host writes 1 to quiesce, 2 to release the NIU
        uint32_t done_addr[kNFillers] = {};  // drainer publishes 0xD09E**** once its last page is out
        uint32_t results_addr[kNFillers] = {};
        uint32_t n_drisc = kNFillers;
        // core_index -> virtual (x,y) [what the SRC lane resolves to], and virtual -> NOC0 (x,y) [Tracy view].
        std::vector<std::pair<uint32_t, uint32_t>> core_virt;
        std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> virt_to_noc0;
        std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
        bool active = false;
        bool clock_synced = false;
        double freq_ghz = 0.0;  // measured sync frequency (cycles/ns); aiclk fallback

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };

    void start(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    // Put this DRISC's NIU into stream mode (1) or back to NOC2AXI (0). Its own program, launched and
    // waited on, because the socket config has to be able to land in DRISC L1 before the drainer runs.
    static void set_drisc_niu_mode(IDevice* device, const CoreCoord& drisc_logical, uint32_t stream);
    // Flip SEVERAL DRISC NIUs in ONE program launch. See the .cpp: doing them one-per-launch is what
    // hung drainer 1's bring-up, because each launch carries a dram_barrier that runs while an EARLIER
    // drainer is already resident in stream mode.
    static void set_drisc_niu_mode(IDevice* device, const std::vector<CoreCoord>& drisc_logicals, uint32_t stream);
    // Set PROFILER_TERMINATE on every worker, so producers stop BLOCKING on a full ring and just proceed.
    // Must be called on every path where the drainer does not come up: producers are armed by
    // TT_METAL_DEVICE_PROFILER independently of us, they are lossless by design, and with no consumer they
    // block forever -- the workload wedges rather than merely losing its capture.
    void disarm_producers(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t device_id);
    bool wait_producer_rings_drained(DeviceCtx& ctx, std::chrono::milliseconds budget);
    void disarm_producer_backpressure(DeviceCtx& ctx);
    bool boot_device(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    // Read the drainer's LIVE state (done word, heartbeat, phase) mid-run and log it. Distinguishes
    // "kernel exited" from "kernel blocked in the credit wait" from "kernel sweeping with nothing to do" --
    // states the end-of-run results block cannot tell apart because it is only published on exit.
    void dump_drainer_state(DeviceCtx& ctx, uint32_t d, const char* why);
    // COMMON-TRIGGER SYNC EVENT: park every drainer in a tight spin, release them together, and let each stamp
    // its own clock -- so the spread in the DRISC-SYNC zones is anchor + render error only. Called from stop(),
    // i.e. after the workload: a parked drainer is not draining, and the lazy zone-name harvest needs the
    // workload's kernels already compiled.
    void fire_sync_events();
    // After the drainers swept-to-empty and the receiver drained: compare every worker lane's own tail
    // against the receiver's consumed-words mirror, so a stop-path regression can never lose the capture
    // tail silently again.
    void verify_completeness(DeviceCtx& ctx, uint32_t device_index);

    std::vector<DeviceCtx> devices_;
    std::unique_ptr<PerfDebugTracyHandler> tracy_;
    std::unique_ptr<perf_debug::PerfDebugTracyConsumer> tracy_consumer_;
    std::unique_ptr<perf_debug::PerfDebugReceiver> receiver_;
    std::atomic<bool> stopped_{false};
    // chip -> the NOC0 coords of that chip's drainer cores, when DRISC self-profiling is on. Filled during
    // boot_device (the only place a drainer's placement is known) and consumed in start() to pre-create their
    // Tracy contexts, which is off the drain hot path.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> self_zone_cores_;
};

}  // namespace tt::tt_metal
