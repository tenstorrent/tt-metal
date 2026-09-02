// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler: the host-side control plane for the Blackhole device-zone capture path -- relay
// bring-up (resident kernels, NIU stream-mode flips, D2H socket construction, static TLBs), host<->device
// clock sync and Tracy anchoring, the quiesce, and the teardown completeness check. Everything from the
// socket reads to the record consumers lives in StreamingProfilerReceiver.
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
class MeshBuffer;
}  // namespace distributed
namespace streaming_profiler {
class StreamingProfilerReceiver;
class StreamingProfilerTracyConsumer;
}  // namespace streaming_profiler
class StreamingProfilerTracyHandler;
class Program;
class IDevice;

// One StreamingProfiler per MeshDevice. Constructing it boots the relays on every eligible local Blackhole
// device and starts the receiver; destroying it (or calling stop()) quiesces the relays, drains the
// receiver, verifies capture completeness and leaves the resident idle FW alone (no reset). Teardown talks
// to a relay through its stop word: 1 = quiesce, 2 = release the NIU.
class StreamingProfiler {
public:
    explicit StreamingProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~StreamingProfiler();

    StreamingProfiler(const StreamingProfiler&) = delete;
    StreamingProfiler& operator=(const StreamingProfiler&) = delete;

    // Quiesce the relays, drain the receiver to completion, verify capture completeness, report.
    // Idempotent. The idle FW stays resident (no reset).
    void stop();

private:
    // One DRISC relay per DRAM view, each sweeping an eighth of the worker grid into its own D2H socket.
    static constexpr uint32_t kMaxRelays = 8;
    static constexpr uint32_t kMaxSockets = kMaxRelays;
    static constexpr uint32_t kPageSize = 64;
    static constexpr uint32_t kNRisc = 5;

    struct DeviceCtx {
        uint32_t chip_id = 0;
        std::unique_ptr<distributed::D2HSocket> sockets[kMaxSockets];  // moved into the receiver at start()
        uint32_t nl = 0;  // lanes = num_cores * NRISC (+ n_drisc * NRISC with self-profiling on)
        // Where the DRISC self-profiling lane block starts inside core_virt; 0 when self-profiling is off.
        uint32_t n_worker_cores = 0;
        // Resident for the life of the profiler, and launched outside the command queue: a DRAM-only
        // program touches no fast-dispatch resource, so it can sit there across every user workload, while
        // going through the CQ would deadlock the first Finish().
        std::unique_ptr<Program> relay_program[kMaxRelays];
        IDevice* device = nullptr;
        CoreCoord drisc_logical[kMaxRelays];
        CoreCoord drisc_virtual[kMaxRelays];
        uint64_t drisc_l1_noc[kMaxRelays] = {};  // NoC-addressable base of each DRISC L1 window
        uint32_t drisc_l1_base[kMaxRelays] = {};
        uint32_t stop_addr[kMaxRelays] = {};  // host writes 1 to quiesce, 2 to release the NIU
        uint32_t done_addr[kMaxRelays] = {};  // relay publishes 0xD09E**** once its last page is out
        uint32_t n_drisc = 0;  // relays (= sockets) in use, [1, kMaxRelays]; set once by boot_device
        // core index -> virtual (x,y) [what the SRC lane resolves to]; virtual -> NOC0 (x,y) [Tracy's view].
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
    // Put this DRISC's NIU into stream mode (1) or back to NOC2AXI (0).
    static void set_drisc_niu_mode(IDevice* device, const CoreCoord& drisc_logical, uint32_t stream);
    // Flip several DRISC NIUs in one program launch; doing them one per launch hangs bring-up (see the .cpp).
    static void set_drisc_niu_mode(IDevice* device, const std::vector<CoreCoord>& drisc_logicals, uint32_t stream);
    // Set PROFILER_TERMINATE on every worker so producers stop blocking on a full ring. Required on every
    // path where the relay does not come up: producers are armed by TT_METAL_DEVICE_PROFILER independently
    // of us and are lossless, so with no consumer they block forever and the workload wedges.
    void disarm_producers(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t device_id);
    bool wait_producer_rings_drained(DeviceCtx& ctx, std::chrono::milliseconds budget);
    void disarm_producer_backpressure(DeviceCtx& ctx);
    bool boot_device(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    // After the relays swept to empty and the receiver drained: compare every worker lane's own tail
    // against the receiver's consumed-words mirror.
    void verify_completeness(DeviceCtx& ctx, uint32_t device_index);

    std::vector<DeviceCtx> devices_;
    // GDDR spool: one replicated mesh buffer with one interleaved page per DRAM bank, so the same window is
    // reserved in every bank of every device. Held until stop(); nullptr in direct-push runs.
    std::shared_ptr<distributed::MeshBuffer> spool_buffer_;
    std::unique_ptr<StreamingProfilerTracyHandler> tracy_;
    std::unique_ptr<streaming_profiler::StreamingProfilerTracyConsumer> tracy_consumer_;
    std::unique_ptr<streaming_profiler::StreamingProfilerReceiver> receiver_;
    std::atomic<bool> stopped_{false};
};

}  // namespace tt::tt_metal
