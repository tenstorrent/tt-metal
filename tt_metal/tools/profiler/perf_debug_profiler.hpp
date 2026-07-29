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
#include <span>
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
struct RecRingHolder;  // pimpl for BroadcastRing<PerfDebugRec> (keeps the ring header out of this one)

namespace profiler {
class X280Driver;
struct ProfzoneDecodeState;
}  // namespace profiler

// Decoded device record handed writer -> reader. Layout mirrors test_x280_realprof's Rec exactly (ts first
// packs it to 24 B instead of 32 padded), so both paths move the same bytes per record.
struct PerfDebugRec {
    uint64_t ts;
    uint32_t lane;
    uint32_t type;
    uint32_t zone;
    uint32_t prog;
};

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
    static constexpr uint32_t kNRelay = 2;    // dual relay
    static constexpr uint32_t kNSockets = 2;  // one D2H FIFO per relay
    // 12 MiB / socket. RAISED from 4 MiB (1048576 words), which was sized from a "4 MiB knee" measurement
    // that later proved to be 4 MiB's OWN floor, not the hardware's. On a host-bound box 4 MiB pins the FIFO
    // at 100%: relay hostfull 415k -> reader spsc-wait 155M -> producers stall, reader copy% 0.8 (spinning),
    // X280 wall 1921M cyc. At 12 MiB: hostfull 0, spsc-wait 0, copy% 52, wall 30M cyc. Matches the
    // test_x280_realprof --hring default. This is also the CEILING: NOC_2M_WINDOW_COUNT=224 with
    // SOCKET_WIN_BASE=208 leaves 16 TLB windows and the FW maps nwin=ceil((in_off+bytes+64)/2MiB) consecutive
    // windows per socket, so kNSockets * nwin <= 16 (12 MiB -> nwin=7 -> 14). Raising further needs
    // SOCKET_WIN_BASE moved too.
    static constexpr uint32_t kHRingWords = 3145728;
    // Per-read page cap. 0 = UNCAPPED (take whatever the FIFO holds, bounded only by fifo_pages-1).
    // Overridable at runtime with TT_METAL_PERF_DEBUG_MAX_PAGES.
    //
    // Max pages pulled per read() (kPageSize = 64 B, so 1024 pages = 64 KB). Override:
    // TT_METAL_PERF_DEBUG_MAX_PAGES. 0 = uncapped.
    //
    // This has been 1024, then 0, and is now 1024 again. Both changes were right for their time, and the
    // reason it flipped back is worth keeping:
    //
    // ERA 1 -- inline Tracy push on the drain thread. A small cap multiplied a huge fixed per-read cost
    // (socket read + decode + the Tracy push, all on one thread), so the FIFO stayed full and the RELAY sat
    // in HOST-WAIT, back-pressuring the reader into the worker cores. Measured on UFLD-v2 (~99M markers),
    // busier socket:
    //     cap 1024 (64 KB)  -> 24,669 producer stall zones, 9,855 reads
    //     cap 16384 (1 MB)  ->  9,223 producer stall zones,   617 reads
    //     cap 0 (uncapped)  ->    783 producer stall zones,    54 reads
    // Monotonic, hence uncapped.
    //
    // ERA 2 -- sink decoupled onto the BroadcastRing (the push moved to the consumer thread). That removed
    // the term the cap was multiplying, and uncapped became the WORSE choice, because it trades a throughput
    // problem for a LATENCY one: an unbounded read swallows a whole batch and spends ~10 ms in
    // read+decode+publish+resize with the FIFO never polled, so the relay fills 12 MiB, blocks, and stalls
    // every producer once. Measured at the knee (test_perf_debug_zones --delay 950, full grid):
    //     uncapped -> 557 producer stall zones (exactly 1 per lane),   3 reads/socket, decode 6,048 us/pass
    //     cap 1024 ->   0 producer stall zones,                     1107 reads/socket, decode    17 us/pass
    // And the ERA-1 cost is gone: re-measured 2026-07-29, BOTH models are clean either way --
    //     UFLD-v2   uncapped 0 stalls / 97,627 reads   vs cap 1024 0 stalls / 100,595 reads (99,187,072
    //               markers, bit-identical, 0 drops, iter 1.880 vs 1.876 ms)
    //     ResNet-50 uncapped 0 stalls /  2,960 reads   vs cap 1024 0 stalls /   3,569 reads (1,581,952
    //               markers, bit-identical, 0 drops, iter 1.7056 vs 1.7165 ms)
    // At real-model rates the FIFO rarely holds 1024 pages, so the cap barely binds -- it costs nothing and
    // buys ~75 delay units of margin at the knee. See FINDINGS SS27.
    //
    // KNOWN LIMITATION: a PAGE cap does not bound per-pass TIME. Once data is plentiful every read is a full
    // cap-sized read, which is why caps of 64 / 256 / 1024 / 4096 are indistinguishable below the knee. The
    // correct fix is to bound elapsed time per pass; this is the stopgap that behaves well at observed rates.
    // Note also the ack is issued by read() itself, so a bigger read acks MORE data sooner, not later.
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
        bool active = false;
        // --hartzones equivalent (TT_METAL_PERF_DEBUG_HART_ZONES=1): the X280 drain harts inject their own
        // busy/idle spans IN-BAND. {rdcycle, meta} pairs per hart (START,END alternating); each hart is written
        // by exactly one drain thread, so no lock is needed. Mapped to Tracy at stop() using the per-cluster
        // rdcycle->Tensix calibration that hart0 writes at boot when this mode is on.
        struct HZMark {
            uint64_t rdc;
            uint32_t meta;
        };
        // Per-socket state that used to live as locals in the old per-socket drain thread. drain_pass() is
        // now one pass called repeatedly by the single writer thread, so it has to persist here.
        struct SockState {
            std::vector<uint32_t> buf;        // scratch for the page read (+ decoder residual)
            std::vector<PerfDebugRec> batch;  // pre-sized to the per-read record upper bound
            uint64_t iters = 0, pages = 0, emit = 0, stall = 0;
            uint32_t quiesce = 0;
            bool done = false;
        };
        SockState sock_state[kNSockets];
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
        bool synced = false;  // a real host<->device clock sync succeeded -> push RAW device timestamps

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };

    void start(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    bool boot_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    // ONE read+decode pass over (ctx, sock): pages -> decode -> records -> ring. Returns true if it moved data.
    bool drain_pass(DeviceCtx& ctx, uint32_t sock_idx);
    void writer_thread();    // round-robins every (device, socket), publishing each read as one batch
    void consumer_thread();  // BroadcastRing reader -> PerfDebugTracyHandler (the slow sink, now off the drain)

    std::vector<DeviceCtx> devices_;
    std::unique_ptr<PerfDebugTracyHandler> tracy_;
    // ★ Same shape as test_x280_realprof: the drain NEVER blocks on Tracy. One writer publishes decoded
    // records into a BroadcastRing; a separate consumer pushes them to Tracy. A lagging consumer DROPS its
    // own records (reported) instead of back-pressuring the FIFO -> relay -> reader -> worker cores.
    // Measured why this is required: with the push inline, UFLD-v2 put relay0 in HOST-WAIT for 15.85 s of a
    // 19 s run and stalled producers 826x; with the push removed entirely, stalls went to 0.
    std::unique_ptr<RecRingHolder> ring_;
    std::thread writer_;
    std::vector<std::thread> consumers_;
    std::atomic<uint64_t> consumed_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<bool> writer_done_{false};
    size_t read_chunk_recs_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stopped_{false};
    std::unordered_map<uint16_t, std::string> zone_names_;  // srcloc hash -> zone name (Tracy)
    std::once_flag names_once_;  // zone names are loaded LAZILY on first drain (after kernels JIT-compile,
                                 // so the zone-source-location log exists) -- not at start()/bring-up.
};

}  // namespace tt::tt_metal
