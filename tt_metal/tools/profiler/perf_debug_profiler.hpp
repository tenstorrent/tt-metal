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
// construction, static TLBs), host<->device clock sync and Tracy anchoring, the ordered quiesce, the
// device results reporting, and the teardown completeness check. Everything from the socket reads to the
// record consumers lives in PerfDebugReceiver; the Tracy sink is one more paired-record consumer.
// Booted once at MeshDevice bring-up (resident); at teardown the host writes 1 to each drainer's stop word
// to quiesce it, then 2 to release its NIU (no reset).
#pragma once

#include <cstdint>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <array>
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

// Per-risc byte figure the perf-debug role split needs the HAL to size its per-bank DRAM PROFILER region from,
// or 0 when the split is off. Called from get_profiler_dram_bank_size_for_hal_allocation() ONLY -- that path
// feeds the HAL region and nothing else. The role split reuses that region instead of allocating a second DRAM
// buffer, so the region's size and the ring's size are one knob. See FINDINGS §N+39.
uint32_t perf_debug_dram_region_bytes_per_risc();

// One PerfDebugProfiler per MeshDevice. Constructing it boots the drainers on every eligible local
// Blackhole device and starts the receiver; destroying it (or calling stop()) quiesces the drainers in
// role order, drains and shuts down the receiver, and leaves the resident idle FW alone (no reset).
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
    // TWO DRISC drainers without the role split, one D2H socket per drainer -- kNSockets is both counts.
    //
    // WHY 2 (FINDINGS N+34): a 5x knee improvement over 1; below the knee a single drainer falls into a
    // feedback loop (producers stall -> rings pin -> sweeps cost more) that two stay out of entirely.
    // 2 IS THE CEILING, for two independent reasons:
    //   - exactly two DRAM cores measure safe to host a drainer (row y == 0: `0-0` and `9-0`, N+29)
    //   - TLB windows: nwin=7 per 12 MiB socket into 16 available, so 2x7=14 fits and 3x7=21 does not
    //     (see kHRingWords below; raising it needs SOCKET_WIN_BASE moved)
    static constexpr uint32_t kNSocketsMax = 2;
    // ---- ROLE SPLIT (TT_METAL_PERF_DEBUG_ROLE_SPLIT, default on) ----
    //
    // kNSockets stays 2 -- nothing downstream of the socket knows this feature exists. With the knob on
    // there are SIX DRISCs rather than two:
    //
    //   index 0..3  FILLER  sweep a QUARTER of the worker grid -> write frames into their own device-DRAM
    //                       ring. No socket, no PCIe, no host MMIO.
    //   index 4,5   MOVER   read TWO DRAM rings -> push to socket 0 / socket 1. The only two
    //                       host-facing-safe cores (y == 0). Mover m drains fillers m and m + kNSockets.
    //
    // The knee is the FILLER's scan over its share of the grid (FINDINGS N+28/N+34/N+40), and the DRAM ring
    // is the elastic buffer the TLB-capped 12 MiB host FIFO cannot be: a DRISC writes DRAM natively over
    // the NoC with no window at all, so the buffer moves to DRAM and can be hundreds of MB.
    static constexpr uint32_t kMaxDrisc = 7;  // max roster: 6 fillers + 1 mover, or 4 + 2
    // The filler/mover SHAPE is runtime-selectable (TT_METAL_PERF_DEBUG_FILLERS = 4 or 6): 4 fillers ->
    // 2 movers (the sustained-optimized default), 6 fillers -> 1 mover (the onset-optimized shape: every
    // DRAM view drains workers, cores/filler drops 30 -> 20, and the single mover owns all six rings --
    // halving the sustained evacuation ceiling, which the ring runway and the fill gate absorb in the
    // regimes that shape is for). kNFillersMax/kNPeerMax bound the arrays and the device compile args;
    // divisibility (fillers % movers == 0) is enforced at boot.
    static constexpr uint32_t kNFillersMax = 6;
    static constexpr uint32_t kNPeerMax = 6;
    static_assert(kNFillersMax + 1 <= kMaxDrisc, "kMaxDrisc must cover every filler and mover");
    // 12 MiB / socket, and the CEILING: NOC_2M_WINDOW_COUNT=224 with SOCKET_WIN_BASE=208 leaves 16 TLB
    // windows and the FW maps nwin=ceil((in_off+bytes+64)/2MiB) consecutive windows per socket, so
    // kNSockets * nwin <= 16 (12 MiB -> nwin=7 -> 14). Raising further needs SOCKET_WIN_BASE moved too.
    static constexpr uint32_t kHRingWords = 3145728;
    static constexpr uint32_t kPageSize = 64;
    static constexpr uint32_t kNRisc = 5;
    static constexpr uint32_t kNoSocket = 0xFFFFFFFFu;  // DeviceCtx::sock_of for a filler

    struct DeviceCtx {
        uint32_t chip_id = 0;
        std::unique_ptr<distributed::D2HSocket> sockets[kNSocketsMax];  // moved into the receiver at start()
        uint32_t nl = 0;  // lanes = num_cores * NRISC (+ n_drisc * NRISC with self-profiling on)
        // Worker cores only, i.e. where the DRISC self-profiling lane block starts inside core_virt. 0 when
        // self-profiling is off, in which case core_virt holds nothing but workers.
        uint32_t n_worker_cores = 0;
        // ETH COVERAGE: eth cores are appended after the workers in the poll list. eth_start is the index
        // where they begin (== worker count), eth_prof_l1 the L1 base their profiler_msg_t sits at, which
        // differs from the Tensix one. Both are 0 when eth coverage is off.
        uint32_t n_eth_cores = 0;
        uint32_t eth_start = 0;
        uint32_t eth_prof_l1 = 0;
        // ---- DRISC drainer ----
        // The program stays alive for the life of the profiler: its kernel is still running. It was
        // launched OUTSIDE the command queue (detail::LaunchProgram with force_slow_dispatch), which is
        // what makes a resident drainer possible at all -- a DRAM-only program touches none of the fast
        // dispatch worker grid or dispatch column, so it can sit there across every user workload. Going
        // through the CQ instead would deadlock the first Finish().
        std::unique_ptr<Program> drain_program[kMaxDrisc];
        IDevice* device = nullptr;
        // Per-DRISC. Each drainer owns a disjoint slice of the worker grid, its own socket and its own L1
        // window -- nothing is shared between them on the device side.
        CoreCoord drisc_logical[kMaxDrisc];
        CoreCoord drisc_virtual[kMaxDrisc];
        uint64_t drisc_l1_noc[kMaxDrisc] = {};  // NoC-addressable base of each DRISC L1 window
        uint32_t drisc_l1_base[kMaxDrisc] = {};
        uint32_t stop_addr[kMaxDrisc] = {};  // host writes 1 to quiesce, 2 to release the NIU
        uint32_t done_addr[kMaxDrisc] = {};  // drainer publishes 0xD09E**** once its last page is out
        uint32_t results_addr[kMaxDrisc] = {};
        // ---- role split (all zero / kRoleFull when the knob is off) ----
        uint32_t n_drisc = kNSocketsMax;   // 2 normally, n_fillers + n_sockets with the role split on
        uint32_t role[kMaxDrisc] = {};     // 0 = full job, 1 = filler, 2 = mover
        uint32_t sock_of[kMaxDrisc] = {};  // socket index this DRISC owns, or kNoSocket
        uint32_t hs_addr[kMaxDrisc] = {};  // filler's handshake block (head/tail/probes) in its L1
        // mover -> the filler indices it drains, and how many. n_peer is 0 for a filler / full-job drainer.
        uint32_t peer_of[kMaxDrisc][kNPeerMax] = {};
        uint32_t n_peer[kMaxDrisc] = {};
        uint32_t dram_bank[kMaxDrisc] = {};  // ring bank of the ring DRISC d OWNS (fillers; 0 for movers)
        // Bank-relative base of the ring DRISC d owns. PER-DRISC rather than one shared value: the kernel
        // reaches its ring through get_noc_addr_from_bank_id, which adds bank_to_dram_offset[bank] itself, so
        // the host has to subtract THAT BANK's offset. It measures 0 in every bank on bh-26, but with four
        // rings a single shared address would silently mis-address any bank whose offset differed.
        uint32_t dram_addr[kMaxDrisc] = {};
        uint32_t dram_frames = 0;  // ring capacity in whole frames (identical for every ring)
        // No buffer handle: the rings live in the HAL's per-bank DRAM PROFILER region, which is reserved for
        // the profiler's whole lifetime by construction. Nothing to own, nothing to free.
        // core_index -> virtual (x,y) [what the SRC lane resolves to], and virtual -> NOC0 (x,y) [Tracy view].
        std::vector<std::pair<uint32_t, uint32_t>> core_virt;
        std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> virt_to_noc0;
        std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
        bool active = false;
        bool clock_synced = false;
        double freq_ghz = 0.0;  // measured sync frequency (cycles/ns); aiclk fallback
        // The anchor this device's zones are actually placed with, mirrored from the last AddDevice(). Kept
        // so the close-check can ask the only question that matters for placement: how far has this device's
        // clock moved away from the anchor still being used to render it?
        int64_t anchor_host = 0;
        uint64_t anchor_dev = 0;
        bool anchor_valid = false;

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
    bool boot_device(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);

    // ---- DEVICE-TO-DEVICE CLOCK SYNC ------------------------------------------------------------------
    //
    // Measured over the ethernet links at start(), BEFORE any device is booted. That ordering is the whole
    // reason it is cheap: fabric has not claimed the eth cores yet and no drainer is resident, so nothing
    // has to be evicted and LaunchProgram's dram_barrier cannot meet a DRAM core in stream mode.
    //
    // One entry per measured link. Composing these into a per-device transform against the root, and
    // feeding that to Tracy, is the next step; for now they are measured and reported so the numbers can
    // be trusted before anything depends on them.
    struct LinkSync {
        uint32_t sender_chip = 0;
        uint32_t receiver_chip = 0;
        int64_t offset = 0;    // receiver MINUS sender, in cycles, at ref_mid on the sender clock
        uint64_t ref_mid = 0;  // the sender-clock instant the offset is quoted at
        double rate = 1.0;     // receiver cycles per sender cycle
        double residual_rms = 0.0;
        uint64_t rtt_min = 0;
        bool valid = false;
        // Enough to RE-MEASURE this exact link at close: same two eth cores, same direction. Which physical
        // link is used matters at the nanosecond level, so re-measuring a different one would price the
        // link difference as session drift.
        IDevice* snd_dev = nullptr;
        IDevice* rcv_dev = nullptr;
        CoreCoord snd_eth{};
        CoreCoord rcv_eth{};
        // The two eth cores' own host<->device anchors, copied off the trace before it is discarded. Kept so
        // the close-check can compare an INDEPENDENT path (host MMIO) against this link's own measurement.
        int64_t snd_host_anchor = 0, rcv_host_anchor = 0;
        uint64_t snd_dev_at_anchor = 0, rcv_dev_at_anchor = 0;
        bool host_anchors_valid = false;
    };
    std::vector<LinkSync> link_syncs_;
    void sync_devices_over_eth(const std::shared_ptr<distributed::MeshDevice>& mesh_device);

    // The spanning tree the links were measured over: chip -> index into link_syncs_ of the edge reaching
    // it from its parent. The root has no entry.
    std::unordered_map<uint32_t, size_t> eth_sync_parent_edge_;
    uint32_t eth_sync_root_chip_ = 0;

    // Worst redundant-link closure from sync_devices_over_eth(), kept so the Tracy SYNC_ANCHOR marker can
    // publish the accuracy bound into the trace itself instead of leaving it only in the run log. `_valid`
    // is false when the topology had no cycle, which is NOT the same as a closure of zero.
    // Raw round trips per measured link, kept until the anchors exist so they can be drawn on the two
    // ETH cores' own Tracy lanes: the sender's [t0,t2] as a zone, the receiver's t1 as a marker inside it.
    // Unlike SYNC_ANCHOR (which is derived from the anchors and so aligns by construction), these are
    // measurements -- a t1 that renders OUTSIDE its zone is proof the alignment is wrong.
    struct EthSyncTrace {
        uint32_t sender_chip = 0;
        uint32_t receiver_chip = 0;
        uint32_t snd_x = 0, snd_y = 0;  // NOC0 coords of the sender's eth core
        uint32_t rcv_x = 0, rcv_y = 0;  // NOC0 coords of the receiver's eth core
        uint32_t snd_vx = 0, snd_vy = 0;  // VIRTUAL coords, for reading that tile's own wall clock
        uint32_t rcv_vx = 0, rcv_vy = 0;
        // Each eth core's own host<->device anchor, captured AT SYNC TIME rather than later. Anchoring far
        // from the samples is not free: the anchor is a point plus a slope, and the slope carries a few ppm
        // of error, so extrapolating back N seconds costs N * ppm. Measured the hard way -- anchoring 7.55 s
        // after the samples put the peer's echo 36 us OUTSIDE its own round trip (4.8 ppm x 7.55 s), which
        // discards the sync's 1.9 ns precision by four orders of magnitude.
        int64_t snd_host_anchor = 0, rcv_host_anchor = 0;
        uint64_t snd_dev_at_anchor = 0, rcv_dev_at_anchor = 0;
        bool snd_anchor_valid = false, rcv_anchor_valid = false;
        std::vector<std::array<uint64_t, 3>> trips;  // t0 (snd clock), t1 (rcv clock), t2 (snd clock)
    };
    std::vector<EthSyncTrace> eth_sync_traces_;
    // Draw the stored traces onto eth lanes. Called once, after every device has its anchor.
    void emit_eth_sync_lanes();

    // Re-measure every transform AFTER bring-up, replacing the ones taken at init. Bring-up is the whole
    // problem: booting drainers on 4 devices took 26.3 s in a measured run, and the anchors were fitted
    // BEFORE it while the first zone is timestamped AFTER it. At ~5.4 us/s of accumulated clock stepping
    // that bakes ~140 us of misalignment into a capture before it starts. Re-anchoring here shrinks the
    // gap from 26 s to ~0.1 s. It works because Tracy contexts are created LAZILY on each core's first
    // zone, so anchors replaced at the end of start() are the ones every context bakes in.
    void reanchor_after_boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);

    // Re-measure every tree link at CLOSE and compare against what the init fit predicted for that instant.
    // The init sync says how well the clocks agreed WHEN MEASURED; this says how far the session had drifted
    // by the time it ended, which is the number that actually bounds a zone's placement late in a capture.
    void check_sync_drift_at_close();
    // Host-only late re-anchor: a fresh host<->device fit per device after bring-up. Needs no eth core and
    // launches nothing, so unlike the eth re-anchor it is legal with fabric up.
    void host_reanchor_after_boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    // Measures each device anchor's staleness against a fresh fit. No eth, no composition.
    void check_anchor_staleness_at_close();
    // Repeatedly measures the eth-derived and host-derived inter-device offset and reports whether their
    // DIFFERENCE holds constant. The constant intra-chip domain gap cancels, so a trend is real divergence.
    void cross_path_tracking_test();
    // Background re-anchor: refits each device against the host every N ms and publishes the resulting
    // correction so zones are placed with a CURRENT anchor instead of the one fitted at bring-up.
    void start_drift_corrector(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    void stop_drift_corrector();
    std::thread drift_thread_;
    std::atomic<bool> drift_stop_{false};

    // RESIDENT ETH TRACKER (TT_METAL_PERF_DEBUG_ETH_TRACK_HZ, 0 = off). Keeps a bounded sync pair RESIDENT
    // on a SECOND channel of every spanning-tree link -- one launch at start(), rounds commanded through an
    // L1 mailbox -- and converts each link's live deviation from its init fit into the SAME per-device zone
    // corrections the drift corrector feeds. Zones and the close-check prediction both read those
    // corrections, so the close-check stops scoring a one-shot extrapolation and starts scoring the
    // tracked truth. The init one-shot and the close-check keep the pair's FIRST channel; residents live on
    // a second one, so the two never collide.
    struct EthTrackState;
    EthTrackState* eth_track_ = nullptr;
    void start_eth_tracker(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    void stop_eth_tracker();

    // Chips whose AI clock this profiler pinned, so stop() can release exactly those.
    std::vector<int> forced_aiclk_chips_;

    int64_t eth_sync_worst_closure_ = 0;
    bool eth_sync_closure_valid_ = false;

    // The ROOT's host<->device fit, kept so every other device can hang off it instead of measuring its
    // own. Plain fields rather than the PerfDebugSync struct, which is .cpp-local.
    bool root_sync_valid_ = false;
    int64_t root_host_anchor_ = 0;
    uint64_t root_dev_at_anchor_ = 0;
    double root_freq_ghz_ = 0.0;

    // Map a ROOT-clock instant onto `chip`'s own clock by walking the tree, and report how much faster
    // that chip ticks than the root. False if `chip` has no measured route to the root, in which case the
    // caller falls back to giving that device its own host sync.
    //
    // Composed in DIFFERENCES from each edge's reference instant, never as an absolute intercept: the
    // clocks sit near 1e13 cycles, where a double resolves only ~0.001 cycle, so an intercept form would
    // spend its precision representing the epoch rather than the answer.
    bool eth_sync_anchor_for(uint32_t chip, uint64_t root_clock, uint64_t& chip_clock, double& rate_vs_root) const;
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
