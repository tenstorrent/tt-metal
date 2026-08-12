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
// Engine:
//   boot_device -> 2 D2HSockets
//   -> producer (poll/copy/ack into ping-pong staging)
//   -> decoder  (staging -> ping-pong publish batches)  // overlapped with the next sock-read
//   -> publisher (batches -> BroadcastRing)             // overlapped with decode
//   -> consumer (ring -> Tracy)
// Shares the marker wire format with the drain kernel through spsc_marker_decode.hpp.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "hostdevcommon/profiler_common.h"

#include <tt-metalium/core_coord.hpp>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class D2HSocket;
class MeshBuffer;
}  // namespace distributed
class PerfDebugTracyHandler;
class Program;
class IDevice;
struct RecRingHolder;  // pimpl for BroadcastRing<PerfDebugRec> (keeps the ring header out of this one)

namespace profiler {
struct SpscDecodeState;
}  // namespace profiler

// Per-risc byte figure the perf-debug role split needs the HAL to size its per-bank DRAM PROFILER region from,
// or 0 when the split is off. Called from get_profiler_dram_bank_size_for_hal_allocation() ONLY -- that path
// feeds the HAL region and nothing else. The role split reuses that region instead of allocating a second DRAM
// buffer, so the region's size and the ring's size are one knob. See FINDINGS §N+39.
uint32_t perf_debug_dram_region_bytes_per_risc();

// Decoded device record handed writer -> reader. Packed to 12 B: full device timestamp + meta.
//   meta = [31:29] record type | [28:26] device index | [25:16] lane | [15:0] zone srcloc hash / id16
// Zone records are self-contained. A DATA/EVENT head is followed by EXT (full 20-bit id + word count
// in ts) then CONT payload pairs in ts -- rare path.
// Per-record `prog` stays dropped (runtime_host_id=0 until a sticky-PROG side channel is restored);
// timestamps stay full-width -- truncating them is not an acceptable publish-bandwidth cheat.
struct __attribute__((packed)) PerfDebugRec {
    uint64_t ts;
    uint32_t meta;
};
static_assert(sizeof(PerfDebugRec) == 12);
static_assert(std::is_trivially_copyable_v<PerfDebugRec>);
inline constexpr uint32_t kRecTypeShift = 29;
inline constexpr uint32_t kRecDevShift = 26;
inline constexpr uint32_t kRecLaneShift = 16;
inline constexpr uint32_t kRecDevMax = 8;
inline constexpr uint32_t kRecLaneMax = 1024;
// Record type codes. START/END equal the wire's PP_ZONE_START/END so the hot emit stores the wire type
// unmapped; the rest are this stream's own codes (the wire's 5-bit space does not fit 3 bits).
inline constexpr uint32_t kRecZoneStart = 0;
inline constexpr uint32_t kRecZoneEnd = 1;
inline constexpr uint32_t kRecData = 3;
inline constexpr uint32_t kRecEvent = 4;
inline constexpr uint32_t kRecCont = 5;
inline constexpr uint32_t kRecExt = 6;

// One PerfDebugProfiler per MeshDevice. Constructing it boots the drainer drainer on every eligible local
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
    // Fixed drain config -- the silicon-validated defaults (see the standalone drain harness / the knee + Tracy
    // sweeps): 2 reader harts + 2 relay harts (dual relay), one 12 MiB D2HSocket FIFO per relay, adaptive per-core
    // drain.
    static constexpr uint32_t kNRead = 2;
    static constexpr uint32_t kNRelay = 2;  // dual relay
    // TWO DRISC drainers, and one D2H socket per drainer -- kNSockets is both counts. Each drainer owns
    // a disjoint contiguous slice of the grid (cores [0,60) / [60,120) at 120 cores), its own L1, its own
    // head mirrors and its own 12 MiB socket, so the two drain loops share nothing on the device.
    //
    // WHY 2 (FINDINGS N+34). It is a 5x knee improvement, measured on the fixed bring-up path with 3 warm
    // repeats per point and no failed runs anywhere in the sweep:
    //
    //     drainers   knee   idle sweep   busy sweep   worst sweep
    //        1        100     32.6 us      70.0 us      81.2 us
    //        2         20     18.4 us      38.0 us      58.8 us
    //
    // Knee is sharp: 23,000 stalls with rings pinned at 511/512 at delay 10, and 0 stalls at occ ~435 at
    // delay 20. The gain exceeds the ~2x that halving per-sweep cost predicts, because below the knee a
    // single drainer falls into a feedback loop (producers stall -> rings pin -> sweeps cost more) that
    // two drainers stay out of entirely.
    //
    // WHY IT WAS 1 BEFORE, AND WHAT CHANGED. Two drainers used to hang ~4/80 runs and froze the host
    // twice. The instrumented repro named the site 4 times out of 4 --
    // `niu-mode(3,1)->1:LaunchProgram(dram_barrier+wait_until_cores_done)` -- i.e. the SECOND drainer's
    // NIU flip, whose LaunchProgram barriers across every DRAM channel while the FIRST drainer is already
    // resident in stream mode. Flipping every drainer's NIU in ONE launch, before any core enters stream
    // mode, removed it: 80/80 clean, no freeze (was 42/80 clean with 4 hangs, 3 wedges and a freeze).
    //
    // 2 IS THE CEILING, for two independent reasons:
    //   - exactly two DRAM cores measure safe to host a drainer (row y == 0: `0-0` and `9-0`, N+29)
    //   - TLB windows: nwin=7 per 12 MiB socket into 16 available, so 2x7=14 fits and 3x7=21 does not
    //     (see kHRingWords below; raising it needs SOCKET_WIN_BASE moved)
    //
    // Judge any change here on the WORST sweep, not the mean: an early 2-drainer attempt improved the mean
    // busy sweep 1.43x while leaving the worst sweep at ~95 us, and the knee did not move at all.
    static constexpr uint32_t kNSockets = 2;
    // ---- ROLE SPLIT (TT_METAL_PERF_DEBUG_ROLE_SPLIT=1) ----
    //
    // kNSockets is still 2, and that is the point: the number of D2H sockets, host reader threads, decoder
    // threads and decode streams is unchanged, so nothing downstream of the socket knows this feature exists.
    // What changes is that with the knob on there are SIX DRISCs rather than two, and the job is split:
    //
    //   index 0..3  FILLER  sweep a QUARTER of the worker grid -> write frames into their own device-DRAM ring.
    //                       No socket, no PCIe, no host MMIO. Banks 5, 6, 4, 1 (cores 9-9, 9-5, 9-2, 0-3).
    //   index 4,5   MOVER   read TWO DRAM rings -> push to socket 0 / socket 1, exactly as today.
    //                       Banks 0 and 3 (cores 0-0 and 9-0) -- the only two host-facing-safe cores.
    //                       Mover m drains fillers m and m + kNSockets (so 4 -> {0,2}, 5 -> {1,3}).
    //
    // WHY 4 FILLERS AND ONLY 2 MOVERS. The knee is the FILLER's SCAN over its share of the grid (FINDINGS
    // N+28), so halving each filler's slice from 60 cores to 30 is the same lever that gave 5x going 1 -> 2
    // drainers (N+34). Movers cannot be scaled the same way -- there are exactly two host-facing-safe cores
    // (y == 0) and only 16 TLB windows for 7-window sockets -- but they are ~97% idle (0.3-0.4 us idle
    // sweeps, ~3,900 busy of ~450,000), so each absorbs a second ring.
    //
    // A dual-ring mover walks its peers SEQUENTIALLY, each with the WHOLE staging area, with the existing
    // per-push write barrier in between. So `max batch` stays 7 per peer -- splitting the 7 slots two ways
    // (which would have raised per-frame egress overhead exactly where the credit-wait knee lives) is not
    // needed, because egress is serialized on one socket anyway. MEASURED: max batch 7 on all four peers.
    //
    // The four rings are FREE in DRAM: the HAL reserves the profiler region at the same offset in EVERY bank
    // (7 of them, 64.0 MiB each = 448 MiB) and only 2 banks carried a ring before, so rings 2 and 3 cost
    // nothing. Full ring/drainer channel disjointness is no longer possible (6 drainer channels + 4 rings >
    // 7 allocator banks); rings stay off the two MOVER banks, which is where the measured hazard lives.
    //
    // WHY. The knee is the worst sweep's CREDIT-WAIT (50-97 us of an 86-127 us worst sweep, vs 1.3 us when
    // the host keeps up), and that is burst absorption, not bandwidth: sustained egress is ~1.37 GB/s
    // aggregate but a single busy sweep ships 581 KB in 35.2 us, i.e. exactly the 16.5 GB/s ceiling. The
    // 12 MiB FIFO is ~21 busy sweeps of slack and CANNOT be grown -- kHRingWords is capped by the device
    // TLB budget (kNSockets * nwin <= 16; 12 MiB is nwin=7, so 14 of 16 are already spent). A DRISC writes
    // DRAM natively over the NoC with no window at all, so the elastic buffer moves to DRAM and can be
    // hundreds of MB. Expected floor is proc 12.9 + write + wr-barrier ~= 42 us, so knee ~60-80, not single
    // digits.
    //
    // WHY THE FILLERS MAY SIT ON y != 0 BANKS when kSafeBanks says only y==0 is safe: that finding is about
    // the HOST-FACING role. Measured on bank 5 (N+29's worst core, 5/25 for a full-job drainer): 0/25
    // failures held in stream mode and 0/25 doing filler-only duty. The role-aware TT_FATAL in boot_device
    // enforces kSafeBanks for movers only.
    static constexpr uint32_t kMaxDrisc = 6;
    // FOUR fillers, two movers. kNFillers must stay a whole multiple of kNSockets: mover m takes fillers
    // m, m + kNSockets, ... and kNPeerMax bounds the per-mover peer state (device compile args and L1
    // telemetry words are per-peer, so raising this is not free).
    static constexpr uint32_t kNFillers = 4;
    static constexpr uint32_t kNPeerMax = 2;
    static_assert(kNFillers % kNSockets == 0, "each mover must drain a whole number of fillers");
    static_assert(kNFillers / kNSockets <= kNPeerMax, "a mover cannot drain more than kNPeerMax rings");
    static_assert(kNFillers + kNSockets <= kMaxDrisc, "kMaxDrisc must cover every filler and mover");
    // 12 MiB / socket. RAISED from 4 MiB (1048576 words), which was sized from a "4 MiB knee" measurement
    // that later proved to be 4 MiB's OWN floor, not the hardware's. On a host-bound box 4 MiB pins the FIFO
    // at 100%: relay hostfull 415k -> reader spsc-wait 155M -> producers stall, reader copy% 0.8 (spinning),
    // drainer wall 1921M cyc. At 12 MiB: hostfull 0, spsc-wait 0, copy% 52, wall 30M cyc. Matches the
    // the standalone drain harness --hring default. This is also the CEILING: NOC_2M_WINDOW_COUNT=224 with
    // SOCKET_WIN_BASE=208 leaves 16 TLB windows and the FW maps nwin=ceil((in_off+bytes+64)/2MiB) consecutive
    // windows per socket, so kNSockets * nwin <= 16 (12 MiB -> nwin=7 -> 14). Raising further needs
    // SOCKET_WIN_BASE moved too.
    static constexpr uint32_t kHRingWords = 3145728;
    // Per-read page cap. 0 = UNCAPPED (take whatever the FIFO holds, bounded only by fifo_pages-1).
    // Overridable at runtime with TT_METAL_PERF_DEBUG_MAX_PAGES.
    //
    // Socket page = one staged span frame (kPageSize = 10,560 B). FINDINGS egress winner was
    // ~80 KB x 8 pages/read ≈ 640 KB per host read → 25.4 GB/s with memcpy (57.6 GB/s discard).
    // Default 60 pages ≈ 619 KB. Override: TT_METAL_PERF_DEBUG_MAX_PAGES.
    //
    // HISTORY (when kPageSize was 64 B):
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
    // ERA 3 -- live-pack on the mover (variable frames) needs 64 B pages again. Cap 10240 = 640 KB/read
    // so we still amortize; uncapped risks a multi-ms stall on one socket while the other starves.
    // Optional MIN_PAGES (TT_METAL_PERF_DEBUG_MIN_PAGES) holds a poll until a FINDINGS-sized chunk is
    // pending; default 0 -- device-side pack coalesce already widens bursts, and a host hold stranded
    // FIFO tails at teardown until stop-drain was added.
    //
    // KNOWN LIMITATION: a PAGE cap does not bound per-pass TIME. Once data is plentiful every read is a full
    // cap-sized read, which is why caps of 64 / 256 / 1024 / 4096 are indistinguishable below the knee. The
    // correct fix is to bound elapsed time per pass; this is the stopgap that behaves well at observed rates.
    // Note also the ack is issued by read() itself, so a bigger read acks MORE data sooner, not later.
    // 64 B socket pages (SPSC_SPAN_PAGE_WORDS). Live-pack emits variable-length frames; a large fixed
    // page would pad every frame back to a full slot and erase the PCIe win (measured). Cap ~640 KB/read.
    static constexpr uint32_t kMaxPagesPerRead = 10240;  // 640 KB @ 64 B; uncapped was slower/noisier
    static constexpr uint32_t kPageSize = 64;
    static_assert(kPageSize == kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u, "host/device socket page contract");
    // Bound on the decoder's cross-call residual (a trailing partial packet, at most one incomplete
    // BULK_SPAN frame ~2640 words). Sizes the publish batches so one decode call can never overflow.
    static constexpr uint32_t kDecodeCarryWords = 4096;
    // Decode-worker fan-out ceiling (TT_METAL_PERF_DEBUG_DECODE_WORKERS picks the actual count, default 3).
    // Cores shard CONTIGUOUSLY across workers (worker = core * N / num_cores), which preserves per-lane
    // record order structurally: a lane never changes worker, and each worker consumes its slots in
    // stream order. Cap 4: w=3 matches w=4 on marker-wire; w=2 is decode-bound.
    static constexpr uint32_t kMaxDecodeWorkers = 4;
    static constexpr uint32_t kNRisc = 5;
    static constexpr uint32_t kNoSocket = 0xFFFFFFFFu;  // DeviceCtx::sock_of for a filler

    struct DeviceCtx {
        uint32_t chip_id = 0;
        std::unique_ptr<distributed::D2HSocket> sockets[kNSockets];
        uint64_t params_addr = 0;  // profzone MBOX_PARAMS (P_STOP at teardown)
        uint32_t nl = 0;           // lanes = num_cores * NRISC
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
        uint32_t n_drisc = kNSockets;      // 2 normally, kNFillers + kNSockets with the role split on
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
        std::unique_ptr<profiler::SpscDecodeState> decode[kNSockets];
        // Per-(worker, socket) decode state for the parallel workers. Per-lane fields shard trivially (a
        // lane's frames always land on one worker); cur_prog is naturally per-core because each core's
        // BRISC emits its own STICKY_PROG and a core's frames stay on one worker.
        std::unique_ptr<profiler::SpscDecodeState> wdecode[kMaxDecodeWorkers * kNSockets];
        bool active = false;
        // --hartzones equivalent (TT_METAL_PERF_DEBUG_HART_ZONES=1): the drain harts inject their own
        // busy/idle spans IN-BAND. {rdcycle, meta} pairs per hart (START,END alternating); each hart is written
        // by exactly one drain thread, so no lock is needed. Mapped to Tracy at stop() using the per-cluster
        // rdcycle->Tensix calibration that hart0 writes at boot when this mode is on.
        struct HZMark {
            uint64_t rdc;
            uint32_t meta;
        };
        // Per-socket state; drain_pass() is called repeatedly by this socket's producer thread, so it
        // persists here. Staging slots: the producer fills slot (fill_seq & mask), ACKs, splits the slot
        // into whole-frame ranges per decode worker, and advances fill_seq; each WORKER consumes its
        // ranges from every slot in order and advances its own wdone. A slot is reusable when every
        // worker has passed it. Eight slots: the mover ships same-filler frame batches, so a
        // single slot's frames often belong to ONE worker -- pipeline depth across slots is what lets
        // several workers run concurrently. Four slots stalls decode under role-split (50–120 ms).
        //
        // ZERO-COPY (decode in place on D2H peek) lost twice to this staging copy: branchy decode
        // pulls ~5 GB/s from cold PCIe lines; memcpy + L3-warm decode is ~3x. Do not remove.
        static constexpr size_t kStageSlots = 8;
        struct SockState {
            std::array<std::vector<uint32_t>, kStageSlots> buf{};
            std::array<size_t, kStageSlots> words{};
            // Producer-authored per-slot, per-worker WHOLE-frame ranges (pointer, words) into the slot's
            // staging (or into carry[slot] for the one frame that straddled the previous slot's end).
            std::array<std::array<std::vector<std::pair<const uint32_t*, uint32_t>>, kMaxDecodeWorkers>, kStageSlots>
                ranges{};
            // A frame cut by a slot boundary is reassembled here (indexed by the slot that completes it);
            // carry_pend holds the partial tail copied out before its slot is recycled.
            std::array<std::vector<uint32_t>, kStageSlots> carry{};
            std::vector<uint32_t> carry_pend;
            alignas(64) std::atomic<uint64_t> fill_seq{0};
            std::array<std::atomic<uint64_t>, kMaxDecodeWorkers> wdone{};
            uint64_t iters = 0, pages = 0, stall = 0;
            // Producer-thread-owned phase counters (two producer threads now, so these cannot be members
            // of the profiler).
            uint64_t read_ns = 0, wait_ns = 0, copy_ns = 0, ack_ns = 0, poll_ns = 0, polls = 0, reads = 0, bytes = 0,
                     wall_ns = 0;
            uint32_t quiesce = 0;
            bool done = false;
            bool overflow_reported = false;  // one-shot: pages_available() exceeded the FIFO (see drain_pass)
        };
        SockState sock_state[kNSockets];
        std::vector<std::vector<HZMark>> hz_raw;  // sized kNRead + kNRelay when enabled
        uint64_t nharts = 0;
        // Marker rebase origin (first worker-kernel device ts seen by ANY socket), published so the hart-zone
        // push can share it. The hart spans MUST use the same origin as the markers: the drainer starts draining
        // at MeshDevice bring-up, seconds before the model runs, so rebasing the harts on their own first span
        // shifts their whole lane left of the kernels.
        // Plain (not atomic): DeviceCtx must stay movable, and the only contention is two drain threads both
        // wanting to record "the first marker ts". Whichever wins differs by microseconds -- irrelevant for an
        // origin, so the benign race is preferable to making the struct unmovable.
        uint64_t marker_ts_base = 0;
        bool synced = false;  // a real host<->device clock sync succeeded -> push RAW device timestamps
        // Device cycles per nanosecond (GHz). From clock sync when valid, else aiclk. Used for
        // first→last zone sustained-throughput reporting.
        double freq_ghz = 0.0;

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };

    void report_sustained_throughput();  // device first→last zone vs host busy/wall, same numerators

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
    bool boot_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    // ONE read pass over (ctx, sock): pages -> early ACK -> split the staged slot into per-worker
    // whole-frame ranges -> publish the slot to the decode workers. Returns true if it moved data.
    bool drain_pass(DeviceCtx& ctx, uint32_t sock_idx);
    // Producer-side splitter: walk the staged words packet-by-packet (frame hops -- 2 loads per ~10 KB
    // frame), route each whole frame to its core's worker, and reassemble the frame cut at the slot
    // boundary via carry_pend/carry[slot]. Adjacent same-worker frames coalesce into one range.
    void split_slot(DeviceCtx& ctx, uint32_t sock_idx, size_t slot, const uint32_t* stage, size_t words);
    // Decode one slot's ranges for one worker into that worker's PublishBatch and submit it.
    void decode_ranges(DeviceCtx& ctx, uint32_t sock_idx, uint32_t worker, size_t slot);
    // PRODUCER STAGGER PROBE. first_ts_[lane] = device timestamp of the first marker seen on that lane.
    // The SPREAD across lanes says whether the 110 cores start together or staggered -- which is the open
    // question behind the "degraded" batching difference (12 vs 57 cores with data per drainer sweep).
    // Indexed by the record's lane field; 0 = not yet seen. Written only by the producer thread.
    std::vector<uint64_t> first_ts_;
    void report_lane_spread();
    // Read the drainer's LIVE state (done word, heartbeat, phase) mid-run and log it. Distinguishes
    // "kernel exited" from "kernel blocked in the credit wait" from "kernel sweeping with nothing to do" --
    // states the end-of-run results block cannot tell apart because it is only published on exit.
    void dump_drainer_state(DeviceCtx& ctx, uint32_t d, const char* why);
    // Producer for ONE socket: poll -> read -> ack -> split into per-worker ranges.
    void producer_thread(uint32_t sock_idx);
    // Decode worker: consumes its ranges from every socket's slots, in slot order.
    void decode_worker(uint32_t worker);
    // Dedicated BroadcastRing writer: drains worker scratch batches via publish_batch.
    void publisher_thread();
    void consumer_thread();  // BroadcastRing reader -> PerfDebugTracyHandler
    bool decoder_work_pending() const;

    struct PublishBatch {
        std::vector<PerfDebugRec> recs;
        size_t n = 0;
    };
    static constexpr size_t kPublishBatchSlots = 4;  // slack so decode never waits on publisher
    // Per-worker double-buffered SPSC handoff (worker -> publisher) plus that worker's decode counters
    // (single-writer, summed/maxed at report time).
    struct WorkerPub {
        std::array<PublishBatch, kPublishBatchSlots> batches{};
        alignas(64) std::atomic<uint64_t> prod{0};
        alignas(64) std::atomic<uint64_t> cons{0};
        uint64_t decode_ns = 0;
        uint64_t recs = 0;
        uint64_t zone_recs = 0;
        uint64_t stall = 0;
        uint64_t emit[kNSockets] = {0, 0};
        uint64_t last_rec_ns = 0;
    };
    PublishBatch* publish_acquire_batch(uint32_t worker);  // wait until a free slot
    void publish_submit_batch(uint32_t worker);
    void publish_wait_idle();  // every worker's cons == prod
    void publish_stop();

    std::vector<DeviceCtx> devices_;
    std::unique_ptr<PerfDebugTracyHandler> tracy_;
    // Single BroadcastRing: decode workers submit L3 scratch batches; one publisher NT-copies in.
    std::unique_ptr<RecRingHolder> ring_;
    std::array<std::thread, kNSockets> producers_;
    std::vector<std::thread> workers_;
    std::thread publisher_;
    std::thread consumer_;
    uint32_t n_workers_ = 3;
    std::array<WorkerPub, kMaxDecodeWorkers> wpub_{};
    std::atomic<bool> publisher_stop_{false};
    std::once_flag first_data_once_;
    std::atomic<bool> decoder_stop_{false};
    std::atomic<uint64_t> consumed_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<bool> writer_done_{false};
    size_t read_chunk_recs_ = 0;
    // TEMPORARY DIAGNOSTICS (nesting investigation): where markers vanish between decode and the ring, and
    // whether per-lane ts order is already broken at publish or only after the ring.
    std::atomic<uint64_t> w_drop_lane_{0};    // markers whose lane is outside the known core grid
    std::atomic<uint64_t> w_batch_flush_{0};  // mid-decode claim re-arms because the reservation filled
    std::atomic<uint64_t> w_pub_regress_{0};  // per-lane ts going backwards AT PUBLISH (must stay 0)
    std::atomic<uint64_t> w_pub_ok_{0};
    std::vector<uint64_t> pub_last_ts_;
    std::atomic<uint64_t> w_con_regress_{0};
    std::atomic<uint64_t> w_con_seen_{0};
    uint64_t w_read_ns_ = 0;
    uint64_t w_wait_ns_ = 0;
    uint64_t w_copy_ns_ = 0;
    // sock-read split. `read()` is wait_for_bytes + two memcpys + pop_bytes; the ack is issued separately.
    // Only the memcpy scales with BYTES. The ack is a single ~180 ns PCIe write.
    uint64_t w_ack_ns_ = 0;
    uint64_t w_predrain_ns_ = 0;
    uint64_t w_decode_ns_ = 0;
    uint64_t w_publish_ns_ = 0;  // dedicated publisher occupancy
    uint64_t w_reads_ = 0;
    uint64_t w_bytes_ = 0;
    uint64_t w_recs_ = 0;
    uint64_t w_stalls_ = 0;
    uint64_t w_poll_ns_ = 0;
    uint64_t w_polls_ = 0;
    uint64_t w_wall_ns_ = 0;
    // Sustained-throughput window. Device: min/max ZONE marker timestamps (device cycles). Host: steady_clock
    // ns from first successful drain through last decode that published records. Same numerators (D2H bytes,
    // marker-wire bytes, zone count) over each window => apples-to-apples GB/s and Mzones/s.
    uint64_t w_zone_recs_ = 0;         // PP_ZONE_START/END only (excludes PP_DATA continuations)
    uint64_t host_first_data_ns_ = 0;  // steady_clock ns since epoch; 0 = none
    uint64_t host_last_rec_ns_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stopped_{false};
    std::unordered_map<uint16_t, std::string> zone_names_;  // srcloc hash -> zone name (Tracy)
    std::once_flag names_once_;  // zone names are loaded LAZILY on first drain (after kernels JIT-compile,
                                 // so the zone-source-location log exists) -- not at start()/bring-up.
};

}  // namespace tt::tt_metal
