// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's FILLER: sweep a sixth of the worker grid's SPSC rings and push each core's
// frame straight into this filler's own D2H socket. The live runs are GATHER-READ off the worker into
// packed wire order in staging, so a frame ships as ONE PCIe write -- the read side fans out across the
// slice's cores while the write side, where six fillers converge on one PCIe tile, stays at one
// arbitration event per frame. Six fillers cover the grid; there is no intermediate device-DRAM ring and
// no mover role, so back-pressure is the socket credit wait against the host FIFO.
//
// Placement evidence and wire format: tools/drisc_drain/FINDINGS.md.

#include "drisc_drain_common.hpp"

void kernel_main() {
    constexpr uint32_t kStageBase = get_compile_time_arg_val(0);  // slot 0's PREFIX (not its span)
    constexpr uint32_t kNStage = get_compile_time_arg_val(1);     // cores per batch = max cores per push
    constexpr uint32_t kHeadScratch = get_compile_time_arg_val(2);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
    constexpr uint32_t kDoneAddr = get_compile_time_arg_val(4);
    constexpr uint32_t kStopAddr = get_compile_time_arg_val(5);  // host: 1 = quiesce, 2 = free the NIU
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(6);
    constexpr uint32_t kMaxSweeps = get_compile_time_arg_val(7);
    constexpr uint32_t kMaxCores = get_compile_time_arg_val(8);
    // Fixed inter-sweep gap in cycles. 0 = continuous.
    constexpr uint32_t kGapCycles = get_compile_time_arg_val(9);
    // 1 = resync the software NoC mirrors from hardware at entry (see the wedge note below). 0 = diagnostic.
    constexpr uint32_t kNocInit = get_compile_time_arg_val(11);
    // Args 10, 12..15 and 17..18 are retired. The indices stay occupied: arg positions appear in JIT cache
    // keys and in FINDINGS notes.
    constexpr uint32_t kPcieEncOverride = get_compile_time_arg_val(16);

    constexpr uint32_t kNumRisc = 5;
    static_assert(kNumRisc == 5, "the control scans are unrolled for exactly five RISCs");
    constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
    constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;  // 2,624 words: the raw self frame payload
    // LIVE capacity = the rings alone; kSpanWords also counts the 64-word control vector.
    constexpr uint32_t kLiveWords = kNumRisc * kRingWords;
    constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    // DRISC self-profiling master switch (full doc at the self-profiling block below); read early
    // because the slot geometry depends on it.
    constexpr uint32_t kSelfZones = get_compile_time_arg_val(32);
    // FULL-SPAN slots on purpose: every sub-span cap tried deferred whole lanes at speed and starved
    // TRISC2's producer outright -- the capped configs' knee wins were that load shed.
    constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);
    constexpr uint32_t kSlotBytes = kSlotWords * 4u;
    constexpr uint32_t kWireCtrl = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
    constexpr uint32_t kPayloadCapWords = kSlotWords - kPrefix - kWireCtrl;
    constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
    constexpr uint32_t kPageBytes = kPageWords * 4u;
    // Reads take the NoC the writes do not; NOC_INDEX (the kernel's configured NoC) carries egress.
    constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
    // ---- GDDR SPOOL (arg 43 != 0): frames ship by DMA into a ring in this DRISC's own GDDR bank and a
    // non-blocking drain pump forwards spool bytes to the host FIFO from OFF the service path. The hot
    // loop then never touches the PCIe tile: its egress is a local DMA whose acceptance is deterministic,
    // so host-side pressure (decode, memory traffic) lands in spool occupancy instead of in the sweep
    // interval. 0 = the direct-push path, byte-identical to before the spool existed.
    constexpr uint32_t kSpoolBase = get_compile_time_arg_val(42);
    constexpr uint32_t kSpoolBytes = get_compile_time_arg_val(43);
    constexpr bool kSpool = kSpoolBytes != 0;
    constexpr uint8_t kDmaShip = 0;   // TX stream 0: staging -> spool
    constexpr uint8_t kDmaDrain = 1;  // TX stream 1: spool -> bounce
    // Two-core batches; every slot beyond the CV slot becomes pipeline depth. Spool mode spends two
    // slots on the drain's bounce buffers -- affordable because the third staging generation existed to
    // ride out PCIe acceptance jitter, which a DMA ship has none of.
    constexpr uint32_t kGenSlots = 2;
    constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
    // Self-zone builds park the CV staging inside the self slot, so no array slot is spent on it.
    constexpr uint32_t kCvOwnSlot = kSelfZones != 0 ? 0u : 1u;
    constexpr uint32_t kNGens = (kNStage - kCvOwnSlot - kNBounce) / kGenSlots;
    static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
    constexpr uint32_t kBounceSlot0 = kNGens * kGenSlots + kCvOwnSlot;
    static_assert(kBounceSlot0 + kNBounce <= kNStage, "bounce slots must fit inside the mapped staging arena");
    static_assert(!kSpool || kSpoolBytes % (kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u) == 0, "spool wraps on pages");
    // The static VC this filler's PCIe pushes ride, spread across fillers by the host: per-hop NoC
    // arbitration is per-VC, so six pushers on one VC starved the far cores' share of the PCIe tile.
    constexpr uint32_t kWriteVc = get_compile_time_arg_val(20);
    // Args 21..31 retired (RAW_ONLY, then the DRAM-ring role split).
    // ---- DRISC SELF-PROFILING (kSelfZones above; 0 = off, every use behind `if constexpr`) ----
    // The drainer's own zones, framed exactly like a worker span and shipped down the path it already
    // owns: no side channel, no second wire format, host decoder untouched. Only ring 0 is live.
    constexpr uint32_t kSelfHoldCycles = get_compile_time_arg_val(33);
    constexpr uint32_t kSelfXY = get_compile_time_arg_val(34);  // this DRISC's own virtual (y<<16)|x
    constexpr uint32_t kSelfMaxFrames = get_compile_time_arg_val(35);
    // Detail 0 = SWEEP + PACE only; 1 = also the per-batch child phases, ~25x the volume.
    constexpr uint32_t kSelfDetail = get_compile_time_arg_val(36);
    // NoC FOOTPRINT: the drainer's own NIU master counters (local MMIO loads -- cannot perturb what
    // they measure).
    constexpr uint32_t kNocFootprint = get_compile_time_arg_val(37);
    // COMMON-TRIGGER SYNC EVENT: the host parks every drainer in a tight spin and one release makes
    // them all stamp the same instant.
    constexpr uint32_t kSyncEvent = get_compile_time_arg_val(38);
    // Per-core service-interval histogram, ~0.5-1 us of a knee sweep (TT_METAL_PERF_DEBUG_DRISC_SVC).
    constexpr uint32_t kSvcInstr = get_compile_time_arg_val(40);
    // Master gate on the base instrumentation tier: the phase cycle counters, ~55 wall-clock reads per
    // sweep (~1 us of a 15 us knee sweep); 0 compiles them out for record runs. Functional clock reads
    // (credit and barrier deadlines, the pace gap, stop-path timing) are NOT behind this.
    constexpr uint32_t kInstr = get_compile_time_arg_val(41);
    // PER-CORE SHIP THRESHOLD: a frame costs the pipe the same whatever it carries, so a core ships only
    // when it is worth the frame. Binds on the core's FULLEST lane, not its span -- the producer that
    // blocks is always a lane, and a span percent under-reads the binding ring (span-5% = 551 stalls at
    // delay 10 vs lane-5% = 0).
    constexpr uint32_t kShipMinPct = get_compile_time_arg_val(39);
    constexpr uint32_t kShipMaxAgeSweeps = 512u;
    constexpr uint32_t kLaneShipWords = (kRingWords * kShipMinPct) / 100u;
    // CV-FIRST SWEEPS: read each core's ring TAILS (32 B), decide the ship set, then GATHER-READ only
    // the ship set's live runs, each straight to its packed wire offset in staging -- a staged frame is
    // its own wire image. The tails read is authoritative for the frames it parents: a frame claims
    // exactly [mirror, tail-at-this-read), and a producer only appends PAST a published tail (fenced
    // after its data), so the gather can trail the tail read by a whole batch without tearing.
    constexpr uint32_t kCvReadBytes = 32;
    constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
    // Idle backoff ceiling, ~5 us: collapse on work, creep when idle. 20 us exceeded a lane's fill time
    // at high rates and blinded the filler for most of a fill window (workload 806 -> 70 ms at delay 45).
    constexpr uint32_t kCvIdleGapMax = 6750;
    // Per-lane ship trigger, and (via kCvBusyPeak) the point past which the idle gap must stop growing:
    // a head only reaches a producer on a ship, so backing off on "shipped no frame" alone puts the
    // filler to sleep exactly while lanes fill toward the trigger.
    constexpr uint32_t kLaneTrigger = kRingWords / 2u;
    constexpr uint32_t kCvBusyPeak = kLaneTrigger / 2u;
    // Deliberately NO whole-span (raw) ship mode for worker cores: a raw sweep's dead ring bytes multiply
    // pressure exactly where direct push is narrowest (measured 60k producer stalls from the HIGH mode
    // this replaced; 0 with it unreachable). Only the self frame ships the raw span layout.
    static_assert(
        kSelfZones != 0 || kNGens * kGenSlots < kNStage,
        "CV staging needs a slot past the 2-generation pipeline (kNStage must be odd when self-zones are off)");
    // CV staging: its own slot past the pipeline, or -- under self-zones -- the self slot's ring 1..4
    // area, which is safe because only the self frame's ring 0 is ever live and the host walks a lane
    // only between its head and tail, so those bytes ship but are never decoded.
    constexpr uint32_t kCvSlot = kSelfZones != 0 ? kNStage : kNGens * kGenSlots;
    constexpr uint32_t kCvBase = kStageBase + kCvSlot * kSlotBytes +
                                 (kSelfZones != 0 ? (kPrefix + kCtrlWords + kRingWords) * 4u : 0u);
    static_assert(
        (kSelfZones != 0 ? (kPrefix + kCtrlWords + kRingWords) * 4u : 0u) + kCvReadBytes * kMaxCores <= kSlotBytes,
        "CV staging must fit inside the slot past the pipeline");
    // Drain bounce buffers, WIDE: CV staging uses at most 4 KiB of its slot and its tail is contiguous
    // with the bounce slots, so the bounces span the remainder as two 13,888 B chunks -- +31% drain bytes
    // per pump pass, the surplus that pulls the sustained equilibrium below the cap.
    constexpr uint32_t kBounceBase0 = kSelfZones != 0 ? kStageBase + kBounceSlot0 * kSlotBytes
                                                      : kCvBase + kCvReadBytes * kMaxCores;
    constexpr uint32_t kBounceBytes =
        kSelfZones != 0 ? kSlotBytes
                        : (((kNBounce + 1u) * kSlotBytes - kCvReadBytes * kMaxCores) / 2u) &
                              ~(kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u - 1u);
    static_assert(kBounceBase0 % (kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u) == 0, "bounces start on a page");
    static_assert(
        !kSpool || kBounceBase0 + kNBounce * kBounceBytes <= kStageBase + kNStage * kSlotBytes,
        "bounces must fit inside the mapped staging arena");
    constexpr bool kSelfPhases = kSelfZones != 0 && kSelfDetail != 0;
    // The self frame lives in slot kNStage -- one PAST every slot the drain pipeline can touch (the host
    // reserves it by passing nstage - 1 when this is on, so the OFF build is byte-identical).
    constexpr uint32_t kSelfSlot = kNStage;
    static_assert(kGenSlots >= 1, "need at least one slot per staging generation");

    static_assert(kSelfZones == 0 || kSelfHoldCycles >= 1, "a 0-cycle window hold would trace nothing");
    static_assert(kSelfDetail <= 1, "detail is 0 (SWEEP + PACE) or 1 (full per-batch phases)");
    static_assert(kSelfZones == 0 || kSelfMaxFrames >= 1, "self-profiling with a 0 frame budget captures nothing");
    // The sync event rides the self-zone marker ring, so with zones off it would have nowhere to go
    // and would silently measure nothing.
    static_assert(kSyncEvent == 0 || kSelfZones != 0, "the sync event rides the self-zone ring; enable zones");

    static_assert(kSpanWords * 4u <= NOC_MAX_BURST_SIZE, "a raw span read must fit one NoC burst");
    static_assert(kRingWords * 4u <= NOC_MAX_BURST_SIZE, "a gather read of a whole ring must fit one NoC burst");
    static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
    static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");
    // The packed gather's congruence argument (profiler_common.h): pads bring each run to its ring phase,
    // and everything else -- the slot base, the payload base, a wrap continuation -- must land congruent
    // with NO pad, which these divisibilities are the proof of. One pad rule serves both hops: the gather
    // READ into staging needs src == dst (mod NOC_L1_READ_ALIGNMENT_BYTES) and the frame's PCIe write
    // needs it mod NOC_PCIE_WRITE_ALIGNMENT_BYTES.
    static_assert(
        kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_PCIE_WRITE_ALIGNMENT_BYTES &&
            kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_L1_READ_ALIGNMENT_BYTES,
        "the shared pad rule no longer matches this part's NoC congruence");
    static_assert(
        kRingWords % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
            (kPrefix + kWireCtrl) % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
            kStageBase % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0 &&
            kSlotBytes % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0,
        "packed-gather congruence broken");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // start of profiler_msg_t on the worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));

    // RESYNC THE SOFTWARE NOC COUNTERS ON BOTH NOCS, ALWAYS: the barriers compare hardware counters
    // against software mirrors that persist across launches on this never-reset core, and firmware only
    // runs noc_local_state_init() on FW boot -- a run that ends with unacked writes wedges the next
    // launch's first barrier. kNocInit=0 brings the wedge back on demand for repro.
    if constexpr (kNocInit) {
        noc_local_state_init(NOC_INDEX);
        noc_local_state_init(kReadNoc);
    }
    // Does constructing Noc{kReadNoc} move the runtime global `noc_index`? The library barriers default
    // to that global while the writes are issued on the compile-time NOC_INDEX; if they diverge, the
    // barrier guarding staging reuse watches the wrong NoC.
    const uint32_t noc_index_before = noc_index;
    Noc noc{kReadNoc};
    const uint32_t noc_index_after = noc_index;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = kPcieEncOverride != 0 ? kPcieEncOverride : sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);
    // Egress write cmd state, programmed ONCE for both modes: nothing else on this core touches
    // write_cmd_buf on the egress NoC (head write-backs ride the read NoC), and re-programming it per
    // push was ~0.5 us a sweep on the direct knee.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    // The host-written ack word (a filler has exactly one downstream); the pump prices partial ships
    // against it.
    volatile tt_l1_ptr uint32_t* acked0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender.bytes_acked_base_addr);

    // Rendezvous words in the 64 B pad behind `stop` (only word 0 was used before; the `done` pad is full):
    //   +4 req (host asks) | +8 ack (kernel: parked in the spin) | +12 go (release = the measured instant)
    // The ack is what makes it a barrier -- without it the host would release a core still mid-sweep.
    volatile tt_l1_ptr uint32_t* sync_req = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr + 4);
    volatile tt_l1_ptr uint32_t* sync_ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr + 8);
    volatile tt_l1_ptr uint32_t* sync_go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr + 12);
    uint32_t sync_seen = 0;      // last generation this kernel has already served
    uint32_t sync_events = 0;    // releases observed and marked
    uint32_t sync_timeouts = 0;  // barriers abandoned because the release never came -- MUST stay 0
    uint32_t sync_spin_cyc = 0;  // cycles spent parked in the last barrier (host reports it as a sanity read)

    // Live liveness window, readable by the host WHILE the loop runs: without it the host cannot tell
    // "kernel exited" from "blocked" from "spinning with nothing to do". Every blocking call needs its
    // own phase marker or the phase word lies by omission.
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    volatile tt_l1_ptr uint32_t* phase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 8);
    constexpr uint32_t kPhaseInit = 1, kPhasePoll = 2, kPhaseReserve = 3, kPhaseWrite = 4, kPhaseExit = 5;
    // WRITE sub-phases: 6=chunked NoC write to the PCIe tile, 7=push_pages bookkeeping, 8=bytes_sent
    // notify, 9=write issued.
    constexpr uint32_t kPhWrChunk = 6, kPhWrPush = 7, kPhWrNotify = 8, kPhWrDone = 9;
    // 10=frame dropped, 11=sweep-body write barrier, 14=exit socket_barrier, 15=post-loop barrier.
    constexpr uint32_t kPhDropped = 10, kPhBar1 = 11;
    constexpr uint32_t kPhTailBar = 15;
    constexpr uint32_t kPhSockBar = 14;
    *hb = 0;
    *phase = kPhaseInit;

    // Every frame's prefix is IDENTICAL, and of the control words only the heads, tails and SPSC_CORE_XY
    // are staged per frame -- the rest must read zero on the wire -- so both are written once here.
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefix + kCtrlWords; k++) {
            pfx[k] = 0;
        }
    }

    static uint32_t head_mirror[kMaxCores * kNumRisc];
    // kSvcInstr only -- per-core service interval as a distribution (the stalls live in its TAIL, not
    // its mean). Buckets double from 8192 cycles; sampled at the head write-back, the moment a ship
    // releases a producer.
    static uint32_t last_ship[kMaxCores];
    static uint32_t svc_hist[8];
    uint32_t svc_max = 0;
    static uint8_t seeded[kMaxCores];
    static uint8_t hot[kMaxCores];  // shipped real words last scan; a hot core scanning live==0 is publish lag
    static uint16_t ship_age[kMaxCores];
    static uint8_t ship_list[kMaxCores];  // CV-first: this sweep's ship set, dense core indices
    // Per-slot frame geometry, written at gather-read issue and consumed a whole batch later by the ship --
    // stored, not recomputed, so the two phases cannot diverge. slot_payload[kSelfSlot] is the raw self
    // frame's constant payload.
    static uint32_t slot_runs[kNStage * kNumRisc];
    static uint8_t slot_core[kNStage];
    static uint32_t slot_payload[kNStage + 1];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
        hot[i] = 0;
        ship_age[i] = 0;
        last_ship[i] = 0;
    }
    for (uint32_t i = 0; i < 8; i++) {
        svc_hist[i] = 0;
    }
    uint32_t ship_deferred = 0;  // core visits left unstaged by the ship threshold
    uint32_t ship_aged = 0;      // ships forced by kShipMaxAgeSweeps

    uint64_t total_words = 0;
    uint32_t pages = 0;
    uint32_t frames = 0;
    uint32_t pushes = 0;
    uint32_t sweeps = 0;
    uint32_t max_occ = 0;
    // Seeded at 0 = sweep immediately: seeding high to skip the idle ramp stalled producers at burst
    // onset.
    uint32_t gap = kGapCycles;
    uint32_t overflows = 0;
    uint32_t hb_slot = 0;
    uint32_t scan_rot = 0;
    uint32_t deferred_seen = 0;  // ship_deferred as of the last rotation decision
    uint32_t fill_hist[8] = {};  // busy-sweep PEAK lane fill, 1/8-of-ring buckets

    // ================================ INSTRUMENTATION START (kInstr phase/window counters) ====
    uint64_t c_scan = 0;     // decide loop alone
    uint64_t c_read = 0;     // bulk span reads: issue + whatever wait survives the ship overlap
    uint64_t c_issue = 0;    // the issue half of c_read
    uint64_t c_cv = 0;       // the sweep-start CV pass alone
    uint32_t n_gather_rd = 0, n_cv_rd = 0;
    uint64_t c_proc = 0;     // control-vector inspection, prefix + head patch, head write-back
    uint64_t c_reserve = 0;  // host credit wait
    uint64_t c_write = 0;    // PCIe write + push + notify
    uint64_t c_barrier = 0;  // write barrier before staging is reused
    uint64_t c_ph_head = 0;  // the per-core head write-back inside proc (kSvcInstr)
    uint64_t c_wr_chunk = 0;
    uint64_t c_wr_push = 0;
    uint64_t c_wr_notify = 0;
    uint64_t c_idle = 0;
    uint64_t c_busy = 0;
    // WINDOW-SCOPED phases: lifetime percentages are dominated by idle CV polling (a drainer is resident
    // for seconds, a capture is tens of ms), so snapshot at the first shipping sweep and diff.
    bool win2_open = false;
    uint64_t w0_t = 0, w1_t = 0, w0_busy = 0, w1_busy = 0, w0_idle = 0, w1_idle = 0, w0_pace = 0, w1_pace = 0;
    uint64_t w0_cv = 0, w1_cv = 0, w0_issue = 0, w1_issue = 0;
    uint32_t w0_frames = 0, w1_frames = 0, w0_sweeps = 0, w1_sweeps = 0;
    uint32_t sweeps_idle = 0;
    uint32_t max_sweep = 0;
    // Worst-sweep phase split: the knee is set by the WORST sweep (~2.5x the mean) beating ring fill.
    uint32_t ws_read = 0, ws_proc = 0, ws_rsv = 0, ws_wr = 0, ws_bar = 0;
    uint32_t max_reserve = 0;
    uint64_t c_pace = 0;
    // ================================ INSTRUMENTATION END (kInstr phase/window counters) ====
    // Set when a bounded write barrier expires: egress is dead, so STOP SHIPPING for good.
    // Never means "continue anyway" -- staging reuse depends on that barrier having flushed.
    bool egress_dead = false;
    // Which staging generations hold a ship possibly still in flight. Persists ACROSS sweeps: the write
    // wait happens at the slots' next refill, never at sweep end, so a sweep's final ship drains under
    // the pace gap or the next sweep's CV pass instead of on the sweep's critical path.
    bool gen_shipped[kNGens] = {};

    // ---- GDDR spool state (all dead code when kSpool == 0) ----
    // Byte counters are MONOTONIC 64-bit (long captures exceed 4 GiB per filler); ring offsets are kept
    // incrementally so the hot path never takes a runtime modulo.
    uint64_t spool_wr = 0;        // bytes appended by the DMA ship
    uint64_t spool_done = 0;      // bytes whose DMA writes have completed (observable to stream-1 reads)
    uint64_t spool_rd_iss = 0;    // bytes a bounce refill has been ISSUED for (the read cursor)
    uint64_t spool_rd = 0;        // bytes whose refill reads COMPLETED -- only these spool slots may be rewritten
    uint32_t spool_wr_off = 0;    // spool_wr % kSpoolBytes
    uint32_t spool_rd_iss_off = 0;  // spool_rd_iss % kSpoolBytes
    uint32_t spool_max = 0;        // peak occupancy, bytes
    uint32_t spool_drops = 0;      // frames dropped because the spool was full
    uint32_t dma_issued = 0;       // cumulative stream-0 writes, for the per-generation completion gate
    uint32_t gen_dma_mark[kNGens] = {};  // dma_issued as of each generation's ship
    // Drain bounce buffers: at most one READING (stream-1 reads in flight) and one SHIPPING (egress NoC
    // writes unflushed) at a time, so every pump pass is a poll and the drain can never stall the sweep.
    constexpr uint32_t kBounceEmpty = 0, kBounceReading = 1, kBounceReady = 2, kBounceShipping = 3;
    uint32_t b_state[2] = {kBounceEmpty, kBounceEmpty};
    uint32_t b_bytes[2] = {};       // spool bytes held
    uint32_t b_off[2] = {};         // bytes of those already pushed to the host (partial ships under credit)
    uint32_t b_ack_target[2] = {};  // noc_nonposted_writes_acked snapshot at ship: this bounce's flush line
    uint32_t b_seq[2] = {};         // refill order, so a both-READY pass ships the older bytes first
    uint32_t b_rd_mark[2] = {};     // dma_rd_issued as of each bounce's refill: its completion line
    uint64_t b_rd_end[2] = {};      // spool_rd_iss after its refill: what spool_rd advances to on READY
    uint32_t dma_rd_issued = 0;     // cumulative stream-1 reads
    bool notify_pending = false;    // pump ships owe the host a bytes_sent notify (batched per sweep)
    bool pump_pressure = false;     // hysteresis state of the on-sweep pressure pump
    bool spool_lossy = false;  // a full-spool wait expired: consumer gone, later frames drop without waiting
    uint32_t drain_chunks = 0;   // bounce fills
    uint32_t drain_ships = 0;    // host pushes issued by the pump (partials count)
    uint32_t drain_starved = 0;  // pump passes that held a READY bounce but had zero credit
    uint64_t c_drain = 0;        // pump cycles OUTSIDE the pace gap (in-gap pumping is already c_pace)
    bool drain_dead = false;     // exit drain deadline expired: bytes stranded in the spool

    uint32_t credit_timeouts = 0;  // bounded credit wait expired -> frame dropped instead of deadlocking
    uint32_t dropped_frames = 0;
    // ================================ INSTRUMENTATION START (self-zone + NoC-footprint state and marker path) ====
    // ---- NoC FOOTPRINT state (all compiled out when kNocFootprint == 0). Index order kNfRdW..kNfWrT is
    // shared with the host's out[] report -- wire format, not an implementation detail.
    NocFpState nf{};
    // Self-profiling producer: 2-word markers + sticky timers into a 512-word ring, same wall-clock
    // register as the workers, so nothing needs calibrating. Zones stamp their own clock reads.
    volatile tt_l1_ptr uint32_t* self_ctrl =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + kSelfSlot * kSlotBytes + kPrefix * 4u);
    volatile tt_l1_ptr uint32_t* self_ring = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kStageBase + kSelfSlot * kSlotBytes + (kPrefix + kCtrlWords) * 4u);
    uint32_t self_head = 0;           // words the host has been shown (consumer side, kept by us)
    uint32_t self_tail = 0;           // words written (producer side)
    uint32_t self_hi = 0xFFFFFFFFu;   // last wall-clock high half emitted; ~0 forces a first sticky
    uint32_t self_frames = 0;         // self frames shipped
    uint32_t self_markers = 0;        // markers written into the ring
    uint32_t self_dropped = 0;        // markers refused because a publish could not free the ring
    uint32_t self_sweeps = 0;         // sweeps instrumented -- every sweep inside an active window
    uint32_t self_sweeps_work = 0;    // of those, the ones that did real work
    uint32_t self_windows = 0;        // times the window was armed from cold
    uint32_t self_words_shipped = 0;  // words actually SHIPPED; must end equal to self_tail
    uint32_t self_over = 0;           // sweeps left uninstrumented because the frame budget was spent
    uint64_t c_self = 0;              // cycles spent publishing self frames (the perturbation)
    uint64_t self_t_sweep0 = 0;       // this sweep's start; self_arm keys the hold window off it
    uint64_t self_armed_until = 0;    // wall clock: instrument every sweep that STARTS before this
    bool self_on = false;             // this sweep is being instrumented
    bool self_busy = false;           // inside self_publish: suppress marker emission (re-entrancy)
    bool self_work = false;           // THIS sweep did work (set by self_arm / the end-of-sweep check)
    bool self_from_start = false;     // instrumented from the top of the sweep, not armed part-way in
    // Phase totals over the instrumented-FROM-THE-START sweeps only, so summed Tracy zone durations
    // have something to check against. A sweep armed part-way through would mismatch for a reason
    // that is not a bug.
    uint32_t self_ck_sweeps = 0;
    uint64_t self_ck_read = 0, self_ck_proc = 0, self_ck_rsv = 0, self_ck_write = 0, self_ck_bar = 0;
    if constexpr (kSelfZones != 0) {
        volatile tt_l1_ptr uint32_t* pfx =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + kSelfSlot * kSlotBytes);
        // The self frame ships RAW -- whole rings, decoded circularly against its control vector -- so the
        // marker ring never needs packing into wire order. Its geometry is constant, staged once here.
        pfx[0] = kernel_profiler::spsc_span_w0() | kernel_profiler::SPSC_SPAN_RAW_FLAG;
        pfx[1] = kSpanWords;
        slot_payload[kSelfSlot] = kSpanWords;
        for (uint32_t k = 2; k < kPrefix; k++) {
            pfx[k] = 0;
        }
        // The whole control vector, not just the words we use: it ships verbatim and the host reads a head
        // and a tail for all five RISCs. Rings 1-4 must read tail == head == 0 forever or the decoder would
        // walk uninitialised L1 as markers.
        for (uint32_t k = 0; k < kCtrlWords; k++) {
            self_ctrl[k] = 0;
        }
        self_ctrl[kernel_profiler::SPSC_CORE_XY] = kSelfXY;
    }
    // Publish trampoline: self_mark_w0's ring-full path must call self_publish, which is defined after
    // the egress lambdas because it ships THROUGH them -- a reference cycle no lambda declaration order
    // can express. Wired immediately after self_publish is defined; re-entrancy guarded by self_busy.
    struct SelfPub {
        void* ctx = nullptr;
        void (*fn)(void*) = nullptr;
    } self_pub;
    // Append one 2-word marker at a fresh timestamp; publishes first if the ring cannot hold a sticky
    // plus a marker, and drops (counted) only if that publish could not free room.
    auto self_mark_w0 = [&](uint32_t w0) -> bool {
        if constexpr (kSelfZones == 0) {
            (void)w0;
            return false;
        } else {
            if (!self_on || self_busy) {
                return false;
            }
            // 9-word margin covers every emitted shape (sticky + marker, sticky + PP_DATA sample).
            // RING FULL -> PUBLISH AND CARRY ON: publishing here rather than once per sweep is what makes
            // tracing every sweep affordable -- a frame carries as many markers as the ring holds.
            if (self_tail - self_head > kRingWords - 9u) {
                if (self_pub.fn != nullptr) {
                    self_pub.fn(self_pub.ctx);
                }
                if (self_tail - self_head > kRingWords - 9u) {
                    self_dropped++;
                    return false;
                }
            }
            // Clock read AFTER any publish, so a marker never carries a time from before the publish
            // that made room for it.
            const uint64_t ts = get_timestamp();
            const uint32_t hi = static_cast<uint32_t>(ts >> 32);
            if (hi != self_hi) {
                self_ring[self_tail % kRingWords] = kernel_profiler::spsc_sticky_timer_w0(hi);
                self_tail++;
                self_hi = hi;
            }
            self_ring[self_tail % kRingWords] = w0;
            self_tail++;
            self_ring[self_tail % kRingWords] = static_cast<uint32_t>(ts & 0xFFFFFFFFu);
            self_tail++;
            self_markers++;
            return true;
        }
    };
    // Guards inline at the scope site: an uninstrumented sweep pays a flag check, never a call.
    auto self_mark_now = [&](uint32_t w0) -> bool {
        if constexpr (kSelfZones == 0) {
            (void)w0;
            return false;
        } else {
            if (!self_on || self_busy) {
                return false;
            }
            return self_mark_w0(w0);
        }
    };
    auto self_mark_phase = [&](uint32_t w0) -> bool {
        if constexpr (!kSelfPhases) {
            (void)w0;
            return false;
        } else {
            if (!self_on || self_busy) {
                return false;
            }
            return self_mark_w0(w0);
        }
    };
    using SelfMarkNow = decltype(self_mark_now);
    using SelfMarkPhase = decltype(self_mark_phase);
    // ================================ INSTRUMENTATION END (self-zone + NoC-footprint state and marker path) ====
    // ~50 ms at 1.35 GHz. Enormously above anything healthy (worst observed credit wait is ~0.1 us), so it
    // never fires in normal operation -- it exists purely to convert "wait forever" into "lose a frame".
    constexpr uint64_t kCreditWaitCycles = 67500000ull;

    // ---- SPOOL DRAIN PUMP: one poll pass, never a spin -- every wait this stage could have is a STATE a
    // later pass observes, so the pump can delay host delivery but never the sweep. Runs at sweep end,
    // in the pace gap, and in the exit tail; NEVER unconditionally per pipeline batch (a shipping pass
    // costs ~1.3 us and per-batch placement cost 119k stalls at d10). Bursts are the spool's job to
    // absorb; the pump only has to win on average.
    auto drain_pump = [&]() -> bool {
        if constexpr (!kSpool) {
            return false;
        } else {
            // L1-only early-out so the idle-sweep and pace-gap passes cost no DMA/NIU MMIO reads.
            if (spool_wr == spool_rd && b_state[0] == kBounceEmpty && b_state[1] == kBounceEmpty) {
                return false;
            }
            bool did = false;
            // SHIPPING -> EMPTY once the egress writes are ACKED -- full flush, not "sent": the bounce's
            // next writer is the DMA engine, which the sent-only gate does not fence (measured as decode
            // order regressions). Per-bounce flush lines so both bounces ride the wire at once.
            for (uint32_t i = 0; i < 2; i++) {
                if (b_state[i] == kBounceShipping &&
                    static_cast<int32_t>(NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_WR_ACK_RECEIVED) - b_ack_target[i]) >=
                        0) {
                    b_state[i] = kBounceEmpty;
                    did = true;
                }
            }
            // READING -> READY when a bounce's stream-1 reads retire (spool_rd, the ship side's free-space
            // fence, then advances). FIFO completion order gives each bounce its own line from one
            // outstanding count, so BOTH bounces can be filling at once.
            if (b_state[0] == kBounceReading || b_state[1] == kBounceReading) {
                const uint32_t rd_out = experimental::dma_get_reads_outstanding(kDmaDrain);
                for (uint32_t i = 0; i < 2; i++) {
                    if (b_state[i] == kBounceReading && rd_out <= dma_rd_issued - b_rd_mark[i]) {
                        b_state[i] = kBounceReady;
                        if (b_rd_end[i] > spool_rd) {
                            spool_rd = b_rd_end[i];
                        }
                        did = true;
                    }
                }
            }
            // Refill an EMPTY bounce BEFORE shipping the READY one, so the stream-1 read runs under the
            // ship's NoC issue. Only SHIP-COMPLETED bytes are readable: nothing short of a stream-0
            // write's completion orders a stream-1 read of the same address behind it.
            if (spool_done != spool_wr && experimental::dma_get_writes_outstanding(kDmaShip) == 0) {
                spool_done = spool_wr;
            }
            // The SECOND concurrent refill only under pressure: at a burst the extra in-flight GDDR read
            // deepens the bank queue exactly when the ship DMA needs it (d8: 0 -> 3.2k stalls).
            const uint32_t emp = b_state[0] == kBounceEmpty ? 0u : (b_state[1] == kBounceEmpty ? 1u : 2u);
            if (emp != 2u && (pump_pressure || b_state[emp ^ 1u] != kBounceReading) && spool_done != spool_rd_iss) {
                uint32_t len = static_cast<uint32_t>(spool_done - spool_rd_iss);
                if (len > kBounceBytes) {
                    len = kBounceBytes;
                }
                if (len > kSpoolBytes - spool_rd_iss_off) {
                    len = kSpoolBytes - spool_rd_iss_off;
                }
                dma_read_unchecked(kDmaDrain, kSpoolBase + spool_rd_iss_off, kBounceBase0 + emp * kBounceBytes, len);
                spool_rd_iss += len;
                spool_rd_iss_off += len;
                if (spool_rd_iss_off == kSpoolBytes) {
                    spool_rd_iss_off = 0;
                }
                b_rd_mark[emp] = ++dma_rd_issued;
                b_rd_end[emp] = spool_rd_iss;
                b_bytes[emp] = len;
                b_off[emp] = 0;
                b_seq[emp] = drain_chunks;
                b_state[emp] = kBounceReading;
                drain_chunks++;
                did = true;
            }
            // Ship a READY bounce, as much as the host FIFO has credit for RIGHT NOW -- partial ships keep
            // the FIFO fed under credit pressure. No init and no notify here (cmd state programmed once
            // at entry, notify batched per sweep -- together most of a shipping pass's 1.3 us). OLDEST
            // first when both are READY: the socket is a byte stream and the younger bounce would reorder
            // the wire.
            uint32_t rdy = 2u;
            if (b_state[0] == kBounceReady && b_state[1] == kBounceReady) {
                rdy = static_cast<int32_t>(b_seq[0] - b_seq[1]) < 0 ? 0u : 1u;
            } else if (b_state[0] == kBounceReady) {
                rdy = 0u;
            } else if (b_state[1] == kBounceReady) {
                rdy = 1u;
            }
            if (rdy != 2u && !egress_dead) {
                invalidate_l1_cache();
                const uint32_t bytes_free = sender.downstream_fifo_total_size - (sender.bytes_sent - *acked0);
                uint32_t nb = b_bytes[rdy] - b_off[rdy];
                if (bytes_free < nb) {
                    nb = bytes_free & ~(kPageBytes - 1u);
                }
                if (nb == 0) {
                    drain_starved++;
                } else {
                    const uint32_t fifo_size = sender.downstream_fifo_curr_size;
                    uint32_t wr = sender.write_ptr;
                    if (wr >= fifo_size) {
                        wr -= fifo_size;
                    }
                    const uint32_t src = kBounceBase0 + rdy * kBounceBytes + b_off[rdy];
                    const uint32_t first = (wr + nb > fifo_size) ? fifo_size - wr : nb;
                    write_to_host_chunked(pcie_xy_enc, src, pcie_base + wr, first);
                    if (first < nb) {
                        write_to_host_chunked(pcie_xy_enc, src + first, pcie_base, nb - first);
                    }
                    socket_push_pages(sender, nb / kPageBytes);
                    notify_pending = true;
                    pages += nb / kPageBytes;
                    pushes++;
                    drain_ships++;
                    b_off[rdy] += nb;
                    if (b_off[rdy] == b_bytes[rdy]) {
                        b_state[rdy] = kBounceShipping;
                        b_off[rdy] = 0;
                        // The mirror is cumulative, so this line also covers any earlier partial ships.
                        b_ack_target[rdy] = noc_nonposted_writes_acked[NOC_INDEX];
                    }
                    did = true;
                }
            }
            return did;
        }
    };

    // Ship `count` adjacent staged slots, ONE write per frame (two across a wrap): a staged slot IS its
    // frame's wire image. The trailing page fill is never written -- the host derives every offset from
    // the control vector and reads past it.
    auto emit_slots = [&](uint32_t start, uint32_t count) {
        if (count == 0) {
            return;
        }
        if (egress_dead) {
            *phase = kPhDropped;
            dropped_frames += count;
            return;
        }
        if constexpr (kSpool) {
            uint32_t bytes = 0;
            for (uint32_t f = 0; f < count; f++) {
                bytes += kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) * 4u;
            }
            // Full spool: PUMP UNTIL THERE IS ROOM -- this wait, not a drop, is the spool's back-pressure
            // (frames safe in staging, sweep slows, producers stall, the direct path's credit-wait shape).
            // A frame drops only past the same escapes as reserve_pages_bounded: host stop, consumer
            // already gone, or the dead-consumer deadline -- one expiry flags the consumer gone so later
            // frames do not re-pay it.
            if (kSpoolBytes - static_cast<uint32_t>(spool_wr - spool_rd) < bytes && !egress_dead && !spool_lossy) {
                *phase = kPhaseReserve;
                const uint64_t t_a = get_timestamp() + kCreditWaitCycles;
                while (kSpoolBytes - static_cast<uint32_t>(spool_wr - spool_rd) < bytes && *stop == 0) {
                    drain_pump();
                    if (get_timestamp() >= t_a) {
                        spool_lossy = true;
                        break;
                    }
                }
            }
            if (kSpoolBytes - static_cast<uint32_t>(spool_wr - spool_rd) < bytes) {
                *phase = kPhDropped;
                spool_drops += count;
                dropped_frames += count;
                return;
            }
            // The DMA engine reads the control and length words the scalar core staged; Blackhole stores
            // can reach SRAM out of order.
            asm volatile("fence" ::: "memory");
            const uint64_t t0 = kInstr != 0 ? get_timestamp() : 0;
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfMarkPhase> z_write(self_mark_phase);
            *phase = kPhaseWrite;
            for (uint32_t f = 0; f < count; f++) {
                uint32_t src = kStageBase + (start + f) * kSlotBytes;
                // WHOLE page-rounded frames, dead tail bytes included: the spool offset then advances in
                // lockstep with the FIFO write pointer, so the spool is a byte-exact image of the wire
                // and the drain needs no frame geometry at all.
                uint32_t len = kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) * 4u;
                while (len != 0) {
                    const uint32_t piece = len > kSpoolBytes - spool_wr_off ? kSpoolBytes - spool_wr_off : len;
                    dma_write_unchecked(kDmaShip, src, kSpoolBase + spool_wr_off, piece);
                    dma_issued++;
                    spool_wr += piece;
                    spool_wr_off += piece;
                    if (spool_wr_off == kSpoolBytes) {
                        spool_wr_off = 0;
                    }
                    src += piece;
                    len -= piece;
                }
            }
            const uint32_t occ = static_cast<uint32_t>(spool_wr - spool_rd);
            if (occ > spool_max) {
                spool_max = occ;
            }
            c_write += kInstr != 0 ? (get_timestamp() - t0) : 0;
            *phase = kPhWrDone;
            return;
        }
        uint32_t npages = 0;
        for (uint32_t f = 0; f < count; f++) {
            npages += kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) / kPageWords;
        }
        // The NIU reads the control and length words the scalar core staged; Blackhole stores can reach
        // SRAM out of order.
        asm volatile("fence" ::: "memory");
        const uint64_t t0 = get_timestamp();
        *phase = kPhaseReserve;
        bool credited;
        {
            // Suppressed while self_publish ships the self frame through this same path, so the self
            // frame's own egress is never a zone.
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_CREDIT_WAIT, SelfMarkPhase> z_credit(
                self_mark_phase);
            credited = reserve_pages_bounded(sender, npages, t0 + kCreditWaitCycles, stop);
        }
        *phase = kPhaseWrite;
        if (!credited) {
            // DROP rather than block: the heads for these slots were already written back, so the producers
            // stay unblocked and the workload completes. Capture is best-effort; the workload is not.
            *phase = kPhDropped;
            credit_timeouts++;
            dropped_frames += count;
            c_reserve += kInstr != 0 ? (get_timestamp() - t0) : 0;
            return;
        }
        // t0 stays real (it is the credit deadline); anchoring the gated chain to it keeps every diff 0.
        const uint64_t t1 = kInstr != 0 ? get_timestamp() : t0;
        c_reserve += t1 - t0;
        if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
            max_reserve = static_cast<uint32_t>(t1 - t0);
        }
        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfMarkPhase> z_write(self_mark_phase);
        *phase = kPhWrChunk;
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        uint32_t wr = sender.write_ptr;
        // dst is a FIFO offset; socket_push_pages only wraps the pointer, it does not split a transfer, so
        // a piece crossing the FIFO wrap is split here. The split preserves the src/dst NoC congruence the
        // pack pads establish: fifo_size is a whole number of 64 B pages.
        auto put = [&](uint32_t src, uint32_t dst, uint32_t len) {
            if (dst >= fifo_size) {
                dst -= fifo_size;
            }
            const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
            write_to_host_chunked(pcie_xy_enc, src, pcie_base + dst, first);
            if (first < len) {
                write_to_host_chunked(pcie_xy_enc, src + first, pcie_base, len - first);
            }
        };
        for (uint32_t f = 0; f < count; f++) {
            const uint32_t payload = slot_payload[start + f];
            put(kStageBase + (start + f) * kSlotBytes, wr, (kPrefix + payload) * 4u);
            wr += kernel_profiler::spsc_span_frame_words(payload) * 4u;
            if (wr >= fifo_size) {
                wr -= fifo_size;
            }
        }
        const uint64_t t2 = kInstr != 0 ? get_timestamp() : t1;
        c_wr_chunk += t2 - t1;
        *phase = kPhWrPush;
        socket_push_pages(sender, npages);
        const uint64_t t3 = kInstr != 0 ? get_timestamp() : t2;
        c_wr_push += t3 - t2;
        *phase = kPhWrNotify;
        // NOT socket_notify_receiver: that re-inits write_cmd_buf onto NOC_UNICAST_WRITE_VC, and on a
        // filler whose data rides the other unicast VC the mesh can deliver the bytes_sent word ahead of
        // the data it announces (measured: 325 resyncs / 1,957 bad frames on one socket). Same cmd state,
        // VC and route as the data makes delivery order the issue order again.
        volatile tt_l1_ptr sender_socket_md* cfg =
            reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr);
        cfg->bytes_sent = sender.bytes_sent;
        asm volatile("fence" ::: "memory");
        write_to_host_chunked(
            pcie_xy_enc,
            sender.config_addr,
            (static_cast<uint64_t>(sender.d2h.bytes_sent_addr_hi) << 32) | sender.downstream_bytes_sent_addr,
            4u);
        const uint64_t t4 = kInstr != 0 ? get_timestamp() : t3;
        c_wr_notify += t4 - t3;
        c_write += t4 - t1;
        *phase = kPhWrDone;
        pages += npages;
        pushes++;
    };

    // The batched bytes_sent notify for pump ships: once per sweep instead of once per chunk (the host
    // learns of new bytes ~15 us later, against multi-MiB FIFOs). Same cmd state, VC and route as the
    // data, for the delivery-order argument the direct path's notify documents.
    auto drain_notify = [&]() {
        if constexpr (!kSpool) {
            return;
        } else {
            if (!notify_pending || egress_dead) {
                return;
            }
            volatile tt_l1_ptr sender_socket_md* cfg =
                reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr);
            cfg->bytes_sent = sender.bytes_sent;
            asm volatile("fence" ::: "memory");
            write_to_host_chunked(
                pcie_xy_enc,
                sender.config_addr,
                (static_cast<uint64_t>(sender.d2h.bytes_sent_addr_hi) << 32) | sender.downstream_bytes_sent_addr,
                4u);
            notify_pending = false;
        }
    };

    // ================================ INSTRUMENTATION START (self-frame publish, NoC-footprint sampling, window arming) ====
    // Barrier at the END: the next marker overwrites a word an in-flight frame may still be shipping, so
    // the wait belongs before the NEXT publish. Phase counters are saved/restored around the egress call
    // so the self frame does not bill itself.
    auto self_publish = [&]() {
        if constexpr (kSelfZones == 0) {
            return;
        } else {
            if (self_tail == self_head) {
                return;
            }
            const uint64_t t_s0 = get_timestamp();
            self_busy = true;
            self_ctrl[kernel_profiler::SPSC_RING_HEAD_0] = self_head;
            self_ctrl[kernel_profiler::SPSC_RING_TAIL_0] = self_tail;
            asm volatile("fence" ::: "memory");
            const uint64_t s_rsv = c_reserve, s_wr = c_write, s_ch = c_wr_chunk, s_pu = c_wr_push, s_no = c_wr_notify;
            const uint32_t s_pages = pages, s_pushes = pushes, s_maxr = max_reserve;
            emit_slots(kSelfSlot, 1);
            c_reserve = s_rsv;
            c_write = s_wr;
            c_wr_chunk = s_ch;
            c_wr_push = s_pu;
            c_wr_notify = s_no;
            pages = s_pages;
            pushes = s_pushes;
            max_reserve = s_maxr;
            const bool self_out = kSpool ? dma_wait_writes_bounded(kDmaShip, 0, get_timestamp() + kCreditWaitCycles)
                                         : write_barrier_bounded(get_timestamp() + kCreditWaitCycles);
            if (self_out) {
                self_words_shipped += self_tail - self_head;
                self_head = self_tail;
                self_frames++;
            } else {
                // Egress is dead. emit_slots has already accounted the drop; stop instrumenting rather
                // than keep writing into a ring nothing will ever read.
                egress_dead = true;
            }
            self_busy = false;
            c_self += get_timestamp() - t_s0;
        }
    };

    // Captureless, so it converts to a plain function pointer.
    self_pub.ctx = static_cast<void*>(&self_publish);
    self_pub.fn = [](void* p) { (*static_cast<decltype(self_publish)*>(p))(); };
    // ARM MID-SWEEP: deciding at sweep top misses the first sweep of every burst. Scopes already
    // open stay unrecorded, so the discovery sweep is partial and later ones whole.
    auto self_arm = [&]() {
        if constexpr (kSelfZones == 0) {
            return;
        } else {
            self_work = true;
            // Refreshed on every later discovery of work, so a burst keeps the window open and coverage inside
            // it stays CONTIGUOUS.
            self_armed_until = self_t_sweep0 + kSelfHoldCycles;
            if (self_on || self_busy) {
                return;
            }
            if (self_frames >= kSelfMaxFrames) {
                return;
            }
            self_on = true;
            self_windows++;
        }
    };


    // NoC FOOTPRINT sampling: the only place NIU registers are read. Both NoCs are sampled -- which NoC
    // carries what is the thing being verified, so the zeros are part of the measurement.
    auto nf_end = [&](uint64_t t_sweep0, uint32_t sweep_cyc, bool did_work) {
        if constexpr (kNocFootprint == 0) {
            (void)t_sweep0;
            (void)sweep_cyc;
            (void)did_work;
            return;
        } else {
            nf_sweep_end(&nf, sweeps, t_sweep0, sweep_cyc, did_work);
        }
    };
    // ================================ INSTRUMENTATION END (self-frame publish, NoC-footprint sampling, window arming) ====

    if constexpr (kNocFootprint != 0) {
        // Seed the mirrors so the first sweep's delta is measured from HERE, not from whatever the counters
        // held at chip reset -- everything the bring-up path did is therefore excluded.
        nf_sample_regs(&nf);
        for (uint32_t i = 0; i < kNfSlots; i++) {
            nf.life[i] = 0;
        }
        nf.cost = 0;
    }

    // Stop-path sweep-to-empty: on stop=1 keep sweeping until one whole sweep moves nothing, so markers
    // still in worker rings ship instead of being stranded. Exiting on the stop word directly is what
    // silently truncated captures.
    constexpr uint64_t kStopDrainCycles = 1350000000;
    uint64_t stop_seen_at = 0;
    uint64_t words_at_stop = 0;
    uint32_t frames_at_stop_check = 0;
    uint32_t stop_sweeps = 0;
    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && !egress_dead) {
        invalidate_l1_cache();
        if (*stop != 0) {
            if (stop_seen_at == 0) {
                stop_seen_at = get_timestamp();
                words_at_stop = total_words;
            } else if (frames == frames_at_stop_check || get_timestamp() - stop_seen_at > kStopDrainCycles) {
                break;
            }
            frames_at_stop_check = frames;
            stop_sweeps++;
        }
        // ================================ INSTRUMENTATION START (common-trigger sync rendezvous) ====
        // ---- COMMON-TRIGGER SYNC EVENT: the rendezvous ----
        // FIRST in the loop body, before sweeps++ and before t_sweep0, so a barrier wait is billed to no
        // sweep and cannot perturb the phase accounting.
        if constexpr (kSyncEvent != 0) {
            invalidate_l1_cache();
            const uint32_t req = *sync_req;
            if (req != sync_seen) {
                sync_seen = req;
                const uint64_t t_park = get_timestamp();
                *sync_ack = req;  // parked; the host may release once every drainer has done this
                uint64_t t_go = 0;
                // Bounded, so a host that never releases DEGRADES instead of wedging the workload.
                uint32_t guard = 0xFFFFFFFFu;
                for (;;) {
                    invalidate_l1_cache();
                    if (*sync_go == req) {
                        t_go = get_timestamp();  // THE MEASURED INSTANT -- ~5 instructions after the release
                        break;
                    }
                    if (*stop != 0 || --guard == 0) {
                        break;
                    }
                }
                if (t_go != 0) {
                    // Force emission: the sync zone is NOT part of the work-armed window and must land whether or not
                    // this sweep would have been instrumented.
                    self_on = true;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SYNC, SelfMarkNow> z_sync(
                            self_mark_now);
                    }
                    self_publish();  // ship it now; do not let it wait on the ring filling
                    sync_events++;
                    sync_spin_cyc = static_cast<uint32_t>(t_go - t_park);
                } else {
                    sync_timeouts++;
                }
            }
        }
        // ================================ INSTRUMENTATION END (common-trigger sync rendezvous) ====
        sweeps++;
        *hb = sweeps;
        *phase = kPhasePoll;
        const uint64_t t_sweep0 = (kInstr != 0 || kSelfZones != 0) ? get_timestamp() : 0;
        const uint32_t frames_at_sweep_start = frames;
        const uint64_t s_read0 = c_read, s_proc0 = c_proc, s_rsv0 = c_reserve, s_wr0 = c_write, s_bar0 = c_barrier;
        const uint64_t words_at_sweep_start = total_words;

        // Inside an active window EVERY sweep is instrumented, one register compare against a deadline.
        // Only the FIRST sweep of a window is partial, arming from self_arm when work is discovered.
        if constexpr (kSelfZones != 0) {
            self_t_sweep0 = t_sweep0;
            self_work = false;
            self_on = false;
            self_from_start = false;
            if (self_frames >= kSelfMaxFrames) {
                self_over++;
                // The budget running out is not a reason to discard zones already written.
                self_publish();
            } else if (t_sweep0 < self_armed_until) {
                self_on = true;
                self_from_start = true;
            }
        }

        uint32_t sweep_cyc = 0;
        uint32_t sweep_peak = 0;
        {
            // Constructed AFTER the arming block decided self_on, so an armed-window sweep records its whole
            // body. A sweep that arms mid-body gets only its post-arm children.
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SWEEP, SelfMarkNow> z_sweep(self_mark_now);
            {
                // ---- software pipeline: gather generation G on kReadNoc while G^1 ships on NOC_INDEX ----
                uint32_t gen = 0;
                uint32_t pend_n = 0, pend_gen = 0;
                bool have_pend = false;

                // ---- CV-FIRST phases 0+1: read every core's control words, decide the ship set. ----
                uint32_t n_ship = 0;
                {
                    const uint64_t t_cv0 = kInstr != 0 ? get_timestamp() : 0;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ, SelfMarkPhase> z_cv(
                            self_mark_phase);
                        for (uint32_t i = 0; i < num_cores; i++) {
                            const uint32_t xy = coords[i];
                            CoreLocalMem<uint32_t> dst(kCvBase + i * kCvReadBytes);
                            noc.async_read<NocOptions::DEFAULT, kCvReadBytes>(
                                src,
                                dst,
                                kCvReadBytes,
                                {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src + kCvReadSrcOff},
                                {});
                            n_cv_rd++;
                        }
                        noc.async_read_barrier();
                    }
                    const uint64_t t_cv1 = kInstr != 0 ? get_timestamp() : 0;
                    c_read += t_cv1 - t_cv0;
                    c_cv += t_cv1 - t_cv0;
                    // Half a lane; dithering it per-core removed the fillers' synchronized ship bursts but
                    // regressed the knee everywhere -- the extra frames' gather reads cost the sweep more
                    // than the burst did (the read side is the saturated-sweep wall).
                    const uint32_t lane_trigger = kLaneTrigger;
                    // ROTATED start, only while the threshold is actually deferring: scan order is service
                    // order, and under deferral a fixed order handed the last-scanned cores every stall.
                    // The rotation steps BACKWARD -- forward makes the previously-first core suddenly
                    // last (a near-2-sweep service interval per step, measured twice); backward, every
                    // core's interval grows by one visit, which every lane's margin absorbs. Wrap by
                    // compare, never %: a runtime modulo is a soft-div on this core.
                    if (ship_deferred != deferred_seen) {
                        deferred_seen = ship_deferred;
                        scan_rot = scan_rot == 0 ? num_cores - 1u : scan_rot - 1u;
                    }
                    const uint64_t t_scan0 = kInstr != 0 ? get_timestamp() : 0;
                    uint32_t c = scan_rot;
                    for (uint32_t k = 0; k < num_cores; k++, (++c >= num_cores ? c = 0 : c)) {
                        const tt_l1_ptr uint32_t* tails =
                            reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                        uint32_t* mine = &head_mirror[c * kNumRisc];
                        if (!seeded[c]) {
                            // Seed from the TAILS: everything before this filler first saw the core
                            // predates the workload.
                            for (uint32_t r = 0; r < kNumRisc; r++) {
                                mine[r] = tails[r];
                            }
                            seeded[c] = 1;
                        }
                        // SCAN, UNROLLED INTO REGISTERS: a loop over indexed arrays spills on this core,
                        // and each spilled word is another L1 round trip per core per sweep.
                        const uint32_t d0 = tails[0] - mine[0];
                        const uint32_t d1 = tails[1] - mine[1];
                        const uint32_t d2 = tails[2] - mine[2];
                        const uint32_t d3 = tails[3] - mine[3];
                        const uint32_t d4 = tails[4] - mine[4];
                        const uint32_t c0 = d0 > kRingWords ? kRingWords : d0;  // overflow is counted at issue
                        const uint32_t c1 = d1 > kRingWords ? kRingWords : d1;
                        const uint32_t c2 = d2 > kRingWords ? kRingWords : d2;
                        const uint32_t c3 = d3 > kRingWords ? kRingWords : d3;
                        const uint32_t c4 = d4 > kRingWords ? kRingWords : d4;
                        const uint32_t live = c0 + c1 + c2 + c3 + c4;
                        uint32_t peak = c0;
                        if (c1 > peak) {
                            peak = c1;
                        }
                        if (c2 > peak) {
                            peak = c2;
                        }
                        if (c3 > peak) {
                            peak = c3;
                        }
                        if (c4 > peak) {
                            peak = c4;
                        }
                        if (peak > max_occ) {
                            max_occ = peak;
                        }
                        if (peak > sweep_peak) {
                            sweep_peak = peak;
                        }
                        if (live == 0) {
                            // A HOT core scanning empty is almost certainly the producer's 64-word batched
                            // tail publish, not idleness; skipping it hands the core a 2-sweep service
                            // interval and the stalls sat on exactly these cores. Ship it -- by issue time
                            // the in-flight tail refresh has usually crossed a boundary. One-shot: a
                            // genuinely idle core wastes at most one empty frame before going cold.
                            if (hot[c] == 0) {
                                continue;
                            }
                            hot[c] = 0;
                            ship_list[n_ship++] = static_cast<uint8_t>(c);
                            continue;
                        }
                        // Level check only -- no growth term, so the producer's batched tail publish
                        // cannot fool it.
                        if (stop_seen_at == 0 && peak < kLaneShipWords && peak < lane_trigger &&
                            ship_age[c] < kShipMaxAgeSweeps) {
                            ship_age[c]++;
                            ship_deferred++;
                            continue;
                        }
                        if (ship_age[c] >= kShipMaxAgeSweeps) {
                            ship_aged++;
                        }
                        ship_age[c] = 0;
                        hot[c] = 1;
                        ship_list[n_ship++] = static_cast<uint8_t>(c);
                    }
                    const uint64_t t_scan1 = kInstr != 0 ? get_timestamp() : 0;
                    c_scan += t_scan1 - t_scan0;
                    c_proc += t_scan1 - t_cv1;
                }

                // Stage one core's frame: write the prefix and control words locally, then GATHER-READ each
                // live run straight to its packed wire offset. The pads bring each destination to its ring
                // phase, so read src == dst (mod 16 B) holds for every piece -- including a wrap split,
                // whose continuation is congruent because the capacity is a multiple of the alignment.
                auto issue_core = [&](uint32_t c, uint32_t sl) {
                    const uint32_t slot = kStageBase + sl * kSlotBytes;
                    const uint32_t xy = coords[c];
                    const tt_l1_ptr uint32_t* tails =
                        reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                    const uint32_t* mine = &head_mirror[c * kNumRisc];
                    volatile tt_l1_ptr uint32_t* cv =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                    uint32_t off = kPrefix + kWireCtrl;
                    ncrisc_noc_read_set_state<DM_DEDICATED_NOC, false, false>(
                        kReadNoc, read_cmd_buf, get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src));
                    // The per-lane walk stays a LOOP, unlike the scans: lane r's bookkeeping hides behind
                    // lane r-1's NIU acceptance. Unrolling front-loaded the bookkeeping against every
                    // issue and measurably regressed.
                    for (uint32_t r = 0; r < kNumRisc; r++) {
                        const uint32_t tail = tails[r];
                        uint32_t run = tail - mine[r];
                        if (run > kRingWords) {
                            overflows++;
                            run = kRingWords;
                        }
                        const uint32_t start = tail - run;
                        // Cap the frame at the slot's payload capacity, WHOLE LANES ONLY: a published
                        // tail is a packet boundary but an arbitrary word count is not -- clamping
                        // mid-run split packets across frames and corrupted the lane stream.
                        uint32_t take = run;
                        uint32_t pad = 0;
                        if (take != 0) {
                            pad = kernel_profiler::spsc_span_pack_pad(start, off);
                            const uint32_t used = off - (kPrefix + kWireCtrl);
                            const uint32_t room =
                                kPayloadCapWords > used + pad ? kPayloadCapWords - used - pad : 0;
                            if (take > room) {
                                take = 0;
                                pad = 0;
                            }
                        }
                        slot_runs[sl * kNumRisc + r] = take;
                        cv[kernel_profiler::SPSC_WIRE_HEAD_0 + r] = start;
                        cv[kernel_profiler::SPSC_WIRE_TAIL_0 + r] = start + take;
                        if (take == 0) {
                            continue;
                        }
                        off += pad;
                        const uint32_t hm = start & (kRingWords - 1u);
                        const uint32_t ring_src = cv_src + (kCtrlWords + r * kRingWords) * 4u;
                        const uint32_t chunk = take <= kRingWords - hm ? take : kRingWords - hm;
                        n_gather_rd++;
                        ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                            kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, chunk * 4u);
                        if (chunk < take) {
                            n_gather_rd++;
                            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                                kReadNoc, read_cmd_buf, ring_src, slot + (off + chunk) * 4u, (take - chunk) * 4u);
                        }
                        off += take;
                    }
                    cv[kernel_profiler::SPSC_WIRE_XY] = xy;
                    volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
                    pfx[0] = kernel_profiler::spsc_span_w0();
                    pfx[1] = off - kPrefix;
                    slot_payload[sl] = off - kPrefix;
                    slot_core[sl] = static_cast<uint8_t>(c);
                };

                // Heads go out the moment the batch's read barrier passes, NOT with the frame emit: the
                // payload is resident in staging once the reads land, so those ring slots are free
                // regardless of when the frame reaches the host.
                auto advance_heads = [&](uint32_t n, uint32_t g) {
                    const uint64_t t_p0 = kInstr != 0 ? get_timestamp() : 0;
                    kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PROC, SelfMarkPhase> z_proc(
                        self_mark_phase);
                    for (uint32_t i = 0; i < n; i++) {
                        const uint32_t sl = g * kGenSlots + i;
                        const uint32_t c = slot_core[sl];
                        uint32_t* mine = &head_mirror[c * kNumRisc];
                        uint64_t t_h0 = 0;
                        if constexpr (kSvcInstr != 0) {
                            t_h0 = get_timestamp();
                            if (last_ship[c] != 0) {
                                const uint32_t dt = static_cast<uint32_t>(t_h0) - last_ship[c];
                                if (dt > svc_max) {
                                    svc_max = dt;
                                }
                                uint32_t b = 0;
                                for (uint32_t q = dt >> 13; q != 0 && b < 7u; q >>= 1) {
                                    b++;
                                }
                                svc_hist[b]++;
                            }
                            last_ship[c] = static_cast<uint32_t>(t_h0);
                        }
                        const uint32_t sc = kHeadScratch + hb_slot * 32u;
                        volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                        const uint32_t* runs = &slot_runs[sl * kNumRisc];
                        uint32_t live = 0;
                        for (uint32_t r = 0; r < kNumRisc; r++) {
                            const uint32_t m = mine[r] + runs[r];
                            mine[r] = m;
                            scp[r] = m;
                            live += runs[r];
                        }
                        // POSTED (the barriers protect staging reuse, which a head write never touches;
                        // scratch reuse stays safe on the slot rotation) and on the READ NoC: on the
                        // egress NoC this 32 B packet queues behind the previous batch's frames, so head
                        // visibility inherited the PCIe tile's acceptance jitter -- the term that binds
                        // at the knee.
                        noc_async_write_one_packet<true, true>(
                            sc,
                            get_noc_addr(
                                coords[c] & 0xFFFFu, coords[c] >> 16, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u),
                            kNumRisc * 4u,
                            kReadNoc);
                        hb_slot = (hb_slot + 1u) & (kMaxCores - 1u);
                        if constexpr (kSvcInstr != 0) {
                            c_ph_head += get_timestamp() - t_h0;
                        }

                        frames++;
                        total_words += live;
                    }
                    c_proc += kInstr != 0 ? (get_timestamp() - t_p0) : 0;
                };

                auto ship_frames = [&](uint32_t n, uint32_t g) {
                    // A LISTED CORE is live by construction, so the batch having cores IS the work signal.
                    if constexpr (kSelfZones != 0) {
                        if (!self_on && n != 0) {
                            self_arm();  // opens the window at EVERY detail level
                        }
                    }
                    emit_slots(g * kGenSlots, n);
                    if constexpr (kSpool) {
                        gen_dma_mark[g] = dma_issued;
                    }
                    if (!egress_dead) {
                        gen_shipped[g] = true;
                    }
                };

                uint32_t cur = 0;
                while (cur < n_ship) {

                    // This generation's previous ship must be out of staging before its slots refill (SENT
                    // suffices: the next writer is this core's own NIU read responses). gen_shipped
                    // persists across sweeps, so a sweep's LAST ship is never waited on inside its own
                    // sweep -- this is the wait that catches it if the gap has not drained it.
                    if (gen_shipped[gen]) {
                        const uint64_t t_b0 = get_timestamp();
                        *phase = kPhBar1;
                        bool flushed;
                        {
                            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfMarkPhase> z_bar(
                                self_mark_phase);
                            if constexpr (kSpool) {
                                // Completion of THIS generation's ship writes; anything issued since may
                                // stay in flight (stream completion is FIFO, so outstanding <= later-issues
                                // means this generation retired).
                                const uint32_t since = dma_issued - gen_dma_mark[gen];
                                flushed = dma_wait_writes_bounded(
                                    kDmaShip, since > 15u ? 15u : static_cast<uint8_t>(since), t_b0 + kCreditWaitCycles);
                            } else {
                                flushed = write_barrier_bounded<true>(t_b0 + kCreditWaitCycles);
                            }
                        }
                        c_barrier += kInstr != 0 ? (get_timestamp() - t_b0) : 0;
                        if (!flushed) {
                            egress_dead = true;
                            break;
                        }
                        gen_shipped[gen] = false;
                    }

                    // c_read is two disjoint intervals per batch: the issue, and whatever wait survives
                    // the concurrent ship.
                    const uint64_t t_batch0 = kInstr != 0 ? get_timestamp() : 0;
                    uint32_t n = 0;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ, SelfMarkPhase> z_issue(
                            self_mark_phase);
                        uint32_t slots = 0;
                        while (slots < kGenSlots && cur < n_ship) {
                            issue_core(ship_list[cur], gen * kGenSlots + slots);
                            cur++;
                            n++;
                            slots++;
                        }
                        // Refresh the NEXT batch's tails in the same flight: on the sweep-start snapshot
                        // alone the last batch's cores are consumed ~a whole sweep stale, and the
                        // scan-order-last core took all the stalls. This generation's read barrier covers
                        // these too, so the next issue_core sees landed tails.
                        const uint32_t nn = (n_ship - cur) < kGenSlots ? (n_ship - cur) : kGenSlots;
                        for (uint32_t i = 0; i < nn; i++) {
                            const uint32_t c = ship_list[cur + i];
                            const uint32_t xy = coords[c];
                            CoreLocalMem<uint32_t> dst(kCvBase + c * kCvReadBytes);
                            noc.async_read<NocOptions::DEFAULT, kCvReadBytes>(
                                src,
                                dst,
                                kCvReadBytes,
                                {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src + kCvReadSrcOff},
                                {});
                            n_cv_rd++;
                        }
                    }
                    const uint64_t t_issue = kInstr != 0 ? get_timestamp() : 0;

                    // The overlap: the previous batch's PCIe writes go out on NOC_INDEX while the gather reads
                    // above fly on kReadNoc.
                    if (have_pend) {
                        ship_frames(pend_n, pend_gen);
                    }
                    // UNDER SPOOL PRESSURE, also pump on the sweep itself: the read-shadow passes below
                    // vanish exactly when a sustained load needs them (a loaded steady-state sweep barely
                    // waits on its read barrier), and this ~2.4 us is what direct push paid inline. WITH
                    // HYSTERESIS -- a single threshold duty-cycles around itself and the off-phases bled
                    // occupancy back to the cap. Engage 5/8 (the loaded steady state settles ~62%, so 3/4
                    // engaged too late); release near empty (a 1/4 release left drain-less windows in the
                    // sustained oscillation). A 10k burst peaks ~42% and never enters the band.
                    if constexpr (kSpool) {
                        const uint32_t occ = static_cast<uint32_t>(spool_wr - spool_rd);
                        if (occ > kSpoolBytes / 2u + kSpoolBytes / 8u) {
                            pump_pressure = true;
                        } else if (occ < kSpoolBytes / 16u) {
                            pump_pressure = false;
                        }
                        if (pump_pressure) {
                            drain_pump();
                        }
                    }

                    // Issue cost plus only the wait REMAINING after the concurrent ship (timing to the
                    // barrier instead double-counted the frame emit; phases summed 133%).
                    const uint64_t t_after_proc = kInstr != 0 ? get_timestamp() : 0;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ_WAIT, SelfMarkPhase> z_wait(
                            self_mark_phase);
                        // The barrier spin is the pump's slot -- cycles the core burns anyway. UNDER
                        // PRESSURE ONLY, even here: at a burst the pump's GDDR reads and bounce-L1 writes
                        // contend with the ship DMA and the landing gather responses (d8 paid 3.5k stalls
                        // for drain the spool did not need). Same predicate + trailing invalidate as
                        // noc.async_read_barrier().
                        while (!ncrisc_noc_reads_flushed(kReadNoc)) {
                            if constexpr (kSpool) {
                                if (pump_pressure && spool_wr != spool_rd) {
                                    drain_pump();
                                }
                            }
                        }
                        invalidate_l1_cache();
                    }
                    const uint64_t t_read_end = kInstr != 0 ? get_timestamp() : 0;
                    advance_heads(n, gen);
                    c_issue += t_issue - t_batch0;
                    c_read += (t_issue - t_batch0) + (t_read_end - t_after_proc);

                    pend_n = n;
                    pend_gen = gen;
                    have_pend = true;
                    gen = gen + 1u == kNGens ? 0u : gen + 1u;
                }
                if (have_pend) {
                    ship_frames(pend_n, pend_gen);
                    have_pend = false;
                }
            }

            if constexpr (kInstr != 0 || kSelfZones != 0) {
                sweep_cyc = static_cast<uint32_t>(get_timestamp() - t_sweep0);
            }
        }
        if (sweep_cyc > max_sweep) {
            max_sweep = sweep_cyc;
            ws_read = static_cast<uint32_t>(c_read - s_read0);
            ws_proc = static_cast<uint32_t>(c_proc - s_proc0);
            ws_rsv = static_cast<uint32_t>(c_reserve - s_rsv0);
            ws_wr = static_cast<uint32_t>(c_write - s_wr0);
            ws_bar = static_cast<uint32_t>(c_barrier - s_bar0);
        }
        const bool win2_work = frames != frames_at_sweep_start;
        if (win2_work && !win2_open) {
            win2_open = true;  // snapshot BEFORE this sweep lands in c_busy
            w0_t = t_sweep0;
            w0_busy = c_busy;
            w0_idle = c_idle;
            w0_pace = c_pace;
            w0_frames = frames_at_sweep_start;
            w0_sweeps = sweeps;
            if constexpr (kSelfZones == 0) {
                w0_cv = c_cv;
                w0_issue = c_issue;
            }
        }
        if (frames == frames_at_sweep_start) {
            sweeps_idle++;
            c_idle += sweep_cyc;
        } else {
            c_busy += sweep_cyc;
        }
        nf_end(t_sweep0, sweep_cyc, frames != frames_at_sweep_start);
        if (frames != frames_at_sweep_start) {
            const uint32_t b = sweep_peak / (kRingWords >> 3u);  // constexpr divisor: folds to a shift
            fill_hist[b > 7u ? 7u : b]++;
        }
        // After sweep_cyc is captured, so drain time is never billed to the sweep it trails.
        if constexpr (kSpool) {
            const uint64_t t_d0 = kInstr != 0 ? get_timestamp() : 0;
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_DRAIN, SelfMarkNow> z_drain(self_mark_now);
            drain_pump();
            drain_notify();
            c_drain += kInstr != 0 ? (get_timestamp() - t_d0) : 0;
        }
        // No per-sweep publish: mid-window frames are published by self_mark_w0 when the ring fills.
        if constexpr (kSelfZones != 0) {
            const bool busy = frames != frames_at_sweep_start;
            if (busy) {
                self_armed_until = t_sweep0 + sweep_cyc + kSelfHoldCycles;
            }
            if (self_on) {
                self_sweeps++;
                if (busy || self_work) {
                    self_sweeps_work++;
                }
                if (self_from_start) {
                    self_ck_sweeps++;
                    self_ck_read += c_read - s_read0;
                    self_ck_proc += c_proc - s_proc0;
                    self_ck_rsv += c_reserve - s_rsv0;
                    self_ck_write += c_write - s_wr0;
                    self_ck_bar += c_barrier - s_bar0;
                }
            }
        }

        // Collapse the gap on work, creep toward kCvIdleGapMax when idle: widening only saves idle probe
        // traffic, and a producer must never wait on it. Live-but-untriggered counts as work here (see
        // kCvBusyPeak).
        if (frames != frames_at_sweep_start || sweep_peak >= kCvBusyPeak) {
            gap = 0;
        } else {
            uint32_t inc = gap >> 1;
            if (inc < 256u) {
                inc = 256u;
            }
            gap = (gap + inc > kCvIdleGapMax) ? kCvIdleGapMax : gap + inc;
        }
        // THE PACING GAP as its own depth-0 zone, a SIBLING of the SWEEP scope that closed above, never its
        // child.
        if (gap != 0) {
            const uint64_t t_g0 = get_timestamp();
            {
                kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PACE, SelfMarkNow> z_pace(self_mark_now);
                const uint64_t until = t_g0 + gap;
                while (get_timestamp() < until) {
                    if constexpr (kSpool) {
                        drain_pump();  // idle time is drain time; billed to pace, as the spin was
                    }
                }
            }
            c_pace += get_timestamp() - t_g0;
        }
        if (win2_work) {
            w1_t = get_timestamp();
            w1_busy = c_busy;
            w1_idle = c_idle;
            w1_pace = c_pace;
            w1_frames = frames;
            w1_sweeps = sweeps;
            if constexpr (kSelfZones == 0) {
                w1_cv = c_cv;
                w1_issue = c_issue;
            }
        }
        // The window's LAST sweep must flush, or its zones sit in the ring until the next window, or
        // forever. After the gap, so PACE rides in the same frame as the SWEEP it follows.
        if constexpr (kSelfZones != 0) {
            if (self_on) {
                if (get_timestamp() >= self_armed_until) {
                    self_publish();
                }
                self_on = false;
            }
        }
    }

    // FLUSH WHAT IS STILL IN THE RING. Gated on `self_tail != self_head`, NOT on self_on -- self_on is
    // cleared at the end of every sweep, so a check on it can never fire here.
    if constexpr (kSelfZones != 0) {
        self_on = false;  // no more markers; self_publish() must not think it is inside a traced sweep
        if (!egress_dead) {
            self_publish();
        }
    }

    // Everything the run spooled must reach the host FIFO before the socket barrier can pass; bounded,
    // so a consumer that stopped acking strands bytes (counted below) instead of wedging the teardown.
    if constexpr (kSpool) {
        if (!egress_dead) {
            (void)dma_wait_writes_bounded(kDmaShip, 0, get_timestamp() + kCreditWaitCycles);
            const uint64_t t_dl = get_timestamp() + kStopDrainCycles;
            while (spool_rd_iss != spool_wr || b_state[0] != kBounceEmpty || b_state[1] != kBounceEmpty) {
                drain_pump();
                // Per pass, not per sweep: with a host FIFO smaller than the backlog, the acks that free
                // credit only come after the host has SEEN the bytes.
                drain_notify();
                if (get_timestamp() >= t_dl) {
                    drain_dead = true;
                    break;
                }
            }
            drain_notify();
        }
    }

    // socket_barrier() waits for the host to ack everything, so it hangs on a dead consumer just
    // like the write barrier did. Skip both when we already know the consumer is gone.
    const bool consumer_gone = egress_dead || credit_timeouts != 0 || drain_dead || spool_lossy;
    *phase = kPhSockBar;
    if (!consumer_gone) {
        socket_barrier(sender);
    }
    *phase = kPhTailBar;
    (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles);
    // The posted head write-backs are outside that barrier's predicate; drain their SENT counter (20 B
    // packets stream out in ns) so no scratch slot or unstreamed head is left behind at report time.
    {
        const uint64_t t_ps = get_timestamp() + 1350000u;
        while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc)) &&
               get_timestamp() < t_ps) {
        }
    }
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    // ================================ INSTRUMENTATION START (results block: every counter the host reads) ====
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = static_cast<uint32_t>(total_words & 0xFFFFFFFFu);
    out[3] = static_cast<uint32_t>(total_words >> 32);
    out[4] = sweeps;
    out[5] = pages;
    out[6] = frames;
    out[7] = max_occ;
    out[8] = overflows;
    out[9] = pushes;
    out[10] = static_cast<uint32_t>(c_read & 0xFFFFFFFFu);
    out[11] = static_cast<uint32_t>(c_read >> 32);
    out[12] = static_cast<uint32_t>(c_proc & 0xFFFFFFFFu);
    out[13] = static_cast<uint32_t>(c_proc >> 32);
    out[14] = static_cast<uint32_t>(c_reserve & 0xFFFFFFFFu);
    out[15] = static_cast<uint32_t>(c_reserve >> 32);
    out[16] = static_cast<uint32_t>(c_write & 0xFFFFFFFFu);
    out[17] = static_cast<uint32_t>(c_write >> 32);
    out[18] = static_cast<uint32_t>(c_barrier & 0xFFFFFFFFu);
    out[19] = static_cast<uint32_t>(c_barrier >> 32);
    out[20] = sweeps_idle;
    out[21] = static_cast<uint32_t>(c_idle & 0xFFFFFFFFu);
    out[22] = static_cast<uint32_t>(c_idle >> 32);
    out[23] = static_cast<uint32_t>(c_busy & 0xFFFFFFFFu);
    out[24] = static_cast<uint32_t>(c_busy >> 32);
    out[25] = max_sweep;
    out[26] = max_reserve;
    out[27] = static_cast<uint32_t>(c_wr_chunk & 0xFFFFFFFFu);
    out[28] = static_cast<uint32_t>(c_wr_chunk >> 32);
    out[29] = static_cast<uint32_t>(c_wr_push & 0xFFFFFFFFu);
    out[30] = static_cast<uint32_t>(c_wr_push >> 32);
    out[31] = static_cast<uint32_t>(c_wr_notify & 0xFFFFFFFFu);
    out[32] = static_cast<uint32_t>(c_wr_notify >> 32);
    out[33] = credit_timeouts;
    out[34] = dropped_frames;
    out[35] = egress_dead ? 1u : 0u;
    // ---- GDDR spool / drain pump (all zero in direct-push builds) ----
    out[48] = spool_drops;
    out[49] = drain_chunks;
    out[50] = drain_starved;
    out[51] = spool_max;
    out[52] = drain_ships;
    out[135] = drain_dead ? 1u : 0u;
    out[138] = static_cast<uint32_t>(c_drain & 0xFFFFFFFFu);
    out[139] = static_cast<uint32_t>(c_drain >> 32);
    out[140] = static_cast<uint32_t>(spool_wr & 0xFFFFFFFFu);
    out[141] = static_cast<uint32_t>(spool_wr >> 32);
    out[142] = static_cast<uint32_t>(spool_wr - spool_rd);  // stranded bytes; nonzero only with drain_dead
    out[144] = spool_lossy ? 1u : 0u;
    out[36] = noc_index_before;
    out[37] = noc_index_after;
    out[38] = NOC_INDEX;
    out[39] = kReadNoc;
    out[43] = ws_read;
    out[44] = ws_proc;
    out[45] = ws_rsv;
    out[46] = ws_wr;
    out[47] = ws_bar;
    out[42] = gap;  // where the pacing controller settled
    out[40] = static_cast<uint32_t>(c_ph_head & 0xFFFFFFFFu);
    out[41] = static_cast<uint32_t>(c_ph_head >> 32);
    // out[48..52] retired (HIGH-mode telemetry, then the DRAM-ring role split before it).
    for (uint32_t k = 0; k < 8u; k++) {
        out[53 + k] = fill_hist[k];
    }
    out[61] = static_cast<uint32_t>(c_scan & 0xFFFFFFFFu);
    out[62] = static_cast<uint32_t>(c_scan >> 32);
    out[63] = sweeps * num_cores;  // decide-loop core visits: every sweep scans them all
    // ---- DRISC SELF-PROFILING counters ----
    out[64] = self_frames;       // self frames shipped
    out[65] = self_markers;      // markers written into the self ring
    out[66] = self_sweeps;       // sweeps whose zones were shipped
    out[67] = self_sweeps_work;  // of those, the ones that actually did work -- the ones worth having
    out[68] = self_windows;      // distinct active windows covered
    out[69] = self_over;         // sweeps left uninstrumented because the frame budget was spent
    out[70] = self_dropped;      // markers lost because a publish could not free the ring (should be 0)
    out[71] = static_cast<uint32_t>(c_self & 0xFFFFFFFFu);
    out[72] = static_cast<uint32_t>(c_self >> 32);
    out[73] = self_tail;  // total words ever appended (monotonic)
    // Phase totals over the sweeps the zones cover, so a host summing zone durations can assert an
    // equality rather than eyeball a plausible shape.
    out[74] = self_ck_sweeps;
    out[75] = static_cast<uint32_t>(self_ck_read & 0xFFFFFFFFu);
    out[76] = static_cast<uint32_t>(self_ck_read >> 32);
    out[77] = static_cast<uint32_t>(self_ck_proc & 0xFFFFFFFFu);
    out[78] = static_cast<uint32_t>(self_ck_proc >> 32);
    out[79] = static_cast<uint32_t>(self_ck_rsv & 0xFFFFFFFFu);
    out[80] = static_cast<uint32_t>(self_ck_rsv >> 32);
    out[81] = static_cast<uint32_t>(self_ck_write & 0xFFFFFFFFu);
    out[82] = static_cast<uint32_t>(self_ck_write >> 32);
    out[83] = static_cast<uint32_t>(self_ck_bar & 0xFFFFFFFFu);
    out[84] = static_cast<uint32_t>(self_ck_bar >> 32);
    out[85] = kSelfMaxFrames;  // the budget, so the host reports utilisation without re-deriving it
    out[86] = kSelfDetail;     // 0 = SWEEP+PACE only, 1 = full per-batch phases
    out[87] = self_words_shipped;  // MUST equal out[73]: less means trace lost in the ring at teardown
    // sync_timeouts MUST be 0: a timed-out drainer never contributed to the fiducial, voiding the trigger.
    out[130] = sync_events;
    out[131] = sync_timeouts;
    out[132] = sync_spin_cyc;
    out[133] = stop_sweeps;
    out[134] = static_cast<uint32_t>(total_words - words_at_stop);
    out[136] = static_cast<uint32_t>(c_pace & 0xFFFFFFFFu);
    out[137] = static_cast<uint32_t>(c_pace >> 32);
    out[170] = ship_deferred;
    out[171] = ship_aged;
    {
        const uint64_t wc = win2_open ? w1_t - w0_t : 0u;
        out[181] = static_cast<uint32_t>(wc & 0xFFFFFFFFu);
        out[182] = static_cast<uint32_t>(wc >> 32);
        const uint64_t wb = win2_open ? w1_busy - w0_busy : 0u;
        out[183] = static_cast<uint32_t>(wb & 0xFFFFFFFFu);
        out[184] = static_cast<uint32_t>(wb >> 32);
        const uint64_t wi = win2_open ? w1_idle - w0_idle : 0u;
        out[185] = static_cast<uint32_t>(wi & 0xFFFFFFFFu);
        out[186] = static_cast<uint32_t>(wi >> 32);
        const uint64_t wp = win2_open ? w1_pace - w0_pace : 0u;
        out[187] = static_cast<uint32_t>(wp & 0xFFFFFFFFu);
        out[188] = static_cast<uint32_t>(wp >> 32);
        out[189] = win2_open ? w1_frames - w0_frames : 0u;
        out[190] = win2_open ? w1_sweeps - w0_sweeps : 0u;
        out[191] = win2_open ? 1u : 0u;
        out[192] = num_cores;
    }
    // The read-split window (out[202..205]) and the self-zone build trade the same code-region bytes;
    // zones supersede it as the diagnostic when both are requested.
    if constexpr (kSelfZones == 0) {
        const uint64_t wcv = win2_open ? w1_cv - w0_cv : 0u;
        out[202] = static_cast<uint32_t>(wcv & 0xFFFFFFFFu);
        out[203] = static_cast<uint32_t>(wcv >> 32);
        const uint64_t wis = win2_open ? w1_issue - w0_issue : 0u;
        out[204] = static_cast<uint32_t>(wis & 0xFFFFFFFFu);
        out[205] = static_cast<uint32_t>(wis >> 32);
    }
    out[193] = svc_max;
    for (uint32_t i = 0; i < 8; i++) {
        out[194 + i] = svc_hist[i];
    }
    out[172] = static_cast<uint32_t>(c_issue & 0xFFFFFFFFu);
    out[173] = static_cast<uint32_t>(c_issue >> 32);
    out[174] = n_gather_rd;
    out[175] = n_cv_rd;
    out[176] = static_cast<uint32_t>(c_cv & 0xFFFFFFFFu);
    out[177] = static_cast<uint32_t>(c_cv >> 32);
    // ---- NoC FOOTPRINT counters ----
    // TWO BLOCKS, NEVER BLENDED: `life` covers every sweep, `win` the workload window only.
    {
        // Final sample, so the lifetime block includes the last sweep, the exit drain wait and the barrier.
        if constexpr (kNocFootprint != 0) {
            nf_sample_regs(&nf);
        }
        // out[88..103]: life[noc][rd_words, rd_txns, wr_words, wr_txns], 64-bit lo/hi.
        // out[104..119]: the window delta, same order. One loop over a two-entry table, for code size.
        uint32_t o = 88;
        for (uint32_t i = 0; i < kNfSlots; i++) {
            out[o++] = static_cast<uint32_t>(nf.life[i] & 0xFFFFFFFFu);
            out[o++] = static_cast<uint32_t>(nf.life[i] >> 32);
        }
        for (uint32_t i = 0; i < kNfSlots; i++) {
            const uint64_t v = nf.win_last[i] - nf.win_base[i];
            out[o++] = static_cast<uint32_t>(v & 0xFFFFFFFFu);
            out[o++] = static_cast<uint32_t>(v >> 32);
        }
        // out[104..119]: win[noc][...] in the same order
        out[120] = nf.win_open ? (nf.win_sweep_last - nf.win_sweep_first + 1u) : 0u;
        const uint64_t win_cyc = nf.win_open ? (nf.win_t1 - nf.win_t0) : 0u;
        out[121] = static_cast<uint32_t>(win_cyc & 0xFFFFFFFFu);
        out[122] = static_cast<uint32_t>(win_cyc >> 32);
        // The instrument's own cost (8 NIU loads + 2 clock reads per sweep), so the host can report it.
        out[123] = static_cast<uint32_t>(nf.cost & 0xFFFFFFFFu);
        out[124] = static_cast<uint32_t>(nf.cost >> 32);
        // Retired posted-blind-spot check; zeroed to keep the layout.
        out[125] = 0;
        out[126] = 0;
        out[127] = nf.win_open ? 1u : 0u;  // did the window ever open? (0 = no sweep did work)
        out[128] = kNocFootprint;          // echo, so the host never guesses whether this block is valid
        out[129] = NOC_WORD_BYTES;         // the byte scale, from the header -- host never hardcodes it
    }
    static_assert(
        kernel_profiler::SPSC_DRAIN_RESULT_WORDS >= 202,
        "the results block must hold the self-profiling, NoC-footprint, stop-drain and histogram counters");

    // ================================ INSTRUMENTATION END (results block: every counter the host reads) ====
    *phase = kPhaseExit;
    // Written back only for a live consumer: after dropped frames the socket's view of bytes_sent is
    // already out of sync with the host's, and the socket is being torn down either way.
    if (!consumer_gone) {
        update_socket_config(sender);
    }

    // Published last, after the socket barrier, so the host only sees `done` once every page is out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = 0xD09E0000u | (frames & 0xFFFFu);

    // NIU restore, on the host's word: NIU_CFG_0 persists until chip reset, so whoever set stream mode
    // owns putting it back -- and LAST, because the flip to NOC2AXI takes this L1 (`done`, the results,
    // bytes_acked) out of the host's view.
    for (uint32_t spins = 0; spins < 200000000u && *stop != 2u; spins++) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
