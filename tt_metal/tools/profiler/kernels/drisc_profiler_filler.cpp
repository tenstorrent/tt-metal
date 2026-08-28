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
    // Read early: the slot geometry depends on it. (Full doc at the self-profiling block below.)
    constexpr uint32_t kSelfZonesEarly = get_compile_time_arg_val(32);
    // CAPPED slots, half the full-span worst case: a frame's consume clamps at the payload cap and the
    // residue ships next sweep, so a slot never needs to hold a whole core's backlog -- and halving the
    // slot doubles the staging generations the same L1 holds, which is what keeps egress acceptance
    // jitter off the service loop (measured: knee floor ~8 with egress ablated vs 13 coupled, and
    // three generations recovered ~2 points of that). Self-zone builds keep the full-span slot: the
    // self frame ships the RAW span layout and cannot be split.
    // Full-span slots on purpose. Capping the slot to deepen the staging ring (arg 42, 2026-08-28) was
    // deleted after its measured wins proved fake: any cap below a core's worst case makes the consume
    // defer whole lanes, and at speed first-fit deferral STARVED lane 4 outright -- lanes 0-3 produce
    // ~cap words per sweep, so TRISC2's producer on every core blocked ~30 us in and stayed blocked all
    // run. The "knee" numbers of the capped configs were that 20% load shed. With all lanes honestly
    // flowing, every capped variant (1,262/1,598/1,982 payload words; fullest-first and scan-side defer
    // selection) lost to full slots at every delay measured.
    constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);
    constexpr uint32_t kSlotBytes = kSlotWords * 4u;
    constexpr uint32_t kWireCtrl = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
    constexpr uint32_t kPayloadCapWords = kSlotWords - kPrefix - kWireCtrl;
       // 10,560
    constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
    constexpr uint32_t kPageBytes = kPageWords * 4u;
    // Reads take the NoC the writes do not; NOC_INDEX (the kernel's configured NoC) carries egress.
    constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
    // Two-core batches; every slot the arena holds beyond the CV slot becomes pipeline depth.
    constexpr uint32_t kGenSlots = 2;
    constexpr uint32_t kNGens = (kNStage - 1) / kGenSlots;
    static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
    // The static VC this filler's PCIe pushes ride (0 or 1, the two unicast request VCs). Spread across
    // the fillers by the host: per-hop NoC arbitration is per-VC, so six pushers on one VC gave the far
    // cores a geometrically starved share of the PCIe tile while near ones stayed fast.
    constexpr uint32_t kWriteVc = get_compile_time_arg_val(20);
    // Args 21..31 retired (RAW_ONLY, then the DRAM-ring role split: ring geometry and the mover handshake).
    // ---- DRISC SELF-PROFILING (0 = off, every use behind `if constexpr`) ----
    // The drainer's own zones, framed exactly like a worker span and shipped down the path it already
    // owns: no side channel, no second wire format, host decoder untouched. Only ring 0 is live.
    constexpr uint32_t kSelfZones = get_compile_time_arg_val(32);
    constexpr uint32_t kSelfHoldCycles = get_compile_time_arg_val(33);
    constexpr uint32_t kSelfXY = get_compile_time_arg_val(34);  // this DRISC's own virtual (y<<16)|x
    constexpr uint32_t kSelfMaxFrames = get_compile_time_arg_val(35);
    // Detail 0 = SWEEP + PACE only; 1 = also the per-batch child phases, ~25x the volume.
    constexpr uint32_t kSelfDetail = get_compile_time_arg_val(36);
    // ---- NoC FOOTPRINT: the drainer's OWN NIU MASTER counters. NOC_STATUS_READ_REG is a local MMIO
    // load and issues no NoC transaction, so the instrument cannot perturb what it measures. ----
    constexpr uint32_t kNocFootprint = get_compile_time_arg_val(37);
    // COMMON-TRIGGER SYNC EVENT. The host parks every drainer in a TIGHT SPIN and one release makes
    // them all stamp the same instant; a per-sweep poll would report sweep phase, not the trigger.
    constexpr uint32_t kSyncEvent = get_compile_time_arg_val(38);
    // Per-core service-interval instrumentation: two wall-clock reads + a histogram update per shipped
    // core, ~0.5-1 us of a knee sweep. The svc lines it feeds are the diagnostic that found the rotation
    // and staleness mechanisms; opt-in via TT_METAL_PERF_DEBUG_DRISC_SVC when hunting the next one.
    constexpr uint32_t kSvcInstr = get_compile_time_arg_val(40);
    // Master gate on the BASE instrumentation tier: the phase cycle counters and their ~55 wall-clock
    // reads per sweep (~1 us of a 15 us knee sweep). 0 compiles them out for record runs -- the
    // LIFETIME/WINDOW/WORST/read-split report lines then print zeros. Functional clock reads (credit
    // and barrier deadlines, the pace gap, stop-path timing) are NOT behind this.
    constexpr uint32_t kInstr = get_compile_time_arg_val(41);
    // PER-CORE SHIP THRESHOLD (0 = ship every live core every sweep). A frame costs the pipe the same
    // whether it carries 200 live words or 2,000, so a core ships only when it is worth the frame:
    // enough live words, any lane past kLaneShipWords, or the age bound below.
    constexpr uint32_t kShipMinPct = get_compile_time_arg_val(39);
    constexpr uint32_t kShipMaxAgeSweeps = 512u;
    constexpr uint32_t kShipMinWords = (kLiveWords * kShipMinPct) / 100u;
    // CV-FIRST SWEEPS: read each core's ring TAILS (32 B), decide the ship set, then GATHER-READ only the
    // ship set's live runs -- each straight to its packed wire offset in staging, so a staged frame is its
    // own wire image and ships in ONE PCIe write (two across the FIFO wrap). The tails read is
    // authoritative for the frames it parents: a frame claims exactly [mirror, tail-at-this-read), and a
    // producer only ever appends PAST a published tail (fenced after its data), so the gather can trail
    // the tail read by a whole batch without tearing. The frame's control words are synthesized from the
    // mirror, these tails and coords[] -- heads, tails and SPSC_CORE_XY are all the decoder reads.
    constexpr uint32_t kCvReadBytes = 32;
    constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
    // Idle backoff ceiling (~20 us): collapse on work, creep when idle.
    // 5 us. Was 20 us, which exceeds the time a lane takes to fill its ring at high production rates, so
    // the ramp could blind the filler for most of a fill window (measured: workload 806 -> 70 ms at delay
    // 45). The filler is busy ~22% of a workload, so the probe traffic a wider gap saved was never
    // contended for.
    constexpr uint32_t kCvIdleGapMax = 6750;
    // Per-lane ship trigger, and the point past which the idle gap must stop growing. "Shipped no frame"
    // is a statement about the TRIGGER, not about the producers, so backing off on it alone puts the filler
    // to sleep exactly while lanes fill toward it -- and a head only reaches a producer on a ship, so a
    // late ship is a late head. Both derive from one constant so the two cannot drift apart.
    // STATE-REUSE READS. Every lane read of a core targets the SAME worker, so the coordinate registers
    // (NOC_TARG_ADDR_MID, NOC_TARG_ADDR_COORDINATE) are identical across all of them. set_async_read_state
    // programs those once per core and async_read_with_state then issues with four register writes instead
    // of six. max_page_size must stay above NOC_MAX_BURST_SIZE: below it the length is programmed in
    // set_state and every read would have to be the same size, which per-lane runs are not.
    //
    // Not obviously a win -- it trades ~2 register writes per read for one extra cmd-buf-ready poll per
    // core -- so it is a switch, not a rewrite. Gather issue is ~9.6 us of a ~13 us sweep at 103 ns/read.
    constexpr bool kReadState = true;
    constexpr uint32_t kStateMaxPage = NOC_MAX_BURST_SIZE + 1u;
    constexpr uint32_t kLaneTrigger = kRingWords / 2u;
    constexpr uint32_t kCvBusyPeak = kLaneTrigger / 2u;
    // There is deliberately NO whole-span (raw) ship mode for worker cores. Direct-push egress shares ONE
    // PCIe tile across all six fillers, so a raw sweep's dead ring bytes (kSpanWords against ~10% live at
    // the loads where such a mode would engage) multiply pressure exactly where the pipeline is narrowest.
    // Measured at delay 35, SHIP_MIN_PCT=1, host out of the loop: the peak-lane-hysteresis HIGH mode this
    // replaced engaged on all fillers, the worst one pinned at ring capacity (write 20.7 us of a 38 us
    // sweep) and took 60k producer stalls; with it unreachable the same load ran 0-stall. Only the DRISC's
    // own self frame still ships the raw span layout -- its dead bytes stage no reads and ship rarely.
    static_assert(kShipMinPct != 0, "CV-first sweeps exist to feed the per-core ship decision");
    static_assert(
        kSelfZones != 0 || kNGens * kGenSlots < kNStage,
        "CV staging needs a slot past the 2-generation pipeline (kNStage must be odd when self-zones are off)");
    // The control snapshots land in the ring-1 area of the slot past the pipeline. With self-zones ON that
    // slot holds the self FRAME, and this placement is still safe: only the self frame's ring 0 is ever
    // live, so its ring 1..4 storage (8 KiB) carries no markers -- the raw self frame ships these bytes,
    // but the host walks a lane only between its head and tail, so they are never decoded. That shared
    // dead space is what lets CV-first and drainer self-profiling coexist.
    constexpr uint32_t kCvSlot = kSelfZones != 0 ? kNStage : kNGens * kGenSlots;
    // Under self-zones the CV staging shares the self slot and must sit past its live ring 0; otherwise
    // it owns the whole slot past the pipeline and starts at the base.
    constexpr uint32_t kCvBase = kStageBase + kCvSlot * kSlotBytes +
                                 (kSelfZones != 0 ? (kPrefix + kCtrlWords + kRingWords) * 4u : 0u);
    static_assert(
        (kSelfZones != 0 ? (kPrefix + kCtrlWords + kRingWords) * 4u : 0u) + kCvReadBytes * kMaxCores <= kSlotBytes,
        "CV staging must fit inside the slot past the pipeline");
    // Per-sweep PP_DATA series FORCED OFF: zones + footprint + CV-first measured 396 B over the
    // 11,264 B code region, and the out[] byte totals answer the traffic questions without it.
    constexpr uint32_t kNocFpSeries = 0u;
    constexpr bool kSelfPhases = kSelfZones != 0 && kSelfDetail != 0;
    // The self frame lives in staging slot kNStage -- one PAST every slot the drain pipeline can touch. The
    // host reserves it by passing (nstage - 1) as kNStage when this is on, so DRISC L1 does not grow and the
    // OFF build is byte-identical. The pipeline only ever reaches slot 2*kGenSlots-1, so nothing else can
    // write here.
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

    // Reads on one NoC, writes on the other: a batch of span reads stays in flight while the previous
    // batch pushes to the host -- on one NoC the barriers would serialize them.
    //
    // RESYNC THE SOFTWARE NOC COUNTERS ON BOTH NOCS, ALWAYS. The barriers compare hardware counters against
    // SOFTWARE MIRRORS that persist across kernel launches on this never-reset core; a run that ends with
    // unacked writes leaves a mirror permanently ahead of hardware and the next launch wedges in its first
    // barrier (measured: HW 14768 vs SW 14770, frozen -- looked like "DRISC cannot run under slow dispatch"
    // until a tt-smi -r). Firmware runs noc_local_state_init() on FW BOOT only, not per launch, so do it
    // here. kNocInit=0 (TT_METAL_PERF_DEBUG_NO_NOC_INIT=1) brings the wedge back on demand for repro.
    if constexpr (kNocInit) {
        noc_local_state_init(NOC_INDEX);
        noc_local_state_init(kReadNoc);
    }
    // Does constructing Noc{kReadNoc} move the RUNTIME global `noc_index`? It matters: the library
    // noc_async_write_barrier() defaults to that global, while the writes are issued on the COMPILE-TIME
    // NOC_INDEX. If they diverge, the barrier guarding staging reuse watches the wrong NoC.
    const uint32_t noc_index_before = noc_index;
    Noc noc{kReadNoc};
    const uint32_t noc_index_after = noc_index;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = kPcieEncOverride != 0 ? kPcieEncOverride : sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    // The host-written ack word, read for the HIGH-mode credit veto (reserve_pages_bounded walks the same
    // address; a filler has exactly one downstream).
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

    // ---- live liveness window, readable by the host WHILE the loop runs ----
    //
    // The results block only publishes after the loop exits, so without these the host cannot tell
    // "kernel exited" from "blocked" from "spinning with nothing to do".
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    volatile tt_l1_ptr uint32_t* phase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 8);
    constexpr uint32_t kPhaseInit = 1, kPhasePoll = 2, kPhaseReserve = 3, kPhaseWrite = 4, kPhaseExit = 5;
    // Sub-phases of WRITE, so a stuck egress says WHICH call blocks: 6=chunked NoC write to the
    // PCIe tile, 7=socket_push_pages bookkeeping, 8=the bytes_sent notify (a PCIe write of the
    // producer pointer), 9=write issued, back in the sweep body.
    constexpr uint32_t kPhWrChunk = 6, kPhWrPush = 7, kPhWrNotify = 8, kPhWrDone = 9;
    // 10 = frame dropped (credit wait gave up); 11/13 = the sweep-body write barriers, which are
    // OUTSIDE ship_run and so reported a stale phase before they had markers of their own.
    constexpr uint32_t kPhDropped = 10, kPhBar1 = 11, kPhBarTail = 13;
    constexpr uint32_t kPhTailBar = 15;  // the post-loop write barrier, which shared phase 11 and hid there
    // 14 = socket_barrier() in the exit tail. Every blocking call needs its own marker or the phase
    // word lies by omission.
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
    // PER-CORE SERVICE INTERVAL, as a distribution. The MEAN cannot explain the stalls: at delay 30 it is
    // 22.6 us against a 47.7 us fill -- 2.1x headroom -- and producers still block 65 k times, so what
    // blocks them is this distribution's TAIL. Buckets double from 8192 cycles (~6.1 us at 1.35 GHz);
    // the host converts. Sampled at the head write-back, which IS the moment a ship releases a producer.
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
    // Seeded at 0 = sweep immediately. Seeding high to skip the idle ramp was measured to stall producers
    // at burst onset: the first sweep must never wait.
    uint32_t gap = kGapCycles;
    uint32_t overflows = 0;
    uint32_t hb_slot = 0;
    uint32_t scan_rot = 0;
    uint32_t deferred_seen = 0;  // ship_deferred as of the last rotation decision
    uint32_t fill_hist[8] = {};  // busy-sweep PEAK lane fill, 1/8-of-ring buckets -- the hysteresis signal
    uint64_t c_scan = 0;  // decide-loop cycles alone, for a ns/core figure the host can print

    uint64_t c_read = 0;     // bulk span reads: issue + barrier
    // ISSUE COST, split out from the barrier wait. Programming a cmd buf occupies the DRISC; the bytes then
    // move on the NIU, overlapped with the previous generation's ship. So the sweep's critical path is issue
    // time plus whatever wait survives that overlap, and these two say which of them it is.
    uint64_t c_issue = 0;
    uint64_t c_cv = 0;  // the sweep-start CV pass alone: issue of the per-core control reads plus their barrier
    uint32_t n_gather_rd = 0, n_cv_rd = 0;
    uint64_t c_proc = 0;     // control-vector inspection, prefix + head patch, head write-back
    uint64_t c_reserve = 0;  // socket_reserve_pages -- host credit wait
    uint64_t c_write = 0;    // PCIe write + push + notify
    uint64_t c_barrier = 0;  // write barrier before staging is reused
    // `write` sub-split: the chunked NoC write moves bytes (and can block on command-buffer
    // availability), push_pages is local bookkeeping, notify_receiver is a PCIe write.
    uint64_t c_ph_head = 0;  // the per-core head write-back inside proc (see ship_batch)
    uint64_t c_wr_chunk = 0;
    uint64_t c_wr_push = 0;
    uint64_t c_wr_notify = 0;
    uint64_t c_idle = 0;
    uint64_t c_busy = 0;
    // WINDOW-SCOPED PHASES. Every lifetime percentage is dominated by idle CV polling: a drainer is
    // resident for seconds while a capture is tens of milliseconds, so a lifetime "pace 63%" is the wait
    // FOR a workload, not headroom inside one -- and reading it as headroom is wrong in the direction that
    // matters. Snapshot at the first shipping sweep and again at every later one; the difference is the
    // phases as they stood while data actually flowed.
    bool win2_open = false;
    uint64_t w0_t = 0, w1_t = 0, w0_busy = 0, w1_busy = 0, w0_idle = 0, w1_idle = 0, w0_pace = 0, w1_pace = 0;
    uint64_t w0_cv = 0, w1_cv = 0, w0_issue = 0, w1_issue = 0;
    uint32_t w0_frames = 0, w1_frames = 0, w0_sweeps = 0, w1_sweeps = 0;
    uint32_t sweeps_idle = 0;
    uint32_t max_sweep = 0;
    // The knee is set by the WORST sweep beating ring fill time, and the worst is ~2.5x the mean, so
    // averages cannot say why.
    uint32_t ws_read = 0, ws_proc = 0, ws_rsv = 0, ws_wr = 0, ws_bar = 0;
    uint32_t max_reserve = 0;
    uint64_t c_pace = 0;
    // Set when a bounded write barrier expires: egress is dead, so STOP SHIPPING for good.
    // Never means "continue anyway" -- staging reuse depends on that barrier having flushed.
    bool egress_dead = false;
    // Which staging generations hold a ship possibly still in flight. Persists ACROSS sweeps: the write
    // wait happens at the slots' next refill, never at sweep end, so a sweep's final ship drains under
    // the pace gap or the next sweep's CV pass instead of on the sweep's critical path.
    bool gen_shipped[kNGens] = {};

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
    // Publish trampoline. self_mark_w0's ring-full path must publish, but the publisher (self_publish,
    // defined after the egress lambdas because it ships THROUGH them) is exactly what opens zones through
    // self_mark_w0 from inside that egress -- a genuine reference cycle no lambda declaration order can
    // express. One captureless trampoline breaks it; it is wired immediately after self_publish is defined,
    // long before the main loop can emit a marker. (Re-entrancy stays guarded by self_busy, as before.)
    struct SelfPub {
        void* ctx = nullptr;
        void (*fn)(void*) = nullptr;
    } self_pub;
    // Append one 2-word marker at a fresh timestamp. Publishes first if the ring cannot hold a sticky
    // plus a marker; a marker is dropped only if that publish could not free room, and is counted.
    // Takes a RAW word0 so the PP_DATA sample shares this prologue -- two copies do not fit the code
    // region. Returns whether the words were written: a caller appending a PAYLOAD must not write
    // orphan words after a dropped header. Out of line because this reaches self_publish().
    auto self_mark_w0 = [&](uint32_t w0) -> bool {
        if constexpr (kSelfZones == 0) {
            (void)w0;
            return false;
        } else {
            if (!self_on || self_busy) {
                return false;
            }
            // The margin is 9 words, which covers every shape that uses this: 1-word sticky + 2-word marker = 3,
            // and 1 + 3 + SPSC_NOCFP_WORDS = 8 for a footprint sample (PP_DATA grew a word2 when its id
            // widened to the full 27 bits). Leave a margin so a run can never exceed the ring capacity,
            // which the host clamps (spsc_span_live) rather than trusts.
            // RING FULL -> PUBLISH AND CARRY ON. Publishing HERE rather than once per sweep is what makes tracing
            // every sweep affordable: a frame carries as many markers as the ring holds (~250 at full detail, ~63
            // SWEEP/PACE pairs at detail 0) instead of the handful one sweep produces. A marker is only lost if the
            // publish could not free the ring -- egress dead -- and that is counted, never silent.
            if (self_tail - self_head > kRingWords - 9u) {
                if (self_pub.fn != nullptr) {
                    self_pub.fn(self_pub.ctx);
                }
                if (self_tail - self_head > kRingWords - 9u) {
                    self_dropped++;
                    return false;
                }
            }
            // Clock read AFTER any publish, so a marker can never carry a time from before the publish that
            // made room for it (the same ordering argument mark_time makes in kernel_profiler.hpp).
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
    // The guard is INLINE at the scope site, so an uninstrumented sweep pays a flag check and never a
    // call; only then does the shared out-of-line prologue run.
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

    // Ship `count` adjacent staged slots straight into the host FIFO, ONE write per frame (two when it
    // straddles the FIFO wrap): the gather reads already landed every live run at its packed wire offset,
    // so a staged slot IS its frame's wire image. Only the trailing page fill is never written -- the host
    // derives every offset from the control vector and reads past it (profiler_common.h).
    auto emit_slots = [&](uint32_t start, uint32_t count) {
        if (count == 0) {
            return;
        }
        if (egress_dead) {
            *phase = kPhDropped;
            dropped_frames += count;
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
        const uint64_t t1 = kInstr != 0 ? get_timestamp() : t0;  // t0 stays real (the credit deadline); anchoring the gated chain to it keeps every diff zero
        c_reserve += t1 - t0;
        if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
            max_reserve = static_cast<uint32_t>(t1 - t0);
        }
        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfMarkPhase> z_write(self_mark_phase);
        *phase = kPhWrChunk;
        // Not hoisted out of emit_slots: the head write-backs between pushes program the same command
        // buffer, so the state must be re-established per push.
        noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);
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
        // filler whose data rides the OTHER unicast VC the mesh may deliver the bytes_sent word ahead of
        // the data it announces -- the host then decodes bytes that have not landed (measured: one socket
        // with 325 resyncs / 1,957 bad frames in an otherwise clean 150k run). Same cmd state, same VC,
        // same route as the data pieces above makes delivery order the issue order again.
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

    // Barrier AT THE END: after a publish the next marker overwrites a word the in-flight frame is
    // still shipping, so the wait belongs before the NEXT publish, not after the previous one.
    // Phase counters are saved/restored around the egress call so the self frame does not bill itself.
    // ================================ INSTRUMENTATION START (self-frame publish, NoC-footprint sampling, window arming) ====
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
            if (write_barrier_bounded(get_timestamp() + kCreditWaitCycles)) {
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
    // One VARIABLE-LENGTH PP_DATA packet of this sweep's NoC counter deltas. Payload layout is the
    // shared contract in profiler_common.h (see SpscNocFpWord), not a local convention.
    auto self_nocfp = [&]() {
        if constexpr (kSelfZones == 0 || kNocFootprint == 0) {
            return;
        } else {
            // Header + timestamp through the shared prologue (which reads the clock, so this is a POINT
            // marker at emission); payload only if that landed.
            if (!self_mark_w0(kernel_profiler::spsc_data_w0(kernel_profiler::SPSC_DATA_ID_NOCFP))) {
                return;
            }
            self_ring[self_tail % kRingWords] = kernel_profiler::spsc_data_w2(kernel_profiler::SPSC_NOCFP_WORDS);
            self_tail++;
            // Read straight out of nf.last[] at compile-time indices: reads on kReadNoc, writes on NOC_INDEX.
            constexpr uint32_t kRd = uint32_t{kReadNoc} * kNfN;
            constexpr uint32_t kWr = uint32_t{NOC_INDEX} * kNfN;
            // A constexpr INDEX table, not a value array: building the values on the stack first cost code
            // the region does not have.
            static constexpr uint32_t kIdx[kernel_profiler::SPSC_NOCFP_WORDS] = {
                kRd + kNfRdW, kRd + kNfRdT, kWr + kNfWrW, kWr + kNfWrT};
            for (uint32_t i = 0; i < kernel_profiler::SPSC_NOCFP_WORDS; i++) {
                self_ring[self_tail % kRingWords] = nf.last[kIdx[i]];
                self_tail++;
            }
        }
    };
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


    // ---- NoC FOOTPRINT sampling: the only place NIU registers are read. BOTH NoCs are sampled --
    // which NoC carries what is the thing being verified, so the zeros are part of the measurement.
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
                    // Half a lane. Re-derived for the gather-read frame and it still wins: a per-core
                    // dithered trigger over [1/4, 1/2) removed the six fillers' synchronized ship bursts
                    // (worst-sweep write 9.7-11.4 -> 2 us) yet REGRESSED the knee everywhere (delay 50:
                    // 256k -> 900k stalls) -- the extra frames' gather reads cost the sweep more than the
                    // burst did, because the read side, not the PCIe write, is the saturated-sweep wall.
                    const uint32_t lane_trigger = kLaneTrigger;
                    // ROTATED start, but only while the ship threshold is actually deferring cores. The
                    // scan/ship order is also the service order; under deferral a fixed order gives the
                    // last cores a whole sweep less headroom every sweep -- the same handful of cores took
                    // every producer stall while their slice-mates took none. But when every live core
                    // ships every sweep, rotation is the harm instead: the start advancing one slot per
                    // sweep hands every core a near-2-sweep service interval once per rotation period
                    // (measured at delay 48, SHIP_MIN_PCT=1: svc max 36.8 us vs the 21 us sweep, exactly
                    // 1/num_cores of services in the 2-sweep bucket). Wrap by compare, never %: a runtime
                    // modulo is a soft-div on this core (N+64).
                    if (ship_deferred != deferred_seen) {
                        deferred_seen = ship_deferred;
                        if (++scan_rot >= num_cores) {
                            scan_rot = 0;
                        }
                    }
                    const uint64_t t_scan0 = kInstr != 0 ? get_timestamp() : 0;
                    uint32_t c = scan_rot;
                    for (uint32_t k = 0; k < num_cores; k++, (++c >= num_cores ? c = 0 : c)) {
                        const tt_l1_ptr uint32_t* tails =
                            reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                        uint32_t* mine = &head_mirror[c * kNumRisc];
                        if (!seeded[c]) {
                            // Seed from the TAILS: the mirror means "consumed up to here", and everything
                            // before this filler first saw the core predates the workload -- the filler is
                            // resident and verified before any producer runs.
                            for (uint32_t r = 0; r < kNumRisc; r++) {
                                mine[r] = tails[r];
                            }
                            seeded[c] = 1;
                        }
                        // SCAN, UNROLLED INTO REGISTERS: it is L1-access bound, not arithmetic bound, and a
                        // loop over indexed arrays spills on this core -- each spilled word is another L1
                        // round trip per core per sweep.
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
                            // A core that shipped real words last sweep and now scans empty is almost
                            // certainly the producer's 64-word batched tail publish, not an idle core --
                            // at rates where it matters, all five lanes sitting inside an unpublished
                            // batch is the only way to read 0. Skipping it hands the core a 2-sweep
                            // service interval (measured at delay 48: ~380 uncounted skips per filler,
                            // svc max 44 us, and the stalls sat on exactly these cores). Ship it: by
                            // issue time the in-flight tail refresh has usually crossed a boundary and
                            // the frame carries real words. One-shot -- a genuinely idle core wastes at
                            // most one empty frame before going cold.
                            if (hot[c] == 0) {
                                continue;
                            }
                            hot[c] = 0;
                            ship_list[n_ship++] = static_cast<uint8_t>(c);
                            continue;
                        }
                        // PER-LANE trigger, not span fill: one hot lane at 90% of its own ring is only ~18% of the span,
                        // and the producer that blocks is always a LANE. Level check only -- no growth term, so the
                        // producer's batched tail publish cannot fool it.
                        if (stop_seen_at == 0 && live < kShipMinWords && peak < lane_trigger &&
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
                    // The per-lane walk stays a LOOP, unlike the scans: lane r's bookkeeping runs
                    // while lane r-1's read is still being accepted by the NIU, so the L1 accesses hide
                    // behind the issues. Unrolling to scalars with a shared emit helper was measured
                    // 2026-08-27 at the 8-filler knee and REGRESSED: gather-issue 6.4 -> 8.3 us/sweep,
                    // service interval 16.5 -> 18.4 us, d23 stalls 72 -> 54k -- front-loading all five
                    // lanes' bookkeeping serializes it against every issue instead of interleaving.
                    for (uint32_t r = 0; r < kNumRisc; r++) {
                        const uint32_t tail = tails[r];
                        uint32_t run = tail - mine[r];
                        if (run > kRingWords) {
                            overflows++;
                            run = kRingWords;
                        }
                        const uint32_t start = tail - run;
                        // Cap the frame at the slot's payload capacity, WHOLE LANES ONLY: a lane that
                        // does not fit ships nothing this frame and goes next sweep. The published tail
                        // is always a packet boundary but an arbitrary word count is not -- clamping
                        // mid-run split packets across frames and corrupted the lane stream (measured:
                        // 300-550 order regressions per socket). A full lane ends at the published
                        // tail, so all-or-nothing keeps every claim on packet boundaries.
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

                auto ship_batch = [&](uint32_t n, uint32_t g) {
                    const uint64_t t_p0 = kInstr != 0 ? get_timestamp() : 0;
                    // c_self joins the nested term because self_publish RESTORES c_reserve/c_write, so without it a
                    // mid-batch self publish would be charged to `proc`.
                    const uint64_t flush_at = c_reserve + c_write + (kSelfZones != 0 ? c_self : 0);
                    // PROC as an ordinary RAII scope over the whole batch, so its children (the credit wait and the
                    // write inside emit_slots) nest under it.
                    kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PROC, SelfMarkPhase> z_proc(
                        self_mark_phase);
                    for (uint32_t i = 0; i < n; i++) {
                        const uint32_t sl = g * kGenSlots + i;
                        const uint32_t c = slot_core[sl];
                        uint32_t* mine = &head_mirror[c * kNumRisc];
                        // HEAD WRITE-BACK: it releases the producer, and is safe at once -- the payload is
                        // resident in staging (this generation's read barrier passed), so those ring slots
                        // are free regardless of when the frame reaches the host.
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
                        // POSTED: the barriers exist to protect STAGING reuse, which a head write never
                        // touches, so its worker ACK round-trip bought nothing -- and the ack packet itself
                        // rode the congested worker route. Scratch reuse stays safe on the slot rotation.
                        // Issuing this on the read NoC instead, to decouple head visibility from the egress
                        // queue, was measured stall-neutral at the delay-16 saturation wall and deleted.
                        noc_async_write_one_packet<true, true>(
                            sc,
                            get_noc_addr(
                                coords[c] & 0xFFFFu, coords[c] >> 16, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u),
                            kNumRisc * 4u);
                        hb_slot = (hb_slot + 1u) & (kMaxCores - 1u);
                        if constexpr (kSvcInstr != 0) {
                            c_ph_head += get_timestamp() - t_h0;
                        }

                        frames++;
                        total_words += live;
                    }
                    // A LISTED CORE is live by construction, so the batch having cores IS the work signal.
                    if constexpr (kSelfZones != 0) {
                        if (!self_on && n != 0) {
                            self_arm();  // opens the window at EVERY detail level
                        }
                    }
                    emit_slots(g * kGenSlots, n);
                    if (!egress_dead) {
                        gen_shipped[g] = true;
                    }
                    // SATURATING: the nested emit_slots time is subtracted so it is not double-counted, but an unsigned
                    // wrap here once produced "proc 18727729111430.1%".
                    {
                        const uint64_t t_p1 = kInstr != 0 ? get_timestamp() : 0;
                        const uint64_t span = t_p1 - t_p0;
                        const uint64_t nested = (c_reserve + c_write + (kSelfZones != 0 ? c_self : 0)) - flush_at;
                        c_proc += (span > nested) ? (span - nested) : 0;
                    }
                    // z_proc closes here: PROC spans the whole batch, i.e. c_proc plus its nested children, which is
                    // what a Tracy parent is.
                };

                // A batch's FIRST frame may go WIDE (spill into the generation's second slot) when its
                // core is genuinely backlogged, and the batch then carries that one frame. This is how a
                // capture-onset backlog (every ring full) drains in one visit instead of trickling out
                // over three capped sweeps that hold the already-blocked producers for a flat ~130 stalls.
                // The trigger must sit well above the cap: per-sweep production RIDES the cap at the
                // lowest workable delays, and a wide frame there would burn a whole generation on an
                // ordinary core, halving the pipeline depth that bought the low-end knee.
                uint32_t cur = 0;
                while (cur < n_ship) {

                    // This generation's previous ship must be out of staging before its slots refill. SENT is
                    // enough: the next writer of this staging is this core's own NIU read responses, so the
                    // usual source-reuse gate applies. gen_shipped persists across sweeps, so a sweep's LAST
                    // ship is never waited on inside its own sweep -- it drains under the pace gap or the next
                    // sweep's CV pass, and this is the wait that catches it if it has not.
                    if (gen_shipped[gen]) {
                        const uint64_t t_b0 = get_timestamp();
                        *phase = kPhBar1;
                        bool flushed;
                        {
                            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfMarkPhase> z_bar(
                                self_mark_phase);
                            flushed = write_barrier_bounded<true>(t_b0 + kCreditWaitCycles);
                        }
                        c_barrier += kInstr != 0 ? (get_timestamp() - t_b0) : 0;
                        if (!flushed) {
                            egress_dead = true;
                            break;
                        }
                        gen_shipped[gen] = false;
                    }

                    // c_read is TWO disjoint intervals per batch -- the issue, and whatever wait survives the concurrent
                    // ship -- so it takes two zones.
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
                        // Refresh the NEXT batch's tails in the same flight: issue_core consumes
                        // [mirror, tail-at-read), so on the sweep-start snapshot alone the last batch's
                        // cores are consumed ~a whole sweep stale, and their backlog rides one sweep of
                        // production higher than the first batch's (measured at delay 48, SHIP_MIN_PCT=1:
                        // peak lanes 256-320 words against interval x rate ~150, and the scan-order-last
                        // core of every filler took all the stalls). This generation's read barrier covers
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
                        ship_batch(pend_n, pend_gen);
                    }

                    // Issue cost plus only the wait REMAINING after the concurrent ship. Timing to the barrier instead
                    // would swallow ship_batch and double-count it against c_proc -- it did, and phases summed 133%.
                    const uint64_t t_after_proc = kInstr != 0 ? get_timestamp() : 0;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ_WAIT, SelfMarkPhase> z_wait(
                            self_mark_phase);
                        noc.async_read_barrier();
                    }
                    const uint64_t t_read_end = kInstr != 0 ? get_timestamp() : 0;
                    c_issue += t_issue - t_batch0;
                    c_read += (t_issue - t_batch0) + (t_read_end - t_after_proc);

                    pend_n = n;
                    pend_gen = gen;
                    have_pend = true;
                    gen = gen + 1u == kNGens ? 0u : gen + 1u;
                }
                if (have_pend) {
                    ship_batch(pend_n, pend_gen);
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
        // Stamped at the sweep's END so it lines up with the DRISC-SWEEP zone that just closed.
        if constexpr (kNocFpSeries != 0) {
            self_nocfp();
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
        // traffic, and a producer must never wait on it. Live-but-untriggered counts as work here -- see
        // kCvBusyPeak. (A 16-sweep hold-down before the first growth was measured stall- and pace-neutral
        // at the delay-16 saturation wall: the losing fillers' published-empty streaks outlast any
        // plausible hold anyway.)
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

    // socket_barrier() waits for the host to ack everything, so it hangs on a dead consumer just
    // like the write barrier did. Skip both when we already know the consumer is gone.
    const bool consumer_gone = egress_dead || credit_timeouts != 0;
    *phase = kPhSockBar;
    if (!consumer_gone) {
        socket_barrier(sender);
    }
    *phase = kPhBarTail;
    *phase = kPhTailBar;  // distinct from kPhBar1: the tail barrier used to run while phase still read 11
    (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles);
    // The posted head write-backs are outside that barrier's predicate; drain their SENT counter (20 B
    // packets stream out in ns) so no scratch slot or unstreamed head is left behind at report time.
    {
        const uint64_t t_ps = get_timestamp() + 1350000u;
        while (!ncrisc_noc_posted_writes_sent(NOC_INDEX) && get_timestamp() < t_ps) {
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
    // MUST equal out[73] (self_tail). Less means trace LOST IN THE RING at teardown -- what the tail
    // flush above prevents, and what no other counter shows.
    out[87] = self_words_shipped;
    // sync_timeouts MUST be 0: a timeout means this drainer parked at the barrier and never
    // contributed to the fiducial, so the whole common trigger is void.
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
        // The instrument's own cost, reported rather than hidden: 8 NIU register loads plus 2 clock reads per
        // sweep. The host prints it as a share of the run so "the footprint counters are free" is a measured
        // claim and not an assertion.
        out[123] = static_cast<uint32_t>(nf.cost & 0xFFFFFFFFu);
        out[124] = static_cast<uint32_t>(nf.cost >> 32);
        // Retired: the write totals now sum posted + non-posted words (see nf_sample_regs), so the old
        // posted-must-be-zero blind-spot check has nothing left to guard. Zeroed to keep the layout.
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

    // -------- NIU restore, on the host's word --------
    //
    // NIU_CFG_0 persists until a chip reset, so whoever set stream mode owns putting it back. It must
    // happen HERE (by teardown the mesh device can no longer launch a program) and LAST: in NOC2AXI mode an
    // inbound DRAM-range address is forwarded to GDDR, so the flip takes this L1 -- `done`, the results,
    // the socket's bytes_acked -- out of the host's view.
    for (uint32_t spins = 0; spins < 200000000u && *stop != 2u; spins++) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
