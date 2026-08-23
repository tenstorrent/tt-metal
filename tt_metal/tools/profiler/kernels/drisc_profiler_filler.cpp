// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's FILLER: sweep a quarter of the worker grid's SPSC rings and write complete
// frames into this filler's own device-DRAM ring. No socket, no PCIe, no host MMIO -- its
// back-pressure is a DRAM ring of tens of MB, not the 12 MiB host FIFO. Four fillers cover the grid.
//
// Roles, placement evidence and wire format: tools/drisc_drain/FINDINGS.md.

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
    // Arg 10 retired (SHIP_REPEAT egress amplifier).
    constexpr uint32_t kShipRepeat = get_compile_time_arg_val(10);
    // 1 = resync the software NoC mirrors from hardware at entry (see the wedge note below). 0 = diagnostic.
    constexpr uint32_t kNocInit = get_compile_time_arg_val(11);
    // Args 12..15 and 17..18 are retired. The indices stay occupied: arg positions appear in JIT cache
    // keys and in FINDINGS notes.
    constexpr uint32_t kPcieEncOverride = get_compile_time_arg_val(16);

    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
    constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;  // 2,624 words = 10,496 B
    constexpr uint32_t kSpanBytes = kSpanWords * 4u;
    // LIVE capacity = the rings alone; kSpanWords also counts the 64-word control vector.
    constexpr uint32_t kLiveWords = kNumRisc * kRingWords;
    constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    // Sized for the PACKED worst case, not the raw span -- see spsc_span_slot_words(). This is what makes
    // packing unconditional: the pads can push a nearly-full span's packed image past the raw span's size.
    constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);  // 2,656
    constexpr uint32_t kSlotBytes = kSlotWords * 4u;       // 10,560
    constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
    constexpr uint32_t kPageBytes = kPageWords * 4u;
    constexpr uint32_t kPagesPerSlot = kSlotWords / kPageWords;  // 166
    // Reads take the NoC the writes do not; NOC_INDEX (the kernel's configured NoC) carries egress.
    constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
    // Two staging generations: one fills while the other drains.
    constexpr uint32_t kGenSlots = kNStage / 2;
    // ---- ROLE SPLIT (see the header) ----
    constexpr uint32_t kRoleFiller = 1, kRoleMover = 2;
    constexpr uint32_t kRole = kRoleFiller;  // this file IS the filler; carg 20 stays reserved
    constexpr uint32_t kDramBank = get_compile_time_arg_val(21);    // allocator bank id of this ring
    constexpr uint32_t kDramAddr = get_compile_time_arg_val(22);    // bank-relative base of this ring
    constexpr uint32_t kDramFrames = get_compile_time_arg_val(23);  // ring capacity in whole frames
    constexpr uint32_t kHsAddr = get_compile_time_arg_val(24);      // FILLER: its own handshake block, in L1
    constexpr uint32_t kPeerXY = get_compile_time_arg_val(25);      // MOVER: (y<<16)|x virtual of peer 0
    constexpr uint32_t kPeerHsAddr = get_compile_time_arg_val(26);  // MOVER: peer 0's handshake block
    // kNPeer is a COMPILE-time 1 or 2, so a single-ring mover emits exactly the code it did before this
    // existed. Args 21/22 are peer 0's (bank, addr); a filler reuses those for its own ring.
    constexpr uint32_t kNPeer = get_compile_time_arg_val(27);
    constexpr uint32_t kPeerXY1 = get_compile_time_arg_val(28);
    constexpr uint32_t kPeerHsAddr1 = get_compile_time_arg_val(29);
    constexpr uint32_t kDramBank1 = get_compile_time_arg_val(30);
    constexpr uint32_t kDramAddr1 = get_compile_time_arg_val(31);
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
    // PER-CORE SHIP THRESHOLD (0 = ship every live core every sweep). A frame costs the pipe the same
    // whether it carries 200 live words or 2,000, so a core ships only when it is worth the frame:
    // enough live words, any lane past kLaneShipWords, or the age bound below.
    constexpr uint32_t kShipMinPct = get_compile_time_arg_val(39);
    // Half a lane's ring, as OBSERVED tail: the batched publish can hide up to SPSC_PUBLISH_BATCH_WORDS
    // more, so the true worst occupancy at trigger is ~62%, leaving ~190 words of service headroom.
    constexpr uint32_t kLaneShipWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE / 2u;
    constexpr uint32_t kShipMaxAgeSweeps = 512u;
    constexpr uint32_t kShipMinWords = (kLiveWords * kShipMinPct) / 100u;
    // CV-FIRST SWEEPS: read each core's ring TAILS (32 B), decide the ship set, bulk-read spans only
    // for it. The tails read is a HINT -- the span read re-reads the control vector as its leading
    // bytes and remains the authoritative snapshot.
    //
    // Heads publish at write-ISSUE time, which keeps the DRAM ack wait off the sweep path. Sound only
    // because the mover VERIFIES landing per frame (kSeqWord): NoC-write vs DMA-read ordering at one
    // address is UNDEFINED and WR_ACK is not a cross-master fence.
    constexpr uint32_t kSeqWord = 3u;
    constexpr uint32_t kSeqSrcWord = 7u;
    // In-place re-reads per sub-batch before deferring to the next visit, sized to cover the trailing
    // stamp's NoC flight rather than to poll. TAILS ONLY: the decision needs each ring's tail; heads
    // come from the local mirror.
    constexpr uint32_t kCvReadBytes = 32;
    constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
    // Idle backoff ceiling (~20 us): collapse on work, creep when idle. A filler-sized version of the
    // mover's policy.
    constexpr uint32_t kCvIdleGapMax = 27000;
    static_assert(kShipMinPct != 0, "CV-first sweeps exist to feed the per-core ship decision");
    static_assert(
        kSelfZones != 0 || 2u * kGenSlots < kNStage,
        "CV staging needs a slot past the 2-generation pipeline (kNStage must be odd when self-zones are off)");
    // The tails land in the ring-1 area of the slot past the pipeline. With self-zones ON that slot holds
    // the self FRAME, and this placement is still safe: only the self frame's ring 0 is ever live, so its
    // ring 1..4 storage (8 KiB) is written by nobody and never ships -- the pack skips empty rings, and a
    // filler's raw stage of the self slot carries these bytes only as dead DRAM-ring padding. That shared
    // dead space is what lets CV-first and drainer self-profiling coexist.
    constexpr uint32_t kCvSlot = kSelfZones != 0 ? kNStage : 2u * kGenSlots;
    constexpr uint32_t kCvBase = kStageBase + kCvSlot * kSlotBytes + (kPrefix + kCtrlWords + kRingWords) * 4u;
    static_assert(
        kCvReadBytes * kMaxCores <= 4u * kRingWords * 4u,
        "CV tails staging must fit the self slot's dead ring space");
    // Per-sweep PP_DATA series FORCED OFF: zones + footprint + CV-first measured 396 B over the
    // 11,264 B code region, and the out[] byte totals answer the traffic questions without it.
    constexpr uint32_t kNocFpSeries = 0u;
    constexpr bool kSelfPhases = kSelfZones != 0 && kSelfDetail != 0;
    // The self frame lives in staging slot kNStage -- one PAST every slot the drain pipeline can touch. The
    // host reserves it by passing (nstage - 1) as kNStage when this is on, so DRISC L1 does not grow and the
    // OFF build is byte-identical. A filler's pipeline only ever reaches slot 2*kGenSlots-1 and a mover's
    // batch is capped at kNStage, so nothing else can write here.
    constexpr uint32_t kSelfSlot = kNStage;
    // Indexed by peer slot. constexpr arrays of compile-time args, so nothing is loaded from memory to reach
    // them; the loop over kNPeer is a fully-known trip count.
    // Handshake block: four words, 16 B apart so each is independently addressable by a 4 B NoC write
    // without a read-modify-write on its neighbours.
    //   +0  head     filler stores locally,  mover NoC-reads
    //   +16 tail     mover NoC-writes,       filler reads locally
    //   +32 probe_f  host writes a magic,    mover NoC-reads and echoes into its own L1  (proves reads work)
    //   +48 probe_m  mover NoC-writes magic, host reads it off the filler                (proves writes work)
    // Both directions are probed at bring-up rather than assumed: a wrong peer-L1 coordinate yields
    // plausible counters and a garbage handshake, which is how this class of bug hides. The magics carry
    // the peer INDEX, so a mover that named the wrong filler cannot pass by reading someone else's probe.
    constexpr uint32_t kHsHead = 0, kHsTail = 16, kHsProbeF = 32, kHsProbeM = 48;
    constexpr uint32_t kProbeMoverMagic = 0x5A0FE1EDu;
    static_assert(kRole == kRoleFiller || kRole == kRoleMover, "unknown drainer role");
    static_assert(kGenSlots >= 1, "need at least one slot per staging generation");

    static_assert(kSelfZones == 0 || kSelfHoldCycles >= 1, "a 0-cycle window hold would trace nothing");
    static_assert(kSelfDetail <= 1, "detail is 0 (SWEEP + PACE) or 1 (full per-batch phases)");
    static_assert(kSelfZones == 0 || kSelfMaxFrames >= 1, "self-profiling with a 0 frame budget captures nothing");
    // The sync event rides the self-zone marker ring, so with zones off it would have nowhere to go
    // and would silently measure nothing.
    static_assert(kSyncEvent == 0 || kSelfZones != 0, "the sync event rides the self-zone ring; enable zones");

    static_assert(kSpanBytes <= NOC_MAX_BURST_SIZE, "the fused span read must fit one NoC burst");
    static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
    static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");
    // The packed gather's congruence argument (profiler_common.h): pads bring each run to its ring phase,
    // and everything else -- the slot base, the payload base, a wrap continuation -- must land congruent
    // with NO pad, which these divisibilities are the proof of.
    static_assert(
        kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_PCIE_WRITE_ALIGNMENT_BYTES,
        "the shared pad rule no longer matches this part's NoC write congruence");
    static_assert(
        kRingWords % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
            (kPrefix + kCtrlWords) % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
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


    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;

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
    // PCIe tile, 7=socket_push_pages bookkeeping, 8=socket_notify_receiver (a PCIe write of the
    // producer pointer), 9=write issued, back in the sweep body.
    constexpr uint32_t kPhWrChunk = 6, kPhWrPush = 7, kPhWrNotify = 8, kPhWrDone = 9;
    // 10 = frame dropped (credit wait gave up); 11/12/13 = the sweep-body write barriers, which are
    // OUTSIDE ship_run and so reported a stale phase before they had markers of their own.
    constexpr uint32_t kPhDropped = 10, kPhBar1 = 11, kPhBar2 = 12, kPhBarTail = 13;
    constexpr uint32_t kPhTailBar = 15;  // the post-loop write barrier, which shared phase 11 and hid there
    // 14 = socket_barrier() in the exit tail. Every blocking call needs its own marker or the phase
    // word lies by omission.
    constexpr uint32_t kPhSockBar = 14;
    // 16 = the DRAM ring capacity wait (a FILLER's only blocking wait, and the one c_reserve now measures).
    constexpr uint32_t kPhRingWait = 16;
    *hb = 0;
    *phase = kPhaseInit;

    // FILLER: its own handshake block. head is stored locally and read over the NoC by its mover; tail is
    // written over the NoC by that mover and read locally here.
    volatile tt_l1_ptr uint32_t* hs_head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHsAddr + kHsHead);
    volatile tt_l1_ptr uint32_t* hs_tail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHsAddr + kHsTail);
    uint32_t frames_staged = 0;   // FILLER: frames written into DRAM (monotonic; == published head)
    uint32_t frames_flushed = 0;  // FILLER: of those, how many are barrier-flushed and safe to publish
    // Per peer ring, and nothing here may be shared between peers: one mv_tail for two rings would ack
    // frames on one that were only read from the other, handing the filler room it does not have.
    uint32_t ring_hi[2] = {0, 0};
    uint32_t ring_blocked = 0;  // FILLER: stage_run calls that had to wait for ring room at all
    if constexpr (kRole == kRoleFiller) {
        *hs_head = 0;
        // NOT hs_tail: the host zeroes it before launch and the MOVER owns it from then on. Zeroing it here
        // would race a mover that has already started acking.
    }

    // Every frame's prefix is IDENTICAL and the bulk read lands past it (at slot + 16 words), so it is
    // written once here. Word 1 -- the packed payload length -- is the exception: ship_once patches it at
    // push time, the first moment the fill is known.
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefix; k++) {
            pfx[k] = 0;
        }
    }

    static uint32_t head_mirror[kMaxCores * kNumRisc];
    static uint8_t seeded[kMaxCores];
    static uint16_t ship_age[kMaxCores];
    static uint8_t ship_list[kMaxCores];  // CV-first: this sweep's ship set, dense core indices
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
        ship_age[i] = 0;
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
    // SATURATION BYPASS: when nearly every core ships every sweep the tails pass decides nothing, so
    // sweep the old full-span way. Hysteresis on a 1/8-of-cores slack keeps it from flapping.
    bool cv_bypass = false;
    uint32_t cv_below = 0;

    uint64_t c_read = 0;     // bulk span reads: issue + barrier
    uint64_t c_proc = 0;     // control-vector inspection, prefix + head patch, head write-back
    uint64_t c_reserve = 0;  // socket_reserve_pages -- host credit wait
    uint64_t c_write = 0;    // PCIe write + push + notify
    uint64_t c_barrier = 0;  // write barrier before staging is reused
    // `write` sub-split: the chunked NoC write moves bytes (and can block on command-buffer
    // availability), push_pages is local bookkeeping, notify_receiver is a PCIe write.
    uint64_t c_ph_head = 0;  // the per-core head write-back inside proc (see process_batch)
    uint64_t c_wr_chunk = 0;
    uint64_t c_wr_push = 0;
    uint64_t c_wr_notify = 0;
    uint64_t c_idle = 0;
    uint64_t c_busy = 0;
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
        pfx[0] = kernel_profiler::spsc_span_w0();
        pfx[1] = kSpanWords;  // the full five-ring span: only ring 0 is live, and the host skips the rest
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

    auto stage_run = [&](uint32_t start, uint32_t count) {
        // Guarded as a whole rather than per-statement: kDramFrames is 0 on every non-filler build, and
        // `frames_staged % kDramFrames` would then be a compile-time divide by zero even though nothing
        // ever calls this.
        if constexpr (kRole != kRoleFiller) {
            (void)start;
            (void)count;
            return;
        } else {
            if (count == 0) {
                return;
            }
            if (egress_dead) {
                *phase = kPhDropped;
                dropped_frames += count;
                return;
            }
            const uint64_t t0 = get_timestamp();
            *phase = kPhRingWait;
            // BOUNDED, with the same three ways out as reserve_pages_bounded: room granted, host asked us to
            // stop, or the deadline passed. Dropping a frame keeps the producers running -- the heads for these
            // slots are already written back -- and losing capture beats wedging the workload.
            bool room = false;
            bool waited = false;
            {
                // A filler's only blocking wait (DRAM ring room), under the same CREDIT-WAIT name the mover's
                // socket wait carries -- what c_reserve means for this role. Ordinary RAII zone.
                kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_CREDIT_WAIT, SelfMarkPhase> z_credit(
                    self_mark_phase);
                for (;;) {
                    invalidate_l1_cache();
                    // tail is only ever ADVANCED by the mover and head only by us, so this cannot underflow.
                    const uint32_t inflight = frames_staged - *hs_tail;
                    if (inflight > ring_hi[0]) {
                        ring_hi[0] = inflight;
                    }
                    if (inflight + count <= kDramFrames) {
                        room = true;
                        break;
                    }
                    waited = true;
                    if (*stop != 0 || get_timestamp() >= t0 + kCreditWaitCycles) {
                        break;
                    }
                }
            }
            if (waited) {
                ring_blocked++;
            }
            const uint64_t t1 = get_timestamp();
            c_reserve += t1 - t0;
            if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
                max_reserve = static_cast<uint32_t>(t1 - t0);
            }
            if (!room) {
                *phase = kPhDropped;
                credit_timeouts++;
                dropped_frames += count;
                return;
            }
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfMarkPhase> z_write(self_mark_phase);
            *phase = kPhWrChunk;
            // Each frame's seq stamp is a separate trailing 4 B write, issued AFTER that frame's bulk
            // writes on the same route: in-order delivery then makes a visible stamp prove every packet
            // of the frame landed. 1-BASED, because never-written GDDR can read 0 and frame 0 must not
            // alias that. The value rides in kSeqSrcWord (28 == 12 mod 16) to satisfy the BH src/dst
            // congruence, and doubles as the small write's L1 source, so it must persist until sent --
            // which the generation reuse gate covers.
            // PACK INTO THE RING rather than leaving the gather to the mover. Same wire layout either way,
            // but not the same cost: gathering at ship time was ~11 NIU write issues per frame on the
            // write-bound mover, against a filler write path with headroom to spare and twice as many
            // fillers. A frame that lands contiguous here ships in ONE issue.
            const uint32_t slot0 = frames_staged % kDramFrames;
            uint32_t fpages = 0;
            for (uint32_t f = 0; f < count; f++) {
                const uint32_t slot = kStageBase + (start + f) * kSlotBytes;
                const tt_l1_ptr uint32_t* cv = reinterpret_cast<const tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
                uint32_t off = kPrefix + kCtrlWords;
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
                    const uint32_t run =
                        kernel_profiler::spsc_span_live(cv[kernel_profiler::SPSC_RING_HEAD_0 + r], tail, kRingWords);
                    if (run != 0) {
                        off += kernel_profiler::spsc_span_pack_pad(tail - run, off) + run;
                    }
                }
                pfx[0] = kernel_profiler::spsc_span_w0();
                pfx[1] = off - kPrefix;
                fpages += kernel_profiler::spsc_span_frame_words(off - kPrefix) / kPageWords;
                pfx[kSeqWord] = 0;
                pfx[kSeqSrcWord] = frames_staged + f + 1u;
            }
            // The NIU reads the patched length and stamp words; Blackhole stores can reach SRAM out of order.
            asm volatile("fence" ::: "memory");
            for (uint32_t f = 0; f < count; f++) {
                const uint32_t slot = kStageBase + (start + f) * kSlotBytes;
                const uint32_t dslot = (slot0 + f < kDramFrames) ? (slot0 + f) : (slot0 + f - kDramFrames);
                const uint64_t dbase =
                    get_noc_addr_from_bank_id<true>(kDramBank, kDramAddr + dslot * kSlotBytes, NOC_INDEX);
                const tt_l1_ptr uint32_t* cv = reinterpret_cast<const tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                noc_async_write(slot, dbase, (kPrefix + kCtrlWords) * 4u, NOC_INDEX);
                uint32_t off = kPrefix + kCtrlWords;
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
                    const uint32_t run =
                        kernel_profiler::spsc_span_live(cv[kernel_profiler::SPSC_RING_HEAD_0 + r], tail, kRingWords);
                    if (run == 0) {
                        continue;
                    }
                    off += kernel_profiler::spsc_span_pack_pad(tail - run, off);
                    // A lane's live window can straddle its own ring wrap; the pad above is what keeps both
                    // halves congruent with their destination, so each half is a plain write from there.
                    const uint32_t hm = (tail - run) & (kRingWords - 1u);
                    const uint32_t ring_l1 = slot + (kPrefix + kCtrlWords + r * kRingWords) * 4u;
                    const uint32_t chunk = run <= kRingWords - hm ? run : kRingWords - hm;
                    noc_async_write(ring_l1 + hm * 4u, dbase + off * 4u, chunk * 4u, NOC_INDEX);
                    if (chunk < run) {
                        noc_async_write(ring_l1, dbase + (off + chunk) * 4u, (run - chunk) * 4u, NOC_INDEX);
                    }
                    off += run;
                }
            }
            for (uint32_t f = 0; f < count; f++) {
                const uint32_t dslot = (slot0 + f < kDramFrames) ? (slot0 + f) : (slot0 + f - kDramFrames);
                noc_async_write(
                    kStageBase + (start + f) * kSlotBytes + kSeqSrcWord * 4u,
                    get_noc_addr_from_bank_id<true>(
                        kDramBank, kDramAddr + dslot * kSlotBytes + kSeqWord * 4u, NOC_INDEX),
                    4u,
                    NOC_INDEX);
            }

            const uint64_t t2 = get_timestamp();
            c_wr_chunk += t2 - t1;
            c_write += t2 - t1;
            *phase = kPhWrDone;
            frames_staged += count;
            pages += fpages;
            pushes++;
        }
    };

    // Publish the staged-frame head at ISSUE time -- the frames' DRAM writes may still be in flight. This
    // is only sound because the mover verifies each frame's trailing seq stamp before consuming it (stamp
    // visible => whole frame landed, by same-route ordering). Weakening either side of that pair -- the
    // stamps here or the verification there -- reopens torn-frame reads the moment the mover chases closely.
    auto publish_head = [&]() {
        if constexpr (kRole == kRoleFiller) {
            if (frames_flushed != frames_staged) {
                frames_flushed = frames_staged;
                *hs_head = frames_flushed;
            }
        }
    };

    auto emit_run = [&](uint32_t start, uint32_t count, bool do_notify = true) {
        (void)do_notify;
        stage_run(start, count);
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
            emit_run(kSelfSlot, 1);
            c_reserve = s_rsv;
            c_write = s_wr;
            c_wr_chunk = s_ch;
            c_wr_push = s_pu;
            c_wr_notify = s_no;
            pages = s_pages;
            pushes = s_pushes;
            max_reserve = s_maxr;
            if (write_barrier_bounded(get_timestamp() + kCreditWaitCycles)) {
                publish_head();  // a filler's self frame is flushed, so the mover may be told about it
                self_words_shipped += self_tail - self_head;
                self_head = self_tail;
                self_frames++;
            } else {
                // Egress is dead. ship_run/stage_run have already accounted the drop; stop instrumenting rather
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
            // Read straight out of nf.last[] at compile-time indices. A filler reads on kReadNoc and writes
            // on NOC_INDEX; a mover does both on NOC_INDEX, so its unused counters are provable zeros.
            constexpr uint32_t kRd = ((kRole == kRoleMover) ? uint32_t{NOC_INDEX} : uint32_t{kReadNoc}) * kNfN;
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
        const uint64_t t_sweep0 = get_timestamp();
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
        {
            // Constructed AFTER the arming block decided self_on, so an armed-window sweep records its whole
            // body. A sweep that arms mid-body gets only its post-arm children.
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SWEEP, SelfMarkNow> z_sweep(self_mark_now);
            {
                // ---- software pipeline: read generation G on kReadNoc while G^1 ships on NOC_INDEX ----
                uint32_t gen = 0;
                uint32_t pend_base = 0, pend_n = 0, pend_gen = 0;
                bool have_pend = false;
                bool gen_shipped[2] = {false, false};

                // ---- CV-FIRST phases 0+1: gather every core's ring tails, decide the ship set ----
                // Same arithmetic process_batch runs, from a 32 B read instead of a 10,496 B one.
                uint32_t n_ship = 0;
                {
                    if (cv_bypass && cv_below > num_cores / 8u) {
                        cv_bypass = false;
                    }
                    cv_below = 0;
                    if (cv_bypass) {
                        for (uint32_t c = 0; c < num_cores; c++) {
                            ship_list[c] = static_cast<uint8_t>(c);
                        }
                        n_ship = num_cores;
                    } else {
                        const uint64_t t_cv0 = get_timestamp();
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
                            }
                            noc.async_read_barrier();
                        }
                        const uint64_t t_cv1 = get_timestamp();
                        c_read += t_cv1 - t_cv0;
                        for (uint32_t c = 0; c < num_cores; c++) {
                            if (!seeded[c]) {
                                ship_list[n_ship++] = static_cast<uint8_t>(c);
                                continue;
                            }
                            const tt_l1_ptr uint32_t* tails =
                                reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                            uint32_t* mine = &head_mirror[c * kNumRisc];
                            uint32_t live = 0, peak = 0;
                            for (uint32_t r = 0; r < kNumRisc; r++) {
                                uint32_t d = tails[r] - mine[r];
                                if (d > kRingWords) {
                                    d = kRingWords;  // counted as an overflow by the authoritative scan, not here
                                }
                                live += d;
                                if (d > peak) {
                                    peak = d;
                                }
                            }
                            if (peak > max_occ) {
                                max_occ = peak;
                            }
                            if (live == 0) {
                                continue;
                            }
                            // PER-LANE trigger, not span fill: one hot lane at 90% of its own ring is only ~18% of the span,
                            // and the producer that blocks is always a LANE. Level check only -- no growth term, so the
                            // producer's batched tail publish cannot fool it.
                            if (stop_seen_at == 0 && live < kShipMinWords && peak < kLaneShipWords &&
                                ship_age[c] < kShipMaxAgeSweeps) {
                                ship_age[c]++;
                                ship_deferred++;
                                continue;
                            }
                            if (ship_age[c] >= kShipMaxAgeSweeps) {
                                ship_aged++;
                            }
                            ship_age[c] = 0;
                            ship_list[n_ship++] = static_cast<uint8_t>(c);
                        }
                        c_proc += get_timestamp() - t_cv1;
                        if (n_ship + num_cores / 8u >= num_cores) {
                            cv_bypass = true;
                        }
                    }
                }
                // With CV-first the pipeline walks the ship set; without it, every core.
                const uint32_t n_poll = n_ship;

                auto process_batch = [&](uint32_t base_c, uint32_t n, uint32_t g) {
                    const uint64_t t_p0 = get_timestamp();
                    // c_self joins the nested term because self_publish RESTORES c_reserve/c_write, so without it a
                    // mid-batch self publish would be charged to `proc`.
                    const uint64_t flush_at = c_reserve + c_write + (kSelfZones != 0 ? c_self : 0);
                    // PROC as an ordinary RAII scope over the whole batch, so its children (the credit wait and the
                    // write inside emit_run) nest under it.
                    kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PROC, SelfMarkPhase> z_proc(
                        self_mark_phase);
                    // `frames` advances once per LIVE core, so this asks whether the batch found work without a flag
                    // in the scan loop.
                    const uint32_t frames_at_p0 = frames;
                    uint32_t run_start = 0, run_len = 0;
                    for (uint32_t i = 0; i < n; i++) {
                        const uint32_t c = ship_list[base_c + i];
                        const uint32_t sl = g * kGenSlots + i;
                        const uint32_t slot = kStageBase + sl * kSlotBytes;
                        // NON-volatile on purpose. This control vector is in STAGING -- a snapshot the bulk read landed and
                        // the read barrier waited on -- so nothing mutates it while it is scanned, and a volatile pointer
                        // would force a reload per access in the one loop that must stay in registers.
                        const tt_l1_ptr uint32_t* cv = reinterpret_cast<const tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                        uint32_t* mine = &head_mirror[c * kNumRisc];

                        if (!seeded[c]) {
                            for (uint32_t r = 0; r < kNumRisc; r++) {
                                mine[r] = cv[kernel_profiler::SPSC_RING_HEAD_0 + r];
                            }
                            seeded[c] = 1;
                        }

                        // SCAN, UNROLLED INTO REGISTERS: it is L1-access bound, not arithmetic bound, and indexed arrays
                        // spill on this core.
                        uint32_t m0 = mine[0], m1 = mine[1], m2 = mine[2], m3 = mine[3], m4 = mine[4];
                        uint32_t r0 = cv[kernel_profiler::SPSC_RING_TAIL_0 + 0] - m0;
                        uint32_t r1 = cv[kernel_profiler::SPSC_RING_TAIL_0 + 1] - m1;
                        uint32_t r2 = cv[kernel_profiler::SPSC_RING_TAIL_0 + 2] - m2;
                        uint32_t r3 = cv[kernel_profiler::SPSC_RING_TAIL_0 + 3] - m3;
                        uint32_t r4 = cv[kernel_profiler::SPSC_RING_TAIL_0 + 4] - m4;
                        if (r0 > kRingWords) {
                            overflows++;
                            r0 = kRingWords;
                        }
                        if (r1 > kRingWords) {
                            overflows++;
                            r1 = kRingWords;
                        }
                        if (r2 > kRingWords) {
                            overflows++;
                            r2 = kRingWords;
                        }
                        if (r3 > kRingWords) {
                            overflows++;
                            r3 = kRingWords;
                        }
                        if (r4 > kRingWords) {
                            overflows++;
                            r4 = kRingWords;
                        }
                        uint32_t peak = r0;
                        if (r1 > peak) {
                            peak = r1;
                        }
                        if (r2 > peak) {
                            peak = r2;
                        }
                        if (r3 > peak) {
                            peak = r3;
                        }
                        if (r4 > peak) {
                            peak = r4;
                        }
                        if (peak > max_occ) {
                            max_occ = peak;
                        }
                        const uint32_t live = r0 + r1 + r2 + r3 + r4;
                        {
                            cv_below += live < kShipMinWords ? 1u : 0u;
                        }
                        if (live == 0) {
                            emit_run(run_start, run_len);
                            run_len = 0;
                            continue;
                        }
                        // Under CV-first the ship decision was made in phase 1; a listed core ships unconditionally.
                        if (run_len == 0) {
                            run_start = sl;
                        }
                        run_len++;

                        // Head write-back releases the producer, and is safe at once: the payload is already resident in
                        // staging, so those ring slots are free regardless of when it reaches the host.
                        m0 += r0;
                        m1 += r1;
                        m2 += r2;
                        m3 += r3;
                        m4 += r4;
                        mine[0] = m0;
                        mine[1] = m1;
                        mine[2] = m2;
                        mine[3] = m3;
                        mine[4] = m4;
                        // HEAD WRITE-BACK, timed separately: `proc` is the largest busy-sweep phase and it is two unlike
                        // things -- a local scan of staged control vectors, and one 20 B noc_async_write per live core.
                        const uint64_t t_h0 = get_timestamp();
                        const uint32_t sc = kHeadScratch + hb_slot * 32u;
                        volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                        scp[0] = m0;
                        scp[1] = m1;
                        scp[2] = m2;
                        scp[3] = m3;
                        scp[4] = m4;
                        noc_async_write(
                            sc,
                            get_noc_addr(
                                coords[c] & 0xFFFFu, coords[c] >> 16, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u),
                            kNumRisc * 4u);
                        hb_slot = (hb_slot + 1u) & (kMaxCores - 1u);
                        c_ph_head += get_timestamp() - t_h0;

                        frames++;
                        total_words += live;
                    }
                    // A LIVE CORE is a filler's definition of work, and this is the first point after the scan where it
                    // can be acted on without putting anything inside the scan.
                    if constexpr (kSelfZones != 0) {
                        if (!self_on && frames != frames_at_p0) {
                            self_arm();  // opens the window at EVERY detail level
                        }
                    }
                    emit_run(run_start, run_len);
                    gen_shipped[g] = true;
                    // Publish at ISSUE time: the ack wait leaves the sweep path entirely, and the mover's stamp
                    // verification consumes only frames whose writes are visible.
                    publish_head();
                    // SATURATING: the nested ship_run time is subtracted so it is not double-counted, but an unsigned
                    // wrap here once produced "proc 18727729111430.1%".
                    {
                        const uint64_t t_p1 = get_timestamp();
                        const uint64_t span = t_p1 - t_p0;
                        const uint64_t nested = (c_reserve + c_write + (kSelfZones != 0 ? c_self : 0)) - flush_at;
                        c_proc += (span > nested) ? (span - nested) : 0;
                    }
                    // z_proc closes here: PROC spans the whole batch, i.e. c_proc plus its nested children, which is
                    // what a Tracy parent is.
                };

                for (uint32_t base_c = 0; base_c < n_poll; base_c += kGenSlots) {
                    const uint32_t n = (n_poll - base_c) < kGenSlots ? (n_poll - base_c) : kGenSlots;

                    // This generation's previous ship must be out of staging before its slots refill. SENT is enough
                    // here and only here: the next writer of a filler's staging is its own NIU read responses. It is
                    // NOT a fence against another master -- the mover's staging is refilled by the DMA engine, where
                    // a sent-based wait produced tens of thousands of decode order regressions.
                    if (gen_shipped[gen]) {
                        const uint64_t t_b0 = get_timestamp();
                        *phase = kPhBar1;
                        bool flushed;
                        {
                            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfMarkPhase> z_bar(
                                self_mark_phase);
                            flushed = write_barrier_bounded<true>(t_b0 + kCreditWaitCycles);
                        }
                        c_barrier += get_timestamp() - t_b0;
                        if (!flushed) {
                            egress_dead = true;
                            break;
                        }
                        gen_shipped[gen] = false;
                    }

                    // c_read is TWO disjoint intervals per batch -- the issue, and whatever wait survives the concurrent
                    // ship -- so it takes two zones.
                    const uint64_t t_batch0 = get_timestamp();
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ, SelfMarkPhase> z_issue(
                            self_mark_phase);
                        for (uint32_t i = 0; i < n; i++) {
                            const uint32_t xy = coords[ship_list[base_c + i]];
                            CoreLocalMem<uint32_t> dst(kStageBase + (gen * kGenSlots + i) * kSlotBytes + kPrefix * 4u);
                            // ONE read per span, NEVER split across NoCs: the shipped control vector is the span's own first
                            // bytes, and only a single ascending read guarantees every tail is captured BEFORE the data it
                            // points at. A second NoC's half completes in any order, so a tail could claim a record the
                            // other half's capture predates.
                            noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                                src, dst, kSpanBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
                        }
                    }
                    const uint64_t t_issue = get_timestamp();

                    // The overlap: these writes go out on NOC_INDEX while the reads above fly on kReadNoc.
                    if (have_pend) {
                        process_batch(pend_base, pend_n, pend_gen);
                    }

                    // Issue cost plus only the wait REMAINING after the concurrent ship. Timing to the barrier instead
                    // would swallow process_batch and double-count it against c_proc -- it did, and phases summed 133%.
                    const uint64_t t_after_proc = get_timestamp();
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ_WAIT, SelfMarkPhase> z_wait(
                            self_mark_phase);
                        noc.async_read_barrier();
                    }
                    const uint64_t t_read_end = get_timestamp();
                    c_read += (t_issue - t_batch0) + (t_read_end - t_after_proc);

                    pend_base = base_c;
                    pend_n = n;
                    pend_gen = gen;
                    have_pend = true;
                    gen ^= 1u;
                }
                if (have_pend) {
                    process_batch(pend_base, pend_n, pend_gen);
                    have_pend = false;
                }
                {
                    const uint64_t t_b0 = get_timestamp();
                    *phase = kPhBar2;
                    bool flushed;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfMarkPhase> z_bar(
                            self_mark_phase);
                        flushed = write_barrier_bounded<true>(t_b0 + kCreditWaitCycles);
                    }
                    if (!flushed) {
                        egress_dead = true;
                    } else {
                        publish_head();
                    }
                    c_barrier += get_timestamp() - t_b0;
                }
            }

            sweep_cyc = static_cast<uint32_t>(get_timestamp() - t_sweep0);
        }
        if (sweep_cyc > max_sweep) {
            max_sweep = sweep_cyc;
            ws_read = static_cast<uint32_t>(c_read - s_read0);
            ws_proc = static_cast<uint32_t>(c_proc - s_proc0);
            ws_rsv = static_cast<uint32_t>(c_reserve - s_rsv0);
            ws_wr = static_cast<uint32_t>(c_write - s_wr0);
            ws_bar = static_cast<uint32_t>(c_barrier - s_bar0);
        }
        if (frames == frames_at_sweep_start) {
            sweeps_idle++;
            c_idle += sweep_cyc;
        } else {
            c_busy += sweep_cyc;
        }
        nf_end(t_sweep0, sweep_cyc, frames != frames_at_sweep_start);
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

        // Collapse the gap on work, creep toward ~20 us when idle: widening only saves idle probe traffic,
        // and a producer must never wait on it.
        if constexpr (kRole != kRoleMover) {
            if (frames != frames_at_sweep_start) {
                gap = 0;
            } else {
                uint32_t inc = gap >> 1;
                if (inc < 256u) {
                    inc = 256u;
                }
                gap = (gap + inc > kCvIdleGapMax) ? kCvIdleGapMax : gap + inc;
            }
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

    // socket_barrier() waits for the host to ack everything, so it hangs on a dead consumer exactly as
    // the write barrier would. A FILLER has no socket, so both calls fold away.
    *phase = kPhSockBar;
    *phase = kPhBarTail;
    *phase = kPhTailBar;  // distinct from kPhBar1: the tail barrier used to run while phase still read 11
    (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles);
    // Publish the LAST staged frames. Without this the final batch is written to the ring but never announced,
    // so the mover cannot drain it and the tail of every capture is silently short by up to one sweep.
    publish_head();
    // Retire the ring HERE, not next to the flip: `done` below releases the host to set stop == 2, which
    // ends the spin at once, so a retirement published there is only microseconds ahead of the flip and a
    // mover polling at ~10 us misses it. Published before the drain wait, the mover has that whole wait
    // plus teardown to see it.
    if constexpr (kRole == kRoleFiller) {
        *hs_head = frames_staged | kHsRetireBit;
    }
    // Wait for the mover to drain what we published before reporting: exiting with a STALE tail mirror
    // makes `inflight = frames_staged - *hs_tail` overstate the ring, so a filler can wait for room
    // that is already free. Bounded on the write-barrier deadline so a dead mover cannot wedge us.
    if constexpr (kRole == kRoleFiller) {
        const uint64_t drain_deadline = get_timestamp() + kCreditWaitCycles;
        while (*hs_tail != frames_staged && get_timestamp() < drain_deadline) {
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
    // ---- role-split instrumentation ----
    out[48] = kRole;
    out[49] = frames_staged;
    out[50] = ring_hi[0];    // head - tail high-water on peer 0's ring (the filler's own ring), in frames
    out[51] = ring_blocked;  // FILLER: pushes that had to wait for ring room
    out[52] = kDramFrames;
    out[53] = *hs_tail;
    out[54] = 0;
    out[55] = 0;
    out[56] = 0;
    // MUST BE ZERO. Non-zero means the mover read something that cannot be a head, so every frame it
    // shipped from then on is suspect.
    out[57] = 0;
    // ---- PEER 1 of a dual-ring mover. Mirrors 49/53/54/55/50 above. A per-peer copy of everything the
    // verification bar checks: staged == moved must hold PER RING. ----
    out[58] = 0;
    out[59] = 0;
    out[60] = 0;
    out[61] = 0;
    out[62] = 0;
    out[63] = ring_hi[1];
    static_assert(kNPeer <= 2, "the results block only carries two peers (out[58..63])");
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
    out[172] = 0;
    out[173] = 0;
    out[174] = 0;
    out[175] = 0;
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
        kernel_profiler::SPSC_DRAIN_RESULT_WORDS >= 176,
        "the results block must hold the self-profiling, NoC-footprint, stop-drain and histogram counters");

    // ================================ INSTRUMENTATION END (results block: every counter the host reads) ====
    *phase = kPhaseExit;

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
