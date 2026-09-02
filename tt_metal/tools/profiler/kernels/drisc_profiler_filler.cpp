// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's FILLER. Each filler is resident on one DRAM bank's free DRISC and owns a
// slice of the worker grid. It polls each worker's per-RISC SPSC ring tails, gather-reads the live
// runs into packed wire frames in L1 staging, DMAs the frames into a spool ring in its own GDDR
// bank, and a non-blocking pump forwards spool bytes to the host FIFO through its own D2H socket.
// Producers are lossless: a full ring blocks the worker, so the whole pipeline is flow-controlled
// end to end and the producer stall counter is the perturbation ground truth.
//
// Wire format and placement history: tools/drisc_drain/FINDINGS.md; the retired diagnostic tiers
// and their findings: tools/drisc_drain/INSTRUMENTATION_NOTES.md.

#include "drisc_drain_common.hpp"

// ---- kernel arguments and derived constants --------------------------------------------------------

constexpr uint32_t kStageBase = get_named_compile_time_arg_val("stage_base");
constexpr uint32_t kNStage = get_named_compile_time_arg_val("n_stage");
constexpr uint32_t kHeadScratch = get_named_compile_time_arg_val("head_scratch");
constexpr uint32_t kDoneAddr = get_named_compile_time_arg_val("done_addr");
// Host writes 1 = quiesce (drain everything, every wait holds), 2 = kill switch (abandon waits, free the NIU).
constexpr uint32_t kStopAddr = get_named_compile_time_arg_val("stop_addr");
constexpr uint32_t kSocketConfigAddr = get_named_compile_time_arg_val("socket_config_addr");
constexpr uint32_t kMaxCores = get_named_compile_time_arg_val("max_cores");
static_assert(kMaxCores <= 256, "ship_list, slot_core and hot index cores as bytes");
// Static VC for PCIe pushes, spread across fillers by the host.
constexpr uint32_t kWriteVc = get_named_compile_time_arg_val("write_vc");
// Ship threshold, percent of one ring. Binds on the core's fullest LANE, not its span: the
// producer that blocks is always a single lane, and a span-percent under-reads the binding ring.
constexpr uint32_t kShipMinPct = get_named_compile_time_arg_val("ship_min_pct");
// GDDR spool ring in this DRISC's own bank; 0 bytes selects the direct-push path (frames go
// straight from staging to the host FIFO).
constexpr uint32_t kSpoolBase = get_named_compile_time_arg_val("spool_base");
constexpr uint32_t kSpoolBytes = get_named_compile_time_arg_val("spool_bytes");

constexpr uint32_t kNumRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
static_assert(kNumRisc == 5, "the control scans are unrolled for exactly five RISCs");
constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;
constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
// Slots hold a full span on purpose: every sub-span cap tried deferred whole lanes at speed and
// starved TRISC2's producer.
constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);
constexpr uint32_t kSlotBytes = kSlotWords * 4u;
constexpr uint32_t kWireCtrl = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPayloadCapWords = kSlotWords - kPrefix - kWireCtrl;
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
// Reads take the NoC the writes do not: NOC_INDEX carries egress, the other NoC carries gathers.
constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
constexpr bool kSpool = kSpoolBytes != 0;
constexpr uint8_t kDmaShip = 0;   // TX stream 0: staging -> spool
constexpr uint8_t kDmaDrain = 1;  // TX stream 1: spool -> bounce
// The TX stream status register's num_writes_outstanding field is 4 bits wide (gddr_dma_regs.h).
constexpr uint32_t kDmaOutstandingMax = 15;
// Staging layout: two-core batches in kNGens generations, one slot of CV staging, and (spool mode)
// two drain bounce buffers.
constexpr uint32_t kGenSlots = 2;
constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
constexpr uint32_t kNGens = (kNStage - 1u - kNBounce) / kGenSlots;
static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
constexpr uint32_t kCvBase = kStageBase + kNGens * kGenSlots * kSlotBytes;
constexpr uint32_t kCvReadBytes = 32;
constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
static_assert(kCvReadBytes * kMaxCores <= kSlotBytes, "CV staging must fit its slot");
// Five head words per core, padded to the NoC write alignment.
constexpr uint32_t kHeadScratchStride = 32;
// The bounces take the rest of the CV slot's space plus their own slots, split in two and
// page-rounded: wide bounces are what pull the sustained drain equilibrium below production.
constexpr uint32_t kBounceBase0 = kCvBase + kCvReadBytes * kMaxCores;
constexpr uint32_t kBounceBytes = (((kNBounce + 1u) * kSlotBytes - kCvReadBytes * kMaxCores) / 2u) & ~(kPageBytes - 1u);
static_assert(kBounceBase0 % kPageBytes == 0, "bounces start on a page");
static_assert(
    !kSpool || kBounceBase0 + kNBounce * kBounceBytes <= kStageBase + kNStage * kSlotBytes,
    "bounces must fit inside the mapped staging arena");
static_assert(!kSpool || kSpoolBytes % kPageBytes == 0, "spool wraps on pages");
constexpr uint32_t kLaneShipWords = (kRingWords * kShipMinPct) / 100u;
// Per-lane ship trigger; kCvBusyPeak is also where idle backoff must stop growing, because a head
// only reaches a producer on a ship -- backing off while lanes fill toward the trigger would blind
// the filler exactly when it is needed.
constexpr uint32_t kLaneTrigger = kRingWords / 2u;
constexpr uint32_t kCvBusyPeak = kLaneTrigger / 2u;
constexpr uint64_t kCyclesPerUs = 1350;  // DRISC wall clock at the 1.35 GHz AICLK
// Idle backoff ceiling. 20 us exceeded a lane's fill time at high rates.
constexpr uint32_t kCvIdleGapMax = 5 * kCyclesPerUs;
// Worst-case host staleness for a workload too light to reach the occupancy bands.
constexpr uint64_t kSpoolFreshCycles = 50'000 * kCyclesPerUs;
constexpr uint64_t kStopDrainCycles = 1'000'000 * kCyclesPerUs;
// How long the exit lets the posted head writes stream out; small packets leave in nanoseconds.
constexpr uint64_t kPostedDrainCycles = 1000 * kCyclesPerUs;
// How long the exit waits for the host's NIU-restore word before restoring anyway.
constexpr uint64_t kNiuRestoreWaitCycles = 10'000'000 * kCyclesPerUs;

static_assert(kSpanWords * 4u <= NOC_MAX_BURST_SIZE, "a span read must fit one NoC burst");
static_assert(kRingWords * 4u <= NOC_MAX_BURST_SIZE, "a whole-ring gather must fit one NoC burst");
static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");
// Packed-gather congruence: pads bring each run to its ring phase, and everything else -- slot
// base, payload base, wrap continuations -- must land congruent with no pad. One pad rule serves
// both hops (the gather read into staging and the frame's PCIe write).
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

// Control-vector wave: read cores [lo, hi)'s tails into CV staging, then wait until `expect`
// responses have landed since `rd0`. Counted, not barriered: gather responses in flight also bump
// the counter, which can only hand a scan stale-but-valid tails (they are monotonic).
__attribute__((always_inline)) inline void cv_wave(
    const uint64_t* core_noc, uint32_t lo, uint32_t hi, uint32_t rd0, uint32_t expect) {
    for (uint32_t i = lo; i < hi; i++) {
        noc_async_read<kCvReadBytes>(core_noc[i] + kCvReadSrcOff, kCvBase + i * kCvReadBytes, kCvReadBytes, kReadNoc);
    }
    while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED) - rd0 < expect) {
    }
    invalidate_l1_cache();
}

__attribute__((always_inline)) inline uint32_t head_scratch(uint32_t c) {
    return kHeadScratch + c * kHeadScratchStride;
}

// Posted (the barriers protect staging reuse, which a head write never touches; scratch reuse is
// safe on the slot rotation) and on the read NoC: on the egress NoC this small packet queues behind
// frame data, so head visibility inherited the PCIe tile's acceptance jitter.
__attribute__((always_inline)) inline void post_heads(const uint64_t* core_noc, uint32_t c) {
    noc_async_write_one_packet<true, true>(
        head_scratch(c), core_noc[c] + kernel_profiler::SPSC_RING_HEAD_0 * 4u, kNumRisc * 4u, kReadNoc);
}

__attribute__((always_inline)) inline uint32_t prev_gen(uint32_t g) { return g == 0u ? kNGens - 1u : g - 1u; }

// A frame occupies whole socket pages on the wire.
__attribute__((always_inline)) inline uint32_t page_round(uint32_t bytes) {
    return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u);
}

void kernel_main() {
    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // profiler_msg_t base on every worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));
    // Per-core NoC address of the profiler control block, computed once: get_noc_addr's coordinate
    // arithmetic would otherwise run at every issue site of a sweep that is instruction-stream
    // bound.
    static uint64_t core_noc[kMaxCores];
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t xy = coords[i];
        core_noc[i] = get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src);
    }
    // Resync the software NoC counter mirrors from hardware. They persist across launches on this
    // never-reset core and firmware only initialises them at boot, so a previous run that ended
    // with unacked writes would wedge this run's first barrier.
    noc_local_state_init(NOC_INDEX);
    noc_local_state_init(kReadNoc);

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    set_sender_socket_page_size(sender, kPageBytes);
    // Egress write command state, programmed once: nothing else on this core touches write_cmd_buf
    // on the egress NoC, and re-programming per push cost ~0.5 us a sweep.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    // The host's launch check polls this: a DRISC that never leaves reset would otherwise wedge every
    // producer on a full ring with no error anywhere.
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    *hb = 0;

    // Every frame's prefix is identical, and of the control words only heads, tails and the core
    // identity are staged per frame -- the rest must read zero on the wire. Written once here.
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefix + kWireCtrl; k++) {
            pfx[k] = 0;
        }
    }

    // Statics persist across launches on this core, so everything the loop trusts is re-initialised
    // explicitly.
    // Sum of a core's five tails at its last scan. Tails are monotonic, so the delta is exactly the
    // words produced in one service interval -- the growth term the ship deferral needs.
    static uint32_t tails_seen[kMaxCores];
    static uint8_t hot[kMaxCores];        // shipped real words last scan; hot + empty scan = publish lag
    static uint8_t ship_list[kMaxCores];  // this sweep's ship set, dense core indices
    // Per-slot frame geometry, written at gather issue and consumed a whole batch later by the
    // ship. Stored rather than recomputed so the two phases cannot diverge.
    static uint8_t slot_core[kNStage];
    static uint32_t slot_bytes[kNStage];  // wire bytes per staged frame, prefix included
    for (uint32_t i = 0; i < num_cores; i++) {
        hot[i] = 0;
    }
    // Seed the heads from the tails as they stand now: everything published before this launch
    // predates the capture. The scratch is the only copy of the heads: the scan reads it, the
    // issue advances it, and the posted head write ships it.
    cv_wave(core_noc, 0, num_cores, NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED), num_cores);
    for (uint32_t c = 0; c < num_cores; c++) {
        const tt_l1_ptr uint32_t* tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
        volatile tt_l1_ptr uint32_t* heads = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(head_scratch(c));
        uint32_t tsum = 0;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            heads[r] = tails[r];
            tsum += tails[r];
        }
        tails_seen[c] = tsum;
    }

    uint32_t relieved = 0;
    uint32_t sweeps = 0;
    uint32_t gap = 0;
    // Ship-threshold arming. Batching must never hold pre-burst trickle across a burst onset (a
    // pre-loaded ring tips over during the detection latency), and occupancy alone cannot tell
    // one-shot trickle from a light workload's steady sub-threshold lanes. Growth persistence can:
    // defer only after kBatchArmSweeps consecutive growing sweeps, flush after kFlushQuietSweeps
    // dead ones.
    bool grid_busy = false;
    uint32_t grow_streak = 0;
    uint32_t quiet_streak = 0;
    constexpr uint32_t kBatchArmSweeps = 3;
    constexpr uint32_t kFlushQuietSweeps = 8;
    // A head relief is one posted write; re-posting every core's head on this cadence bounds a
    // lost one to a stall instead of parking the producer for the rest of the run.
    constexpr uint32_t kHeadRefreshSweeps = 64;
    static_assert((kHeadRefreshSweeps & (kHeadRefreshSweeps - 1u)) == 0, "the refresh cadence is a mask");
    // Which staging generations may still have a ship in flight. Persists across sweeps so a
    // sweep's final ship drains under the pace gap or the next CV pass, not on its own critical
    // path.
    bool gen_shipped[kNGens] = {};

    uint32_t gen_dma_mark[kNGens] = {};
    SpoolPump<kSpoolBase, kSpoolBytes, kBounceBase0, kBounceBytes, kPageBytes, kDmaShip, kDmaDrain> pump(sender);
    bool killed = false;  // the kill switch (stop=2) broke a wait: the consumer is gone, bytes are stranded

    // Ship `count` adjacent staged slots. A staged slot is already its frame's wire image, so a
    // frame is one write (or one DMA), and the trailing page fill is never written -- the host
    // derives every offset from the control vector and reads past it.
    auto emit_slots = [&](uint32_t start, uint32_t count) {
        // Never true, and load-bearing: it hands the compiler count >= 1, which is what keeps the
        // emit loops rotated and the slot address strength-reduced (without it: a spill, a mul, and
        // +3.4% d1 stalls).
        if (count == 0) {
            return;
        }
        if constexpr (kSpool) {
            // Whole page-rounded frames, dead tail bytes included: the spool offset then advances
            // in lockstep with the FIFO write pointer, so the spool is a byte-exact image of the
            // wire and the drain needs no frame geometry at all.
            uint32_t len[kGenSlots];
            uint32_t bytes = 0;
            for (uint32_t f = 0; f < count; f++) {
                len[f] = page_round(slot_bytes[start + f]);
                bytes += len[f];
            }
            // Full spool: pump until there is room. This wait, not a drop, is the spool's
            // back-pressure -- frames stay safe in staging, the sweep slows, producers stall.
            // It holds through quiesce; only the kill switch breaks it.
            while (!pump.has_room(bytes)) {
                if (*stop == 2u) {
                    killed = true;
                    return;
                }
                invalidate_l1_cache();
                pump.pass();
            }
            // The DMA engine reads the control and length words the scalar core staged; Blackhole
            // stores can reach SRAM out of order.
            asm volatile("fence" ::: "memory");
            for (uint32_t f = 0; f < count;) {
                const uint32_t fsrc = kStageBase + (start + f) * kSlotBytes;
                // A full-span frame fills its slot exactly, so adjacent full frames are
                // wire-contiguous in staging and ship as one DMA write.
                uint32_t piece = len[f];
                uint32_t nfused = 1;
                while (f + nfused < count && piece == nfused * kSlotBytes) {
                    piece += len[f + nfused];
                    nfused++;
                }
                f += nfused;
                pump.append(fsrc, piece);
            }
            pump.rebalance();
            return;
        }
        // Direct push: reserve host FIFO credit, then write the frames straight to the host.
        uint32_t npages = 0;
        for (uint32_t f = 0; f < count; f++) {
            npages += page_round(slot_bytes[start + f]) / kPageBytes;
        }
        asm volatile("fence" ::: "memory");
        if (!reserve_pages(sender, npages, stop)) {
            killed = true;
            return;
        }
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        uint32_t wr = sender.write_ptr;
        for (uint32_t f = 0; f < count; f++) {
            const uint32_t bytes = slot_bytes[start + f];
            push_fifo(sender, kStageBase + (start + f) * kSlotBytes, wr, bytes);
            wr += page_round(bytes);
            if (wr >= fifo_size) {
                wr -= fifo_size;
            }
        }
        socket_push_pages(sender, npages);
        notify_bytes_sent(sender);
    };

    // Main loop. On stop=1, keep sweeping until one whole sweep moves nothing, so markers still in
    // worker rings ship instead of being stranded; exiting on the stop word directly is what
    // silently truncated captures.
    uint64_t stop_seen_at = 0;
    uint32_t relieved_at_stop_check = 0;
    while (true) {
        invalidate_l1_cache();
        if (*stop != 0) {
            if (stop_seen_at == 0) {
                stop_seen_at = get_timestamp();
            } else if (relieved == relieved_at_stop_check || get_timestamp() - stop_seen_at > kStopDrainCycles) {
                break;
            }
            relieved_at_stop_check = relieved;
        }
        sweeps++;
        *hb = sweeps;
        const uint32_t relieved_at_sweep_start = relieved;

        uint32_t sweep_peak = 0;
        bool sweep_grew = false;
        // Software pipeline: gather generation G on the read NoC while G^1 ships on the egress
        // side. The CV pass is pipelined into the batch flights: all tail reads issue up front,
        // the wait covers only the first chunk's responses, and the rest of the grid is scanned
        // just in time when the ship list runs low.
        //
        // No lambdas anywhere in the scan region: wrapping it in a by-reference lambda cost
        // 1-2% of sweep time in capture codegen alone, and the saturation boundary amplifies
        // that ~200x. The scan must stay in this exact compilation context.
        uint32_t gen = 0;
        uint32_t pend_n = 0;
        bool have_pend = false;
        uint32_t n_ship = 0;

        const uint32_t cv_chunk = num_cores < kGenSlots * 2u ? num_cores : kGenSlots * 2u;
        const uint32_t rd0 = NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED);
        // Responses can arrive out of order, so a counted response may belong to a
        // later core -- a chunk core then scans last sweep's tails, which are stale but
        // valid (tails are monotonic): it under-ships and catches up next visit.
        // Only the first chunk's CVs are read here; the rest of the grid's are issued
        // at the refill pause, mid-sweep, so the late scan sees tails fresh enough to
        // catch a core that started producing in this very sweep.
        cv_wave(core_noc, 0, cv_chunk, rd0, cv_chunk);
        uint32_t scan_lo = 0;
        uint32_t scan_hi = cv_chunk;
        // Stage one core's frame: write the prefix and control words locally, then
        // gather-read each live run straight to its packed wire offset. The pads bring each
        // destination to its ring phase, so read src == dst (mod 16 B) holds for every
        // piece, including a wrap split, whose continuation is congruent because the ring
        // capacity is a multiple of the alignment.
        auto issue_core = [&](uint32_t c, uint32_t sl) {
            const uint32_t slot = kStageBase + sl * kSlotBytes;
            const uint32_t xy = coords[c];
            const tt_l1_ptr uint32_t* __restrict tails =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
            volatile tt_l1_ptr uint32_t* __restrict cv =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
            // The head advance is staged here, hidden behind the NIU's acceptance of the same
            // lane's gather read; after the batch barrier only the posted head write remains on
            // the release path. Safe because nothing reads the scratch between issue and that
            // barrier.
            volatile tt_l1_ptr uint32_t* __restrict heads =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(head_scratch(c));
            uint32_t live = 0;
            uint32_t off = kPrefix + kWireCtrl;
            ncrisc_noc_read_set_state<DM_DEDICATED_NOC, false, false>(kReadNoc, read_cmd_buf, core_noc[c]);
            // The per-lane walk stays a loop, unlike the scan: lane r's bookkeeping hides
            // behind lane r-1's NIU acceptance, and unrolling front-loaded it against every
            // issue and measurably regressed.
            for (uint32_t r = 0; r < kNumRisc; r++) {
                const uint32_t tail = tails[r];
                const uint32_t head = heads[r];
                uint32_t run = tail - head;
                if (run > kRingWords) {
                    run = kRingWords;
                }
                const uint32_t start = tail - run;
                // Frames cap at the slot's payload capacity in whole lanes only: a
                // published tail is a packet boundary but an arbitrary word count is not,
                // and clamping mid-run split packets across frames and corrupted the lane
                // stream.
                uint32_t take = run;
                uint32_t pad = 0;
                const bool img = kernel_profiler::spsc_span_wrap_image(start, take, kRingWords);
                if (take != 0) {
                    pad = kernel_profiler::spsc_span_pack_pad(img ? 0u : start, off);
                    const uint32_t used = off - (kPrefix + kWireCtrl);
                    const uint32_t room = kPayloadCapWords > used + pad ? kPayloadCapWords - used - pad : 0;
                    // A ring-image ship occupies the whole ring in the slot, not its extent.
                    const uint32_t need = img ? kRingWords : take;
                    if (need > room) {
                        take = 0;
                        pad = 0;
                    }
                }
                heads[r] = head + take;
                live += take;
                cv[kernel_profiler::SPSC_WIRE_HEAD_0 + r] = start;
                cv[kernel_profiler::SPSC_WIRE_TAIL_0 + r] = start + take;
                if (take == 0) {
                    continue;
                }
                off += pad;
                const uint32_t ring_src = cv_src + (kCtrlWords + r * kRingWords) * 4u;
                const uint32_t hm = start & (kRingWords - 1u);
                if (img) {
                    // A near-full wrapping run ships as its whole ring image in one read
                    // (the decoder linearises by head with the same predicate). Never coalesce
                    // adjacent ring images into one read: it starves the producer's L1 port,
                    // up to ~70x the stall floor at five rings per read.
                    ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                        kReadNoc, read_cmd_buf, ring_src, slot + off * 4u, kRingWords * 4u);
                    off += kRingWords;
                } else if (hm + take > kRingWords) {
                    // A small wrapping run ships as the two-piece split, byte-exact: at
                    // sustained rates the image's dead remainder is most of the ring, and
                    // there the drain, not the sweep, is the binding resource.
                    const uint32_t first = kRingWords - hm;
                    ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                        kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, first * 4u);
                    ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                        kReadNoc, read_cmd_buf, ring_src, slot + (off + first) * 4u, (take - first) * 4u);
                    off += take;
                } else {
                    ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                        kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, take * 4u);
                    off += take;
                }
            }
            cv[kernel_profiler::SPSC_WIRE_XY] = xy;
            // pfx[0] is constant and staged once at init; only the payload word varies.
            volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
            pfx[1] = off - kPrefix;
            slot_bytes[sl] = off * 4u;
            slot_core[sl] = static_cast<uint8_t>(c);
        };

        // Heads go out the moment the batch's read barrier passes, not with the frame
        // emit: the payload is resident in staging once the reads land, so the producer's
        // ring slots are free regardless of when the frame reaches the host.
        auto advance_heads = [&](uint32_t n, uint32_t g) {
            for (uint32_t i = 0; i < n; i++) {
                post_heads(core_noc, slot_core[g * kGenSlots + i]);
                relieved++;
            }
        };

        auto ship_frames = [&](uint32_t n, uint32_t g) {
            emit_slots(g * kGenSlots, n);
            if constexpr (kSpool) {
                gen_dma_mark[g] = pump.dma_issued;
            }
            gen_shipped[g] = true;
        };

        uint32_t cur = 0;
        while (true) {
            for (uint32_t c = scan_lo; c < scan_hi; c++) {
                const tt_l1_ptr uint32_t* __restrict tails =
                    reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                const tt_l1_ptr uint32_t* __restrict mine =
                    reinterpret_cast<const tt_l1_ptr uint32_t*>(head_scratch(c));
                // The scan is unrolled into registers on purpose: a loop over indexed arrays
                // spills on this core, and each spilled word is an L1 round trip per core per
                // sweep.
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
                uint32_t grew = 0;
                if constexpr (kLaneShipWords != 0) {
                    const uint32_t tsum = tails[0] + tails[1] + tails[2] + tails[3] + tails[4];
                    grew = tsum - tails_seen[c];
                    tails_seen[c] = tsum;
                    sweep_grew |= grew != 0;
                }
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
                if (peak > sweep_peak) {
                    sweep_peak = peak;
                }
                if (live == 0) {
                    // A hot core scanning empty is almost always the producer's 64-word batched
                    // tail publish, not idleness. Skipping it would hand the core a two-sweep
                    // service interval, so ship it -- by issue time the in-flight tail refresh
                    // has usually crossed a publish boundary. One-shot: a genuinely idle core
                    // wastes at most one empty frame before going cold.
                    if (hot[c] == 0) {
                        continue;
                    }
                    hot[c] = 0;
                    ship_list[n_ship++] = static_cast<uint8_t>(c);
                    continue;
                }
                // Deferral must be safe against one more service interval of production, and
                // the level alone cannot promise that: a core scanned just under the threshold
                // at a high rate blows the ring-fill margin two sweeps later. `grew` is the
                // words produced in the last interval, so requiring it under the threshold too
                // bounds a deferred core at ~2x threshold next visit, while trickle cores batch
                // as before.
                if (grid_busy && stop_seen_at == 0 && peak < kLaneShipWords && grew < kLaneShipWords &&
                    peak < kLaneTrigger) {
                    continue;
                }
                hot[c] = 1;
                ship_list[n_ship++] = static_cast<uint8_t>(c);
            }
            scan_lo = scan_hi;

            while (cur < n_ship) {
                // Refill the ship list before it runs dry: the per-batch tail refresh below
                // only covers cores already on the list, so a batch issued right after a
                // dry-out scan would consume tails a whole sweep stale.
                if (scan_hi < num_cores && n_ship - cur <= kGenSlots) {
                    break;
                }

                // This generation's previous ship must be out of staging before its slots
                // refill. gen_shipped persists across sweeps, so a sweep's last ship is never
                // waited on inside its own sweep -- this is the wait that catches it if the
                // pace gap has not already drained it.
                if (gen_shipped[gen]) {
                    // Bare waits: both predicates complete on this device alone (the DMA
                    // engine's writes to GDDR, the NIU's sent counter), so no consumer
                    // state can hang them. Host-gated waits keep their bounds.
                    if constexpr (kSpool) {
                        // This generation's ship writes only: stream completion is FIFO,
                        // so outstanding <= later-issues means this generation retired.
                        const uint32_t since = pump.dma_issued - gen_dma_mark[gen];
                        const uint32_t cap = since > kDmaOutstandingMax ? kDmaOutstandingMax : since;
                        while (experimental::dma_get_writes_outstanding(kDmaShip) > cap) {
                        }
                    } else {
                        // Sent-only is legal here because the staging slots' next writer is
                        // this core's own NIU read responses.
                        while (!ncrisc_noc_nonposted_writes_sent(NOC_INDEX)) {
                        }
                    }
                    gen_shipped[gen] = false;
                }

                uint32_t n = 0;
                uint32_t slots = 0;
                while (slots < kGenSlots && cur < n_ship) {
                    issue_core(ship_list[cur], gen * kGenSlots + slots);
                    cur++;
                    n++;
                    slots++;
                }
                // Refresh the next batch's tails in the same flight: on the sweep-start
                // snapshot alone the last cores would be served a sweep stale, and the
                // scan-order-last core took all the stalls. This generation's read barrier
                // covers these reads too.
                const uint32_t nn = (n_ship - cur) < kGenSlots ? (n_ship - cur) : kGenSlots;
                for (uint32_t i = 0; i < nn; i++) {
                    const uint32_t c = ship_list[cur + i];
                    ncrisc_noc_read_set_state<DM_DEDICATED_NOC, false, false>(kReadNoc, read_cmd_buf, core_noc[c]);
                    ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                        kReadNoc, read_cmd_buf, cv_src + kCvReadSrcOff, kCvBase + c * kCvReadBytes, kCvReadBytes);
                }

                // The overlap: the previous batch ships on the egress side while this batch's
                // gather reads fly on the read NoC.
                if (have_pend) {
                    ship_frames(pend_n, prev_gen(gen));
                }
                if constexpr (kSpool) {
                    if (pump.level >= 3u) {
                        pump.pass();
                    }
                }

                // Read barrier before the heads go out. The spin doubles as the pump's slot
                // -- cycles the core burns anyway -- but only at full pressure: below it
                // the pump's GDDR reads contend with the ship DMA and the landing gathers.
                while (!ncrisc_noc_reads_flushed(kReadNoc)) {
                    if constexpr (kSpool) {
                        // Level 3 means occupancy is over the 5/8 line, so nonempty holds.
                        if (pump.level >= 3u) {
                            pump.pass();
                        }
                    }
                }
                invalidate_l1_cache();
                advance_heads(n, gen);

                pend_n = n;
                have_pend = true;
                gen = gen + 1u == kNGens ? 0u : gen + 1u;
            }
            if (cur >= n_ship && scan_hi >= num_cores) {
                if (have_pend) {
                    ship_frames(pend_n, prev_gen(gen));
                    have_pend = false;
                }
                break;
            }
            // The rest of the grid's CV reads issue HERE, mid-sweep, not at sweep start:
            // the late scan runs mid-sweep either way, and sweep-start data maximizes the
            // staleness of exactly the cores scanned last. Read now, their tails can show a
            // core that started producing during this sweep, cutting the join blind window by
            // most of a sweep. Same reads, later timing.
            cv_wave(core_noc, scan_hi, num_cores, rd0, num_cores);
            scan_hi = num_cores;
        }

        // Every issued batch has passed its read barrier by here, so each core's scratch is exactly
        // the head it was last relieved to.
        if ((sweeps & (kHeadRefreshSweeps - 1u)) == 0) {
            for (uint32_t c = 0; c < num_cores; c++) {
                post_heads(core_noc, c);
            }
        }
        // Busy sweeps below the first band skip the post-sweep pump entirely: the spool is the
        // burst absorber, and a capture that fits in it deserves pure gather.
        if constexpr (kSpool) {
            if (pump.level >= 2u || (pump.level == 1u && (sweeps & 1u) != 0) || pump.fresh_boost ||
                relieved == relieved_at_sweep_start) {
                pump.pass();
                pump.notify();
            }
        }

        // Ship-threshold arming (see the declarations above for why growth persistence, not level).
        if constexpr (kLaneShipWords != 0) {
            if (sweep_grew) {
                grow_streak++;
                quiet_streak = 0;
            } else if (++quiet_streak >= kFlushQuietSweeps) {
                grow_streak = 0;
            }
            grid_busy = grow_streak >= kBatchArmSweeps;
        }
        if constexpr (kSpool) {
            pump.freshness_tick(kSpoolFreshCycles);
        }
        // Idle pacing: collapse on work, creep toward the ceiling when idle. Live-but-untriggered
        // lanes count as work here -- a head only reaches a producer on a ship, so sleeping while
        // lanes fill toward the trigger is exactly wrong.
        if (relieved != relieved_at_sweep_start || sweep_peak >= kCvBusyPeak) {
            gap = 0;
        } else {
            uint32_t inc = gap >> 1;
            if (inc < 256u) {
                inc = 256u;
            }
            gap = (gap + inc > kCvIdleGapMax) ? kCvIdleGapMax : gap + inc;
        }
        if (gap != 0) {
            const uint64_t until = get_timestamp() + gap;
            while (get_timestamp() < until) {
                if constexpr (kSpool) {
                    pump.pass();  // idle time is drain time
                }
            }
        }
    }

    // Exit. Everything the run spooled must reach the host FIFO before the socket barrier can
    // pass; bounded, so a consumer that stopped acking strands bytes instead of wedging teardown.
    if constexpr (kSpool) {
        while (!pump.drained()) {
            pump.pass();
            // Notify per pass, not per sweep: with a host FIFO smaller than the backlog, the
            // acks that free credit only come after the host has seen the bytes.
            pump.notify();
            // The host's teardown escalates stop to 2 after its own timeout -- the close path's
            // kill switch for a drain whose consumer will never finish it.
            invalidate_l1_cache();
            if (*stop == 2u) {
                killed = true;
                break;
            }
        }
        pump.notify();
    }

    // socket_barrier waits for the host to ack everything, so it would hang on a dead consumer.
    if (!killed) {
        socket_barrier(sender);
    }
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    // The posted head write-backs are outside that barrier's predicate; drain their sent counter so
    // no unstreamed head is left behind.
    const uint64_t t_ps = get_timestamp() + kPostedDrainCycles;
    while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc)) &&
           get_timestamp() < t_ps) {
    }
    // Written back only for a live consumer: after an abandoned batch the socket's view of bytes_sent
    // is already out of sync with the host's, and the socket is being torn down either way.
    if (!killed) {
        update_socket_config(sender);
    }

    // Published last, after the socket barrier, so the host only sees `done` once every page is
    // out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = 0xD09E0000u;

    // NIU restore, on the host's word. NIU_CFG_0 persists until chip reset, so whoever set stream
    // mode owns putting it back -- and last, because the flip to NOC2AXI takes this L1 (`done`,
    // the results, bytes_acked) out of the host's view.
    const uint64_t t_end = get_timestamp() + kNiuRestoreWaitCycles;
    while (*stop != 2u && get_timestamp() < t_end) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
