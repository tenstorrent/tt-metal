// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The DRISC streaming-profiler drainer: worker SPSC rings -> DRISC -> D2H socket -> host.
//
// The DRISC is a CONDUIT. It bulk-reads a worker's whole profiler_msg_t into a staging slot and writes
// that same slot straight to the host -- it never copies the payload, never re-frames it, and authors
// exactly 7 words about it: a 2-word frame header, 5 patched heads. Identity, per-lane progress and extent
// all reach the host inside the worker's own control vector, which rides at the front of the span.
//
// ---- Why there is no copy: it was 45% of the drainer ----
//
// The previous version packed each lane's live run exactly, with CPU loads and stores out of the staged
// snapshot. Per-phase instrumentation killed it: a busy sweep cost 2,271 us against an idle sweep's 36 us,
// and 2,244 us of that was the copy -- 20.4 us to move one core's 2,490 words, i.e. ~11 CYCLES PER WORD.
// A `volatile tt_l1_ptr` word-at-a-time loop cannot be unrolled, widened or pipelined by the compiler.
// Meanwhile the socket credit wait was 0.0% and the PCIe write 0.1%.
//
// So exact packing traded a resource that cost 0.1% (PCIe bytes) for one that cost 45% (DRISC cycles).
// Shipping the raw span over-sends the dead tail of each ring, which is nearly free at the high occupancy
// bulk reading actually produces -- and it makes the payload untouched by software, which is also why the
// alignment hazard class cannot arise: both ends of the transfer are whole slots.
//
// ---- Layout: the prefix lives IN FRONT of the staging slot ----
//
//   slot = [16-word prefix][2,624-word span: 64-word control vector + 5 x 512-word rings]
//        = 2,640 words = 10,560 B = exactly 165 socket pages, so a frame never needs padding.
//
// The bulk read lands at slot+64 B, so prefix and span are contiguous and one NoC write ships the frame.
// Slots are contiguous too, so a run of adjacent cores that all have data ships as a SINGLE write.
//
// The staged span is reused by the next batch's reads, so the write must land before then -- hence one
// write barrier per batch. That costs PCIe latency rather than bandwidth, and the measured write phase was
// 0.1%, so serialising it is the cheap side of this trade.
//
// The NIU must already be in stream mode -- a DRISC in the default NOC2AXI mode cannot initiate NoC, and
// the socket config write has to be able to land in L1.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "hostdevcommon/profiler_common.h"
#include "internal/tt-1xx/risc_common.h"

// DRAIN_ON_TENSIX builds this same drain loop for a Tensix BRISC instead of a DRAM DRISC. It is a CONTROL,
// not a product path: the loop body, the staging layout and the socket protocol are byte-identical, so a
// behavioural difference between the two is attributable to the core the egress originates from and nothing
// else. Only the three DRISC-specific pieces are compiled out -- the NIU mode flip (a Tensix NIU is already
// a NoC master), the cb_interface shim (Tensix firmware defines it) and the NIU-restore tail.
#ifndef DRAIN_ON_TENSIX
#include "experimental/drisc_mode.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));
#endif

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks.
inline void write_to_host_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    while (size) {
        const uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

// Bounded replacement for socket_reserve_pages (socket_api.h), which spins on `bytes_free < num_bytes`
// with NO escape. That spin is a deadlock trap: the host writer gives up acking after its no-progress
// watchdog, and `*stop` is only re-read at the top of the sweep loop, so a drainer parked here is both
// unkillable and unfeedable -- and because the producers are lossless, the WORKLOAD hangs with it.
//
// Same credit test, three ways out: credit granted (true), host asked us to stop, or the deadline passed.
// Returning false means "ship nothing this time"; the caller drops the frame. That is the right trade --
// the heads have already been written back, so the producers keep running and only capture is lost.
inline bool reserve_pages_bounded(
    const SocketSenderInterface& socket,
    uint32_t num_pages,
    uint64_t deadline,
    volatile tt_l1_ptr uint32_t* stop) {
    const uint32_t num_bytes = num_pages * socket.page_size;
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_base_addr);
    const uint32_t acked_end = socket.bytes_acked_base_addr + socket.num_downstreams * bytes_acked_size_bytes;
    while (reinterpret_cast<uint32_t>(acked) < acked_end) {
        for (;;) {
            invalidate_l1_cache();
            // bytes_acked is never ahead of bytes_sent, so this cannot underflow
            const uint32_t bytes_free = socket.downstream_fifo_total_size - (socket.bytes_sent - *acked);
            if (bytes_free >= num_bytes) {
                break;
            }
            if (*stop != 0 || get_timestamp() >= deadline) {
                return false;
            }
        }
        acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            reinterpret_cast<uint32_t>(acked) + bytes_acked_size_bytes);
    }
    return true;
}

// Bounded noc_async_write_barrier(). Same predicate the real barrier spins on
// (ncrisc_noc_nonposted_writes_flushed: hardware NIU_MST_WR_ACK_RECEIVED == software
// noc_nonposted_writes_acked), but it gives up instead of hanging.
//
// This barrier is NOT optional bookkeeping -- it is what makes staging reuse safe. The staged span is
// overwritten by the next batch's reads, so continuing past an unflushed barrier would let staging be
// rewritten while writes are still in flight, i.e. trade a hung drainer for silently corrupt capture. So
// the caller must treat `false` as "egress is dead": stop shipping entirely and leave the loop, never as
// "carry on". That is safe precisely because it only ever fires when the consumer has already gone away.
inline bool write_barrier_bounded(
    uint64_t deadline,
    volatile tt_l1_ptr uint32_t* dbg_hw = nullptr,
    volatile tt_l1_ptr uint32_t* dbg_sw = nullptr) {
    // Bounded on ITERATIONS as well as cycles. The cycle deadline alone assumes two things that a wedged NIU
    // breaks: that get_timestamp() advances, and that the loop gets to evaluate it at all. Under slow dispatch
    // the DRISC was observed stuck here with the 50 ms deadline never firing (phase=11 forever), which can only
    // happen if control never returns from the flush check or the clock is frozen. An iteration cap escapes the
    // first case and distinguishes it from the second. MEASURED (forced JIT rebuild, 0/18 cache hits): the
    // iteration cap does NOT free it either -- so control never returns from the flush check, and the core is
    // stuck inside the NIU register read. No software bound can help; the cap stays only because a barrier
    // bounded two ways is strictly better than one bounded by a clock it has to be running to read.
    // 4M iterations is far beyond any healthy flush (worst observed is a handful).
    constexpr uint32_t kMaxSpins = 4u << 20;
    uint32_t spins = 0;
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
        invalidate_l1_cache();
        // Publish BOTH sides of the flush predicate so a wedge is diagnosable from the host without a
        // debugger. ncrisc_noc_nonposted_writes_flushed compares a HARDWARE counter against a SOFTWARE
        // mirror; if the mirror is out of sync the predicate can never come true, which looks identical to
        // "the NoC is stalled" but is a completely different bug.
        if (dbg_hw != nullptr) {
            *dbg_hw = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_WR_ACK_RECEIVED);
            *dbg_sw = noc_nonposted_writes_acked[NOC_INDEX];
        }
        if (++spins >= kMaxSpins || get_timestamp() >= deadline) {
            return false;
        }
    }
    return true;
}

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
    // Fixed inter-sweep gap in cycles. 0 = continuous. The hook a pacing controller would drive.
    constexpr uint32_t kGapCycles = get_compile_time_arg_val(9);
    // EGRESS AMPLIFIER. 1 = normal. >1 re-sends each staged frame this many times, so egress bandwidth is
    // decoupled from producer rate: the extra sends skip the read and process phases entirely. Exists to ask
    // "can PCIe egress alone hang the card?" on a drainer whose own bottleneck is read/process, not egress.
    // The host receives duplicate frames, so run it with decode OFF -- it is a stress tool, not a capture.
    constexpr uint32_t kShipRepeat = get_compile_time_arg_val(10);
    // 1 = resync the software NoC mirrors from hardware at entry (see the wedge note below). 0 = diagnostic.
    constexpr uint32_t kNocInit = get_compile_time_arg_val(11);
    // ABLATION knobs. kAblate=1 strips the drain loop down to EGRESS ONLY: no worker reads, no per-core
    // processing, just re-shipping the same pre-staged bytes. kAblateSpin stands in for the sweep so the PCIe
    // push keeps its normal cadence; kAblateSlots is how many staging slots go out per iteration.
    constexpr uint32_t kAblate = get_compile_time_arg_val(12);
    constexpr uint32_t kAblateSpin = get_compile_time_arg_val(13);
    constexpr uint32_t kAblateSlots = get_compile_time_arg_val(14);
    // How many pushes make up one ablated 'sweep'. A real busy sweep walks the grid in batches of kNStage and
    // ships each one, so 110 cores / 7 slots = 16 pushes per sweep. Shipping ONCE per iteration instead
    // under-drives egress badly (measured 4.2 GB/s vs the real 16.2) because the per-push credit wait, write
    // barrier and notify stop being amortised the way they really are.
    constexpr uint32_t kAblateBatches = get_compile_time_arg_val(15);
    // Non-zero => the host recomputed the PCIe tile encoding for THIS NoC's mirrored coordinate space.
    constexpr uint32_t kPcieEncOverride = get_compile_time_arg_val(16);

    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
    constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;  // 2,624 words = 10,496 B
    constexpr uint32_t kSpanBytes = kSpanWords * 4u;
    constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    constexpr uint32_t kSlotWords = kPrefix + kSpanWords;  // 2,640
    constexpr uint32_t kSlotBytes = kSlotWords * 4u;       // 10,560
    constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
    constexpr uint32_t kPageBytes = kPageWords * 4u;
    constexpr uint32_t kPagesPerSlot = kSlotWords / kPageWords;  // 165
    // Reads take the NoC the writes do not; NOC_INDEX (the kernel's configured NoC) carries egress.
    constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
    // Two staging generations: one fills while the other drains.
    constexpr uint32_t kGenSlots = kNStage / 2;
    static_assert(kGenSlots >= 1, "need at least one slot per staging generation");

    static_assert(kSpanBytes <= NOC_MAX_BURST_SIZE, "the fused span read must fit one NoC burst");
    static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
    static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // start of profiler_msg_t on the worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));

    // Reads on one NoC, writes on the other, so a batch of span reads can be IN FLIGHT while the previous
    // batch is pushed to the host. On a single NoC the two are serialized by the barriers however the loop
    // is arranged -- the split is what makes the overlap physically possible.
    //
    // RESYNC THE SOFTWARE NOC COUNTERS ON BOTH NOCS, ALWAYS.
    //
    // The barriers do not watch hardware -- they compare a hardware counter against a SOFTWARE MIRROR
    // (`ncrisc_noc_nonposted_writes_flushed` is `NIU_MST_WR_ACK_RECEIVED == noc_nonposted_writes_acked[noc]`).
    // Those mirrors live in this core's memory and PERSIST ACROSS KERNEL LAUNCHES. A resident drainer is
    // launched repeatedly onto a core that is never reset, so any run that ends with writes still unacked
    // leaves the mirror permanently AHEAD of hardware -- and the next kernel's first write barrier then waits
    // for an equality that can never hold. Measured: HW_ACK_RECEIVED=14768 vs SW_acked=14770, frozen, so the
    // drainer wedged in the barrier on sweep 1 and the host reported FAILED TO START. It reproduced on every
    // run until a `tt-smi -r`, which is what made it look like "the DRISC cannot run under slow dispatch".
    //
    // The comment this replaces asserted DRISC firmware runs noc_local_state_init() for every NOC. It does --
    // on FW boot, which is not the same thing as on every kernel launch. Do it here, unconditionally: it is
    // the only way to guarantee the mirrors match hardware at the start of THIS kernel, it is idempotent, and
    // the Tensix build has always needed it for the read NoC anyway (BRISC firmware inits only its own,
    // brisc.cc:385, and a stale read counter makes noc_async_read_barrier() return EARLY -- silent corruption
    // rather than a wedge).
    // kNocInit=0 (host: TT_METAL_PERF_DEBUG_NO_NOC_INIT=1) skips the resync so the wedge can be brought BACK
    // on demand. Keeping the failure reproducible on one binary is what settles "did this actually fix it" --
    // the claim that it did not was made against an already-wedged core and was wrong.
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

    // ---- live liveness window, readable by the host WHILE the loop runs ----
    //
    // The results block is only published after the loop exits, so a drainer that stops draining mid-run is
    // invisible: the host cannot tell "kernel exited" from "kernel blocked" from "kernel spinning with
    // nothing to do". These two words close that gap. `hb` advances once per sweep; `phase` records where
    // the kernel is, so a drainer parked in the unbounded credit wait (socket_reserve_pages) reads as
    // PHASE_RESERVE with a frozen hb. Both live in the 64 B pad between done and stop.
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    volatile tt_l1_ptr uint32_t* phase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 8);
    // +12/+16: the two sides of the write-barrier flush predicate, live while it spins.
    volatile tt_l1_ptr uint32_t* dbg_hw_ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 12);
    volatile tt_l1_ptr uint32_t* dbg_sw_ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 16);
    constexpr uint32_t kPhaseInit = 1, kPhasePoll = 2, kPhaseReserve = 3, kPhaseWrite = 4, kPhaseExit = 5;
    // Sub-phases of WRITE, so a stuck egress says WHICH call blocks: 6=chunked NoC write to the
    // PCIe tile, 7=socket_push_pages bookkeeping, 8=socket_notify_receiver (a PCIe write of the
    // producer pointer), 9=write issued, back in the sweep body.
    constexpr uint32_t kPhWrChunk = 6, kPhWrPush = 7, kPhWrNotify = 8, kPhWrDone = 9;
    // 10 = frame dropped (credit wait gave up); 11/12/13 = the write barriers in the sweep body,
    // which are OUTSIDE ship_run and so were invisible: a stale phase 4 after a dropped frame
    // made it look like the write path was blocking when execution had already moved on.
    constexpr uint32_t kPhDropped = 10, kPhBar1 = 11, kPhBar2 = 12, kPhBarTail = 13;
    constexpr uint32_t kPhTailBar = 15;  // the post-loop write barrier, which shared phase 11 and hid there
    // 14 = socket_barrier() in the exit tail. It had NO marker, so a kernel blocked there reported the
    // loop's stale 12 and was twice mis-diagnosed as the sweep-body write barrier. Every blocking call
    // needs its own marker or the phase word lies by omission.
    constexpr uint32_t kPhSockBar = 14;
    *hb = 0;
    *phase = kPhaseInit;

    // Every frame's prefix is IDENTICAL and the bulk read lands past it (at slot + 16 words), so it is
    // written once here and never touched again. It used to be 16 stores per core per visit.
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        pfx[1] = kSpanWords;  // constant: control vector + five whole rings
        for (uint32_t k = 2; k < kPrefix; k++) {
            pfx[k] = 0;
        }
    }

    static uint32_t head_mirror[kMaxCores * kNumRisc];
    static uint8_t seeded[kMaxCores];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
    }

    uint64_t total_words = 0;
    uint32_t pages = 0;
    uint32_t frames = 0;
    uint32_t pushes = 0;
    uint32_t sweeps = 0;
    uint32_t max_occ = 0;
    uint32_t overflows = 0;
    uint32_t hb_slot = 0;

    // ---- per-phase instrumentation (see the header: this is what found the copy) ----
    uint64_t c_read = 0;     // bulk span reads: issue + barrier
    uint64_t c_proc = 0;     // control-vector inspection, prefix + head patch, head write-back
    uint64_t c_reserve = 0;  // socket_reserve_pages -- host credit wait
    uint64_t c_write = 0;    // PCIe write + push + notify
    uint64_t c_barrier = 0;  // write barrier before staging is reused
    // `write` sub-split. It is the largest busy-sweep phase (~42%), and these three do very different
    // things: the chunked NoC write moves the bytes (and can block on command-buffer availability), while
    // push_pages is local bookkeeping and notify_receiver is a PCIe write of the producer pointer -- one per
    // push regardless of size, so it is the part that punishes small pushes.
    uint64_t c_ph_head = 0;  // the per-core head write-back inside proc (see process_batch)
    uint64_t c_wr_chunk = 0;
    uint64_t c_wr_push = 0;
    uint64_t c_wr_notify = 0;
    uint64_t c_idle = 0;
    uint64_t c_busy = 0;
    uint32_t sweeps_idle = 0;
    uint32_t max_sweep = 0;
    uint32_t max_reserve = 0;
    // Set when a bounded write barrier expires: egress is dead, so STOP SHIPPING for good.
    // Never means "continue anyway" -- staging reuse depends on that barrier having flushed.
    bool egress_dead = false;
    uint32_t credit_timeouts = 0;  // bounded credit wait expired -> frame dropped instead of deadlocking
    uint32_t dropped_frames = 0;
    // ~50 ms at 1.35 GHz. Enormously above anything healthy (worst observed credit wait is ~0.1 us), so it
    // never fires in normal operation -- it exists purely to convert "wait forever" into "lose a frame".
    constexpr uint64_t kCreditWaitCycles = 67500000ull;

    // Ship `count` adjacent slots as ONE contiguous write. They are already framed in place: nothing is
    // copied, nothing is assembled.
    // Returns false if this send was dropped (credit wait expired), so an amplified run stops repeating
    // into a consumer that is not acking instead of billing one dropped frame per repeat.
    auto ship_once = [&](uint32_t start, uint32_t count) -> bool {
        const uint32_t nbytes = count * kSlotBytes;
        const uint32_t npages = count * kPagesPerSlot;
        const uint64_t t0 = get_timestamp();
        *phase = kPhaseReserve;  // if the host sees this stuck, the credit wait is the deadlock
        const bool credited = reserve_pages_bounded(sender, npages, t0 + kCreditWaitCycles, stop);
        *phase = kPhaseWrite;
        if (!credited) {
            // The consumer is gone or wedged. DROP this frame rather than block: the heads for these slots
            // were already written back, so the producers stay unblocked and the workload runs to
            // completion. Capture is best-effort; the workload is not.
            *phase = kPhDropped;
            credit_timeouts++;
            dropped_frames += count;
            c_reserve += get_timestamp() - t0;
            return false;
        }
        const uint64_t t1 = get_timestamp();
        c_reserve += t1 - t0;
        if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
            max_reserve = static_cast<uint32_t>(t1 - t0);
        }
        // A multi-page write is one contiguous burst, so it must be split where the FIFO wraps;
        // socket_push_pages only wraps the pointer, it does not split the transfer.
        *phase = kPhWrChunk;
        const uint32_t base = kStageBase + start * kSlotBytes;
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        const uint32_t first = (sender.write_ptr + nbytes > fifo_size) ? fifo_size - sender.write_ptr : nbytes;
        write_to_host_chunked(pcie_xy_enc, base, pcie_base + sender.write_ptr, first);
        if (first < nbytes) {
            write_to_host_chunked(pcie_xy_enc, base + first, pcie_base, nbytes - first);
        }
        const uint64_t t2 = get_timestamp();
        c_wr_chunk += t2 - t1;
        *phase = kPhWrPush;
        socket_push_pages(sender, npages);
        const uint64_t t3 = get_timestamp();
        c_wr_push += t3 - t2;
        *phase = kPhWrNotify;
        socket_notify_receiver(sender);
        const uint64_t t4 = get_timestamp();
        c_wr_notify += t4 - t3;
        c_write += t4 - t1;
        *phase = kPhWrDone;
        pages += npages;
        pushes++;
        return true;
    };

    // One logical frame = kShipRepeat sends of the same staged bytes. At kShipRepeat == 1 this is exactly the
    // old ship_run. The drop/dead checks stay OUT here so a dead consumer costs the whole run's worth of
    // frames once, not once per repeat.
    auto ship_run = [&](uint32_t start, uint32_t count) {
        if (count == 0) {
            return;
        }
        if (egress_dead) {
            *phase = kPhDropped;
            dropped_frames += count;
            return;
        }
        for (uint32_t rep = 0; rep < kShipRepeat; rep++) {
            if (!ship_once(start, count) || egress_dead) {
                break;
            }
        }
    };

    // Fill the staging area once for the ablation: egress must ship deterministic bytes, and uninitialised
    // L1 would make a corruption result unreadable if we ever needed one.
    if constexpr (kAblate != 0) {
        volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase);
        const uint32_t nwords = (kAblateSlots * kSlotBytes) / 4;
        for (uint32_t i = 0; i < nwords; i++) {
            stg[i] = 0xAB000000u | (i & 0x00FFFFFFu);
        }
    }

    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && *stop == 0 && !egress_dead) {
        sweeps++;
        *hb = sweeps;
        *phase = kPhasePoll;
        const uint64_t t_sweep0 = get_timestamp();
        const uint32_t frames_at_sweep_start = frames;

        // ---- ABLATION: egress only (kAblate=1) ----
        //
        // Everything that touches the worker grid is compiled out. The staged bytes are a fixed pattern
        // written once at init and never refreshed, so the PCIe push, the socket credit loop, the write
        // barrier and the notify all run exactly as they do in a real capture while the read side does not
        // exist. A spin replaces the sweep so pushes keep their normal spacing -- without it the loop would
        // ship far faster than the real drainer and change the very thing being measured.
        //
        // This is NOT a capture: the host receives the same mock bytes forever. Run with
        // TT_METAL_PERF_DEBUG_NO_DECODE=1 and read the page/byte counters.
        if constexpr (kAblate != 0) {
            if constexpr (kAblateSpin != 0) {
                const uint64_t until = get_timestamp() + kAblateSpin;
                while (get_timestamp() < until) {
                }
            }
            for (uint32_t b = 0; b < kAblateBatches && !egress_dead; b++) {
                ship_run(0, kAblateSlots);
                frames += kAblateSlots;
                // Tick the heartbeat per PUSH, not per iteration. The host's liveness check wants movement
                // within 200 ms, and one ablated iteration is 16 pushes -- most of it legitimately parked in
                // the credit wait at 511/512 occupancy, which reads as "failed to start" if hb only moves
                // once per iteration.
                *hb = sweeps * kAblateBatches + b + 1;
            }
        } else {
        // ---- software pipeline: read generation G on kReadNoc while generation G^1 ships on NOC_INDEX ----
        //
        // Per iteration: free the generation we are about to refill (its ship was issued last iteration),
        // issue its reads, then process the PREVIOUS batch -- whose writes now fly concurrently with those
        // reads. Only then wait for the reads. The read barrier is the last thing, not the first, which is
        // the whole trick: it used to sit between the read and the ship and forced them apart.
        uint32_t gen = 0;
        uint32_t pend_base = 0, pend_n = 0, pend_gen = 0;
        bool have_pend = false;
        bool gen_shipped[2] = {false, false};

        auto process_batch = [&](uint32_t base_c, uint32_t n, uint32_t g) {
            const uint64_t t_p0 = get_timestamp();
            const uint64_t flush_at = c_reserve + c_write;
            uint32_t run_start = 0, run_len = 0;
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t c = base_c + i;
                const uint32_t sl = g * kGenSlots + i;
                const uint32_t slot = kStageBase + sl * kSlotBytes;
                // NON-volatile on purpose. This control vector is in STAGING -- a snapshot the bulk read
                // already landed and the read barrier already waited on -- so nothing mutates it while it
                // is scanned. Through a volatile pointer the compiler must issue these 10 loads strictly
                // one at a time, and this scan is the single largest busy-sweep cost (~45% of busy at 120
                // cores, vs 1.2% for the head write-back). Dropping volatile lets the loads pipeline.
                // The producing core's LIVE control vector is a different address and is still read over
                // the NoC; only the staged copy is treated as plain memory.
                const tt_l1_ptr uint32_t* cv = reinterpret_cast<const tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                uint32_t* mine = &head_mirror[c * kNumRisc];

                if (!seeded[c]) {
                    for (uint32_t r = 0; r < kNumRisc; r++) {
                        mine[r] = cv[kernel_profiler::SPSC_RING_HEAD_0 + r];
                    }
                    seeded[c] = 1;
                }

                uint32_t runs[kNumRisc];
                uint32_t live = 0;
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    uint32_t run = cv[kernel_profiler::SPSC_RING_TAIL_0 + r] - mine[r];
                    if (run > kRingWords) {
                        overflows++;  // torn snapshot: a lossless producer blocks at capacity
                        run = kRingWords;
                    }
                    if (run > max_occ) {
                        max_occ = run;
                    }
                    runs[r] = run;
                    live += run;
                }
                if (live == 0) {
                    ship_run(run_start, run_len);
                    run_len = 0;
                    continue;
                }
                if (run_len == 0) {
                    run_start = sl;
                }
                run_len++;

                // Head write-back releases the producer. Safe at once: the payload is a snapshot already
                // resident in staging, so those ring slots are free regardless of when it reaches the host.
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] += runs[r];
                }
                // HEAD WRITE-BACK, timed separately. `proc` is the largest busy-sweep phase (~46% at
                // 120 cores) and it is two very different things: a local scan of the staged control
                // vectors, and THIS -- one 20 B noc_async_write per live core per sweep, i.e. up to 120
                // separate NoC issues. This drainer is issue-bound rather than bandwidth-bound, so a
                // per-core small write is exactly the shape that hurts. Splitting it says whether to
                // attack the write-back (batch it, or write back less often) or the scan (tighten the
                // per-RISC loops), instead of guessing which half of `proc` matters.
                const uint64_t t_h0 = get_timestamp();
                const uint32_t sc = kHeadScratch + hb_slot * 32u;
                volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    scp[r] = mine[r];
                }
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
            ship_run(run_start, run_len);
            gen_shipped[g] = true;
            c_proc += (get_timestamp() - t_p0) - ((c_reserve + c_write) - flush_at);
        };

        for (uint32_t base_c = 0; base_c < num_cores; base_c += kGenSlots) {
            const uint32_t n = (num_cores - base_c) < kGenSlots ? (num_cores - base_c) : kGenSlots;

            // This generation's previous ship must have landed before its slots are refilled.
            if (gen_shipped[gen]) {
                const uint64_t t_b0 = get_timestamp();
                *phase = kPhBar1;
                const bool flushed = write_barrier_bounded(t_b0 + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack);
                c_barrier += get_timestamp() - t_b0;
                if (!flushed) {
                    egress_dead = true;
                    break;
                }
                gen_shipped[gen] = false;
            }

            const uint64_t t_batch0 = get_timestamp();
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t xy = coords[base_c + i];
                CoreLocalMem<uint32_t> dst(kStageBase + (gen * kGenSlots + i) * kSlotBytes + kPrefix * 4u);
                noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                    src, dst, kSpanBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
            }
            const uint64_t t_issue = get_timestamp();

            // The overlap: these writes go out on NOC_INDEX while the reads above fly on kReadNoc.
            if (have_pend) {
                process_batch(pend_base, pend_n, pend_gen);
            }

            // Issue cost, plus only the wait that REMAINS after the concurrent ship. Measuring
            // t_batch0..barrier would swallow process_batch and double-count it against c_proc -- which it
            // did, and the phases summed to 133%.
            const uint64_t t_after_proc = get_timestamp();
            noc.async_read_barrier();
            c_read += (t_issue - t_batch0) + (get_timestamp() - t_after_proc);

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
            if (!write_barrier_bounded(t_b0 + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack)) {
                egress_dead = true;
            }
            c_barrier += get_timestamp() - t_b0;
        }

        }

        const uint32_t sweep_cyc = static_cast<uint32_t>(get_timestamp() - t_sweep0);
        if (sweep_cyc > max_sweep) {
            max_sweep = sweep_cyc;
        }
        if (frames == frames_at_sweep_start) {
            sweeps_idle++;
            c_idle += sweep_cyc;
        } else {
            c_busy += sweep_cyc;
        }

        if constexpr (kGapCycles != 0) {
            const uint64_t until = get_timestamp() + kGapCycles;
            while (get_timestamp() < until) {
            }
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
    (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
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
    out[40] = static_cast<uint32_t>(c_ph_head & 0xFFFFFFFFu);
    out[41] = static_cast<uint32_t>(c_ph_head >> 32);

    *phase = kPhaseExit;
    // Only hand the socket back if the consumer was still alive. update_socket_config() talks to the same
    // host FIFO the credit wait just gave up on, so on a dead consumer it blocks and the kernel never
    // exits -- which strands a resident drainer on the core and forces a card reset before the next run.
    // Skipping it costs nothing: the socket is being torn down anyway.
    if (!consumer_gone) {
        update_socket_config(sender);
    }

    // Published last, after the socket barrier, so the host only sees `done` once every page is out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = 0xD09E0000u | (frames & 0xFFFFu);

#ifndef DRAIN_ON_TENSIX
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
#endif
}
