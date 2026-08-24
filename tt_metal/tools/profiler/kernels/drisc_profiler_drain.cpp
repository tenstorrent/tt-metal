// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The DRISC streaming-profiler drainer: worker SPSC rings -> DRISC -> D2H socket -> host.
//
// The DRISC is a CONDUIT. It bulk-reads a worker's whole profiler_msg_t into a staging slot and ships out
// of that same slot -- it never copies the payload, and authors exactly 8 words about it: a 2-word frame
// header (word 1 patched to the packed payload length at push), 5 patched heads. Identity, per-lane
// progress and extent all reach the host inside the worker's own control vector, which rides at the front
// of the span.
//
// ---- Why the CPU never touches the payload: a copy was 45% of the drainer ----
//
// An early version packed each lane's live run exactly, with CPU loads and stores out of the staged
// snapshot. Per-phase instrumentation killed it: a busy sweep cost 2,271 us against an idle sweep's 36 us,
// and 2,244 us of that was the copy -- 20.4 us to move one core's 2,490 words, i.e. ~11 CYCLES PER WORD.
// A `volatile tt_l1_ptr` word-at-a-time loop cannot be unrolled, widened or pipelined by the compiler.
//
// Its replacement shipped the raw span whole, which zeroed the copy -- and made host cost frames x
// 10,560 B regardless of fill. ship_once now does the exact packing the copy died for, WITHOUT the
// copy: the NIU gathers each live window straight from the staged slot into the host FIFO (layout and
// the congruence-pad rule in profiler_common.h). The saving is the dead ring bytes, so it scales
// inversely with fill: ~9% on a saturated delay-15 capture whose snapshots run ~90-96% full, up to ~30x
// on a sparse workload whose spans are nearly empty. The price is write ISSUES -- ~11 NoC writes per
// frame instead of one burst per batch, measured 6.98 us per 7-frame push at ~96% fill (~9.9 GB/s per
// mover, still above the device's ~15.9 GB/s offered aggregate, so egress stays off the critical path).
//
// ---- Layout: the prefix lives IN FRONT of the staging slot ----
//
//   slot = [16-word prefix][2,624-word span: 64-word control vector + 5 x 512-word rings]
//        = 2,640 words = 10,560 B = exactly 165 socket pages.
//
// The bulk read lands at slot+64 B, so prefix and span are contiguous and a frame stages in one piece.
// Slots are contiguous too, so a run of adjacent cores that all have data moves to a DRAM ring as a
// SINGLE write (stage_run); only the socket push gathers per frame (ship_once).
//
// The staged span is reused by the next batch's reads, so the write must land before then -- hence one
// write barrier per batch. That costs PCIe latency rather than bandwidth, and the measured write phase was
// 0.1%, so serialising it is the cheap side of this trade.
//
// The NIU must already be in stream mode -- a DRISC in the default NOC2AXI mode cannot initiate NoC, and
// the socket config write has to be able to land in L1.
//
// ---- ROLE SPLIT (compile arg 20; 0 = off, the whole file behaves exactly as above) ----
//
// The knee is set by the WORST sweep and that sweep is dominated by CREDIT-WAIT: 50-97 us of an 86-127 us
// worst sweep, against 1.3 us when the host is keeping up. It is a BURST problem, not a bandwidth one --
// sustained egress is ~1.37 GB/s aggregate (~8% of the 16.5 GB/s ceiling) but one busy sweep ships 581 KB
// in 35.2 us, which IS the ceiling. The 12 MiB host FIFO is only ~21 busy sweeps of slack, and it cannot be
// enlarged: the FW maps nwin = ceil((in_off + bytes + 64) / 2 MiB) consecutive TLB windows per socket and
// kNSockets * nwin <= 16, so 12 MiB already uses 14 of 16.
//
// So split the job across SIX DRISCs instead of making two do all of it:
//
//   FILLER (kRole=1)  sweep worker rings -> write frames into its OWN device-DRAM ring. No socket, no PCIe,
//                     no host MMIO at all. Its back-pressure is a DRAM ring of hundreds of MB, not 12 MiB.
//                     FOUR of them, each owning a quarter of the grid, because the knee is the filler's SCAN.
//   MOVER  (kRole=2)  read frames out of kNPeer DRAM rings -> push to the existing D2H socket, byte-for-byte
//                     the same protocol the full-job drainer uses, so the host path is untouched. TWO of
//                     them, each draining TWO rings, because there are only two host-facing-safe cores.
//
// One ring per filler, so the ring is structurally single-producer/single-consumer and no de-interleaving is
// ever needed -- a DUAL-RING mover is still one consumer per ring, it just visits two of them. The handshake
// is a monotonic frame count: the filler publishes `head` into a word in its own L1; the mover NoC-reads head,
// moves head-tail frames, and NoC-writes `tail` back into that same block.
// Every wait on either side is BOUNDED with a give-up -- an unbounded spin on a DRISC is unkillable and
// unfeedable, and because the producers are lossless it takes the workload down with it.
//
// A dual-ring mover walks its peers SEQUENTIALLY, each getting the WHOLE staging area, with the same
// per-push write barrier in between that a single-ring mover already had. That is why `max batch` stays at
// kNStage (7) per peer instead of halving to 3-4: splitting the staging slots between peers would only pay
// off if the two peers' egress could overlap, and it cannot -- both push into ONE socket.
//
// Placement is evidence-based, not arbitrary. Only DRAM cores in NoC row y == 0 are safe to HOST-FACE
// (FINDINGS N+29: y==0 1/75 failures vs y!=0 16/125, Fisher p ~ 0.006), which is exactly two cores on this
// part -- so the movers take those and the fillers go on y!=0 banks. That is measured to be safe for the
// FILLER duty specifically: two 25-run blocks on bank 5, N+29's worst core at 5/25 for a full-job drainer,
// produced 0/25 failures held in stream mode and 0/25 doing filler-only duty. The hazard lives in the
// egress/host-facing half.
//
// A DRAM ring slot is the same raw 2,640-word prefix+span the full-job drainer stages, so the mover copies
// whole slots out of the ring and packs only at its socket push -- every role puts one wire format on the
// socket and the host decoder cannot tell them apart.

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
#include "experimental/gddr_dma.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));
#endif

// ---- The drainer's own zone ids ---------------------------------------------------------------------
//
// ORDINARY structural zone ids with ORDINARY .tt_zone_meta records, declared here in the drain kernel's
// own translation unit exactly the way a worker kernel declares its zones. The host resolves these names
// out of THIS ELF as it loads, like any other kernel's -- there is no reserved band, no fixed id and no
// hardcoded host name table any more. (They used to be fixed values 0x7FF0..0x7FF8 in a reserved band,
// with their names registered by hand next to PRODUCER-STALL in perf_debug_profiler.cpp.)
//
// Declared inside `namespace kernel_profiler` under their original spellings so the ~20 emission sites
// below are unchanged. The names are the strings a human reads in Tracy, so they keep their old text.
namespace kernel_profiler {
TT_ZONE_DEFINE_ID(DRISC_ZONE_SWEEP, "DRISC-SWEEP");              // one whole poll sweep (the parent)
TT_ZONE_DEFINE_ID(DRISC_ZONE_READ, "DRISC-READ");                // filler: span-read ISSUE. mover: the DRAM read
TT_ZONE_DEFINE_ID(DRISC_ZONE_READ_WAIT, "DRISC-READ-WAIT");      // filler: read-barrier wait left after proc
TT_ZONE_DEFINE_ID(DRISC_ZONE_PROC, "DRISC-PROC");                // control-vector scan + head write-back
TT_ZONE_DEFINE_ID(DRISC_ZONE_CREDIT_WAIT, "DRISC-CREDIT-WAIT");  // filler: DRAM ring room. mover: socket credit
TT_ZONE_DEFINE_ID(DRISC_ZONE_WRITE, "DRISC-WRITE");              // the egress write itself
TT_ZONE_DEFINE_ID(DRISC_ZONE_WR_BARRIER, "DRISC-WR-BARRIER");    // write barrier before staging is reused
TT_ZONE_DEFINE_ID(DRISC_ZONE_PACE, "DRISC-PACE");                // the inter-sweep pacing gap
TT_ZONE_DEFINE_ID(DRISC_ZONE_SYNC, "DRISC-SYNC");                // common-trigger sync fiducial
TT_ZONE_DEFINE_ID(SPSC_DATA_ID_NOCFP, "DRISC-NOC-FOOTPRINT");    // the per-sweep NoC-counter PP_DATA sample
}  // namespace kernel_profiler

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks. The caller runs
// noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC) once per push -- the packed
// gather calls this ~11 times per frame, so a per-call init would repeat command-buffer setup that
// nothing between the calls invalidates.
inline void write_to_host_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
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
    const SocketSenderInterface& socket, uint32_t num_pages, uint64_t deadline, volatile tt_l1_ptr uint32_t* stop) {
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
        acked =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uint32_t>(acked) + bytes_acked_size_bytes);
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
    uint64_t deadline, volatile tt_l1_ptr uint32_t* dbg_hw = nullptr, volatile tt_l1_ptr uint32_t* dbg_sw = nullptr) {
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

#ifndef DRAIN_ON_TENSIX
// Bounded drain of the GDDR DMA write queue -- the stage_run analogue of write_barrier_bounded, needed at
// exactly the same points. Staging-slot reuse and publish_head both require the frame bytes to have LANDED
// in the ring, and a real filler's stage_run now moves them with the tile's own DMA engine, so the NIU
// flush predicate no longer covers them. Same two-way bound, same contract: false means egress is dead.
inline bool dma_write_barrier_bounded(uint64_t deadline) {
    constexpr uint32_t kMaxSpins = 4u << 20;
    uint32_t spins = 0;
    while (experimental::dma_get_writes_outstanding(/*stream=*/0) != 0) {
        if (++spins >= kMaxSpins || get_timestamp() >= deadline) {
            return false;
        }
    }
    return true;
}
#endif

// ---- NoC FOOTPRINT: state and the two single-copy helpers --------------------------------------------
//
// WHY THESE ARE FILE-SCOPE `static noinline` FUNCTIONS AND NOT LAMBDAS. The first version was a pair of
// capturing lambdas over `uint32_t nf_prev[2][4]` / `uint64_t nf_life[2][4]`, with the register reads written
// out as an initializer list. At -Os that inlined at all three call sites and unrolled both loops, and the
// result overflowed the DRISC code region the moment self-profiling was enabled alongside it:
//
//   drisc.elf: segment[0] [0x6660,+0x2da8) overflows region:0 limit of 0x2c00 bytes
//
// 0x2da8 = 11,688 B against an 11,264 B limit -- 424 B over. Each knob fit alone; together they did not, and
// the plots need both. What fixed it was DATA STRUCTURE, and only data structure:
//   1. A FLAT array indexed at runtime => one loop body, not 8 unrolled MMIO address computations.
//   2. The register ids in a runtime-indexed table => the loop is over data, not over code.
//   3. Taking the window base from life[] before sampling => no 64 B stack snapshot per sweep.
// `static` (file scope, not lambdas) is what lets these be ONE body with a state pointer rather than a capture
// set replicated per site, and it means that when kNocFootprint == 0 nothing references them and the compiler
// discards them -- so the OFF build is unchanged, which is what the whole feature is gated on.
//
// DO NOT ADD `noinline` HERE. It was tried and it made things WORSE: with these two functions and
// self_mark_w0 marked noinline the build was 11,216 B, and simply deleting all three attributes took it to
// 11,124 B -- 92 B SMALLER, and 104 B smaller with the per-sweep series enabled. GCC's own inlining decisions
// beat the hint at -Os here. The attribute is genuinely honoured (nm shows real out-of-line symbols), so this
// is not a case of it being ignored; it is a case of the hint being wrong. Outlining also does nothing for
// SMALL bodies: collapsing 47 inlined get_timestamp() call sites into one noinline wrapper changed the ELF by
// exactly 0 bytes, because a call sequence costs what the ~3 inlined instructions did.
//
// SAMPLED ONCE PER SWEEP INTO 64-BIT SOFTWARE ACCUMULATORS, and never differenced end to end. The NIU
// counters are 32 bits wide and they WRAP. A filler pulls one whole 10,496 B span per core per sweep -- 164
// NoC words of 64 B -- so at 30 cores it moves NIU_MST_RD_DATA_WORD_RECEIVED by 164 x 30 = 4,920 words every
// sweep, and 2^32 / 4,920 = ~873,000 sweeps to wrap. A resident filler runs 167k-379k sweeps during a single
// ~2 ms capture (FINDINGS N+41), i.e. the SAME ORDER OF MAGNITUDE as the wrap: an entry/exit difference would
// read plausibly and be silently wrong on any longer run, and would go huge on a run that wrapped once.
// Per-sweep deltas are 32-bit-safe by a wide margin -- one sweep cannot move a counter by anything near 2^32.
constexpr uint32_t kNfRdW = 0;  // NIU_MST_RD_DATA_WORD_RECEIVED       -- read bytes in, in 64 B NoC words
constexpr uint32_t kNfRdT = 1;  // NIU_MST_RD_REQ_SENT                 -- read transactions issued
constexpr uint32_t kNfWrW = 2;  // NIU_MST_NONPOSTED_WR_DATA_WORD_SENT -- write bytes out, in NoC words
constexpr uint32_t kNfWrT = 3;  // NIU_MST_NONPOSTED_WR_REQ_SENT       -- write transactions issued
constexpr uint32_t kNfN = 4;
constexpr uint32_t kNfSlots = 2 * kNfN;  // flat: noc * kNfN + k, NoC 0 first

struct NocFpState {
    uint32_t prev[kNfSlots];
    uint64_t life[kNfSlots];
    // THIS sweep's deltas, i.e. what the per-sweep PP_DATA sample ships. Free: nf_sample_regs already computes
    // each delta to fold it into life[], so keeping it costs one store and no extra register read.
    uint32_t last[kNfSlots];
    // The workload window: from the START of the first sweep that did work to the END of the last one. Same
    // work-triggered definition self-profiling settled on (FINDINGS N+41), and the whole reason there are two
    // blocks rather than one -- a drainer is resident from device open, so a lifetime figure is dominated by
    // polling OUTSIDE any workload, and blending the two is the wrong-population trap. Idle sweeps INTERLEAVED
    // inside the window are included, deliberately: their span reads are real traffic that a workload paid for.
    uint64_t win_base[kNfSlots];
    uint64_t win_last[kNfSlots];
    uint64_t win_t0;
    uint64_t win_t1;
    uint64_t cost;  // cycles this instrument spent on its own register reads -- reported, never hidden
    uint32_t win_sweep_first;
    uint32_t win_sweep_last;
    // Posted writes, sampled ONCE at entry and once at exit rather than per sweep. Every write this kernel
    // issues is non-posted (the D2H push passes posted=false, and the write barrier waits on WR_ACK which a
    // posted write never returns), so these MUST come out 0. They exist because the word totals count
    // non-posted words only: if a posted write ever appears the byte figures are UNDER-reported, and the host
    // warns instead of printing a plausible low number. Entry/exit differencing is safe here precisely
    // because the expected value is 0 -- there is nothing to wrap.
    uint32_t posted_entry[2];
    uint32_t posted_txns[2];
    bool win_open;
};

// Read all kNfSlots counters and fold the wrap-safe deltas into the 64-bit accumulators.
//
// The 32-bit subtract happens FIRST and is only then widened: the truncated difference of two 32-bit samples
// is the true delta for any delta < 2^32, whether or not the counter rolled over between them.
//
// Both NoCs are read rather than just the one this role uses. A filler reads on kReadNoc and writes on
// NOC_INDEX; a mover does both on NOC_INDEX -- but that is the thing being verified, not an assumption worth
// baking in. The zero columns ARE the measurement: a non-zero filler read count on NoC 0 would mean the
// read/write NoC split is not doing what the code claims.
static void nf_sample_regs(NocFpState* s) {
    // Runtime-indexed so the eight loads collapse into one loop body. NOC_STATUS_READ_REG resolves to a load
    // from (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(id) -- this core's own memory-mapped NIU register
    // block -- so it issues NO NoC transaction and the instrument cannot perturb what it measures.
    // Addresses come from NOC_STATUS() in noc_parameters.h and are never literals: test_cluster_bh.cpp
    // carries a misnamed 0xffb202e0 and copying it would propagate the error into a number nobody can check.
    static const uint32_t kIds[kNfN] = {
        NIU_MST_RD_DATA_WORD_RECEIVED,
        NIU_MST_RD_REQ_SENT,
        NIU_MST_NONPOSTED_WR_DATA_WORD_SENT,
        NIU_MST_NONPOSTED_WR_REQ_SENT};
    const uint64_t t0 = get_timestamp();
    for (uint32_t i = 0; i < kNfSlots; i++) {
        const uint32_t cur = NOC_STATUS_READ_REG(i / kNfN, kIds[i % kNfN]);
        const uint32_t d = cur - s->prev[i];
        s->last[i] = d;
        s->life[i] += static_cast<uint64_t>(d);
        s->prev[i] = cur;
    }
    s->cost += get_timestamp() - t0;
}

// End of one sweep: sample, then decide whether this sweep extends the workload window.
//
// Called after the sweep body and before the pacing gap. The gap issues no NoC traffic, so which side of it
// the sample falls on cannot change a byte total; taking it before means the window's DURATION is measured
// over sweeps rather than over sweeps-plus-whatever-pace-trailed-the-last-one, which is the honest
// denominator for a MB/s figure.
static void nf_sweep_end(NocFpState* s, uint32_t sweep, uint64_t t_sweep0, uint32_t sweep_cyc, bool did_work) {
    // ORDER IS THE TRICK, and it is what removes a whole 64 B stack snapshot. On entry life[] still holds the
    // through-PREVIOUS-sweep totals, so a window opening now can take its base straight from life[] -- no
    // "before" copy is needed at all. Sample AFTER that, and life[] then includes this sweep, which is exactly
    // what win_last wants. The earlier version snapshotted life[] into a local array first and was 32 B over
    // the DRISC code limit; this is both smaller and simpler for the same numbers.
    if (did_work && !s->win_open) {
        s->win_open = true;
        s->win_sweep_first = sweep;
        s->win_t0 = t_sweep0;
        for (uint32_t i = 0; i < kNfSlots; i++) {
            s->win_base[i] = s->life[i];
        }
    }
    nf_sample_regs(s);
    if (!did_work) {
        return;
    }
    s->win_sweep_last = sweep;
    s->win_t1 = t_sweep0 + sweep_cyc;
    for (uint32_t i = 0; i < kNfSlots; i++) {
        s->win_last[i] = s->life[i];
    }
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
    // PACING CONTROLLER (FINDINGS N+36). The drainer ships the WHOLE span per core per frame -- 10,560 B,
    // 165 pages -- regardless of how much of it is live, so host cost is frames x 10,560 and the fill ratio
    // is what decides bytes-per-marker. Sweeping continuously against slow producers reads spans that are
    // only ~37% live, which costs ~2x the host bytes for the same payload and is why producer stalls get
    // WORSE as producers get SLOWER. Measured at 120 cores / 6M markers: delay 20 -> 70% fill, 3,341 frames,
    // 67 MB; delay 125 -> 37% fill, 6,383 frames, 120 MB.
    //
    // So pace the sweeps: hold the inter-sweep gap wherever the spans come back >= kFillPct full. This is
    // the closed loop the fixed kGapCycles hook was always meant to drive.
    constexpr uint32_t kFillPct = get_compile_time_arg_val(17);  // 0 = controller off (fixed kGapCycles)
    constexpr uint32_t kGapMaxCycles = get_compile_time_arg_val(18);
    // EGRESS AMPLIFIER. 1 = normal. >1 re-sends each staged frame this many times, so egress bandwidth is
    // decoupled from producer rate: the extra sends skip the read and process phases entirely. Exists to ask
    // "can PCIe egress alone hang the card?" on a drainer whose own bottleneck is read/process, not egress.
    // The host receives duplicate frames, so run it with decode OFF -- it is a stress tool, not a capture.
    constexpr uint32_t kShipRepeat = get_compile_time_arg_val(10);
    // 1 = resync the software NoC mirrors from hardware at entry (see the wedge note below). 0 = diagnostic.
    constexpr uint32_t kNocInit = get_compile_time_arg_val(11);
    // Args 12..15 were the ABLATION knobs (kAblate / kAblateSpin / kAblateSlots / kAblateBatches) and are now
    // DEAD -- deliberately left in place rather than renumbered, because arg indices appear in JIT cache keys
    // and in the N+41 notes, and a silent shift would make old logs unreadable against new ones. The host
    // still passes four values here; nothing reads them.
    //
    // Removed because the technique was superseded three times over, not because it stopped working:
    //   * The ROLE SPLIT already IS the ablated case -- a MOVER has no worker grid, no per-core scan and
    //     reports `proc 0.0 us`, so a switch that strips those stages is redundant on a kernel without them.
    //   * INSTRUMENTATION replaced SUBTRACTION: per-phase self-zones plus per-NoC NIU counters separate read
    //     from write cost inside ONE run, and decomposition needs no second arm to difference against.
    //   * It could never reach where this is going: `static_assert(kSelfZones == 0 || kAblate == 0)` made it
    //     mutually exclusive with self-profiling, so it can never run with the zones or the footprint series.
    // kShipRepeat (arg 10) KEEPS the one capability that was unique to it -- zero-read egress, re-shipping
    // staged frames with no read or proc phase -- and unlike the ablation it composes with self-zones.
    // Non-zero => the host recomputed the PCIe tile encoding for THIS NoC's mirrored coordinate space.
    constexpr uint32_t kPcieEncOverride = get_compile_time_arg_val(16);

    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
    constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;  // 2,624 words = 10,496 B
    constexpr uint32_t kSpanBytes = kSpanWords * 4u;
    // Pacing-controller derived limits. NOTE the target is against LIVE capacity (the rings), not
    // kSpanWords, which also counts the 64-word control vector that ships whether or not it is live.
    constexpr uint32_t kLiveWords = kNumRisc * kRingWords;
    constexpr uint32_t kFillTarget = (kLiveWords * kFillPct) / 100u;
    // The producers are LOSSLESS and block at ring capacity, so the controller must never pace a core into
    // a stall. Above this per-RISC occupancy the gap collapses to 0 regardless of fill. The scan reads
    // the PUBLISHED tail, which lags true occupancy by up to SPSC_PUBLISH_BATCH_WORDS (the producer's
    // batched fence) -- subtract that lag so the valves fire on what the lag could be hiding, or a paced
    // slow-rate capture stalls producers the valves never saw coming (measured: delay 200 went 0 -> ~8k
    // stalls when the batch landed without this).
    constexpr uint32_t kPaceHighWater = (kRingWords * 3u) / 4u - kernel_profiler::SPSC_PUBLISH_BATCH_WORDS;
    constexpr uint32_t kPaceCritical =
        (kRingWords * 7u) / 8u - kernel_profiler::SPSC_PUBLISH_BATCH_WORDS;  // hard stop: about to block
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
    // READ-NOC SPLIT (compile arg 19; 0 = off, all reads on kReadNoc).
    //
    // The busy sweep is READ-LATENCY bound: unrolling the scan halved `proc` (42% -> 23%) and the busy
    // sweep barely moved, because the read time it had been hiding simply surfaced (28% -> 46%). The batch
    // cannot grow -- DRISC L1 holds only kNStage=7 spans of 10,560 B, so kGenSlots is 3 and just 3 cores'
    // reads are ever in flight. More generations does not help either: the read barrier is global, so it
    // waits on every outstanding read regardless.
    //
    // What is left is issuing those reads on BOTH NoCs, which doubles outstanding transactions without
    // needing more L1. Writes are only ~0.9% of the sweep, so sharing NOC_INDEX with them costs little.
    constexpr uint32_t kReadSplit = get_compile_time_arg_val(19);
    // STAGING FILL GATE (arg 39; 0 = off). Skip shipping a core whose live words are under this fraction
    // of its span's live capacity, so its words accumulate in the worker rings and the frame that finally
    // ships is FULL. Attacks the sustained ceiling at its actual currency -- frames-per-word: the mover
    // cost is per fixed-size FRAME (~1.16 us each), so at 46% fill half the mover's work moves padding
    // (FINDINGS N+65). Three overrides ship a below-threshold core anyway, each protecting an invariant:
    //   occupancy  -- any RISC past kPaceHighWater ships regardless (the valves keep seeing true occupancy:
    //                 the gate sits AFTER the peak bookkeeping, so pacing safety is untouched);
    //   latency    -- every 8th sweep is a force-ship sweep, bounding capture latency at ~8 sweeps and
    //                 batching sparse traffic (a near-idle grid ships once per 8 sweeps with 8x the fill);
    //   teardown   -- the stop-word drain ships everything (the completeness check counts every word).
    // The gate does NOT save the span read (the bulk read lands before fill is known); it saves FRAMES:
    // DRAM-ring reserves/writes on the filler and per-frame evacuation on the mover.
    constexpr uint32_t kStageMinFillPct = get_compile_time_arg_val(39);
    // MOVER TRAFFIC SHAPING (arg 40; 0 = off): cycles of deliberate pace per frame moved, applied only to
    // a productive sweep that KEPT UP (no peer backlog beyond one batch). See the controller below.
    constexpr uint32_t kMoverFrameGap = get_compile_time_arg_val(40);
    // MOVER peer slots 2..5 (args 41-56): the 6-filler shape's single mover drains six rings. Zero on the
    // default 4+2 shape and for fillers.
    constexpr uint32_t kPeerXY2 = get_compile_time_arg_val(41);
    constexpr uint32_t kPeerHsAddr2 = get_compile_time_arg_val(42);
    constexpr uint32_t kDramBank2 = get_compile_time_arg_val(43);
    constexpr uint32_t kDramAddr2 = get_compile_time_arg_val(44);
    constexpr uint32_t kPeerXY3 = get_compile_time_arg_val(45);
    constexpr uint32_t kPeerHsAddr3 = get_compile_time_arg_val(46);
    constexpr uint32_t kDramBank3 = get_compile_time_arg_val(47);
    constexpr uint32_t kDramAddr3 = get_compile_time_arg_val(48);
    constexpr uint32_t kPeerXY4 = get_compile_time_arg_val(49);
    constexpr uint32_t kPeerHsAddr4 = get_compile_time_arg_val(50);
    constexpr uint32_t kDramBank4 = get_compile_time_arg_val(51);
    constexpr uint32_t kDramAddr4 = get_compile_time_arg_val(52);
    constexpr uint32_t kPeerXY5 = get_compile_time_arg_val(53);
    constexpr uint32_t kPeerHsAddr5 = get_compile_time_arg_val(54);
    constexpr uint32_t kDramBank5 = get_compile_time_arg_val(55);
    constexpr uint32_t kDramAddr5 = get_compile_time_arg_val(56);
    constexpr uint32_t kGateForceMask = 7;  // force-ship every 8th sweep
    constexpr uint32_t kStageMinWords = (kNumRisc * kernel_profiler::PROFILER_L1_VECTOR_SIZE * kStageMinFillPct) / 100u;
    // ---- ROLE SPLIT (see the header). 0 = today's full-job drainer, and every arg below is then 0. ----
    constexpr uint32_t kRoleFull = 0, kRoleFiller = 1, kRoleMover = 2;
    constexpr uint32_t kRole = get_compile_time_arg_val(20);
    constexpr uint32_t kDramBank = get_compile_time_arg_val(21);    // allocator bank id of this ring
    constexpr uint32_t kDramAddr = get_compile_time_arg_val(22);    // bank-relative base of this ring
    constexpr uint32_t kDramFrames = get_compile_time_arg_val(23);  // ring capacity in whole frames
    constexpr uint32_t kHsAddr = get_compile_time_arg_val(24);      // FILLER: its own handshake block, in L1
    constexpr uint32_t kPeerXY = get_compile_time_arg_val(25);      // MOVER: (y<<16)|x virtual of peer 0
    constexpr uint32_t kPeerHsAddr = get_compile_time_arg_val(26);  // MOVER: peer 0's handshake block
    // MOVER, second ring. kNPeer is 1 or 2 (0 for every other role) and is a COMPILE-time value, so a
    // single-ring mover emits exactly the code it did before this existed. Args 21/22 above are peer 0's
    // (bank, addr); a filler reuses those two for its own ring.
    constexpr uint32_t kNPeer = get_compile_time_arg_val(27);
    constexpr uint32_t kPeerXY1 = get_compile_time_arg_val(28);
    constexpr uint32_t kPeerHsAddr1 = get_compile_time_arg_val(29);
    constexpr uint32_t kDramBank1 = get_compile_time_arg_val(30);
    constexpr uint32_t kDramAddr1 = get_compile_time_arg_val(31);
    // ---- DRISC SELF-PROFILING (args 32..35; 0 = off, and every use is behind `if constexpr`) ----
    //
    // The drainer emits its OWN device zones by framing them exactly like a worker span and shipping them
    // down the path it already owns: a FILLER stages the frame into its DRAM ring, a MOVER pushes it to its
    // socket. There is no side channel and no second wire format -- the frame IS a worker span (16-word
    // prefix, a 64-word control vector carrying THIS core's identity, five 512-word rings), so the host
    // decoder is untouched. Only ring 0 is ever live, because myRiscID == PROCESSOR_INDEX == 0 on a DRISC and
    // the decoder yields nothing from a ring whose tail equals its head; that wastes 4/5 of a self frame,
    // which is fine at DRISC zone volume (measured in FINDINGS).
    constexpr uint32_t kSelfZones = get_compile_time_arg_val(32);
    // ---- CONTINUOUS COVERAGE, not sampling ----
    //
    // Sampling was built first and REJECTED ON USE: at ~4.5% of busy sweeps a drainer's Tracy row is a handful
    // of disconnected zones, and the gaps between them read as idle time when they are really uncaptured
    // sweeps. An instrument that has to be caveated that way is worse than none.
    //
    // So every sweep is instrumented WHILE THE DRAINER IS DOING SOMETHING. It cannot be every sweep full stop:
    // a drainer is resident for the whole process and is >99% idle, so a filler runs 22k-380k sweeps and a
    // mover up to 946k, nearly all of them outside any workload. The window is ARMED BY WORK and held open for
    // kSelfHoldCycles of wall clock past the last work seen.
    //
    // The hold is in CYCLES, not sweeps, deliberately: the two roles' cadences differ by ~20x (a filler idles
    // at 8.5 us plus a 12.7 us pacing gap, a mover at 0.7 us with no gap at all), so a sweep-count hold would
    // mean two entirely different durations on the two roles.
    constexpr uint32_t kSelfHoldCycles = get_compile_time_arg_val(33);
    constexpr uint32_t kSelfXY = get_compile_time_arg_val(34);  // this DRISC's own virtual (y<<16)|x
    // Frame budget. With continuous coverage this is a COVERAGE LIMIT, not a safety net: frames scale with how
    // long the drainer stays busy, so on a long workload it binds and the capture covers the first N ms of it.
    // Reported loudly rather than left to be inferred from a short-looking row.
    constexpr uint32_t kSelfMaxFrames = get_compile_time_arg_val(35);
    // DETAIL LEVEL (arg 36). 0 = SWEEP + PACE only, the two depth-0 zones that make a drainer's row a complete
    // account of its cadence with no unexplained whitespace. 1 = also the per-batch child phases (read,
    // read-wait, proc, credit-wait, write, wr-barrier).
    //
    // Level 0 is ~4 markers per sweep against ~100 at level 1: 25x less injected work, and 63 sweeps fit one
    // frame instead of 2.5 -- so it is the level at which tracing EVERY sweep is nearly free. Detail is a
    // compile-time knob rather than a code change, so the cheap level stays available as the one left on.
    constexpr uint32_t kSelfDetail = get_compile_time_arg_val(36);
    // ---- NoC FOOTPRINT (compile arg 37; 0 = off, and then every line below folds away to nothing) ----
    //
    // Measures the drainer's OWN NoC traffic off its NIU MASTER counters: bytes and transactions, per NoC,
    // split read vs write. It exists because the footprint figures in FINDINGS are arithmetic over
    // `sweeps x cores x kSpanBytes`, and a counter is exactly what arithmetic like that should be replaced by.
    //
    // LOCAL LOADS ONLY, so the instrument cannot perturb the quantity it measures. NOC_STATUS_READ_REG is a
    // plain load from (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(id) -- this core's own memory-mapped NIU
    // register block -- and issues NO NoC transaction. write_barrier_bounded above already reads
    // NIU_MST_WR_ACK_RECEIVED the same way. Every address comes from NOC_STATUS() in noc_parameters.h and
    // none is ever written as a literal: test_cluster_bh.cpp carries a misnamed 0xffb202e0 and copying a
    // literal from there would propagate the error into a number nobody could check.
    constexpr uint32_t kNocFootprint = get_compile_time_arg_val(37);
    // COMMON-TRIGGER SYNC EVENT (compile arg 38; 0 = off, nothing emitted). A rendezvous at the top of the sweep
    // loop: the host parks every drainer in a TIGHT SPIN and one release makes all of them stamp the same instant.
    // The spin is essential -- a per-sweep poll would report sweep PHASE (up to ~157 us) instead of the residual.
    // Observe-to-stamp is ~5 instructions (fence, load, branch, 2 latching register reads) = O(10 ns).
    // DEFAULT OFF: a parked drainer is not draining, so the host only ever fires this after the workload.
    constexpr uint32_t kSyncEvent = get_compile_time_arg_val(38);
    // PER-SWEEP SERIES (the plot source), separate from the out[] totals above and currently FORCED OFF.
    //
    // 1 = also ship a per-sweep PP_DATA sample (the plot source) on top of the out[] totals. It FITS, but only
    // just, and the road there is worth keeping because every intuition about it was wrong. All measured on the
    // DRISC code region with zones + footprint both on, limit 11,264 B:
    //   series, first cut ..................... 11,508  (244 over)
    //   + shared self_mark_w0 prologue ........ 11,348  ( 84 over)  -- real win, -160 B
    //   + constexpr index table ............... 11,332  ( 68 over)  -- marginal, -16 B
    //   + noinline on the prologue ............ 11,332  ( 68 over)  -- ZERO effect
    //   + noinline wrapper on get_timestamp ... 11,332  ( 68 over)  -- ZERO effect, 47 sites collapsed
    //   - DELETE all three noinline attributes  11,228  (36 SPARE)  -- -104 B, the thing that actually fit it
    // So the sample fits because three `noinline` hints were REMOVED, not added. Nothing had to be sacrificed:
    // the posted-write validation and the instrument-cost timing both stay, and the four deltas stay 32-bit
    // (packing them into 16-bit pairs would have risked silent truncation on a wider filler slice).
    //
    // 36 B of headroom is not much. Before adding anything here, re-measure -- and prefer restructuring data
    // over hinting at the inliner, which is the one lesson this whole exercise supports.
    constexpr uint32_t kNocFpSeries = 1;
    constexpr bool kSelfPhases = kSelfZones != 0 && kSelfDetail != 0;
    // The self frame lives in staging slot kNStage -- one PAST every slot the drain pipeline can touch. The
    // host reserves it by passing (nstage - 1) as kNStage when this is on, so DRISC L1 does not grow and the
    // OFF build is byte-identical. A filler's pipeline only ever reaches slot 2*kGenSlots-1 and a mover's
    // batch is capped at kNStage, so nothing else can write here.
    constexpr uint32_t kSelfSlot = kNStage;
    // Indexed by peer slot. constexpr arrays of compile-time args, so nothing is loaded from memory to reach
    // them; the loop over kNPeer is a fully-known trip count.
    constexpr uint32_t kPeerXYs[6] = {kPeerXY, kPeerXY1, kPeerXY2, kPeerXY3, kPeerXY4, kPeerXY5};
    constexpr uint32_t kPeerHss[6] = {
        kPeerHsAddr, kPeerHsAddr1, kPeerHsAddr2, kPeerHsAddr3, kPeerHsAddr4, kPeerHsAddr5};
    constexpr uint32_t kPeerBanks[6] = {kDramBank, kDramBank1, kDramBank2, kDramBank3, kDramBank4, kDramBank5};
    constexpr uint32_t kPeerAddrs[6] = {kDramAddr, kDramAddr1, kDramAddr2, kDramAddr3, kDramAddr4, kDramAddr5};
    static_assert(kNPeer <= 6, "a mover drains at most six rings (see kNPeerMax on the host)");
    static_assert(kRole != kRoleMover || kNPeer >= 1, "a mover with no ring would spin forever doing nothing");
    // Handshake block: four words, 16 B apart so each is independently addressable by a 4 B NoC write
    // without a read-modify-write on its neighbours.
    //   +0  head     filler stores locally,  mover NoC-reads
    //   +16 tail     mover NoC-writes,       filler reads locally
    //   +32 probe_f  host writes a magic,    mover NoC-reads and echoes into its own L1  (proves reads work)
    //   +48 probe_m  mover NoC-writes magic, host reads it off the filler                (proves writes work)
    // Both probes exist because this is a SECOND producer/consumer path, and the last one of those silently
    // destroyed a capture (1.03M records lost, every lane's nesting corrupt) while every existing counter
    // read clean. An unverified peer-L1 coordinate would fail exactly that way: plausible counters, garbage
    // handshake. So the addressing is checked at bring-up rather than assumed.
    //
    // The magics are PER-PEER, not one constant. With four fillers a single shared magic proves only "the
    // mover read SOME filler's probe word" -- a mover whose peer-1 coordinate accidentally named peer 0's
    // filler (or any other filler) would pass. The host plants kProbeFillerMagic + <filler index> and checks
    // the echo against the index it MEANT; the mover writes back kProbeMoverMagic + <peer slot>, so the host
    // can also tell peer 0's write from peer 1's.
    constexpr uint32_t kHsHead = 0, kHsTail = 16, kHsProbeF = 32, kHsProbeM = 48;
    constexpr uint32_t kProbeMoverMagic = 0x5A0FE1EDu;
    static_assert(kRole == kRoleFull || kRole == kRoleFiller || kRole == kRoleMover, "unknown drainer role");
    static_assert(kGenSlots >= 1, "need at least one slot per staging generation");

    static_assert(kSelfZones == 0 || kSelfHoldCycles >= 1, "a 0-cycle window hold would trace nothing");
    static_assert(kSelfDetail <= 1, "detail is 0 (SWEEP + PACE) or 1 (full per-batch phases)");
    static_assert(kSelfZones == 0 || kSelfMaxFrames >= 1, "self-profiling with a 0 frame budget captures nothing");
    // The sync event's only carrier is the self-zone marker ring: it emits a DRISC_ZONE_SYNC scope through
    // the self marker path and ships it with self_publish(). With zones off there is no ring, no framed
    // prefix and no control vector, so the event would have nowhere to go and would silently measure nothing.
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
    Noc noc_b{static_cast<uint8_t>(NOC_INDEX)};  // second read NoC when kReadSplit != 0
    const uint32_t noc_index_after = noc_index;
    UnicastEndpoint src;

    // A FILLER owns no socket: nothing wrote a config into its L1, so `create_sender_socket_interface` would
    // read uninitialised L1 and `set_sender_socket_page_size` would then compute pointers from garbage and
    // scribble somewhere unrelated. Skip the whole thing rather than pass it a zeroed decoy.
    SocketSenderInterface sender;
    uint32_t pcie_xy_enc = 0;
    uint64_t pcie_base = 0;
    if constexpr (kRole != kRoleFiller) {
        sender = create_sender_socket_interface(kSocketConfigAddr);
        pcie_xy_enc = kPcieEncOverride != 0 ? kPcieEncOverride : sender.d2h.pcie_xy_enc;
        pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
        set_sender_socket_page_size(sender, kPageBytes);
    }

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
    // 16 = the DRAM ring capacity wait (a FILLER's only blocking wait, and the one c_reserve now measures).
    constexpr uint32_t kPhRingWait = 16;
    *hb = 0;
    *phase = kPhaseInit;

    // ---- role-split state ----
    //
    // The four probe/telemetry words live in the same 64 B pad behind `done` that hb/phase/dbg already use,
    // so the host can read them WHILE the loop runs -- the results block is only published on exit, which is
    // useless for verifying a handshake that has to work before any data flows.
    // PER PEER, 16 B apart: probe_f echo | first frame word | live head | live tail. Peer 0 keeps the
    // addresses it always had (+20..+32) so nothing that reads them had to move; peer 1's block is +36..+48.
    // 13 words = 52 B of the 64 B pad, so it still fits behind `done` with room to spare.
    // Peers 0/1 keep their historical done-pad addresses; peers 2-5 (6-filler shape) live in the
    // RESULTS region at out[92 + (p-2)*4 .. +3] = {probe_f echo, first frame word, live head, live tail}
    // -- the pad has no room for six 16 B blocks, and out[88..107] was free. Written live; the host
    // probe check and exit report both know this layout.
#define MV_LIVE_W(p, k) (kResultsAddr + (92u + ((p) - 2u) * 4u + (k)) * 4u)
    volatile tt_l1_ptr uint32_t* mv_probe_f[6] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 20),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 36),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(2, 0)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(3, 0)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(4, 0)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(5, 0))};
    volatile tt_l1_ptr uint32_t* mv_probe_frame[6] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 24),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 40),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(2, 1)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(3, 1)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(4, 1)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(5, 1))};
    volatile tt_l1_ptr uint32_t* mv_live_head[6] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 28),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 44),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(2, 2)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(3, 2)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(4, 2)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(5, 2))};
    volatile tt_l1_ptr uint32_t* mv_live_tail[6] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 32),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 48),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(2, 3)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(3, 3)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(4, 3)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MV_LIVE_W(5, 3))};
#undef MV_LIVE_W
    // FILLER: its own handshake block. head is stored locally and read over the NoC by its mover; tail is
    // written over the NoC by that mover and read locally here.
    volatile tt_l1_ptr uint32_t* hs_head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHsAddr + kHsHead);
    volatile tt_l1_ptr uint32_t* hs_tail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHsAddr + kHsTail);
    uint32_t frames_staged = 0;   // FILLER: frames written into DRAM (monotonic; == published head)
    uint32_t frames_flushed = 0;  // FILLER: of those, how many are barrier-flushed and safe to publish
    // MOVER, per peer ring. Nothing here may be shared between peers: one `mv_tail` for two rings would ack
    // frames on one ring that were only read from the other, i.e. hand the filler room it does not have.
    uint32_t mv_tail[6] = {};   // frames consumed out of peer p's ring (monotonic)
    uint32_t mv_moved[6] = {};  // frames shipped to the host out of peer p's ring
    uint32_t mv_max_n[6] = {};  // largest batch moved in one visit to peer p
    // head - tail high-water per ring: how much elastic buffer is REALLY used. A FILLER has one ring and
    // uses slot 0 only.
    uint32_t ring_hi[6] = {};
    uint32_t ring_blocked = 0;  // FILLER: stage_run calls that had to wait for ring room at all
    uint32_t hs_bad = 0;        // MOVER: head reads that were structurally impossible -- MUST stay 0
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
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
    }

    uint64_t total_words = 0;
    uint32_t pages = 0;
    uint32_t frames = 0;
    uint32_t pushes = 0;
    uint32_t sweeps = 0;
    uint32_t max_occ = 0;
    uint32_t sweep_max_run = 0;  // per-sweep peak occupancy, the controller's safety input
    // Starts at 0 (kGapCycles default) = sweep immediately. REJECTED ALTERNATIVE, measured: seeding this at
    // kGapMaxCycles to avoid the ramp cost moved the knee badly -- delay 15 went 0/0/0/0 -> 35/100/107/75 and
    // delay 10 roughly doubled, with ring-room waits still 0, i.e. the L1 MARKER rings filled because the
    // drainer waited up to 148 us before its first sweep. The premise ("start where it settles") was false:
    // on the synthetic workload the controller settles at 8k-34k cycles, not the ceiling -- it only pins at
    // the ceiling on a sparse real model (ResNet-50 trace, rings nearly empty). See FINDINGS §N+44.
    //
    // The ramp waste on a sparse workload is real (a filler burned ~475k-564k sweeps there against 15.4k when
    // seeded high) but must be fixed WITHOUT delaying the first sweep -- e.g. a faster ramp, or a low-fill
    // floor -- not by seeding the gap.
    uint32_t gap = kGapCycles;
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
    // Phase breakdown of the WORST sweep specifically. The knee is set by the worst sweep beating ring
    // fill time, and the worst is ~2.5x the mean (105-143 us vs ~46 us) -- averages cannot say why.
    uint32_t ws_read = 0, ws_proc = 0, ws_rsv = 0, ws_wr = 0, ws_bar = 0;
    uint32_t max_reserve = 0;
    // Set when a bounded write barrier expires: egress is dead, so STOP SHIPPING for good.
    // Never means "continue anyway" -- staging reuse depends on that barrier having flushed.
    bool egress_dead = false;
    uint32_t credit_timeouts = 0;  // bounded credit wait expired -> frame dropped instead of deadlocking
    uint32_t dropped_frames = 0;
    // ---- NoC FOOTPRINT state (all of it compiled out when kNocFootprint == 0) ----------------------------
    //
    // SAMPLED ONCE PER SWEEP INTO 64-BIT SOFTWARE ACCUMULATORS, and never differenced end to end. The NIU
    // counters are 32 bits wide and they WRAP. A filler pulls one whole 10,496 B span per core per sweep --
    // 164 NoC words of 64 B -- so at 30 cores it moves NIU_MST_RD_DATA_WORD_RECEIVED by 164 x 30 = 4,920
    // words every sweep, and 2^32 / 4,920 = ~873,000 sweeps to wrap. A resident filler runs 167k-379k sweeps
    // during a single ~2 ms capture (FINDINGS N+41), i.e. the SAME ORDER OF MAGNITUDE as the wrap: an
    // entry/exit difference would read plausibly and be silently wrong on any longer run, and would go
    // NEGATIVE-looking (huge) on a run that wrapped once. Per-sweep deltas are 32-bit-safe by a wide margin
    // because one sweep cannot move any counter by anything close to 2^32.
    //
    // Index order is fixed by kNfRdW..kNfWrT below and is shared with the host's report, which reads these
    // straight out of out[] -- so the order is part of the wire format, not an implementation detail.
    // State is a flat struct in FLAT index order (noc * kNfN + k), operated on by the file-scope nf_* helpers
    // above. Flat rather than [2][kNfN] and helpers rather than lambdas for one reason: CODE SIZE. See the
    // size note on nf_sample_regs -- the nested-array lambda version overflowed the DRISC code region by
    // 424 B once both this and self-profiling were enabled, which is the combination the plots need.
    NocFpState nf{};
    // ---- DRISC SELF-PROFILING: state, one ring, and the marker path --------------------------------------
    //
    // The self frame is a WORKER SPAN, so this is a producer of exactly the shape kernel_profiler.hpp is:
    // 2-word markers (type|zone-id , timer_low) preceded by a 1-word STICKY_TIMER whenever the wall clock's
    // high half ticks, appended into a 512-word circular ring, with monotonic head/tail word counters in the
    // control vector. The clock is the SAME register the workers use (RISCV_DEBUG_REG_WALL_CLOCK_L, read here
    // through get_timestamp()), which is why no anchor, graft or calibration is needed anywhere.
    //
    // Zones are ATOMIC RAII SCOPES (kernel_profiler::SpscZoneScope): the constructor only reads the clock
    // and the destructor ships the whole zone as one ZONE_ATOMIC packet (end + duration) -- exactly like a
    // worker's DeviceZoneScopedN, with only the transport differing. Nothing is stamped from timestamps the sweep
    // loop read for its phase counters; those counters keep their own reads, so the host cross-check
    // (out[74..84]) is now approximate to a few cycles per zone boundary rather than exact by construction.
    // The cost of that honesty is the retroactive captures: a scope constructed while instrumentation was
    // off stays off (SpscZoneScope::start_ == 0), so the sweep that DISCOVERS work is captured only from the
    // arm point onward -- the armed window makes the next sweeps whole.
    //
    // This block sits ABOVE the egress lambdas because the egress phases (CREDIT-WAIT, WRITE) are
    // themselves zones now, opened inside ship_once/stage_run through the marker path below.
    volatile tt_l1_ptr uint32_t* self_ctrl =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + kSelfSlot * kSlotBytes + kPrefix * 4u);
    volatile tt_l1_ptr uint32_t* self_ring = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kStageBase + kSelfSlot * kSlotBytes + (kPrefix + kCtrlWords) * 4u);
    uint32_t self_head = 0;           // words the host has been shown (consumer side, kept by us)
    uint32_t self_tail = 0;           // words written (producer side)
    uint32_t self_hi = 0xFFFFFFFFu;   // last wall-clock high half emitted; ~0 forces a first sticky
    uint64_t self_last_ts = 0;        // the fresh timestamp self_mark_w0 stamped last; the atomic close reads it
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
    // PHASE TOTALS OVER THE INSTRUMENTED-FROM-THE-START SWEEPS ONLY -- the cross-check the zones are worth
    // nothing without. The out[10..19] lifetime totals cannot verify a sampled instrument: they cover the whole
    // run while the zones cover ~0.5% of it. These cover exactly the sweeps the zones do, so the host can
    // compare zone durations summed out of the Tracy capture against them:
    //   sum(READ) + sum(READ-WAIT) ~= c_read delta      sum(CREDIT-WAIT) ~= c_reserve delta
    //   sum(PROC) - sum(CREDIT-WAIT) - sum(WRITE) ~= c_proc delta      sum(WR-BARRIER) ~= c_barrier delta
    // APPROXIMATE, not exact: a zone boundary is its own clock read, a few cycles away from the counter's.
    // Restricted to from-the-start sweeps because a sweep ARMED part-way through has zones for only part of it,
    // and comparing those against a whole-sweep counter delta would fail for a reason that is not a bug.
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
    // Append one 2-word marker at a FRESH timestamp, read here. Publishes first if the ring is too full to
    // hold a sticky plus a marker; a marker is only ever dropped if that publish could not free it (egress
    // dead), and then it is counted rather than lost silently.
    // Takes a RAW word0 rather than (type, zone_id) so the PP_DATA sample can share this prologue: the room
    // check, the publish-and-carry-on path and the sticky-timer refresh are the fiddly parts, and having two
    // copies of them cost 244 B of DRISC code the region does not have. Returns whether the words were
    // actually written -- a caller appending a PAYLOAD after the header must not write orphan words when the
    // header was dropped.
    // NOINLINE-by-size (the call sites guard with their own inline self_on checks -- the N+41 lesson: a call
    // that only checks a flag and returns still costs a real call in the scan loop). What staying out of
    // line buys is large: this prologue reaches self_publish(), so inlining it at ~10 sites would replicate
    // the whole publish path ~10 times.
    auto self_mark_w0 = [&](uint32_t w0) -> bool {
        if constexpr (kSelfZones == 0) {
            (void)w0;
            return false;
        } else {
            if (!self_on || self_busy) {
                return false;
            }
            // The margin is 9 words, which covers every shape that uses this: 1-word sticky + 3-word atomic zone = 4,
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
            self_last_ts = ts;
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
    // The RAII zone transports (SpscZoneScope's NowFn + CloseFn). The OPEN is just a guarded clock read
    // -- an uninstrumented sweep pays a flag check, never a call (the N+41 lesson); returning 0 keeps the
    // zone silent (mid-sweep arming: a zone opened while off stays off). _phase compiles out entirely
    // below full detail, so a detail-0 build carries no phase-zone code at all. The CLOSE is shared and
    // out of line: it reuses the whole self_mark_w0 prologue (room check, publish-and-carry-on, sticky,
    // fresh END stamp) and appends the duration word. Duration SATURATES at 0xFFFFFFFF instead of taking
    // the workers' exact legacy-pair fallback: a >3.2 s self-zone means this drainer is wedged, and the
    // fallback's code bytes are not worth carrying in this region's budget.
    auto self_now = [&]() -> uint64_t {
        if constexpr (kSelfZones == 0) {
            return 0;
        } else {
            return (!self_on || self_busy) ? 0 : get_timestamp();
        }
    };
    auto self_now_phase = [&]() -> uint64_t {
        if constexpr (!kSelfPhases) {
            return 0;
        } else {
            return (!self_on || self_busy) ? 0 : get_timestamp();
        }
    };
    auto self_zone_close = [&](uint32_t w0, uint64_t start) -> void {
        if constexpr (kSelfZones != 0) {
            if (self_mark_w0(w0)) {  // wrote w0 + end timer_low at a fresh stamp (self_last_ts)
                const uint64_t d = self_last_ts - start;
                self_ring[self_tail % kRingWords] = (d >> 32) != 0 ? 0xFFFFFFFFu : static_cast<uint32_t>(d);
                self_tail++;
            }
        } else {
            (void)w0;
            (void)start;
        }
    };
    using SelfNow = decltype(self_now);
    using SelfNowPhase = decltype(self_now_phase);
    using SelfClose = decltype(self_zone_close);
    // ~50 ms at 1.35 GHz. Enormously above anything healthy (worst observed credit wait is ~0.1 us), so it
    // never fires in normal operation -- it exists purely to convert "wait forever" into "lose a frame".
    constexpr uint64_t kCreditWaitCycles = 67500000ull;

    // Above this payload the frame ships RAW: packing costs ~10 extra NoC write issues (~0.35 us,
    // measured 6.88 vs ~4.5 us per 7-frame push), worth ~3.5 KB of transfer at the mover's ~10 GB/s --
    // so below ~2/3 fill packing wins on bytes, above it the single burst wins on issues, and the
    // mover's worst-case (full-span) cost stays exactly the raw path's.
    constexpr uint32_t kPackMaxPayload = kCtrlWords + (kLiveWords * 2u) / 3u;

    // Frame geometry, derived from the staged control vector exactly as the host re-derives it
    // (profiler_common.h), and the header patch that publishes the chosen layout. Non-volatile loads
    // on purpose, same argument as process_batch: staging is a landed, barrier-waited snapshot.
    auto pack_frame_words = [&](uint32_t slot) -> uint32_t {
        const tt_l1_ptr uint32_t* cv = reinterpret_cast<const tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
        uint32_t off = kPrefix + kCtrlWords;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
            const uint32_t run =
                kernel_profiler::spsc_span_live(cv[kernel_profiler::SPSC_RING_HEAD_0 + r], tail, kRingWords);
            if (run != 0) {
                off += kernel_profiler::spsc_span_pack_pad(tail - run, off) + run;
            }
        }
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        if (off - kPrefix > kPackMaxPayload) {
            pfx[0] = kernel_profiler::spsc_span_w0() | kernel_profiler::SPSC_SPAN_RAW_FLAG;
            pfx[1] = kSpanWords;
            return kSlotWords;
        }
        pfx[0] = kernel_profiler::spsc_span_w0();
        pfx[1] = off - kPrefix;
        return kernel_profiler::spsc_span_frame_words(off - kPrefix);
    };

    // Ship `count` adjacent staged slots as PACKED frames: per frame, the prefix + control vector and then
    // each live lane's window, NIU-gathered straight out of the slot. Nothing is copied; the dead ring
    // tails simply never ship.
    // Returns false if this send was dropped (credit wait expired), so an amplified run stops repeating
    // into a consumer that is not acking instead of billing one dropped frame per repeat.
    auto ship_once = [&](uint32_t start, uint32_t count) -> bool {
        uint32_t npages = 0;
        for (uint32_t f = 0; f < count; f++) {
            npages += pack_frame_words(kStageBase + (start + f) * kSlotBytes) / kPageWords;
        }
        // The NIU reads the patched length words; Blackhole stores can reach SRAM out of order.
        asm volatile("fence" ::: "memory");
        const uint64_t t0 = get_timestamp();
        *phase = kPhaseReserve;  // if the host sees this stuck, the credit wait is the deadlock
        bool credited;
        {
            // The credit wait, as an ordinary zone. Suppressed automatically while self_publish ships the
            // self frame through this same path (self_busy), so the self frame's own egress is never a zone.
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_CREDIT_WAIT, SelfNowPhase, SelfClose> z_credit(
                self_now_phase, self_zone_close);
            credited = reserve_pages_bounded(sender, npages, t0 + kCreditWaitCycles, stop);
        }
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
        // The egress write (gather + push + notify), as one zone; the t1..t4 reads stay for the counters.
        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfNowPhase, SelfClose> z_write(
            self_now_phase, self_zone_close);
        *phase = kPhWrChunk;
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        uint32_t wr = sender.write_ptr;
        // Every gather piece is split where the FIFO wraps; socket_push_pages only wraps the pointer, it
        // does not split a transfer. Pad and page-tail words are skipped, never written.
        auto put = [&](uint32_t src, uint32_t len) {
            const uint32_t first = (wr + len > fifo_size) ? fifo_size - wr : len;
            write_to_host_chunked(pcie_xy_enc, src, pcie_base + wr, first);
            if (first < len) {
                write_to_host_chunked(pcie_xy_enc, src + first, pcie_base, len - first);
            }
            wr += len;
            if (wr >= fifo_size) {
                wr -= fifo_size;
            }
        };
        auto skip = [&](uint32_t len) {
            wr += len;
            if (wr >= fifo_size) {
                wr -= fifo_size;
            }
        };
        for (uint32_t f = 0; f < count; f++) {
            const uint32_t slot = kStageBase + (start + f) * kSlotBytes;
            const tt_l1_ptr uint32_t* cv = reinterpret_cast<const tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
            if (reinterpret_cast<const tt_l1_ptr uint32_t*>(slot)[0] & kernel_profiler::SPSC_SPAN_RAW_FLAG) {
                put(slot, kSlotBytes);
                continue;
            }
            put(slot, (kPrefix + kCtrlWords) * 4u);
            uint32_t off = kPrefix + kCtrlWords;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
                const uint32_t run =
                    kernel_profiler::spsc_span_live(cv[kernel_profiler::SPSC_RING_HEAD_0 + r], tail, kRingWords);
                if (run == 0) {
                    continue;
                }
                const uint32_t pad = kernel_profiler::spsc_span_pack_pad(tail - run, off);
                skip(pad * 4u);
                const uint32_t hm = (tail - run) & (kRingWords - 1u);
                const uint32_t ring_l1 = slot + (kPrefix + kCtrlWords + r * kRingWords) * 4u;
                const uint32_t chunk = run <= kRingWords - hm ? run : kRingWords - hm;
                put(ring_l1 + hm * 4u, chunk * 4u);
                if (chunk < run) {
                    put(ring_l1, (run - chunk) * 4u);
                }
                off += pad + run;
            }
            skip((kernel_profiler::spsc_span_frame_words(off - kPrefix) - off) * 4u);
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

    // FILLER egress: the same `count` adjacent, already-framed slots, but written into this filler's DRAM
    // ring instead of to the host. Structurally identical to ship_run -- wait for room, one write, account
    // the phases -- and deliberately so, because that shape is what the WORST-sweep breakdown is built on.
    //
    // c_reserve stays pointed at the blocking wait (now ring room rather than socket credit), so the host's
    // "WORST sweep = read + proc + credit-wait + write + wr-barrier" line keeps meaning the same thing.
    // Getting that wrong would hide the very quantity this change exists to move.
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
                kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_CREDIT_WAIT, SelfNowPhase, SelfClose>
                    z_credit(self_now_phase, self_zone_close);
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
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfNowPhase, SelfClose> z_write(
                self_now_phase, self_zone_close);
            *phase = kPhWrChunk;
            const uint32_t src = kStageBase + start * kSlotBytes;
            const uint32_t slot0 = frames_staged % kDramFrames;
            // The ring is a whole number of frames, so a FRAME never straddles the wrap -- but a RUN of adjacent
            // frames can, and socket_push_pages' trick of only wrapping a pointer does not apply here. Split it,
            // exactly as ship_once splits at the FIFO wrap.
            const uint32_t first = (slot0 + count > kDramFrames) ? (kDramFrames - slot0) : count;
#ifdef DRAIN_ON_TENSIX
            // A Tensix control build has no GDDR engine and its ring is a remote bank anyway.
            noc_async_write(
                src,
                get_noc_addr_from_bank_id<true>(kDramBank, kDramAddr + slot0 * kSlotBytes, NOC_INDEX),
                first * kSlotBytes,
                NOC_INDEX);
            if (first < count) {
                noc_async_write(
                    src + first * kSlotBytes,
                    get_noc_addr_from_bank_id<true>(kDramBank, kDramAddr, NOC_INDEX),
                    (count - first) * kSlotBytes,
                    NOC_INDEX);
            }
#else
            // The filler LIVES on the DRAM tile fronting this ring's GDDR channel, so the L1 -> ring hop uses
            // the tile's own DMA engine (gddr_dma.h, tensor_prefetcher's path) instead of looping the payload
            // out through the NoC port and straight back. That keeps the NIU for what actually needs it --
            // span reads from workers and posted head write-backs. The DMA takes the CHANNEL-LOCAL byte
            // address: the same bank-relative kDramAddr the NoC helper took, plus the same per-bank offset
            // get_noc_addr_from_bank_id<true> adds. One descriptor moves at most 262,128 B (14-bit size field
            // in 16 B words), so chunk at the largest whole-frame multiple: a run can exceed it in the
            // 4-filler shape (30 slots x 10,560 B). dma_async_write itself spins for a free queue slot (15
            // outstanding max), unbounded -- acceptable for the same reason write_barrier_bounded's iteration
            // cap documents: no software bound frees a wedged engine, and the barrier every generation keeps
            // the queue depth at a handful.
            static_assert(kSlotBytes % 16u == 0, "GDDR DMA moves 16 B multiples");
            static_assert(kStageBase % 16u == 0, "GDDR DMA source must be 16 B aligned");
            constexpr uint32_t kDmaChunkBytes = (262128u / kSlotBytes) * kSlotBytes;
            const uint64_t ring_base = static_cast<uint64_t>(kDramAddr) + bank_to_dram_offset[kDramBank];
            auto dma_push = [](uint32_t src_l1, uint64_t dst_gddr, uint32_t bytes) {
                while (bytes != 0) {
                    const uint32_t n = bytes > kDmaChunkBytes ? kDmaChunkBytes : bytes;
                    experimental::dma_async_write(/*stream=*/0, src_l1, dst_gddr, n);
                    src_l1 += n;
                    dst_gddr += n;
                    bytes -= n;
                }
            };
            dma_push(src, ring_base + slot0 * kSlotBytes, first * kSlotBytes);
            if (first < count) {
                dma_push(src + first * kSlotBytes, ring_base, (count - first) * kSlotBytes);
            }
#endif
            const uint64_t t2 = get_timestamp();
            c_wr_chunk += t2 - t1;
            c_write += t2 - t1;
            *phase = kPhWrDone;
            frames_staged += count;
            pages += count * kPagesPerSlot;
            pushes++;
        }
    };

    // Publish head, but only as far as the write barrier has actually FLUSHED. The mover reads whatever this
    // says and immediately treats those frames as complete, so publishing an unflushed frame hands it bytes
    // that are still in flight -- the same corruption trade the staging-reuse barrier exists to prevent.
    // Called only from the success path of a bounded barrier.
    auto publish_head = [&]() {
        if constexpr (kRole == kRoleFiller) {
            if (frames_flushed != frames_staged) {
                frames_flushed = frames_staged;
                *hs_head = frames_flushed;
            }
        }
    };

    // One bounded barrier for "the frames this role issued have LANDED": the NIU flush covers NoC writes
    // (mover socket pushes, Tensix-control staging writes), and a real filler additionally drains the GDDR
    // DMA queue its stage_run now writes through -- invisible to the NIU predicate.
    auto egress_barrier = [&](uint64_t deadline) -> bool {
        bool ok = write_barrier_bounded(deadline, dbg_hw_ack, dbg_sw_ack);
#ifndef DRAIN_ON_TENSIX
        if constexpr (kRole == kRoleFiller) {
            ok = ok && dma_write_barrier_bounded(deadline);
        }
#endif
        return ok;
    };

    // What process_batch calls. One name, so the sweep body is byte-identical across roles.
    auto emit_run = [&](uint32_t start, uint32_t count) {
        if constexpr (kRole == kRoleFiller) {
            stage_run(start, count);
        } else {
            ship_run(start, count);
        }
    };

    // Publish the ring's live window as one self frame, then wait for it to land.
    //
    // The barrier is at the END, and that placement is the correctness argument: after a publish the live
    // window is empty, so the very next marker overwrites a word the in-flight frame is still shipping. The
    // reverse order (barrier before the next publish) would leave exactly that hole. The fence is the same
    // one publish_tail() needs -- the NIU reads L1, and Blackhole stores can reach SRAM out of order, so the
    // marker stores must be ordered before the write that ships them.
    //
    // Every counter the existing phase breakdown is built on is SAVED AND RESTORED around the egress call.
    // The self frame really does go through stage_run/ship_run -- that is the whole point, one wire format and
    // one ring protocol -- but if its credit wait, write and pages leaked into c_reserve/c_write/pages then
    // turning this feature on would silently change the numbers it exists to explain. frames_staged,
    // dropped_frames, ring_hi, ring_blocked and egress_dead are deliberately NOT restored: those describe the
    // ring and the consumer, and a self frame occupies a real ring slot.
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
            if (egress_barrier(get_timestamp() + kCreditWaitCycles)) {
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

    // Wire the publish trampoline (declared next to self_mark_w0, far above): captureless, so it converts
    // to a plain function pointer. From here on a full self ring publishes itself mid-sweep, exactly as the
    // old direct call did.
    self_pub.ctx = static_cast<void*>(&self_publish);
    self_pub.fn = [](void* p) { (*static_cast<decltype(self_publish)*>(p))(); };
    // One VARIABLE-LENGTH PP_DATA packet carrying this sweep's NoC counter deltas: 1 header + 1 timestamp +
    // SPSC_NOCFP_WORDS payload. The payload layout is the shared contract in profiler_common.h, not a local
    // convention -- see SpscNocFpWord there for why one timestamp covers all four values.
    //
    // Needs BOTH knobs: the sample rides the self-zone marker stream, so TT_METAL_PERF_DEBUG_NOC_FOOTPRINT
    // alone produces the out[] totals but NO per-sweep series and hence no plots.
    //
    // The kRingWords - 9 headroom check already covers the worst case here: 1 sticky + 1 header +
    // 1 timestamp + 1 size word + 4 payload = 8 words.
    auto self_nocfp = [&]() {
        if constexpr (kSelfZones == 0 || kNocFootprint == 0) {
            return;
        } else {
            // Header + timestamp through the shared prologue (which reads the clock itself -- this sample is
            // a POINT marker stamped at emission, right after the SWEEP zone closes); the payload only if
            // that actually landed. PP_DATA is 3 + N words now: word0 carries the full 27-bit structural id
            // (so this sample is named from the ELF like a zone), and the payload LENGTH moved out to its
            // own word2, written here between the timestamp and the payload.
            if (!self_mark_w0(kernel_profiler::spsc_data_w0(kernel_profiler::SPSC_DATA_ID_NOCFP))) {
                return;
            }
            self_ring[self_tail % kRingWords] = kernel_profiler::spsc_data_w2(kernel_profiler::SPSC_NOCFP_WORDS);
            self_tail++;
            // ROLE-SPECIALISED, read straight out of nf.last[] at compile-time indices -- no intermediate
            // array, which is where the last of the code budget went. A filler reads on kReadNoc and writes on
            // NOC_INDEX; a mover does both on NOC_INDEX. The four counters this role does NOT use are the
            // provable zeros (out[] still ships all eight, which is where that proof is kept).
            constexpr uint32_t kRd = ((kRole == kRoleMover) ? uint32_t{NOC_INDEX} : uint32_t{kReadNoc}) * kNfN;
            constexpr uint32_t kWr = uint32_t{NOC_INDEX} * kNfN;
            // A constexpr INDEX table, not a value array: the four slots to ship are known at compile time, so
            // this lands in .rodata and the loop copies nf.last[] -> ring directly. Building a value array on
            // the stack first cost ~40 B of code for nothing.
            static constexpr uint32_t kIdx[kernel_profiler::SPSC_NOCFP_WORDS] = {
                kRd + kNfRdW, kRd + kNfRdT, kWr + kNfWrW, kWr + kNfWrT};
            for (uint32_t i = 0; i < kernel_profiler::SPSC_NOCFP_WORDS; i++) {
                self_ring[self_tail % kRingWords] = nf.last[kIdx[i]];
                self_tail++;
            }
        }
    };
    // TURN INSTRUMENTATION ON MID-SWEEP, at the moment work is discovered.
    //
    // This is what makes the feature catch bursts instead of the idle loop. Deciding at the top of a sweep
    // cannot work -- whether a sweep has anything to do is not knowable until the head read (mover) or the
    // control-vector scan (filler) says so, and arming from history alone misses the first sweep of every
    // burst. Arming sets self_on, so every zone SCOPE CONSTRUCTED FROM HERE ON in this sweep records; the
    // scopes already open (the SWEEP scope, a filler's PROC scope) were constructed while off and stay off
    // (SpscZoneScope::start_ == 0) -- there is deliberately no retroactive back-fill any more, because a zone's
    // timestamps are its scope's own clock reads. The discovery sweep is therefore captured only from the
    // arm point onward (its egress children -- CREDIT-WAIT, WRITE, WR-BARRIER -- land parentless at depth
    // 0); the hold window makes every LATER sweep of the burst whole. That asymmetry is real and is stated
    // in FINDINGS rather than hidden.
    auto self_arm = [&]() {
        if constexpr (kSelfZones == 0) {
            return;
        } else {
            self_work = true;
            // Hold the window open from here, refreshed on every later discovery of work, so a burst keeps it open
            // and the coverage inside it is CONTIGUOUS -- the entire point of tracing over sampling.
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

    // ---- MOVER bring-up probe: prove BOTH directions of the peer-L1 handshake before any data moves ----
    //
    // A wrong peer coordinate or a wrong L1 address does not fail loudly -- it reads a plausible-looking
    // garbage `head`, and the mover then ships whatever DRAM happens to contain. So read the magic the host
    // planted in the filler's probe_f word and echo it into our own L1, and write our own magic into the
    // filler's probe_m word. The host checks both during the heartbeat verify and refuses the run if either
    // is wrong. probe_m is a separate word from tail on purpose: writing a magic into TAIL would make the
    // filler read an enormous consumed-count and overwrite frames.
    if constexpr (kRole == kRoleMover) {
        // Both directions, for EVERY peer. A dual-ring mover has two independent chances to be pointed at the
        // wrong L1, and the failure mode is silent either way.
        for (uint32_t p = 0; p < kNPeer; p++) {
            const uint32_t pxy = kPeerXYs[p];
            const uint64_t peer_hs = get_noc_addr(pxy & 0xFFFFu, pxy >> 16, kPeerHss[p] + kHsProbeF);
            noc_async_read(peer_hs, kHeadScratch, 4u, NOC_INDEX);
            noc_b.async_read_barrier();
            invalidate_l1_cache();
            *mv_probe_f[p] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHeadScratch);
            volatile tt_l1_ptr uint32_t* msrc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHeadScratch + 32u);
            *msrc = kProbeMoverMagic + p;  // + slot, so the host can tell peer 0's write from peer 1's
            noc_async_write(
                kHeadScratch + 32u, get_noc_addr(pxy & 0xFFFFu, pxy >> 16, kPeerHss[p] + kHsProbeM), 4u, NOC_INDEX);
            // Barrier per peer: the scratch word is reused by the next peer's write, so it must have landed.
            (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack);
        }
    }

    // ---- NoC FOOTPRINT: the one sampling primitive, and the only place NIU registers are read ------------
    //
    // Folds to nothing when the feature is off: `if constexpr` on a compile arg, so the OFF build has no
    // check, no call and no register pressure -- which is the standard this file already holds itself to
    // (FINDINGS N+41 measured a 6x difference between a runtime check inside an emitter and a compile-time
    // one at the call site).
    //
    // Both NoCs are read unconditionally rather than just the one this role uses. A filler reads on kReadNoc
    // and writes on NOC_INDEX, a mover does both on NOC_INDEX -- but that is the thing being verified, not an
    // assumption to bake in. The zeros are the measurement: if a filler's NoC 0 read counter is non-zero the
    // read/write NoC split is not doing what the code claims.
    // Called at the end of every sweep: sample, then decide whether this sweep extends the workload window.
    //
    // Sampled HERE, after the sweep body and before the pacing gap. The gap issues no NoC traffic, so which
    // side of it the sample falls on cannot change any byte total; taking it before means the window's
    // DURATION is measured over sweeps rather than over sweeps-plus-whatever-pace-trailed-the-last-one, which
    // is the honest denominator for a MB/s figure.
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

    if constexpr (kNocFootprint != 0) {
        // Seed the mirrors so the first sweep's delta is measured from HERE, not from whatever the counters
        // held at chip reset. Everything the bring-up path did (NIU flip, socket config, the mock staging
        // fill) is therefore excluded from both blocks, which is what makes the lifetime figure mean "this
        // drain loop" rather than "this core since power-on".
        // Reusing nf_sample_regs for the seed rather than open-coding the reads keeps the read sequence to ONE
        // copy in the binary. The deltas this first call folds in are against a zeroed mirror, so they are
        // meaningless and are cleared immediately below; only prev[] matters here.
        nf_sample_regs(&nf);
        for (uint32_t i = 0; i < kNfSlots; i++) {
            nf.life[i] = 0;
        }
        nf.cost = 0;
        for (uint32_t n = 0; n < 2; n++) {
            nf.posted_entry[n] = NOC_STATUS_READ_REG(n, NIU_MST_POSTED_WR_REQ_SENT);
        }
    }

    // Stop-path sweep-to-empty: on stop=1 keep sweeping until one whole sweep moves nothing, so markers
    // still in worker rings (or DRAM-ring frames not yet moved) ship instead of being stranded -- exiting
    // on the stop word directly is what silently cut the capture tail on every lane. Producers are
    // quiescent at close, so this converges in a sweep or two; the deadline covers one that is not.
    constexpr uint64_t kStopDrainCycles = 135000000;
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
        // ---- COMMON-TRIGGER SYNC EVENT: the rendezvous ----
        //
        // FIRST thing in the loop body, before `sweeps++` and before t_sweep0 is taken, so a barrier wait is
        // not billed to any sweep's duration and cannot perturb the phase accounting the zones are checked
        // against.
        if constexpr (kSyncEvent != 0) {
            invalidate_l1_cache();
            const uint32_t req = *sync_req;
            if (req != sync_seen) {
                sync_seen = req;
                const uint64_t t_park = get_timestamp();
                *sync_ack = req;  // parked; the host may release once every drainer has done this
                uint64_t t_go = 0;
                // Bounded, so a host that never releases DEGRADES instead of wedging the workload. An
                // unbounded spin here would be indistinguishable from the resident-drainer failure mode that
                // hangs a run with a perfectly healthy card.
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
                    // Force emission: the sync zone is NOT part of the work-armed window and must land whether
                    // or not this sweep would have been instrumented. The arming block immediately below
                    // rewrites self_on/self_from_start for this sweep, so nothing has to be restored here.
                    // The scope's OWN clock read is the fiducial: it sits a fixed handful of instructions
                    // after the release detection at t_go, identically on every drainer, so releasing them
                    // together still marks the same physical instant.
                    self_on = true;
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SYNC, SelfNow, SelfClose> z_sync(
                            self_now, self_zone_close);
                    }
                    self_publish();  // ship it now; do not let it wait on the ring filling
                    sync_events++;
                    sync_spin_cyc = static_cast<uint32_t>(t_go - t_park);
                } else {
                    sync_timeouts++;
                }
            }
        }
        sweeps++;
        *hb = sweeps;
        *phase = kPhasePoll;
        const uint64_t t_sweep0 = get_timestamp();
        const uint32_t frames_at_sweep_start = frames;
        const uint64_t s_read0 = c_read, s_proc0 = c_proc, s_rsv0 = c_reserve, s_wr0 = c_write, s_bar0 = c_barrier;
        const uint64_t words_at_sweep_start = total_words;
        sweep_max_run = 0;
        // MOVER: the largest raw peer backlog seen this sweep = the fullest DRAM ring, in frames. The
        // shaper's safety signal: pace only while the ring has headroom to absorb the shaped rate.
        [[maybe_unused]] uint32_t mv_max_backlog = 0;

        // Inside an active window EVERY sweep is instrumented, which costs one register compare against a
        // deadline. Only the FIRST sweep of a window is partial -- it arms late, from self_arm, at the moment
        // work is discovered.
        if constexpr (kSelfZones != 0) {
            self_t_sweep0 = t_sweep0;
            self_work = false;
            self_on = false;
            self_from_start = false;
            if (self_frames >= kSelfMaxFrames) {
                self_over++;
                // Flush what the ring still holds, once. The budget running out is not a reason to discard
                // zones already written; publish() is a no-op on every later sweep because head == tail.
                self_publish();
            } else if (t_sweep0 < self_armed_until) {
                self_on = true;
                self_from_start = true;
                // The SWEEP zone itself is the RAII scope opened just below, once self_on is decided.
            }
        }

        uint32_t sweep_cyc = 0;
        {
            // ---- DRISC-SWEEP: an ordinary RAII zone over this sweep's work -----------------------------
            // Constructed AFTER the arming block decided self_on, so an armed-window sweep records its whole
            // body from its own entry/exit clock reads, and an uninstrumented sweep costs one flag check. A
            // sweep that arms itself mid-body (self_arm) gets no SWEEP zone -- this scope had already been
            // constructed off -- only its post-arm children. Closed BEFORE the pacing gap below, because
            // PACE is deliberately SWEEP's sibling at depth 0, not its child.
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SWEEP, SelfNow, SelfClose> z_sweep(
                self_now, self_zone_close);
            if constexpr (kRole == kRoleMover) {
                // ---- MOVER: kNPeer DRAM rings -> staging -> the existing D2H socket ----
                //
                // No worker grid, no control-vector scan, no head write-back. The frames in the ring are already
                // complete (prefix + span, written by the filler), so this is a copy and a push -- which is why the
                // socket protocol, the host FIFO and the host decoder are all untouched by the role split.
                //
                // The peers are visited SEQUENTIALLY and each gets the whole staging area. The write barrier at the
                // end of a peer's visit is what makes that safe (it already had to be there for staging reuse), and
                // it is also why splitting the slots between peers would buy nothing: the two pushes go into ONE
                // socket, so they could not have overlapped anyway. Cost of getting this wrong is direct -- halving
                // the batch doubles the per-frame credit-wait/notify overhead, which is where the knee lives.
                for (uint32_t peer = 0; peer < kNPeer; peer++) {
                    const uint32_t pxy = kPeerXYs[peer];
                    const uint64_t t_r0 = get_timestamp();
                    uint32_t head;
                    uint32_t n;
                    {
                        // The visit's READ -- the head poll, plus the contiguous DRAM batch read when the ring
                        // has frames -- as ONE ordinary RAII zone (there is no issue/wait split to make on a
                        // mover). On the visit that DISCOVERS work this scope was constructed before self_arm()
                        // ran, so that visit's READ goes unrecorded (SpscZoneScope::start_ == 0); the knee phases
                        // that follow -- CREDIT-WAIT, WRITE, WR-BARRIER -- arm in time and are captured. Within
                        // an armed window every visit's READ is whole, idle visits included: the head poll and
                        // its barrier are what an idle mover sweep is made of (185,097 of 185,409 sweeps), and
                        // leaving them out would hide the whole idle cost.
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ, SelfNowPhase, SelfClose>
                            z_read(self_now_phase, self_zone_close);
                        noc_async_read(
                            get_noc_addr(pxy & 0xFFFFu, pxy >> 16, kPeerHss[peer] + kHsHead),
                            kHeadScratch,
                            4u,
                            NOC_INDEX);
                        // The FREE-function barrier, matching the free-function read above. Noc{}::async_read_barrier()
                        // is a different accounting path, and a barrier that watches the wrong counters returns EARLY
                        // -- which here would mean reading a head value that has not landed yet.
                        noc_async_read_barrier(NOC_INDEX);
                        invalidate_l1_cache();
                        head = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHeadScratch);
                        n = head - mv_tail[peer];
                        // HANDSHAKE SANITY, and it is not paranoia -- this exact check is what turns a silent capture
                        // corruption into a reported number.
                        //
                        // head is monotonic and the filler can never be more than kDramFrames ahead, so n > kDramFrames
                        // is structurally impossible and means the value is not a head at all. Observed for real:
                        // releasing the filler's NIU (stop=2) flips it back to NOC2AXI, where an inbound DRAM-range
                        // address is forwarded to GDDR instead of terminating at L1 -- so this read started returning
                        // GDDR contents (0xF5AE93CB), n underflowed to ~4.1e9, the `n > kNStage` clamp quietly turned
                        // that into "7 frames are ready", and the mover shipped 1,800 frames of garbage that no
                        // existing counter noticed. Bail instead of clamp.
                        if (n > kDramFrames) {
                            hs_bad++;
                            n = 0;
                            egress_dead = true;  // the peer is gone or unreadable; shipping anything more is corruption
                        }
                        if (n > ring_hi[peer]) {
                            ring_hi[peer] = n;
                        }
                        if constexpr (kMoverFrameGap != 0) {
                            if (n > mv_max_backlog) {
                                mv_max_backlog = n;  // raw backlog, before the batch clamps
                            }
                        }
                        if (n != 0) {
                            // A MOVER's definition of work, known before anything expensive happens. This is the arm
                            // that matters most: a mover is idle in ~199,584 of ~199,918 sweeps (0.17% busy), and the
                            // socket credit wait that sets the knee exists only on this path and only when n != 0.
                            if constexpr (kSelfZones != 0) {
                                if (!self_on) {
                                    self_arm();
                                }
                            }
                            // Bounded by staging, then by the ring wrap -- clamping at the wrap costs one short read
                            // per lap and keeps this a SINGLE contiguous DRAM read, which matters because reads are
                            // what the mover is made of.
                            if (n > kNStage) {
                                n = kNStage;
                            }
                            const uint32_t off = mv_tail[peer] % kDramFrames;
                            if (off + n > kDramFrames) {
                                n = kDramFrames - off;
                            }
                            noc_async_read(
                                get_noc_addr_from_bank_id<true>(
                                    kPeerBanks[peer], kPeerAddrs[peer] + off * kSlotBytes, NOC_INDEX),
                                kStageBase,
                                n * kSlotBytes,
                                NOC_INDEX);
                            noc_b.async_read_barrier();
                        }
                    }  // z_read closes: the READ zone ends at the read barrier, busy or idle
                    c_read += get_timestamp() - t_r0;
                    if (n != 0) {
                        if (*mv_probe_frame[peer] == 0) {
                            // First frame word the mover ever saw ON THIS RING. The host checks it against
                            // spsc_span_w0(), which proves the filler's DRAM write and this read agree on the address
                            // -- end to end, with no host-side DRAM read needed and no way for a plausible-but-wrong
                            // ring address to pass. Per ring, because the two rings are in different DRAM banks and
                            // only one of them may be wrong.
                            *mv_probe_frame[peer] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase);
                        }
                        // Release the ring region NOW: the bytes are in staging, so the filler may reuse it
                        // immediately. Doing this before the push (rather than after) is what keeps the filler off the
                        // ring ceiling.
                        mv_tail[peer] += n;
                        volatile tt_l1_ptr uint32_t* tsrc =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHeadScratch + 32u);
                        *tsrc = mv_tail[peer];
                        noc_async_write(
                            kHeadScratch + 32u,
                            get_noc_addr(pxy & 0xFFFFu, pxy >> 16, kPeerHss[peer] + kHsTail),
                            4u,
                            NOC_INDEX);
                        *mv_live_head[peer] = head;
                        *mv_live_tail[peer] = mv_tail[peer];
                        // A mover's emit_run IS ship_run; its CREDIT-WAIT and WRITE zones are RAII scopes inside it.
                        emit_run(0, n);
                        frames += n;
                        mv_moved[peer] += n;
                        if (n > mv_max_n[peer]) {
                            mv_max_n[peer] = n;
                        }
                        // ONE barrier covers the PCIe push (staging is about to be refilled -- by the NEXT PEER as well
                        // as the next sweep), the tail write (the filler cannot see room until it lands) and the reuse
                        // of the +32 scratch word the tail write sources from.
                        const uint64_t t_b0 = get_timestamp();
                        *phase = kPhBar2;
                        {
                            kernel_profiler::
                                SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfNowPhase, SelfClose>
                                    z_bar(self_now_phase, self_zone_close);
                            if (!write_barrier_bounded(t_b0 + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack)) {
                                egress_dead = true;
                            }
                        }
                        c_barrier += get_timestamp() - t_b0;
                    }
                    // Never start the second peer once egress is dead: staging may hold unflushed bytes, and an
                    // impossible head on one ring says nothing good about the other.
                    if (egress_dead) {
                        break;
                    }
                }  // for peer
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

                // Fill-gate override for THIS sweep: force-ship period, or the stop-word drain (stop_seen_at
                // is the sweep loop's own latch -- no extra L1 read). Folds to `true` when the gate is off.
                [[maybe_unused]] const bool gate_ship_all =
                    kStageMinFillPct == 0 || (sweeps & kGateForceMask) == 0 || stop_seen_at != 0;

                auto process_batch = [&](uint32_t base_c, uint32_t n, uint32_t g) {
                    const uint64_t t_p0 = get_timestamp();
                    // c_self joins the nested term because self_publish RESTORES c_reserve/c_write (see there), so
                    // without it the time a mid-batch self publish spends would be charged to `proc`. With kSelfZones
                    // == 0 the ternary folds to a literal 0 and this is the expression it always was.
                    const uint64_t flush_at = c_reserve + c_write + (kSelfZones != 0 ? c_self : 0);
                    // PROC as an ordinary RAII scope over the whole batch: START goes out here, so its children
                    // (the credit wait and the write inside emit_run) follow it in the stream and nest under it.
                    // On the batch that DISCOVERS work (self_arm below, after the scan) this scope was constructed
                    // while instrumentation was off and stays off (SpscZoneScope::start_ == 0), so that batch
                    // contributes only its post-arm egress children -- there is no retroactive PROC any more.
                    kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PROC, SelfNowPhase, SelfClose> z_proc(
                        self_now_phase, self_zone_close);
                    // `frames` advances once per LIVE core, so this is a zero-cost way to ask at the end of the batch
                    // whether it found any work -- no flag in the scan loop, which is the one place nothing may be
                    // added (see the SCAN comment below: it is register-pressure bound, and a call in there spilled
                    // the head mirror and cost 40% of proc).
                    const uint32_t frames_at_p0 = frames;
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

                        // SCAN, UNROLLED INTO REGISTERS. This is the single largest busy-sweep cost (~42% of busy,
                        // ~356 ns/core = ~480 cycles) and it is not arithmetic-bound -- ~40 ops cannot cost 480
                        // cycles. It is L1-ACCESS bound: `runs[]` and the head mirror are arrays the compiler
                        // spills, so each core paid ~25 L1 round trips (5 tail loads + 10 for runs[] + 10 for
                        // mine[]). Hoisting the mirror into scalars and dropping runs[] entirely leaves only the
                        // 5 tail loads and one mirror load/store per RISC, which is the irreducible part.
                        //
                        // kNumRisc is 5 and fixed, so this unrolls cleanly; a loop over an indexed array does not
                        // keep its elements in registers on this core.
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
                        if (peak > sweep_max_run) {
                            sweep_max_run = peak;
                        }
                        const uint32_t live = r0 + r1 + r2 + r3 + r4;
                        if (live == 0) {
                            emit_run(run_start, run_len);
                            run_len = 0;
                            continue;
                        }
                        // The staging fill gate (see kStageMinFillPct above). AFTER the peak bookkeeping, so the
                        // pacing valves always see true occupancy; mirrors and head stay untouched, so the words
                        // remain the producer's until a sweep that ships them.
                        if constexpr (kStageMinFillPct != 0) {
                            if (live < kStageMinWords && peak < kPaceHighWater && !gate_ship_all) {
                                emit_run(run_start, run_len);
                                run_len = 0;
                                continue;
                            }
                        }
                        if (run_len == 0) {
                            run_start = sl;
                        }
                        run_len++;

                        // Head write-back releases the producer. Safe at once: the payload is a snapshot already
                        // resident in staging, so those ring slots are free regardless of when it reaches the host.
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
                        // HEAD WRITE-BACK, timed separately. Immediate, per core, on NOC_INDEX -- both measured
                        // optimal (2026-08-24), keep the falsifications: (a) batching/deferring the issues to a
                        // post-scan loop was WORSE at the onset edge (d22 140 -> 566 stalls) -- release IMMEDIACY
                        // dominates the 0.5-0.6%-of-busy issue cost; (b) moving the write to kReadNoc was
                        // CATASTROPHIC (d22 140 -> 14k) -- this write travels TO the worker cores, the exact
                        // route the span-read requests/31 KB responses occupy, while NOC_INDEX's staging and
                        // mover traffic goes to DRAM rows and never touches worker routes.
                        //
                        // POSTED, deliberately: a nonposted head write's worker ACK round-trip was swept into
                        // every write_barrier_bounded wait (the barrier polls NIU_MST_WR_ACK_RECEIVED for ALL
                        // nonposted writes), and those barriers exist only for the STAGING writes -- slot reuse
                        // and publish_head gating. The ack protected nothing: heads touch neither staging nor
                        // the DRAM ring. Posted also removes the ack packet itself from the congested worker
                        // route. Scratch-reuse safety is the 32-slot rotation, same in-flight margin the
                        // nonposted version relied on between barriers (<= kGenSlots live cores per batch).
                        const uint64_t t_h0 = get_timestamp();
                        const uint32_t sc = kHeadScratch + hb_slot * 32u;
                        volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                        scp[0] = m0;
                        scp[1] = m1;
                        scp[2] = m2;
                        scp[3] = m3;
                        scp[4] = m4;
                        noc_async_write_one_packet<true, true>(
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
                    // can be acted on without putting anything inside the scan. Arming here is what makes busy sweeps
                    // get captured at all -- 114 of ~21,000 sweeps have live cores, so a rule keyed on sweep number
                    // samples the other 99.5%. COST: this batch's own READ and PROC scopes were constructed while
                    // instrumentation was off, so the first batch of a newly-armed sweep contributes only the egress
                    // children of the emit_run below. Every later batch of that sweep is complete.
                    if constexpr (kSelfZones != 0) {
                        if (!self_on && frames != frames_at_p0) {
                            self_arm();  // opens the window at EVERY detail level
                        }
                    }
                    emit_run(run_start, run_len);
                    gen_shipped[g] = true;
                    // SATURATING. The nested ship_run time is subtracted out so it is not double-counted against
                    // proc, but if that term ever exceeds the elapsed span the unsigned subtract wraps -- observed
                    // once as "proc 18727729111430.1%", which silently corrupts the whole phase breakdown.
                    {
                        const uint64_t t_p1 = get_timestamp();
                        const uint64_t span = t_p1 - t_p0;
                        const uint64_t nested = (c_reserve + c_write + (kSelfZones != 0 ? c_self : 0)) - flush_at;
                        c_proc += (span > nested) ? (span - nested) : 0;
                    }
                    // z_proc closes here: the PROC zone spans the whole batch, i.e. c_proc PLUS its nested
                    // credit-wait/write children. That is what a Tracy parent is; the host cross-check
                    // subtracts the children.
                };

                for (uint32_t base_c = 0; base_c < num_cores; base_c += kGenSlots) {
                    const uint32_t n = (num_cores - base_c) < kGenSlots ? (num_cores - base_c) : kGenSlots;

                    // This generation's previous ship must have landed before its slots are refilled.
                    if (gen_shipped[gen]) {
                        const uint64_t t_b0 = get_timestamp();
                        *phase = kPhBar1;
                        bool flushed;
                        {
                            kernel_profiler::
                                SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfNowPhase, SelfClose>
                                    z_bar(self_now_phase, self_zone_close);
                            flushed = egress_barrier(t_b0 + kCreditWaitCycles);
                        }
                        c_barrier += get_timestamp() - t_b0;
                        if (!flushed) {
                            egress_dead = true;
                            break;
                        }
                        publish_head();  // those DRAM writes are now flushed, so the mover may have them
                        gen_shipped[gen] = false;
                    }

                    // c_read is TWO disjoint intervals per batch -- the issue, and whatever wait survives the
                    // concurrent ship -- so it takes two zones (READ here, READ-WAIT below). One zone spanning
                    // both would swallow process_batch and claim read time the overlap actually hides, which is
                    // the very error the c_read accounting comment below was written to fix.
                    const uint64_t t_batch0 = get_timestamp();
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ, SelfNowPhase, SelfClose>
                            z_issue(self_now_phase, self_zone_close);
                        for (uint32_t i = 0; i < n; i++) {
                            const uint32_t xy = coords[base_c + i];
                            CoreLocalMem<uint32_t> dst(kStageBase + (gen * kGenSlots + i) * kSlotBytes + kPrefix * 4u);
                            if constexpr (kReadSplit == 2) {
                                // SPLIT WITHIN THE CORE: both NoCs carry half of the SAME span. Alternating whole
                                // cores (kReadSplit==1) left only ~3 transactions outstanding -- kGenSlots is 3 -- and
                                // measured as a no-op. Halving each span doubles outstanding transactions (3 cores x 2
                                // halves) without needing more L1, which is the only free variable left.
                                constexpr uint32_t kHalf = (kSpanBytes / 2u) & ~0x1Fu;  // 32 B aligned
                                noc.async_read<NocOptions::DEFAULT, kHalf>(
                                    src, dst, kHalf, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
                                CoreLocalMem<uint32_t> dst2(
                                    kStageBase + (gen * kGenSlots + i) * kSlotBytes + kPrefix * 4u + kHalf);
                                noc_b.async_read<NocOptions::DEFAULT, kSpanBytes - kHalf>(
                                    src,
                                    dst2,
                                    kSpanBytes - kHalf,
                                    {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src + kHalf},
                                    {});
                            } else if constexpr (kReadSplit == 1) {
                                // Alternate cores between the two NoCs so both have transactions outstanding.
                                if ((i & 1u) == 0u) {
                                    noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                                        src,
                                        dst,
                                        kSpanBytes,
                                        {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src},
                                        {});
                                } else {
                                    noc_b.async_read<NocOptions::DEFAULT, kSpanBytes>(
                                        src,
                                        dst,
                                        kSpanBytes,
                                        {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src},
                                        {});
                                }
                            } else {
                                noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                                    src,
                                    dst,
                                    kSpanBytes,
                                    {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src},
                                    {});
                            }
                        }
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
                    {
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ_WAIT, SelfNowPhase, SelfClose>
                            z_wait(self_now_phase, self_zone_close);
                        noc.async_read_barrier();
                        if constexpr (kReadSplit != 0) {
                            noc_b.async_read_barrier();  // staging reuse is only safe once BOTH read NoCs have landed
                        }
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
                        kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfNowPhase, SelfClose>
                            z_bar(self_now_phase, self_zone_close);
                        flushed = egress_barrier(t_b0 + kCreditWaitCycles);
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
        // Per-sweep NoC sample, stamped at the sweep's END so it lines up with the DRISC-SWEEP zone that just
        // closed. ROLE-SPECIALISED: a filler reads on kReadNoc and writes on NOC_INDEX, a mover does both on
        // NOC_INDEX, so only that role's four counters are compiled -- the other four are the provable zeros
        // (out[] still ships all eight, which is where that proof is kept).
        if constexpr (kNocFpSeries != 0) {
            self_nocfp();
        }

        // ---- this sweep's SWEEP zone closed with its scope above; the PACE zone and the publish come AFTER
        // the gap ----
        //
        // There is deliberately NO per-sweep publish: mid-window frames are published by self_mark_w0 when the
        // ring fills, which keeps the publish off the per-sweep critical path and fills each frame.
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
                    // The counter side of the cross-check, over exactly the sweep the zones just described.
                    self_ck_sweeps++;
                    self_ck_read += c_read - s_read0;
                    self_ck_proc += c_proc - s_proc0;
                    self_ck_rsv += c_reserve - s_rsv0;
                    self_ck_write += c_write - s_wr0;
                    self_ck_bar += c_barrier - s_bar0;
                }
            }
        }

        // ---- pacing controller ----
        //
        // Asymmetric on purpose. Widening the gap raises fill but walks toward the ring ceiling, where a
        // lossless producer BLOCKS and we would have traded host bytes for a stalled workload; narrowing it
        // only costs bytes. So: creep up, collapse down.
        // A MOVER is excluded: it has no worker grid, so sweep_max_run and total_words are permanently 0 and
        // the controller would read "spans arriving 0% full", creep the gap to its 200,000-cycle ceiling and
        // sleep ~148 us between pushes -- pacing the consumer instead of the producer, which is backwards.
        if constexpr (kFillPct != 0 && kRole != kRoleMover) {
            // THREE-LEVEL RESPONSE. The first version collapsed the gap to 0 whenever the single hottest
            // core crossed 3/4, which at 120 cores fires nearly every sweep -- so the gap never held and
            // pacing did nothing at low producer rates (delay 125: gap stuck ~1,200 of a 20,000 ceiling,
            // occupancy still 510). Only a core in real danger of blocking should stop pacing outright.
            if (sweep_max_run >= kPaceCritical) {
                gap = 0;  // about to block a lossless producer: drain now, fill be damned
            } else if (sweep_max_run >= kPaceHighWater) {
                gap -= gap >> 2;  // getting warm: ease off 25%, do not abandon pacing
            } else {
                const uint32_t frames_now = frames - frames_at_sweep_start;
                if (frames_now != 0) {
                    // One sweep's words fit 32 bits with huge margin (<= grid x live capacity, ~300K);
                    // dividing the u64 directly drags the 956 B __udivdi3 soft-div into a code region
                    // with less than that in total headroom.
                    static_assert(
                        static_cast<uint64_t>(kMaxCores) * kNumRisc * kRingWords <= 0xFFFFFFFFull,
                        "a sweep's word delta no longer fits the 32-bit fill division");
                    const uint32_t mean_fill = static_cast<uint32_t>(total_words - words_at_sweep_start) / frames_now;
                    if (mean_fill < kFillTarget) {
                        // Under-full: wait longer. STEEPER THAN IT LOOKS, and deliberately tuned by constants
                        // rather than by adding a branch -- `gap >> 1` is the same instruction as `gap >> 2`
                        // with a different immediate, and the floor is a literal, so this costs ZERO bytes of
                        // DRISC code. That matters: the instrumented build (self-zones + NoC footprint) has
                        // ~36 B of headroom in an 11,264 B code region, and a two-regime version of this ramp
                        // overflowed it -- the drainer then FAILED TO LOAD and the run silently produced no
                        // device zones at all.
                        //
                        // WHY STEEPER. The controller is asymmetric: up x1.5 here, down x0.875 on over-target,
                        // and a kPaceCritical collapse throws the gap to 0 outright. On a sparse real model
                        // (ResNet-50 trace: 96 work windows per filler, rings nearly empty) collapses fire
                        // constantly, so at x1.25 with a floor of 64 the gap needed ~35-40 sweeps to climb back
                        // and effectively never arrived -- the PACE distribution came out bimodal with 24-32%
                        // of samples under 1 us. At x1.5 with a floor of 512 it is ~13 sweeps, roughly 3x
                        // faster, without delaying the FIRST sweep (which is what moved the knee when the gap
                        // was seeded high instead). See FINDINGS §N+44.
                        uint32_t inc = gap >> 1;
                        if (inc < 512u) {
                            inc = 512u;
                        }
                        gap = (gap + inc > kGapMaxCycles) ? kGapMaxCycles : gap + inc;
                    } else if (mean_fill > kFillTarget) {
                        gap -= gap >> 3;  // over-target: ease down and settle
                    }
                }
            }
        }
        // MOVER-SIDE PACING, with a criterion and a ceiling of its own.
        //
        // A mover's job is to empty the DRAM frame ring, so the right signal is FRAMES AVAILABLE, not span
        // fill: collapse to 0 the instant there is anything to move, back off only when the ring is empty.
        // That never delays a real drain, and it targets the traffic that measurement showed is nearly all
        // overhead -- ~1.6 M transactions to move ~380 MB on ResNet-50 (~240 B/txn), i.e. mostly 4 B head
        // polls, plus 13-15% of a mover's egress spent on self-description simply because it samples every
        // sweep and it sweeps every ~1.27 us.
        //
        // CEILING IS 10 us, not the filler's 148 us, and that asymmetry is the whole point. The earlier
        // accident that seeded a mover at 148 us with no feedback path made it ~114x slower to drain and a
        // 20 ms trace replay took ~100 ms to finish (FINDINGS §N+44). Pacing a CONSUMER is only safe while it
        // is provably idle and while the backoff is short enough that one missed observation cannot extend the
        // drain tail materially -- movers already trail the workload by ~2.5-2.9 ms.
        //
        // Room to do this at all: each role compiles its own ELF, and the mover binary is ~6,788 B against the
        // 11,264 B code region -- ~4.4 KB spare, where a filler has ~36 B. This is the one place new device
        // code fits.
        if constexpr (kFillPct != 0 && kRole == kRoleMover) {
            constexpr uint32_t kMoverGapMax = 13500;  // 10 us at 1.35 GHz
            // Same busy/idle signal the sweep bookkeeping already uses at both roles, so there is no second
            // definition of "did this sweep do anything" to drift.
            if (frames != frames_at_sweep_start) {
                // Frames were there. The shaper's criterion is RING HEADROOM, deliberately NOT backlog:
                // at the onset-with-runway regime the mover is *always* backlogged (the ring is doing its
                // job) yet its burst density is pure harm -- compressed sweeps inflate the landing tail of
                // the fillers' posted head write-backs, the write whose landing releases a blocked producer
                // (FINDINGS N+66: decode-paced acks shaped this by accident, arm B 164-212 stalls vs arm
                // A's 2.2-2.6k at the same device config; a backlog-keyed first cut of this shaper was a
                // measured no-op for exactly this reason). So: while the fullest peer ring is under HALF,
                // pace kMoverFrameGap x frames just moved through the ordinary PACE zone below -- the ring
                // absorbs the shaped rate, that is what runway is. Past half, drain flat out: from half to
                // full there are thousands of sweeps of reaction room, and near-full is the one regime
                // where mover rate gates the fillers (the sustained ceiling), which pausing would move.
                // A stop-drain is never paced.
                gap = 0;
                if constexpr (kMoverFrameGap != 0) {
                    // Engage ONLY when this sweep's own credit wait was ~nothing -- i.e. acks are instant
                    // (NO_DECODE / an idle consumer) and no one else is shaping. Decode-paced acks ARE the
                    // shaping this replaces; adding a gap on top of them double-paces the mover in the one
                    // regime where its rate feeds back to producers (measured: full-pipe d100 stalls rose
                    // with an unconditional shaper, and are untouched with this guard).
                    const uint64_t rsv_this_sweep = c_reserve - s_rsv0;
                    if (mv_max_backlog < kDramFrames / 2u && rsv_this_sweep < 1350u && stop_seen_at == 0) {
                        const uint32_t moved = frames - frames_at_sweep_start;
                        const uint32_t shaped = kMoverFrameGap * moved;
                        gap = shaped > kMoverGapMax ? kMoverGapMax : shaped;
                    }
                }
            } else {
                uint32_t inc = gap >> 1;
                if (inc < 256u) {
                    inc = 256u;
                }
                gap = (gap + inc > kMoverGapMax) ? kMoverGapMax : gap + inc;
            }
        }
        // THE PACING GAP, as its own depth-0 zone -- an ordinary RAII scope over the wait, a sibling of the
        // SWEEP scope that closed above, never its child. A FILLER's controller settled it at 17,156 cycles
        // (12.7 us) against an 8.5 us sweep, so it is the majority of that core's wall time -- and without a
        // zone it is unexplained whitespace between SWEEPs, which is exactly what makes a drainer row
        // unreadable. A MOVER is excluded from the controller (see below), so its gap stays 0 and this zone
        // appears on a mover row only via the traffic shaper (kMoverFrameGap, keeping-up sweeps): a PACE
        // zone after a productive mover sweep IS the shaper firing; after an idle one it is the idle ramp.
        if (gap != 0) {
            kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PACE, SelfNow, SelfClose> z_pace(
                self_now, self_zone_close);
            const uint64_t until = get_timestamp() + gap;
            while (get_timestamp() < until) {
            }
        }
        // The window's LAST sweep has to flush, or its zones sit in the ring until the next window -- or
        // forever. Done after the gap so PACE rides in the same frame as the SWEEP it follows.
        if constexpr (kSelfZones != 0) {
            if (self_on) {
                if (get_timestamp() >= self_armed_until) {
                    self_publish();
                }
                self_on = false;
            }
        }
    }

    // FLUSH WHATEVER IS STILL IN THE RING. Gated on `self_tail != self_head`, NOT on self_on -- self_on is
    // cleared at the end of every sweep, so a check on it can never fire here, and this tail flush silently did
    // nothing. MEASURED before the fix: a filler wrote 945 words of zones and shipped 504, losing the last 55
    // sweeps of its own trace while every counter read clean. Every SWEEP zone is already closed inside the
    // loop, so there is nothing to emit here -- only bytes to ship. Publishing needs a live egress, so skip it
    // if egress is already declared dead.
    if constexpr (kSelfZones != 0) {
        self_on = false;  // no more markers; self_publish() must not think it is inside a traced sweep
        if (!egress_dead) {
            self_publish();
        }
    }

    // socket_barrier() waits for the host to ack everything, so it hangs on a dead consumer just
    // like the write barrier did. Skip both when we already know the consumer is gone.
    // A FILLER has no socket at all, so both socket calls in this tail are skipped for it -- not because they
    // would be slow but because `sender` was never initialised.
    const bool consumer_gone = egress_dead || credit_timeouts != 0 || kRole == kRoleFiller;
    *phase = kPhSockBar;
    if (!consumer_gone) {
        socket_barrier(sender);
    }
    *phase = kPhBarTail;
    *phase = kPhTailBar;  // distinct from kPhBar1: the tail barrier used to run while phase still read 11
    (void)egress_barrier(get_timestamp() + kCreditWaitCycles);
    // The posted head write-backs are not in the barrier above; drain their SENT counter (bounded spin --
    // 20 B packets stream out in ns) so no scratch slot or unstreamed head is left behind at report time.
    if constexpr (kRole != kRoleMover) {
        const uint64_t t_ps = get_timestamp() + 1350000;  // 1 ms, absurdly generous
        while (!ncrisc_noc_posted_writes_sent(NOC_INDEX) && get_timestamp() < t_ps) {
        }
    }
    // Publish the LAST staged frames. Without this the final batch is written to the ring but never announced,
    // so the mover cannot drain it and the tail of every capture is silently short by up to one sweep.
    publish_head();
    // FILLER: wait until the mover has drained everything we published before reporting anything.
    // Without this the filler exits holding a STALE tail mirror -- observed `tail 2414` against 3089 frames
    // staged -- because the mover is still draining when the filler's results are written. That is not just a
    // cosmetic log: `inflight = frames_staged - *hs_tail` is the ring-room predicate, so a filler whose mirror
    // lags believes the ring is fuller than it is and can wait for room that is already free.
    // Bounded on the same deadline the write barrier uses: a dead or stopped mover must never wedge us here,
    // and the host's quiesce order (fillers -> drain rings -> movers) guarantees the mover outlives this wait.
    if constexpr (kRole == kRoleFiller) {
        const uint64_t drain_deadline = get_timestamp() + kCreditWaitCycles;
        while (*hs_tail != frames_staged && get_timestamp() < drain_deadline) {
        }
    }
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
    out[43] = ws_read;
    out[44] = ws_proc;
    out[45] = ws_rsv;
    out[46] = ws_wr;
    out[47] = ws_bar;
    out[42] = gap;  // where the pacing controller settled
    out[40] = static_cast<uint32_t>(c_ph_head & 0xFFFFFFFFu);
    out[41] = static_cast<uint32_t>(c_ph_head >> 32);
    // ---- role-split instrumentation (0 on the default path) ----
    // Shipped WITH the feature, not added after a bad capture: the last single-producer ring driven by two
    // threads lost 1.03M records and corrupted every lane's nesting while every pre-existing counter read
    // clean. ring_hi says how much of the elastic buffer is actually used (the whole justification for it);
    // ring_blocked says whether the ring ever became the new bottleneck.
    out[48] = kRole;
    out[49] = (kRole == kRoleFiller) ? frames_staged : mv_moved[0];
    out[50] = ring_hi[0];    // head - tail high-water on peer 0's ring (the filler's own ring), in frames
    out[51] = ring_blocked;  // FILLER: pushes that had to wait for ring room
    out[52] = kDramFrames;
    out[53] = (kRole == kRoleFiller) ? *hs_tail : mv_tail[0];
    out[54] = mv_max_n[0];
    out[55] = (kRole == kRoleMover) ? *mv_probe_frame[0] : 0u;
    out[56] = (kRole == kRoleMover) ? *mv_probe_f[0] : 0u;
    // MUST BE ZERO. Non-zero means the mover read something that cannot be a head (see the check site), so
    // every frame it shipped from then on is suspect. Summed over both rings -- there is nothing to gain from
    // knowing WHICH ring lied, because the mover declares egress dead either way.
    out[57] = hs_bad;
    // ---- PEER 1 of a dual-ring mover. Mirrors 49/53/54/55/50 above, so nothing had to be renumbered. ----
    // A per-peer copy of every quantity the verification bar checks: frames staged == frames moved must hold
    // PER RING, and a single summed figure could hide one ring shipping short while the other over-ships.
    out[58] = (kRole == kRoleMover) ? mv_moved[1] : 0u;
    out[59] = (kRole == kRoleMover) ? mv_tail[1] : 0u;
    out[60] = mv_max_n[1];
    out[61] = (kRole == kRoleMover) ? *mv_probe_frame[1] : 0u;
    out[62] = (kRole == kRoleMover) ? *mv_probe_f[1] : 0u;
    out[63] = ring_hi[1];
    // Peers 2-5 (6-filler shape): out[108 + (p-2)*5] = {moved, ring_hi, tail, max_batch, first frame}.
    for (uint32_t p = 2; p < 6; p++) {
        const uint32_t w = 108u + (p - 2u) * 5u;
        out[w + 0] = mv_moved[p];
        out[w + 1] = ring_hi[p];
        out[w + 2] = mv_tail[p];
        out[w + 3] = mv_max_n[p];
        out[w + 4] = (kRole == kRoleMover && kNPeer > p) ? *mv_probe_frame[p] : 0u;
    }
    static_assert(kNPeer <= 6, "the results block carries six peers (out[58..63] + out[108..127])");
    // ---- DRISC SELF-PROFILING counters (0 on the default path) ----
    //
    // Shipped WITH the feature for the same reason the role split's were: a sampled, capped instrument that
    // reports only "it worked" is indistinguishable from one that captured the wrong 0.5% of the run. These say
    // how much was captured, how much was deliberately discarded, how much was refused for budget, and what it
    // cost -- so "silently truncated" is not a state this can be in.
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
    // The cross-check block: the phase totals over the sweeps the zones cover, so a host that sums zone
    // durations out of the Tracy capture can assert an equality rather than eyeball a plausible shape.
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
    // MUST equal out[73] (self_tail). Anything less is trace LOST IN THE RING at teardown -- the defect the
    // tail flush above exists to prevent, and one that no other counter shows.
    out[87] = self_words_shipped;
    // ---- COMMON-TRIGGER SYNC EVENT counters (0 on the default path) --------------------------------------
    //
    // sync_timeouts MUST be 0. A timeout means this drainer parked at the barrier and was never released, so
    // it contributed no marker to that trigger -- and a trigger with a missing participant would otherwise be
    // read as a tight spread over the drainers that DID answer, which is the most flattering possible way for
    // this measurement to be wrong.
    out[130] = sync_events;
    out[131] = sync_timeouts;
    out[132] = sync_spin_cyc;
    out[133] = stop_sweeps;
    out[134] = static_cast<uint32_t>(total_words - words_at_stop);
    // ---- NoC FOOTPRINT counters (0 on the default path) --------------------------------------------------
    //
    // TWO BLOCKS, NEVER BLENDED. `life` covers every sweep this drain loop ran; `win` covers the workload
    // window only (first sweep that did work .. last sweep that did work, inclusive of the idle sweeps
    // between them). They differ by orders of magnitude and mean different things: a resident drainer polls
    // from device open, so the lifetime figure is dominated by traffic that no workload asked for. Reporting
    // one number for both would be the wrong-population trap this file has already been burned by twice.
    //
    // Word units, not bytes: these are NoC words as the NIU counts them. The host multiplies by the NoC word
    // size ONCE, in one place, rather than every producer of a byte figure guessing at it.
    {
        // Final sample, so the lifetime block includes the last sweep, the exit drain wait and the teardown
        // barrier. Without it the tail of the loop is missing and the totals read slightly low -- the same
        // shape of silent shortfall as the self-profiling tail flush that could never fire (N+41).
        if constexpr (kNocFootprint != 0) {
            nf_sample_regs(&nf);
            for (uint32_t n = 0; n < 2; n++) {
                nf.posted_txns[n] =
                    static_cast<uint32_t>(NOC_STATUS_READ_REG(n, NIU_MST_POSTED_WR_REQ_SENT) - nf.posted_entry[n]);
            }
        }
        // out[88..103]: life[noc][rd_words, rd_txns, wr_words, wr_txns], each 64-bit lo/hi
        // out[104..119]: the window delta, same order.
        //
        // ONE emission loop over a two-entry block table, not two loops. That is a code-size decision, not
        // style: the two-loop version was 32 B over the DRISC code limit with self-profiling also enabled.
        // win_last is turned into the DELTA IN PLACE first so both blocks have the same shape; nothing reads
        // win_last after this point.
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
        // MUST BE ZERO on both NoCs. Non-zero means a posted write happened, and the word totals above count
        // non-posted words only -- so the byte figures would be UNDER-reported. The host warns instead of
        // printing a plausible low number.
        out[125] = nf.posted_txns[0];
        out[126] = nf.posted_txns[1];
        out[127] = nf.win_open ? 1u : 0u;  // did the window ever open? (0 = no sweep did work)
        out[128] = kNocFootprint;          // echo, so the host never guesses whether this block is valid
        out[129] = NOC_WORD_BYTES;         // the byte scale, from the header -- host never hardcodes it
    }
    static_assert(
        kernel_profiler::SPSC_DRAIN_RESULT_WORDS >= 136,
        "the results block must hold the self-profiling, NoC-footprint and stop-drain counters");

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
