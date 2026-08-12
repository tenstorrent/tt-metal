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
// Filler→DRAM keeps the whole-slot layout (fixed ring indexing). The mover may LIVE-PACK before PCIe
// (SPSC_SPAN_PACKED_FLAG): only live ring runs cross the wire. Pack uses NoC L1 loopback DMA -- a CPU
// word walk was the FINDINGS 45% tax; memcpy still left mover busy ~50 us. NoC pack is ~10 us proc.

#include <cstdint>
#include <cstring>

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
    // PACING CONTROLLER (FINDINGS N+36). The drainer ships the WHOLE span per core per frame -- 10,560 B,
    // 165 pages -- regardless of how much of it is live, so host cost is frames x 10,560 and the fill ratio
    // is what decides bytes-per-marker. Sweeping continuously against slow producers reads spans that are
    // only ~37% live, which costs ~2x the host bytes for the same payload and is why producer stalls get
    // WORSE as producers get SLOWER. Measured at 120 cores / 6M markers: delay 20 -> 70% fill, 3,341 frames,
    // 67 MB; delay 125 -> 37% fill, 6,383 frames, 120 MB.
    //
    // So pace the sweeps: hold the inter-sweep gap wherever the spans come back >= kFillPct full. This is
    // the closed loop the fixed kGapCycles hook was always meant to drive.
    constexpr uint32_t kFillPct = get_compile_time_arg_val(17);   // 0 = controller off (fixed kGapCycles)
    constexpr uint32_t kGapMaxCycles = get_compile_time_arg_val(18);
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
    // Pacing-controller derived limits. NOTE the target is against LIVE capacity (the rings), not
    // kSpanWords, which also counts the 64-word control vector that ships whether or not it is live.
    constexpr uint32_t kLiveWords = kNumRisc * kRingWords;
    constexpr uint32_t kFillTarget = (kLiveWords * kFillPct) / 100u;
    // The producers are LOSSLESS and block at ring capacity, so the controller must never pace a core into
    // a stall. Above this per-RISC occupancy the gap collapses to 0 regardless of fill.
    constexpr uint32_t kPaceHighWater = (kRingWords * 3u) / 4u;
    constexpr uint32_t kPaceCritical = (kRingWords * 7u) / 8u;  // hard stop: a producer is about to block
    constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    constexpr uint32_t kSlotWords = kPrefix + kSpanWords;  // 2,640
    constexpr uint32_t kSlotBytes = kSlotWords * 4u;       // 10,560
    // Socket page = wire pad quantum (64 B). Live-pack produces variable-length frames, so the page must
    // stay small: a 10,560 B page would force every packed frame back up to a full slot and erase the win.
    // Whole-slot ships (filler→DRAM, ablate) use kPagesPerSlot pages per frame.
    constexpr uint32_t kSocketPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4u;
    constexpr uint32_t kPagesPerSlot = kSlotBytes / kSocketPageBytes;  // 165
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
    // ---- ROLE SPLIT (see the header). 0 = today's full-job drainer, and every arg below is then 0. ----
    constexpr uint32_t kRoleFull = 0, kRoleFiller = 1, kRoleMover = 2;
    constexpr uint32_t kRole = get_compile_time_arg_val(20);
    // LIVE-PACK on the mover: rewrite each DRAM frame into the packed wire layout (SPSC_SPAN_PACKED_FLAG)
    // before the PCIe push. Filler→DRAM stays whole-slot (fixed ring indexing); only host-facing bytes shrink.
    // Host decoder already understands the flag (spsc_marker_decode.hpp).
    // Input depth 2 (not kGenSlots=3) leaves 5 slots ≈ 52 KB for the pack accumulate buffer so both peers
    // of a dual-ring mover usually fit in ONE PCIe ship.
    constexpr bool kLivePack = (kRole == kRoleMover);
    constexpr uint32_t kPackInSlots = kLivePack ? 2u : kNStage;
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
    // Indexed by peer slot. constexpr arrays of compile-time args, so nothing is loaded from memory to reach
    // them; the loop over kNPeer is a fully-known trip count.
    constexpr uint32_t kPeerXYs[2] = {kPeerXY, kPeerXY1};
    constexpr uint32_t kPeerHss[2] = {kPeerHsAddr, kPeerHsAddr1};
    constexpr uint32_t kPeerBanks[2] = {kDramBank, kDramBank1};
    constexpr uint32_t kPeerAddrs[2] = {kDramAddr, kDramAddr1};
    static_assert(kNPeer <= 2, "a mover drains at most two rings (see kNPeerMax on the host)");
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

    static_assert(kSpanBytes <= NOC_MAX_BURST_SIZE, "the fused span read must fit one NoC burst");
    static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
    static_assert(kSlotBytes % kSocketPageBytes == 0, "a slot must be a whole number of socket pages");
    static_assert(kPagesPerSlot == kSlotBytes / kSocketPageBytes, "kPagesPerSlot mismatch");
    static_assert(kSocketPageBytes == 64u, "live-pack assumes 64 B socket pages");

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
        set_sender_socket_page_size(sender, kSocketPageBytes);
    }

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
    volatile tt_l1_ptr uint32_t* mv_probe_f[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 20),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 36)};
    volatile tt_l1_ptr uint32_t* mv_probe_frame[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 24),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 40)};
    volatile tt_l1_ptr uint32_t* mv_live_head[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 28),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 44)};
    volatile tt_l1_ptr uint32_t* mv_live_tail[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 32),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 48)};
    // FILLER: its own handshake block. head is stored locally and read over the NoC by its mover; tail is
    // written over the NoC by that mover and read locally here.
    volatile tt_l1_ptr uint32_t* hs_head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHsAddr + kHsHead);
    volatile tt_l1_ptr uint32_t* hs_tail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHsAddr + kHsTail);
    uint32_t frames_staged = 0;    // FILLER: frames written into DRAM (monotonic; == published head)
    uint32_t frames_flushed = 0;   // FILLER: of those, how many are barrier-flushed and safe to publish
    // MOVER, per peer ring. Nothing here may be shared between peers: one `mv_tail` for two rings would ack
    // frames on one ring that were only read from the other, i.e. hand the filler room it does not have.
    uint32_t mv_tail[2] = {0, 0};   // frames consumed out of peer p's ring (monotonic)
    uint32_t mv_moved[2] = {0, 0};  // frames shipped to the host out of peer p's ring
    uint32_t mv_max_n[2] = {0, 0};  // largest batch moved in one visit to peer p
    // head - tail high-water per ring: how much elastic buffer is REALLY used. A FILLER has one ring and
    // uses slot 0 only.
    uint32_t ring_hi[2] = {0, 0};
    uint32_t ring_blocked = 0;     // FILLER: stage_run calls that had to wait for ring room at all
    uint32_t hs_bad = 0;           // MOVER: head reads that were structurally impossible -- MUST stay 0
    if constexpr (kRole == kRoleFiller) {
        *hs_head = 0;
        // NOT hs_tail: the host zeroes it before launch and the MOVER owns it from then on. Zeroing it here
        // would race a mover that has already started acking.
    }

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
    uint32_t sweep_max_run = 0;   // per-sweep peak occupancy, the controller's safety input
    uint32_t gap = kGapCycles;    // runtime, driven by the pacing controller below
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
    // ~50 ms at 1.35 GHz. Enormously above anything healthy (worst observed credit wait is ~0.1 us), so it
    // never fires in normal operation -- it exists purely to convert "wait forever" into "lose a frame".
    constexpr uint64_t kCreditWaitCycles = 67500000ull;

    // Ship `count` adjacent FULL slots as ONE contiguous write. They are already framed in place: nothing is
    // copied, nothing is assembled.
    // Returns false if this send was dropped (credit wait expired), so an amplified run stops repeating
    // into a consumer that is not acking instead of billing one dropped frame per repeat.
    auto ship_bytes = [&](uint32_t src_l1, uint32_t nbytes) -> bool {
        const uint32_t npages = nbytes / kSocketPageBytes;
        const uint64_t t0 = get_timestamp();
        *phase = kPhaseReserve;
        const bool credited = reserve_pages_bounded(sender, npages, t0 + kCreditWaitCycles, stop);
        *phase = kPhaseWrite;
        if (!credited) {
            *phase = kPhDropped;
            credit_timeouts++;
            c_reserve += get_timestamp() - t0;
            return false;
        }
        const uint64_t t1 = get_timestamp();
        c_reserve += t1 - t0;
        if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
            max_reserve = static_cast<uint32_t>(t1 - t0);
        }
        *phase = kPhWrChunk;
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        const uint32_t first = (sender.write_ptr + nbytes > fifo_size) ? fifo_size - sender.write_ptr : nbytes;
        write_to_host_chunked(pcie_xy_enc, src_l1, pcie_base + sender.write_ptr, first);
        if (first < nbytes) {
            write_to_host_chunked(pcie_xy_enc, src_l1 + first, pcie_base, nbytes - first);
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

    auto ship_once = [&](uint32_t start, uint32_t count) -> bool {
        if (!ship_bytes(kStageBase + start * kSlotBytes, count * kSlotBytes)) {
            dropped_frames += count;
            return false;
        }
        return true;
    };

    // L1→L1 bulk via NoC loopback. CPU memcpy on a DRISC is the FINDINGS 45% tax (~11 cyc/word even when
    // non-volatile); the NIU moves aligned spans for nearly free. Src/dst must be 16 B aligned; size too.
    // Tiny / misaligned edges stay on the CPU.
    auto noc_l1_copy = [&](uint32_t dst_l1, uint32_t src_l1, uint32_t nbytes) {
        if (nbytes == 0) {
            return;
        }
        // Peel to 16 B alignment (same misalignment on both sides after the packed lead pad).
        const uint32_t head_peel = (16u - (dst_l1 & 15u)) & 15u;
        if (head_peel != 0) {
            const uint32_t n = head_peel < nbytes ? head_peel : nbytes;
            memcpy(reinterpret_cast<void*>(dst_l1), reinterpret_cast<const void*>(src_l1), n);
            dst_l1 += n;
            src_l1 += n;
            nbytes -= n;
        }
        const uint32_t noc_n = nbytes & ~15u;
        // Skip NoC setup for crumbs -- issue cost dominates below ~64 B.
        if (noc_n >= 64u) {
            // Pack DMAs on kReadNoc so they can overlap NOC_INDEX credit/PCIe work.
            noc_async_read(get_noc_addr(src_l1, kReadNoc), dst_l1, noc_n, kReadNoc);
            dst_l1 += noc_n;
            src_l1 += noc_n;
            nbytes -= noc_n;
        }
        if (nbytes != 0) {
            memcpy(reinterpret_cast<void*>(dst_l1), reinterpret_cast<const void*>(src_l1), nbytes);
        }
    };

    // Pack one whole-slot frame at `src` into packed wire layout at `dst`. Returns frame size in BYTES
    // (page-aligned). Matches spsc_marker_decode.hpp packed walk: pad to 16 B, (head&3) lead words, then
    // the live run unwrapped. Lead words may be zero -- the host skips them without inspecting contents.
    // Ring runs go through noc_l1_copy (queued); caller must barrier before reading the packed bytes.
    auto pack_frame = [&](uint32_t src, uint32_t dst) -> uint32_t {
        const uint32_t* in = reinterpret_cast<const uint32_t*>(src);
        uint32_t* out_base = reinterpret_cast<uint32_t*>(dst);
        const uint32_t* ctrl = in + kPrefix;
        const uint32_t* rings = ctrl + kCtrlWords;
        uint32_t* out = out_base + kPrefix;
        noc_l1_copy(reinterpret_cast<uint32_t>(out), reinterpret_cast<uint32_t>(ctrl), kCtrlWords * 4u);
        out += kCtrlWords;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            const uint32_t head = ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r];
            const uint32_t tail = ctrl[kernel_profiler::SPSC_RING_TAIL_0 + r];
            const uint32_t run = kernel_profiler::spsc_span_live(head, tail, kRingWords);
            if (run == 0) {
                continue;
            }
            const uint32_t off = static_cast<uint32_t>(out - out_base);
            const uint32_t pad_lead = ((4u - (off & 3u)) & 3u) + (head & 3u);
            if (pad_lead != 0) {
                memset(out, 0, pad_lead * 4u);
                out += pad_lead;
            }
            const uint32_t* ring = rings + r * kRingWords;
            const uint32_t head_mod = head % kRingWords;
            const uint32_t first = (kRingWords - head_mod) < run ? (kRingWords - head_mod) : run;
            noc_l1_copy(reinterpret_cast<uint32_t>(out), reinterpret_cast<uint32_t>(ring + head_mod), first * 4u);
            out += first;
            if (run > first) {
                noc_l1_copy(reinterpret_cast<uint32_t>(out), reinterpret_cast<uint32_t>(ring), (run - first) * 4u);
                out += run - first;
            }
        }
        const uint32_t payload = static_cast<uint32_t>(out - out_base) - kPrefix;
        const uint32_t frame = kernel_profiler::spsc_span_frame_words(payload);
        const uint32_t have = static_cast<uint32_t>(out - out_base);
        if (have < frame) {
            memset(out, 0, (frame - have) * 4u);
        }
        out_base[0] = kernel_profiler::spsc_span_w0() | kernel_profiler::SPSC_SPAN_PACKED_FLAG;
        out_base[1] = payload;
        for (uint32_t k = 2; k < kPrefix; k++) {
            out_base[k] = 0;
        }
        return frame * 4u;
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
        *phase = kPhWrChunk;
        const uint32_t src = kStageBase + start * kSlotBytes;
        const uint32_t slot0 = frames_staged % kDramFrames;
        // The ring is a whole number of frames, so a FRAME never straddles the wrap -- but a RUN of adjacent
        // frames can, and socket_push_pages' trick of only wrapping a pointer does not apply here. Split it,
        // exactly as ship_once splits at the FIFO wrap.
        const uint32_t first = (slot0 + count > kDramFrames) ? (kDramFrames - slot0) : count;
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

    // What process_batch calls. One name, so the sweep body is byte-identical across roles.
    auto emit_run = [&](uint32_t start, uint32_t count) {
        if constexpr (kRole == kRoleFiller) {
            stage_run(start, count);
        } else {
            ship_run(start, count);
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
                kHeadScratch + 32u,
                get_noc_addr(pxy & 0xFFFFu, pxy >> 16, kPeerHss[p] + kHsProbeM),
                4u,
                NOC_INDEX);
            // Barrier per peer: the scratch word is reused by the next peer's write, so it must have landed.
            (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack);
        }
    }

    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && *stop == 0 && !egress_dead) {
        sweeps++;
        *hb = sweeps;
        *phase = kPhasePoll;
        const uint64_t t_sweep0 = get_timestamp();
        const uint32_t frames_at_sweep_start = frames;
        const uint64_t s_read0 = c_read, s_proc0 = c_proc, s_rsv0 = c_reserve, s_wr0 = c_write,
                       s_bar0 = c_barrier;
        const uint64_t words_at_sweep_start = total_words;
        sweep_max_run = 0;

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
        } else if constexpr (kRole == kRoleMover) {
            // ---- MOVER: kNPeer DRAM rings -> staging -> (optional live-pack) -> D2H socket ----
            //
            // No worker grid, no control-vector scan, no head write-back. The frames in the ring are already
            // complete (prefix + span, written by the filler), so this is a copy and a push -- which is why the
            // socket protocol, the host FIFO and the host decoder are all untouched by the role split.
            //
            // LIVE-PACK coalesce: pack every peer's frames into one L1 buffer, barrier pack (frees staging for
            // the next peer), then ONE PCIe ship for the sweep. Host sees fewer/larger FIFO bursts. Pack DMAs
            // use kReadNoc; credit-reserve spins on the host pointer while those DMAs drain.
            //
            // Peers still visit SEQUENTIALLY (shared staging). The write barrier per peer covers the tail write.
            const uint32_t pack_base = kLivePack ? (kStageBase + kPackInSlots * kSlotBytes) : 0u;
            const uint32_t pack_cap = kLivePack ? ((kNStage - kPackInSlots) * kSlotBytes) : 0u;
            uint32_t packed_acc = 0;
            uint32_t packed_frames = 0;

            auto flush_packed = [&]() {
                if constexpr (!kLivePack) {
                    return;
                }
                if (packed_acc == 0) {
                    return;
                }
                // Size is known from CPU layout even while payload DMAs are still in flight on kReadNoc --
                // start the credit wait so a non-zero reserve overlaps the barrier.
                const uint32_t npages = packed_acc / kSocketPageBytes;
                const uint64_t t0 = get_timestamp();
                *phase = kPhaseReserve;
                const bool credited = reserve_pages_bounded(sender, npages, t0 + kCreditWaitCycles, stop);
                noc_async_read_barrier(kReadNoc);
                invalidate_l1_cache();
                *phase = kPhaseWrite;
                if (!credited) {
                    *phase = kPhDropped;
                    credit_timeouts++;
                    dropped_frames += packed_frames;
                    c_reserve += get_timestamp() - t0;
                    packed_acc = 0;
                    packed_frames = 0;
                    return;
                }
                const uint64_t t1 = get_timestamp();
                c_reserve += t1 - t0;
                if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
                    max_reserve = static_cast<uint32_t>(t1 - t0);
                }
                // Inline the PCIe write/push/notify (same as ship_bytes body) now that credits are held and
                // the pack buffer is valid.
                *phase = kPhWrChunk;
                const uint32_t fifo_size = sender.downstream_fifo_curr_size;
                const uint32_t first =
                    (sender.write_ptr + packed_acc > fifo_size) ? fifo_size - sender.write_ptr : packed_acc;
                write_to_host_chunked(pcie_xy_enc, pack_base, pcie_base + sender.write_ptr, first);
                if (first < packed_acc) {
                    write_to_host_chunked(pcie_xy_enc, pack_base + first, pcie_base, packed_acc - first);
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
                packed_acc = 0;
                packed_frames = 0;
            };

            for (uint32_t peer = 0; peer < kNPeer; peer++) {
                const uint32_t pxy = kPeerXYs[peer];
                const uint64_t t_r0 = get_timestamp();
                noc_async_read(
                    get_noc_addr(pxy & 0xFFFFu, pxy >> 16, kPeerHss[peer] + kHsHead), kHeadScratch, 4u, NOC_INDEX);
                // The FREE-function barrier, matching the free-function read above. Noc{}::async_read_barrier() is a
                // different accounting path, and a barrier that watches the wrong counters returns EARLY -- which
                // here would mean reading a head value that has not landed yet.
                noc_async_read_barrier(NOC_INDEX);
                invalidate_l1_cache();
                const uint32_t head = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHeadScratch);
                uint32_t n = head - mv_tail[peer];
                // HANDSHAKE SANITY, and it is not paranoia -- this exact check is what turns a silent capture
                // corruption into a reported number.
                //
                // head is monotonic and the filler can never be more than kDramFrames ahead, so n > kDramFrames is
                // structurally impossible and means the value is not a head at all. Observed for real: releasing the
                // filler's NIU (stop=2) flips it back to NOC2AXI, where an inbound DRAM-range address is forwarded to
                // GDDR instead of terminating at L1 -- so this read started returning GDDR contents (0xF5AE93CB), n
                // underflowed to ~4.1e9, the `n > kNStage` clamp quietly turned that into "7 frames are ready", and
                // the mover shipped 1,800 frames of garbage that no existing counter noticed. Bail instead of clamp.
                if (n > kDramFrames) {
                    hs_bad++;
                    n = 0;
                    egress_dead = true;  // the peer is gone or unreadable; shipping anything more is corruption
                }
                if (n > ring_hi[peer]) {
                    ring_hi[peer] = n;
                }
                if (n != 0) {
                    // Bounded by staging, then by the ring wrap -- clamping at the wrap costs one short read per lap
                    // and keeps this a SINGLE contiguous DRAM read, which matters because reads are what the mover
                    // is made of.
                    // Live-pack: kPackInSlots of staging for DRAM frames; the rest accumulates packed wire bytes.
                    const uint32_t n_cap = kLivePack ? kPackInSlots : kNStage;
                    if (n > n_cap) {
                        n = n_cap;
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
                    c_read += get_timestamp() - t_r0;
                    if (*mv_probe_frame[peer] == 0) {
                        // First frame word the mover ever saw ON THIS RING. The host checks it against spsc_span_w0(),
                        // which proves the filler's DRAM write and this read agree on the address -- end to end, with
                        // no host-side DRAM read needed and no way for a plausible-but-wrong ring address to pass.
                        // Per ring, because the two rings are in different DRAM banks and only one of them may be
                        // wrong.
                        *mv_probe_frame[peer] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase);
                    }
                    // Release the ring region NOW: the bytes are in staging, so the filler may reuse it immediately.
                    // Doing this before the push (rather than after) is what keeps the filler off the ring ceiling.
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
                    if constexpr (kLivePack) {
                        const uint64_t t_pack0 = get_timestamp();
                        uint64_t t_pack_mark = t_pack0;
                        for (uint32_t i = 0; i < n; i++) {
                            // Worst-case packed frame is a whole slot; flush before we would overrun the pack buf.
                            if (packed_acc + kSlotBytes > pack_cap) {
                                c_proc += get_timestamp() - t_pack_mark;
                                flush_packed();
                                const uint64_t t_bflush = get_timestamp();
                                *phase = kPhBar2;
                                if (!write_barrier_bounded(t_bflush + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack)) {
                                    egress_dead = true;
                                    break;
                                }
                                c_barrier += get_timestamp() - t_bflush;
                                t_pack_mark = get_timestamp();
                            }
                            if (egress_dead) {
                                break;
                            }
                            packed_acc += pack_frame(kStageBase + i * kSlotBytes, pack_base + packed_acc);
                            packed_frames++;
                        }
                        // Staging is the NoC-read source -- must complete before the next peer overwrites it.
                        noc_async_read_barrier(kReadNoc);
                        invalidate_l1_cache();
                        c_proc += get_timestamp() - t_pack_mark;
                    } else {
                        ship_run(0, n);
                    }
                    frames += n;
                    mv_moved[peer] += n;
                    if (n > mv_max_n[peer]) {
                        mv_max_n[peer] = n;
                    }
                    // ONE barrier covers the PCIe push (staging is about to be refilled -- by the NEXT PEER as well
                    // as the next sweep), the tail write (the filler cannot see room until it lands) and the reuse of
                    // the +32 scratch word the tail write sources from.
                    const uint64_t t_b0 = get_timestamp();
                    *phase = kPhBar2;
                    if (!write_barrier_bounded(t_b0 + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack)) {
                        egress_dead = true;
                    }
                    c_barrier += get_timestamp() - t_b0;
                } else {
                    c_read += get_timestamp() - t_r0;
                }
                // Never start the second peer once egress is dead: staging may hold unflushed bytes, and an
                // impossible head on one ring says nothing good about the other.
                if (egress_dead) {
                    break;
                }
            }  // for peer
            if constexpr (kLivePack) {
                if (packed_acc != 0 && !egress_dead) {
                    flush_packed();
                    const uint64_t t_b1 = get_timestamp();
                    *phase = kPhBar2;
                    if (!write_barrier_bounded(t_b1 + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack)) {
                        egress_dead = true;
                    }
                    c_barrier += get_timestamp() - t_b1;
                }
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
                if (r0 > kRingWords) { overflows++; r0 = kRingWords; }
                if (r1 > kRingWords) { overflows++; r1 = kRingWords; }
                if (r2 > kRingWords) { overflows++; r2 = kRingWords; }
                if (r3 > kRingWords) { overflows++; r3 = kRingWords; }
                if (r4 > kRingWords) { overflows++; r4 = kRingWords; }
                uint32_t peak = r0;
                if (r1 > peak) { peak = r1; }
                if (r2 > peak) { peak = r2; }
                if (r3 > peak) { peak = r3; }
                if (r4 > peak) { peak = r4; }
                if (peak > max_occ) { max_occ = peak; }
                if (peak > sweep_max_run) { sweep_max_run = peak; }
                const uint32_t live = r0 + r1 + r2 + r3 + r4;
                if (live == 0) {
                    emit_run(run_start, run_len);
                    run_len = 0;
                    continue;
                }
                if (run_len == 0) {
                    run_start = sl;
                }
                run_len++;

                // Head write-back releases the producer. Safe at once: the payload is a snapshot already
                // resident in staging, so those ring slots are free regardless of when it reaches the host.
                m0 += r0; m1 += r1; m2 += r2; m3 += r3; m4 += r4;
                mine[0] = m0; mine[1] = m1; mine[2] = m2; mine[3] = m3; mine[4] = m4;
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
                scp[0] = m0; scp[1] = m1; scp[2] = m2; scp[3] = m3; scp[4] = m4;
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
            emit_run(run_start, run_len);
            gen_shipped[g] = true;
            // SATURATING. The nested ship_run time is subtracted out so it is not double-counted against
            // proc, but if that term ever exceeds the elapsed span the unsigned subtract wraps -- observed
            // once as "proc 18727729111430.1%", which silently corrupts the whole phase breakdown.
            {
                const uint64_t span = get_timestamp() - t_p0;
                const uint64_t nested = (c_reserve + c_write) - flush_at;
                c_proc += (span > nested) ? (span - nested) : 0;
            }
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
                publish_head();  // those DRAM writes are now flushed, so the mover may have them
                gen_shipped[gen] = false;
            }

            const uint64_t t_batch0 = get_timestamp();
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
                        src, dst2, kSpanBytes - kHalf,
                        {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src + kHalf}, {});
                } else if constexpr (kReadSplit == 1) {
                    // Alternate cores between the two NoCs so both have transactions outstanding.
                    if ((i & 1u) == 0u) {
                        noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                            src, dst, kSpanBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
                    } else {
                        noc_b.async_read<NocOptions::DEFAULT, kSpanBytes>(
                            src, dst, kSpanBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
                    }
                } else {
                    noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                        src, dst, kSpanBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
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
            noc.async_read_barrier();
            if constexpr (kReadSplit != 0) {
                noc_b.async_read_barrier();  // staging reuse is only safe once BOTH read NoCs have landed
            }
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
            } else {
                publish_head();
            }
            c_barrier += get_timestamp() - t_b0;
        }

        }

        const uint32_t sweep_cyc = static_cast<uint32_t>(get_timestamp() - t_sweep0);
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
                gap = 0;                 // about to block a lossless producer: drain now, fill be damned
            } else if (sweep_max_run >= kPaceHighWater) {
                gap -= gap >> 2;         // getting warm: ease off 25%, do not abandon pacing
            } else {
                const uint32_t frames_now = frames - frames_at_sweep_start;
                if (frames_now != 0) {
                    const uint32_t mean_fill = static_cast<uint32_t>((total_words - words_at_sweep_start) / frames_now);
                    if (mean_fill < kFillTarget) {
                        // Under-full: wait longer. MULTIPLICATIVE with an additive floor -- the old
                        // `1 + (err>>3)` crept up ~1 cycle per sweep and could not reach the thousands of
                        // cycles a slow producer needs within a run.
                        uint32_t inc = gap >> 2;
                        if (inc < 64u) {
                            inc = 64u;
                        }
                        gap = (gap + inc > kGapMaxCycles) ? kGapMaxCycles : gap + inc;
                    } else if (mean_fill > kFillTarget) {
                        gap -= gap >> 3;  // over-target: ease down and settle
                    }
                }
            }
        }
        if (gap != 0) {
            const uint64_t until = get_timestamp() + gap;
            while (get_timestamp() < until) {
            }
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
    (void)write_barrier_bounded(get_timestamp() + kCreditWaitCycles, dbg_hw_ack, dbg_sw_ack);
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
    out[50] = ring_hi[0];   // head - tail high-water on peer 0's ring (the filler's own ring), in frames
    out[51] = ring_blocked; // FILLER: pushes that had to wait for ring room
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
    static_assert(kNPeer <= 2, "the results block only carries two peers (out[58..63])");

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
