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
#include "experimental/drisc_mode.h"
#include "hostdevcommon/profiler_common.h"
#include "internal/tt-1xx/risc_common.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

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

    static_assert(kSpanBytes <= NOC_MAX_BURST_SIZE, "the fused span read must fit one NoC burst");
    static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
    static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // start of profiler_msg_t on the worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));

    Noc noc;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;

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
    uint64_t c_wr_chunk = 0;
    uint64_t c_wr_push = 0;
    uint64_t c_wr_notify = 0;
    uint64_t c_idle = 0;
    uint64_t c_busy = 0;
    uint32_t sweeps_idle = 0;
    uint32_t max_sweep = 0;
    uint32_t max_reserve = 0;

    // Ship `count` adjacent slots as ONE contiguous write. They are already framed in place: nothing is
    // copied, nothing is assembled.
    auto ship_run = [&](uint32_t start, uint32_t count) {
        if (count == 0) {
            return;
        }
        const uint32_t nbytes = count * kSlotBytes;
        const uint32_t npages = count * kPagesPerSlot;
        const uint64_t t0 = get_timestamp();
        socket_reserve_pages(sender, npages);
        const uint64_t t1 = get_timestamp();
        c_reserve += t1 - t0;
        if (static_cast<uint32_t>(t1 - t0) > max_reserve) {
            max_reserve = static_cast<uint32_t>(t1 - t0);
        }
        // A multi-page write is one contiguous burst, so it must be split where the FIFO wraps;
        // socket_push_pages only wraps the pointer, it does not split the transfer.
        const uint32_t base = kStageBase + start * kSlotBytes;
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        const uint32_t first = (sender.write_ptr + nbytes > fifo_size) ? fifo_size - sender.write_ptr : nbytes;
        write_to_host_chunked(pcie_xy_enc, base, pcie_base + sender.write_ptr, first);
        if (first < nbytes) {
            write_to_host_chunked(pcie_xy_enc, base + first, pcie_base, nbytes - first);
        }
        const uint64_t t2 = get_timestamp();
        c_wr_chunk += t2 - t1;
        socket_push_pages(sender, npages);
        const uint64_t t3 = get_timestamp();
        c_wr_push += t3 - t2;
        socket_notify_receiver(sender);
        const uint64_t t4 = get_timestamp();
        c_wr_notify += t4 - t3;
        c_write += t4 - t1;
        pages += npages;
        pushes++;
    };

    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && *stop == 0) {
        sweeps++;
        const uint64_t t_sweep0 = get_timestamp();
        const uint32_t frames_at_sweep_start = frames;

        for (uint32_t base_c = 0; base_c < num_cores; base_c += kNStage) {
            const uint32_t n = (num_cores - base_c) < kNStage ? (num_cores - base_c) : kNStage;

            // -------- BULK: one fused span read per core, all outstanding, one barrier --------
            const uint64_t t_batch0 = get_timestamp();
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t xy = coords[base_c + i];
                CoreLocalMem<uint32_t> dst(kStageBase + i * kSlotBytes + kPrefix * 4u);
                noc.async_read<NocOptions::DEFAULT, kSpanBytes>(
                    src, dst, kSpanBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
            }
            noc.async_read_barrier();
            const uint64_t t_rd = get_timestamp();
            c_read += t_rd - t_batch0;
            const uint64_t flush_at = c_reserve + c_write;

            // Frame in place, then ship adjacent shippable slots together. A core with nothing breaks the
            // run: shipping it would send 10,560 B of dead ring.
            uint32_t run_start = 0, run_len = 0;
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t c = base_c + i;
                const uint32_t slot = kStageBase + i * kSlotBytes;
                volatile tt_l1_ptr uint32_t* cv =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                uint32_t* mine = &head_mirror[c * kNumRisc];

                // Seed from the worker's own heads on first sight. Tails are monotonic for the whole
                // firmware session -- the stream may already be far along when a drainer arrives.
                if (!seeded[c]) {
                    for (uint32_t r = 0; r < kNumRisc; r++) {
                        mine[r] = cv[kernel_profiler::SPSC_RING_HEAD_0 + r];
                    }
                    seeded[c] = 1;
                }

                uint32_t runs[kNumRisc];
                uint32_t live = 0;
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
                    uint32_t run = tail - mine[r];
                    if (run > kRingWords) {
                        // A lossless producer blocks at capacity, so a wider run means a torn snapshot.
                        overflows++;
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

                // -------- FRAME: nothing to do --------
                //
                // The prefix was written once at startup and the payload is the untouched snapshot, so the
                // DRISC writes ZERO words inside a frame. It used to patch the 5 heads to its own mirror;
                // the host reconstructs those instead (head of this frame == tail of the previous frame for
                // that lane, exact because the FIFO is ordered and lossless), and uses the head field that
                // rides along in the control vector as a consistency check rather than a dependency.
                if (run_len == 0) {
                    run_start = i;
                }
                run_len++;

                // -------- HEAD write-back: releases the producer --------
                //
                // Safe here: the payload is a SNAPSHOT already resident in our staging, so those ring slots
                // are free regardless of when the span reaches the host.
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] += runs[r];
                }
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

                frames++;
                total_words += live;
            }
            ship_run(run_start, run_len);
            c_proc += (get_timestamp() - t_rd) - ((c_reserve + c_write) - flush_at);

            // The staging slots are about to be re-read, so every write out of them must have landed.
            const uint64_t t_b0 = get_timestamp();
            noc_async_write_barrier();
            c_barrier += get_timestamp() - t_b0;
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

    socket_barrier(sender);
    noc_async_write_barrier();
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

    update_socket_config(sender);

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
