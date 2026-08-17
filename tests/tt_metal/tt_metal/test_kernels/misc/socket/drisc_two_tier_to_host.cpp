// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-tier adaptive drainer.
//
// Per sweep:
//   1. POLL    one 64-word (256 B) control-vector read per core, all outstanding, one barrier
//   2. DECIDE  per core, compute run = tail - head for each of the 5 RISCs
//   3a. BULK   if ANY lane is at or above kBulkThresholdWords (70% of the ring), take the whole core
//              in one read -- control vector plus all five rings, 10,496 B, one NoC packet
//   3b. PARTIAL otherwise, read only the valid run of each non-empty lane
//
// The point of the partial tier is egress, not ingest. A read costs ~40 cycles regardless of payload,
// so per-lane reads are not much cheaper to *issue* -- but they fetch only live markers, and egress is
// the scarce resource at ~24 GB/s. Bulk-reading a near-empty core would spend 10 KB of host bandwidth
// to deliver a few hundred bytes of markers.
//
// Variable-size items are packed into fixed-size socket pages, since the socket page size is fixed.
// An item that does not fit forces the current page out (padded), so pushed bytes >= valid bytes; the
// kernel reports both so the padding waste is visible. Packing continues across sweeps -- the stream
// is continuous.
//
// The NIU must already be in stream mode -- set by a prior program so the socket config lands in L1.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "hostdevcommon/profiler_common.h"
#include "internal/tt-1xx/risc_common.h"
#include "pcie_noc_utils.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kPollBytes = get_compile_time_arg_val(0);           // 256
    constexpr uint32_t kRingCapBytes = get_compile_time_arg_val(1);        // 2048 per lane
    constexpr uint32_t kCoreSpan = get_compile_time_arg_val(2);            // 10496 = cv + 5 rings
    constexpr uint32_t kBulkThresholdWords = get_compile_time_arg_val(3);  // 70% of the ring, in words
    constexpr uint32_t kPageBytes = get_compile_time_arg_val(4);
    constexpr uint32_t kPollRing = get_compile_time_arg_val(5);
    constexpr uint32_t kPageBuf = get_compile_time_arg_val(6);
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(7);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(8);
    // Hysteresis: a core that went bulk last sweep is assumed still bulk, so its poll is skipped and it
    // goes straight to the whole-core read. That read carries a fresh control vector, so the core's tier
    // is re-evaluated from data already fetched. Only cores currently in the partial tier are polled.
    // A mispredict costs one bulk read of a core that has cooled -- which is what the 70%-vs-100%
    // headroom in the threshold absorbs.
    constexpr uint32_t kHysteresis = get_compile_time_arg_val(9);
    // No adaptive logic at all: skip the poll, and unconditionally read all five rings of every core
    // as five separate full-ring transfers. The naive baseline the two tiers are measured against --
    // same bytes as a whole-core read, but 5 transfers per core instead of 1.
    constexpr uint32_t kNoAdaptive = get_compile_time_arg_val(10);
    // Paced mode: always bulk, but space the sweeps so each one arrives when the rings are worth
    // reading. A closed loop on observed occupancy -- above the high watermark the drainer is behind
    // and runs flat out; below the low watermark it waits longer.
    //
    // Production is EMULATED: occupancy is modelled as rate x elapsed rather than read from the
    // workers, because the harness primes static tails and a real controller needs occupancy to
    // respond to the pacing. Reads, pushes and all timing are real; only the occupancy value is
    // synthetic, and it drives both the controller and the live-byte accounting.
    constexpr uint32_t kPaced = get_compile_time_arg_val(11);
    constexpr uint32_t kProdMilliWordsPerUs = get_compile_time_arg_val(12);  // per lane
    constexpr uint32_t kHighWatermark = get_compile_time_arg_val(13);        // words
    constexpr uint32_t kLowWatermark = get_compile_time_arg_val(14);         // words
    constexpr uint32_t kClkMhz = get_compile_time_arg_val(15);
    constexpr uint32_t kDelayStepCycles = get_compile_time_arg_val(16);
    // Head write-back. The ring is flow-controlled: SPSC_RING_HEAD_0..4 in the worker's control vector
    // are CONSUMER-written (profiler_common.h:157-161), so advancing them is what unblocks the
    // producers. Without it a real
    // producer stalls. Five head words are staged locally and published in ONE 20 B NoC write rather
    // than five inline writes, since issue cost dominates. Posted -- a stale head only makes the
    // producer conservative, never unsafe.
    constexpr uint32_t kWriteBackHeads = get_compile_time_arg_val(17);
    constexpr uint32_t kHeadScratch = get_compile_time_arg_val(18);
    constexpr uint32_t kHeadSlots = 16;
    constexpr uint32_t kMirrorCores = 128;
    constexpr uint32_t kRingCapWordsK = kRingCapBytes / 4;

    // Offsets come from the shared enum, never from literals. Hardcoding word 5 is exactly how a
    // reader silently stops draining when PROFILER_SPSC_MAX_RISC moves 5 -> 24: the tails
    // relocate to 24..28, the reader keeps reading 5..9, those read 0, tail always equals head, and
    // every producing RISC blocks forever.
    constexpr uint32_t kHeadWordOffset = kernel_profiler::SPSC_RING_HEAD_0;
    constexpr uint32_t kTailWordOffset = kernel_profiler::SPSC_RING_TAIL_0;
    constexpr uint32_t kCoreXyOffset = kernel_profiler::SPSC_CORE_XY;
    constexpr uint32_t kNumRisc = 5;
    static_assert(
        (kernel_profiler::SPSC_CONTROL_END * 4u) <= kPollBytes,
        "the SPSC control layout must fit inside the polled control vector");
    constexpr uint32_t kMaxCores = 256;
    constexpr uint32_t kMaxBulkPerPage = kPageBytes / kCoreSpan + 1;
    static_assert(kCoreSpan <= NOC_MAX_BURST_SIZE, "whole-core read must fit one NoC packet");
    static_assert(kCoreSpan <= kPageBytes, "a bulk item must fit in one page");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t num_sweeps = get_arg_val<uint32_t>(1);
    const uint32_t cv_src = get_arg_val<uint32_t>(2);     // start of profiler_msg_t on the worker
    const uint32_t ring0_src = get_arg_val<uint32_t>(3);  // first ring, just past the control vector
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(4));

    Noc noc;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    uint8_t was_bulk[kMaxCores];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        was_bulk[i] = 0;
    }
    // Bulk items placed in the page still in flight, for deferred tier re-evaluation at flush.
    uint32_t rec_core[kMaxBulkPerPage];
    uint32_t rec_off[kMaxBulkPerPage];
    uint32_t n_recs = 0;
    uint32_t polls = 0;

    uint32_t src_w0_acc = 0;
    uint64_t t_head = 0;
    uint32_t hb_slot = 0;
    uint32_t head_ctr = 0;
    // Local head mirror. The head is consumer-written, so the drainer already knows it -- reading it
    // back from the worker would be pointless work. Only the tail is producer-written and must be
    // fetched, and it rides along in the control vector.
    static uint32_t head_mirror[kMirrorCores * kNumRisc];
    uint64_t delay_cycles = 0;
    uint64_t last_sweep = get_timestamp();
    uint64_t t_wait = 0;
    uint32_t overflows = 0;
    uint64_t occ_sum = 0;

    uint64_t t_poll = 0;
    uint64_t t_decide = 0;
    uint64_t t_fetch = 0;
    uint64_t t_push = 0;
    uint32_t pages = 0;
    uint32_t bulk_cores = 0;
    uint32_t partial_cores = 0;
    uint64_t valid_bytes = 0;  // bytes actually transferred (bulk counts whole rings)
    uint64_t live_bytes = 0;   // bytes that are real markers -- the only thing the host cares about
    uint32_t fill = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        uint32_t paced_occ = 0;
        if constexpr (kPaced) {
            const uint64_t w0 = get_timestamp();
            while ((get_timestamp() - last_sweep) < delay_cycles) {
            }
            const uint64_t now = get_timestamp();
            t_wait += now - w0;
            const uint64_t elapsed_us = (now - last_sweep) / kClkMhz;
            last_sweep = now;
            uint64_t occ = (kProdMilliWordsPerUs * elapsed_us) / 1000u;
            if (occ >= kRingCapWordsK) {
                occ = kRingCapWordsK;
                overflows++;  // a real producer would have wrapped and lost markers here
            }
            paced_occ = static_cast<uint32_t>(occ);
            occ_sum += paced_occ;
        }

        // -------- 1. poll --------
        const uint64_t p0 = get_timestamp();
        for (uint32_t c = 0; c < num_cores && !kNoAdaptive && !kPaced; c++) {
            if constexpr (kHysteresis) {
                if (was_bulk[c]) {
                    continue;  // its bulk read will carry a fresh control vector
                }
            }
            const uint32_t xy = coords[c];
            CoreLocalMem<uint32_t> dst(kPollRing + c * kPollBytes);
            noc.async_read<NocOptions::DEFAULT, kPollBytes>(
                src, dst, kPollBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
            polls++;
        }
        noc.async_read_barrier();
        const uint64_t p1 = get_timestamp();
        t_poll += p1 - p0;

        // -------- 2/3. decide, then fetch at the chosen granularity --------
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint64_t d0 = get_timestamp();
            volatile tt_l1_ptr uint32_t* cv =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPollRing + c * kPollBytes);
            uint32_t runs[kNumRisc];
            bool bulk = false;
            if constexpr (kPaced) {
                bulk = true;  // paced mode is always bulk; the pacing chooses when, not how much
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    runs[r] = paced_occ;
                }
            } else if constexpr (kNoAdaptive) {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    runs[r] = kRingCapBytes / 4;  // whole ring, unconditionally
                }
            } else if (kHysteresis && was_bulk[c]) {
                bulk = true;  // assumed; confirmed or cleared when its bulk read lands
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    runs[r] = 0;  // tails unknown without a poll; live-byte accounting is off in this mode
                }
            } else {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    runs[r] = cv[kTailWordOffset + r];  // heads are 0 here, so tail - head == tail
                    if (runs[r] >= kBulkThresholdWords) {
                        bulk = true;
                    }
                }
            }
            for (uint32_t r = 0; r < kNumRisc; r++) {
                live_bytes += runs[r] * 4u;
            }
            // Identity comes from the core itself: (y<<16)|x in the control vector, stamped once by
            // BRISC FW. Nothing is constructed or injected here, and nothing is looked up host-side.
            src_w0_acc += cv[kCoreXyOffset];
            const uint32_t xy = coords[c];
            const uint64_t d1 = get_timestamp();
            t_decide += d1 - d0;

            // Emit one item (bulk) or up to five (partial). An item that does not fit flushes the page.
            uint32_t n_items = bulk ? 1u : kNumRisc;
            for (uint32_t it = 0; it < n_items; it++) {
                uint32_t bytes;
                uint32_t remote_addr;
                if (bulk) {
                    bytes = kCoreSpan;
                    remote_addr = cv_src;
                } else {
                    bytes = (runs[it] * 4u + 15u) & ~15u;  // 16 B aligned
                    remote_addr = ring0_src + it * kRingCapBytes;
                    if (bytes == 0) {
                        continue;
                    }
                }

                if (fill + bytes > kPageBytes) {
                    // Flush: the reads issued into this page must land before it is written out.
                    const uint64_t f0 = get_timestamp();
                    noc.async_read_barrier();
                    const uint64_t f1 = get_timestamp();
                    socket_reserve_pages(sender, 1);
                    noc_write_page_chunked(pcie_xy_enc, kPageBuf, pcie_base + sender.write_ptr, kPageBytes);
                    socket_push_pages(sender, 1);
                    socket_notify_receiver(sender);
                    noc_async_write_barrier();  // single page buffer
                    const uint64_t f2 = get_timestamp();
                    t_fetch += f1 - f0;
                    t_push += f2 - f1;
                    pages++;
                    fill = 0;
                    if constexpr (kHysteresis) {
                        // The page has landed, so every bulk item in it carries a current control
                        // vector: re-evaluate those cores' tiers before the buffer is reused.
                        for (uint32_t k = 0; k < n_recs; k++) {
                            volatile tt_l1_ptr uint32_t* bcv =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPageBuf + rec_off[k]);
                            uint8_t still = 0;
                            for (uint32_t r = 0; r < kNumRisc; r++) {
                                if (bcv[kTailWordOffset + r] >= kBulkThresholdWords) {
                                    still = 1;
                                }
                            }
                            was_bulk[rec_core[k]] = still;
                        }
                        n_recs = 0;
                    }
                }

                if constexpr (kHysteresis) {
                    if (bulk && n_recs < kMaxBulkPerPage) {
                        rec_core[n_recs] = c;
                        rec_off[n_recs] = fill;
                        n_recs++;
                    }
                }
                CoreLocalMem<uint32_t> dst(kPageBuf + fill);
                noc.async_read<NocOptions::DEFAULT, kCoreSpan>(
                    src, dst, bytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = remote_addr}, {});
                fill += bytes;
                valid_bytes += bytes;
            }
            if constexpr (kWriteBackHeads) {
                const uint64_t h0 = get_timestamp();
                // Advance the local mirror by what was just drained. No read of the worker: the head
                // is ours, we wrote it last time.
                uint32_t* mine = &head_mirror[(c & (kMirrorCores - 1u)) * kNumRisc];
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] += runs[r];
                }
                head_ctr += mine[0];
                // Stage the five new heads, publish in one write.
                const uint32_t sc = kHeadScratch + hb_slot * 32u;
                volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    scp[r] = mine[r];
                }
                noc_async_write(sc, get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src + kHeadWordOffset * 4u), kNumRisc * 4u);
                hb_slot = (hb_slot + 1u) & (kHeadSlots - 1u);
                t_head += get_timestamp() - h0;
            }
            if (bulk) {
                bulk_cores++;
                (void)0;
                if constexpr (kHysteresis) {
                    was_bulk[c] = 1;  // provisional; the flush confirms or clears it
                }
            } else {
                partial_cores++;
                if constexpr (kHysteresis) {
                    was_bulk[c] = 0;
                }
            }
        }
        if constexpr (kPaced) {
            // Integral controller: ease the delay down when rings run hot, up when they run cold.
            // Zeroing on overshoot would oscillate.
            constexpr uint64_t kMaxDelay = 4000000;  // ~3 ms, enough for very low production rates
            if (paced_occ > kHighWatermark) {
                delay_cycles = (delay_cycles > kDelayStepCycles) ? delay_cycles - kDelayStepCycles : 0;
            } else if (paced_occ < kLowWatermark) {
                delay_cycles =
                    (delay_cycles + kDelayStepCycles > kMaxDelay) ? kMaxDelay : delay_cycles + kDelayStepCycles;
            }
        }
    }
    // Flush whatever is left so the host sees a whole number of pages.
    if (fill > 0) {
        noc.async_read_barrier();
        socket_reserve_pages(sender, 1);
        noc_write_page_chunked(pcie_xy_enc, kPageBuf, pcie_base + sender.write_ptr, kPageBytes);
        socket_push_pages(sender, 1);
        socket_notify_receiver(sender);
        pages++;
    }
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = pages;
    out[3] = bulk_cores;
    out[4] = partial_cores;
    out[5] = static_cast<uint32_t>(valid_bytes & 0xFFFFFFFFu);
    out[6] = static_cast<uint32_t>(valid_bytes >> 32);
    out[7] = static_cast<uint32_t>(t_poll & 0xFFFFFFFFu);
    out[8] = static_cast<uint32_t>(t_poll >> 32);
    out[9] = static_cast<uint32_t>(t_decide & 0xFFFFFFFFu);
    out[10] = static_cast<uint32_t>(t_fetch & 0xFFFFFFFFu);
    out[11] = static_cast<uint32_t>(t_fetch >> 32);
    out[12] = static_cast<uint32_t>(t_push & 0xFFFFFFFFu);
    out[13] = static_cast<uint32_t>(t_push >> 32);
    out[14] = polls;
    out[15] = static_cast<uint32_t>(live_bytes & 0xFFFFFFFFu);
    out[16] = static_cast<uint32_t>(live_bytes >> 32);
    out[17] = overflows;
    out[18] = static_cast<uint32_t>(occ_sum / (num_sweeps ? num_sweeps : 1));
    out[19] = static_cast<uint32_t>(t_wait & 0xFFFFFFFFu);
    out[20] = static_cast<uint32_t>(t_wait >> 32);
    out[21] = static_cast<uint32_t>(delay_cycles);
    out[22] = static_cast<uint32_t>(t_head & 0xFFFFFFFFu);
    out[23] = static_cast<uint32_t>(t_head >> 32);
    out[24] = head_ctr;    // keeps the head staging from being optimized away
    out[25] = src_w0_acc;  // keeps the identity read from being optimized away

    update_socket_config(sender);
}
