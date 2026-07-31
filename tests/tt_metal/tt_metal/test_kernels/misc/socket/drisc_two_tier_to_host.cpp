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

    constexpr uint32_t kTailWordOffset = 5;
    constexpr uint32_t kNumRisc = 5;
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

    uint64_t t_poll = 0;
    uint64_t t_decide = 0;
    uint64_t t_fetch = 0;
    uint64_t t_push = 0;
    uint32_t pages = 0;
    uint32_t bulk_cores = 0;
    uint32_t partial_cores = 0;
    uint64_t valid_bytes = 0;
    uint32_t fill = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        // -------- 1. poll --------
        const uint64_t p0 = get_timestamp();
        for (uint32_t c = 0; c < num_cores; c++) {
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
            if (kHysteresis && was_bulk[c]) {
                bulk = true;  // assumed; confirmed or cleared when its bulk read lands
            } else {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    runs[r] = cv[kTailWordOffset + r];  // heads are 0 here, so tail - head == tail
                    if (runs[r] >= kBulkThresholdWords) {
                        bulk = true;
                    }
                }
            }
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
            if (bulk) {
                bulk_cores++;
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

    update_socket_config(sender);
}
