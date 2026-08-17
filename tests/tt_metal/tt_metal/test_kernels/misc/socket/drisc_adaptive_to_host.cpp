// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The full drainer: monitor first, then drain only what is worth draining, then push to host.
//
// Per sweep:
//   1. POLL   one 64-word (256 B) control-vector read per core, all outstanding, one barrier
//   2. DECIDE sum (tail - head) across the 5 RISC tails; a core reaching ADAPT_THRESH goes on the list
//   3. DRAIN  whole-core 10 KB reads for listed cores, accumulated into a socket page, pushed when full
//
// This is the adaptive switch with the bulk read wired straight to a D2H socket instead of
// a relay. The deliberate departure is the same as before: no per-RISC fallback below the threshold. A
// read costs ~40 cycles regardless of payload, so five per-lane reads cost 5x one whole-core read that
// fetches the same data plus slack.
//
// L1 is the binding constraint: a 120-core poll ring is 30,720 B, which leaves room for a 4-core
// (40,960 B) page rather than the 8-core page the pure drainer prefers.
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
    constexpr uint32_t kPollBytes = get_compile_time_arg_val(0);     // 256 = 64-word control vector
    constexpr uint32_t kBytesPerCore = get_compile_time_arg_val(1);  // 10240 = whole core
    constexpr uint32_t kCoresPerPage = get_compile_time_arg_val(2);
    constexpr uint32_t kThresholdWords = get_compile_time_arg_val(3);  // ADAPT_THRESH
    constexpr uint32_t kPollRing = get_compile_time_arg_val(4);
    constexpr uint32_t kPageBuf = get_compile_time_arg_val(5);
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(6);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(7);

    constexpr uint32_t kPageBytes = kCoresPerPage * kBytesPerCore;
    constexpr uint32_t kTailWordOffset =
        5;  // = kernel_profiler::SPSC_RING_TAIL_0, the first of the 5 per-RISC tails in the control vector
    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kMaxCores = 256;
    static_assert(kBytesPerCore <= NOC_MAX_BURST_SIZE, "whole-core read must fit one NoC packet");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t num_sweeps = get_arg_val<uint32_t>(1);
    const uint32_t cv_src = get_arg_val<uint32_t>(2);    // control vector on the worker
    const uint32_t bulk_src = get_arg_val<uint32_t>(3);  // first ring, just past the control vector
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(4));

    Noc noc;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    uint32_t bulk_list[kMaxCores];
    uint64_t t_poll = 0;
    uint64_t t_decide = 0;
    uint64_t t_bulk = 0;
    uint64_t t_push = 0;
    uint32_t pages = 0;
    uint32_t bulk_total = 0;
    uint32_t pending_acc = 0;
    uint32_t page_fill = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t sweep = 0; sweep < num_sweeps; sweep++) {
        // -------- 1. poll every core, all reads outstanding --------
        const uint64_t p0 = get_timestamp();
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint32_t xy = coords[c];
            CoreLocalMem<uint32_t> dst(kPollRing + c * kPollBytes);
            noc.async_read<NocOptions::DEFAULT, kPollBytes>(
                src, dst, kPollBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
        }
        noc.async_read_barrier();
        const uint64_t p1 = get_timestamp();

        // -------- 2. the adaptive decision --------
        uint32_t nbulk = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            volatile tt_l1_ptr uint32_t* cv =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPollRing + c * kPollBytes);
            uint32_t full = 0;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                full += cv[kTailWordOffset + r];  // heads are 0 here, so tail - head == tail
            }
            pending_acc += full;
            if (full >= kThresholdWords) {
                bulk_list[nbulk++] = c;
            }
        }
        bulk_total += nbulk;
        const uint64_t p2 = get_timestamp();

        // -------- 3. whole-core drain of the listed cores, pushed a page at a time --------
        // `last` walks forward across pages; charging every page from p2 would count the earlier pages'
        // time again in each subsequent page.
        uint64_t last = p2;
        for (uint32_t i = 0; i < nbulk; i++) {
            const uint32_t xy = coords[bulk_list[i]];
            CoreLocalMem<uint32_t> dst(kPageBuf + page_fill * kBytesPerCore);
            noc.async_read<NocOptions::DEFAULT, kBytesPerCore>(
                src, dst, kBytesPerCore, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = bulk_src}, {});
            page_fill++;

            if (page_fill == kCoresPerPage) {
                noc.async_read_barrier();
                const uint64_t b1 = get_timestamp();

                socket_reserve_pages(sender, 1);
                noc_write_page_chunked(pcie_xy_enc, kPageBuf, pcie_base + sender.write_ptr, kPageBytes);
                socket_push_pages(sender, 1);
                socket_notify_receiver(sender);
                noc_async_write_barrier();  // single page buffer: the write must land before refill
                const uint64_t b2 = get_timestamp();

                t_bulk += b1 - last;  // read time since the previous page boundary
                t_push += b2 - b1;
                last = b2;
                pages++;
                page_fill = 0;
            }
        }
        const uint64_t p3 = get_timestamp();

        t_poll += p1 - p0;
        t_decide += p2 - p1;
        t_bulk += p3 - last;  // trailing issue cost for a partial page, or the whole phase if nbulk==0
    }
    socket_barrier(sender);
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = pages;
    out[3] = bulk_total;
    out[4] = static_cast<uint32_t>(t_poll & 0xFFFFFFFFu);
    out[5] = static_cast<uint32_t>(t_poll >> 32);
    out[6] = static_cast<uint32_t>(t_decide & 0xFFFFFFFFu);
    out[7] = static_cast<uint32_t>(t_decide >> 32);
    out[8] = static_cast<uint32_t>(t_bulk & 0xFFFFFFFFu);
    out[9] = static_cast<uint32_t>(t_bulk >> 32);
    out[10] = static_cast<uint32_t>(t_push & 0xFFFFFFFFu);
    out[11] = static_cast<uint32_t>(t_push >> 32);
    out[12] = pending_acc;

    update_socket_config(sender);
}
