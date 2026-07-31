// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The full path: real Tensix producers -> DRISC -> D2H socket -> host.
//
// drisc_service_workers.cpp proved the flow-control loop closes but threw the data away. This ships it,
// framed, so the host can attribute every word to a (core, lane).
//
// Per sweep:
//   POLL   one 256 B control-vector read per core, all outstanding, one barrier
//   FRAME  for each lane with a run, append a 2-word header to the page and NoC-read the run's words
//          STRAIGHT INTO the page buffer behind it
//   PUSH   when the next frame would not fit, pad and push the page over the socket
//   HEAD   publish the five advanced heads in one 20 B write -- what unblocks the producer
//
// Reading per lane directly into the page is the point. The alternative -- one whole-core 10 KB read
// into a staging buffer, then a local copy of the live words into the page -- costs a device-side
// memcpy and ships dead ring space. Here each read lands exactly where it belongs and only real words
// cross PCIe. It costs more read ISSUES (up to 5 per core rather than 1), which the measurements say is
// the dominant term, so this is deliberately trading issue cost for bytes; see FINDINGS.
//
// Ring wrap is resolved here, not on the host: a run that crosses the ring end is split into two reads
// that land contiguously in the page, so the payload the host sees is already linear.
//
// The NIU must already be in stream mode -- a DRISC in the default NOC2AXI mode cannot initiate NoC,
// and the socket config write has to be able to land in L1.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "internal/tt-1xx/risc_common.h"
#include "pcie_noc_utils.h"

// Parent directory: the JIT include path is the kernel's own directory (that is how the sibling
// pcie_noc_utils.h resolves), so the shared wire-format header needs the relative hop.
#include "../drisc_drain_frame.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kPollBytes = get_compile_time_arg_val(0);  // 256 = 64-word control vector
    constexpr uint32_t kRingWords = get_compile_time_arg_val(1);  // 512 = PROFILER_L1_VECTOR_SIZE
    constexpr uint32_t kPollRing = get_compile_time_arg_val(2);
    constexpr uint32_t kPageBuf = get_compile_time_arg_val(3);
    constexpr uint32_t kHeadScratch = get_compile_time_arg_val(4);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(5);
    constexpr uint32_t kDoneAddr = get_compile_time_arg_val(6);
    constexpr uint32_t kPageBytes = get_compile_time_arg_val(7);
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(8);
    constexpr uint32_t kQuietStop = get_compile_time_arg_val(9);
    constexpr uint32_t kMaxSweeps = get_compile_time_arg_val(10);

    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kMaxCores = 128;
    constexpr uint32_t kHeadSlots = 16;
    constexpr uint32_t kPageWords = kPageBytes / 4;

    constexpr uint32_t kHeadWordOffset = kernel_profiler::SPSC_RING_HEAD_0;
    constexpr uint32_t kTailWordOffset = kernel_profiler::SPSC_RING_TAIL_0;
    constexpr uint32_t kCoreXyOffset = kernel_profiler::SPSC_CORE_XY;
    static_assert(
        (kernel_profiler::SPSC_CONTROL_END * 4u) <= kPollBytes,
        "the SPSC control layout must fit inside the polled control vector");
    // A run can be a whole ring, so a page must hold at least one maximal frame or it could never make
    // progress -- the fit check would fail forever on a freshly flushed page.
    static_assert(kPageWords >= drisc_drain::FRAME_HEADER_WORDS + kRingWords, "page too small for a max frame");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);     // start of profiler_msg_t on the worker
    const uint32_t ring0_src = get_arg_val<uint32_t>(2);  // first ring, just past the control vector
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(3));

    Noc noc;
    UnicastEndpoint src;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);

    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPageBuf);

    static uint32_t head_mirror[kMaxCores * kNumRisc];
    static uint8_t seeded[kMaxCores];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
    }

    uint64_t total_words = 0;
    uint32_t page_words = 0;
    uint32_t pages = 0;
    uint32_t frames = 0;
    uint32_t sweeps = 0;
    uint32_t max_run = 0;
    uint32_t overflows = 0;
    uint32_t quiet = 0;
    uint32_t hb_slot = 0;
    bool seen_work = false;

    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && quiet < kQuietStop) {
        sweeps++;

        // -------- POLL --------
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint32_t xy = coords[c];
            CoreLocalMem<uint32_t> dst(kPollRing + c * kPollBytes);
            noc.async_read<NocOptions::DEFAULT, kPollBytes>(
                src, dst, kPollBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src}, {});
        }
        noc.async_read_barrier();

        uint32_t sweep_words = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            volatile tt_l1_ptr uint32_t* cv =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPollRing + c * kPollBytes);
            uint32_t* mine = &head_mirror[c * kNumRisc];

            // Seed from the worker's own heads on first sight; tails are monotonic for the whole FW
            // session, so the stream may already be far along.
            if (!seeded[c]) {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] = cv[kHeadWordOffset + r];
                }
                seeded[c] = 1;
            }

            const uint32_t xy = coords[c];
            const uint32_t core_xy = cv[kCoreXyOffset];  // identity, free in the poll
            bool touched = false;

            for (uint32_t r = 0; r < kNumRisc; r++) {
                uint32_t run = cv[kTailWordOffset + r] - mine[r];
                if (run == 0) {
                    continue;
                }
                if (run > max_run) {
                    max_run = run;
                }
                // A lossless producer blocks at capacity, so a run can never exceed the ring.
                if (run > kRingWords) {
                    overflows++;
                    run = kRingWords;
                }

                // -------- PUSH if this frame would not fit --------
                if (page_words + drisc_drain::FRAME_HEADER_WORDS + run > kPageWords) {
                    if (page_words < kPageWords) {
                        page[page_words] = drisc_drain::frame_w0(drisc_drain::KIND_PAD, 0, 0);
                    }
                    noc.async_read_barrier();  // every frame's payload must have landed
                    socket_reserve_pages(sender, 1);
                    noc_write_page_chunked(pcie_xy_enc, kPageBuf, pcie_base + sender.write_ptr, kPageBytes);
                    socket_push_pages(sender, 1);
                    socket_notify_receiver(sender);
                    noc_async_write_barrier();  // single page buffer: the write must land before refill
                    pages++;
                    page_words = 0;
                }

                // -------- FRAME: header, then the run read straight in behind it --------
                page[page_words] = drisc_drain::frame_w0(drisc_drain::KIND_DATA, r, run);
                page[page_words + 1] = core_xy;
                uint32_t doff = page_words + drisc_drain::FRAME_HEADER_WORDS;

                uint32_t si = mine[r] % kRingWords;
                uint32_t left = run;
                const uint32_t ring_base = ring0_src + r * kRingWords * 4u;
                while (left > 0) {
                    uint32_t chunk = kRingWords - si;  // to the ring end
                    if (chunk > left) {
                        chunk = left;
                    }
                    CoreLocalMem<uint32_t> dst(kPageBuf + doff * 4u);
                    noc.async_read<NocOptions::DEFAULT, 0>(
                        src,
                        dst,
                        chunk * 4u,
                        {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = ring_base + si * 4u},
                        {});
                    doff += chunk;
                    left -= chunk;
                    si = 0;  // only the first chunk can start mid-ring
                }

                page_words += drisc_drain::FRAME_HEADER_WORDS + run;
                mine[r] += run;
                sweep_words += run;
                frames++;
                touched = true;
            }

            // -------- HEAD write-back: what actually unblocks the producer --------
            if (touched) {
                const uint32_t sc = kHeadScratch + hb_slot * 32u;
                volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    scp[r] = mine[r];
                }
                noc_async_write(sc, get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src + kHeadWordOffset * 4u), kNumRisc * 4u);
                hb_slot = (hb_slot + 1u) & (kHeadSlots - 1u);
            }
        }

        total_words += sweep_words;
        if (sweep_words != 0) {
            seen_work = true;
            quiet = 0;
        } else if (seen_work) {
            quiet++;
        }
    }

    // -------- final flush --------
    if (page_words > 0) {
        if (page_words < kPageWords) {
            page[page_words] = drisc_drain::frame_w0(drisc_drain::KIND_PAD, 0, 0);
        }
        noc.async_read_barrier();
        socket_reserve_pages(sender, 1);
        noc_write_page_chunked(pcie_xy_enc, kPageBuf, pcie_base + sender.write_ptr, kPageBytes);
        socket_push_pages(sender, 1);
        socket_notify_receiver(sender);
        noc_async_write_barrier();
        pages++;
        page_words = 0;
    }
    socket_barrier(sender);
    noc_async_write_barrier();  // the last head must land too
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
    out[7] = max_run;
    out[8] = overflows;

    update_socket_config(sender);

    // Published last, after the socket barrier, so the host only sees `done` once every page is out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = 0xD09E0000u | (pages & 0xFFFFu);
}
