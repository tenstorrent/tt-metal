// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The DRISC streaming-profiler drainer: worker SPSC rings -> DRISC -> D2H socket -> host.
//
// It drains and it ships. It does not interpret. Every frame it emits is the worker's own profiler
// control vector followed by the ring bytes that control vector describes -- so the core's identity, the
// per-RISC progress and the extent of the live data all reach the host from the WORKER, never from here.
// The only word this kernel authors about someone else's data is the head it writes back, which is its
// own progress and nothing else. See SPSC_SPAN_* in hostdevcommon/profiler_common.h for the layout and
// the slice geometry; the host recomputes that geometry with the same helper, so there is nothing on the
// wire the two sides can disagree about.
//
// Per sweep:
//   POLL   one 256 B control-vector read per core, all outstanding, one barrier
//   FRAME  for each core with work: copy its control vector into the frame (heads patched to OUR mirror,
//          which is what the host's geometry must match), then read each aligned ring slice in after it
//   PUSH   the frame, padded to a whole number of socket pages
//   HEAD   publish the advanced heads in one 20 B write -- what unblocks the producer
//
// Slices are cut on 4-word boundaries so both ends of every NoC copy stay 16 B-aligned. That is not a
// nicety: a misaligned NoC transfer is MIS-DELIVERED rather than rejected, and it shows up as a single
// substituted word at a frame boundary -- totals still reconcile perfectly while the host's marker walk
// desynchronises. Cutting to alignment is also why nothing has to be copied locally: each slice is read
// straight into its final place in the frame.
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

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks. Same body as the socket benchmarks'
// pcie_noc_utils.h, inlined here so a production kernel does not reach into a test header.
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
    constexpr uint32_t kPollRing = get_compile_time_arg_val(0);  // per-core polled control vectors
    constexpr uint32_t kFrameBuf = get_compile_time_arg_val(1);  // one frame, staged then pushed
    constexpr uint32_t kHeadScratch = get_compile_time_arg_val(2);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
    constexpr uint32_t kDoneAddr = get_compile_time_arg_val(4);
    constexpr uint32_t kStopAddr = get_compile_time_arg_val(5);  // host-written: 1 = quiesce and exit
    constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(6);
    constexpr uint32_t kQuietStop = get_compile_time_arg_val(7);  // 0 = run until the host says stop
    constexpr uint32_t kMaxSweeps = get_compile_time_arg_val(8);
    constexpr uint32_t kMaxCores = get_compile_time_arg_val(9);

    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
    constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
    constexpr uint32_t kPollBytes = kCtrlWords * 4u;
    constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
    constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
    constexpr uint32_t kPageBytes = kPageWords * 4u;
    // A run can be a whole ring, and an aligned slice never exceeds it, so this bounds a frame exactly.
    constexpr uint32_t kMaxFrameWords = kPrefix + kCtrlWords + kNumRisc * kRingWords;
    // One head-scratch slot per core: a posted head write must not have its scratch reused underneath it.
    constexpr uint32_t kHeadSlots = 128;

    static_assert(kRingWords * 4u <= NOC_MAX_BURST_SIZE, "a ring slice must fit one NoC packet");
    static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");

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

    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kFrameBuf);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    for (uint32_t i = 2; i < kPrefix; i++) {
        frame[i] = 0;  // reserved prefix words, written once
    }

    static uint32_t head_mirror[kMaxCores * kNumRisc];
    static uint8_t seeded[kMaxCores];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
    }

    uint64_t total_words = 0;
    uint32_t pages = 0;
    uint32_t frames = 0;
    uint32_t sweeps = 0;
    uint32_t max_run = 0;
    uint32_t overflows = 0;
    uint32_t quiet = 0;
    uint32_t hb_slot = 0;
    bool seen_work = false;

    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && *stop == 0 && (kQuietStop == 0 || quiet < kQuietStop)) {
        sweeps++;

        // -------- POLL: every core's control vector, all reads outstanding, one barrier --------
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

            // Seed from the worker's own heads on first sight. Tails are monotonic for the whole firmware
            // session -- the stream may already be far along when a drainer arrives.
            if (!seeded[c]) {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] = cv[kernel_profiler::SPSC_RING_HEAD_0 + r];
                }
                seeded[c] = 1;
            }

            kernel_profiler::SpscSpanRun geom[kNumRisc];
            uint32_t live = 0;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                geom[r] =
                    kernel_profiler::spsc_span_run(mine[r], cv[kernel_profiler::SPSC_RING_TAIL_0 + r], kRingWords);
                if (geom[r].run > max_run) {
                    max_run = geom[r].run;
                }
                // A lossless producer blocks at capacity, so a wider run means a torn control-vector read.
                if (cv[kernel_profiler::SPSC_RING_TAIL_0 + r] - mine[r] > kRingWords) {
                    overflows++;
                }
                live += geom[r].run;
            }
            if (live == 0) {
                continue;
            }

            // -------- FRAME: the worker's control vector, then the slices it describes --------
            //
            // The heads are patched to OUR mirror rather than shipped as polled. They are the drainer's
            // own progress, and the host cuts the same slices from them that we just did -- a stale head
            // (our previous write-back not yet visible in this poll) would make the two disagree.
            volatile tt_l1_ptr uint32_t* fctrl = frame + kPrefix;
            for (uint32_t i = 0; i < kCtrlWords; i++) {
                fctrl[i] = cv[i];
            }
            for (uint32_t r = 0; r < kNumRisc; r++) {
                fctrl[kernel_profiler::SPSC_RING_HEAD_0 + r] = mine[r];
            }

            const uint32_t xy = coords[c];
            uint32_t off = kPrefix + kCtrlWords;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                const uint32_t ring_base = ring0_src + r * kRingWords * 4u;
                for (uint32_t s = 0; s < geom[r].nslices; s++) {
                    const uint32_t bytes = geom[r].slice[s].words * 4u;
                    CoreLocalMem<uint32_t> dst(kFrameBuf + off * 4u);
                    noc.async_read<NocOptions::DEFAULT, kRingWords * 4u>(
                        src,
                        dst,
                        bytes,
                        {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = ring_base + geom[r].slice[s].start * 4u},
                        {});
                    off += geom[r].slice[s].words;
                }
            }
            const uint32_t payload_words = off - kPrefix;
            const uint32_t frame_words = kernel_profiler::spsc_span_frame_words(payload_words);
            frame[0] = kernel_profiler::spsc_span_w0();
            frame[1] = payload_words;
            for (uint32_t i = off; i < frame_words; i++) {
                frame[i] = 0;  // pad up to the socket page
            }
            noc.async_read_barrier();  // the frame must be whole before it is pushed

            // -------- PUSH --------
            //
            // A multi-page write is one contiguous burst, so it has to be split where the FIFO wraps;
            // socket_push_pages only wraps the pointer, it does not split the transfer.
            {
                const uint32_t nbytes = frame_words * 4u;
                const uint32_t npages = frame_words / kPageWords;
                socket_reserve_pages(sender, npages);
                const uint32_t fifo_size = sender.downstream_fifo_curr_size;
                const uint32_t first = (sender.write_ptr + nbytes > fifo_size) ? fifo_size - sender.write_ptr : nbytes;
                write_to_host_chunked(pcie_xy_enc, kFrameBuf, pcie_base + sender.write_ptr, first);
                if (first < nbytes) {
                    write_to_host_chunked(pcie_xy_enc, kFrameBuf + first, pcie_base, nbytes - first);
                }
                socket_push_pages(sender, npages);
                socket_notify_receiver(sender);
                noc_async_write_barrier();  // single frame buffer: the write must land before it is refilled
                pages += npages;
            }
            frames++;

            // -------- HEAD write-back: what actually unblocks the producer --------
            //
            // Only now, after the barrier above -- publishing the head before the reads complete frees the
            // producer to overwrite the very slots still in flight. That fails plausibly: the stream
            // decodes for a while and then a mid-marker word is parsed as a header.
            {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] += geom[r].run;
                    sweep_words += geom[r].run;
                }
                const uint32_t sc = kHeadScratch + hb_slot * 32u;
                volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    scp[r] = mine[r];
                }
                noc_async_write(
                    sc,
                    get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src + kernel_profiler::SPSC_RING_HEAD_0 * 4u),
                    kNumRisc * 4u);
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
    *done = 0xD09E0000u | (frames & 0xFFFFu);

    // -------- NIU restore, on the host's word --------
    //
    // NIU_CFG_0 persists until a chip reset, so whoever set stream mode owns putting it back. It has to
    // happen HERE rather than from the host, because by the time the profiler stops the mesh device is
    // already coming down and cannot launch a program -- and it has to happen LAST, because in NOC2AXI
    // mode an inbound address in the DRAM range is forwarded to GDDR, so the host would lose its view of
    // this L1 (`done`, the results, the socket's bytes_acked) the instant we flip.
    //
    // So the host acknowledges with 2 once it has read everything it wants. The wait is bounded: leaving
    // the NIU in stream mode is untidy, but hanging a DRISC at teardown is worse, and the next firmware
    // boot forces NOC2AXI back anyway.
    for (uint32_t spins = 0; spins < 200000000u && *stop != 2u; spins++) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
