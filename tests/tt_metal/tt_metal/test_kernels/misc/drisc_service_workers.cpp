// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The first end-to-end DRISC drainer against REAL producers: Tensix RISCs emitting zones through the
// ordinary DeviceZoneScopedN path. Everything before this drained synthetic tails the host had written.
//
// No socket, no host egress, no tiering. The single question this answers is whether a DRISC can keep
// real producers from blocking -- i.e. whether the flow-control loop closes:
//
//   POLL  one 256 B control-vector read per core (all outstanding, one barrier)
//   DRAIN for each core with work, one 10,240 B whole-core read of the five rings
//   HEAD  publish the five advanced heads in ONE 20 B write -- this is what unblocks the producer
//
// Ordering: poll the control vector, then read the rings only -- never the control vector twice.
//
// This is NOT because a fused read of the whole 10,496 B span would tear. It would not: the control
// vector sits at LOWER addresses than the rings, so one burst samples the tail BEFORE the data, which
// makes the tail conservative relative to the data it authorises -- the safe direction. (The unsafe
// order is data-then-tail, which this layout cannot produce.) Wrap is covered separately, since the
// producer blocks rather than overwrite [head, tail).
//
// The fused shape is therefore worth taking later: it deletes the poll entirely and cost here is
// issue-dominated. It rests on NoC bursts sampling their source in address order -- an assumption
// still to be confirmed against the NoC spec, which is the only reason this kernel keeps two reads.
//
// Head seeding. Tails are MONOTONIC for the whole FW session -- kernel_profiler.hpp seeds wIndex from L1
// once and never resets per launch (init_profiler, "do NOT re-read TAIL_INDEX per launch"). So a drainer
// must not assume the stream starts at zero; if a profiled program already ran, the tails are well past
// it. The heads are the right seed and they cost nothing: they are words 0..4 of the same control vector
// the POLL already fetches.
//
// The NIU must already be in stream mode -- a DRISC in the default NOC2AXI mode cannot initiate NoC.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "internal/tt-1xx/risc_common.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

void kernel_main() {
    constexpr uint32_t kPollBytes = get_compile_time_arg_val(0);  // 256 = 64-word control vector
    constexpr uint32_t kDataBytes = get_compile_time_arg_val(1);  // 10240 = five 2 KB rings
    constexpr uint32_t kRingWords = get_compile_time_arg_val(2);  // 512 = PROFILER_L1_VECTOR_SIZE
    constexpr uint32_t kPollRing = get_compile_time_arg_val(3);
    constexpr uint32_t kDataBuf = get_compile_time_arg_val(4);
    constexpr uint32_t kHeadScratch = get_compile_time_arg_val(5);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(6);
    constexpr uint32_t kQuietStop = get_compile_time_arg_val(7);  // consecutive idle sweeps => done
    constexpr uint32_t kMaxSweeps = get_compile_time_arg_val(8);  // hard cap, so a stuck run still ends

    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kMaxCores = 128;
    constexpr uint32_t kHeadSlots = 16;

    // Offsets come from the shared enum, never from literals. Hardcoding word 5 is exactly how a reader
    // silently stops draining when PROFILER_SPSC_MAX_RISC moves 5 -> 24.
    constexpr uint32_t kHeadWordOffset = kernel_profiler::SPSC_RING_HEAD_0;
    constexpr uint32_t kTailWordOffset = kernel_profiler::SPSC_RING_TAIL_0;
    static_assert(
        (kernel_profiler::SPSC_CONTROL_END * 4u) <= kPollBytes,
        "the SPSC control layout must fit inside the polled control vector");
    static_assert(kDataBytes <= NOC_MAX_BURST_SIZE, "whole-core read must fit one NoC packet");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);     // start of profiler_msg_t on the worker
    const uint32_t ring0_src = get_arg_val<uint32_t>(2);  // first ring, just past the control vector
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(3));

    Noc noc;
    UnicastEndpoint src;

    static uint32_t head_mirror[kMaxCores * kNumRisc];
    static uint8_t seeded[kMaxCores];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
    }

    uint64_t total_words = 0;
    uint32_t sweeps = 0;
    uint32_t visits_with_work = 0;
    uint32_t max_run = 0;
    uint32_t overflows = 0;
    uint32_t checksum = 0;
    uint32_t quiet = 0;
    uint32_t hb_slot = 0;
    bool seen_work = false;

    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps && quiet < kQuietStop) {
        sweeps++;

        // -------- POLL: every core's control vector, all reads outstanding --------
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

            // Seed from the worker's own heads on first sight; the stream may already be far along.
            if (!seeded[c]) {
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    mine[r] = cv[kHeadWordOffset + r];
                }
                seeded[c] = 1;
            }

            uint32_t runs[kNumRisc];
            uint32_t total = 0;
            for (uint32_t r = 0; r < kNumRisc; r++) {
                runs[r] = cv[kTailWordOffset + r] - mine[r];  // monotonic counts, wrap-safe subtraction
                if (runs[r] > max_run) {
                    max_run = runs[r];
                }
                // A lossless producer BLOCKS at capacity, so a run can never exceed the ring. If one
                // does, either the producer overran us or the mirror desynced -- both are data loss.
                if (runs[r] > kRingWords) {
                    overflows++;
                    runs[r] = kRingWords;
                }
                total += runs[r];
            }
            if (total == 0) {
                continue;
            }

            // -------- DRAIN: whole core, one packet --------
            const uint32_t xy = coords[c];
            CoreLocalMem<uint32_t> dst(kDataBuf);
            noc.async_read<NocOptions::DEFAULT, kDataBytes>(
                src, dst, kDataBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = ring0_src}, {});
            noc.async_read_barrier();

            // Prove real marker words moved, not zeros: fold in the first newly-arrived word per lane.
            volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDataBuf);
            for (uint32_t r = 0; r < kNumRisc; r++) {
                if (runs[r] != 0) {
                    checksum += data[r * kRingWords + (mine[r] % kRingWords)];
                }
            }

            // -------- HEAD: publish, which is what actually unblocks the producer --------
            for (uint32_t r = 0; r < kNumRisc; r++) {
                mine[r] += runs[r];
            }
            const uint32_t sc = kHeadScratch + hb_slot * 32u;
            volatile tt_l1_ptr uint32_t* scp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
            for (uint32_t r = 0; r < kNumRisc; r++) {
                scp[r] = mine[r];
            }
            noc_async_write(sc, get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src + kHeadWordOffset * 4u), kNumRisc * 4u);
            hb_slot = (hb_slot + 1u) & (kHeadSlots - 1u);

            sweep_words += total;
            visits_with_work++;
        }

        total_words += sweep_words;
        if (sweep_words != 0) {
            seen_work = true;
            quiet = 0;
        } else if (seen_work) {
            // Only count quiet once the stream has actually started, or the drainer would declare
            // victory during the launch skew before any producer has emitted its first zone.
            quiet++;
        }
    }
    noc_async_write_barrier();  // the last head must land before the kernel exits
    const uint64_t t_end = get_timestamp();

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = static_cast<uint32_t>(total_words & 0xFFFFFFFFu);
    out[3] = static_cast<uint32_t>(total_words >> 32);
    out[4] = sweeps;
    out[5] = visits_with_work;
    out[6] = max_run;
    out[7] = overflows;
    out[8] = checksum;
    out[9] = quiet;
}
