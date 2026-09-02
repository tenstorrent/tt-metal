// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC adaptive drain -- the steady-state sweep.
//
// One sweep is three phases, structured for a core whose reads are issue-bound rather than
// bandwidth-bound:
//
//   1. POLL   issue a 64-word (256 B) control-vector read to every core, all outstanding, one barrier.
//   2. DECIDE sum (tail - head) over the 5 RISC tails per core; a core whose total reaches
//             ADAPT_THRESH (4 * RING_CAP words) goes on the bulk list.
//   3. BULK   one whole-core read (5 contiguous rings, 10240 B) per listed core, kBulkDepth
//             outstanding at a time.
//
// No per-RISC fallback below the threshold, which an adaptive drain might otherwise do.
// That would be wrong here -- a read costs ~40 cycles regardless of payload, so 5 per-lane reads cost 5x
// one whole-core read that fetches the same data plus slack. On DRISC, over-reading is free and
// per-lane draining is the expensive path.
//
// Reports total sweep cycles and the bulk count so the host can report a sweep period against the
// 8 us kernel-train budget. Timings come from the DRISC's own wall clock.

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/drisc_mode.h"
#include "internal/tt-1xx/risc_common.h"

void kernel_main() {
    constexpr uint32_t kPollBytes = get_compile_time_arg_val(0);       // 256 = 64-word control vector
    constexpr uint32_t kBulkBytes = get_compile_time_arg_val(1);       // 10240 = 5 rings, whole core
    constexpr uint32_t kThresholdWords = get_compile_time_arg_val(2);  // ADAPT_THRESH = 4 * RING_CAP
    constexpr uint32_t kPollRingBase = get_compile_time_arg_val(3);
    constexpr uint32_t kBulkRingBase = get_compile_time_arg_val(4);
    constexpr uint32_t kBulkDepth = get_compile_time_arg_val(5);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(6);

    constexpr uint32_t kTailWordOffset =
        5;  // = kernel_profiler::SPSC_RING_TAIL_0, the first of the 5 per-RISC tails in the control vector
    constexpr uint32_t kNumRisc = 5;
    constexpr uint32_t kMaxCores = 256;

    static_assert(kPollBytes <= NOC_MAX_BURST_SIZE, "poll read must fit one NoC packet");
    static_assert(kBulkBytes <= NOC_MAX_BURST_SIZE, "bulk read must fit one NoC packet");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    const uint32_t cv_src_addr = get_arg_val<uint32_t>(2);    // control vector on the worker
    const uint32_t bulk_src_addr = get_arg_val<uint32_t>(3);  // first ring, right after the control vector
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(4));

    Noc noc;
    UnicastEndpoint src;

    experimental::drisc_set_stream_mode();

    uint64_t timer_overhead = 0;
    {
        constexpr uint32_t kTimerProbes = 1024;
        const uint64_t tp_start = get_timestamp();
        for (uint32_t i = 0; i < kTimerProbes; i++) {
            (void)get_timestamp();
        }
        timer_overhead = (get_timestamp() - tp_start) / kTimerProbes;
    }

    uint32_t bulk_list[kMaxCores];
    uint32_t bulk_total = 0;
    uint32_t pending_acc = 0;

    const uint64_t t_start = get_timestamp();
    for (uint32_t iter = 0; iter < iters; iter++) {
        // -------- phase 1: poll every core, all reads outstanding, single barrier --------
        for (uint32_t c = 0; c < num_cores; c++) {
            const uint32_t xy = coords[c];
            CoreLocalMem<uint32_t> dst(kPollRingBase + c * kPollBytes);
            noc.async_read<NocOptions::DEFAULT, kPollBytes>(
                src, dst, kPollBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = cv_src_addr}, {});
        }
        noc.async_read_barrier();

        // -------- phase 2: the adaptive decision --------
        uint32_t nbulk = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            volatile tt_l1_ptr uint32_t* cv =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPollRingBase + c * kPollBytes);
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

        // -------- phase 3: one whole-core read per core that tripped the threshold --------
        uint32_t i = 0;
        while (i < nbulk) {
            uint32_t slot = 0;
            for (; slot < kBulkDepth && i < nbulk; slot++, i++) {
                const uint32_t xy = coords[bulk_list[i]];
                CoreLocalMem<uint32_t> dst(kBulkRingBase + slot * kBulkBytes);
                noc.async_read<NocOptions::DEFAULT, kBulkBytes>(
                    src, dst, kBulkBytes, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = bulk_src_addr}, {});
            }
            noc.async_read_barrier();
        }
    }
    const uint64_t t_end = get_timestamp();

    experimental::drisc_set_noc2axi_mode();

    uint32_t checksum = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPollRingBase);

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = checksum;
    out[3] = iters;
    out[4] = static_cast<uint32_t>(timer_overhead);
    out[5] = bulk_total;   // whole-core bulk reads performed across all sweeps
    out[6] = pending_acc;  // keeps the decision loop from being optimized away
}
