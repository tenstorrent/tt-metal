// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// DRISC scatter-read microbenchmark.
//
// Models the ingest half of a profiler drainer hosted on a DRAM core: walk a table of
// worker-core NoC coords, pull a profiler-sized marker burst out of each core's L1 into a
// local landing ring, and repeat. Reports device cycles for the whole timed loop so the
// host can derive aggregate GB/s, ns per marker, and ns per core visit (the poll floor).
//
// Two knobs, both compile-time so the issue sequence codegens cleanly:
//   kMarkersPerRead  bytes pulled per core visit (K * 8B). Sweeping this finds the knee
//                    where per-transaction overhead stops dominating.
//   kReadsInFlight   how many reads are issued before the barrier -- read-level parallelism via
//                    the NoC's own outstanding-read tracking, not multiple loads in flight.
//
// The landing ring is kReadsInFlight slots of kMarkersPerRead markers. Slots are only
// reused after a barrier, so there is no write-after-read hazard on the destination.
//
// NOTE: this measures reads only. The egress side (D2H socket push) is deliberately absent
// so the number is a clean ingest ceiling.

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/drisc_mode.h"
#include "internal/tt-1xx/risc_common.h"

void kernel_main() {
    constexpr uint32_t kMarkersPerRead = get_compile_time_arg_val(0);
    constexpr uint32_t kReadsInFlight = get_compile_time_arg_val(1);
    constexpr uint32_t kRingBase = get_compile_time_arg_val(2);
    constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
    // Poll mode: after each barrier, examine the landed control vectors the way an adaptive drain
    // switch does -- sum (tail - head) across the 5 RISC tails per core,
    //   for (r) full += tails[r] - heads[c*NRISC + r];
    // so the measured per-core cost includes the CPU work a real poll pays, not just the NoC read.
    // Heads are not advanced (they stay 0), which is cost-equivalent but not functionally a drain.
    constexpr uint32_t kPollExamine = get_compile_time_arg_val(4);
    // Distinct landing slots. Normally == kReadsInFlight (every outstanding read gets its own buffer).
    // Setting it lower lets reads-in-flight exceed what DRISC L1 can hold, so overlapping reads land on
    // top of each other. That is NOT a valid drainer configuration -- it corrupts the landed data -- but
    // it measures the transport honestly, isolating "how deep can the NIU usefully go" from "how much
    // buffer do we have". Only ever use kRingSlots < kReadsInFlight for bandwidth measurement.
    constexpr uint32_t kRingSlots = get_compile_time_arg_val(5);
    constexpr uint32_t kTailWordOffset =
        5;  // = kernel_profiler::SPSC_RING_TAIL_0, the first of the 5 per-RISC tails in the control vector
    constexpr uint32_t kNumRisc = 5;
    static_assert(kRingSlots >= 1, "need at least one landing slot");
    static_assert(!kPollExamine || kRingSlots >= kReadsInFlight, "poll-examine needs one slot per read");

    // Real kernel_profiler marker: 2 words.
    constexpr uint32_t kMarkerBytes = 8;
    constexpr uint32_t kBytesPerRead = kMarkersPerRead * kMarkerBytes;

    // Keep every read on the one-packet path; passing kBytesPerRead as max_page_size below
    // only selects that path if the burst actually fits, otherwise the reads get chunked and
    // the per-visit cost being measured is no longer what the name says.
    static_assert(kBytesPerRead <= NOC_MAX_BURST_SIZE, "marker burst must fit one NoC packet");
    static_assert(
        !kPollExamine || kBytesPerRead >= (kTailWordOffset + kNumRisc) * 4,
        "poll-examine needs the read to cover the control vector's tail words");

    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    const uint32_t src_addr = get_arg_val<uint32_t>(2);
    // Packed virtual coords, one word per polled core: x in the low half, y in the high half.
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(3));

    Noc noc;
    UnicastEndpoint src;

    // Required before the DRISC can initiate any NoC traffic at all -- in the default
    // NOC2AXI mode its NIU is a DRAM subordinate only.
    experimental::drisc_set_stream_mode();

    // Cost of reading the on-DRISC wall clock itself, measured on the DRISC. This is the noise floor
    // for every cycle number below -- a per-read cost of ~40 cycles cannot be timed with an instrument
    // that costs more than that, so phases are separated by ablation (kPollExamine on/off) rather than
    // by bracketing them with timer reads.
    uint64_t timer_overhead = 0;
    {
        constexpr uint32_t kTimerProbes = 1024;
        const uint64_t tp_start = get_timestamp();
        for (uint32_t i = 0; i < kTimerProbes; i++) {
            (void)get_timestamp();
        }
        const uint64_t tp_end = get_timestamp();
        timer_overhead = (tp_end - tp_start) / kTimerProbes;
    }

    uint32_t pending = 0;
    const uint64_t t_start = get_timestamp();
    for (uint32_t iter = 0; iter < iters; iter++) {
        uint32_t core = 0;
        while (core < num_cores) {
            uint32_t issued = 0;
            uint32_t slot = 0;
            for (; issued < kReadsInFlight && core < num_cores; issued++, core++) {
                const uint32_t xy = coords[core];
                CoreLocalMem<uint32_t> dst(kRingBase + slot * kBytesPerRead);
                noc.async_read<NocOptions::DEFAULT, kBytesPerRead>(
                    src, dst, kBytesPerRead, {.noc_x = xy & 0xFFFFu, .noc_y = xy >> 16, .addr = src_addr}, {});
                if constexpr (kRingSlots > 1) {
                    slot++;
                    if (slot == kRingSlots) {
                        slot = 0;
                    }
                }
            }
            noc.async_read_barrier();
            if constexpr (kPollExamine) {
                // The adaptive switch's work: sum (tail - head) over the 5 RISC tails of each core that
                // just landed. Heads are 0, so this is the tail sum -- same instruction count.
                for (uint32_t s = 0; s < issued; s++) {
                    volatile tt_l1_ptr uint32_t* cv =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kRingBase + s * kBytesPerRead);
                    for (uint32_t r = 0; r < kNumRisc; r++) {
                        pending += cv[kTailWordOffset + r];
                    }
                }
            }
        }
    }
    const uint64_t t_end = get_timestamp();

    // Always restore NOC2AXI so subsequent context observes the default -- NIU_CFG_0 persists
    // across programs and only a chip reset puts it back.
    experimental::drisc_set_noc2axi_mode();

    // Cheap liveness guard: the host primes every polled window with a nonzero pattern, so a
    // zero checksum means the reads never landed (bad src addr, or the NIU never left NOC2AXI)
    // and any bandwidth number from this run is measuring nothing. Summed after the timed loop
    // so the loads stay out of the measurement.
    uint32_t checksum = 0;
    for (uint32_t slot = 0; slot < kRingSlots; slot++) {
        checksum += *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kRingBase + slot * kBytesPerRead);
    }

    const uint64_t cycles = t_end - t_start;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);
    out[0] = static_cast<uint32_t>(cycles & 0xFFFFFFFFu);
    out[1] = static_cast<uint32_t>(cycles >> 32);
    out[2] = checksum;
    out[3] = num_cores * iters;                      // core visits, for the ns/visit poll-floor metric
    out[4] = static_cast<uint32_t>(timer_overhead);  // cycles per get_timestamp(), the instrument's own cost
    out[5] = pending;                                // keeps the poll-examine loop from being optimized away
}
