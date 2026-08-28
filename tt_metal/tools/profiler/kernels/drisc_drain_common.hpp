// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Support spine of the DRISC drain kernel (drisc_profiler_filler.cpp): the D2H write/credit/barrier
// primitives, the drainer's own zone ids and the NoC-footprint instrument.
#pragma once

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "hostdevcommon/profiler_common.h"
#include "internal/tt-1xx/risc_common.h"

#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

// ---- The drainer's own zone ids ---------------------------------------------------------------------
//
// ORDINARY structural zone ids with ORDINARY .tt_zone_meta records, declared here in the drain kernel's
// own translation unit exactly the way a worker kernel declares its zones. The host resolves these names
// out of THIS ELF as it loads, like any other kernel's -- there is no reserved band, no fixed id and no
// hardcoded host name table any more. (They used to be fixed values 0x7FF0..0x7FF8 in a reserved band,
// with their names registered by hand next to PRODUCER-STALL in perf_debug_profiler.cpp.)
//
// Declared inside `namespace kernel_profiler` under their original spellings so the ~20 emission sites
// below are unchanged. The names are the strings a human reads in Tracy, so they keep their old text.
namespace kernel_profiler {
TT_ZONE_DEFINE_ID(DRISC_ZONE_SWEEP, "DRISC-SWEEP");              // one whole poll sweep (the parent)
TT_ZONE_DEFINE_ID(DRISC_ZONE_READ, "DRISC-READ");                // span-read ISSUE
TT_ZONE_DEFINE_ID(DRISC_ZONE_READ_WAIT, "DRISC-READ-WAIT");      // read-barrier wait left after proc
TT_ZONE_DEFINE_ID(DRISC_ZONE_PROC, "DRISC-PROC");                // control-vector scan + head write-back
TT_ZONE_DEFINE_ID(DRISC_ZONE_CREDIT_WAIT, "DRISC-CREDIT-WAIT");  // socket credit against the host FIFO
TT_ZONE_DEFINE_ID(DRISC_ZONE_WRITE, "DRISC-WRITE");              // the egress: PCIe write, or the DMA spool ship
TT_ZONE_DEFINE_ID(DRISC_ZONE_WR_BARRIER, "DRISC-WR-BARRIER");    // staging-reuse wait: NoC sent, or DMA complete
TT_ZONE_DEFINE_ID(DRISC_ZONE_DRAIN, "DRISC-DRAIN");              // spool->host drain pump (spool mode only)
TT_ZONE_DEFINE_ID(DRISC_ZONE_PACE, "DRISC-PACE");                // the inter-sweep pacing gap
TT_ZONE_DEFINE_ID(DRISC_ZONE_SYNC, "DRISC-SYNC");                // common-trigger sync fiducial
TT_ZONE_DEFINE_ID(SPSC_DATA_ID_NOCFP, "DRISC-NOC-FOOTPRINT");    // the per-sweep NoC-counter PP_DATA sample
}  // namespace kernel_profiler

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks. The caller runs
// noc_write_init_state<write_cmd_buf>(NOC_INDEX, vc) once per push -- a push makes several calls (one
// per frame plus the notify), so a per-call init would repeat command-buffer setup that nothing between
// the calls invalidates. Alternating two command buffers to pipeline NIU acceptance was measured
// stall-neutral at the delay-16 saturation wall and deleted.
inline void write_to_host_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
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
    const SocketSenderInterface& socket, uint32_t num_pages, uint64_t deadline, volatile tt_l1_ptr uint32_t* stop) {
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
        acked =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uint32_t>(acked) + bytes_acked_size_bytes);
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
// `sent_only` (NONPOSTED_WR_REQ_SENT, the usual source-reuse gate) is legal ONLY when the buffer's next
// writer is this core's own NIU (staging, refilled by its own read responses). It is NOT a fence against
// another L1 master -- a sent-based wait against a DMA-engine refill produced 77k/42k decode order
// regressions in one run.
template <bool sent_only = false>
inline bool write_barrier_bounded(uint64_t deadline) {
    // Bounded on ITERATIONS as well as cycles: the cycle deadline assumes get_timestamp() advances AND
    // that the loop gets to evaluate it, and a wedged NIU breaks both. The cap does not actually free a
    // wedged core -- measured; control never returns from the flush check -- but a barrier bounded two
    // ways beats one bounded by a clock it must be running to read.
    // 4M iterations is far beyond any healthy flush (worst observed is a handful).
    constexpr uint32_t kMaxSpins = 4u << 20;
    uint32_t spins = 0;
    while (sent_only ? !ncrisc_noc_nonposted_writes_sent(NOC_INDEX)
                     : !ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
        invalidate_l1_cache();
        if (++spins >= kMaxSpins || get_timestamp() >= deadline) {
            return false;
        }
    }
    return true;
}

// Bounded dma_async_write_wait_n (gddr_dma.h): the spool-mode staging-reuse gate. Completion, not "sent" --
// the DMA engine has no sent analog, and completion (AXI write response received) is also what makes the
// spool bytes observable to the stream-1 reads that consume them.
inline bool dma_wait_writes_bounded(uint8_t stream, uint8_t n, uint64_t deadline) {
    while (experimental::dma_get_writes_outstanding(stream) > n) {
        if (get_timestamp() >= deadline) {
            return false;
        }
    }
    return true;
}

// ---- NoC FOOTPRINT: per-sweep NIU-counter deltas into 64-bit accumulators ------------------------------
//
// Data-driven (flat runtime-indexed arrays, register ids in a table) because unrolled per-register code
// overflowed the 11,264 B DRISC code region; keep it table-shaped. The NIU counters are 32-bit and WRAP
// within one long capture, so deltas are taken per sweep -- never end-to-end.
constexpr uint32_t kNfRdW = 0;  // NIU_MST_RD_DATA_WORD_RECEIVED       -- read bytes in, in 64 B NoC words
constexpr uint32_t kNfRdT = 1;  // NIU_MST_RD_REQ_SENT                 -- read transactions issued
constexpr uint32_t kNfWrW = 2;  // NIU_MST_NONPOSTED_WR_DATA_WORD_SENT -- write bytes out, in NoC words
constexpr uint32_t kNfWrT = 3;  // NIU_MST_NONPOSTED_WR_REQ_SENT       -- write transactions issued
constexpr uint32_t kNfN = 4;
constexpr uint32_t kNfSlots = 2 * kNfN;  // flat: noc * kNfN + k, NoC 0 first

struct NocFpState {
    uint32_t prev[kNfSlots];
    uint64_t life[kNfSlots];
    // THIS sweep's deltas, i.e. what the per-sweep PP_DATA sample ships. Free: nf_sample_regs already computes
    // each delta to fold it into life[], so keeping it costs one store and no extra register read.
    uint32_t last[kNfSlots];
    // The workload window: from the START of the first sweep that did work to the END of the last one. Same
    // work-triggered definition self-profiling settled on (FINDINGS N+41), and the whole reason there are two
    // blocks rather than one -- a drainer is resident from device open, so a lifetime figure is dominated by
    // polling OUTSIDE any workload, and blending the two is the wrong-population trap. Idle sweeps INTERLEAVED
    // inside the window are included, deliberately: their span reads are real traffic that a workload paid for.
    uint64_t win_base[kNfSlots];
    uint64_t win_last[kNfSlots];
    uint64_t win_t0;
    uint64_t win_t1;
    uint64_t cost;  // cycles this instrument spent on its own register reads -- reported, never hidden
    uint32_t win_sweep_first;
    uint32_t win_sweep_last;
    bool win_open;
};

// Read all kNfSlots counters and fold the wrap-safe deltas into the 64-bit accumulators.
//
// The 32-bit subtract happens FIRST and is only then widened: the truncated difference of two 32-bit samples
// is the true delta for any delta < 2^32, whether or not the counter rolled over between them.
//
// Both NoCs are read rather than just the ones the kernel claims to use. The zero columns ARE the
// measurement: a non-zero read count on the write NoC would mean the read/write NoC split is not doing
// what the code claims.
static void nf_sample_regs(NocFpState* s) {
    // Runtime-indexed so the loads collapse into one loop body. NOC_STATUS_READ_REG resolves to a load
    // from (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(id) -- this core's own memory-mapped NIU register
    // block -- so it issues NO NoC transaction and the instrument cannot perturb what it measures.
    // Addresses come from NOC_STATUS() in noc_parameters.h and are never literals: test_cluster_bh.cpp
    // carries a misnamed 0xffb202e0 and copying it would propagate the error into a number nobody can check.
    //
    // The write slots sum the POSTED and NON-POSTED counters: a total that counted only one flavor would
    // silently under-report bytes -- which is what the old posted-must-be-zero entry/exit check existed to
    // guard against, and why summing retires it. The delta of a SUM of two wrapping 32-bit counters is
    // still exact for any true delta < 2^32.
    static const uint32_t kIds[kNfN] = {
        NIU_MST_RD_DATA_WORD_RECEIVED,
        NIU_MST_RD_REQ_SENT,
        NIU_MST_NONPOSTED_WR_DATA_WORD_SENT,
        NIU_MST_NONPOSTED_WR_REQ_SENT};
    static const uint32_t kIds2[kNfN] = {0, 0, NIU_MST_POSTED_WR_DATA_WORD_SENT, NIU_MST_POSTED_WR_REQ_SENT};
    const uint64_t t0 = get_timestamp();
    for (uint32_t i = 0; i < kNfSlots; i++) {
        uint32_t cur = NOC_STATUS_READ_REG(i / kNfN, kIds[i % kNfN]);
        if (kIds2[i % kNfN] != 0) {
            cur += NOC_STATUS_READ_REG(i / kNfN, kIds2[i % kNfN]);
        }
        const uint32_t d = cur - s->prev[i];
        s->last[i] = d;
        s->life[i] += static_cast<uint64_t>(d);
        s->prev[i] = cur;
    }
    s->cost += get_timestamp() - t0;
}

// End of one sweep: sample, then decide whether this sweep extends the workload window.
//
// Called after the sweep body and before the pacing gap. The gap issues no NoC traffic, so which side of it
// the sample falls on cannot change a byte total; taking it before means the window's DURATION is measured
// over sweeps rather than over sweeps-plus-whatever-pace-trailed-the-last-one, which is the honest
// denominator for a MB/s figure.
static void nf_sweep_end(NocFpState* s, uint32_t sweep, uint64_t t_sweep0, uint32_t sweep_cyc, bool did_work) {
    // ORDER IS THE TRICK, and it is what removes a whole 64 B stack snapshot. On entry life[] still holds the
    // through-PREVIOUS-sweep totals, so a window opening now can take its base straight from life[] -- no
    // "before" copy is needed at all. Sample AFTER that, and life[] then includes this sweep, which is exactly
    // what win_last wants. The earlier version snapshotted life[] into a local array first and was 32 B over
    // the DRISC code limit; this is both smaller and simpler for the same numbers.
    if (did_work && !s->win_open) {
        s->win_open = true;
        s->win_sweep_first = sweep;
        s->win_t0 = t_sweep0;
        for (uint32_t i = 0; i < kNfSlots; i++) {
            s->win_base[i] = s->life[i];
        }
    }
    nf_sample_regs(s);
    if (!did_work) {
        return;
    }
    s->win_sweep_last = sweep;
    s->win_t1 = t_sweep0 + sweep_cyc;
    for (uint32_t i = 0; i < kNfSlots; i++) {
        s->win_last[i] = s->life[i];
    }
}
