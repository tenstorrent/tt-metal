// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/transport_bandwidth/backends/x280_to_tensix.hpp -- X280 backend for
 * the X280 -> Tensix transport_bandwidth member.
 *
 * X280 is the initiator. It programs one paired-range NOC TLB slot at the
 * Tensix worker's L1 destination in posted mode, then issues bursts of
 * `transfer_size / 4` back-to-back u32 stores through the window. BUG-4
 * forbids memcpy and wider stores through the window, so the burst is a
 * straight sequence of `sw` instructions the compiler unrolls at
 * `issue<TransferSize>`.
 *
 * The source of every store is a stack-resident vector loaded from LIM
 * once at INIT; subsequent iterations hit the X280's cache path to the
 * stack and never touch LIM. What we want to measure is transport
 * throughput, not LIM -> window bandwidth.
 *
 * Timing uses TWO counters around the same bracket. mtime (fixed 50 MHz,
 * PLL-independent) is the PRIMARY -- it is a wall clock that needs no
 * PLL conversion on the host to publish rate in seconds.
 * rdcycle (at the running PLL) is the SECONDARY, published from the
 * same bracket as a cross-check between two independent clocks. rdcycle
 * does not advance while the hart stalls on memory or a BRESP
 * (MEASUREMENT.md "Unconfirmed hardware assumptions", item 5); the two
 * counters have counted in lockstep on every run so far (see §4.3 in
 * x280/CHARACTERIZATION_RESULTS.md).
 *
 * The host retains secondary_batch_delta as raw CPU cycles. Only mtime
 * supplies elapsed time.
 *
 * See rt/x280_measurement.h for the fenced helpers and
 * x280/CHARACTERIZATION_UBENCH_DESIGN.md § Transport bandwidth for the
 * frontend/backend contract this file supplies.
 */
#ifndef X280_TRANSPORT_BW_BACKEND_X280_TO_TENSIX_HPP
#define X280_TRANSPORT_BW_BACKEND_X280_TO_TENSIX_HPP

#include <cstdint>

#include "transport_bandwidth_common.h"
#include "transport_bandwidth_frontend_cxx.hpp"
#include "x280_to_tensix.h"
#include "../../shared/ubench_units.h"

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/measurement.h"
#include "rt/x280_measurement.h"
#include "rt/x280_noc_map.h"

namespace transport_bandwidth::backend_x280_to_tensix {

/* Path-verification magic. Different from the Tensix -> X280 member's magic
 * so a mis-launched image can never accidentally pass; picked so that
 * (nonce ^ magic) is not congruent with the nonce for any small nonce the
 * host can mint. */
constexpr uint32_t kVerifyMagic = 0x9B7C1D3Eu;

/* Reads verify_path takes before it reports PATH_MISMATCH. Each is one NoC
 * round trip, outside the timed loop. */
constexpr uint32_t kVerifyPollBound = 500000u;

/* Admitted transfer sizes. Each `sw` through the uncached window is its own
 * 4 B NoC request, so a larger transfer_size does not grow a packet. */
using sizes = size_list<64u, 256u>;
static_assert(
    sizes::max() == TB_X2T_STAGING_MAX_BYTES,
    "the host sizes LIM staging and the Tensix L1 destination to the largest transfer");

/* Ceiling for the source-vector cache, in u32 words. Sized for the largest
 * supported transfer_size; smaller transfers use a prefix. */
constexpr uint32_t kMaxWords = sizes::max() / sizeof(uint32_t);

/* --------------------------------------------------------------------------
 * Backend types
 * -------------------------------------------------------------------------- */

/* Timestamp: both counters, read atomically-in-order at t0 and t1. Reading
 * (mtime, rdcycle) at both ends gives equal-width brackets for the two
 * counters (they differ by a handful of csr cycles at each end, well below
 * one mtime tick at 50 MHz / 20 ns per tick). */
struct timestamp_t {
    uint64_t mtime_ticks;
    uint64_t cycle_ticks;
};

/* Delta: primary is mtime (PLL-independent wall clock), secondary is
 * raw rdcycle. */
struct counter_delta_t {
    uint64_t primary;
    uint64_t secondary;
};

struct state_t {
    /* Parsed protocol args (transfer_size, warmup_count, transfer_count,
     * boundary, run_nonce). The frontend consults these through args(). */
    transport_bw_protocol_args_t args;

    /* Programmed TLB window (base + zero offset). Store target for issue(),
     * read target for drain(). */
    volatile uint32_t* window;

    /* Stack-resident source vector. Kept inside state_t so the frontend's
     * `Backend::issue<TransferSize>(s)` can access it through a stable
     * offset; the compiler hoists s.window and s.staging into registers
     * for the inner unrolled loop.
     *
     * kMaxWords is sized for the largest supported transfer; smaller
     * transfers touch only the prefix. */
    uint32_t staging[kMaxWords];

    /* Result region (writable, plain). The volatile-vs-plain cast is done
     * once at image entry. */
    transport_bw_result_t* r;

    /* TLB slot to clear on shutdown, so a residual coordinate does not
     * leak into the next case. */
    uint32_t tlb_index;

    /* Echoed for the sentinel computation. */
    uint32_t run_nonce;
};

/* --------------------------------------------------------------------------
 * Section validation and copy
 * --------------------------------------------------------------------------
 * sanity_check refuses every arg combination that would corrupt the timed
 * loop or the result region. The image publishes ARG_LAYOUT_MISMATCH on a
 * false return.
 * -------------------------------------------------------------------------- */
inline bool sanity_check(const transport_bw_x280_to_tensix_section_t& s) {
    if (s.cookie_common != transport_bw_cookie_for(UBENCH_SECTION_COMMON)) {
        return false;
    }
    if (s.cookie_backend != transport_bw_cookie_for(UBENCH_SECTION_BACKEND)) {
        return false;
    }
    if (!sizes::contains(s.common.transfer_size)) {
        return false;
    }
    if (s.common.transfer_count == 0u) {
        return false;
    }
    if (s.common.boundary != TRANSPORT_BW_BOUNDARY_DRAINED && s.common.boundary != TRANSPORT_BW_BOUNDARY_INJECTION) {
        return false;
    }
    if (s.common.run_nonce == 0u) {
        return false;
    }
    if (s.backend.peer_dst_addr == 0u) {
        return false;
    }
    if ((s.backend.peer_dst_addr % UBENCH_NOC_UNIT_BYTES) != 0u) {
        return false;
    }
    if (s.backend.own_staging_addr == 0u) {
        return false;
    }
    if ((s.backend.own_staging_addr % UBENCH_NOC_UNIT_BYTES) != 0u) {
        return false;
    }
    if (s.backend.own_result_addr == 0u) {
        return false;
    }
    return x280_tlb_paired_slot(s.backend.peer_tlb_index);
}

/* --------------------------------------------------------------------------
 * Initialisation
 * --------------------------------------------------------------------------
 * Populates state_t and the result region's static fields, programs the
 * paired-range TLB slot, and cache-loads the
 * staging vector.
 *
 * Returns false on any TLB failure. A programmed window that failed its
 * verification stays recorded in state.window, so the caller (the image)
 * ends the run through ubench_finish: ARG_LAYOUT_MISMATCH, or
 * WINDOW_CLEANUP_FAILED when shutdown()'s clear does not read back zero.
 * -------------------------------------------------------------------------- */
inline bool initialize(state_t& state, const transport_bw_x280_to_tensix_section_t& s, transport_bw_result_t& r) {
    state.args = s.common;
    state.r = &r;
    state.tlb_index = s.backend.peer_tlb_index;
    state.run_nonce = s.common.run_nonce;
    state.window = nullptr;

    /* Initialise the result region. terminal_state advances as the state
     * machine moves; the host reads it once at DONE (or its refusal
     * terminal). */
    r.backend_id = TRANSPORT_BW_BACKEND_ID_X280_TO_TENSIX;
    r.boundary = s.common.boundary;
    r.terminal_state = static_cast<uint32_t>(TRANSPORT_BW_STATE_INIT);
    r.transfer_size = s.common.transfer_size;
    r.transfer_count = s.common.transfer_count;
    r.warmup_completed = 0u;
    r.logical_bytes = 0ull;
    r.raw_batch_delta = 0ull;
    r.secondary_batch_delta = 0ull;
    r.path_sentinel = 0u;
    r.reserved = 0u;
    x280_fence_ow();

    /* Ordering mode 2, Posted Writes. See BlackholeA0/L2CPUTile/TLBWindows.md.
     *
     * Mode 3 (Counted Writes) pairs posted writes with NoC ordering
     * and so looks like the better fit here. It is refused because one store
     * through a mode-3 window permanently stops completion blocking on the
     * whole L2CPU tile: later non-posted stores no longer wait for their
     * acknowledgement, so every later latency measurement on that tile reads
     * short, and only an ASIC reset clears it. In hardware, the
     * outstanding-request limiter and the post-strict ordering logic operate
     * independently, so the master takes an early AXI BRESP. Staying under the
     * tracked 256-outstanding-write workaround does not help; the failure is
     * observed from four stores.
     *
     * An unverified window is kept in state.window so shutdown() clears it and
     * a failed clear ends WINDOW_CLEANUP_FAILED: the host assigned this paired
     * slot to this run alone. */
    volatile void* w = nullptr;
    const noc_window_status_t opened =
        noc_open_window_2m(s.backend.peer_tlb_index, s.backend.peer_noc, s.backend.peer_dst_addr, NOC_ORDER_POSTED, &w);
    state.window = reinterpret_cast<volatile uint32_t*>(w);
    if (opened != NOC_WINDOW_OPEN) {
        return false;
    }

    /* Prepare a stack-resident source vector. Reading the store source from
     * a register-cached local array keeps the hot loop's data path off LIM:
     * what we want to measure is transport throughput, not LIM -> window
     * bandwidth. Copies enough words for the largest supported transfer;
     * smaller transfers touch only the prefix. */
    const volatile uint32_t* staging_lim = reinterpret_cast<volatile const uint32_t*>(s.backend.own_staging_addr);
    for (uint32_t k = 0u; k < kMaxWords; ++k) {
        state.staging[k] = staging_lim[k];
    }
    __asm__ volatile("" : : "r"(&state.staging[0]) : "memory");
    return true;
}

/* One-shot cleanup: clear the paired-TLB slot. Called on every exit path
 * past `initialize()` succeeding. Returns false when the clear did not read
 * back zero. */
inline bool shutdown(state_t& state) {
    if (state.window == nullptr) {
        return true;
    }
    state.window = nullptr;
    return noc_clear_tlb_2m(state.tlb_index) == 0;
}

/* --------------------------------------------------------------------------
 * Frontend adapter
 * -------------------------------------------------------------------------- */
struct Backend {
    using state_t = ::transport_bandwidth::backend_x280_to_tensix::state_t;
    using timestamp_t = ::transport_bandwidth::backend_x280_to_tensix::timestamp_t;
    using counter_delta_t = ::transport_bandwidth::backend_x280_to_tensix::counter_delta_t;
    using sizes = ::transport_bandwidth::backend_x280_to_tensix::sizes;

    static const transport_bw_protocol_args_t& args(state_t& s) { return s.args; }
    static transport_bw_result_t& result(state_t& s) { return *s.r; }

    /* Path verification: write sentinel through the window, then read the
     * same word until it matches or kVerifyPollBound reads have returned. On
     * posted TLB nothing orders a read behind the write, so one read that
     * overtook it is retried rather than reported; a match proves the path
     * delivered this run's word. A stale word from an earlier run cannot match
     * because (nonce ^ magic) is unique to this run.
     *
     * Publishes r.path_sentinel as an observation (not a comparison
     * outcome), so the host can distinguish "path answered with unexpected
     * value" from "path did not answer at all". */
    static bool verify_path(state_t& s) {
        const uint32_t sentinel = s.run_nonce ^ kVerifyMagic;
        s.window[0] = sentinel;
        x280_fence_iorw();
        uint32_t observed = 0u;
        for (uint32_t i = 0u; i < kVerifyPollBound; ++i) {
            observed = s.window[0];
            if (observed == sentinel) {
                break;
            }
        }
        x280_fence_iorw();
        s.r->path_sentinel = observed;
        return observed == sentinel;
    }

    /* One issue: TransferSize/4 back-to-back u32 stores through the TLB
     * window. Templated so the compiler unrolls each supported burst
     * length; `always_inline` because a call boundary here is exactly what
     * the measured loop is otherwise counting. */
    template <uint32_t TransferSize>
    static inline __attribute__((always_inline)) void issue(state_t& s) {
        static_assert(sizes::contains(TransferSize), "TransferSize is not in sizes");
        static_assert(TransferSize % sizeof(uint32_t) == 0u, "transfer_size must be a multiple of 4B");
        constexpr uint32_t K = TransferSize / sizeof(uint32_t);
        volatile uint32_t* window = s.window;
        const uint32_t* staging = s.staging;
#pragma GCC unroll 64
        for (uint32_t k = 0u; k < K; ++k) {
            window[k] = staging[k];
        }
    }

    /* Fenced read through the window. It is not proof that preceding posted
     * writes completed; host L1 readback verifies delivery. */
    static inline __attribute__((always_inline)) void drain(state_t& s) {
        x280_fence_iorw();
        uint32_t v = *s.window;
        __asm__ volatile("" : : "r"(v) : "memory");
        x280_fence_iorw();
    }

    /* Read both counters at this instant. Order is (mtime, rdcycle) at both
     * t0 and t1, so the two brackets have equal width. */
    static inline __attribute__((always_inline)) timestamp_t measurement_now(state_t&) {
        const x280_paired_t now = x280_paired_now();
        return timestamp_t{now.wall, now.cpu};
    }

    static counter_delta_t delta(state_t& s, const timestamp_t& t0, const timestamp_t& t1) {
        counter_delta_t d;
        if (!measurement_delta(t0.mtime_ticks, t1.mtime_ticks, &d.primary)) {
            d.primary = UINT64_MAX;
        }
        if (!measurement_delta(t0.cycle_ticks, t1.cycle_ticks, &d.secondary)) {
            d.secondary = UINT64_MAX;
        }
        return d;
    }

    static void publish_delta(state_t& s, const counter_delta_t& d) {
        s.r->raw_batch_delta = d.primary;
        s.r->secondary_batch_delta = d.secondary;
    }

    static void publish_barrier(state_t&) { x280_fence_ow(); }

    static bool publish_footer(state_t& s) { return shutdown(s); }

    static constexpr uint32_t BACKEND_ID = TRANSPORT_BW_BACKEND_ID_X280_TO_TENSIX;
};

}  // namespace transport_bandwidth::backend_x280_to_tensix

#endif /* X280_TRANSPORT_BW_BACKEND_X280_TO_TENSIX_HPP */
