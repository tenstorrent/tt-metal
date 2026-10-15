// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/transport_bandwidth/transport_bandwidth_frontend_cxx.hpp -- the
 * templated C++ frontend for a transport_bandwidth member.
 *
 * One suite frontend owns the state-machine, verify/warmup/timed phase order,
 * boundary specialization, and result publication. One statically selected
 * backend per component class supplies the hardware operations (issue, drain,
 * verify_path, measurement_now/delta, publish_barrier, publish_footer).
 *
 * The design's point (§Frontend and backend contract, §Kernel entry point,
 * §Timed loop) is that a source-code divergence between members must not be
 * mistaken for a hardware difference. The X280 -> Tensix and Tensix -> X280
 * members share this frontend so their timed loop, warmup discipline,
 * verify sequencing, and DONE publication all come from one file.
 *
 * A backend supplies (via a `struct Backend` adapter that the frontend
 * consumes as a template parameter):
 *
 *   struct state_t;
 *   struct timestamp_t;      -- POD; may carry multiple counters
 *   struct counter_delta_t;  -- POD; carries primary, secondary (0 if none)
 *
 *   static const transport_bw_protocol_args_t& args(state_t& s);
 *   static transport_bw_result_t& result(state_t& s);
 *
 *   static bool verify_path(state_t& s);
 *     -- Untimed. Publishes r.path_sentinel as a side effect. Returns true
 *        when the observed value matched the expected sentinel.
 *
 *   static bool rendezvous(state_t& s);
 *     -- Untimed, after warm-up and before t0. Returns true once every
 *        participant of the run has reached it, so the participants' timed
 *        brackets start together; true at once for a single participant.
 *        False when the others did not arrive within the backend's bound;
 *        the run then ends RENDEZVOUS_TIMEOUT.
 *
 *   template <uint32_t TransferSize>
 *   static void issue(state_t& s);   -- one issue; compile-time-specialized
 *
 *   static void drain(state_t& s);
 *     -- Ends the DRAINED bracket. Tensix->X280's write barrier forces
 *        completion of every previously-issued transfer; X280->Tensix's
 *        fenced read does not under the posted ordering mode, and host L1
 *        readback verifies delivery. Backends must include any pre/post
 *        fences they need; the frontend does not emit fences around this
 *        call.
 *
 *   static timestamp_t measurement_now(state_t& s);
 *   static counter_delta_t delta(state_t& s, const timestamp_t& t0,
 *                                             const timestamp_t& t1);
 *
 *   static void publish_delta(state_t& s, const counter_delta_t& d);
 *     -- Writes into r.raw_batch_delta and r.secondary_batch_delta. Called
 *        exactly once per invocation.
 *
 *   static void publish_barrier(state_t& s);
 *     -- Release fence so the host observes the terminal state after every
 *        result-region field. `fence ow, ow` on X280; no-op on Tensix (L1 is
 *        write-through).
 *
 *   static bool publish_footer(state_t& s);
 *     -- Cleanup before the terminal state is published (X280 TLB clear;
 *        no-op on Tensix). Returns false when the cleanup did not take, and
 *        the run ends WINDOW_CLEANUP_FAILED instead of its own state.
 *
 *   static constexpr uint32_t BACKEND_ID;
 *
 *   using sizes = size_list<...>;
 *     -- The transfer sizes (bytes) the member admits. The frontend refuses
 *        any other args.transfer_size and instantiates issue<S> only for the
 *        listed S.
 *
 * The backend has already parsed member-specific arguments and populated the
 * result region's static fields (backend_id, boundary, transfer_size,
 * transfer_count) before the frontend is called.
 * The frontend then advances the state machine:
 *
 *   INIT (already set) -> VERIFY_PATH -> WARMUP -> [rendezvous] ->
 *     TIMED_{DRAINED|INJECTION} -> [DRAIN, only under INJECTION] ->
 *     PUBLISH -> DONE
 *
 * or, on a path-verification failure, INIT -> VERIFY_PATH -> PATH_MISMATCH,
 * or, when the rendezvous gives up, ... -> WARMUP -> RENDEZVOUS_TIMEOUT.
 * A failed footer replaces the terminal state with WINDOW_CLEANUP_FAILED.
 *
 * Timed-loop rule, for every X280 benchmark: a timed loop keeps its operands
 * in registers. Every hart's stack is in uncached LIM, so each reload of a
 * stack variable, or of an object whose address has escaped, is one LIM read
 * inside the loop, and every fence's "memory" clobber forces those reloads.
 *
 * Every function on the path from run_member to Backend::issue is
 * always_inline: taking `s` out of line makes every NoC counter store force a
 * reload of the transfer operands inside the timed loop. An entry point named
 * `main` that calls run_member is marked `hot`, or GCC lays its timed loops out
 * as run-once code.
 *
 * See docs/benchmarks/design.md, § Transport bandwidth.
 */
#ifndef X280_TRANSPORT_BANDWIDTH_FRONTEND_CXX_HPP
#define X280_TRANSPORT_BANDWIDTH_FRONTEND_CXX_HPP

#include <stdint.h>

#include "transport_bandwidth_common.h"
#include "../shared/ubench_finish.hpp"

namespace transport_bandwidth {

/* A backend's admitted transfer sizes, in bytes. contains() is the runtime
 * membership test; max() is the largest listed size. */
template <uint32_t... Sizes>
struct size_list {
    static_assert(sizeof...(Sizes) > 0u, "a backend admits at least one transfer size");

    static constexpr bool contains(uint32_t size) { return ((size == Sizes) || ...); }

    static constexpr uint32_t max() {
        uint32_t m = 0u;
        ((m = Sizes > m ? Sizes : m), ...);
        return m;
    }
};

template <uint32_t Size>
struct size_constant {
    static constexpr uint32_t value = Size;
};

/* Calls f(size_constant<S>{}) for the listed S equal to `size`. Returns false,
 * calling nothing, when no S matches. */
template <uint32_t... Sizes, typename F>
inline __attribute__((always_inline)) bool dispatch_size(size_list<Sizes...>, uint32_t size, F&& f) {
    return ((size == Sizes ? (f(size_constant<Sizes>{}), true) : false) || ...);
}

/* --------------------------------------------------------------------------
 * Timed loop, compile-time-specialized on TransferSize and Boundary.
 *
 * There is no boundary or size check inside the loop: `run_timed_dispatch`
 * picks both specializations before t0. DRAINED brackets t0..drain..t1;
 * INJECTION brackets t0..t1 and drains after. The mandatory drain on
 * INJECTION is transitioned into TRANSPORT_BW_STATE_DRAIN so the host can
 * tell "batch finished but drain outstanding" from "batch clean".
 * -------------------------------------------------------------------------- */
template <typename Backend, uint32_t TransferSize, transport_bw_boundary_t Boundary>
inline __attribute__((always_inline)) void run_timed_specialized(typename Backend::state_t& s, uint32_t N) {
    transport_bw_result_t& r = Backend::result(s);
    r.terminal_state = static_cast<uint32_t>(
        Boundary == TRANSPORT_BW_BOUNDARY_DRAINED ? TRANSPORT_BW_STATE_TIMED_DRAINED
                                                  : TRANSPORT_BW_STATE_TIMED_INJECTION);

    auto t0 = Backend::measurement_now(s);
    for (uint32_t i = 0u; i < N; ++i) {
        Backend::template issue<TransferSize>(s);
    }

    if constexpr (Boundary == TRANSPORT_BW_BOUNDARY_DRAINED) {
        Backend::drain(s);
        auto t1 = Backend::measurement_now(s);
        Backend::publish_delta(s, Backend::delta(s, t0, t1));
    } else {
        auto t1 = Backend::measurement_now(s);
        Backend::publish_delta(s, Backend::delta(s, t0, t1));
        /* Mandatory drain outside the bracket. Advertise it in the state
         * machine so a host that sees TRANSPORT_BW_STATE_DRAIN in the
         * result knows the drain phase is still owed by device. */
        r.terminal_state = static_cast<uint32_t>(TRANSPORT_BW_STATE_DRAIN);
        Backend::drain(s);
    }
}

/* --------------------------------------------------------------------------
 * Runtime dispatch on {transfer_size, boundary} -> compile-time-specialized
 * timed loop. Both axes are host-provided but constrained to a small set;
 * this branch is BEFORE t0 so the hot loop has no branches on either.
 *
 * run_member has refused every size outside Backend::sizes, so exactly one
 * specialization runs.
 * -------------------------------------------------------------------------- */
template <typename Backend>
inline __attribute__((always_inline)) void run_timed_dispatch(typename Backend::state_t& s) {
    const transport_bw_protocol_args_t& a = Backend::args(s);
    const auto boundary = static_cast<transport_bw_boundary_t>(a.boundary);
    dispatch_size(typename Backend::sizes{}, a.transfer_size, [&](auto size) __attribute__((always_inline)) {
        constexpr uint32_t TransferSize = decltype(size)::value;
        if (boundary == TRANSPORT_BW_BOUNDARY_DRAINED) {
            run_timed_specialized<Backend, TransferSize, TRANSPORT_BW_BOUNDARY_DRAINED>(s, a.transfer_count);
        } else {
            run_timed_specialized<Backend, TransferSize, TRANSPORT_BW_BOUNDARY_INJECTION>(s, a.transfer_count);
        }
    });
}

/* --------------------------------------------------------------------------
 * Warmup: W untimed issues followed by one drain. Measurement starts warm
 * but empty, per §Timed loop.
 *
 * Compile-time-specialized on TransferSize for consistency with the timed
 * loop -- some backends require a static TransferSize for issue() (X280's
 * unrolled store burst) and gracefully ignore it on backends that don't
 * (Tensix's noc_async_write passes the size as a runtime argument to the NIU).
 * -------------------------------------------------------------------------- */
template <typename Backend>
inline __attribute__((always_inline)) void warmup(typename Backend::state_t& s) {
    const transport_bw_protocol_args_t& a = Backend::args(s);
    dispatch_size(typename Backend::sizes{}, a.transfer_size, [&](auto size) __attribute__((always_inline)) {
        for (uint32_t i = 0u; i < a.warmup_count; ++i) {
            Backend::template issue<decltype(size)::value>(s);
        }
    });
    Backend::drain(s);
}

/* --------------------------------------------------------------------------
 * State-machine driver. The backend has parsed args and populated result's
 * static fields (backend_id, boundary, transfer_size, transfer_count).
 * Called with terminal_state == INIT.
 * -------------------------------------------------------------------------- */
template <typename Backend>
inline __attribute__((always_inline)) void run_member(typename Backend::state_t& s) {
    transport_bw_result_t& r = Backend::result(s);

    /* warmup() and run_timed_dispatch() run nothing for an unlisted size,
     * so refuse it here rather than relying on each backend's validation. */
    if (!Backend::sizes::contains(Backend::args(s).transfer_size)) {
        ubench_finish<Backend>(s, TRANSPORT_BW_STATE_ARG_LAYOUT_MISMATCH, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED);
        return;
    }

    /* VERIFY_PATH. */
    r.terminal_state = static_cast<uint32_t>(TRANSPORT_BW_STATE_VERIFY_PATH);
    if (!Backend::verify_path(s)) {
        ubench_finish<Backend>(s, TRANSPORT_BW_STATE_PATH_MISMATCH, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED);
        return;
    }

    /* WARMUP. */
    r.terminal_state = static_cast<uint32_t>(TRANSPORT_BW_STATE_WARMUP);
    warmup<Backend>(s);
    r.warmup_completed = Backend::args(s).warmup_count;
    if (!Backend::rendezvous(s)) {
        ubench_finish<Backend>(s, TRANSPORT_BW_STATE_RENDEZVOUS_TIMEOUT, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED);
        return;
    }

    /* TIMED. run_timed_dispatch selects both the burst length and the
     * boundary specialization before t0 and publishes the batch delta. */
    run_timed_dispatch<Backend>(s);

    /* PUBLISH -> DONE. logical_bytes is uint64_t(N) * uint64_t(SIZE);
     * secondary_batch_delta was set inside publish_delta. */
    r.terminal_state = static_cast<uint32_t>(TRANSPORT_BW_STATE_PUBLISH);
    const auto& a = Backend::args(s);
    r.logical_bytes = static_cast<uint64_t>(a.transfer_count) * static_cast<uint64_t>(a.transfer_size);

    ubench_finish<Backend>(s, TRANSPORT_BW_STATE_DONE, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED);
}

}  // namespace transport_bandwidth

#endif /* X280_TRANSPORT_BANDWIDTH_FRONTEND_CXX_HPP */
