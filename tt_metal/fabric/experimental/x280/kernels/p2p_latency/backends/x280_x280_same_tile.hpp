// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/p2p_latency/backends/x280_x280_same_tile.hpp -- x280_same_tile X280 backend.
 *
 * ID 2 is X280 hart <-> X280 hart on the same L2CPU tile. Both participants
 * share one LIM, so a transmit is a plain LIM store from the initiator's
 * hart into the responder's inbound mailbox and a poll is a plain load from
 * this hart's own inbound. No NOC window is consumed, no address translation
 * is required. The medium selector chooses coherent cached LIM (initial) or
 * uncached LIM (deferred; the header rejects it until its explicit visibility
 * operation is separately established).
 *
 * The design's clock-solipsistic rule holds: each hart reads only its own
 * rdcycle and mtime and never sends or receives timestamps. The result region's raw
 * deltas are in this hart's cycles, which never become seconds (docs/measurement.md,
 * "Native CPU cycles never become nanoseconds").
 *
 * See docs/benchmarks/design.md §Roles and mailbox model,
 * §Timing and clock domains, and §Participant rows for ID 2's semantic
 * contract.
 */
#ifndef X280_P2P_LATENCY_BACKEND_X280_X280_SAME_TILE_HPP
#define X280_P2P_LATENCY_BACKEND_X280_X280_SAME_TILE_HPP

#include <cstdint>

#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "p2p_latency_frontend_cxx.hpp"
#include "x280_same_tile_args.h"

#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "rt/x280_measurement.h"

namespace p2p_latency::backend_x280_x280_same_tile {

/* --------------------------------------------------------------------------
 * Backend types
 * -------------------------------------------------------------------------- */

struct timestamp_t {
    uint64_t cycles;
    uint64_t secondary;
};

using counter_delta_t = uint64_t;

struct state_t {
    /* Addresses reified from the arg block into typed pointers so the hot
     * loop does not repeat the cast. `volatile` on both endpoints because a
     * peer hart is a concurrent writer at inbound and a concurrent reader at
     * peer_inbound. */
    volatile uint32_t* own_inbound_token;  /* own_inbound_addr + token_offset */
    volatile uint32_t* peer_inbound_token; /* peer_inbound_addr + token_offset */

    /* Hart identity for STOP polling. Read once at init; never inside the hot
     * loop's arithmetic path. */
    uint64_t hartid;
    uint64_t run_value;

    /* Set by initialize(). Not used inside the timed loop for ID 2, but the
     * frontend calls bind_direction() every phase and future members may want a
     * per-direction rebinding. Kept for parity. */
    uint32_t direction;
};

/* --------------------------------------------------------------------------
 * Initialisation.
 *
 * Called before either phase runs. Parses the backend args section, binds
 * pointers, and publishes backend_id into the result.
 * Called EXACTLY once per participant per invocation.
 *
 * Returns false on any layout mismatch or invalid backend arg. The image
 * calls this and, on false, sets STATUS = ARG_LAYOUT_MISMATCH and returns
 * without executing the loop.
 * -------------------------------------------------------------------------- */
inline bool initialize(
    state_t& state,
    const p2p_latency_common_args_t& common,
    const p2p_latency_x280_same_tile_backend_args_t& backend_args,
    uint64_t hartid,
    uint64_t run_value,
    p2p_latency_result_t& result) {
    if (backend_args.medium != P2P_LATENCY_X280_SAME_TILE_MEDIUM_COHERENT) {
        /* Uncached LIM medium is deferred: the design admits it only after
         * its explicit visibility operation is separately established, and
         * this member does not stand up an unproven artifact/configuration
         * pair. */
        return false;
    }

    if (common.payload_size != 4u) {
        /* ID 2 implements only the token-only specialisation. Larger
         * payloads are a separate template instantiation of the frontend
         * and its backend transmit; they can be added alongside the
         * payload sweep without touching this member. */
        return false;
    }

    if (backend_args.own_inbound_addr == 0u || backend_args.peer_inbound_addr == 0u ||
        backend_args.own_result_addr == 0u) {
        return false;
    }
    if ((backend_args.own_inbound_addr % 4u) != 0u || (backend_args.peer_inbound_addr % 4u) != 0u) {
        return false;
    }

    const uint32_t tok = p2p_latency_token_offset(common.payload_size);
    state.own_inbound_token = reinterpret_cast<volatile uint32_t*>(backend_args.own_inbound_addr + tok);
    state.peer_inbound_token = reinterpret_cast<volatile uint32_t*>(backend_args.peer_inbound_addr + tok);
    state.hartid = hartid;
    state.run_value = run_value;
    state.direction = 0u;

    p2p_latency::initialize_result(result, P2P_LATENCY_BACKEND_ID_X280_X280_SAME_TILE);
    return true;
}

/* Phase entry hook. ID 2's addresses are direction-agnostic (both harts read
 * their own inbound and write to the peer's inbound regardless of who is
 * initiator this phase), so this only records the current direction. */
inline void bind_direction(state_t& state, p2p_latency_direction_t direction) {
    state.direction = static_cast<uint32_t>(direction);
}

/* --------------------------------------------------------------------------
 * Measurement primitives
 * --------------------------------------------------------------------------
 * The fenced counter read comes from rt/x280_measurement.h; the delta is
 * modulo-u64 subtraction. The frontend rejects deltas exceeding the compact
 * UINT32_MAX per-sample storage bound before publishing them.
 */
inline timestamp_t measurement_now(const state_t&) {
    const x280_paired_t now = x280_paired_now();
    return timestamp_t{now.cpu, now.wall};
}

inline bool delta(const state_t&, timestamp_t t0, timestamp_t t1, counter_delta_t* primary) {
    return measurement_delta(t0.cycles, t1.cycles, primary) != 0;
}

/* --------------------------------------------------------------------------
 * Transmit
 * --------------------------------------------------------------------------
 * A store into peer_inbound_token. FIRE_AND_FORGET is a bare volatile store.
 * WAIT_FOR_COMPLETION follows the store with `fence ow, ow` so a later local
 * load-of-observation (or the peer's inbound-store-buffer drain via its own
 * `fence ir, ir` before the poll) sees the write.
 *
 * WAIT_FOR_DEPARTURE has no established primitive on X280 for a local LIM
 * store; the design rejects that mode for X280, and instantiating the
 * template with it fails to compile.
 */
template <p2p_latency_transfer_shape_t Shape, p2p_latency_transmit_mode_t TransmitMode>
inline void transmit(state_t& state, uint32_t token) {
    static_assert(Shape == P2P_LATENCY_TRANSFER_TOKEN_ONLY, "ID 2 supports only the token-only specialisation");
    static_assert(
        TransmitMode != P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE,
        "X280 has no established departure-only primitive; the design rejects this mode");

    *state.peer_inbound_token = token;
    if constexpr (TransmitMode == P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION) {
        x280_fence_ow();
    }
    /* FIRE_AND_FORGET: no fence. The measurement bracket's own fence io, io
     * on the enclosing rdcycle_end anchors the store's position for the
     * compiler, but does not force the store to have committed to LIM before
     * t1 is read. That is precisely what this mode measures. */
}

/* --------------------------------------------------------------------------
 * Poll
 * --------------------------------------------------------------------------
 * The load-side fence pattern is `fence ir, ir` before every read: without
 * it a LIM poll can wedge on a stale word even through a `volatile`
 * (rt/x280_idle.h).
 *
 * The warm-up poll checks STOP so the host can always end the wait
 * (x280/README.md § 6, obligation 1). The timed poll checks it only under
 * P2P_LATENCY_POLL_CHECKS_STOP; without it the timed poll is unbounded and a
 * token that never arrives strands the hart until `tt-smi -r`. The timed poll
 * returns true once expected arrives, false on STOP.
 */
template <p2p_latency_poll_mode_t PollMode>
inline bool poll(state_t& state, uint32_t expected) {
    /* Held in a register: the fence's "memory" clobber would otherwise reload
     * it from `state`, on the uncached LIM stack, every pass. */
    volatile uint32_t* const token = state.own_inbound_token;
    for (;;) {
        x280_fence_ir();
        if (*token == expected) {
            return true;
        }
#ifdef P2P_LATENCY_POLL_CHECKS_STOP
        if (x280_hart_stop_requested(state.hartid, state.run_value)) {
            return false;
        }
#endif
        if constexpr (PollMode == P2P_LATENCY_POLL_BACKOFF) {
            x280_pause();
        }
    }
}

/* Bounded poll for warm-up. `max_ticks` is in this participant's rdcycle
 * ticks, which stop while the hart stalls and so do not track wall time
 * (docs/measurement.md "Unconfirmed hardware assumptions",
 * item 5). Returns true if the token arrived within the budget, false
 * otherwise or on STOP.
 *
 * The budget only has to be generous enough that legitimate warm-up traffic
 * cannot exhaust it; the tail we care about is the timed loop, not the
 * warm-up. On STOP the caller falls out via the shared stop-check.
 */
template <p2p_latency_poll_mode_t PollMode>
inline bool poll_for(state_t& state, uint32_t expected, uint64_t max_ticks) {
    volatile uint32_t* const token = state.own_inbound_token;
    const uint64_t t0 = x280_rdcycle();
    for (;;) {
        x280_fence_ir();
        if (*token == expected) {
            return true;
        }
        if (x280_hart_stop_requested(state.hartid, state.run_value)) {
            return false;
        }
        if ((x280_rdcycle() - t0) > max_ticks) {
            return false;
        }
        if constexpr (PollMode == P2P_LATENCY_POLL_BACKOFF) {
            x280_pause();
        }
    }
}

inline uint32_t observed(state_t& state) {
    x280_fence_ir();
    return *state.own_inbound_token;
}

/* --------------------------------------------------------------------------
 * Adapter for the frontend's Backend template argument.
 *
 * The frontend calls Backend::transmit<>, Backend::poll<>, etc. via the
 * template parameter. Wrapping the functions in a struct with static methods
 * gives it exactly one type to bind Backend to.
 */
struct Backend {
    using state_t = ::p2p_latency::backend_x280_x280_same_tile::state_t;
    using timestamp_t = ::p2p_latency::backend_x280_x280_same_tile::timestamp_t;
    using counter_delta_t = ::p2p_latency::backend_x280_x280_same_tile::counter_delta_t;

    static inline void bind_direction(state_t& s, p2p_latency_direction_t d) {
        ::p2p_latency::backend_x280_x280_same_tile::bind_direction(s, d);
    }
    static inline timestamp_t measurement_now(const state_t& s) {
        return ::p2p_latency::backend_x280_x280_same_tile::measurement_now(s);
    }
    static inline bool delta(const state_t& s, timestamp_t t0, timestamp_t t1, counter_delta_t* primary) {
        return ::p2p_latency::backend_x280_x280_same_tile::delta(s, t0, t1, primary);
    }
    template <p2p_latency_transfer_shape_t Shape, p2p_latency_transmit_mode_t Tx>
    static inline void transmit(state_t& s, uint32_t token) {
        ::p2p_latency::backend_x280_x280_same_tile::transmit<Shape, Tx>(s, token);
    }
    template <p2p_latency_poll_mode_t Pm>
    static inline bool poll(state_t& s, uint32_t expected) {
        return ::p2p_latency::backend_x280_x280_same_tile::poll<Pm>(s, expected);
    }
    template <p2p_latency_poll_mode_t Pm>
    static inline bool poll_for(state_t& s, uint32_t expected, uint64_t max_ticks) {
        return ::p2p_latency::backend_x280_x280_same_tile::poll_for<Pm>(s, expected, max_ticks);
    }
    static inline uint32_t observed(state_t& s) { return ::p2p_latency::backend_x280_x280_same_tile::observed(s); }
};

}  // namespace p2p_latency::backend_x280_x280_same_tile

#endif /* X280_P2P_LATENCY_BACKEND_X280_X280_SAME_TILE_HPP */
