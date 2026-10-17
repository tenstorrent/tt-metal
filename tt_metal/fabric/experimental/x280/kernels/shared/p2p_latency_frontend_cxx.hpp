// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/shared/p2p_latency_frontend_cxx.hpp -- the templated C++ frontend
 * loop for a p2p_latency participant.
 *
 * One suite frontend owns rendezvous, warm-up, both roles, both direction
 * phases, timing, status, and result publication. One statically selected
 * backend per component class supplies hardware operations. No NOC, DMA, or
 * Ethernet API appears here -- the design's point is that participant rows
 * share one protocol loop so a source-code divergence is not mistaken for a
 * hardware difference.
 *
 * See docs/benchmarks/design.md, §Kernel frontend and backend,
 * §Kernel entry point, and §Timed loop for the semantics this file implements.
 *
 * A backend supplies:
 *
 *   struct state_t;
 *   struct timestamp_t;
 *   struct counter_delta_t;
 *
 *   void            bind_direction(state_t& b, p2p_latency_direction_t d);
 *   timestamp_t     measurement_now(state_t& b);
 *   bool            delta(state_t& b, timestamp_t t0, timestamp_t t1,
 *                         counter_delta_t* primary);
 *     -- Writes the primary counter's t1 - t0 through measurement_delta.
 *        False, leaving *primary unchanged, when the interval is refused or
 *        either read failed.
 *
 *   template <p2p_latency_role_t role,
 *             p2p_latency_transfer_shape_t shape,
 *             p2p_latency_transmit_mode_t transmit_mode>
 *   void transmit(state_t& b, uint32_t token);
 *
 *   template <p2p_latency_poll_mode_t poll_mode>
 *   bool poll(state_t& b, uint32_t expected);
 *     -- Timed poll, no iteration bound. True once expected arrives; false
 *        on host STOP or the backend's own hang guard.
 *
 *   template <p2p_latency_poll_mode_t poll_mode>
 *   bool poll_for(state_t& b, uint32_t expected, uint64_t max_poll_cycles);
 *
 *   uint32_t observed(state_t& b);
 *
 * A member descriptor validates its flat argument section and initializes the
 * backend state before instantiating run_participant<Backend>().
 */
#ifndef X280_P2P_LATENCY_FRONTEND_CXX_HPP
#define X280_P2P_LATENCY_FRONTEND_CXX_HPP

#include <stdint.h>

#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "../../include/rt/measurement.h"

namespace p2p_latency {

/* --------------------------------------------------------------------------
 * Result publication helpers.
 *
 * A phase record and the result envelope have simple POD stores; a backend
 * publishes them outside the measurement bracket. Fencing is the backend's
 * business -- LIM stores on X280 need a preceding fence ow, ow if their
 * ordering vs a subsequent commit matters. The frontend never fences here
 * because these writes are never in the measurement path.
 * -------------------------------------------------------------------------- */
inline void initialize_result(p2p_latency_result_t& r, uint32_t backend_id) {
    r.backend_id = backend_id;
    r.reserved = 0u;
    r.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_INIT);
    r.path_sentinel = 0u;
    for (uint32_t p = 0u; p < 2u; ++p) {
        auto& ph = r.phases[p];
        ph.phase_ordinal = p;
        ph.direction = 0u;
        ph.role = 0u;
        ph.sample_kind = 0u;
        ph.samples_completed = 0u;
        ph.warmup_progress = 0u;
        ph.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_INIT);
        ph.reserved = 0u;
    }
}

inline constexpr p2p_latency_direction_t opposite(p2p_latency_direction_t d) {
    return d == P2P_LATENCY_DIRECTION_A_TO_B ? P2P_LATENCY_DIRECTION_B_TO_A : P2P_LATENCY_DIRECTION_A_TO_B;
}

inline constexpr bool this_participant_initiates(p2p_latency_participant_t self, p2p_latency_direction_t direction) {
    // A initiates when A_TO_B; B initiates when B_TO_A.
    return (self == P2P_LATENCY_PARTICIPANT_A && direction == P2P_LATENCY_DIRECTION_A_TO_B) ||
           (self == P2P_LATENCY_PARTICIPANT_B && direction == P2P_LATENCY_DIRECTION_B_TO_A);
}

/* --------------------------------------------------------------------------
 * Token derivation
 * --------------------------------------------------------------------------
 * A per-exchange token is monotonically greater than every token that could
 * appear earlier in a run, so a mailbox left latched by an earlier exchange
 * cannot satisfy a later poll. Host validation ensures the whole
 *
 *     run_nonce + total_tokens - 1 = run_nonce + 2 * (W + N) - 1
 *
 * interval fits in u32; the kernel increments token in the hot loop without
 * a wrap branch (see design §Roles and mailbox model).
 */
inline constexpr uint32_t token_for(
    uint32_t run_nonce,
    uint32_t phase_ordinal,
    uint32_t warmup_plus_samples, /* W + N */
    uint32_t iteration /* 0..W+N-1 */) {
    return run_nonce + phase_ordinal * warmup_plus_samples + iteration;
}

/* --------------------------------------------------------------------------
 * Role derivation from (self, direction).
 *
 * Every member: A initiates A_TO_B; B initiates B_TO_A. So the
 * role is a pure function of self and direction, not a runtime choice.
 * -------------------------------------------------------------------------- */
inline constexpr p2p_latency_role_t role_for(p2p_latency_participant_t self, p2p_latency_direction_t direction) {
    return this_participant_initiates(self, direction) ? P2P_LATENCY_ROLE_INITIATOR : P2P_LATENCY_ROLE_RESPONDER;
}

/* --------------------------------------------------------------------------
 * Templated phase loop.
 * --------------------------------------------------------------------------
 * The measurement bracket differs by role. An initiator brackets the whole
 * exchange (transmit -> observed reply); a responder brackets its own turn
 * (observed request -> transmit-departure or completion). Neither timestamps
 * anything the peer reads.
 *
 * The transmit and poll modes are compile-time template parameters so that
 * one instantiation compiles to the sequence a single mode would in a hand-
 * written kernel. A backend supplies templated transmit() and poll()
 * operations parameterised over these same modes; there is no runtime
 * branch on TransmitMode inside the timed loop.
 *
 * On TIMEOUT during warm-up, the phase returns false with terminal_state set
 * to TIMEOUT and warmup_progress set to the iteration index that failed.
 * Timed polling is unbounded; the harness's host STOP is the outer safety
 * net (see design §Rendezvous and warm-up). A timed poll that returns false
 * ends the phase the same way, with samples_completed the samples before it.
 */
template <
    typename Backend,
    p2p_latency_transfer_shape_t Shape,
    p2p_latency_poll_mode_t PollMode,
    p2p_latency_transmit_mode_t TransmitMode>
bool run_phase(
    typename Backend::state_t& backend_state,
    const p2p_latency_common_args_t& args,
    p2p_latency_direction_t direction,
    uint32_t phase_ordinal,
    p2p_latency_result_t& result) {
    p2p_latency_phase_record_t& phase = result.phases[phase_ordinal];

    const p2p_latency_participant_t self = static_cast<p2p_latency_participant_t>(args.self);
    const p2p_latency_role_t role = role_for(self, direction);
    const uint32_t W = args.warmup_count;
    const uint32_t N = args.sample_count;
    const uint32_t total = W + N;

    phase.phase_ordinal = phase_ordinal;
    phase.direction = static_cast<uint32_t>(direction);
    phase.role = static_cast<uint32_t>(role);
    phase.sample_kind = static_cast<uint32_t>(
        role == P2P_LATENCY_ROLE_INITIATOR ? P2P_LATENCY_SAMPLE_KIND_RTT : P2P_LATENCY_SAMPLE_KIND_TURN);
    phase.samples_completed = 0u;
    phase.warmup_progress = 0u;
    phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_RENDEZVOUS);

    Backend::bind_direction(backend_state, direction);

    /* Warm-up and rendezvous share the same shape: one exchange per iteration
     * with a bounded poll. The rendezvous is iteration 0 and is not called
     * out separately -- the initiator's first transmit is what makes it a
     * rendezvous, and the bounded poll ensures we never spin forever waiting
     * for a peer that never launched. Phase 0 publishes the observed token,
     * including on timeout, to distinguish a wrong reply from no reply. */
    const bool publish_rendezvous = (phase_ordinal == 0u);

    for (uint32_t i = 0u; i < W; ++i) {
        const uint32_t out_token = token_for(args.run_nonce, phase_ordinal, total, i);
        const uint32_t in_token = out_token; /* responder echoes same token */
        bool arrived;

        if (role == P2P_LATENCY_ROLE_INITIATOR) {
            Backend::template transmit<Shape, TransmitMode>(backend_state, out_token);
            arrived = Backend::template poll_for<PollMode>(backend_state, in_token, args.warmup_max_poll_cycles);
        } else {
            arrived = Backend::template poll_for<PollMode>(backend_state, in_token, args.warmup_max_poll_cycles);
            if (arrived) {
                Backend::template transmit<Shape, TransmitMode>(backend_state, out_token);
            }
        }

        if (i == 0u && publish_rendezvous) {
            result.path_sentinel = Backend::observed(backend_state);
        }
        if (!arrived) {
            phase.warmup_progress = i;
            phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_TIMEOUT);
            return false;
        }
        if (i == 0u) {
            phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_WARMUP);
        }
    }

    phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_TIMED);

    /* Timed loop. Bracket differs by role. Polls here are unbounded -- the
     * host STOP is the ultimate cutoff where backend::poll checks it (X280
     * backends: only under P2P_LATENCY_POLL_CHECKS_STOP), and a bounded
     * rdcycle poll would truncate the latency tail this suite exists to
     * characterise. */
    for (uint32_t i = 0u; i < N; ++i) {
        const uint32_t iteration = W + i;
        const uint32_t out_token = token_for(args.run_nonce, phase_ordinal, total, iteration);
        const uint32_t in_token = out_token;

        typename Backend::timestamp_t t0;
        typename Backend::timestamp_t t1;
        bool arrived;

        /* The initiator tests arrival after t1, so the bracket holds the same
         * instructions whether or not the poll succeeded. */
        if (role == P2P_LATENCY_ROLE_INITIATOR) {
            t0 = Backend::measurement_now(backend_state);
            Backend::template transmit<Shape, TransmitMode>(backend_state, out_token);
            arrived = Backend::template poll<PollMode>(backend_state, in_token);
            t1 = Backend::measurement_now(backend_state);
        } else {
            arrived = Backend::template poll<PollMode>(backend_state, in_token);
            if (arrived) {
                t0 = Backend::measurement_now(backend_state);
                Backend::template transmit<Shape, TransmitMode>(backend_state, out_token);
                t1 = Backend::measurement_now(backend_state);
            }
        }
        if (!arrived) {
            phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_TIMEOUT);
            phase.samples_completed = i;
            return false;
        }

        /* Post-bracket: the delta assignment lives outside the timed window
         * so the store is not itself accounted for. samples_completed is
         * updated once at loop exit rather than every iteration -- the
         * timed loop is unbounded on-device, so the only ways we leave with
         * partial completion are a poll returning false, handled above, and
         * a refused delta, handled below. */
        uint64_t primary = 0;
        uint64_t secondary = 0;
        if (!Backend::delta(backend_state, t0, t1, &primary) || t0.secondary == UINT64_MAX ||
            t1.secondary == UINT64_MAX || primary > UINT32_MAX ||
            !measurement_delta(t0.secondary, t1.secondary, &secondary) || secondary > UINT32_MAX) {
            phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_TRAP);
            phase.samples_completed = i;
            return false;
        }
        phase.raw_deltas[i] = static_cast<uint32_t>(primary);
        phase.secondary_deltas[i] = static_cast<uint32_t>(secondary);
    }

    phase.samples_completed = N;
    phase.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_DONE);
    return true;
}

/* --------------------------------------------------------------------------
 * Two-phase driver.
 *
 * Runs first_direction as phase 0 and its opposite as phase 1. The direction
 * swap is a runtime phase, not a separate build (design §Roles and mailbox
 * model). Returns false if either phase failed; the phase records carry the
 * per-phase terminal state.
 */
template <
    typename Backend,
    p2p_latency_transfer_shape_t Shape,
    p2p_latency_poll_mode_t PollMode,
    p2p_latency_transmit_mode_t TransmitMode>
bool run_participant(
    typename Backend::state_t& backend_state, const p2p_latency_common_args_t& args, p2p_latency_result_t& result) {
    const p2p_latency_direction_t first = static_cast<p2p_latency_direction_t>(args.first_direction);
    const p2p_latency_direction_t second = opposite(first);

    const bool phase0_ok = run_phase<Backend, Shape, PollMode, TransmitMode>(backend_state, args, first, 0u, result);
    if (!phase0_ok) {
        result.terminal_state = result.phases[0].terminal_state;
        return false;
    }

    const bool phase1_ok = run_phase<Backend, Shape, PollMode, TransmitMode>(backend_state, args, second, 1u, result);
    if (!phase1_ok) {
        result.terminal_state = result.phases[1].terminal_state;
        return false;
    }

    result.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_DONE);
    return true;
}

}  // namespace p2p_latency

#endif /* X280_P2P_LATENCY_FRONTEND_CXX_HPP */
