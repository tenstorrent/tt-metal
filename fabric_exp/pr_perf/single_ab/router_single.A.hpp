// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The single-ERISC router (Blackhole with two-ERISC mode off; Wormhole): one ERISC runs both roles. Each role
// keeps its two-ERISC pass (sender::pass, receiver::pass) and its state. The receiver's pass runs first: it frees slots
// and sends the credits the remote sender is waiting on.
// Where the passes run (SINGLE_ERISC_PASS): on Blackhole out of line, each over its state in memory, so neither role's
// code holds the other's state in registers; on Wormhole inline, with the receiver's loop-local progress a local of
// the loop as in receiver::run, since there reloading and storing each role's state every pass costs more than keeping
// the roles' states apart saves.
// Rounds, router commands, the heartbeat and the context switch are the sender loop's (router_round.hpp), with no other
// ERISC to park. A role channel trimming left without channels makes empty passes.

#include <type_traits>

#include "router_receiver.hpp"
#include "router_sender.hpp"

// Where the passes run (above): out of line on Blackhole, inline on Wormhole.
#if defined(ARCH_BLACKHOLE)
#define SINGLE_ERISC_PASS __attribute__((noinline))
#elif defined(ARCH_WORMHOLE)
#define SINGLE_ERISC_PASS FORCE_INLINE
#endif

namespace fabric_router::single {

// Whether the passes run inline (SINGLE_ERISC_PASS), which lets the loop carry the receiver's loop-local progress.
constexpr bool inline_passes = is_wormhole_arch;

// The sender's state between passes: sender::run's objects, with the loop-local send state kept here, not copied.
struct SenderState {
    sender::SenderChannelStates states;
    uint32_t channels_in_flight = 0;
};

// The receiver's state between passes: receiver::run's objects, with every VC's progress kept here, not copied, except
// the loop-local VC's on Wormhole, which the loop copies as receiver::run does.
struct ReceiverState {
    receiver::ReceiverChannelStates states;
    receiver::Credits credits{0};
    bool credit_frame_due = false;  // the credit counts changed since the last frame
};

SINGLE_ERISC_PASS void sender_pass(SenderState& s, bool& send_yielded) {
    sender::pass(s.states.loop_local_progress(), s.states, s.channels_in_flight, send_yielded);
}

template <typename LoopCopy>
SINGLE_ERISC_PASS void receiver_pass(
    ReceiverState& r, LoopCopy& loop_copy, DownstreamConnections& downstreams, const ChipCoordinates& my_chip) {
    receiver::pass(loop_copy, r.credit_frame_due, r.states, r.credits, downstreams, my_chip);
}

__attribute__((noinline)) void run(DownstreamConnections& downstreams) {
    const auto& routing_table = l1_routing_table();
    const ChipCoordinates my_chip{
        routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    SenderState s;
    sender::start_forwarding_channels(s.states);
    ReceiverState r;
    // The receiver's loop-local progress: a copy carried through the loop where the passes are inline, else the
    // record's own.
    using RxProgress = std::remove_reference_t<decltype(r.states.loop_local_progress())>;
    std::conditional_t<inline_passes, RxProgress, RxProgress&> rx_loop_copy = r.states.loop_local_progress();
    local_write_init();

    sender::SenderRound round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        sender::restore_worker_credit_presets(s.states.loop_local_progress(), s.states);
        while (next_round()) {
            sender::before_round(round, s.states.loop_local_progress(), s.states);
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, rx_loop_copy, downstreams, my_chip);
                sender_pass(s, round.send_yielded);
            }
            sender::after_round<true>(
                round, s.states.loop_local_progress(), s.states, [] { return receiver::vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::single

#undef SINGLE_ERISC_PASS
