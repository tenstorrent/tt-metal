// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The single-ERISC router (Wormhole; Blackhole with two-ERISC mode off): one ERISC runs both roles. Each role keeps its
// two-ERISC pass (sender::pass, receiver::pass) and its own state; here each pass runs out of line over a record of its
// state in memory, so neither role's code has to hold the other's state in registers (V4_SINGLE_ERISC.md). The
// receiver's pass runs first: it frees ring slots and returns the credits the far sender is waiting on.
//
// Rounds, router commands, the heartbeat and the context switch are the sender loop's (router_round.hpp); with one ERISC
// there is no other ERISC to park or follow. Channel trimming may leave a role without channels: its pass is then empty.

#include "router_receiver.hpp"
#include "router_sender.hpp"

namespace fabric_router::single {

// The sender's state between passes (sender::run keeps the same objects; there the loop-local send state is copied
// into a loop local, here it stays in the state: sender::SenderChannelStates).
struct SenderState {
    sender::SenderChannelStates states;
    uint32_t channels_in_flight = 0;
};

// The receiver's state between passes (receiver::run keeps the same objects; there the loop-local VC's progress is
// copied into a loop local, here every VC's stays in the state: receiver::ReceiverChannelStates).
struct ReceiverState {
    receiver::ReceiverChannelStates states;
    receiver::Credits credits{0};
    bool credit_frame_due = false;  // the credit counts changed since the last frame
};

__attribute__((noinline)) void sender_pass(SenderState& s, bool& send_yielded) {
    sender::pass(s.states.loop_local_progress(), s.states, s.channels_in_flight, send_yielded);
}

__attribute__((noinline)) void receiver_pass(
    ReceiverState& r, DownstreamConnections& downstreams, const ChipCoordinates& my_chip) {
    receiver::pass(
        r.states.loop_local_progress(), r.credit_frame_due, r.states, r.credits, downstreams, my_chip);
}

__attribute__((noinline)) void run(DownstreamConnections& downstreams) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const ChipCoordinates my_chip{
        routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    SenderState s;
    sender::init_forwarding<sender::SenderChannels::forwarding_channels>(s.states);
    ReceiverState r;
    local_write_init();

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        sender::restore_worker_credit_presets(s.states.loop_local_progress(), s.states);
        while (next_round()) {
            sender::before_round(round, s.states.loop_local_progress(), s.states);
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, downstreams, my_chip);
                sender_pass(s, round.send_yielded);
            }
            sender::after_round<true>(
                round, s.states.loop_local_progress(), s.states, [] { return receiver::vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::single
