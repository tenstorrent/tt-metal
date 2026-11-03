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
template <typename Connections>
struct SenderState {
    Connections connections;
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

template <typename State>
__attribute__((noinline)) void sender_pass(State& s, bool& send_yielded) {
    sender::pass(s.states.loop_local_progress(), s.states, s.connections, s.channels_in_flight, send_yielded);
}

__attribute__((noinline)) void receiver_pass(
    ReceiverState& r, DownstreamConnections& downstreams, const ChipCoordinates& my_chip) {
    receiver::pass(
        r.states.loop_local_progress(), r.credit_frame_due, r.states, r.credits, downstreams, my_chip);
}

template <typename WorkerInterfaces>
__attribute__((noinline)) void run(
    WorkerInterfaces& interfaces, const sender::UpstreamCredits& upstream, DownstreamConnections& downstreams) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const ChipCoordinates my_chip{
        routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    SenderState<decltype(sender::worker_connections<sender::worker_channels>(interfaces))> s{
        sender::worker_connections<sender::worker_channels>(interfaces)};
    sender::init_forwarding<sender::forwarding_channels>(s.states, upstream);
    ReceiverState r;
    local_write_init();

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        sender::restore_worker_credit_preset(s.states.loop_local_progress(), s.states, s.connections);
        while (next_round()) {
            round.send_yielded = false;
            round.sent_at_round_start = sender::total_sent(s.states.loop_local_progress(), s.states);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, downstreams, my_chip);
                sender_pass(s, round.send_yielded);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    sender::total_sent(s.states.loop_local_progress(), s.states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, true>(
                    [] { return sender::any_channel_holds_packet(); }, [] { return receiver::vc0_register_empty(); });
            }
            if (firmware_ran) {
                sender::restore_worker_credit_preset(s.states.loop_local_progress(), s.states, s.connections);
            }
        }
    }
}

}  // namespace fabric_router::single
