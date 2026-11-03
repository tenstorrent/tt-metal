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

// The sender's state between passes (sender::run keeps the same groups as loop locals).
template <typename Connections>
struct SenderState {
    Connections connections;
    sender::Workers workers;
    sender::ForwardingTable forwarding;
    uint32_t in_flight = 0;
    sender::FarRings far;
};

// The receiver's state between passes (receiver::run keeps the same groups as loop locals).
struct ReceiverState {
    [[no_unique_address]] receiver::CursorOf<0> cursor0 = receiver::cursor_start<0>();
    receiver::Tables tables;
    receiver::Credits credits{0};
    bool credit_frame_due = false;  // the credit counts changed since the last frame
};

template <typename State>
__attribute__((noinline)) void sender_pass(State& s, bool& attempted) {
    sender::pass(s.workers, s.connections, s.forwarding, s.in_flight, s.far, attempted);
}

__attribute__((noinline)) void receiver_pass(ReceiverState& r, DownstreamConnections& downstreams, const Coordinates& me) {
    receiver::pass(r.cursor0, r.credit_frame_due, r.tables, r.credits, downstreams, me);
}

template <typename WorkerInterfaces>
__attribute__((noinline)) void run(
    WorkerInterfaces& interfaces, const sender::UpstreamCredits& upstream, DownstreamConnections& downstreams) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    SenderState<decltype(sender::worker_connections(interfaces))> s{
        sender::worker_connections(interfaces), {}, {}, 0, sender::far_rings_start()};
    s.workers = sender::workers_start(s.connections);
    sender::init_forwarding<sender::all_forwarding>(s.forwarding, upstream);
    ReceiverState r;
    local_write_init();

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        sender::restore_worker_credit_preset(s.workers, s.connections);
        while (next_round()) {
            round.attempted = false;
            round.sent_at_round_start = sender::total_sent(s.workers, s.forwarding);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, downstreams, me);
                sender_pass(s, round.attempted);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.attempted ||
                    sender::total_sent(s.workers, s.forwarding) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, true>(
                    [] { return sender::any_channel_holds_packet(); }, [] { return receiver::vc0_register_empty(); });
            }
            if (firmware_ran) {
                sender::restore_worker_credit_preset(s.workers, s.connections);
            }
        }
    }
}

}  // namespace fabric_router::single
