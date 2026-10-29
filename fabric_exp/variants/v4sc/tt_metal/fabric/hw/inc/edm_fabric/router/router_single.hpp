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
// there is no other ERISC to park or follow.

#include "router_receiver.hpp"
#include "router_sender.hpp"

namespace fabric_router::single {

// The sender's state between passes (sender::run keeps the same as loop locals).
template <typename WorkerInterface>
struct SenderState {
    WorkerInterface worker_interface;
    volatile tt_l1_ptr uint32_t* connection_semaphore;
    sender::Worker worker;
    sender::ForwardingTable forwarding;
    sender::FarRing far_vc0;
    sender::FarRing far_vc1;
};

// The receiver's state between passes (receiver::run keeps the same as loop locals).
struct ReceiverState {
    using C0 = receiver::Channel<0>;
    receiver::Position position0{0, C0::first_slot, 0};
    CompletionCounts counts0;
    SlotRecords<C0::num_slots> records0{};
    std::conditional_t<receiver::runs_vc1_receiver, receiver::ChannelState<1>, receiver::NoChannelState> vc1;
    // One credits object for every receiver channel, as receiver::run.
    ReceiverCredits credits = make_receiver_credits();
    bool credits_due = false;  // the credit counts changed since they last went out
};

template <typename State>
__attribute__((noinline)) void sender_pass(State& s, bool& attempted) {
    sender::pass(s.worker, s.worker_interface, s.connection_semaphore, s.forwarding, s.far_vc0, s.far_vc1, attempted);
}

template <typename Downstreams0, typename Downstreams1>
__attribute__((noinline)) void receiver_pass(
    ReceiverState& r, Downstreams0& downstreams0, Downstreams1& downstreams1, const Coordinates& me) {
    receiver::pass(r.position0, r.counts0, r.records0, r.vc1, r.credits, r.credits_due, downstreams0, downstreams1, me);
}

template <typename WorkerInterfaces, typename Downstreams1>
__attribute__((noinline)) void run(
    WorkerInterfaces& interfaces,
    const sender::UpstreamCredits& upstream,
    Downstreams<receiver::Channel<0>::downstream_vc>& downstreams0,
    Downstreams1& downstreams1) {
    using namespace tt::tt_fabric::connection_interface;
    using FarVc0 = sender::FarRingTraits<VC0_RECEIVER_CHANNEL>;
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    auto worker_interface = interfaces.template get<0>();
    SenderState<decltype(worker_interface)> s{
        worker_interface,
        worker_interface.connection_live_semaphore,
        {sender::Channel<0>::first_slot, 0, 0, 0, unused_connection_value},
        {},
        {FarVc0::num_slots, FarVc0::first_slot},
        {}};
    if constexpr (sender::vc1_forwarding != 0) {
        s.far_vc1 = sender::far_ring_start<VC1_RECEIVER_CHANNEL>();
    }
    sender::init_forwarding<sender::vc0_forwarding | sender::vc1_forwarding>(s.forwarding, upstream);
    ReceiverState r;
    local_write_init();

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        if (s.worker.expected_connection != unused_connection_value) {
            sender::preset_worker_credit(s.worker_interface);
        }
        while (next_round()) {
            round.attempted = false;
            round.sent_at_round_start = sender::total_sent(s.worker, s.forwarding);
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                receiver_pass(r, downstreams0, downstreams1, me);
                sender_pass(s, round.attempted);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.attempted ||
                    sender::total_sent(s.worker, s.forwarding) != round.sent_at_round_start);
            if (firmware_ran && s.worker.expected_connection != unused_connection_value) {
                sender::preset_worker_credit(s.worker_interface);
            }
        }
    }
}

}  // namespace fabric_router::single
