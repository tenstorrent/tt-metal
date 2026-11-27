// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "router_init.hpp"
#include "router_receiver.hpp"
#include "router_sender.hpp"

namespace fabric_router {

FORCE_INLINE bool sync_requested(Phase phase) {
    return (read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) ==
           (0x0A00 | static_cast<uint32_t>(phase));
}

template <Phase PHASE>
FORCE_INLINE bool finish_step(bool outcome, InterDomainState s) {
    if constexpr (!is_teardown_master) {
        while (!sync_requested(PHASE)) {
        }
        outcome = inter_domain_control()->state == s;
    }
    sync_erisc(PHASE);
    return outcome;
}

}  // namespace fabric_router

namespace fabric_router::receiver {

FORCE_INLINE void drain() {
    while (!sync_requested(Phase::DRAINED)) {
        for_each_index<NUM_RECEIVER_CHANNELS>([](auto rx) {
            constexpr uint32_t stream = to_receiver_packets_sent_streams[decltype(rx)::value];
            const uint32_t n = read_stream_count<stream>();
            if (n != 0) {
                inter_domain_control()->drained_packets += n;
                increment_local_update_ptr_val<stream>(-static_cast<int32_t>(n));
            }
        });
    }
}

__attribute__((noinline, cold)) void reset_link() {
    zero_credit_counters(local_receiver_ack_counters_base_address);
    zero_credit_counters(local_receiver_completion_counters_base_address);
    init_channel_streams();
}

}  // namespace fabric_router::receiver

namespace fabric_router::sender {

__attribute__((noinline)) void restart_worker_channels(
    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& connection_semaphores,
    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& producer_cursors) {
    for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
        constexpr size_t CH = decltype(i)::value;
        if constexpr (is_sender_channel_serviced[CH] && is_sender_channel_worker[CH]) {
            reset_connection<CH>(connection_semaphores, producer_cursors);
            init_ptr_val<SenderChannel<CH>::free_slots_stream>(SENDER_NUM_BUFFERS_ARRAY[CH]);
            SenderChannelInterfaces::start_one<CH>(connection_semaphores);
        }
    });
}

FORCE_INLINE void run_routing_now_and_then(uint32_t& polls) {
    if (polls++ == POLLS_PER_ROUTING_RUN) {
        polls = 0;
        run_routing();
    }
}

FORCE_INLINE bool drain() {
    set_inter_domain_state(InterDomainState::DRAIN);
    uint32_t polls = 0;
    while (!terminating()) {
        if (inter_domain_command() == InterDomainCommand::CONNECT) {
            set_inter_domain_state(InterDomainState::CONNECT);
            return true;
        }
        run_routing_now_and_then(polls);
    }
    return false;
}

__attribute__((noinline, cold)) bool connect() {
    using namespace erisc::datamover::handshake;
    using Remote = RemoteReceiverChannel<0>;
    auto* c = inter_domain_control();
    c->peer_marker = 0;
    bool connected = c->peer_domain_id != c->local_domain_id && c->peer_vc2_receiver_addr == Remote::first_slot_addr &&
                     c->peer_vc2_receiver_slots == Remote::num_slots &&
                     c->peer_max_packet_size_bytes == channel_buffer_size;
    if (connected) {
        const auto& rt = l1_routing_table();
        const auto aborted = [] { return terminating() || inter_domain_command() == InterDomainCommand::ABORT; };
        if (c->local_domain_id < c->peer_domain_id) {
            fabric_sender_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                rt.my_mesh_id,
                rt.my_device_id,
                termination_signal(),
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT,
                aborted);
        } else {
            fabric_receiver_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                rt.my_mesh_id,
                rt.my_device_id,
                termination_signal(),
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT,
                aborted);
        }
        auto* info = reinterpret_cast<volatile tt_l1_ptr handshake_info_t*>(handshake_addr);
        connected = info->local_value == MAGIC_HANDSHAKE_VALUE;
        if (connected) {
            telemetry_block()->static_info.neighbor_mesh_id = info->neighbor_mesh_id;
            telemetry_block()->static_info.neighbor_device_id = info->neighbor_device_id;
        }
    }
    if (connected) {
        c->connects++;
    } else {
        c->connect_failures++;
    }
    set_inter_domain_state(connected ? InterDomainState::SERVICE_TRAFFIC : InterDomainState::DRAIN);
    c->command = InterDomainCommand::NONE;
    return connected;
}

FORCE_INLINE bool start_disconnect() {
    if (terminating()) {
        return false;
    }
    set_inter_domain_state(InterDomainState::DISCONNECT);
    return true;
}

__attribute__((noinline, cold)) void exchange_markers() {
    auto* c = inter_domain_control();
    if (inter_domain_command() == InterDomainCommand::DISCONNECT) {
        c->marker[0] = c->connects;
        SenderLinkTransport::send_with_context_switch(
            reinterpret_cast<uint32_t>(&c->marker[0]),
            reinterpret_cast<uint32_t>(&c->peer_marker),
            PACKET_WORD_SIZE_BYTES);
        set_inter_domain_state(InterDomainState::WAIT_PEER_MARKER);
        uint32_t polls = 0;
        while (c->peer_marker == 0 && inter_domain_command() != InterDomainCommand::FORCE_DISCONNECT &&
               !terminating()) {
            run_routing_now_and_then(polls);
        }
    }
    c->peer_marker = 0;
    c->command = InterDomainCommand::NONE;
}

__attribute__((noinline, cold)) void reset_link(const RuntimeArgs& args) {
    zero_credit_counters(to_sender_remote_ack_counters_base_address);
    zero_credit_counters(to_sender_remote_completion_counters_base_address);
    restart_worker_channels(args.connection_semaphores, args.producer_cursors);
    for_each_index<NUM_SENDER_CHANNELS>([](auto i) {
        constexpr size_t CH = decltype(i)::value;
        if constexpr (((SenderChannels::forwarding_channels >> CH) & 1u) != 0) {
            auto& state = carried_forwarding.template get<CH>();
            auto& progress = state.progress;
            const uint32_t returned =
                SenderChannel<CH>::first_level_ack ? progress.first_level_acked : progress.completed;
            if (progress.sent != returned) {
                ForwardingChannel<CH>::credit(state.upstream_free_slots_addr, progress.sent - returned);
            }
            progress.sent = progress.completed = progress.first_level_acked = 0;
        }
    });
}

}  // namespace fabric_router::sender

namespace fabric_router {

template <typename Loop>
FORCE_INLINE void run_inter_domain(const RuntimeArgs& args, Loop&& loop) {
    for (;;) {
        bool on = false;
        if constexpr (is_teardown_master) {
            on = sender::drain();
        } else {
            receiver::drain();
        }
        if (!finish_step<Phase::DRAINED>(on, InterDomainState::CONNECT)) {
            break;
        }
        if constexpr (is_teardown_master) {
            on = sender::connect();
        }
        if (!finish_step<Phase::PEER_CONNECTED>(on, InterDomainState::SERVICE_TRAFFIC)) {
            continue;
        }
        loop();
        if constexpr (is_teardown_master) {
            on = sender::start_disconnect();
        }
        if (!finish_step<Phase::SERVICE_DONE>(on, InterDomainState::DISCONNECT)) {
            break;
        }
        init_context_switch_state();
        if constexpr (is_teardown_master) {
            sender::exchange_markers();
        }
        sync_erisc(Phase::MARKERS_EXCHANGED);
        if constexpr (is_receiver_role) {
            receiver::reset_link();
        }
        if constexpr (is_sender_role) {
            sender::reset_link(args);
        }
    }
}

}  // namespace fabric_router
