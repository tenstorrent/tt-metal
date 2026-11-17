// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Inter-domain routers: a router on a link that leaves the fabric domain, its peer bound at run time. It skips the
// Ethernet handshake at init and sits in DRAIN. The host's CONNECT checks the peer's metadata and does the handshake
// with it, then the router routes (SERVICE_TRAFFIC) until the host's DISCONNECT: markers both ways (or none, forced),
// the link-facing state back to its start, DRAIN again. The host drives it through InterDomainRouterControl in L1.
// One ERISC only (fabric_erisc_router_ct_args.hpp).

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_link_transport.hpp"

#include "router_init.hpp"
#include "router_receiver.hpp"
#include "router_sender.hpp"

namespace fabric_router::inter_domain {

static_assert(!is_inter_domain_router || has_vc2, "inter-domain traffic rides VC2");

FORCE_INLINE InterDomainCommand command() {
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    return inter_domain_control()->command;
}

FORCE_INLINE void set_state(InterDomainState s) { inter_domain_control()->state = s; }

// Done with the host's command: the host waits for the state.
FORCE_INLINE void finish_command(InterDomainState s) {
    inter_domain_control()->command = InterDomainCommand::NONE;
    set_state(s);
}

// Lets the base firmware run now and then while the router waits, as the init waits do, so the link stays up.
struct FirmwareTicks {
    uint32_t polls = 0;
    FORCE_INLINE void tick() {
        if (++polls == sender::POLLS_PER_ROUTING_RUN) {
            polls = 0;
#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
            run_routing();
#endif
        }
    }
};

// Packets that land with no peer connected are counted and dropped.
template <size_t RX>
FORCE_INLINE void drop_stray_packets(volatile tt_l1_ptr InterDomainRouterControl* c) {
    constexpr uint32_t stream = to_receiver_packets_sent_streams[RX];
    const uint32_t n = read_stream_count<stream>();
    if (n != 0) {
        c->drained_packets += n;
        increment_local_update_ptr_val<stream>(-static_cast<int32_t>(n));
    }
}

// DRAIN until the host's CONNECT (true) or termination (false).
__attribute__((noinline)) bool drain() {
    auto* c = inter_domain_control();
    set_state(InterDomainState::DRAIN);
    FirmwareTicks ticks;
    while (!terminating()) {
        if (command() == InterDomainCommand::CONNECT) {
            return true;
        }
        for_each_index<NUM_RECEIVER_CHANNELS>([c](auto rx) { drop_stray_packets<decltype(rx)::value>(c); });
        ticks.tick();
    }
    return false;
}

// The peer's VC2 receiver must be where this router's VC0 senders send, as it was built to see it (no remote layout
// at run time yet), with the same packet size (no resizing yet). Domain ids must differ.
FORCE_INLINE bool peer_matches(volatile tt_l1_ptr InterDomainRouterControl* c) {
    using Remote = sender::RemoteReceiverChannel<0>;
    return c->peer_domain_id != c->local_domain_id && c->peer_vc2_receiver_addr == Remote::first_slot_addr &&
           c->peer_vc2_receiver_slots == Remote::num_slots && c->peer_max_packet_size_bytes == channel_buffer_size;
}

// CONNECT: the peer's metadata, then the Ethernet handshake with it; the lower domain id sends. True when connected;
// false (HANDLE_CONNECT_FAIL, then DRAIN) on a peer that doesn't match, the host's ABORT or termination.
__attribute__((noinline)) bool connect() {
    using namespace erisc::datamover::handshake;
    auto* c = inter_domain_control();
    set_state(InterDomainState::CONNECT);
    c->peer_marker = 0;
    bool connected = peer_matches(c);
    if (connected) {
        const auto& routing_table = l1_routing_table();
        const auto aborted = [] { return terminating() || command() == InterDomainCommand::ABORT; };
        if (c->local_domain_id < c->peer_domain_id) {
            fabric_sender_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table.my_mesh_id,
                routing_table.my_device_id,
                termination_signal(),
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT,
                aborted);
        } else {
            fabric_receiver_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table.my_mesh_id,
                routing_table.my_device_id,
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
    if (!connected) {
        set_state(InterDomainState::HANDLE_CONNECT_FAIL);
        c->connect_failures++;
        finish_command(InterDomainState::DRAIN);
        return false;
    }
    c->connects++;
    finish_command(InterDomainState::SERVICE_TRAFFIC);
    return true;
}

// The link-facing state back to its start for the next peer: credit counters, packets-received registers and the
// worker channels. Forwarding channels face this chip and keep theirs.
__attribute__((noinline)) void reset_channels(const RuntimeArgs& args) {
    zero_credit_counters(local_receiver_ack_counters_base_address);
    zero_credit_counters(local_receiver_completion_counters_base_address);
    zero_credit_counters(to_sender_remote_ack_counters_base_address);
    zero_credit_counters(to_sender_remote_completion_counters_base_address);
    receiver::init_channel_streams();
    sender::restart_worker_channels(args.connection_semaphores, args.producer_cursors);
}

// DISCONNECT: healthy, this router's marker to the peer and wait for the peer's (the host may force it on); forced,
// no markers. Then the channels reset and DRAIN.
__attribute__((noinline)) void disconnect(const RuntimeArgs& args) {
    auto* c = inter_domain_control();
    set_state(InterDomainState::DISCONNECT);
    if (command() == InterDomainCommand::DISCONNECT) {
        c->marker[0] = c->connects;  // nonzero after a connect
        SenderLinkTransport::send_with_context_switch(
            reinterpret_cast<uint32_t>(&c->marker[0]),
            reinterpret_cast<uint32_t>(&c->peer_marker),
            PACKET_WORD_SIZE_BYTES);
        set_state(InterDomainState::WAIT_PEER_MARKER);
        FirmwareTicks ticks;
        while (c->peer_marker == 0 && command() != InterDomainCommand::FORCE_DISCONNECT && !terminating()) {
            ticks.tick();
        }
    }
    c->peer_marker = 0;
    reset_channels(args);
    finish_command(InterDomainState::DRAIN);
}

}  // namespace fabric_router::inter_domain
