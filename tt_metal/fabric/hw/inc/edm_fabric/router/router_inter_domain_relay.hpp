// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// An inter-domain router whose link is down (routing plane mismatch, scenario 1: the domain routes on more planes than
// it has live links to the peer domain). The domain still lands its traffic for this side on this router's plane, so
// the router stays: it sends nothing over its link and relays all it would send, every sender channel on every VC,
// over the NoC into the relay sender (VC0) of a same-side router on another plane (is_inter_domain_relay; the host's
// pick, fabric_graph_inputs.cpp relay_planes), which sends it over its own link (or relays it on, down a chain that
// ends at a live link).
//
// To that router this one is the relay sender's one producer, as an upstream router is a forwarding channel's: it
// connects once, keeps a copy of the relay sender's free slots that the target credits, and rings its free-slots
// register (-1) after each packet. To its own producers it is a sender: worker connections, and a slot credited back
// once its packet has left this router's L1. One packet at a time per pass, in channel order; a channel's packets stay
// in order, and the relay sender keeps them so.
//
// Only the sender ERISC works. The receiver ERISC has no link to receive from and runs rounds only.

#include "router_init.hpp"
#include "router_round.hpp"
#include "router_sender.hpp"

namespace fabric_router::relay {

using sender::ForwardingChannel;
using sender::SenderChannel;
using sender::SenderChannels;
using sender::SenderChannelState;
using sender::WorkerChannel;

// The relay sender this router relays into, on a router of this chip.
struct Target {
    static constexpr uint32_t num_slots = inter_domain_relay_num_slots;
    static constexpr uint32_t last_slot_offset = (num_slots - 1) * channel_buffer_size;
    // Its free-slots register (the doorbell): the table is by fabric position, the same on every router.
    static constexpr uint32_t remote_free_slots_stream = sender_free_slots_streams[inter_domain_relay_sender_position];
    // This router's copy of the target's free slots, which the target credits (a VC0 downstream register: a perimeter
    // router has no VC0 receiver, so nothing else here uses it; the host checks).
    static constexpr uint32_t local_credit_stream = inter_domain_relay_credit_stream;
    // The sender ERISC's NoC (its producers' credits and connections use it too).
    static constexpr uint8_t noc = tt::tt_fabric::worker_handshake_noc;
    // A command buffer nothing else on this ERISC uses (no preset to keep), for the packet copies.
    static constexpr uint8_t data_cmd_buf = [] {
        // the read buffer too: relay() sets the source coordinate, which a write does not (init presets it only on the
        // write buffers)
        for (uint8_t buf :
             {static_cast<uint8_t>(write_cmd_buf),
              static_cast<uint8_t>(write_reg_cmd_buf),
              static_cast<uint8_t>(read_cmd_buf)}) {
            if (buf != CommandBuffers::atomics && CommandBuffers::kinds_on[buf] == 0) {
                return buf;
            }
        }
        return static_cast<uint8_t>(NUM_NOC_CMD_BUFS);
    }();
    // A transaction ID for the copies: receiver channel 0's first; the receiver ERISC here never issues one.
    static constexpr uint8_t trid = RX_CH_TRID_STARTS[0];

    static_assert(!is_inter_domain_relay || (num_slots > 0 && num_slots <= stream_count_mask));
    static_assert(!is_inter_domain_relay || local_credit_stream != tt::tt_fabric::k_unused_stream_id);
    // a slot is credited once its packet leaves this router: no first-level acks (rings and tori with deadlock
    // avoidance), whose worker credits count acks
    static_assert(
        !is_inter_domain_relay ||
            !(is_vc_first_level_ack_enabled[0] || is_vc_first_level_ack_enabled[1] || is_vc_first_level_ack_enabled[2]),
        "an inter-domain relay does not take first-level acks");
    static_assert(
        !is_inter_domain_relay || !is_sender_role || data_cmd_buf < NUM_NOC_CMD_BUFS,
        "no free command buffer for the relay");

    uint64_t noc_base;     // the target's NoC coordinate, as a NoC address's high bits
    uint32_t local_coord;  // this core's NoC coordinate: the copies' source
    uint64_t doorbell;  // the target's free-slots register
    uint32_t first_slot_addr;
    uint32_t next_slot_addr;
    uint32_t free_slots;  // known free there: a lower bound of the local copy, re-read only at zero

    // Has a free slot. The known count is tested first; the local copy is re-read only when it is used up.
    FORCE_INLINE bool admit() {
        if (free_slots == 0) {
            free_slots = read_stream_count<local_credit_stream>();
        }
        return free_slots != 0;
    }

    // The packet at `src` (`size` bytes) into the target's next slot, then its doorbell; returns once the copy has left
    // this router's L1 (the slot at `src` is free again).
    FORCE_INLINE void relay(uint32_t src, uint32_t size) {
        const uint32_t dst = next_slot_addr;
        while (!noc_cmd_buf_ready(noc, data_cmd_buf)) {
        }
        noc_cmd_buf_set_targ_addr_coordinate(noc, data_cmd_buf, local_coord);
        noc_async_write_one_packet_with_trid</*update_counter=*/false, /*posted=*/true>(
            src, noc_base | dst, size, trid, data_cmd_buf, noc);
        // after the data on the same NoC and VC, as a forward's credit follows its packet
        noc_inline_dw_write<InlineWriteDst::REG>(
            doorbell, pack_value_for_inc_on_write_stream_reg_write(-1), INLINE_WRITE_ALL_BYTES, noc);
        next_slot_addr = dst == first_slot_addr + last_slot_offset ? first_slot_addr : dst + channel_buffer_size;
        free_slots--;
        increment_local_update_ptr_val<local_credit_stream>(-1);
        while (!noc_cmd_buf_ready(noc, data_cmd_buf)) {
        }
        while (!ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)) {
        }
    }
};

// Before the Ethernet handshake: the local copy of the target's free slots at its slot count, the teardown word at 0
// and the link from the runtime args.
__attribute__((noinline)) void init(Target& target, const InterDomainRelayArgs& args) {
    init_ptr_val<Target::local_credit_stream>(Target::num_slots);
    *reinterpret_cast<volatile uint32_t*>(args.teardown_word) = 0;
    target.noc_base = get_noc_addr(args.noc_x, args.noc_y, 0, Target::noc);
    target.local_coord =
        static_cast<uint32_t>(
            get_noc_addr(my_x[Target::noc], my_y[Target::noc], 0, Target::noc) >> NOC_ADDR_COORD_SHIFT) &
        NOC_COORDINATE_MASK;
    target.doorbell =
        get_noc_addr(args.noc_x, args.noc_y, get_stream_reg_write_addr(Target::remote_free_slots_stream), Target::noc);
    target.first_slot_addr = args.first_slot_addr;
    target.next_slot_addr = args.first_slot_addr;
    target.free_slots = 0;
}

// After the handshake: register with the target as its relay sender's producer, its credits to go to the local copy
// (as a downstream connection opens, router_downstream.hpp open_downstream).
__attribute__((noinline)) void open(const InterDomainRelayArgs& args) {
    constexpr bool posted = !is_blackhole_arch;
    tt::tt_fabric::connection::open_start</*SEND_CREDIT_ADDR=*/false, posted, tt::tt_fabric::worker_handshake_noc>(
        args.location_info,
        get_stream_reg_write_addr(Target::local_credit_stream),
        args.teardown_word,
        args.noc_x,
        args.noc_y);
    tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
        args.registration, reinterpret_cast<volatile uint32_t*>(args.teardown_word), args.noc_x, args.noc_y);
}

template <size_t CH>
FORCE_INLINE void credit_producer(SenderChannelState<CH>& state) {
    // the absolute count to a worker (out of line), +1 on an upstream router's register
    state.progress.completed++;
    if constexpr (SenderChannel<CH>::worker) {
        WorkerChannel<CH>::credit(state.progress.completed, WorkerChannel<CH>::connected(state));
    } else {
        ForwardingChannel<CH>::credit(state.upstream_free_slots_addr, 1);
    }
}

// Channel CH: its worker's connection where it has one, then its next packet into the target where both have one.
// Returns whether it relayed.
template <size_t CH>
FORCE_INLINE bool visit(SenderChannelState<CH>& state, Target& target) {
    if constexpr (SenderChannel<CH>::worker) {
        const uint32_t semaphore =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WorkerChannel<CH>::connection_semaphore_addr);
        WorkerChannel<CH>::check_connection(state, semaphore);
    }
    if (!SenderChannel<CH>::holds_packet(read_stream_raw<SenderChannel<CH>::free_slots_stream>()) || !target.admit()) {
        return false;
    }
    const uint32_t slot = state.progress.next_slot_addr;
    auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(slot);
    const uint32_t size = header->get_payload_size_excluding_header() + sizeof(PACKET_HEADER_TYPE);
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(CH);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(CH, static_cast<uint16_t>(size));
    }
    target.relay(slot, size);
    state.progress.next_slot_addr =
        slot == SenderChannel<CH>::last_slot_addr ? SenderChannel<CH>::first_slot_addr : slot + channel_buffer_size;
    state.progress.sent++;
    increment_local_update_ptr_val<SenderChannel<CH>::free_slots_stream>(1);
    credit_producer<CH>(state);
    return true;
}

template <uint32_t MASK, typename States>
FORCE_INLINE bool visit_all(States& states, Target& target) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool relayed = visit<CH>(states.template get<CH>(), target);
        return visit_all<without_lowest_bit(MASK)>(states, target) || relayed;
    } else {
        return false;
    }
}

template <uint32_t MASK, typename States>
FORCE_INLINE void start_forwarding_channels(States& states) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        states.template get<CH>().upstream_free_slots_addr = ForwardingChannel<CH>::upstream_free_slots_addr();
        start_forwarding_channels<without_lowest_bit(MASK)>(states);
    }
}

// The base firmware may have used the command buffers (a router command, the context switch): the connected worker
// channels' credit presets again.
template <typename States>
FORCE_INLINE void restore_worker_credit_presets(States& states) {
    for_each_index<NUM_SENDER_CHANNELS>([&](auto i) {
        constexpr size_t CH = decltype(i)::value;
        if constexpr (CommandBuffers::presets && ((SenderChannels::worker_channels >> CH) & 1u) != 0) {
            if (WorkerChannel<CH>::connected(states.template get<CH>())) {
                WorkerChannel<CH>::preset_credit();
            }
        }
    });
}

// Whether to run another round: until termination or (master) a router command. Inter-domain commands are not for a
// relaying router (it never connects), so unlike next_round they do not end its rounds.
FORCE_INLINE bool next_relay_round() {
    if constexpr (is_context_switch_master) {
        return !terminating() && !router_state()->is_non_run_command_pending();
    } else {
        return !terminating();
    }
}

// The relay loop (sender ERISC), for the router's lifetime: rounds of passes over every serviced sender channel, with
// router commands between runs of rounds. Its state is its own: it never connects, so nothing carries over.
__attribute__((noinline)) void run(Target& target) {
    Each<SenderChannels::serviced, SenderChannelState> states;
    start_forwarding_channels<SenderChannels::forwarding_channels>(states);
    RoundCounters counters;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_presets(states);
        while (next_relay_round()) {
            bool relayed = false;
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                relayed = visit_all<SenderChannels::serviced>(states, target) || relayed;
            }
            if (end_round(counters, relayed)) {
                restore_worker_credit_presets(states);
            }
        }
    }
}

// The receiver ERISC of a relaying router: nothing to receive; rounds only (it follows the context switch), until
// termination.
__attribute__((noinline)) void run_idle() {
    RoundCounters counters;
    while (!terminating()) {
        end_round(counters, false);
    }
}

}  // namespace fabric_router::relay
