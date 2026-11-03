// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the far router's sender channels, and for each one
// forwards it to the next router on this chip, writes it to this chip, or both, then returns the slot's credit.
//
// Each pass, one step per serviced receiver channel (in channel order; step), the same for every channel:
//   ack       channels with first-level acks (VC0 on rings and tori): ack every packet that has arrived
//   poll      packets known present: the count from the packets-received register (read in the pass's register group)
//             is taken only when the count kept from its last read is used up (the register stays the source of truth;
//             the kept count is a lower bound of it); under first-level acks, the packets acked and not yet processed
//   decode    the route: deliver here, and which downstream routers to forward to (router_routing.hpp; with the
//             landing and exit steps of multi-mesh fabrics)
//   admit     every downstream the packet goes to has a free slot (router_downstream.hpp)
//   forward   one copy per downstream
//   deliver   the packet's NoC commands on this chip (router_local_delivery.hpp); in UDM mode, a copy to the fabric's
//             relay instead, which does them (router_downstream.hpp; admitted with the forwards)
//   retire    the slot's record (its sender channel), packets-received register -1, next slot
//   release   completion credits back to the far sender (router_completions.hpp)
// A channel that only delivers (VC2's, and VC0's on a speedy-path router) skips decode and admit (Channel::routes).
// Around the steps, once per pass:
//   registers every channel's packets-received register and, where a channel has first-level acks (a frame goes out
//             most passes), the TX queue's status, as one group
//   frame     one credit frame, when the pass's acks or completions changed the counts
//
// Channels and VCs. A receiver channel's features are its VC's (Channel<CH>, the one place a channel's VC is named):
// VC0's channel is 0, VC1's 1, VC2's 2, or 1 on routers without VC1.
//
// State, by placement class (V4_RECEIVER_SHAPE.md):
//   - left to the compiler (registers where they fit): VC0's cursor (its position in the ring and its completion counts,
//     per packet) and whether a credit frame is due, each a loop local of its own (inside a larger aggregate GCC would
//     keep parts of them in memory);
//   - in memory (`Tables`): each channel's table (its slot records, indexed by ring slot at run time); VC1's and VC2's
//     cursor with its table in one record (E1's loop is at its register limit with VC0's cursor, and VC1's as more
//     loop-carried values cost VC0 +8 cycles/packet, RESULTS §98 v4p);
//   - the credit sender, an object of its own: its counts are in memory (indexed by sender channel), its block
//     addresses are constants (inside an aggregate with the tables they would be loaded from memory);
//   - compile-time: ring bounds, stream ids and command buffers.
// The downstream connections are passed in, by VC (router_downstream.hpp).

#include "router_common.hpp"
#include "router_completions.hpp"
#include "router_downstream.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::receiver {

// ---------------------------------------------------------------------------------------------------------------------
// Channels and their features
// ---------------------------------------------------------------------------------------------------------------------

// Per VC: whether its receiver traffic reaches this router (VC1's only where the router services VC1 traffic, as main:
// an intermesh router without VC1 pass-through has the channel, but no VC1 traffic comes to it; VC2's where the router
// has VC2).
constexpr std::array<bool, 3> vc_is_received = {
    true,
#if defined(FABRIC_2D_VC1_SERVICED)
    true,
#else
    false,
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    true,
#else
    false,
#endif
};
// Per VC: whether its packets are first-level acked (the host's settings; VC2's never).
constexpr std::array<bool, 3> vc_has_first_level_acks = {ENABLE_FIRST_LEVEL_ACK_VC0, ENABLE_FIRST_LEVEL_ACK_VC1, false};
// Per VC: whether its channel only delivers (VC2's; VC0's on a speedy-path router: the host's ENABLE_SPEEDY_VC0, a
// router whose VC0 carries only its worker's packets to the next chip, e.g. neighbour exchange, as main's speedy
// receiver).
constexpr std::array<bool, 3> vc_delivers_only = {super_speedy_mode, false, true};

// Whether this router runs receiver channel CH: a channel of its VCs, unless channel trimming took it out. (From the VC
// tables, not Channel<CH>: an absent channel's traits are never computed.)
template <size_t CH>
constexpr bool runs_channel() {
    return CH < NUM_RECEIVER_CHANNELS && vc_is_received[vc_of_receiver_channel(CH)] && is_receiver_channel_serviced[CH];
}

// Receiver channel CH: its VC, its ring and packets-received register, and its features, which are its VC's.
template <size_t CH>
struct Channel {
    static constexpr size_t vc = vc_of_receiver_channel(CH);

    // The ring: num_slots slots of channel_buffer_size bytes from first_slot_addr, each starting with its header.
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot_addr = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[CH];

    // Features.
    // Decodes each packet's route, then forwards and/or delivers; or (a channel that only delivers) every packet is for
    // this chip: no route to decode, nothing to admit.
    static constexpr bool routes = !vc_delivers_only[vc];
    // First-level acks: the far sender returns a packet's slot to its producer when the packet has arrived here, on an
    // ack, rather than when it is completed (acknowledge_arrivals).
    static constexpr bool acks = vc_has_first_level_acks[vc];
    // Intermesh ingress: the channel's packets come from another mesh and land here (router_routing.hpp).
    static constexpr bool lands = receiver_channel_is_intermesh_ingress[CH];
    // The VC of the downstream routers the channel forwards to: its own, or VC1 on an intermesh router with crossover.
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    static constexpr size_t downstream_vc = 1;
#else
    static constexpr size_t downstream_vc = vc;
#endif
    // UDM: packets for this chip go to the fabric's relay (router_downstream.hpp), admitted with the forwards.
    static constexpr bool relays_delivery = udm_mode;
#if defined(FABRIC_2D)
    using Routing = Routing2D<downstream_vc, lands>;
#else
    using Routing = Routing1D;
#endif

    // Completions (router_completions.hpp). Slots per completion group: under first-level acks one at a time, as the
    // generic loop does: grouped there, completion credits were held, which stalls bubble flow control (RESULTS §98,
    // v4f2, v4p).
    static constexpr uint32_t group_size = acks ? 1 : completion_group_size<num_slots>;
    // The transaction IDs of the slots' NoC commands: one per completion group, from the channel's own range.
    using TransactionIds = GroupIds<group_size, RX_CH_TRID_STARTS[CH], NUM_TRANSACTION_IDS>;

    // The header address of slot `index`.
    static FORCE_INLINE uint32_t slot_addr(uint32_t index) { return first_slot_addr + index * channel_buffer_size; }
    // The ring index after `index`.
    static FORCE_INLINE uint32_t next_index(uint32_t index) { return index + 1 == num_slots ? 0 : index + 1; }

    // Credits go back as counts in one frame (send_credit_frame); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(CH));
    // The host turns a VC's forwarding off (DISABLE_RX_CH<vc>_FORWARDING) where its channel delivers every packet, and
    // where a channel-trimming capture saw nothing forwarded or written on the VC; such a channel also saw no packet, so
    // channel trimming does not service it. A serviced channel that routes with forwarding off would be a contradictory
    // capture, which the builder rejects (TT_FATAL); the router does not handle it.
    static_assert(
        !is_receiver_channel_serviced[CH] || !routes || !is_vc_forwarding_disabled[vc],
        "a serviced receiver channel that routes has forwarding off: an inconsistent channel-trimming capture");
    static_assert(num_slots > 0 && num_slots <= 255 && num_slots <= stream_count_mask);
};

// Whether channel CH runs here with first-level acks.
template <size_t CH>
constexpr bool channel_acks() {
    if constexpr (runs_channel<CH>()) {
        return Channel<CH>::acks;
    } else {
        return false;
    }
}
constexpr bool any_channel_acks = channel_acks<0>() || channel_acks<1>() || channel_acks<2>();

// The far router's sender channel that sent a packet, given the packet's header field. In Tensix mux mode the far router
// sends everything from its worker channel and does not stamp the field (it holds what the packet's previous hop
// wrote), so the channel is the worker channel and the field is not used.
FORCE_INLINE uint8_t source_channel(uint8_t header_src_ch_id) {
    if constexpr (skip_src_ch_id_update) {
        return 0;
    } else {
        return header_src_ch_id;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------------------------------

// Where a receiver channel is in its ring.
struct Position {
    uint32_t ready;      // packets this channel may process (present, and acked under first-level acks), not retired
    uint32_t slot_addr;  // header address of the next slot
    uint32_t index;      // ring index of the next slot
};

// Moves `position` to the next slot of channel CH's ring.
template <size_t CH>
FORCE_INLINE void advance(Position& position) {
    position.index = Channel<CH>::next_index(position.index);
    position.slot_addr =
        position.index == 0 ? Channel<CH>::first_slot_addr : position.slot_addr + channel_buffer_size;
}

// A receiver channel's per-packet scalars: its position and its completion counts.
struct Cursor {
    Position position;
    CompletionCounts counts;
};

// Channel CH's cursor where this router runs the channel (Present, router_common.hpp), and its value before the
// channel's first packet.
template <size_t CH>
using CursorOf = Present<runs_channel<CH>(), Cursor, CH>;

template <size_t CH>
constexpr CursorOf<CH> start() {
    if constexpr (runs_channel<CH>()) {
        return Cursor{Position{0, Channel<CH>::first_slot_addr, 0}, CompletionCounts{}};
    } else {
        return {};
    }
}

// Channel CH's table, in memory (indexed by ring slot at run time): the sender channel each retired slot's packet came
// from, for its completion credit.
template <size_t CH>
struct ChannelTable {
    SlotRecords<Channel<CH>::num_slots> sources{};
};

// A channel's cursor and table kept together in one record (VC1's, VC2's). The cursor stays in memory because GCC does
// not split a record with a member indexed at run time (the table) into registers: a placement by GCC's choice, not by
// contract. If that ever changes, VC1's and VC2's cursors take registers from VC0's (v4rx2: +9.4 on TorusY1HopNS scatter
// with VC2, V4_RECEIVER_SHAPE.md).
template <size_t CH>
struct ChannelState {
    Cursor cursor = start<CH>();
    ChannelTable<CH> table;
};

// The receiver's state by placement class (see the top of this file). A member exists only where this router runs its
// channel (Present, router_common.hpp).
struct Tables {
    [[no_unique_address]] Present<runs_channel<0>(), ChannelTable<0>, 0> ch0{};
    [[no_unique_address]] Present<runs_channel<1>(), ChannelState<1>, 1> ch1;
    [[no_unique_address]] Present<runs_channel<2>(), ChannelState<2>, 2> ch2;
};

// One credit sender for every receiver channel: counter credits are one L1 block indexed by the far sender channel, and
// the channels' far sender channels are different channels, so one frame per pass carries every channel's credits.
using Credits = ReceiverChannelCounterBasedResponseCreditSender;

// The downstreams of a channel that forwards nothing (one that only delivers).
struct NoDownstreams {};

// The downstream connections receiver channel CH forwards on: those of its downstream VC, none on a channel that only
// delivers.
template <size_t CH, typename Vc1Downstreams>
FORCE_INLINE auto& downstreams_for(
    Downstreams<0>& vc0_downstreams, Vc1Downstreams& vc1_downstreams, NoDownstreams& none) {
    if constexpr (!Channel<CH>::routes) {
        return none;
    } else if constexpr (Channel<CH>::downstream_vc == 0) {
        return vc0_downstreams;
    } else {
        return vc1_downstreams;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------------------------------------------------

// The packet in the next slot is done here (its NoC commands are issued): its source recorded for its completion credit
// and the slot counted uncompleted, the packets-received register -1 (under first-level acks, consumed at the ack),
// the next slot.
template <size_t CH>
FORCE_INLINE void retire(Cursor& cursor, SlotRecords<Channel<CH>::num_slots>& sources, uint32_t source) {
    Position& position = cursor.position;
    slot_retired(cursor.counts, sources, position.index, source);
    position.ready--;
    if constexpr (!Channel<CH>::acks) {
        increment_local_update_ptr_val<Channel<CH>::packets_received_stream>(-1);
    }
    advance<CH>(position);
}

// A packet taken on channel CH, `size` bytes with its header: the channel-trimming capture's and telemetry's count.
template <size_t CH>
FORCE_INLINE void note_received(uint32_t size) {
    channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
    if constexpr (telemetry::counts_received) {
        telemetry::count_received(size);
    }
}

// Channel-trimming capture: the sender channels, on the downstream routers, a packet forwarded on receiver channel CH went
// to (`forward_to`: the routing names of its downstreams), as main records them: in 2D each downstream's channel from
// the host's packed table (by the downstream's compact index), in 1D the downstream's forwarding channel, 1.
template <size_t CH>
FORCE_INLINE void record_forwarded(uint32_t forward_to) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        if constexpr (is_2d) {
            constexpr uint32_t packed = Channel<CH>::downstream_vc == 0 ? PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS
                                                                       : PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS;
            uint32_t entries = 0;
            for (uint32_t names = forward_to; names != 0; names &= names - 1) {
                entries |= 1u << downstream_entry(__builtin_ctz(names));
            }
            channel_trimming_usage_recorder.merge_sender_channel_forwarded_to(
                CH, tt::tt_fabric::forwarded_slots_to_sender_mask(static_cast<uint8_t>(entries), packed));
        } else {
            channel_trimming_usage_recorder.set_sender_channel_forwarded_to(CH, 1);
        }
    }
}

// A 1D sparse multicast that goes on past this chip: this chip writes its own page group (counts[chip_idx] pages from
// write_idx), write_idx and chip_idx advance so the next chip picks the next group, then the forward carries the
// advanced header. The write reads the indices before the advance and the forward reads the header after it, so for
// this send type the local write comes first (as on main). Out of line: rare.
template <size_t CH>
__attribute__((noinline)) void sparse_multicast_hop(
    tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint16_t payload_size,
    uint32_t route_word,
    uint32_t forward_to,
    Downstreams<Channel<CH>::downstream_vc>& downstreams,
    uint32_t slot_addr,
    uint32_t size,
    uint8_t trid) {
    execute_chip_unicast_to_local_chip_impl(
        header, payload_size, tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE, trid, CH);
    auto& sparse = header->command_fields.sparse_mcast_write;
    sparse.write_idx += sparse.counts[sparse.chip_idx];
    sparse.chip_idx++;
    Channel<CH>::Routing::before_forward(header, route_word);
    downstreams.forward_all(forward_to, slot_addr, size, trid);
    record_forwarded<CH>(forward_to);
}

// First-level acks. A packet that has arrived is acked to the far sender before it is processed, and the far sender
// returns its slot to the producer on the ack. Every packet that has arrived since the last pass is acked: its sender
// channel's ack count goes up (the frame goes out with the pass's other credits), and the packets-received register is
// consumed here rather than at retire. The packets acked are the oldest not yet acked: `ready` slots past the next
// one to process. `arrived` is the packets-received count, read at the start of the pass. Returns whether it acked any.
template <size_t CH>
FORCE_INLINE bool acknowledge_arrivals(Position& position, Credits& credits, uint32_t arrived) {
    if (arrived == 0) {
        return false;
    }
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    increment_local_update_ptr_val<Channel<CH>::packets_received_stream>(-static_cast<int32_t>(arrived));
    uint32_t index = ring_add<Channel<CH>::num_slots>(position.index, position.ready);
    for (uint32_t i = 0; i < arrived; i++) {
        const auto* header =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(Channel<CH>::slot_addr(index));
        add_ack(credits, source_channel(header->src_ch_id));
        index = ring_add<Channel<CH>::num_slots>(index, 1);
    }
    position.ready += arrived;
    return true;
}

// The packet in the cursor's slot, from its header word and (on a channel that routes) its route word: admit, forward
// and/or deliver, retire. Returns whether it handled it (false: a downstream had no room; the packet stays for a later
// pass).
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE bool handle(
    Cursor& cursor,
    ChannelTable<CH>& table,
    ChannelDownstreams& downstreams,
    uint32_t header_word,
    uint32_t route_word) {
    using Routing = typename Channel<CH>::Routing;
    const HeaderWord packet = HeaderWord::of(header_word);
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(cursor.position.slot_addr);
    if constexpr (Channel<CH>::routes) {
        const typename Routing::Route route(route_word);
        if (route.forwards() && !downstreams.admit(route.forward_to())) {
            return false;
        }
        if constexpr (Channel<CH>::relays_delivery) {
            if (route.delivers() && !downstreams.admit_relay()) {
                return false;
            }
        }
        const uint8_t trid = Channel<CH>::TransactionIds::of(cursor.position.index);
        const uint32_t size = packet.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
        note_received<CH>(size);
        if (!is_2d && route.delivers() && route.forwards() &&
            packet.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
            sparse_multicast_hop<CH>(
                header,
                packet.payload_size_bytes,
                route_word,
                route.forward_to(),
                downstreams,
                cursor.position.slot_addr,
                size,
                trid);
        } else {
            // Forward first (the next router starts sooner), then write here.
            if (route.forwards()) {
                Routing::before_forward(header, route_word);
                downstreams.forward_all(route.forward_to(), cursor.position.slot_addr, size, trid);
                record_forwarded<CH>(route.forward_to());
            }
            if (route.delivers()) {
                if constexpr (Channel<CH>::relays_delivery) {
                    downstreams.forward_to_relay(cursor.position.slot_addr, size, trid);
                } else {
                    deliver<CH>(header, packet.payload_size_bytes, packet.noc_send_type, trid);
                }
            }
        }
    } else {
        note_received<CH>(packet.payload_size_bytes + sizeof(PACKET_HEADER_TYPE));
        deliver<CH>(
            header,
            packet.payload_size_bytes,
            packet.noc_send_type,
            Channel<CH>::TransactionIds::of(cursor.position.index));
    }
    retire<CH>(cursor, table.sources, source_channel(packet.src_ch_id));
    return true;
}

// One packet, when one is present (acked, under first-level acks; otherwise counted present: `arrived` is the
// channel's packets-received count, read at the start of the pass) and (on a channel that routes) its downstreams have
// room: its header word and route word read, then handled. Returns whether it handled one.
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE bool receive(
    Cursor& cursor,
    ChannelTable<CH>& table,
    ChannelDownstreams& downstreams,
    uint32_t arrived,
    const Coordinates& me) {
    Position& position = cursor.position;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (position.ready == 0) {
        if constexpr (Channel<CH>::acks) {
            return false;  // nothing acked to process
        } else {
            position.ready = arrived;  // the register's count, read in the pass's group
            if (position.ready == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(position.slot_addr);
    const uint32_t header_word = HeaderWord::bits(header);
    uint32_t route_word = 0;
    if constexpr (Channel<CH>::routes) {
        route_word = Channel<CH>::Routing::read(header, me);
    }
    return handle<CH>(cursor, table, downstreams, header_word, route_word);
}

// ---------------------------------------------------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------------------------------------------------

// One step of receiver channel CH (`arrived_raw`: its packets-received register, read at the start of the pass): ack
// what has arrived (a channel with first-level acks), at most one packet (timed for code profiling, as main
// times every receiver channel), then its completions. `frame_due` is set when the credit counts changed.
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE void step(
    Cursor& cursor,
    ChannelTable<CH>& table,
    ChannelDownstreams& downstreams,
    Credits& credits,
    bool& frame_due,
    uint32_t arrived_raw,
    const Coordinates& me) {
    using ThisChannel = Channel<CH>;
    const uint32_t arrived = arrived_raw & stream_count_mask;
    if constexpr (ThisChannel::acks) {
        frame_due |= acknowledge_arrivals<CH>(cursor.position, credits, arrived);
    }
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    timer.set_should_dump(cursor.position.ready != 0);
    timer.open();
    const bool progress = receive<CH>(cursor, table, downstreams, arrived, me);
    timer.close();
    using Ids = typename ThisChannel::TransactionIds;
    frame_due |= release_completions<ThisChannel::group_size, ThisChannel::num_slots, Ids>(
        cursor.counts, table.sources, cursor.position.index, progress, credits);
}

// Channel CH's packets-received register, raw, for the pass's register group; none (0) where the channel does not run.
template <size_t CH>
FORCE_INLINE uint32_t read_packets_received() {
    if constexpr (runs_channel<CH>()) {
        return read_stream_raw<Channel<CH>::packets_received_stream>();
    } else {
        return 0;
    }
}

// Sends the credit frame, all the counts, unless the TX queue is busy (the next pass sends it then). Returns whether it
// was sent. `group_txq_status`: the TX queue's status, read in the pass's register group on routers with first-level
// acks (unused elsewhere); only this ERISC sends on the queue, so a queue seen free there is still free.
FORCE_INLINE bool send_credit_frame(Credits& credits, uint32_t group_txq_status) {
    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        }
    } else {
        const bool seen_free = any_channel_acks && !txq_busy(group_txq_status);
        if (!seen_free && internal_::eth_txq_is_busy(receiver_txq_id)) {
            return false;
        }
    }
    credits.flush_completion_credits();  // the whole credit block: completions and acks
    return true;
}

// Whether VC0's packets-received register reads 0 (telemetry's RX heartbeat, once per round).
FORCE_INLINE bool vc0_register_empty() { return read_stream_count<Channel<0>::packets_received_stream>() == 0; }

// One pass over the serviced receiver channels: the pass's register group, one step per channel, and the credit frame
// when the counts changed. The group: every channel's packets-received register and (with first-level acks, where a
// frame goes out most passes) the TX queue's status are read together, their reads overlapping, raw (each tested where
// it is used), into scalars of their own (a record across the fences is kept in memory). The body of run()'s loop,
// which calls it on its own locals; a single-ERISC router runs it as the receiver half of each pass over
// its state in memory (router_single.hpp). The state types are template parameters so that an absent channel's state is
// named only in template-dependent code (Present).
template <typename Vc0Cursor, typename TablesT, typename Vc1Downstreams>
FORCE_INLINE void pass(
    Vc0Cursor& vc0_cursor,
    bool& frame_due,
    TablesT& tables,
    Credits& credits,
    Downstreams<0>& vc0_downstreams,
    Vc1Downstreams& vc1_downstreams,
    const Coordinates& me) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t arrived0 = read_packets_received<0>();
    const uint32_t arrived1 = read_packets_received<1>();
    const uint32_t arrived2 = read_packets_received<2>();
    uint32_t txq_status = 0;
    if constexpr (any_channel_acks) {
        txq_status = read_txq_status(receiver_txq_id);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    NoDownstreams none;
    if constexpr (runs_channel<0>()) {
        step<0>(
            vc0_cursor,
            tables.ch0,
            downstreams_for<0>(vc0_downstreams, vc1_downstreams, none),
            credits,
            frame_due,
            arrived0,
            me);
    }
    if constexpr (runs_channel<1>()) {
        step<1>(
            tables.ch1.cursor,
            tables.ch1.table,
            downstreams_for<1>(vc0_downstreams, vc1_downstreams, none),
            credits,
            frame_due,
            arrived1,
            me);
    }
    if constexpr (runs_channel<2>()) {
        step<2>(
            tables.ch2.cursor,
            tables.ch2.table,
            downstreams_for<2>(vc0_downstreams, vc1_downstreams, none),
            credits,
            frame_due,
            arrived2,
            me);
    }
    if (frame_due) {
        frame_due = !send_credit_frame(credits, txq_status);
    }
}

// The receiver loop: rounds of passes over the serviced receiver channels until termination. A function of its own:
// its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
//
// The downstream connections are passed in by VC (VC1's an unused stand-in without VC1), and each channel forwards on
// its downstream VC's (downstreams_for): forwards use them, but packets delivered here do not, so they live in a table
// in memory rather than in the registers every packet needs (RESULTS §98, v4 bring-up).
template <typename Vc1Downstreams>
__attribute__((noinline)) void run(Downstreams<0>& vc0_downstreams, Vc1Downstreams& vc1_downstreams) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    CursorOf<0> vc0_cursor = start<0>();
    Tables tables;
    local_write_init();
    Credits credits(0);
    bool frame_due = false;  // the credit counts changed since the last frame

    RoundCounters round;
    while (next_round()) {
        if constexpr (telemetry::any_option) {
            telemetry::round_start();
        }
        for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
            pass(vc0_cursor, frame_due, tables, credits, vc0_downstreams, vc1_downstreams, me);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end<false, true>([] { return false; }, [] { return vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
