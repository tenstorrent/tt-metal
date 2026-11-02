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
// A channel that only delivers (VC2's in neighbour mode, and VC0's on a speedy-path router) skips decode and admit
// (Channel::routes). VC2's channel in line mode routes as VC0's does, on the one downstream of a line VC.
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
// The downstream connections are passed in, by VC (router_downstream.hpp): VC0's, VC1's (a stand-in without VC1), and
// VC2's where VC2 forwards (line mode).

#include "router_common.hpp"
#include "router_completions.hpp"
#include "router_downstream.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::receiver {

// Per VC: whether its receiver traffic reaches this router (VC1's only where the router services VC1 traffic, as main:
// an intermesh router without VC1 pass-through has the channel, but no VC1 traffic comes to it; VC2's where the router
// has VC2), whether its packets are first-level acked (the host's settings: VC0's and, in line mode on a ring axis,
// VC2's; VC1's never), and whether its channel only
// delivers (VC2's in neighbour mode; VC0's on a speedy-path router: the host's ENABLE_SPEEDY_VC0, a router whose VC0
// carries only its worker's packets to the next chip, e.g. neighbour exchange, as main's speedy receiver).
constexpr std::array<bool, 3> vc_received = {
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
constexpr std::array<bool, 3> first_level_ack_on = {
    ENABLE_FIRST_LEVEL_ACK_VC0, ENABLE_FIRST_LEVEL_ACK_VC1, ENABLE_FIRST_LEVEL_ACK_VC2};
constexpr std::array<bool, 3> delivers_only = {super_speedy_mode, false, !vc2_line_mode};

// Whether this router runs receiver channel CH: a channel of its VCs, unless channel trimming took it out. (From the VC
// tables, not Channel<CH>: an absent channel's traits are never computed.)
template <size_t CH>
constexpr bool runs_channel() {
    return CH < NUM_RECEIVER_CHANNELS && vc_received[vc_of_receiver_channel(CH)] && is_receiver_channel_serviced[CH];
}

// Receiver channel CH: its VC, its ring and packets-received register, and its features, which are its VC's.
template <size_t CH>
struct Channel {
    static constexpr size_t vc = vc_of_receiver_channel(CH);
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[CH];
    // Decodes each packet's route, then forwards and/or delivers; or (a channel that only delivers) every packet is for
    // this chip: no route to decode, nothing to admit.
    static constexpr bool routes = !delivers_only[vc];
    // First-level acks: the far sender returns a packet's slot to its producer when the packet has arrived here, on an
    // ack, rather than when it is completed (acknowledge_arrivals).
    static constexpr bool acks = first_level_ack_on[vc];
    // Intermesh ingress: the channel's packets come from another mesh and land here (router_routing.hpp).
    static constexpr bool lands = receiver_channel_is_intermesh_ingress[CH];
    // The VC of the downstream routers the channel forwards to: its own, or VC1 on an intermesh router with crossover
    // (VC0's packets from another mesh, and in VC2 line mode VC2's, fan into VC1 there).
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    static constexpr size_t downstream_vc = 1;
#else
    static constexpr size_t downstream_vc = vc;
#endif
    // UDM: packets for this chip go to the fabric's relay (router_downstream.hpp), admitted with the forwards.
    static constexpr bool relays_delivery = udm_mode;
    // Slots per completion group (router_completions.hpp). Under first-level acks one at a time, as the generic loop
    // does: grouped there, completion credits were held, which stalls bubble flow control (RESULTS §98, v4f2, v4p).
    static constexpr uint32_t completion_group = acks ? 1 : completion_group_size<num_slots>;
    // The transaction IDs of the slots' NoC commands: one per completion group, from the channel's own range.
    using Ids = GroupIds<completion_group, RX_CH_TRID_STARTS[CH], NUM_TRANSACTION_IDS>;
#if defined(FABRIC_2D)
    using Routing = Routing2D<downstream_vc, lands>;
#else
    using Routing = Routing1D;
#endif
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

// Where a receiver channel is in its ring.
struct Position {
    uint32_t available;  // packets this channel may process (present, and acked under first-level acks), not retired
    uint32_t slot;       // header address of the next slot
    uint32_t index;      // ring index of the next slot
};

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
        return Cursor{Position{0, Channel<CH>::first_slot, 0}, CompletionCounts{}};
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

// The packet in the next slot is done here (its NoC commands are issued): its source recorded for its completion credit
// and the slot counted uncompleted, the packets-received register -1 (under first-level acks, consumed at the ack),
// the next slot.
template <size_t CH>
FORCE_INLINE void retire(Cursor& c, SlotRecords<Channel<CH>::num_slots>& sources, uint32_t source) {
    using C = Channel<CH>;
    Position& p = c.position;
    slot_retired(c.counts, sources, p.index, source);
    p.available--;
    if constexpr (!C::acks) {
        increment_local_update_ptr_val<C::packets_received_stream>(-1);
    }
    p.index = p.index + 1 == C::num_slots ? 0 : p.index + 1;
    p.slot = p.index == 0 ? C::first_slot : p.slot + channel_buffer_size;
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
// to (`to`: the routing names of its downstreams), as main records them: in 2D each downstream's channel from the
// host's packed table (by the downstream's compact index), in 1D the downstream's forwarding channel, 1.
template <size_t CH>
FORCE_INLINE void record_forwarded(uint32_t to) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        if constexpr (is_2d) {
            constexpr uint32_t packed = Channel<CH>::downstream_vc == 0   ? PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS
                                        : Channel<CH>::downstream_vc == 1 ? PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS
                                                                          : PACKED_DOWNSTREAM_VC2_SENDER_CHANNEL_IDS;
            uint32_t compact = 0;
            for (uint32_t rest = to; rest != 0; rest &= rest - 1) {
                compact |= 1u << downstream_entry(__builtin_ctz(rest));
            }
            channel_trimming_usage_recorder.merge_sender_channel_forwarded_to(
                CH, tt::tt_fabric::forwarded_slots_to_sender_mask(static_cast<uint8_t>(compact), packed));
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
    uint32_t to,
    Downstreams<Channel<CH>::downstream_vc>& downstreams,
    uint32_t slot,
    uint32_t size,
    uint8_t trid) {
    execute_chip_unicast_to_local_chip_impl(
        header, payload_size, tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE, trid, CH);
    auto& sparse = header->command_fields.sparse_mcast_write;
    sparse.write_idx += sparse.counts[sparse.chip_idx];
    sparse.chip_idx++;
    Channel<CH>::Routing::before_forward(header, route_word);
    downstreams.forward_all(to, slot, size, trid);
    record_forwarded<CH>(to);
}

// First-level acks. A packet that has arrived is acked to the far sender before it is processed, and the far sender
// returns its slot to the producer on the ack. Every packet that has arrived since the last pass is acked: its sender
// channel's ack count goes up (the frame goes out with the pass's other credits), and the packets-received register is
// consumed here rather than at retire. The packets acked are the oldest not yet acked: `available` slots past the next
// one to process. `arrived` is the packets-received count, read at the start of the pass. Returns whether it acked any.
template <size_t CH>
FORCE_INLINE bool acknowledge_arrivals(Position& p, Credits& credits, uint32_t arrived) {
    using C = Channel<CH>;
    if (arrived == 0) {
        return false;
    }
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    increment_local_update_ptr_val<C::packets_received_stream>(-static_cast<int32_t>(arrived));
    uint32_t index = ring_add<C::num_slots>(p.index, p.available);
    for (uint32_t k = 0; k < arrived; k++) {
        const auto* header =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(C::first_slot + index * channel_buffer_size);
        add_ack(credits, source_channel(header->src_ch_id));
        index = ring_add<C::num_slots>(index, 1);
    }
    p.available += arrived;
    return true;
}

// The downstreams of a channel that forwards nothing (one that only delivers).
struct NoDownstreams {};

// The downstream connections receiver channel CH forwards on: those of its downstream VC (d0: VC0's, d1: VC1's, d2:
// VC2's where VC2 forwards), none on a channel that only delivers.
template <size_t CH, typename D1, typename... D2>
FORCE_INLINE auto& downstreams_for(Downstreams<0>& d0, D1& d1, NoDownstreams& none, D2&... d2) {
    using C = Channel<CH>;
    if constexpr (!C::routes) {
        return none;
    } else if constexpr (C::downstream_vc == 0) {
        return d0;
    } else if constexpr (C::downstream_vc == 1) {
        return d1;
    } else {
        static_assert(sizeof...(D2) == 1, "a channel forwarding on VC2 needs VC2's downstreams");
        return (d2, ...);
    }
}

// The packet in the cursor's slot, from its header word and (on a channel that routes) its route word: admit, forward
// and/or deliver, retire. Returns whether it handled it (false: a downstream had no room; the packet stays for a later
// pass).
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE bool handle(
    Cursor& c, ChannelTable<CH>& t, ChannelDownstreams& downstreams, uint32_t word_bits, uint32_t route_word) {
    using C = Channel<CH>;
    const HeaderWord word = HeaderWord::of(word_bits);
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(c.position.slot);
    if constexpr (C::routes) {
        const typename C::Routing::Route route(route_word);
        // A line VC's packets only go straight on: a route that turns (or exits the mesh sideways) is a flow the host
        // should have refused; its turn is masked and the packet would not arrive (watcher builds stop here).
        if constexpr (vc_routes_as_line(C::downstream_vc)) {
            ASSERT(route.turns() == 0);
        }
        if (route.forwards() && !downstreams.admit(route.forward_to())) {
            return false;
        }
        if constexpr (C::relays_delivery) {
            if (route.delivers() && !downstreams.admit_relay()) {
                return false;
            }
        }
        const uint8_t trid = C::Ids::of(c.position.index);
        const uint32_t size = word.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
        note_received<CH>(size);
        if (!is_2d && route.delivers() && route.forwards() &&
            word.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
            sparse_multicast_hop<CH>(
                header, word.payload_size_bytes, route_word, route.forward_to(), downstreams, c.position.slot, size, trid);
        } else {
            // Forward first (the next router starts sooner), then write here.
            if (route.forwards()) {
                C::Routing::before_forward(header, route_word);
                downstreams.forward_all(route.forward_to(), c.position.slot, size, trid);
                record_forwarded<CH>(route.forward_to());
            }
            if (route.delivers()) {
                if constexpr (C::relays_delivery) {
                    downstreams.forward_to_relay(c.position.slot, size, trid);
                } else {
                    deliver<CH>(header, word.payload_size_bytes, word.noc_send_type, trid);
                }
            }
        }
    } else {
        note_received<CH>(word.payload_size_bytes + sizeof(PACKET_HEADER_TYPE));
        deliver<CH>(header, word.payload_size_bytes, word.noc_send_type, C::Ids::of(c.position.index));
    }
    retire<CH>(c, t.sources, source_channel(word.src_ch_id));
    return true;
}

// One packet, when one is present (acked, under first-level acks; otherwise counted present: `arrived` is the
// channel's packets-received count, read at the start of the pass) and (on a channel that routes) its downstreams have
// room: its header word and route word read, then handled. Returns whether it handled one.
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE bool receive(
    Cursor& c, ChannelTable<CH>& t, ChannelDownstreams& downstreams, uint32_t arrived, const Coordinates& me) {
    using C = Channel<CH>;
    Position& p = c.position;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (p.available == 0) {
        if constexpr (C::acks) {
            return false;  // nothing acked to process
        } else {
            p.available = arrived;  // the register's count, read in the pass's group
            if (p.available == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(p.slot);
    const uint32_t word = HeaderWord::bits(header);
    uint32_t route = 0;
    if constexpr (C::routes) {
        route = C::Routing::read(header, me);
    }
    return handle<CH>(c, t, downstreams, word, route);
}

// One step of receiver channel CH (`arrived_raw`: its packets-received register, read at the start of the pass): ack
// what has arrived (a channel with first-level acks), at most one packet (timed for code profiling, as main
// times every receiver channel), then its completions. `frame_due` is set when the credit counts changed.
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE void step(
    Cursor& c,
    ChannelTable<CH>& t,
    ChannelDownstreams& downstreams,
    Credits& credits,
    bool& frame_due,
    uint32_t arrived_raw,
    const Coordinates& me) {
    using C = Channel<CH>;
    const uint32_t arrived = arrived_raw & stream_count_mask;
    if constexpr (C::acks) {
        frame_due |= acknowledge_arrivals<CH>(c.position, credits, arrived);
    }
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    timer.set_should_dump(c.position.available != 0);
    timer.open();
    const bool progress = receive<CH>(c, t, downstreams, arrived, me);
    timer.close();
    frame_due |= release_completions<C::completion_group, C::num_slots, typename C::Ids>(
        c.counts, t.sources, c.position.index, progress, credits);
}

// Whether channel CH runs here with first-level acks.
template <size_t CH>
constexpr bool acking() {
    if constexpr (runs_channel<CH>()) {
        return Channel<CH>::acks;
    } else {
        return false;
    }
}
constexpr bool any_channel_acks = acking<0>() || acking<1>() || acking<2>();

// Channel CH's packets-received register, raw, for the pass's register group; none (0) where the channel does not run.
template <size_t CH>
FORCE_INLINE uint32_t arrival_register() {
    if constexpr (runs_channel<CH>()) {
        return read_stream_raw<Channel<CH>::packets_received_stream>();
    } else {
        return 0;
    }
}

// Sends the credit frame, all the counts, unless the TX queue is busy (the next pass sends it then). Returns whether it
// was sent. `group_txq`: the TX queue's status, read in the pass's register group on routers with first-level acks
// (unused elsewhere); only this ERISC sends on the queue, so a queue seen free there is still free.
FORCE_INLINE bool send_credit_frame(Credits& credits, uint32_t group_txq) {
    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        }
    } else {
        const bool seen_free = any_channel_acks && !txq_busy(group_txq);
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
// named only in template-dependent code (Present). `d2`: VC2's downstreams where VC2 forwards (line mode), else none.
template <typename Cursor0, typename TablesT, typename Downstreams1, typename... Downstreams2>
FORCE_INLINE void pass(
    Cursor0& c0,
    bool& frame_due,
    TablesT& tables,
    Credits& credits,
    Downstreams<0>& d0,
    Downstreams1& d1,
    const Coordinates& me,
    Downstreams2&... d2) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t arrived0 = arrival_register<0>();
    const uint32_t arrived1 = arrival_register<1>();
    const uint32_t arrived2 = arrival_register<2>();
    uint32_t txq = 0;
    if constexpr (any_channel_acks) {
        txq = read_txq_status(receiver_txq_id);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    NoDownstreams none;
    if constexpr (runs_channel<0>()) {
        step<0>(c0, tables.ch0, downstreams_for<0>(d0, d1, none, d2...), credits, frame_due, arrived0, me);
    }
    if constexpr (runs_channel<1>()) {
        step<1>(
            tables.ch1.cursor, tables.ch1.table, downstreams_for<1>(d0, d1, none, d2...), credits, frame_due, arrived1, me);
    }
    if constexpr (runs_channel<2>()) {
        step<2>(
            tables.ch2.cursor, tables.ch2.table, downstreams_for<2>(d0, d1, none, d2...), credits, frame_due, arrived2, me);
    }
    if (frame_due) {
        frame_due = !send_credit_frame(credits, txq);
    }
}

// The receiver loop: rounds of passes over the serviced receiver channels until termination. A function of its own:
// its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
//
// The downstream connections are passed in by VC (d0: VC0's; d1: VC1's, an unused stand-in without VC1; d2: VC2's, only
// where VC2 forwards), and each channel forwards on its downstream VC's (downstreams_for): forwards use them, but
// packets delivered here do not, so they live in a table in memory rather than in the registers every packet needs
// (RESULTS §98, v4 bring-up).
template <typename Downstreams1, typename... Downstreams2>
__attribute__((noinline)) void run(Downstreams<0>& d0, Downstreams1& d1, Downstreams2&... d2) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    CursorOf<0> cursor0 = start<0>();
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
            pass(cursor0, frame_due, tables, credits, d0, d1, me, d2...);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end<false, true>([] { return false; }, [] { return vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
