// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the far router's sender channels, and for each one
// forwards it to the next router on this chip, writes it to this chip, or both, then returns the slot's credit.
//
// Each pass, one step per serviced receiver channel (VC0's, VC1's, VC2's, in that order; channel_step):
//   ack       channels with first-level acks (VC0 on rings and tori): ack the oldest packet not yet acked, one per
//             pass, its header read in one group with the handled packet's (acking_step)
//   poll      packets known present: the packets-received register is read only when the count from its last read is
//             used up (the register stays the source of truth; the kept count is a lower bound of it); under
//             first-level acks, the packets acked and not yet processed
//   decode    the route: deliver here, and which downstream routers to forward to (router_routing.hpp; with the
//             landing and exit steps of multi-mesh fabrics)
//   admit     every downstream the packet goes to has a free slot (router_downstream.hpp)
//   forward   one copy per downstream
//   deliver   the packet's NoC commands on this chip (router_local_delivery.hpp); in UDM mode, a copy to the fabric's
//             relay instead, which does them (router_downstream.hpp; admitted with the forwards)
//   retire    the slot's record (its sender channel), packets-received register -1, next slot
//   release   completion credits back to the far sender (router_completions.hpp); with first-level acks at the
//             pass's end, one slot at a time (release_and_frame)
// A channel that only delivers (VC2's, and VC0's on a speedy-path router) skips decode and admit (Channel::kind).
// Around the steps, once per pass:
//   registers the packets-received counts of the channels with first-level acks, as one group, before the steps
//   frame     one credit frame after the steps, when the pass's acks or completions changed the counts (with
//             first-level acks, the TX queue's status read in one group with the completion's proof)
//
// State, by placement class (V4_RECEIVER_SHAPE.md):
//   - left to the compiler (registers where they fit): VC0's cursor (its position in the ring and its completion counts,
//     per packet) and whether a credit frame is due, each a loop local of its own (inside a larger aggregate GCC would
//     keep parts of them in memory);
//   - in memory (`Tables`): VC0's slot records (indexed by ring slot at run time), VC1's and VC2's channel state (one
//     record each, cursor and records together: E1's loop is at its register limit with VC0's cursor, and VC1's as more
//     loop-carried values cost VC0 +8 cycles/packet, RESULTS §98 v4p);
//   - the credit sender, an object of its own: its counts are in memory (indexed by sender channel), its block
//     addresses are constants (inside an aggregate with the tables they would be loaded from memory);
//   - compile-time: ring bounds, stream ids and command buffers.
// The downstream connections are passed in (router_downstream.hpp).

#include "router_common.hpp"
#include "router_completions.hpp"
#include "router_downstream.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::receiver {

// Whether this router runs its VC1 receiver channel: on routers that service VC1 traffic (as main; an intermesh router
// without VC1 pass-through has the channel, but no VC1 traffic comes to it), unless channel trimming took it out.
constexpr bool runs_vc1_receiver =
#if defined(FABRIC_2D_VC1_SERVICED)
    is_receiver_channel_serviced[1];
#else
    false;
#endif

// Whether this router runs a VC2 receiver channel (VC2_RECEIVER_CHANNEL: 1 without VC1, 2 with), unless channel
// trimming took it out.
constexpr bool runs_vc2_receiver =
#if defined(FABRIC_2D_VC2_SERVICED)
    is_receiver_channel_serviced[VC2_RECEIVER_CHANNEL];
#else
    false;
#endif

// Whether this router runs receiver channel 0 (VC0's): channel trimming takes it out when the capture saw no packet on it.
constexpr bool runs_vc0_receiver = is_receiver_channel_serviced[0];

// Whether this router runs receiver channel CH.
template <size_t CH>
constexpr bool runs_channel() {
    return (CH == 0 && runs_vc0_receiver) || (CH == 1 && runs_vc1_receiver) ||
           (CH == VC2_RECEIVER_CHANNEL && runs_vc2_receiver);
}

// What a receiver channel does with its packets.
enum class Kind {
    routes,    // decode, admit, forward and/or deliver
    delivers,  // every packet is for this chip: no route to decode, nothing to admit
};

// Receiver channel CH: its ring, its packets-received register, and the VC of the downstream routers it forwards to (VC1
// for VC0 on an intermesh router with crossover).
template <size_t CH>
struct Channel {
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    static constexpr size_t downstream_vc = 1;
#else
    static constexpr size_t downstream_vc = CH;
#endif
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[CH];
    // VC2's channel (neighbour exchange).
    static constexpr bool is_vc2 = ACTUAL_VC2_SENDER_CHANNELS > 0 && CH == VC2_RECEIVER_CHANNEL;
    // First-level acks (VC0 on rings and tori): the far sender returns a packet's slot to its producer when the packet
    // has arrived here, on an ack, rather than when it is completed (acknowledge_one). The host's setting per VC.
    static constexpr bool first_level_ack =
        !is_vc2 && (CH == 0 ? ENABLE_FIRST_LEVEL_ACK_VC0 : ENABLE_FIRST_LEVEL_ACK_VC1);
    // Slots per completion group (router_completions.hpp). Under first-level acks one at a time, as the generic loop
    // does: grouped there, completion credits were held, which stalls bubble flow control (RESULTS §98, v4f2, v4p).
    static constexpr uint32_t completion_group = first_level_ack ? 1 : completion_group_size<num_slots>;
    // The transaction IDs of the slots' NoC commands: one per completion group, from the channel's own range.
    using Ids = GroupIds<completion_group, RX_CH_TRID_STARTS[CH], NUM_TRANSACTION_IDS>;
    // Credits go back as counts in one frame (router_completions.hpp); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(CH));
    // The kind. VC2's channel delivers every packet; so does VC0's on a speedy-path router (the host's
    // ENABLE_SPEEDY_VC0: a router whose VC0 carries only its worker's packets to the next chip, e.g. neighbour
    // exchange), as main's speedy receiver. Every other channel routes.
    static constexpr Kind kind = is_vc2 || (CH == 0 && super_speedy_mode) ? Kind::delivers : Kind::routes;
    // The host turns a channel's forwarding off (DISABLE_RX_CHx_FORWARDING) where the channel delivers every packet,
    // and where a channel-trimming capture saw nothing forwarded or written on its VC; such a channel also saw no packet,
    // so channel trimming does not service it. A serviced channel that routes with forwarding off would be a
    // contradictory capture, which the builder rejects (TT_FATAL); the router does not handle it.
    static_assert(
        !is_receiver_channel_serviced[CH] || kind == Kind::delivers || !is_receiver_channel_forwarding_disabled[CH],
        "a serviced receiver channel that routes has forwarding off: an inconsistent channel-trimming capture");
#if defined(FABRIC_2D)
    using Routing = Routing2D<downstream_vc, receiver_channel_is_intermesh_ingress[CH]>;
#else
    using Routing = Routing1D;
#endif
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

template <size_t CH>
constexpr Cursor cursor_start() {
    return {Position{0, Channel<CH>::first_slot, 0}, CompletionCounts{}};
}

// A channel's cursor and slot records kept together in one record (VC1's, VC2's).
template <size_t CH>
struct ChannelState {
    Cursor cursor = cursor_start<CH>();
    SlotRecords<Channel<CH>::num_slots> records{};
};

// The receiver's state by placement class (see the top of this file). A member exists only where this router runs its
// channel (Present, router_common.hpp).
using Vc0Cursor = Present<runs_vc0_receiver, Cursor, 0>;

struct Tables {
    [[no_unique_address]] Present<runs_vc0_receiver, SlotRecords<Channel<0>::num_slots>, 0> records0{};
    [[no_unique_address]] Present<runs_vc1_receiver, ChannelState<1>, 1> vc1;
    [[no_unique_address]] Present<runs_vc2_receiver, ChannelState<VC2_RECEIVER_CHANNEL>, 2> vc2;
};

// One credit sender for every receiver channel: counter credits are one L1 block indexed by the far sender channel, and
// the channels' far sender channels are different channels, so one frame per pass carries every channel's credits.
using Credits = ReceiverChannelCounterBasedResponseCreditSender;

template <size_t CH>
FORCE_INLINE void retire(Position& p) {
    using C = Channel<CH>;
    p.available--;
    if constexpr (!C::first_level_ack) {
        increment_local_update_ptr_val<C::packets_received_stream>(-1);  // under acks, consumed at the ack
    }
    p.index = p.index + 1 == C::num_slots ? 0 : p.index + 1;
    p.slot = p.index == 0 ? C::first_slot : p.slot + channel_buffer_size;
}

// Channel-trimming capture: the sender channels, on the downstream routers, a packet forwarded on receiver channel CH went
// to (`to`: the routing names of its downstreams), as main records them: in 2D each downstream's channel from the
// host's packed table (by the downstream's compact index), in 1D the downstream's forwarding channel, 1.
template <size_t CH>
FORCE_INLINE void record_forwarded(uint32_t to) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        if constexpr (is_2d) {
            constexpr uint32_t packed = Channel<CH>::downstream_vc == 0 ? PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS
                                                                       : PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS;
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
    const typename Channel<CH>::Routing::Fields fields,
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
    Channel<CH>::Routing::before_forward(header, fields);
    downstreams.forward_all(to, slot, size, trid);
    record_forwarded<CH>(to);
}

// First-level acks. A packet that has arrived is acked to the far sender before it is processed, and the far sender
// returns its slot to the producer on the ack. The packets acked are the oldest not yet acked: `available` slots past
// the next one to process; the packets-received register is consumed at the ack rather than at retire.
// One ack per pass (acking_step): the oldest packet not yet acked, if one has arrived (`arrived`: the packets-received
// count; `src`: that packet's sender channel, read with the pass's header loads). Its sender channel's ack count goes
// up, the register is consumed by one, and the packet may be handled; the rest stay counted for the next passes.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool acknowledge_one(Position& p, CreditSender& credits, uint32_t arrived, uint8_t src) {
    using C = Channel<CH>;
    if (arrived == 0) {
        return false;
    }
    increment_local_update_ptr_val<C::packets_received_stream>(-1);
    add_ack(credits, source_channel(src));
    p.available++;
    return true;
}

// One packet of a channel that only delivers: no route to decode, nothing to admit. Returns whether it handled one.
template <size_t CH>
FORCE_INLINE bool receive_local(Position& p, CompletionCounts& counts, SlotRecords<Channel<CH>::num_slots>& records) {
    using C = Channel<CH>;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (p.available == 0) {
        p.available = read_stream_count<C::packets_received_stream>();
        if (p.available == 0) {
            return false;
        }
    }
    auto* header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(p.slot);
    const HeaderWord word = HeaderWord::load(header);
    channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
    if constexpr (telemetry::counts_received) {
        telemetry::count_received(word.payload_size_bytes + sizeof(PACKET_HEADER_TYPE));
    }
    deliver<CH>(header, word.payload_size_bytes, word.noc_send_type, C::Ids::of(p.index));
    slot_retired(counts, records, p.index, source_channel(word.src_ch_id));
    retire<CH>(p);
    return true;
}

// The downstreams of a channel that forwards nothing (one that only delivers).
struct NoDownstreams {};

// The packet in slot `p.slot` of a channel that routes, from its header word and routing fields: decode, admit,
// forward, deliver, retire. Returns whether it handled it (false: a downstream had no room).
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE bool handle(
    Position& p,
    CompletionCounts& counts,
    SlotRecords<Channel<CH>::num_slots>& records,
    ChannelDownstreams& downstreams,
    const Coordinates& me,
    tt_l1_ptr PACKET_HEADER_TYPE* header,
    const HeaderWord& word,
    const typename Channel<CH>::Routing::Fields& fields) {
    using C = Channel<CH>;
    const auto route = C::Routing::decide(fields, me);
    if (route.forwards() && !downstreams.admit(route.forward_to())) {
        return false;
    }
    if constexpr (udm_mode) {
        if (route.delivers() && !downstreams.admit_relay()) {
            return false;
        }
    }

    channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
    const uint8_t trid = C::Ids::of(p.index);
    const uint32_t size = word.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
    if constexpr (telemetry::counts_received) {
        telemetry::count_received(size);
    }
    if (!is_2d && route.delivers() && route.forwards() &&
        word.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
        sparse_multicast_hop<CH>(
            header, word.payload_size_bytes, fields, route.forward_to(), downstreams, p.slot, size, trid);
    } else {
        // Forward first (the next router starts sooner), then write here.
        if (route.forwards()) {
            C::Routing::before_forward(header, fields);
            downstreams.forward_all(route.forward_to(), p.slot, size, trid);
            record_forwarded<CH>(route.forward_to());
        }
        if (route.delivers()) {
            if constexpr (udm_mode) {
                downstreams.forward_to_relay(p.slot, size, trid);
            } else {
                deliver<CH>(header, word.payload_size_bytes, word.noc_send_type, trid);
            }
        }
    }
    slot_retired(counts, records, p.index, source_channel(word.src_ch_id));
    retire<CH>(p);
    return true;
}

// One packet, when one is present and (on a channel that routes) its downstreams have room. Returns whether it handled
// one.
template <size_t CH, typename ChannelDownstreams>
FORCE_INLINE bool receive(
    Position& p,
    CompletionCounts& counts,
    SlotRecords<Channel<CH>::num_slots>& records,
    ChannelDownstreams& downstreams,
    const Coordinates& me) {
    using C = Channel<CH>;
    if constexpr (C::kind == Kind::delivers) {
        return receive_local<CH>(p, counts, records);
    } else {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        if (p.available == 0) {
            if constexpr (C::first_level_ack) {
                return false;  // nothing acked to process
            } else {
                p.available = read_stream_count<C::packets_received_stream>();
                if (p.available == 0) {
                    return false;
                }
            }
        }
        auto* header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(p.slot);
        const HeaderWord word = HeaderWord::load(header);
        const auto fields = C::Routing::read(header, me);
        return handle<CH>(p, counts, records, downstreams, me, header, word, fields);
    }
}

// Whether channel CH acks: it runs here and has first-level acks.
template <size_t CH>
constexpr bool acks() {
    if constexpr (runs_channel<CH>()) {
        return Channel<CH>::first_level_ack;
    } else {
        return false;
    }
}
constexpr bool any_channel_acks = acks<0>() || acks<1>() || acks<2>();
// Only VC0 has first-level acks (the host never sets ENABLE_FIRST_LEVEL_ACK_VC1; VC2's channel never acks), and its
// channel routes (a speedy-path VC0, which only delivers, is on routers without them): acking_step and
// release_and_frame below are VC0's.
constexpr bool vc0_acks_and_routes() {
    if constexpr (acks<0>()) {
        return Channel<0>::kind == Kind::routes;
    } else {
        return true;
    }
}
static_assert(!acks<1>() && !acks<2>() && vc0_acks_and_routes());

// The pass's register reads with first-level acks: the acking channels' packets-received registers, raw (masked where
// they are used), as one group.
struct PassRegisters {
    std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> arrived{};  // packets-received register, raw, per channel that acks
};

template <size_t CH>
FORCE_INLINE void read_arrived(PassRegisters& r) {
    if constexpr (acks<CH>()) {
        r.arrived[CH] = read_stream_raw<Channel<CH>::packets_received_stream>();
    }
}

FORCE_INLINE PassRegisters read_pass_registers() {
    PassRegisters r;
    if constexpr (any_channel_acks) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        read_arrived<0>(r);
        read_arrived<1>(r);
        read_arrived<2>(r);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    return r;
}

// One step of receiver channel CH without first-level acks: at most one packet (timed for code profiling, as main times
// every receiver channel), then its completions. `frame_due` is set when the credit counts changed.
template <size_t CH, typename ChannelDownstreams, typename CreditSender>
FORCE_INLINE void channel_step(
    Cursor& c,
    SlotRecords<Channel<CH>::num_slots>& records,
    ChannelDownstreams& downstreams,
    const Coordinates& me,
    CreditSender& credits,
    bool& frame_due) {
    using C = Channel<CH>;
    static_assert(!C::first_level_ack);
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    timer.set_should_dump(c.position.available != 0);
    timer.open();
    const bool progress = receive<CH>(c.position, c.counts, records, downstreams, me);
    timer.close();
    frame_due |= release_completions<C::completion_group, C::num_slots, typename C::Ids>(
        c.counts, records, c.position.index, progress, credits);
}

// One step of channel CH with first-level acks: one ack and at most one packet (timed for code profiling); its
// completions are released at the pass's end (release_and_frame). The header loads are one group: the sender channel
// of the packet to ack (the oldest not yet acked) and the header word and routing fields of the packet to handle (the
// next in the ring), unconditionally (unused when nothing arrived or nothing is acked to handle), with nothing computed
// on them inside the group (decoded right after it, so no raw value is kept in memory across the ack). A hardware read
// fence first orders them after the pass's read of the packets-received register (`arrived_raw`), so a packet seen
// counted there has its header in L1. Returns whether it handled a packet.
template <size_t CH, typename ChannelDownstreams, typename CreditSender>
FORCE_INLINE bool acking_step(
    Cursor& c,
    SlotRecords<Channel<CH>::num_slots>& records,
    ChannelDownstreams& downstreams,
    const Coordinates& me,
    CreditSender& credits,
    bool& frame_due,
    uint32_t arrived_raw) {
    using C = Channel<CH>;
    Position& p = c.position;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(p.slot);
    const auto* const acked = reinterpret_cast<const volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
        C::first_slot + ring_add<C::num_slots>(p.index, p.available) * channel_buffer_size);
    asm volatile("fence r, r" ::: "memory");
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint8_t acked_src = acked->src_ch_id;
    const uint32_t word = HeaderWord::load_raw(header);
    const auto routing = C::Routing::load_raw(header, me);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const HeaderWord header_word = HeaderWord::from(word);
    const auto fields = C::Routing::from_raw(routing, me);
    frame_due |= acknowledge_one<CH>(p, credits, arrived_raw & stream_count_mask, acked_src);
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    timer.set_should_dump(p.available != 0);
    timer.open();
    const bool progress =
        p.available != 0 && handle<CH>(p, c.counts, records, downstreams, me, header, header_word, fields);
    timer.close();
    return progress;
}

// The pass's end on a router with first-level acks: channel CH's completion (groups of one slot: the oldest uncredited,
// due when a slot other than the one this pass handled waits), then the credit frame. Its proof (the slot's transaction
// ID) and the TX queue's status are read as one group, the status first, as the proof is tested first; the frame stays
// after the pass's work, and no register value is kept across the pass for it. Scalars only around the group: a record
// there (the slot and whether it is due) is kept in memory across the fences.
template <size_t CH, typename CreditSender>
FORCE_INLINE void release_and_frame(
    Cursor& c, const SlotRecords<Channel<CH>::num_slots>& records, bool progress, CreditSender& credits, bool& frame_due) {
    using C = Channel<CH>;
    constexpr uint32_t N = C::num_slots;
    static_assert(C::completion_group == 1);
    const uint32_t waiting = c.counts.uncompleted;
    const uint32_t slot = ring_add<N>(c.position.index, N - waiting);
    const bool due = waiting > static_cast<uint32_t>(progress);  // the slot this pass handled is not due yet
    if (!due && !frame_due) {
        return;
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t txq_status = read_txq_status(receiver_txq_id);
    const bool sent = writes_left(C::Ids::of(slot));
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (due && sent) {
        credit_slots(records, slot, 1, credits);
        c.counts.uncompleted = waiting - 1;
        frame_due = true;
    }
    if (frame_due && (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK || !txq_busy(txq_status))) {
        frame_due = !send_credit_frame(credits, true);
    }
}

// Whether VC0's packets-received register reads 0 (telemetry's RX heartbeat, once per round).
FORCE_INLINE bool vc0_register_empty() { return read_stream_count<Channel<0>::packets_received_stream>() == 0; }

// One pass over the serviced receiver channels: the pass's registers, one step per channel, and the credit frame when
// the counts changed (with first-level acks: VC0's completion and the frame, release_and_frame). The body of run()'s loop, which calls it on its own locals; a single-ERISC router runs it as the
// receiver half of each pass over its state in memory (router_single.hpp). The state types are template parameters so
// that an absent channel's state is named only in template-dependent code (Present).
template <typename Cursor0, typename TablesT, typename Downstreams1>
FORCE_INLINE void pass(
    Cursor0& cursor0,
    bool& frame_due,
    TablesT& tables,
    Credits& credits,
    Downstreams<Channel<0>::downstream_vc>& downstreams0,
    Downstreams1& downstreams1,
    const Coordinates& me) {
    const PassRegisters registers = read_pass_registers();
    bool progress0 = false;
    if constexpr (acks<0>()) {
        progress0 = acking_step<0>(cursor0, tables.records0, downstreams0, me, credits, frame_due, registers.arrived[0]);
    } else if constexpr (runs_vc0_receiver) {
        channel_step<0>(cursor0, tables.records0, downstreams0, me, credits, frame_due);
    }
    if constexpr (runs_vc1_receiver) {
        channel_step<1>(tables.vc1.cursor, tables.vc1.records, downstreams1, me, credits, frame_due);
    }
    if constexpr (runs_vc2_receiver) {
        NoDownstreams none;
        channel_step<VC2_RECEIVER_CHANNEL>(tables.vc2.cursor, tables.vc2.records, none, me, credits, frame_due);
    }
    if constexpr (acks<0>()) {
        release_and_frame<0>(cursor0, tables.records0, progress0, credits, frame_due);
    } else if (frame_due) {
        frame_due = !send_credit_frame(credits, false);
    }
}

// The receiver loop: rounds of passes over the serviced receiver channels until termination. A function of its own:
// its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
//
// The downstream connections are passed in, the ones each receiver channel forwards on (with crossover both are VC1's;
// without VC1, `downstreams1` is unused): forwards use them, but packets delivered here do not, so they live in a table
// in memory rather than in the registers every packet needs (RESULTS §98, v4 bring-up).
template <typename Downstreams1>
__attribute__((noinline)) void run(Downstreams<Channel<0>::downstream_vc>& downstreams0, Downstreams1& downstreams1) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    Vc0Cursor cursor0 = present<runs_vc0_receiver, 0>(cursor_start<0>());
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
            pass(cursor0, frame_due, tables, credits, downstreams0, downstreams1, me);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end<false, true>([] { return false; }, [] { return vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
