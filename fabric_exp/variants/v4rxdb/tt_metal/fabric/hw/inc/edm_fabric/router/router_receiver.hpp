// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the far router's sender channels, and for each one
// forwards it to the next router on this chip, writes it to this chip, or both, then returns the slot's credit.
//
// Each pass, one step per receiver channel this router runs, in channel order (VC0's, then VC1's or VC2's; step):
//   ack       channels with first-level acks (VC0 on rings and tori): ack the oldest packet not yet acked, one per pass;
//             its header word and route word are read there, once, and kept until it is handled (take_in)
//   poll      packets known present: the packets-received register is read only when the count from its last read is
//             used up (the register stays the source of truth; the kept count is a lower bound of it); under
//             first-level acks, the packets acked and not yet processed
//   decode    the route word: deliver here, and which downstream routers to forward to (router_routing.hpp; with the
//             landing and exit steps of multi-mesh fabrics)
//   admit     every downstream the packet goes to has a free slot (router_downstream.hpp)
//   forward   one copy per downstream
//   deliver   the packet's NoC commands on this chip (router_local_delivery.hpp); in UDM mode, a copy to the fabric's
//             relay instead, which does them (router_downstream.hpp; admitted with the forwards)
//   retire    the slot's record (its sender channel), packets-received register -1, next slot
//   release   completion credits back to the far sender (router_completions.hpp)
// A channel that only delivers (its VC's forwarding is off: VC2's, and VC0's on a speedy-path router) has nothing to
// decode or admit (Channel::routes). Every feature comes from the channel's traits (Channel); no step names a channel.
// Around the steps, once per pass:
//   registers where a channel has first-level acks: the TX queue's status and those channels' packets-received
//             registers, as one group, reduced to one word (`ready`: which channels have a packet to ack, and whether
//             the queue was free)
//   frame     one credit frame, when the pass's acks or completions changed the counts
//
// State, by placement class, one rule for every channel (V4_RECEIVER_SHAPE.md):
//   - left to the compiler (registers where they fit): each channel's cursor (its position in the ring and its
//     completion counts, per packet), whether a credit frame is due, and `ready`, each a local of its own (inside a
//     larger aggregate GCC would keep parts of them in memory);
//   - in memory (`Tables`): every channel's slot records (indexed by ring slot at run time) and the credit counts
//     (indexed by sender channel); the downstream connections (DownstreamSet, kernel_main's, taken by reference);
//   - compile-time: ring bounds, stream ids, command buffers and the credit block's addresses.

#include "router_common.hpp"
#include "router_completions.hpp"
#include "router_downstream.hpp"
#include "router_init.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::receiver {

// The VCs whose receiver channel can run here: VC0's on every router; VC1's on routers that service VC1 traffic (as
// main; an intermesh router without VC1 pass-through has the channel, but no VC1 traffic comes to it); VC2's on routers
// with VC2.
constexpr std::array<bool, MAX_NUM_VCS> vc_serviced = {
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

// Whether this router runs receiver channel `ch`: it has the channel, the channel's VC is serviced here, and channel
// trimming left it.
constexpr bool serviced_channel(size_t ch) {
    return ch < NUM_RECEIVER_CHANNELS && is_receiver_channel_serviced[ch] && vc_serviced[vc_of_receiver_channel(ch)];
}

// Receiver channel CH on a router that does not run it: nothing of it exists here.
template <size_t CH, bool SERVICED = serviced_channel(CH)>
struct Channel {
    static constexpr bool serviced = false;
    static constexpr bool first_level_ack = false;
};

// Receiver channel CH on a router that runs it: its VC, its ring and packets-received register, and what it does with
// its packets.
template <size_t CH>
struct Channel<CH, true> {
    static constexpr bool serviced = true;
    // VC0's channel is 0; VC2's is VC2_RECEIVER_CHANNEL (1 without VC1, 2 with); VC1's is 1.
    static constexpr size_t vc = vc_of_receiver_channel(CH);
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[CH];
    // Whether it routes (decode, admit, forward and/or deliver) or only delivers. The host turns a VC's forwarding off
    // (DISABLE_RX_CH<vc>_FORWARDING) where its channel delivers every packet: VC2's, and VC0's on a speedy-path router
    // (the host's ENABLE_SPEEDY_VC0: a router whose VC0 carries only its worker's packets to the next chip, e.g.
    // neighbour exchange), as main's speedy receiver. It also turns it off where a channel-trimming capture saw nothing
    // forwarded or written on the VC; such a channel also saw no packet, so channel trimming does not service it (the
    // builder rejects a capture that says otherwise, TT_FATAL). So a channel that runs with its forwarding off delivers
    // every packet here.
    static constexpr bool routes = !is_vc_receiver_forwarding_disabled[vc];
    // First-level acks (VC0 on rings and tori): the far sender returns a packet's slot to its producer when the packet
    // has arrived here, on an ack, rather than when it is completed (take_in). The host's setting per VC; VC2 has none.
    static constexpr bool first_level_ack =
        vc == 0 ? ENABLE_FIRST_LEVEL_ACK_VC0 : (vc == 1 ? ENABLE_FIRST_LEVEL_ACK_VC1 : false);
    // The VC of the downstream routers it forwards to: its own, or VC1 on an intermesh router with crossover.
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    static constexpr size_t downstream_vc = 1;
#else
    static constexpr size_t downstream_vc = vc;
#endif
    // UDM: its packets for this chip go to the fabric's relay, one more downstream of VC0 (router_downstream.hpp).
    static constexpr bool via_relay = udm_mode && downstream_vc == 0;
    // How it decides a packet's route (router_routing.hpp).
#if defined(FABRIC_2D)
    using Routing =
        std::conditional_t<routes, Routing2D<downstream_vc, receiver_channel_is_intermesh_ingress[CH]>, NoRouting>;
#else
    using Routing = std::conditional_t<routes, Routing1D, NoRouting>;
#endif
    // Slots per completion group (router_completions.hpp). Under first-level acks one at a time, as the generic loop
    // does: grouped there, completion credits were held, which stalls bubble flow control (RESULTS §98, v4f2, v4p).
    static constexpr uint32_t completion_group = first_level_ack ? 1 : completion_group_size<num_slots>;
    // The transaction IDs of the slots' NoC commands: one per completion group, from the channel's own range.
    using Ids = GroupIds<completion_group, RX_CH_TRID_STARTS[CH], NUM_TRANSACTION_IDS>;
    // Credits go back as counts in one frame (router_completions.hpp); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(CH));
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

// Where receiver channel CH is in its ring, and its completion counts. A member exists only with its feature.
template <size_t CH>
struct Cursor {
    using C = Channel<CH>;
    uint32_t available = 0;         // packets it may process (present, and acked under first-level acks), not retired
    uint32_t slot = C::first_slot;  // header address of the next slot
    uint32_t index = 0;             // ring index of the next slot
    uint32_t uncompleted = 0;       // slots retired (NoC commands issued) but not yet credited
    // First-level acks: header address of the oldest packet not yet acked (`available` slots past `slot`).
    [[no_unique_address]] Present<C::first_level_ack, uint32_t, 0> ack_slot =
        present<C::first_level_ack, 0>(C::first_slot);
    // Completion groups of more than one slot: passes without a packet while a partial group waits.
    [[no_unique_address]] Present<(C::completion_group > 1), uint32_t, 1> idle_passes =
        present<(C::completion_group > 1), 1>(uint32_t{0});
};

// A packet's header word and route word as read at its ack (first-level acks), kept until it is handled.
struct KeptHeader {
    uint32_t word;   // the header word as it is in the header (HeaderWord::bits)
    uint32_t route;  // the route word (router_routing.hpp)
};

// Receiver channel CH's slot records: each slot's sender channel (for its completion credit), and under first-level
// acks each acked packet's kept header (so its header is read from L1 once).
template <size_t CH>
struct Records {
    SlotRecords<Channel<CH>::num_slots> source;
    [[no_unique_address]] Present<Channel<CH>::first_level_ack, std::array<KeptHeader, Channel<CH>::num_slots>, 0> kept;
};

// A channel's state exists only where this router runs the channel (Present, router_common.hpp).
template <size_t CH>
using CursorOf = Present<Channel<CH>::serviced, Cursor<CH>, CH>;
template <size_t CH>
using RecordsOf = Present<Channel<CH>::serviced, Records<CH>, CH>;

// The receiver's state in memory (see the top of this file).
struct Tables {
    [[no_unique_address]] RecordsOf<0> records0;
    [[no_unique_address]] RecordsOf<1> records1;
    [[no_unique_address]] RecordsOf<2> records2;
    CreditCounts credits;

    template <size_t CH>
    FORCE_INLINE auto& records() {
        if constexpr (CH == 0) {
            return records0;
        } else if constexpr (CH == 1) {
            return records1;
        } else {
            return records2;
        }
    }
};

// The downstream connections the receiver forwards on: VC0's, and VC1's on VC1 routers. kernel_main owns them: only
// forwards use them, so they live in memory rather than in the registers every packet needs (RESULTS §98, v4 bring-up).
// Taken by reference, they stay in memory.
struct DownstreamSet {
    Downstreams<0> vc0;
    [[no_unique_address]] Present<has_vc1, Downstreams<vc1_links_vc>, 1> vc1;

    template <size_t VC>
    FORCE_INLINE auto& of() {
        if constexpr (VC == 0) {
            return vc0;
        } else {
            return vc1;
        }
    }
};

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
    uint32_t route,
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
    Channel<CH>::Routing::before_forward(header, route);
    downstreams.forward_all(to, slot, size, trid);
    record_forwarded<CH>(to);
}

// The header address of the slot after `slot` in channel CH's ring.
template <size_t CH>
FORCE_INLINE uint32_t next_slot(uint32_t slot) {
    return slot == Channel<CH>::last_slot ? Channel<CH>::first_slot : slot + channel_buffer_size;
}

// First-level acks: a packet that has arrived is acked to the far sender before it is processed, and the far sender
// returns its slot to the producer on the ack. Acks the oldest packet not yet acked (one per pass: the rest stay counted
// in the packets-received register for the next passes, which this ack consumes by one rather than retire): its sender
// channel's ack count goes up (the frame goes out with the pass's other credits). Its header word and route word are
// read here, once, and kept in its slot's record until it is handled; when it is the next packet to handle, they
// replace the step's `word` and `route`.
template <size_t CH>
FORCE_INLINE void take_in(
    CursorOf<CH>& c, Records<CH>& r, CreditCounts& credits, Coordinates me, uint32_t& word, uint32_t& route) {
    using C = Channel<CH>;
    ASSERT(c.ack_slot == C::first_slot + ring_add<C::num_slots>(c.index, c.available) * channel_buffer_size);
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(c.ack_slot);
    const uint32_t acked_word = HeaderWord::bits(header);
    const uint32_t acked_route = C::Routing::read(header, me);
    c.ack_slot = next_slot<CH>(c.ack_slot);  // advanced while they load
    r.kept[ring_add<C::num_slots>(c.index, c.available)] = {acked_word, acked_route};
    increment_local_update_ptr_val<C::packets_received_stream>(-1);
    credits.add_ack(source_channel(HeaderWord::of(acked_word).src_ch_id));
    if (c.available++ == 0) {
        word = acked_word;
        route = acked_route;
    }
}

// The packet in slot `slot`, whose route (`to`, from its route word `route`) is admitted: forwarded, delivered here, or
// both, its NoC commands tagged with `trid`.
template <size_t CH, typename Route>
FORCE_INLINE void act(
    tt_l1_ptr PACKET_HEADER_TYPE* header,
    HeaderWord f,
    uint32_t route,
    Route to,
    DownstreamSet& ds,
    uint8_t trid,
    uint32_t slot) {
    using C = Channel<CH>;
    channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
    const uint32_t size = f.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
    if constexpr (telemetry::counts_received) {
        telemetry::count_received(size);
    }
    if constexpr (C::routes) {
        auto& downstreams = ds.of<C::downstream_vc>();
        if (!is_2d && to.delivers() && to.forwards() &&
            f.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
            sparse_multicast_hop<CH>(header, f.payload_size_bytes, route, to.forward_to(), downstreams, slot, size, trid);
            return;
        }
        // Forward first (the next router starts sooner), then write here.
        if (to.forwards()) {
            C::Routing::before_forward(header, route);
            downstreams.forward_all(to.forward_to(), slot, size, trid);
            record_forwarded<CH>(to.forward_to());
        }
    }
    if (to.delivers()) {
        if constexpr (C::via_relay) {
            ds.of<0>().forward_to_relay(slot, size, trid);
        } else {
            deliver<CH>(header, f.payload_size_bytes, f.noc_send_type, trid);
        }
    }
}

// The slot just handled: its record (`src`: its packet's sender channel), the packets-received register -1 (under
// first-level acks consumed at the ack), the next slot.
template <size_t CH>
FORCE_INLINE void retire(CursorOf<CH>& c, Records<CH>& r, uint8_t src) {
    using C = Channel<CH>;
    r.source.set(c.index, source_channel(src));
    c.uncompleted++;
    c.available--;
    if constexpr (!C::first_level_ack) {
        increment_local_update_ptr_val<C::packets_received_stream>(-1);
    }
    c.index = c.index + 1 == C::num_slots ? 0 : c.index + 1;
    c.slot = c.index == 0 ? C::first_slot : c.slot + channel_buffer_size;
}

// One packet, when one is present and its downstreams (forwards, the UDM relay) have room. Returns whether it handled
// one. Under first-level acks `word` and `route` are the next packet's, kept at its ack; otherwise they are read here.
template <size_t CH>
FORCE_INLINE bool receive(
    CursorOf<CH>& c, Records<CH>& r, DownstreamSet& ds, Coordinates me, uint32_t word, uint32_t route) {
    using C = Channel<CH>;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (c.available == 0) {
        if constexpr (C::first_level_ack) {
            return false;  // nothing acked to process
        } else {
            c.available = read_stream_count<C::packets_received_stream>();
            if (c.available == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(c.slot);
    if constexpr (!C::first_level_ack) {
        word = HeaderWord::bits(header);
        route = C::Routing::read(header, me);
    }
    const HeaderWord f = HeaderWord::of(word);
    const auto to = C::Routing::decide(route, me);
    if constexpr (C::routes) {
        if (to.forwards() && !ds.of<C::downstream_vc>().admit(to.forward_to())) {
            return false;
        }
    }
    if constexpr (C::via_relay) {
        if (to.delivers() && !ds.of<0>().admit_relay()) {
            return false;
        }
    }
    act<CH>(header, f, route, to, ds, C::Ids::of(c.index), c.slot);
    retire<CH>(c, r, f.src_ch_id);
    return true;
}

// Credits channel CH's oldest uncredited slots when they are due and their writes have left (router_completions.hpp);
// returns whether it did. `progress`: this pass handled a packet.
template <size_t CH>
FORCE_INLINE bool release_completions(CursorOf<CH>& c, const Records<CH>& r, CreditCounts& credits, bool progress) {
    using C = Channel<CH>;
    constexpr uint32_t N = C::num_slots;
    constexpr uint32_t T = C::completion_group;
    static_assert(T >= 1 && T <= 4 && T <= N);
    if constexpr (T == 1) {
        // Groups of one slot: the oldest uncredited, due when a slot other than the one this pass handled waits; its
        // proof read once.
        if (c.uncompleted <= static_cast<uint32_t>(progress)) {
            return false;
        }
        const uint32_t oldest = ring_add<N>(c.index, N - c.uncompleted);
        if (!writes_left(C::Ids::of(oldest))) {
            return false;
        }
        credits.add_completions(r.source.of(oldest), 1);
        c.uncompleted--;
        return true;
    } else {
        // Due.
        const uint32_t waiting = c.uncompleted;
        bool due;
        if (progress) {
            c.idle_passes = 0;
            due = waiting > T;
        } else {
            due = waiting >= T || (waiting != 0 && ++c.idle_passes >= release_after_idle_passes);
        }
        if (!due) {
            return false;
        }
        // The due slots: the oldest uncredited one and the rest of its group that has been retired.
        const uint32_t first = ring_add<N>(c.index, N - waiting);
        const uint32_t group_end = (first | (T - 1)) + 1;
        const uint32_t to_end = (group_end < N ? group_end : N) - first;
        const uint32_t n = waiting < to_end ? waiting : to_end;
        // Proof.
        if (!writes_leave<completion_wait_checks>(C::Ids::of(first))) {
            return false;
        }
        // Credit.
        c.idle_passes = 0;
        credit_slots(r.source, first, n, credits);
        c.uncompleted = waiting - n;
        return true;
    }
}

// The pass's register reads, as one word (`ready`): bit CH set when channel CH (first-level acks) has a packet not yet
// acked, and `txq_free` when the TX queue was free.
constexpr uint32_t txq_free = 1u << MAX_NUM_RECEIVER_CHANNELS;

// Channel CH's packets-received register, raw (masked where it is used), into registers[CH], where CH acks.
template <size_t CH>
FORCE_INLINE void read_received(uint32_t* registers) {
    if constexpr (Channel<CH>::first_level_ack) {
        registers[CH] = read_stream_raw<Channel<CH>::packets_received_stream>();
    }
}

// Channel CH's bit of `ready`, from its register in registers[CH] (0 where CH does not ack).
template <size_t CH>
FORCE_INLINE uint32_t arrival_bit(const uint32_t* registers) {
    if constexpr (Channel<CH>::first_level_ack) {
        return (registers[CH] & stream_count_mask) != 0 ? 1u << CH : 0u;
    } else {
        return 0;
    }
}

// The pass's registers where a channel has first-level acks (a frame then goes out most passes): the TX queue's status
// first (for the frame), then those channels' packets-received registers, read as one group (volatile loads kept
// together between the two fences, so their latencies overlap; as the sender's channels_with_packets), then reduced to
// `ready`, the only value kept across the pass. 0 elsewhere (the frame reads the queue's status itself).
template <size_t... CH>
FORCE_INLINE uint32_t read_pass_registers() {
    if constexpr ((Channel<CH>::first_level_ack || ...)) {
        uint32_t registers[sizeof...(CH)];
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const uint32_t txq_status = read_txq_status(receiver_txq_id);
        (read_received<CH>(registers), ...);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return (arrival_bit<CH>(registers) | ...) | (txq_busy(txq_status) ? 0u : txq_free);
    } else {
        return 0;
    }
}

// One step of receiver channel CH, where this router runs it: ack the oldest packet not yet acked (first-level acks,
// when `ready` says one has arrived), at most one packet (timed for code profiling, as main times every receiver
// channel), then its completions. `frame_due` is set when the credit counts changed.
template <size_t CH>
FORCE_INLINE void step(CursorOf<CH>& c, Tables& t, DownstreamSet& ds, Coordinates me, uint32_t ready, bool& frame_due) {
    using C = Channel<CH>;
    if constexpr (C::serviced) {
        auto& r = t.records<CH>();
        uint32_t word = 0;
        uint32_t route = 0;
        if constexpr (C::first_level_ack) {
            // The next packet's header word and route word, kept at its ack (local memory), loaded first.
            word = r.kept[c.index].word;
            route = r.kept[c.index].route;
            if ((ready & (1u << CH)) != 0) {
                take_in<CH>(c, r, t.credits, me, word, route);
                frame_due = true;
            }
        }
        NamedProfiler<
            CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
            code_profiling_enabled_timers_bitfield,
            code_profiling_buffer_base_addr>
            timer;
        timer.set_should_dump(c.available != 0);
        timer.open();
        const bool progress = receive<CH>(c, r, ds, me, word, route);
        timer.close();
        if (release_completions<CH>(c, r, t.credits, progress)) {
            frame_due = true;
        }
    }
}

// Whether receiver channel 0's packets-received register reads 0 (telemetry's RX heartbeat, once per round, as main).
FORCE_INLINE bool channel0_register_empty() { return read_stream_count<to_receiver_packets_sent_streams[0]>() == 0; }

// One pass over the receiver channels (CH: 0, 1, 2; `c`: their cursors): the pass's registers, one step per channel,
// and the credit frame when the counts changed. The body of run()'s loop; a single-ERISC router runs it as the
// receiver half of each pass over its state in memory (router_single.hpp).
template <size_t... CH>
FORCE_INLINE void pass(
    std::index_sequence<CH...>, Tables& t, DownstreamSet& ds, Coordinates me, bool& frame_due, CursorOf<CH>&... c) {
    const uint32_t ready = read_pass_registers<CH...>();
    (step<CH>(c, t, ds, me, ready, frame_due), ...);
    if (frame_due) {
        frame_due = !try_send_frame((ready & txq_free) != 0);
    }
}

// Receiver channels 0, 1, 2: every channel a router can have (pass).
constexpr auto every_channel = std::make_index_sequence<MAX_NUM_RECEIVER_CHANNELS>{};

// The receiver loop: rounds of passes over the receiver channels until termination. A function of its own: its code
// depends only on this body, and once-per-round work is out of line (router_round.hpp).
__attribute__((noinline)) void run(DownstreamSet& ds) {
    const Coordinates me = coordinates();
    CursorOf<0> c0{};
    CursorOf<1> c1{};
    CursorOf<2> c2{};
    Tables t{};
    bool frame_due = false;  // the credit counts changed since the last frame
    local_write_init();

    RoundCounters round;
    while (next_round()) {
        if constexpr (telemetry::any_option) {
            telemetry::round_start();
        }
        for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
            pass(every_channel, t, ds, me, frame_due, c0, c1, c2);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end<false, true>([] { return false; }, [] { return channel0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
