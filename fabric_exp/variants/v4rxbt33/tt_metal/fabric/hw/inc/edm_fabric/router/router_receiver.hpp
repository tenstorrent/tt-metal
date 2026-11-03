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
//   release   completion credits back to the far sender (Completions, below)
// A channel that only delivers (VC2's, and VC0's on a speedy-path router) skips decode and admit (Channel::routes).
// Around the steps, once per pass:
//   registers every channel's packets-received register and, where a channel has first-level acks (a frame goes out
//             most passes), the TX queue's status, as one group
//   frame     one credit frame, when the pass's acks or completions changed the counts
//
// Channels and VCs. A receiver channel's features are its VC's (Channel<CH>, the one place a channel's VC is named):
// VC0's channel is 0, VC1's 1, VC2's 2, or 1 on routers without VC1.
//
// State, by placement class (the State section's Placement notes; V4_RECEIVER_SHAPE.md): VC0's cursor and whether a
// credit frame is due are loop locals of their own (registers where they fit); each channel's slot records, and VC1's
// and VC2's cursors, are in memory (`Tables`); the credit sender is an object of its own; ring bounds, stream ids and
// command buffers are compile-time. The downstream connections are passed in, by VC (router_downstream.hpp).

#include "router_common.hpp"
#include "router_downstream.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::receiver {

// ---------------------------------------------------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------------------------------------------------

// Passes without a packet after which a partial group goes out.
constexpr uint32_t release_after_idle_passes = 4;
// Status reads a group's proof waits for, at most; then its credit is left to a later pass.
constexpr uint32_t completion_wait_checks = 32;

// ---------------------------------------------------------------------------------------------------------------------
// Ring arithmetic and completion groups
// ---------------------------------------------------------------------------------------------------------------------

// Ring index `index + n` in a ring of NUM_SLOTS slots (n < NUM_SLOTS).
template <uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t ring_add(uint32_t index, uint32_t n) {
    const uint32_t sum = index + n;
    if constexpr (is_power_of_2(NUM_SLOTS)) {
        return sum & (NUM_SLOTS - 1);
    } else {
        return sum >= NUM_SLOTS ? sum - NUM_SLOTS : sum;
    }
}

// The slots per completion group of a ring of NUM_SLOTS slots: a quarter of the ring, at most 4, a power of 2.
template <uint32_t NUM_SLOTS>
constexpr uint32_t completion_group_size = NUM_SLOTS < 8 ? 1 : (NUM_SLOTS < 16 ? 2 : 4);

// The group of slot `index`.
template <uint32_t GROUP_SIZE>
FORCE_INLINE uint32_t group_of(uint32_t index) {
    return index / GROUP_SIZE;
}

// The index just past the group of slot `index` (its last slot + 1; groups are aligned).
template <uint32_t GROUP_SIZE>
FORCE_INLINE uint32_t group_end(uint32_t index) {
    return (index | (GROUP_SIZE - 1)) + 1;
}

// The oldest uncredited slot: `waiting` slots before the next one to retire.
template <uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t oldest_uncredited(uint32_t next_index, uint32_t waiting) {
    return ring_add<NUM_SLOTS>(next_index, NUM_SLOTS - waiting);
}

// ---------------------------------------------------------------------------------------------------------------------
// Transaction IDs
// ---------------------------------------------------------------------------------------------------------------------

// The transaction IDs of a receiver ring: every slot of a group carries its group's ID, one of NUM_IDS IDs from
// FIRST_ID (groups past NUM_IDS reuse them in order).
template <uint32_t GROUP_SIZE, uint32_t FIRST_ID, uint32_t NUM_IDS>
struct GroupIds {
    static_assert(GROUP_SIZE == 1 || GROUP_SIZE == 2 || GROUP_SIZE == 4);
    static_assert(is_power_of_2(NUM_IDS));
    static FORCE_INLINE uint8_t of(uint32_t index) {
        return static_cast<uint8_t>(FIRST_ID + group_of<GROUP_SIZE>(index) % NUM_IDS);
    }
};

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

    // Completions (Completions, below). Slots per completion group: under first-level acks one at a time, as the
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

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
static_assert(MAX_NUM_SENDER_CHANNELS <= 0x100);  // a sender channel fits in a byte

// The low n bytes of `word` (n <= 4), the others 0.
FORCE_INLINE uint32_t low_bytes(uint32_t word, uint32_t n) {
    return n == 4 ? word : word & ((1u << (8 * n)) - 1);
}

// The sender channel each slot's packet came from, one byte per slot: byte k of word w is slot 4w + k, so an aligned
// group (1, 2 or 4 slots) lies in one word.
template <uint32_t NUM_SLOTS>
struct SlotRecords {
    static constexpr uint32_t slots_per_word = 4;
    static constexpr uint32_t num_words = (NUM_SLOTS + slots_per_word - 1) / slots_per_word;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void record(uint32_t index, uint32_t source) { bytes()[index] = static_cast<uint8_t>(source); }

    // Slot `index`'s sender channel.
    FORCE_INLINE uint32_t source_of(uint32_t index) const { return bytes()[index]; }

    // The sender channels of slots `first` .. `first + n - 1`, all in one group: slot `first + k`'s in byte k.
    FORCE_INLINE uint32_t group_sources(uint32_t first, uint32_t n) const {
        const uint32_t word = words[first / slots_per_word];
        return low_bytes(word >> (8 * (first % slots_per_word)), n);
    }

private:
    FORCE_INLINE uint8_t* bytes() { return reinterpret_cast<uint8_t*>(words.data()); }
    FORCE_INLINE const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(words.data()); }
};

// The completion counters of one receiver channel (loop locals).
struct CompletionCounts {
    uint32_t uncompleted = 0;  // slots retired (NoC commands issued) but not yet credited
    uint32_t idle_passes = 0;  // groups of several slots: passes without a packet while a partial group waits
};

// Slot `index` was retired; its packet came from sender channel `source`.
template <uint32_t NUM_SLOTS>
FORCE_INLINE void slot_retired(
    CompletionCounts& counts, SlotRecords<NUM_SLOTS>& records, uint32_t index, uint32_t source) {
    records.record(index, source);
    counts.uncompleted++;
}

// A receiver channel's per-packet scalars: its position and its completion counts.
// Placement: VC0's is a loop local of its own, as is `frame_due`; inside a larger aggregate GCC keeps parts of them in
// memory (+4 stack accesses in the loop, V4_RECEIVER_SHAPE.md §5).
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

// VC1's and VC2's cursor and table, kept in one record.
// Placement: GCC keeps a record with a run-time-indexed member (the table) in memory, so these cursors do not take
// VC0's registers. GCC's choice, not a contract: as loop locals they cost VC0 +9.4 cycles/packet (TorusY1HopNS scatter
// with VC2, v4rx2, V4_RECEIVER_SHAPE.md §6).
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
// Placement: an object of its own; inside an aggregate with the tables its constant block addresses are loaded from the
// stack at each use (+9 stack accesses in the 2D loop, V4_RECEIVER_SHAPE.md §5).
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
// Arrivals and first-level acks
// ---------------------------------------------------------------------------------------------------------------------

// One more ack to sender channel `sender_channel` in the credit counts, as accumulate_completion_credit does for
// completions (counter credits only; stream-register credits, which send each credit on its own, are not brought up).
template <typename CreditSender>
FORCE_INLINE void accumulate_ack(CreditSender& credits, uint8_t sender_channel) {
    static_assert(std::is_same_v<CreditSender, ReceiverChannelCounterBasedResponseCreditSender>);
    const uint32_t acks = credits.ack_counters[sender_channel] + 1;
    credits.ack_counters[sender_channel] = acks;
    credits.ack_counters_base_ptr[sender_channel] = acks;
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
        accumulate_ack(credits, source_channel(header->src_ch_id));
        index = ring_add<Channel<CH>::num_slots>(index, 1);
    }
    position.ready += arrived;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------------------------------------------------

// The header fields every packet needs: payload size, send type and sender channel. They share one header word, read
// with one load and kept raw until the packet is handled (one register rather than three).
struct PacketInfo {
    uint16_t payload_size_bytes;
    tt::tt_fabric::NocSendType noc_send_type;
    uint8_t src_ch_id;

    static constexpr size_t word_offset = offsetof(PacketHeaderBase, payload_size_bytes);
    static_assert(word_offset == 40);
    static constexpr uint32_t send_type_shift = 8 * (offsetof(PacketHeaderBase, noc_send_type) - word_offset);
    static constexpr uint32_t src_ch_id_shift = 8 * (offsetof(PacketHeaderBase, src_ch_id) - word_offset);
    static_assert(send_type_shift == 16 && src_ch_id_shift == 24);

    // The word as it is in the header.
    static FORCE_INLINE uint32_t load_word(const volatile PACKET_HEADER_TYPE* header) {
        return *reinterpret_cast<const volatile uint32_t*>(reinterpret_cast<uintptr_t>(header) + word_offset);
    }

    // Its fields.
    static FORCE_INLINE PacketInfo decode(uint32_t word) {
        return {
            static_cast<uint16_t>(word & 0xFFFFu),
            static_cast<tt::tt_fabric::NocSendType>((word >> send_type_shift) & 0xFFu),
            static_cast<uint8_t>(word >> src_ch_id_shift)};
    }
};

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
    const PacketInfo packet = PacketInfo::decode(header_word);
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
    const uint32_t header_word = PacketInfo::load_word(header);
    uint32_t route_word = 0;
    if constexpr (Channel<CH>::routes) {
        route_word = Channel<CH>::Routing::read(header, me);
    }
    return handle<CH>(cursor, table, downstreams, header_word, route_word);
}

// ---------------------------------------------------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------------------------------------------------

// Completion credits: a receiver slot goes back to the far sender channel its packet came from once every NoC command
// that reads the slot has left the NIU (the far sender may then overwrite it).
//
// - Credit frame: the credits are absolute counts per sender channel in this router's L1 (completions, and acks on
//   first-level-ack VCs), copied to the far sender's L1 as one Ethernet frame. accumulate_ack and release_completions
//   only raise the counts; the receiver loop sends one frame per pass when they changed (send_credit_frame).
// - Groups: a ring is credited in aligned groups of slots (completion_group_size), so the far sender returns one credit
//   to its own upstream per group instead of one per slot. An aligned group never wraps around the ring.
// - Proof: every NoC command that reads a slot carries its group's transaction ID (GroupIds), and the NIU counts each
//   ID's outgoing writes over every command buffer, so one status read proves a whole group free. Where a channel has
//   fewer IDs than groups (rings and tori: one slot per group, 8 IDs; VC1 builds, 4 IDs per channel: rings over 16
//   slots; meshes and 1D, 16 IDs: over 64 slots), two groups share an ID and a group's proof also waits for the
//   other's writes: conservative, never wrong.
//
// When a group goes out (release_completions):
//   due     while packets keep arriving, once more than a group waits (the slot retired this pass is never in it); on
//           a pass without a packet, a full group at once, a partial one after release_after_idle_passes such passes;
//   proof   the oldest group's transaction ID: a single slot is checked once, a group waits for it (bounded);
//   credit  one count of n to the slots' sender channel when they all came from one (usual), else one per slot.
//
// Why (measured; keep unless re-measured):
//   - a channel holds back at most a group minus one of its slots, and only until its next packet or a few idle passes
//     (LEDGER E1-28: VC1 -12 to -83, no bursty losses);
//   - a group waits for its proof rather than leaving it to the next pass, which under load handles the next packet
//     first and delays the credit by a packet (LEDGER E1-24, E1-25);
//   - a single slot is checked once: waiting costs ~3 cycles/packet on rings and tori (RESULTS §98 v4sc2).

// Whether every write tagged `trid` has left this ERISC's NIU (local writes and forwards share one NoC: one register).
static_assert(tt::tt_fabric::local_chip_noc_equals_downstream_noc);
FORCE_INLINE bool writes_left(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(tt::tt_fabric::edm_to_local_chip_noc, trid);
}

// Whether the writes tagged `trid` have left, reading the status up to MAX_CHECKS times.
template <uint32_t MAX_CHECKS>
FORCE_INLINE bool writes_left_within(uint32_t trid) {
    for (uint32_t check = 1; check < MAX_CHECKS; check++) {
        if (writes_left(trid)) {
            return true;
        }
    }
    return writes_left(trid);
}

// Due: whether the oldest uncredited slots go back this pass, and how many (`count`; the Completions introduction).
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS>
FORCE_INLINE bool due(CompletionCounts& counts, uint32_t next_index, bool progress, uint32_t& count) {
    const uint32_t waiting = counts.uncompleted;
    if constexpr (GROUP_SIZE == 1) {
        // One slot as soon as a slot other than this pass's waits (no partial group, so no idle count).
        count = 1;
        return waiting > static_cast<uint32_t>(progress);
    } else {
        bool is_due;
        if (progress) {
            counts.idle_passes = 0;
            is_due = waiting > GROUP_SIZE;
        } else {
            is_due = waiting >= GROUP_SIZE || (waiting != 0 && ++counts.idle_passes >= release_after_idle_passes);
        }
        if (!is_due) {
            return false;
        }
        // The oldest uncredited slot and the rest of its group that has been retired.
        const uint32_t first = oldest_uncredited<NUM_SLOTS>(next_index, waiting);
        const uint32_t left_in_group = std::min(group_end<GROUP_SIZE>(first), NUM_SLOTS) - first;
        count = std::min(waiting, left_in_group);
        return true;
    }
}

// Credit: `count` slots from `first` (one group's) to their sender channels: as one count when they all came from one
// (the usual case), else one per slot.
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS, typename CreditSender>
FORCE_INLINE void credit_slots(
    const SlotRecords<NUM_SLOTS>& records, uint32_t first, uint32_t count, CreditSender& credits) {
    if constexpr (GROUP_SIZE == 1) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(records.source_of(first)), 1);
    } else {
        // One sender channel when every byte of `sources` equals the first. The expressions stay inline: wrapped in
        // helpers, or with the test in a named bool, they change the -Os code (production VC1 builds).
        const uint32_t sources = records.group_sources(first, count);  // slot first + k's sender channel in byte k
        const uint32_t first_source = sources & 0xFF;
        const uint32_t one_per_slot = count == 4 ? 0x01010101u : 0x01010101u & ((1u << (8 * count)) - 1);
        if (sources == first_source * one_per_slot) {
            credits.accumulate_completion_credit(static_cast<uint8_t>(first_source), count);
        } else {
            for (uint32_t k = 0; k < count; k++) {
                credits.accumulate_completion_credit(static_cast<uint8_t>((sources >> (8 * k)) & 0xFF), 1);
            }
        }
    }
}

// Credits channel CH's oldest uncredited slots when they are due and their writes have left; returns whether it did.
// `next_index` is the next slot to retire; `progress` says whether this pass handled a packet.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool release_completions(
    CompletionCounts& counts,
    const SlotRecords<Channel<CH>::num_slots>& records,
    uint32_t next_index,
    bool progress,
    CreditSender& credits) {
    constexpr uint32_t group_size = Channel<CH>::group_size;
    constexpr uint32_t num_slots = Channel<CH>::num_slots;
    static_assert(group_size >= 1 && group_size <= 4 && group_size <= num_slots);
    const uint32_t waiting = counts.uncompleted;
    uint32_t count;
    if (!due<group_size, num_slots>(counts, next_index, progress, count)) {
        return false;
    }
    // Proof: a single slot is checked once; a group waits for it, bounded.
    constexpr uint32_t proof_checks = group_size == 1 ? 1 : completion_wait_checks;
    const uint32_t first = oldest_uncredited<num_slots>(next_index, waiting);
    if (!writes_left_within<proof_checks>(Channel<CH>::TransactionIds::of(first))) {
        return false;
    }
    if constexpr (group_size > 1) {
        counts.idle_passes = 0;
    }
    credit_slots<group_size>(records, first, count, credits);
    counts.uncompleted = waiting - count;
    return true;
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
    const uint32_t arrived = arrived_raw & stream_count_mask;
    if constexpr (Channel<CH>::acks) {
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
    frame_due |= release_completions<CH>(cursor.counts, table.sources, cursor.position.index, progress, credits);
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
// it is used). The body of run()'s loop, which calls it on its own locals; a single-ERISC router runs it as the
// receiver half of each pass over its state in memory (router_single.hpp). The state types are template parameters so
// that an absent channel's state is named only in template-dependent code (Present).
// Placement: the group's values are scalars of their own; a record across the fences is kept in memory (7 to 10
// cycles/packet on first-level-ack rows, v4rx1j vs v4rx1js, V4_RECEIVER_SHAPE.md §12).
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
// its downstream VC's (downstreams_for).
// Placement: forwards use them, but packets delivered here do not, so they live in a table in memory rather than in the
// registers every packet needs (in registers, 2D scatter +5 cycles/packet, v4b vs v4c, RESULTS §98).
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
