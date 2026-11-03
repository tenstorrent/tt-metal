// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the far router's sender channels, and for each one
// forwards it to the next router on this chip, writes it to this chip, or both, then returns the slot's credit.
//
// Each pass, one step per serviced receiver channel (in channel order; step), the same for every channel:
//   ack       channels with first-level acks (the host's ENABLE_FIRST_LEVEL_ACK_VC<n>): ack every packet that arrived
//   poll      packets known present: the count from the packets-received register (one of the pass's grouped reads)
//             is taken only when the count kept from its last read is used up (the register stays the source of truth;
//             the kept count is a lower bound of it); under first-level acks, the packets acked and not yet processed
//   decode    the route: deliver here, and which downstream routers to forward to (router_routing.hpp; with the
//             landing and exit steps of multi-mesh fabrics)
//   admit     every downstream the packet goes to has a free slot (router_downstream.hpp)
//   forward   one copy per downstream
//   deliver   the packet's NoC commands on this chip (router_local_delivery.hpp); in UDM mode, a copy to the fabric's
//             relay instead, which does them (router_downstream.hpp; admitted with the forwards)
//   retire    the slot's source (its sender channel) recorded, packets-received register -1, next slot
//   credit    completion credits back to the far sender (Completions, below)
// A channel that does not forward (its VC's forwarding is off, the host's DISABLE_RX_CH<n>_FORWARDING) only delivers:
// it skips decode and admit (Channel::forwards).
// Around the steps, once per pass:
//   reads     every channel's packets-received register and, where a channel has first-level acks (a frame goes out
//             most passes), the TX queue's status, as one group of reads
//   frame     one credit frame, when the pass's acks or completions changed the counts
//
// Channels and VCs. A receiver channel's features are its VC's (Channel<CH>, the one place a channel's VC is named):
// VC0's channel is 0, VC1's 1, VC2's 2, or 1 on routers without VC1.
//
// State, by placement class: each channel's cursor and slot sources (`ChannelStates`, whose note states where they
// live); whether a credit frame is due, a loop local; the credit sender, an object of its own; ring bounds, stream ids
// and command buffers, compile-time. The downstream connections are passed in, by VC (router_downstream.hpp).
//
// Sections, in order: tuning; ring arithmetic and completion groups; transaction IDs; channels and their features;
// state; arrivals and first-level acks; packets; completions; the pass.

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

// Passes without a packet after which a partial completion group is credited.
constexpr uint32_t idle_passes_before_partial_credit = 4;
// Status reads a completion group's proof waits for, at most; then its credit is left to a later pass.
constexpr uint32_t completion_proof_max_checks = 32;

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
constexpr uint32_t completion_group_size_of_ring = NUM_SLOTS < 8 ? 1 : (NUM_SLOTS < 16 ? 2 : 4);

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
struct CompletionGroupTrids {
    static_assert(GROUP_SIZE == 1 || GROUP_SIZE == 2 || GROUP_SIZE == 4);
    static_assert(is_power_of_2(NUM_IDS));
    static FORCE_INLINE uint8_t of_slot(uint32_t index) {
        const uint32_t group = index / GROUP_SIZE;
        return static_cast<uint8_t>(FIRST_ID + group % NUM_IDS);
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Channels and their features
// ---------------------------------------------------------------------------------------------------------------------

// Whether this router runs receiver channel `ch`: a channel of a VC it services, unless channel trimming took it out.
// (From the host's tables, not Channel<CH>: an absent channel's traits are never computed.)
constexpr bool runs_channel(size_t ch) {
    return ch < NUM_RECEIVER_CHANNELS && is_vc_serviced[vc_of_receiver_channel(ch)] && is_receiver_channel_serviced[ch];
}

// Receiver channel CH: its VC, its ring and packets-received register, and its features, which are its VC's.
template <size_t CH>
struct Channel {
    static constexpr size_t vc = vc_of_receiver_channel(CH);

    // The ring: num_slots slots of channel_buffer_size bytes from first_slot_addr, each starting with its header.
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot_addr = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[CH];

    // Features, the host's per VC or per channel (fabric_erisc_router_ct_args.hpp).
    // Forwards: decodes each packet's route, then forwards and/or delivers. A channel that does not forward only
    // delivers: every packet is for this chip, no route to decode, nothing to admit.
    static constexpr bool forwards = !is_vc_forwarding_disabled[vc];
    // First-level acks (accumulate_first_level_acks).
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[vc];
    // Intermesh ingress: the channel's packets come from another mesh (router_routing.hpp).
    static constexpr bool intermesh_ingress = receiver_channel_is_intermesh_ingress[CH];
    // The VC of the downstream routers the channel forwards to: its own, or VC1 where it crosses over.
    static constexpr size_t forwards_on_vc = is_vc_crossover_to_vc1[vc] ? 1 : vc;
    // UDM: packets for this chip go to the fabric's relay (router_downstream.hpp), admitted with the forwards.
    static constexpr bool delivers_through_relay = udm_mode;
#if defined(FABRIC_2D)
    using Routing = Routing2D<forwards_on_vc, intermesh_ingress>;
#else
    using Routing = Routing1D;
#endif

    // Completions (Completions, below). Slots per completion group: one under first-level acks, where held completion
    // credits stall bubble flow control (RESULTS §98).
    static constexpr uint32_t completion_group_size = first_level_ack ? 1 : completion_group_size_of_ring<num_slots>;
    // The transaction IDs of the slots' NoC commands: one per completion group, from the channel's own range.
    using CompletionTrids = CompletionGroupTrids<completion_group_size, RX_CH_TRID_STARTS[CH], NUM_TRANSACTION_IDS>;

    // The header address of slot `index`.
    static FORCE_INLINE uint32_t slot_addr(uint32_t index) { return first_slot_addr + index * channel_buffer_size; }
    // The ring index after `index`.
    static FORCE_INLINE uint32_t next_index(uint32_t index) { return index + 1 == num_slots ? 0 : index + 1; }

    // Credits go back as counts in one frame (send_credit_frame); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(CH));
    static_assert(num_slots > 0 && num_slots <= 255 && num_slots <= stream_count_mask);
};

// Whether some receiver channel runs here with first-level acks.
constexpr bool any_first_level_ack_channel = [] {
    for (size_t ch = 0; ch < MAX_NUM_RECEIVER_CHANNELS; ch++) {
        if (runs_channel(ch) && is_vc_first_level_ack_enabled[vc_of_receiver_channel(ch)]) {
            return true;
        }
    }
    return false;
}();

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

// The sender channel each slot's packet came from, one byte per slot: byte k of word w is slot 4w + k, so an aligned
// group (1, 2 or 4 slots) lies in one word.
template <uint32_t NUM_SLOTS>
struct SlotSources {
    static constexpr uint32_t slots_per_word = 4;
    static constexpr uint32_t num_words = (NUM_SLOTS + slots_per_word - 1) / slots_per_word;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void set(uint32_t index, uint32_t source) { bytes()[index] = static_cast<uint8_t>(source); }

    // Slot `index`'s sender channel.
    FORCE_INLINE uint32_t source_of(uint32_t index) const { return bytes()[index]; }

    // The sender channels of slots `first` .. `first + n - 1`, all in one group: slot `first + k`'s in byte k.
    FORCE_INLINE uint32_t completion_group_sources(uint32_t first, uint32_t n) const {
        const uint32_t word = words[first / slots_per_word] >> (8 * (first % slots_per_word));
        return n == 4 ? word : word & ((1u << (8 * n)) - 1);  // its low n bytes
    }

private:
    FORCE_INLINE uint8_t* bytes() { return reinterpret_cast<uint8_t*>(words.data()); }
    FORCE_INLINE const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(words.data()); }
};

// The completion counters of one receiver channel (loop locals).
struct CompletionCounts {
    uint32_t uncredited = 0;  // slots retired (NoC commands issued) but not yet credited
    // Groups of several slots: passes without a packet while a partial completion group waits.
    uint32_t idle_passes_with_partial_group = 0;
};

// A receiver channel's per-packet scalars: its position and its completion counts.
struct Cursor {
    Position position;
    CompletionCounts counts;
};

// Channel CH's cursor where this router runs the channel (Present, router_common.hpp), and its value before the
// channel's first packet.
template <size_t CH>
using CursorOf = Present<runs_channel(CH), Cursor, CH>;

template <size_t CH>
constexpr CursorOf<CH> cursor_start() {
    if constexpr (runs_channel(CH)) {
        return Cursor{Position{0, Channel<CH>::first_slot_addr, 0}, CompletionCounts{}};
    } else {
        return {};
    }
}

// Channel CH's slot sources, in memory (indexed by ring slot at run time): the sender channel each retired slot's
// packet came from, for its completion credit.
template <size_t CH>
using SlotSourcesOf = SlotSources<Channel<CH>::num_slots>;

// A receiver channel's state: its cursor and its slot sources.
template <size_t CH>
struct ChannelState {
    Cursor cursor = cursor_start<CH>();
    SlotSourcesOf<CH> sources{};
};

// The first receiver channel: its cursor is the one the receiver loop carries as a local (ChannelStates).
constexpr size_t first_channel = 0;

// Every receiver channel's state. A member exists only where this router runs its channel (Present, router_common.hpp).
// Placement: the first channel's cursor is carried through the receiver loop as a local copy (run), so it keeps its
// registers; every later channel's state stays in memory between its steps (step_channel), so it does not take the
// first channel's registers on its packet path (V4_RECEIVER_OWNERSHIP.md §14; loopvars.py's headroom check guards it).
struct ChannelStates {
    [[no_unique_address]] Present<runs_channel(0), ChannelState<0>, 0> ch0;
    [[no_unique_address]] Present<runs_channel(1), ChannelState<1>, 1> ch1;
    [[no_unique_address]] Present<runs_channel(2), ChannelState<2>, 2> ch2;
};

// The first channel's cursor in `states`: what the receiver loop copies into its local (none where the channel does
// not run). A template so that an absent channel's state is named only in template-dependent code (Present).
template <typename States>
FORCE_INLINE auto& first_channel_cursor_in(States& states) {
    if constexpr (runs_channel(first_channel)) {
        return states.ch0.cursor;
    } else {
        return states.ch0;
    }
}

// One credit sender for every receiver channel: counter credits are one L1 block indexed by the far sender channel, and
// the channels' far sender channels are different channels, so one frame per pass carries every channel's credits.
// Placement: an object of its own; inside an aggregate with the channel states its constant block addresses would be
// loaded from the stack at each use (V4_RECEIVER_SHAPE.md §5).
using Credits = ReceiverChannelCounterBasedResponseCreditSender;

// ---------------------------------------------------------------------------------------------------------------------
// Arrivals and first-level acks
// ---------------------------------------------------------------------------------------------------------------------

// First-level acks. A packet that has arrived is acked to the far sender before it is processed, and the far sender
// returns its slot to the producer on the ack. Every packet that has arrived since the last pass is acked: its sender
// channel's ack count goes up (the frame goes out with the pass's other credits), and the packets-received register is
// consumed here rather than at retire. The packets acked are the oldest not yet acked: `ready` slots past the next
// one to process. `arrived` is the packets-received count, read at the start of the pass. Returns whether it acked any.
template <size_t CH>
FORCE_INLINE bool accumulate_first_level_acks(Position& position, Credits& credits, uint32_t arrived) {
    if (arrived == 0) {
        return false;
    }
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    increment_local_update_ptr_val<Channel<CH>::packets_received_stream>(-static_cast<int32_t>(arrived));
    uint32_t index = ring_add<Channel<CH>::num_slots>(position.index, position.ready);
    for (uint32_t i = 0; i < arrived; i++) {
        const auto* header =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(Channel<CH>::slot_addr(index));
        credits.accumulate_ack_credit(source_channel(header->src_ch_id));
        index = ring_add<Channel<CH>::num_slots>(index, 1);
    }
    position.ready += arrived;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------------------------------------------------

// The packet in the next slot is done here (its NoC commands are issued): its source recorded for its completion credit
// and the slot counted uncredited, the packets-received register -1 (under first-level acks, consumed at the ack),
// the next slot.
template <size_t CH>
FORCE_INLINE void retire(Cursor& cursor, SlotSourcesOf<CH>& sources, uint32_t source) {
    Position& position = cursor.position;
    sources.set(position.index, source);
    cursor.counts.uncredited++;
    position.ready--;
    if constexpr (!Channel<CH>::first_level_ack) {
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
            constexpr std::array<uint32_t, 2> packed_by_vc = {
                PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS, PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS};
            constexpr uint32_t packed = packed_by_vc[Channel<CH>::forwards_on_vc];
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
    Downstreams<Channel<CH>::forwards_on_vc>& downstreams,
    uint32_t slot_addr,
    uint32_t size,
    uint8_t trid) {
    execute_chip_unicast_to_local_chip_impl(
        header, payload_size, tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE, trid, CH);
    auto& sparse = header->command_fields.sparse_mcast_write;
    sparse.write_idx += sparse.counts[sparse.chip_idx];
    sparse.chip_idx++;
    Channel<CH>::Routing::update_for_next_hop(header, route_word);
    downstreams.forward_all(forward_to, slot_addr, size, trid);
    record_forwarded<CH>(forward_to);
}

// The packet in the cursor's slot, from its header word and (on a channel that forwards) its route word: admit, forward
// and/or deliver, retire. Returns whether it handled it (false: a downstream had no room; the packet stays for a later
// pass).
template <size_t CH, typename Connections>
FORCE_INLINE bool handle(
    Cursor& cursor,
    SlotSourcesOf<CH>& sources,
    Connections& downstreams,
    uint32_t header_word,
    uint32_t route_word) {
    using Routing = typename Channel<CH>::Routing;
    const auto packet = PACKET_HEADER_TYPE::PackedPayloadAndSendType::unpack(header_word);
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(cursor.position.slot_addr);
    if constexpr (Channel<CH>::forwards) {
        auto& routers = downstreams.template of<Channel<CH>::forwards_on_vc>();  // the downstream routers it forwards to
        const typename Routing::Route route(route_word);
        if (route.forwards() && !routers.admit(route.forward_to())) {
            return false;
        }
        if constexpr (Channel<CH>::delivers_through_relay) {
            if (route.delivers() && !downstreams.relay.admit()) {
                return false;
            }
        }
        const uint8_t trid = Channel<CH>::CompletionTrids::of_slot(cursor.position.index);
        const uint32_t size = packet.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
        note_received<CH>(size);
        if (!is_2d && route.delivers() && route.forwards() &&
            packet.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
            sparse_multicast_hop<CH>(
                header,
                packet.payload_size_bytes,
                route_word,
                route.forward_to(),
                routers,
                cursor.position.slot_addr,
                size,
                trid);
        } else {
            // Forward first (the next router starts sooner), then write here.
            if (route.forwards()) {
                Routing::update_for_next_hop(header, route_word);
                routers.forward_all(route.forward_to(), cursor.position.slot_addr, size, trid);
                record_forwarded<CH>(route.forward_to());
            }
            if (route.delivers()) {
                if constexpr (Channel<CH>::delivers_through_relay) {
                    downstreams.relay.forward(cursor.position.slot_addr, size, trid);
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
            Channel<CH>::CompletionTrids::of_slot(cursor.position.index));
    }
    retire<CH>(cursor, sources, source_channel(packet.src_ch_id));
    return true;
}

// One packet, when one is present (acked, under first-level acks; otherwise counted present: `arrived` is the
// channel's packets-received count, read at the start of the pass) and (on a channel that forwards) its downstreams
// have room: its header word and route word read, then handled. Returns whether it handled one.
template <size_t CH, typename Connections>
FORCE_INLINE bool receive(
    Cursor& cursor,
    SlotSourcesOf<CH>& sources,
    Connections& downstreams,
    uint32_t arrived,
    const Coordinates& me) {
    Position& position = cursor.position;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (position.ready == 0) {
        if constexpr (Channel<CH>::first_level_ack) {
            return false;  // nothing acked to process
        } else {
            position.ready = arrived;  // the register's count, from the pass's grouped reads
            if (position.ready == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(position.slot_addr);
    // Payload size, send type and source channel: one header word, one load (the header's accessor), kept as it is
    // until the packet is handled.
    const uint32_t header_word = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load_word(header);
    uint32_t route_word = 0;
    if constexpr (Channel<CH>::forwards) {
        route_word = Channel<CH>::Routing::read(header, me);
    }
    return handle<CH>(cursor, sources, downstreams, header_word, route_word);
}

// ---------------------------------------------------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------------------------------------------------

// Completion credits: a receiver slot goes back to the far sender channel its packet came from once every NoC command
// that reads the slot has left the NIU (the far sender may then overwrite it).
//
// - Credit frame: the credits are absolute counts per sender channel in this router's L1 (completions, and acks on
//   first-level-ack VCs), copied to the far sender's L1 as one Ethernet frame. accumulate_first_level_ack and
//   credit_completions only raise the counts; the receiver loop sends one frame per pass when they changed
//   (send_credit_frame).
// - Completion groups: a ring is credited in aligned groups of slots (completion_group_size), so the far sender returns
//   one credit to its own upstream per group instead of one per slot. An aligned group never wraps around the ring.
// - Proof: every NoC command that reads a slot carries its completion group's transaction ID (CompletionGroupTrids),
//   and the NIU counts each ID's outgoing writes over every command buffer, so one status read proves a whole group
//   free. Where a channel has fewer IDs (NUM_TRANSACTION_IDS) than groups, two groups share an ID and a group's proof
//   also waits for the other's writes: conservative, never wrong.
//
// When a group goes out (credit_completions):
//   due     for credit: while packets keep arriving, once more than a group waits (the slot retired this pass is never
//           in it); on a pass without a packet, a full group at once, a partial one after
//           idle_passes_before_partial_credit such passes;
//   proof   the oldest group's transaction ID: a single slot is checked once, a group waits for it (bounded);
//   credit  one count of n to the slots' sender channel when they all came from one (usual), else one per slot.
//
// Why (measured):
//   - a channel holds back at most a group minus one of its slots, and only until its next packet or a few idle passes
//     (LEDGER E1-28);
//   - a group waits for its proof: left to the next pass, which under load handles the next packet first, its credit
//     would come a packet later (LEDGER E1-24);
//   - a single slot is checked once: on rings and tori waiting for it costs more than it saves (RESULTS §98).

// Whether every write tagged `trid` has left this ERISC's NIU (local writes and forwards share one NoC: one register).
static_assert(tt::tt_fabric::local_chip_noc_equals_downstream_noc);
FORCE_INLINE bool trid_writes_sent(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(tt::tt_fabric::edm_to_local_chip_noc, trid);
}

// Whether the writes tagged `trid` have left, reading the status up to MAX_CHECKS times.
template <uint32_t MAX_CHECKS>
FORCE_INLINE bool trid_writes_sent_within(uint32_t trid) {
    for (uint32_t check = 1; check < MAX_CHECKS; check++) {
        if (trid_writes_sent(trid)) {
            return true;
        }
    }
    return trid_writes_sent(trid);
}

// How many of the oldest uncredited slots are due for credit this pass: none (0), or the rest of the oldest slot's
// completion group that has been retired (the Completions introduction).
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t slots_due_for_credit(CompletionCounts& counts, uint32_t next_index, bool progress) {
    const uint32_t waiting = counts.uncredited;
    if constexpr (GROUP_SIZE == 1) {
        // One slot as soon as a slot other than this pass's waits (no partial group, so no idle count).
        return waiting > static_cast<uint32_t>(progress) ? 1 : 0;
    } else {
        bool is_due;
        if (progress) {
            counts.idle_passes_with_partial_group = 0;
            is_due = waiting > GROUP_SIZE;
        } else {
            is_due = waiting >= GROUP_SIZE ||
                     (waiting != 0 && ++counts.idle_passes_with_partial_group >= idle_passes_before_partial_credit);
        }
        if (!is_due) {
            return 0;
        }
        // The oldest uncredited slot and the rest of its group that has been retired.
        const uint32_t first = oldest_uncredited<NUM_SLOTS>(next_index, waiting);
        const uint32_t group_end = (first | (GROUP_SIZE - 1)) + 1;  // just past first's group (groups are aligned)
        const uint32_t left_in_group = std::min(group_end, NUM_SLOTS) - first;
        return std::min(waiting, left_in_group);
    }
}

// Credit: `count` slots from `first` (one group's) to their sender channels: as one count when they all came from one
// (the usual case), else one per slot.
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS, typename CreditSender>
FORCE_INLINE void accumulate_completion_credits(
    const SlotSources<NUM_SLOTS>& sources, uint32_t first, uint32_t count, CreditSender& credits) {
    if constexpr (GROUP_SIZE == 1) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(sources.source_of(first)), 1);
    } else {
        // One sender channel when every byte of `group` equals the first. The expressions stay inline: wrapped in
        // helpers, or with the test in a named bool, they change the -Os code (production VC1 builds).
        // Slot first + k's sender channel in byte k.
        const uint32_t group = sources.completion_group_sources(first, count);
        const uint32_t first_source = group & 0xFF;
        const uint32_t one_per_slot = count == 4 ? 0x01010101u : 0x01010101u & ((1u << (8 * count)) - 1);
        if (group == first_source * one_per_slot) {
            credits.accumulate_completion_credit(static_cast<uint8_t>(first_source), count);
        } else {
            for (uint32_t k = 0; k < count; k++) {
                credits.accumulate_completion_credit(static_cast<uint8_t>((group >> (8 * k)) & 0xFF), 1);
            }
        }
    }
}

// Credits channel CH's oldest uncredited slots when they are due and their writes have left; returns whether it did.
// `next_index` is the next slot to retire; `progress` says whether this pass handled a packet.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool credit_completions(
    CompletionCounts& counts,
    const SlotSourcesOf<CH>& sources,
    uint32_t next_index,
    bool progress,
    CreditSender& credits) {
    constexpr uint32_t completion_group_size = Channel<CH>::completion_group_size;
    constexpr uint32_t num_slots = Channel<CH>::num_slots;
    static_assert(completion_group_size >= 1 && completion_group_size <= 4 && completion_group_size <= num_slots);
    const uint32_t waiting = counts.uncredited;
    const uint32_t count = slots_due_for_credit<completion_group_size, num_slots>(counts, next_index, progress);
    if (count == 0) {
        return false;
    }
    // Proof: a single slot is checked once; a group waits for it, bounded.
    constexpr uint32_t proof_checks = completion_group_size == 1 ? 1 : completion_proof_max_checks;
    const uint32_t first = oldest_uncredited<num_slots>(next_index, waiting);
    if (!trid_writes_sent_within<proof_checks>(Channel<CH>::CompletionTrids::of_slot(first))) {
        return false;
    }
    if constexpr (completion_group_size > 1) {
        counts.idle_passes_with_partial_group = 0;
    }
    accumulate_completion_credits<completion_group_size>(sources, first, count, credits);
    counts.uncredited = waiting - count;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------------------------------------------------

// One step of receiver channel CH (`arrived_raw`: its packets-received register, read at the start of the pass): ack
// what has arrived (a channel with first-level acks), at most one packet (timed for code profiling, as main
// times every receiver channel), then its completions. `credit_frame_due` is set when the credit counts changed.
template <size_t CH, typename Connections>
FORCE_INLINE void step(
    Cursor& cursor,
    SlotSourcesOf<CH>& sources,
    Connections& downstreams,
    Credits& credits,
    bool& credit_frame_due,
    uint32_t arrived_raw,
    const Coordinates& me) {
    const uint32_t arrived = arrived_raw & stream_count_mask;
    if constexpr (Channel<CH>::first_level_ack) {
        credit_frame_due |= accumulate_first_level_acks<CH>(cursor.position, credits, arrived);
    }
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    timer.set_should_dump(cursor.position.ready != 0);
    timer.open();
    const bool progress = receive<CH>(cursor, sources, downstreams, arrived, me);
    timer.close();
    credit_frame_due |= credit_completions<CH>(cursor.counts, sources, cursor.position.index, progress, credits);
}

// Channel CH's packets-received register, raw, for the pass's grouped reads; none (0) where the channel does not run.
template <size_t CH>
FORCE_INLINE uint32_t read_packets_received() {
    if constexpr (runs_channel(CH)) {
        return read_stream_raw<Channel<CH>::packets_received_stream>();
    } else {
        return 0;
    }
}

// Sends the credit frame, all the counts, unless the TX queue is busy (the next pass sends it then). Returns whether it
// was sent. `txq_status_at_pass_start`: the TX queue's status, from the pass's grouped reads on routers with
// first-level acks (unused elsewhere); only this ERISC sends on the queue, so a queue seen free there is still free.
FORCE_INLINE bool send_credit_frame(Credits& credits, uint32_t txq_status_at_pass_start) {
    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        }
    } else {
        const bool seen_free = any_first_level_ack_channel && !txq_busy(txq_status_at_pass_start);
        if (!seen_free && internal_::eth_txq_is_busy(receiver_txq_id)) {
            return false;
        }
    }
    credits.flush_completion_credits();  // the whole credit block: completions and acks
    return true;
}

// Whether VC0's packets-received register reads 0 (telemetry's RX heartbeat, once per round).
FORCE_INLINE bool vc0_register_empty() { return read_stream_count<Channel<0>::packets_received_stream>() == 0; }

// Channel CH's step on its state where the placement puts it (ChannelStates): the first channel on the loop's local
// cursor, a later channel on its state in memory, kept there after the step.
template <size_t CH, typename FirstCursor, typename Connections>
FORCE_INLINE void step_channel(
    FirstCursor& first_channel_cursor,
    ChannelState<CH>& state,
    Connections& downstreams,
    Credits& credits,
    bool& credit_frame_due,
    uint32_t arrived_raw,
    const Coordinates& me) {
    if constexpr (CH == first_channel) {
        step<CH>(first_channel_cursor, state.sources, downstreams, credits, credit_frame_due, arrived_raw, me);
    } else {
        step<CH>(state.cursor, state.sources, downstreams, credits, credit_frame_due, arrived_raw, me);
        keep_in_memory(state);
    }
}

// One pass over the serviced receiver channels: the pass's grouped reads, one step per channel, and the credit frame
// when the counts changed. The group: every channel's packets-received register and (with first-level acks, where a
// frame goes out most passes) the TX queue's status are read together, their reads overlapping, raw (each tested where
// it is used). The body of run()'s loop; a single-ERISC router runs it as the receiver half of each pass over its
// state in memory (router_single.hpp). The state types are template parameters so that an absent channel's state is
// named only in template-dependent code (Present). The three channels are stepped one by one: the group's values are
// scalars of their own (below), which a fold over the channels would gather into a record.
// Placement: the group's values are scalars of their own; a record across the fences would be kept in memory
// (V4_RECEIVER_SHAPE.md §12).
static_assert(MAX_NUM_RECEIVER_CHANNELS == 3, "pass() reads and steps receiver channels 0, 1 and 2");
template <typename FirstCursor, typename States, typename Connections>
FORCE_INLINE void pass(
    FirstCursor& first_channel_cursor,
    bool& credit_frame_due,
    States& states,
    Credits& credits,
    Connections& downstreams,
    const Coordinates& me) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t arrived0 = read_packets_received<0>();
    const uint32_t arrived1 = read_packets_received<1>();
    const uint32_t arrived2 = read_packets_received<2>();
    uint32_t txq_status = 0;
    if constexpr (any_first_level_ack_channel) {
        txq_status = read_txq_status(receiver_txq_id);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if constexpr (runs_channel(0)) {
        step_channel<0>(first_channel_cursor, states.ch0, downstreams, credits, credit_frame_due, arrived0, me);
    }
    if constexpr (runs_channel(1)) {
        step_channel<1>(first_channel_cursor, states.ch1, downstreams, credits, credit_frame_due, arrived1, me);
    }
    if constexpr (runs_channel(2)) {
        step_channel<2>(first_channel_cursor, states.ch2, downstreams, credits, credit_frame_due, arrived2, me);
    }
    if (credit_frame_due) {
        credit_frame_due = !send_credit_frame(credits, txq_status);
    }
}

// The receiver loop: rounds of passes over the serviced receiver channels until termination. A function of its own:
// its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
//
// The downstream connections are passed in (router_downstream.hpp); each channel forwards on its VC's downstream
// routers (Channel<CH>::forwards_on_vc).
__attribute__((noinline)) void run(DownstreamConnections& downstreams) {
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    ChannelStates states;
    CursorOf<first_channel> first_channel_cursor = first_channel_cursor_in(states);  // carried through the loop
    local_write_init();
    Credits credits(0);
    bool credit_frame_due = false;  // the credit counts changed since the last frame

    RoundCounters round;
    while (next_round()) {
        if constexpr (telemetry::any_option) {
            telemetry::round_start();
        }
        for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
            pass(first_channel_cursor, credit_frame_due, states, credits, downstreams, me);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end<false, true>([] { return false; }, [] { return vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
