// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the remote router's sender channels, forwards each to
// downstream routers, writes it to this chip, or both, then credits its slot back to the remote sender.
//
// A pass reads every channel's packets-received register (and, with first-level acks, the TX queue's status) as one
// group, steps each serviced VC's receiver channel in VC order, then sends one credit frame if the counts changed.
// A step: ack (first-level acks), take a packet, decode its route, admit its downstreams, forward, deliver, retire,
// credit completions. A channel that does not forward only delivers: no decode, no admit.
// Decode: router_routing.hpp; admit, forward: router_downstream.hpp; deliver: router_local_delivery.hpp;
// rounds: router_round.hpp.
// The packets-received register is the source of truth for arrivals; the count kept from it is a lower bound.
//
// One channel per serviced VC; the code is keyed by VC (ReceiverChannel<VC>). Only the host's per-channel arrays are
// indexed by receiver channel, receiver_channel_of(VC) (VC2's is 1 on routers without VC1).
// Where the state lives: ReceiverChannelStates (ring progress, slot sources) and Credits (the credit sender).

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"

#include "router_common.hpp"
#include "router_downstream.hpp"
#include "router_init.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::receiver {

// ---------------------------------------------------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------------------------------------------------

// When traffic stops with a completion group part full (say 3 of 4 slots retired), only the next packet would complete
// it, so its slots stay held from the remote sender meanwhile, and a sender with a small window may need them to send
// that packet. After this many passes in a row without a packet, the partial group is credited as it is. Fewer passes
// split groups in the short gaps of a steady stream (more credit frames, a cost on both ERISCs; 1 or 2 cost 2D 3-hop
// writes up to 7 cycles per packet on Blackhole P150); more hold the slots longer after a burst. Groups of 2 or 4 only.
constexpr uint32_t IDLE_PASSES_BEFORE_PARTIAL_CREDIT = 4;
// Status reads a completion group's proof waits for, at most; then its credit is left to a later pass. A safety bound:
// on Blackhole P150 proofs finish within 8 reads (large-write deliverers mostly 5-8); the cap is for slower cases, such
// as non-L1 destinations.
constexpr uint32_t COMPLETION_PROOF_MAX_CHECKS = 32;

// ---------------------------------------------------------------------------------------------------------------------
// Ring arithmetic and completion groups
// ---------------------------------------------------------------------------------------------------------------------

// `index + n` around a ring of NUM_SLOTS slots; n < NUM_SLOTS.
template <uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t index_add(uint32_t index, uint32_t n) {
    const uint32_t sum = index + n;
    if constexpr (is_power_of_2(NUM_SLOTS)) {
        return sum & (NUM_SLOTS - 1);
    } else {
        return sum >= NUM_SLOTS ? sum - NUM_SLOTS : sum;
    }
}

// The oldest uncredited slot: `waiting` slots before the next one to retire.
template <uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t oldest_uncredited(uint32_t next_index, uint32_t waiting) {
    return index_add<NUM_SLOTS>(next_index, NUM_SLOTS - waiting);
}

// ---------------------------------------------------------------------------------------------------------------------
// Transaction IDs
// ---------------------------------------------------------------------------------------------------------------------

// Every slot of a completion group carries its group's ID: the group's own when the channel has an ID per group, else
// shared by mask (Completions).
template <uint32_t GROUP_SIZE, uint32_t FIRST_ID, uint32_t NUM_IDS, uint32_t NUM_GROUPS>
struct CompletionGroupTrids {
    static_assert(GROUP_SIZE == 1 || GROUP_SIZE == 2 || GROUP_SIZE == 4);
    static_assert(NUM_IDS >= 1 && FIRST_ID + NUM_IDS <= NOC_MAX_TRANSACTION_ID + 1);
    static constexpr bool shared = NUM_IDS < NUM_GROUPS;
    static_assert(!shared || is_power_of_2(NUM_IDS), "groups share transaction IDs by mask: a power of 2 of them");
    static FORCE_INLINE uint8_t of_slot(uint32_t index) {
        const uint32_t group = index / GROUP_SIZE;
        if constexpr (shared) {
            return static_cast<uint8_t>(FIRST_ID + (group & (NUM_IDS - 1)));
        } else {
            return static_cast<uint8_t>(FIRST_ID + group);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Channels and their features
// ---------------------------------------------------------------------------------------------------------------------

// The VCs whose receiver channel this router runs, one bit each.
constexpr uint32_t receiver_vcs = [] {
    uint32_t vcs = 0;
    for (size_t vc = 0; vc < MAX_NUM_VCS; vc++) {
        vcs |= runs_vc(vc) ? 1u << vc : 0u;
    }
    return vcs;
}();

// VC's receiver channel: its ring, its packets-received register and its features (the host's, per VC or per channel).
template <size_t VC>
struct ReceiverChannel {
    static constexpr size_t rx = receiver_channel_of(VC);  // the host's index of the channel

    // num_slots slots of channel_buffer_size bytes from first_slot_addr, each starting with its packet header.
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[rx];
    static constexpr uint32_t first_slot_addr = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[rx]>::base_address;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[rx];

    // Without forwarding, a channel only delivers: every packet is for this chip, nothing to decode or admit.
    static constexpr bool forwards = !is_vc_forwarding_disabled[VC];
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[VC];
    // The channel's packets come from another mesh.
    static constexpr bool intermesh_ingress = receiver_channel_is_intermesh_ingress[rx];
    // The VC of the downstream routers the channel forwards to: its own, or VC1 where it crosses over.
    static constexpr size_t forwards_on_vc = forwards_on_vc_of(VC);
    // UDM: packets for this chip go to the fabric's relay, admitted with the forwards.
    static constexpr bool delivers_through_relay = udm_mode;
#if defined(FABRIC_2D)
    using Routing = Routing2D<forwards_on_vc, intermesh_ingress>;
#else
    using Routing = Routing1D;
#endif

    // From the host: one slot per group under first-level acks (held completion credits stall bubble flow control).
    static constexpr uint32_t completion_group_size = RX_CH_COMPLETION_GROUP_SIZE[rx];
    static_assert(!first_level_ack || completion_group_size == 1, "first-level acks credit every slot: groups of 1");
    static constexpr uint32_t completion_groups = (num_slots + completion_group_size - 1) / completion_group_size;
    // The channel's own transaction IDs, from the host: one per completion group, or shared by mask when fewer.
    using CompletionTrids =
        CompletionGroupTrids<completion_group_size, RX_CH_TRID_STARTS[rx], RX_CH_NUM_TRIDS[rx], completion_groups>;

    static FORCE_INLINE uint32_t slot_addr(uint32_t index) { return first_slot_addr + index * channel_buffer_size; }
    static FORCE_INLINE uint32_t index_after(uint32_t index) { return index + 1 == num_slots ? 0 : index + 1; }

    // Credits go back as counts in one frame (send_credit_frame); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(rx));
    static_assert(num_slots > 0 && num_slots <= 255 && num_slots <= stream_count_mask);
};

constexpr bool any_first_level_ack_channel = [] {
    for (size_t vc = 0; vc < MAX_NUM_VCS; vc++) {
        if (runs_vc(vc) && is_vc_first_level_ack_enabled[vc]) {
            return true;
        }
    }
    return false;
}();

// The remote sender channel that sent a packet. In Tensix mux mode the remote router sends everything from its worker
// channel and leaves the header field as the previous hop wrote it, so the field is not used.
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

// A receiver channel's per-packet state.
struct ReceiverChannelProgress {
    uint32_t packets_ready = 0;  // present (and acked under first-level acks), not retired
    uint32_t next_slot_addr = 0;
    uint32_t next_index = 0;
    uint32_t uncredited = 0;                      // retired (NoC commands issued), not yet credited
    uint32_t idle_passes_with_partial_group = 0;  // completion groups of several slots only
};

template <size_t VC>
FORCE_INLINE void advance(ReceiverChannelProgress& progress) {
    progress.next_index = ReceiverChannel<VC>::index_after(progress.next_index);
    progress.next_slot_addr =
        progress.next_index == 0 ? ReceiverChannel<VC>::first_slot_addr : progress.next_slot_addr + channel_buffer_size;
}

// The sender channel each slot's packet came from, byte k of word w for slot 4w + k, so an aligned group lies in one
// word. Only retired, uncredited slots are read, and a group's read masks off the bytes past it, so the initial zeros
// and stale bytes are never used.
template <uint32_t NUM_SLOTS>
struct SlotSources {
    static constexpr uint32_t slots_per_word = 4;
    static constexpr uint32_t num_words = (NUM_SLOTS + slots_per_word - 1) / slots_per_word;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void set(uint32_t index, uint8_t source) { bytes()[index] = source; }

    FORCE_INLINE uint32_t source_of(uint32_t index) const { return bytes()[index]; }

    // Slots `first` .. `first + n - 1`, within one group: slot `first + k`'s sender channel in byte k.
    FORCE_INLINE uint32_t completion_group_sources(uint32_t first, uint32_t n) const {
        const uint32_t word = words[first / slots_per_word] >> (8 * (first % slots_per_word));
        return n == 4 ? word : word & ((1u << (8 * n)) - 1);
    }

private:
    FORCE_INLINE uint8_t* bytes() { return reinterpret_cast<uint8_t*>(words.data()); }
    FORCE_INLINE const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(words.data()); }
};

template <size_t VC>
constexpr ReceiverChannelProgress progress_start() {
    ReceiverChannelProgress progress;
    progress.next_slot_addr = ReceiverChannel<VC>::first_slot_addr;
    return progress;
}

// In memory: indexed by ring slot at run time.
template <size_t VC>
using SlotSourcesOf = SlotSources<ReceiverChannel<VC>::num_slots>;

template <size_t VC>
struct ReceiverChannelState {
    ReceiverChannelProgress progress = progress_start<VC>();
    SlotSourcesOf<VC> sources{};
};

// One credit sender for every channel: the counts are one L1 block indexed by remote sender channel, and the channels'
// remote sender channels differ, so one credit frame per pass carries every channel's credits. An object of its own:
// inside an aggregate with the channel states, its constant block addresses would be loaded from the stack at each use.
using Credits = ReceiverChannelCounterBasedResponseCreditSender;

// ---------------------------------------------------------------------------------------------------------------------
// Arrivals and first-level acks
// ---------------------------------------------------------------------------------------------------------------------

// Acks the `arrived` packets (the register's count at pass start), the oldest unacked, before they are processed; the
// remote sender credits their slots to its producers on the ack. The register is consumed here, not at retire. Returns
// whether it acked.
template <size_t VC>
FORCE_INLINE bool accumulate_first_level_acks(ReceiverChannelProgress& progress, Credits& credits, uint32_t arrived) {
    if (arrived == 0) {
        return false;
    }
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    increment_local_update_ptr_val<ReceiverChannel<VC>::packets_received_stream>(-static_cast<int32_t>(arrived));
    uint32_t index = index_add<ReceiverChannel<VC>::num_slots>(progress.next_index, progress.packets_ready);
    for (uint32_t i = 0; i < arrived; i++) {
        const auto* header =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(ReceiverChannel<VC>::slot_addr(index));
        credits.accumulate_ack_credit(source_channel(header->src_ch_id));
        index = index_add<ReceiverChannel<VC>::num_slots>(index, 1);
    }
    progress.packets_ready += arrived;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------------------------------------------------

// The next slot's NoC commands are issued: record its source for the completion credit, count it uncredited, consume
// its packets-received count (already done at the ack under first-level acks), move on.
template <size_t VC>
FORCE_INLINE void retire(ReceiverChannelProgress& progress, SlotSourcesOf<VC>& sources, uint8_t source) {
    sources.set(progress.next_index, source);
    progress.uncredited++;
    progress.packets_ready--;
    if constexpr (!ReceiverChannel<VC>::first_level_ack) {
        increment_local_update_ptr_val<ReceiverChannel<VC>::packets_received_stream>(-1);
    }
    advance<VC>(progress);
}

// Channel-trimming capture and telemetry: a packet of `size` bytes (header included) taken on VC's channel.
template <size_t VC>
FORCE_INLINE void note_received(uint32_t size) {
    channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(ReceiverChannel<VC>::rx);
    if constexpr (telemetry::counts_received) {
        telemetry::note_received(size);
    }
}

// Channel-trimming capture: the downstream routers' sender channels a packet forwarded on VC's channel went to; in 2D
// from the host's packed table, in 1D the downstream's forwarding channel.
template <size_t VC>
FORCE_INLINE void record_forwarded(uint32_t forward_to) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        if constexpr (is_2d) {
            constexpr std::array<uint32_t, 2> packed_by_vc = {
                PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS, PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS};
            constexpr uint32_t packed = packed_by_vc[ReceiverChannel<VC>::forwards_on_vc];
            uint32_t entries = 0;
            for (uint32_t names = forward_to; names != 0; names &= names - 1) {
                entries |= 1u << downstream_entry(lowest_bit(names));
            }
            channel_trimming_usage_recorder.merge_sender_channel_forwarded_to(
                ReceiverChannel<VC>::rx,
                tt::tt_fabric::forwarded_slots_to_sender_mask(static_cast<uint8_t>(entries), packed));
        } else {
            channel_trimming_usage_recorder.set_sender_channel_forwarded_to(
                ReceiverChannel<VC>::rx,
                DownstreamTraits<ReceiverChannel<VC>::forwards_on_vc>::template forwarding_position<0>());
        }
    }
}

// A 1D sparse multicast going past this chip: write this chip's page group, advance write_idx and chip_idx, forward the
// advanced header. The local write comes first because it reads the indices before the advance. Out of line: rare.
template <size_t VC>
__attribute__((noinline)) void sparse_multicast_hop(
    tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint16_t payload_size,
    uint32_t route_word,
    uint32_t forward_to,
    Downstreams<ReceiverChannel<VC>::forwards_on_vc>& downstreams,
    uint32_t slot_addr,
    uint32_t size,
    uint8_t trid) {
    execute_chip_unicast_to_local_chip_impl(
        header, payload_size, tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE, trid, ReceiverChannel<VC>::rx);
    auto& sparse = header->command_fields.sparse_mcast_write;
    sparse.write_idx += sparse.counts[sparse.chip_idx];
    sparse.chip_idx++;
    ReceiverChannel<VC>::Routing::update_for_next_hop(header, route_word);
    downstreams.forward_all(forward_to, slot_addr, size, trid);
    record_forwarded<VC>(forward_to);
}

// Admits, forwards and/or delivers, then retires the next slot's packet. Returns false when a downstream had no room;
// the packet stays for a later pass.
template <size_t VC, typename Connections>
FORCE_INLINE bool handle(
    ReceiverChannelProgress& progress,
    SlotSourcesOf<VC>& sources,
    Connections& downstreams,
    uint32_t header_word,
    uint32_t route_word) {
    using Routing = typename ReceiverChannel<VC>::Routing;
    const auto packet = PACKET_HEADER_TYPE::PackedPayloadAndSendType::unpack(header_word);
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(progress.next_slot_addr);
    if constexpr (ReceiverChannel<VC>::forwards) {
        auto& routers = downstreams.template of<ReceiverChannel<VC>::forwards_on_vc>();
        const typename Routing::Route route(route_word);
        if (route.forwards() && !routers.admit(route.forward_to())) {
            return false;
        }
        if constexpr (ReceiverChannel<VC>::delivers_through_relay) {
            if (route.delivers() && !downstreams.relay.admit()) {
                return false;
            }
        }
        const uint8_t trid = ReceiverChannel<VC>::CompletionTrids::of_slot(progress.next_index);
        const uint32_t size = packet.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
        note_received<VC>(size);
        if (!is_2d && route.delivers() && route.forwards() &&
            packet.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
            sparse_multicast_hop<VC>(
                header,
                packet.payload_size_bytes,
                route_word,
                route.forward_to(),
                routers,
                progress.next_slot_addr,
                size,
                trid);
        } else {
            // Forward first (the downstream router starts sooner), then write here.
            if (route.forwards()) {
                Routing::update_for_next_hop(header, route_word);
                routers.forward_all(route.forward_to(), progress.next_slot_addr, size, trid);
                record_forwarded<VC>(route.forward_to());
            }
            if (route.delivers()) {
                if constexpr (ReceiverChannel<VC>::delivers_through_relay) {
                    downstreams.relay.forward(progress.next_slot_addr, size, trid);
                } else {
                    deliver<ReceiverChannel<VC>::rx>(header, packet.payload_size_bytes, packet.noc_send_type, trid);
                }
            }
        }
    } else {
        note_received<VC>(packet.payload_size_bytes + sizeof(PACKET_HEADER_TYPE));
        deliver<ReceiverChannel<VC>::rx>(
            header,
            packet.payload_size_bytes,
            packet.noc_send_type,
            ReceiverChannel<VC>::CompletionTrids::of_slot(progress.next_index));
    }
    retire<VC>(progress, sources, source_channel(packet.src_ch_id));
    return true;
}

// Handles one packet when one is ready: acked under first-level acks, else counted in `arrived` (the register's count
// at pass start). Returns whether it handled one.
template <size_t VC, typename Connections>
FORCE_INLINE bool receive(
    ReceiverChannelProgress& progress,
    SlotSourcesOf<VC>& sources,
    Connections& downstreams,
    uint32_t arrived,
    const ChipCoordinates& my_chip) {
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (progress.packets_ready == 0) {
        if constexpr (ReceiverChannel<VC>::first_level_ack) {
            return false;  // nothing acked to process
        } else {
            progress.packets_ready = arrived;
            if (progress.packets_ready == 0) {
                return false;
            }
        }
    }
    auto* const header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(progress.next_slot_addr);
    // Payload size, send type and source channel in one load, kept packed until the packet is handled.
    const uint32_t header_word = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load_word(header);
    uint32_t route_word = 0;
    if constexpr (ReceiverChannel<VC>::forwards) {
        route_word = ReceiverChannel<VC>::Routing::read(header, my_chip);
    }
    return handle<VC>(progress, sources, downstreams, header_word, route_word);
}

// ---------------------------------------------------------------------------------------------------------------------
// Completions
// ---------------------------------------------------------------------------------------------------------------------

// A slot is credited to the remote sender channel its packet came from once every NoC command reading it has left the
// NIU.
// - The credits are absolute counts per remote sender channel in L1 (completions, and acks on first-level-ack VCs); the
//   steps only raise them, and the pass sends them as one credit frame when they changed (send_credit_frame).
// - A ring is credited in aligned completion groups, which never wrap, so the remote sender credits its producer once
//   per group instead of per slot.
// - Proof: each slot's NoC commands carry its group's transaction ID and the NIU counts each ID's outgoing writes, so
//   one status read proves a group free. Groups sharing an ID (a channel with fewer IDs than groups) only make a proof
//   wait longer, never wrong.
// - Due: while packets arrive, once more than a group waits (never the slot retired this pass); on an idle pass, a full
//   group at once, a partial one (traffic stopped mid-group) after IDLE_PASSES_BEFORE_PARTIAL_CREDIT idle passes in a
//   row. So a channel holds back at most a group minus one slot, and not for long.
// - A group waits for its proof (bounded): left to the next pass, which under load takes the next packet first, its
//   credit would come a packet later. A single slot is checked once: on rings and tori waiting costs more than it
//   saves.

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

// How many of the oldest uncredited slots are due this pass: 0, or the retired rest of the oldest one's group.
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS>
FORCE_INLINE uint32_t
slots_due_for_credit(ReceiverChannelProgress& progress, uint32_t next_index, bool handled_packet) {
    const uint32_t waiting = progress.uncredited;
    if constexpr (GROUP_SIZE == 1) {
        // No partial groups, so no idle count: due once a slot other than this pass's waits.
        return waiting > static_cast<uint32_t>(handled_packet) ? 1 : 0;
    } else {
        bool is_due;
        if (handled_packet) {
            progress.idle_passes_with_partial_group = 0;
            is_due = waiting > GROUP_SIZE;
        } else {
            is_due = waiting >= GROUP_SIZE ||
                     (waiting != 0 && ++progress.idle_passes_with_partial_group >= IDLE_PASSES_BEFORE_PARTIAL_CREDIT);
        }
        if (!is_due) {
            return 0;
        }
        const uint32_t first = oldest_uncredited<NUM_SLOTS>(next_index, waiting);
        const uint32_t group_end = (first | (GROUP_SIZE - 1)) + 1;  // groups are aligned
        const uint32_t left_in_group = std::min(group_end, NUM_SLOTS) - first;
        return std::min(waiting, left_in_group);
    }
}

// Credits `count` slots from `first`, within one group: one count when they all came from one sender channel (usual),
// else one per slot.
template <uint32_t GROUP_SIZE, uint32_t NUM_SLOTS, typename CreditSender>
FORCE_INLINE void accumulate_completion_credits(
    const SlotSources<NUM_SLOTS>& sources, uint32_t first, uint32_t count, CreditSender& credits) {
    if constexpr (GROUP_SIZE == 1) {
        credits.accumulate_completion_credit(static_cast<uint8_t>(sources.source_of(first)), 1);
    } else {
        // All from one sender channel: every byte of `group` equals the first.
        const uint32_t group = sources.completion_group_sources(first, count);
        constexpr uint32_t BYTE_MASK = 0xFF;
        constexpr uint32_t ONE_PER_BYTE = 0x01010101u;
        const uint32_t first_source = group & BYTE_MASK;
        const uint32_t one_per_slot = count == 4 ? ONE_PER_BYTE : ONE_PER_BYTE & ((1u << (8 * count)) - 1);
        if (group == first_source * one_per_slot) {
            credits.accumulate_completion_credit(static_cast<uint8_t>(first_source), count);
        } else {
            for (uint32_t k = 0; k < count; k++) {
                credits.accumulate_completion_credit(static_cast<uint8_t>((group >> (8 * k)) & BYTE_MASK), 1);
            }
        }
    }
}

// Credits VC's oldest uncredited slots when they are due and their writes have left; returns whether it did.
template <size_t VC, typename CreditSender>
FORCE_INLINE bool credit_completions(
    ReceiverChannelProgress& progress, const SlotSourcesOf<VC>& sources, bool handled_packet, CreditSender& credits) {
    constexpr uint32_t completion_group_size = ReceiverChannel<VC>::completion_group_size;
    constexpr uint32_t num_slots = ReceiverChannel<VC>::num_slots;
    static_assert(completion_group_size >= 1 && completion_group_size <= 4 && completion_group_size <= num_slots);
    const uint32_t next_index = progress.next_index;
    const uint32_t waiting = progress.uncredited;
    const uint32_t count = slots_due_for_credit<completion_group_size, num_slots>(progress, next_index, handled_packet);
    if (count == 0) {
        return false;
    }
    constexpr uint32_t proof_checks = completion_group_size == 1 ? 1 : COMPLETION_PROOF_MAX_CHECKS;
    const uint32_t first = oldest_uncredited<num_slots>(next_index, waiting);
    if (!trid_writes_sent_within<proof_checks>(ReceiverChannel<VC>::CompletionTrids::of_slot(first))) {
        return false;
    }
    if constexpr (completion_group_size > 1) {
        progress.idle_passes_with_partial_group = 0;
    }
    accumulate_completion_credits<completion_group_size>(sources, first, count, credits);
    progress.uncredited = waiting - count;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------------------------------------------------

// VC's step: ack what arrived (first-level acks), at most one packet, then completions. `arrived_raw`: the channel's
// register from the pass's reads. Sets `credit_frame_due` when the counts changed.
template <size_t VC, typename Connections>
FORCE_INLINE void step(
    ReceiverChannelProgress& progress,
    SlotSourcesOf<VC>& sources,
    Connections& downstreams,
    Credits& credits,
    bool& credit_frame_due,
    uint32_t arrived_raw,
    const ChipCoordinates& my_chip) {
    const uint32_t arrived = arrived_raw & stream_count_mask;
    if constexpr (ReceiverChannel<VC>::first_level_ack) {
        credit_frame_due |= accumulate_first_level_acks<VC>(progress, credits, arrived);
    }
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    timer.set_should_dump(progress.packets_ready != 0);
    timer.open();
    const bool handled_packet = receive<VC>(progress, sources, downstreams, arrived, my_chip);
    timer.close();
    credit_frame_due |= credit_completions<VC>(progress, sources, handled_packet, credits);
}

// VC's packets-received register, raw, for the pass's grouped reads; 0 where its channel does not run.
template <size_t VC>
FORCE_INLINE uint32_t read_packets_received() {
    if constexpr (runs_vc(VC)) {
        return read_stream_raw<ReceiverChannel<VC>::packets_received_stream>();
    } else {
        return 0;
    }
}

// Sends the credit frame unless the TX queue is busy (a later pass retries); returns whether it went. Only this ERISC
// sends on the queue, so a queue free in `txq_status_at_pass_start` (first-level-ack routers) is still free.
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

// Telemetry's RX heartbeat idle test, once per round: VC0's register only, so the heartbeat still moves while another
// VC's channel is stuck.
FORCE_INLINE bool vc0_register_empty() { return read_stream_count<ReceiverChannel<0>::packets_received_stream>() == 0; }

// ---------------------------------------------------------------------------------------------------------------------
// Where the channel states live, and the loop
// ---------------------------------------------------------------------------------------------------------------------

// Every serviced VC's channel state. The loop-local VC's progress is copied into a local of run()'s loop, so it can
// stay in registers; every other VC's state is stepped in memory (a one-ERISC router steps all of them in memory).
// The accessors are templates on VC so an absent channel's state is named only in template-dependent code.
struct ReceiverChannelStates : Each<receiver_vcs, ReceiverChannelState> {
    // The lowest serviced VC (a trimmed router can run VC1's and not VC0's); 0 where none runs.
    static constexpr size_t loop_local_vc = receiver_vcs != 0 ? lowest_bit(receiver_vcs) : 0;

    // What run() copies at its start; without a receiver channel, the states themselves (no step uses it).
    template <size_t VC = loop_local_vc>
    FORCE_INLINE auto& loop_local_progress() {
        if constexpr (runs_vc(VC)) {
            return this->template get<VC>().progress;
        } else {
            return *this;
        }
    }

    template <size_t VC>
    FORCE_INLINE ReceiverChannelProgress& progress_for(ReceiverChannelProgress& loop_copy) {
        if constexpr (VC == loop_local_vc) {
            return loop_copy;
        } else {
            return this->template get<VC>().progress;
        }
    }
};

template <size_t VC, typename LoopCopy, typename Connections>
FORCE_INLINE void step_channel(
    LoopCopy& loop_copy,
    ReceiverChannelStates& states,
    Connections& downstreams,
    Credits& credits,
    bool& credit_frame_due,
    uint32_t arrived_raw,
    const ChipCoordinates& my_chip) {
    if constexpr (runs_vc(VC)) {
        step<VC>(
            states.template progress_for<VC>(loop_copy),
            states.template get<VC>().sources,
            downstreams,
            credits,
            credit_frame_due,
            arrived_raw,
            my_chip);
    }
}

// One pass: the grouped reads first (raw, each tested where it is used), a step per serviced channel, then the credit
// frame when the counts changed. Also the receiver half of a single-ERISC router's pass.
static_assert(MAX_NUM_VCS == 3, "pass() steps the receiver channels of VCs 0, 1 and 2");
template <typename LoopCopy, typename Connections>
FORCE_INLINE void pass(
    LoopCopy& loop_copy,
    bool& credit_frame_due,
    ReceiverChannelStates& states,
    Credits& credits,
    Connections& downstreams,
    const ChipCoordinates& my_chip) {
    const std::array<uint32_t, MAX_NUM_VCS> arrived = {
        read_packets_received<0>(), read_packets_received<1>(), read_packets_received<2>()};
    uint32_t txq_status = 0;
    if constexpr (any_first_level_ack_channel) {
        txq_status = read_txq_status(receiver_txq_id);
    }
    step_channel<0>(loop_copy, states, downstreams, credits, credit_frame_due, arrived[0], my_chip);
    step_channel<1>(loop_copy, states, downstreams, credits, credit_frame_due, arrived[1], my_chip);
    step_channel<2>(loop_copy, states, downstreams, credits, credit_frame_due, arrived[2], my_chip);
    if (credit_frame_due) {
        credit_frame_due = !send_credit_frame(credits, txq_status);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Init (out of line, once)
// ---------------------------------------------------------------------------------------------------------------------

// Before the Ethernet handshake and the ERISCs' first sync, so before the remote sender can increment them: every
// receiver channel's packets-received register at 0, serviced or not.
__attribute__((noinline)) void init_channel_streams() {
    for_each_index<NUM_RECEIVER_CHANNELS>(
        [](auto rx) { init_stream_if_used<to_receiver_packets_sent_streams[decltype(rx)::value]>(0); });
}

// A single-ERISC router's loop takes them whatever channel trimming left it (empty without a receiver channel).
constexpr bool has_downstream_connections = is_receiver_role || NUM_ACTIVE_ERISCS == 1;

// Before the Ethernet handshake: the downstream connections for the receiver loop, set up on the receiver's ERISC, and
// the receiver slot buffers. A template so an absent connection is named only in template-dependent code.
template <typename Connections = Present<has_downstream_connections, DownstreamConnections, 0>>
FORCE_INLINE Connections set_up_channels(const RuntimeArgs& args) {
    Connections downstreams{};
    if constexpr (is_receiver_role) {
        if constexpr (any_forwards_on_vc(0)) {
            init_downstreams(downstreams.vc0, args.vc0());
        }
        if constexpr (any_forwards_on_vc(1)) {
            init_downstreams(downstreams.vc1, args.vc1());
        }
        if constexpr (udm_mode) {
            init_relay(downstreams.relay, args.relay);
        }
        auto receiver_channels =
            ReceiverChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, RECEIVER_TO_ENTRY_IDX>::make();
        receiver_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
    }
    return downstreams;
}

// Tensix mux mode (each direction's mux) and UDM (the relay): each counts itself into this router's sync word once set
// up (the host clears it before launch). Wait for all before opening, so no open lands on one still setting up.
FORCE_INLINE void wait_for_downstream_muxes() {
    if constexpr (num_ds_or_local_tensix_connections != 0) {
        wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
            edm_local_tensix_sync_ptr_addr, num_ds_or_local_tensix_connections, termination_signal());
    }
}

// After the Ethernet handshake, by the host's open gates (rules: fabric_erisc_router_ct_args.hpp). VC0's are opened
// even on an intermesh router whose VC0 crosses over, where no packet uses them: the downstream routers wait for them.
// UDM's relay after the downstream routers.
FORCE_INLINE void open_connections(const RuntimeArgs& args) {
    wait_for_downstream_muxes();
    if constexpr (open_downstream_vc0_connections) {
        open_downstreams<0>(args.vc0());
    }
    if constexpr (has_vc1 && open_downstream_vc1_connections) {
        open_downstreams<1>(args.vc1());
    }
    if constexpr (udm_mode) {
        open_relay(args.relay);
    }
}

// A template on VC so the forward commands of a VC nothing forwards on are named only in template-dependent code.
template <size_t VC>
FORCE_INLINE void preset_forward_pair(const RuntimeArgs& args) {
    if constexpr (CommandBuffers::presets_forward_pair(VC)) {
        ForwardCommands<VC>::preset(VC == 0 ? args.vc0() : args.vc1());
    }
}

// After every connection is open: each forward buffer pair preset by the lowest VC that forwards on it
// (any_forwards_on_vc), so VC1 owns it on a crossover router (CommandBuffers::presets_forward_pair).
FORCE_INLINE void preset_forward_commands(const RuntimeArgs& args) {
    preset_forward_pair<0>(args);
    preset_forward_pair<1>(args);
    static_assert(!any_forwards_on_vc(2), "the runtime args list downstream routers for VC0 and VC1");
}

// The receiver loop: rounds of passes until termination. A function of its own so its code depends only on this body;
// once-per-round work is out of line (router_round.hpp).
__attribute__((noinline)) void run(DownstreamConnections& downstreams) {
    const auto& routing_table = l1_routing_table();
    const ChipCoordinates my_chip{
        routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    ReceiverChannelStates states;
    auto loop_copy = states.loop_local_progress();  // the loop-local VC's progress (ReceiverChannelStates)
    local_write_init();
    Credits credits(0);
    bool credit_frame_due = false;  // the credit counts changed since the last frame

    RoundCounters round;
    while (next_round()) {
        if constexpr (telemetry::any_option) {
            telemetry::round_start();
        }
        for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
            pass(loop_copy, credit_frame_due, states, credits, downstreams, my_chip);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end</*SENDER=*/false, /*RECEIVER=*/true>(
                [] { return false; }, [] { return vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
