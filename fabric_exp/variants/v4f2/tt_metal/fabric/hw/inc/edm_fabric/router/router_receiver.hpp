// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the far router's sender channels, and for each one
// forwards it to the next router on this chip, writes it to this chip, or both, then returns the slot's credit.
//
// One step per receiver channel per pass:
//   ack       first-level-ack VCs (rings and tori): ack every packet that has arrived (acknowledge_arrivals)
//   poll      packets known present: the packets-received register is read only when the count from its last read is
//             used up (the register stays the source of truth; the kept count is a lower bound of it); under
//             first-level acks, the packets acked and not yet processed
//   decode    the route: deliver here, and which downstream routers to forward to (router_routing.hpp)
//   admit     every downstream the packet goes to has a free slot (router_downstream.hpp)
//   forward   one copy per downstream
//   deliver   the packet's NoC commands on this chip (router_local_delivery.hpp)
//   retire    the slot's record, packets-received register -1, next slot
//   release   batched completion credits back to the far sender (router_completions.hpp)
//   frame     one credit frame, when the pass's acks or completions changed the counts
//
// State. Each channel's position and counts, its completion state and its downstream cursors are locals of the loop
// function; ring bounds, stream ids and command buffers are compile-time constants.

#include "router_common.hpp"
#include "router_completions.hpp"
#include "router_downstream.hpp"
#include "router_local_delivery.hpp"
#include "router_round.hpp"
#include "router_routing.hpp"

namespace fabric_router::receiver {

// EXPERIMENT (v4f1 / v4f2): completions one slot at a time under first-level acks (the generic loop's choice), or
// batched as without acks.
constexpr bool exp_ack_completion_per_slot = false;

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
    // First-level acks (VC0 on rings and tori): the far sender returns a packet's slot to its producer when the packet
    // has arrived here, on an ack, rather than when it is completed (acknowledge_arrival).
    static constexpr bool first_level_ack = CH == 0 ? ENABLE_FIRST_LEVEL_ACK_VC0 : ENABLE_FIRST_LEVEL_ACK_VC1;
    // Slots credited per completion batch (router_completions.hpp).
    static constexpr uint32_t completion_batch =
        (first_level_ack && exp_ack_completion_per_slot) ? 1 : completion_batch_size<num_slots>;
    // Credits go back as counts in one frame (router_completions.hpp); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(CH));
#if defined(FABRIC_2D)
    using Routing = Routing2D<downstream_vc>;
#else
    using Routing = Routing1D;
#endif
    using TridTracker = WriteTransactionIdTracker<
        num_slots,
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[CH],
        tt::tt_fabric::edm_to_local_chip_noc,
        tt::tt_fabric::edm_to_downstream_noc>;
    static_assert(num_slots > 0 && num_slots <= 255 && num_slots <= stream_count_mask);
    // The source channel of a packet is read from its header (mux mode, which fixes it, is not brought up).
    static_assert(!skip_src_ch_id_update);
};

// Where a receiver channel is in its ring (loop locals).
struct Position {
    uint32_t available;  // packets this channel may process (present, and acked under first-level acks), not retired
    uint32_t slot;       // header address of the next slot
    uint32_t index;      // ring index of the next slot
};

// This router's coordinates in its mesh (2D routing).
struct Coordinates {
    uint8_t y;
    uint8_t x;
};

// A receiver channel's state across passes is kept as separate objects, not one aggregate: the per-packet counters
// (Position, CompletionCounts) are small and stay register candidates, while the state indexed by ring slot (slot
// records, transaction IDs) and the downstream cursors live in their own objects.

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

// A 1D sparse multicast that goes on past this chip: this chip writes its own page group (counts[chip_idx] pages from
// write_idx), write_idx and chip_idx advance so the next chip picks the next group, then the forward carries the
// advanced header. The write reads the indices before the advance and the forward reads the header after it, so for
// this send type the local write comes first (as on main). Out of line: rare. Returns the slot's read bits.
template <size_t CH>
__attribute__((noinline)) uint32_t sparse_multicast_hop(
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
    return SlotRecord::READ_LOCAL | SlotRecord::READ_FORWARD;
}

// First-level acks. A packet that has arrived is acked to the far sender before it is processed, and the far sender
// returns its slot to the producer on the ack. Every packet that has arrived since the last pass is acked: its sender
// channel's ack count goes up (the frame goes out with the pass's other credits), and the packets-received register is
// consumed here rather than at retire. The packets acked are the oldest not yet acked: `available` slots past the next
// one to process. Returns whether it acked any.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool acknowledge_arrivals(Position& p, CreditSender& credits) {
    using C = Channel<CH>;
    const uint32_t arrived = read_stream_count<C::packets_received_stream>();
    if (arrived == 0) {
        return false;
    }
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    increment_local_update_ptr_val<C::packets_received_stream>(-static_cast<int32_t>(arrived));
    uint32_t index = ring_add<C::num_slots>(p.index, p.available);
    for (uint32_t k = 0; k < arrived; k++) {
        const auto* header =
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(C::first_slot + index * channel_buffer_size);
        add_ack(credits, header->src_ch_id);
        index = ring_add<C::num_slots>(index, 1);
    }
    p.available += arrived;
    return true;
}

// One packet, when one is present and its downstreams have room. Returns whether it handled one.
template <size_t CH>
FORCE_INLINE bool receive(
    Position& p,
    CompletionCounts& counts,
    SlotRecords<Channel<CH>::num_slots>& records,
    typename Channel<CH>::TridTracker& trids,
    Downstreams<Channel<CH>::downstream_vc>& downstreams,
    const Coordinates& me) {
    using C = Channel<CH>;
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
    const auto fields = C::Routing::read(header, me.y, me.x);
    const auto route = C::Routing::decide(fields);
    if (route.forwards() && !downstreams.admit(route.forward_to())) {
        return false;
    }

    channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
    const uint8_t trid =
        trids.update_buffer_slot_to_next_trid_and_advance_trid_counter(tt::tt_fabric::BufferIndex{static_cast<uint8_t>(p.index)});
    const uint32_t size = word.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
    uint32_t record = word.src_ch_id;
    if (!is_2d && route.delivers() && route.forwards() &&
        word.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) {
        record |= sparse_multicast_hop<CH>(
            header, word.payload_size_bytes, fields, route.forward_to(), downstreams, p.slot, size, trid);
    } else {
        // Forward first (the next router starts sooner), then write here.
        if (route.forwards()) {
            C::Routing::before_forward(header, fields);
            downstreams.forward_all(route.forward_to(), p.slot, size, trid);
            record |= SlotRecord::READ_FORWARD;
        }
        if (route.delivers()) {
            record |= deliver<CH>(header, word.payload_size_bytes, word.noc_send_type, trid);
        }
    }
    slot_retired(counts, records, p.index, record);
    retire<CH>(p);
    return true;
}

// The receiver loop: rounds of passes over the serviced receiver channels until termination. A function of its own:
// its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
//
// The downstream connections are passed in: forwards use them, but packets delivered here do not, so they live in a
// table in memory rather than in the registers every packet needs (RESULTS §98, v4 bring-up).
__attribute__((noinline)) void run(Downstreams<Channel<0>::downstream_vc>& downstreams0) {
    using C0 = Channel<0>;
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x};
    auto credit_senders = init_receiver_channel_response_credit_senders<NUM_RECEIVER_CHANNELS>();

    Position position0{0, C0::first_slot, 0};
    CompletionCounts counts0;
    SlotRecords<C0::num_slots> records0{};
    typename C0::TridTracker trids0;
    trids0.init();
    local_write_init();

    auto& credits0 = credit_senders.template get<0>();
    bool frame_due = false;  // the credit counts changed since the last frame

    RoundCounters round;
    while (next_round()) {
        for (size_t pass = 0; pass < iterations_between_ctx_switch_and_teardown_checks; pass++) {
            NamedProfiler<
                CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
                code_profiling_enabled_timers_bitfield,
                code_profiling_buffer_base_addr>
                timer;
            if constexpr (C0::first_level_ack) {
                frame_due |= acknowledge_arrivals<0>(position0, credits0);
            }
            timer.set_should_dump(position0.available != 0);
            timer.open();
            const bool progress = receive<0>(position0, counts0, records0, trids0, downstreams0, me);
            timer.close();
            frame_due |= release_completions<C0::completion_batch>(
                counts0, records0, position0.index, progress, trids0, credits0);
            if (frame_due) {
                frame_due = !send_credit_frame(credits0);
            }
        }
        end_round(round, true);
    }
}

}  // namespace fabric_router::receiver
