// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The receiver program: takes packets off the Ethernet link from the far router's sender channels, and for each one
// forwards it to the next router on this chip, writes it to this chip, or both, then returns the slot's credit.
//
// One step per receiver channel per pass (VC0's channel, then VC1's on multi-mesh routers that service VC1):
//   ack       first-level-ack VCs (rings and tori): ack every packet that has arrived (acknowledge_arrivals)
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
//   release   completion credits back to the far sender (router_completions.hpp)
// VC2's receiver channel (neighbour exchange) only delivers: every packet is for this chip, so it skips decode and admit.
// And once per pass, for all channels:
//   frame     one credit frame, when the pass's acks or completions changed the counts
//
// State. VC0's channel position and counts, its completion state and its downstream cursors are locals of the loop
// function; VC1's channel state is one record (ChannelState) the loop keeps and its step takes by reference. Ring
// bounds, stream ids and command buffers are compile-time constants.

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
    // Slots per completion group (router_completions.hpp). Under first-level acks one at a time, as the generic loop
    // does: grouped there, completion credits were held, which stalls bubble flow control (RESULTS §98, v4f2, v4p).
    static constexpr uint32_t completion_group = first_level_ack ? 1 : completion_group_size<num_slots>;
    // The transaction IDs of the slots' NoC commands: one per completion group, from the channel's own range.
    using Ids = GroupIds<completion_group, RX_CH_TRID_STARTS[CH], NUM_TRANSACTION_IDS>;
    // Credits go back as counts in one frame (router_completions.hpp); stream-register credits are not brought up.
    static_assert(receiver_channel_uses_counter_credits(CH));
    // The speedy path (the host's ENABLE_SPEEDY_VC0: a router whose VC0 carries only its worker's packets to the next
    // chip, e.g. neighbour exchange): VC0's receiver channel only delivers, as main's speedy receiver (the host turns
    // its forwarding off).
    static constexpr bool delivers_only = CH == 0 && super_speedy_mode;
    // Elsewhere the host turns a channel's forwarding off when a channel-trimming capture saw no packet on its VC
    // forwarded or written here (DISABLE_RX_CHx_FORWARDING): as main's generic receiver, a packet that still arrives is
    // consumed and credited, not forwarded or written. (VC2's receiver channel only delivers; the host always sets it.)
    static constexpr bool drops = is_receiver_channel_forwarding_disabled[CH] && !delivers_only &&
                                  !(ACTUAL_VC2_SENDER_CHANNELS > 0 && CH == VC2_RECEIVER_CHANNEL);
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

// Where a receiver channel is in its ring (loop locals).
struct Position {
    uint32_t available;  // packets this channel may process (present, and acked under first-level acks), not retired
    uint32_t slot;       // header address of the next slot
    uint32_t index;      // ring index of the next slot
};

// VC0's receiver channel state across passes is kept as separate objects, not one aggregate: the per-packet counters
// (Position, CompletionCounts) are small and stay register candidates, while the state indexed by ring slot (slot
// records) and the downstream cursors live in their own objects.
//
// VC1's is one record. E1's loop is at its register limit with VC0's state, and a second channel's counters as more
// loop locals would compete with VC0's (two more loop-carried values cost VC0 +8 cycles/packet, RESULTS §98 v4p), so
// VC1's step takes its state by reference. VC1 may pay a few cycles for it; VC0 should not (design note §4.3).
template <size_t CH>
struct ChannelState {
    Position position{0, Channel<CH>::first_slot, 0};
    CompletionCounts counts;
    SlotRecords<Channel<CH>::num_slots> records{};
};

// Each receiver channel's state where this router runs the channel (configuration, channel trimming), absent elsewhere
// (Present, router_common.hpp): VC0's as its three objects, VC1's and VC2's as a record each.
using Vc0Position = Present<runs_vc0_receiver, Position, 0>;
using Vc0Counts = Present<runs_vc0_receiver, CompletionCounts, 10>;
using Vc0Records = Present<runs_vc0_receiver, SlotRecords<Channel<0>::num_slots>, 20>;
using Vc1State = Present<runs_vc1_receiver, ChannelState<1>, 1>;
using Vc2State = Present<runs_vc2_receiver, ChannelState<VC2_RECEIVER_CHANNEL>, 2>;

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
// returns its slot to the producer on the ack. Every packet that has arrived since the last pass is acked: its sender
// channel's ack count goes up (the frame goes out with the pass's other credits), and the packets-received register is
// consumed here rather than at retire. The packets acked are the oldest not yet acked: `available` slots past the next
// one to process. `arrived` is the packets-received count, read at the start of the pass. Returns whether it acked any.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool acknowledge_arrivals(Position& p, CreditSender& credits, uint32_t arrived) {
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

// A receiver channel whose packets are all for this chip: VC2's (neighbour exchange; the host turns its forwarding off),
// and VC0's on a speedy-path router (Channel::delivers_only). No route to decode and nothing to admit. Returns whether
// it handled one. Without DELIVERS, a channel whose forwarding is off (Channel::drops): the packet is consumed only.
template <size_t CH, bool DELIVERS = true>
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
    if constexpr (DELIVERS) {
        deliver<CH>(header, word.payload_size_bytes, word.noc_send_type, C::Ids::of(p.index));
    }
    slot_retired(counts, records, p.index, source_channel(word.src_ch_id));
    retire<CH>(p);
    return true;
}

// One packet, when one is present and its downstreams have room. Returns whether it handled one.
template <size_t CH>
FORCE_INLINE bool receive(
    Position& p,
    CompletionCounts& counts,
    SlotRecords<Channel<CH>::num_slots>& records,
    Downstreams<Channel<CH>::downstream_vc>& downstreams,
    const Coordinates& me) {
    using C = Channel<CH>;
    if constexpr (C::delivers_only) {
        return receive_local<CH>(p, counts, records);
    } else if constexpr (C::drops) {
        return receive_local<CH, false>(p, counts, records);
    }
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
        sparse_multicast_hop<CH>(header, word.payload_size_bytes, fields, route.forward_to(), downstreams, p.slot, size, trid);
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

// One pass of VC2's receiver channel: at most one packet, then its completions. Returns whether the credit counts
// changed.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool deliver_and_complete(ChannelState<CH>& s, CreditSender& credits) {
    using C = Channel<CH>;
    const bool progress = receive_local<CH>(s.position, s.counts, s.records);
    return release_completions<C::completion_group, C::num_slots, typename C::Ids>(
        s.counts, s.records, s.position.index, progress, credits);
}

// One pass of a receiver channel kept in a record (VC1's): at most one packet, then its completions. Returns whether
// the credit counts changed.
template <size_t CH, typename CreditSender>
FORCE_INLINE bool receive_and_complete(
    ChannelState<CH>& s, Downstreams<Channel<CH>::downstream_vc>& downstreams, const Coordinates& me, CreditSender& credits) {
    using C = Channel<CH>;
    const bool progress = receive<CH>(s.position, s.counts, s.records, downstreams, me);
    return release_completions<C::completion_group, C::num_slots, typename C::Ids>(
        s.counts, s.records, s.position.index, progress, credits);
}

// Whether VC0's packets-received register reads 0 (telemetry's RX heartbeat, once per round).
FORCE_INLINE bool vc0_register_empty() { return read_stream_count<Channel<0>::packets_received_stream>() == 0; }

// One pass over the serviced receiver channels: at most one packet each, their completions, and the credit frame when
// the counts changed. The body of run()'s loop, which calls it on its own locals; a single-ERISC router runs it as the
// receiver half of each pass over its state in memory (router_single.hpp).
// (The channels' state types are template parameters so that an absent channel's state is never named: Present.)
template <
    typename Position0,
    typename Counts0,
    typename Records0,
    typename VC1State,
    typename VC2State,
    typename Downstreams1>
FORCE_INLINE void pass(
    Position0& position0,
    Counts0& counts0,
    Records0& records0,
    VC1State& vc1,
    VC2State& vc2,
    ReceiverChannelCounterBasedResponseCreditSender& credits,
    bool& frame_due,
    Downstreams<Channel<0>::downstream_vc>& downstreams0,
    Downstreams1& downstreams1,
    const Coordinates& me) {
    using C0 = Channel<0>;
    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        timer;
    // First-level acks: a frame goes out most passes, so the pass's registers are read as one group, their reads
    // overlapping: the packets-received count and the TX queue's status (for the frame).
    bool txq_free = false;
    if constexpr (runs_vc0_receiver) {
        if constexpr (C0::first_level_ack) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const uint32_t arrived = read_stream_count<C0::packets_received_stream>();
            const uint32_t txq_status = read_txq_status(receiver_txq_id);
            std::atomic_signal_fence(std::memory_order_seq_cst);
            txq_free = !txq_busy(txq_status);
            frame_due |= acknowledge_arrivals<0>(position0, credits, arrived);
        }
        timer.set_should_dump(position0.available != 0);
        timer.open();
        const bool progress = receive<0>(position0, counts0, records0, downstreams0, me);
        timer.close();
        frame_due |= release_completions<C0::completion_group, C0::num_slots, typename C0::Ids>(
            counts0, records0, position0.index, progress, credits);
    }
    if constexpr (runs_vc1_receiver) {
        frame_due |= receive_and_complete(vc1, downstreams1, me, credits);
    }
    if constexpr (runs_vc2_receiver) {
        frame_due |= deliver_and_complete(vc2, credits);
    }
    if (frame_due) {
        frame_due = !send_credit_frame(credits, txq_free);
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
    using C0 = Channel<0>;
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const Coordinates me{routing_table.my_mesh_coord_y, routing_table.my_mesh_coord_x, routing_table.my_mesh_id};

    Vc0Position position0 = present<runs_vc0_receiver, 0>(Position{0, C0::first_slot, 0});
    Vc0Counts counts0;
    Vc0Records records0{};
    Vc1State vc1;
    Vc2State vc2;
    local_write_init();

    // One credit sender for every receiver channel: counter credits are one L1 block indexed by the far sender channel,
    // VC0's and VC1's sender channels are different channels, so one frame per pass carries both channels' credits.
    ReceiverChannelCounterBasedResponseCreditSender credits(0);
    bool frame_due = false;  // the credit counts changed since the last frame

    RoundCounters round;
    while (next_round()) {
        if constexpr (telemetry::any_option) {
            telemetry::round_start();
        }
        for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
            pass(position0, counts0, records0, vc1, vc2, credits, frame_due, downstreams0, downstreams1, me);
        }
        end_round(round, true);
        if constexpr (telemetry::any_option) {
            telemetry::round_end<false, true>([] { return false; }, [] { return vc0_register_empty(); });
        }
    }
}

}  // namespace fabric_router::receiver
