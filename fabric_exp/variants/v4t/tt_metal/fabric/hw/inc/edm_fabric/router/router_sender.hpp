// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender program: sends packets from this router's sender channels over the Ethernet link into the far router's
// receiver channel of the same VC, and returns each slot to its producer once the far router completes it.
//
// Sender channels. Channel 0 of a VC is the worker channel: a worker on this chip connects to it, writes packets into
// its ring and gets credits back in its L1. The others are forwarding channels: the router on this chip facing another
// direction writes into them and gets credits back in its free-slots register.
//
// One pass:
//   1. read the free-slots registers of the VC's channels, as one group
//   2. worker channel: send if it has a packet and the far ring has room; take completions and credit the worker;
//      check the worker connection (every pass, by rule)
//   3. forwarding channels: when none has a packet or a packet in flight, one test skips them all; otherwise visit
//      each that has work
// The pass makes no calls while it sends, completes and takes upstream credits (LEDGER E0-11: forwarded 16 B 119.4 ->
// 112.3); the worker credit, once per completion batch, is out of line.
//
// The loop function takes no closures. A for_each_bit lambda capturing the loop's state by reference is an object
// holding pointers to it; whether the compiler dissolves it depends on the code around it, and when it does not, every
// test in the pass loads through those pointers and the captured state cannot live in registers (an unrelated edit to
// send once tipped it: 2D 3-hop 162 -> 188; with no closures 150, RESULTS §98). Its helpers recurse over their channel
// mask instead, with their inputs as arguments.
//
// State. Changes every pass, as locals of the loop function: the worker channel's counts and the far ring's free slots
// and next slot. Changes only when a channel is visited: each forwarding channel's record (its counts and slot) and the
// mask of forwarding channels with packets in flight, in a table read and written once per visit. Set at connection
// time: the producers' credit addresses, in the records. Everything else (slot bounds, registers, counters) is a
// compile-time constant.

#include "router_common.hpp"
#include "router_round.hpp"

namespace fabric_router::sender {

constexpr uint8_t txq = sender_txq_id;

// Sender channel CH.
template <size_t CH>
struct Channel {
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t free_slots_stream = sender_free_slots_stream<CH>;
    // Counters the far receiver writes over the link.
    static constexpr uint32_t completion_counter = to_sender_remote_completion_counters_base_address + CH * 4;
    static constexpr uint32_t ack_counter = to_sender_remote_ack_counters_base_address + CH * 4;
    static constexpr size_t vc = CH < VC1_LOCAL_CHANNEL_START ? 0 : 1;
    static constexpr size_t far_receiver = vc == 0 ? VC0_RECEIVER_CHANNEL : VC1_RECEIVER_CHANNEL;
    static constexpr bool first_level_ack = vc == 0 ? ENABLE_FIRST_LEVEL_ACK_VC0 : ENABLE_FIRST_LEVEL_ACK_VC1;
    // Injection channels on a deadlock-avoiding ring send only with two free far slots (bubble flow control).
    static constexpr uint32_t far_slots_needed =
        (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel[CH])
            ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS
            : 1;
    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
};

// The serviced sender channels of each VC, one bit per channel; channel 0 of VC0 is the worker channel.
constexpr uint32_t channels_in(size_t first, size_t count) {
    uint32_t mask = 0;
    for (size_t ch = first; ch < first + count && ch < NUM_SENDER_CHANNELS; ch++) {
        mask |= is_sender_channel_serviced[ch] ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr uint32_t vc0_channels = channels_in(0, ACTUAL_VC0_SENDER_CHANNELS);
constexpr uint32_t vc0_forwarding = vc0_channels & ~0x1u;
constexpr bool has_worker = (vc0_channels & 0x1u) != 0;

// A sender channel's addresses, as a send uses them.
struct SenderChannel {
    uint32_t first_slot;          // its ring
    uint32_t last_slot;
    uint32_t completion_counter;  // the counter the far receiver writes over the link
    uint32_t free_slots_stream;   // gets one slot back per packet sent
    uint32_t id;                  // stamped into each header (the far receiver credits this channel)
};

template <size_t CH>
constexpr SenderChannel sender_channel = {
    Channel<CH>::first_slot,
    Channel<CH>::last_slot,
    Channel<CH>::completion_counter,
    Channel<CH>::free_slots_stream,
    CH};

// The far router's receiver ring of a VC: its slots and its packets-received register (on the far router).
template <size_t RX>
struct FarRingTraits {
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[RX];
    static constexpr uint32_t first_slot = eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[RX]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[RX];
};

// The far ring as this router fills it.
struct FarRing {
    uint32_t free_slots;
    uint32_t next_slot;
};

// A forwarding channel's record.
struct ForwardingRecord {
    uint32_t slot;       // next slot to send from
    uint32_t sent;       // packets sent
    uint32_t completed;  // completions taken
    uint32_t acked;      // first-level acks taken (first-level-ack VCs only)
    uint64_t upstream_credit_addr;  // the upstream router's free-slots register, cached when it connected
};

// The forwarding channels' records, and which of them have packets not yet completed (or acked), which the pass reads
// to skip idle channels. Both change only when a channel is visited, so they live in this table, not in registers.
struct ForwardingTable {
    ForwardingRecord records[NUM_SENDER_CHANNELS];  // indexed by channel; the worker channel's entry is unused
    uint32_t in_flight = 0;
};

// The worker channel's state.
struct Worker {
    uint32_t slot;
    uint32_t sent;
    uint32_t completed;
    uint32_t acked;
    uint32_t expected_connection;  // the connection semaphore's value while nothing changes
};

// ---------------------------------------------------------------------------------------------------------------------
// Free-slots registers
// ---------------------------------------------------------------------------------------------------------------------

// The free-slots registers of the channels in MASK, into raw[] (indexed by channel), lowest channel first.
template <uint32_t MASK>
FORCE_INLINE void read_free_slots(uint32_t* raw) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        raw[CH] = read_stream_raw<Channel<CH>::free_slots_stream>();
        read_free_slots<without_lowest_bit(MASK)>(raw);
    }
}

// The channels in MASK whose raw[] free-slots value is below their ring size: their producers filled a slot.
template <uint32_t MASK>
FORCE_INLINE uint32_t filled(const uint32_t* raw) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        return (static_cast<uint32_t>((raw[CH] & stream_count_mask) != Channel<CH>::num_slots) << CH) |
               filled<without_lowest_bit(MASK)>(raw);
    } else {
        return 0;
    }
}

// The channels in MASK that have a packet to send (bit per channel). The registers are read as one group: register
// reads overlap while their values are not yet used, so the group costs about one read. The loads are volatile and kept
// in order between two fences. The result is one word, so only it stays live across the pass, not every register value.
template <uint32_t MASK>
FORCE_INLINE uint32_t channels_with_packets() {
    uint32_t raw[MASK == 0 ? 1 : 32 - __builtin_clz(MASK)];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    read_free_slots<MASK>(raw);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return filled<MASK>(raw);
}

// ---------------------------------------------------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------------------------------------------------

FORCE_INLINE bool txq_busy(uint32_t status) { return ((status >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 1u) != 0; }

// The TX queue status, read as eth_txq_is_busy reads it (a CMD read first, BH-55).
FORCE_INLINE uint32_t read_txq_status() {
    (void)eth_txq_reg_read(txq, ETH_TXQ_CMD);
    return eth_txq_reg_read(txq, ETH_TXQ_STATUS);
}

// Sends channel CH's next packet. Precondition: it has a packet and the far ring has room. Returns false when the TX
// queue was busy (nothing sent; the next pass tries again); after a send, `completions` holds the completions reported
// since `completed`, read right after the send.
//   - read the payload size and the TX queue status as one group, and stamp the packet's source channel;
//     a busy queue yields;
//   - send the packet into the far ring's next slot, advance both slots;
//   - read the queue status and the channel's completion counter as one group while the free counts update;
//   - increment the far receiver's packets-received register (after the data, on the same queue).
template <typename FarT>
FORCE_INLINE bool send(
    const SenderChannel c,
    uint32_t& slot,
    FarRing& far,
    uint32_t& sent,
    uint32_t completed,
    uint32_t& completions,
    bool& attempted) {
    const uint32_t src = slot;
    auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src);

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t payload_size = header->get_payload_size_excluding_header();
    const uint32_t status = read_txq_status();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if constexpr (!skip_src_ch_id_update) {
        header->src_ch_id = static_cast<uint8_t>(c.id);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (txq_busy(status)) {
        attempted = true;
        return false;
    }
    const uint32_t size = payload_size + sizeof(PACKET_HEADER_TYPE);
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(c.id);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(c.id, static_cast<uint16_t>(size));
    }

    internal_::eth_send_packet_bytes_unsafe(txq, src, far.next_slot, size);
    far.next_slot = far.next_slot == FarT::last_slot ? FarT::first_slot : far.next_slot + channel_buffer_size;
    slot = src == c.last_slot ? c.first_slot : src + channel_buffer_size;

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t status_after = read_txq_status();
    const uint32_t reported = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.completion_counter);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    far.free_slots--;
    increment_local_update_ptr_val(c.free_slots_stream, 1);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (txq_busy(status_after)) {
        while (internal_::eth_txq_is_busy(txq)) {
        }
    }
    remote_update_ptr_val<FarT::packets_received_stream, txq>(1);
    sent++;
    completions = reported - completed;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Credits back to producers
// ---------------------------------------------------------------------------------------------------------------------

// The worker credit goes back once per completion batch from the far router (every few packets), not per packet, and
// is a NoC write of about ten command-buffer register writes. Out of line, its constants are built when a credit goes
// out instead of being held in the sender loop's registers for its whole run (2D 1-hop writes 141.8 -> 130.5, RESULTS
// §98).

// To the worker: the absolute count of slots returned, written into the worker's L1 while it is connected and into the
// connection info for the next worker otherwise.
template <typename WorkerInterface>
__attribute__((noinline)) void credit_worker(const WorkerInterface& interface, uint32_t returned, bool connected) {
    if (connected) {
        noc_inline_dw_write<InlineWriteDst::L1, true, enable_read_counter_update_noc_flush>(
            interface.cached_worker_semaphore_address,
            returned,
            0xf,
            tt::tt_fabric::worker_handshake_noc,
            NOC_UNICAST_WRITE_VC,
            notify_worker_of_read_counter_update_src_address);
    } else {
        reinterpret_cast<volatile tt_l1_ptr EDMChannelWorkerLocationInfo*>(local_sender_channel_0_connection_info_addr)
            ->edm_read_counter = returned;
    }
}

// To an upstream router: +n on its free-slots register.
FORCE_INLINE void credit_upstream(uint64_t credit_addr, uint32_t n) {
    noc_inline_dw_write<InlineWriteDst::REG, true>(
        credit_addr, pack_value_for_inc_on_write_stream_reg_write(n), 0xf, tt::tt_fabric::worker_handshake_noc);
}

// The worker connection semaphore changed (open or close request): handle it out of line and return the value it
// holds while nothing changes. The interface's read counter is set to the credits returned so far.
template <typename WorkerInterface>
__attribute__((noinline)) uint32_t worker_connection_event(
    WorkerInterface& interface, uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    interface.local_read_counter.counter = returned;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            establish_worker_connection<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(interface);
            return open_connection_value;
        }
        return unused_connection_value;
    }
    if (semaphore == close_connection_request_value) {
        interface.template teardown_worker_connection<true, ENABLE_RISC_CPU_DATA_CACHE>();
        return unused_connection_value;
    }
    return open_connection_value;
}

// ---------------------------------------------------------------------------------------------------------------------
// Visits
// ---------------------------------------------------------------------------------------------------------------------

// The worker channel: send, take completions (and first-level acks), credit the worker, check the connection.
template <size_t RX, typename WorkerInterface>
FORCE_INLINE void visit_worker(
    Worker& w,
    WorkerInterface& interface,
    volatile tt_l1_ptr uint32_t* connection_semaphore,
    bool has_packet,
    FarRing& far,
    bool& attempted) {
    using C = Channel<0>;
    using namespace tt::tt_fabric::connection_interface;
    uint32_t completions = 0;
    bool sent = false;
    if (far.free_slots >= C::far_slots_needed && has_packet) {
        sent = send<FarRingTraits<RX>>(sender_channel<0>, w.slot, far, w.sent, w.completed, completions, attempted);
    }
    if (!sent && w.sent != w.completed) {
        completions = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::completion_counter) - w.completed;
    }
    const bool connected = w.expected_connection != unused_connection_value;
    if (completions != 0) {
        far.free_slots += completions;
        w.completed += completions;
        if constexpr (!C::first_level_ack) {
            credit_worker(interface, w.completed, connected);
        }
    }
    if constexpr (C::first_level_ack) {
        if (w.sent != w.acked) {
            const uint32_t acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter) - w.acked;
            if (acks != 0) {
                w.acked += acks;
                credit_worker(interface, w.acked, connected);
            }
        }
    }
    // The connection: one load and compare every pass; open and close out of line.
    const uint32_t semaphore = *connection_semaphore;
    if (semaphore != w.expected_connection) {
        w.expected_connection = worker_connection_event(
            interface, semaphore, w.expected_connection, C::first_level_ack ? w.acked : w.completed);
    }
}

// A forwarding channel with work: send if it has a packet and the far ring has room, take completions (and acks),
// credit the upstream router, and keep the channel's bit in `in_flight` equal to "packets not yet completed".
template <size_t CH, size_t RX>
FORCE_INLINE void visit_forwarding(
    ForwardingTable& table, bool has_packet, FarRing& far, bool& attempted) {
    ForwardingRecord& r = table.records[CH];
    using C = Channel<CH>;
    uint32_t sent = r.sent;
    uint32_t completed = r.completed;
    uint32_t completions = 0;
    bool did_send = false;
    if (far.free_slots >= C::far_slots_needed && has_packet) {
        uint32_t slot = r.slot;
        did_send = send<FarRingTraits<RX>>(sender_channel<CH>, slot, far, sent, completed, completions, attempted);
        if (did_send) {
            r.slot = slot;
            r.sent = sent;
        }
    }
    if (!did_send && sent != completed) {
        completions = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::completion_counter) - completed;
    }
    if (completions != 0) {
        far.free_slots += completions;
        completed += completions;
        r.completed = completed;
        if constexpr (!C::first_level_ack) {
            credit_upstream(r.upstream_credit_addr, completions);
        }
    }
    bool busy = sent != completed;
    if constexpr (C::first_level_ack) {
        if (sent != r.acked) {
            const uint32_t acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter) - r.acked;
            if (acks != 0) {
                r.acked += acks;
                credit_upstream(r.upstream_credit_addr, acks);
            }
        }
        busy = busy || sent != r.acked;
    }
    table.in_flight = busy ? (table.in_flight | (1u << CH)) : (table.in_flight & ~(1u << CH));
}

// ---------------------------------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------------------------------

// Visits the forwarding channels in MASK that have a packet (`pending`) or packets in flight (`in_flight`), lowest
// channel first.
template <uint32_t MASK, size_t RX>
FORCE_INLINE void visit_forwarding_channels(
    ForwardingTable& table, uint32_t pending, uint32_t in_flight, FarRing& far, bool& attempted) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((pending >> CH) & 1u) != 0;
        if (has_packet || ((in_flight >> CH) & 1u) != 0) {
            visit_forwarding<CH, RX>(table, has_packet, far, attempted);
        }
        visit_forwarding_channels<without_lowest_bit(MASK), RX>(table, pending, in_flight, far, attempted);
    }
}

// The producers' credit addresses of the forwarding channels, cached when each upstream router connected.
struct UpstreamCredits {
    uint64_t addr[NUM_SENDER_CHANNELS];
};

// The forwarding channels in MASK: each one's record at its first slot with nothing sent, and its producer's credit
// address.
template <uint32_t MASK>
FORCE_INLINE void init_forwarding(ForwardingTable& table, const UpstreamCredits& upstream) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        table.records[CH] = {Channel<CH>::first_slot, 0, 0, 0, upstream.addr[CH]};
        init_forwarding<without_lowest_bit(MASK)>(table, upstream);
    }
}

// Packets sent so far on the forwarding channels in MASK.
template <uint32_t MASK>
FORCE_INLINE uint32_t forwarded(const ForwardingTable& table) {
    if constexpr (MASK != 0) {
        return table.records[lowest_bit(MASK)].sent + forwarded<without_lowest_bit(MASK)>(table);
    } else {
        return 0;
    }
}

// Packets sent so far on every VC0 channel (a round attempted a send if this changed or a send yielded; read by the idle
// context switch only).
FORCE_INLINE uint32_t total_sent(const Worker& worker, const ForwardingTable& table) {
    return worker.sent + forwarded<vc0_forwarding>(table);
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds. A function
// of its own: its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
template <typename WorkerInterfaces>
__attribute__((noinline)) void run(WorkerInterfaces& interfaces, const UpstreamCredits& upstream) {
    using namespace tt::tt_fabric::connection_interface;
    using FarVc0 = FarRingTraits<VC0_RECEIVER_CHANNEL>;
    // The loop's own copy of the worker channel's interface (connection events and the worker's credit address): after
    // init nothing else uses it, and a copy keeps the loop from holding a pointer to the caller's for its whole run.
    auto worker_interface = interfaces.template get<0>();
    volatile tt_l1_ptr uint32_t* const connection_semaphore = worker_interface.connection_live_semaphore;

    FarRing far_vc0{FarVc0::num_slots, FarVc0::first_slot};
    Worker worker{Channel<0>::first_slot, 0, 0, 0, unused_connection_value};
    ForwardingTable forwarding;
    init_forwarding<vc0_forwarding>(forwarding, upstream);

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        while (next_round()) {
            round.attempted = false;
            round.sent_at_round_start = total_sent(worker, forwarding);
            for (size_t pass = 0; pass < iterations_between_ctx_switch_and_teardown_checks; pass++) {
                const uint32_t pending = channels_with_packets<vc0_channels>();
                if constexpr (has_worker) {
                    visit_worker<VC0_RECEIVER_CHANNEL>(
                        worker, worker_interface, connection_semaphore, (pending & 0x1u) != 0, far_vc0, round.attempted);
                }
                if constexpr (vc0_forwarding != 0) {
                    // When no forwarding channel has a packet or a packet in flight (every pass on a router whose
                    // traffic all comes from its worker), one test skips them all.
                    const uint32_t in_flight = forwarding.in_flight;
                    if (((pending | in_flight) & vc0_forwarding) != 0) {
                        visit_forwarding_channels<vc0_forwarding, VC0_RECEIVER_CHANNEL>(
                            forwarding, pending, in_flight, far_vc0, round.attempted);
                    }
                }
            }
            end_round(
                round, !IDLE_CONTEXT_SWITCHING || round.attempted || total_sent(worker, forwarding) != round.sent_at_round_start);
        }
    }
}

}  // namespace fabric_router::sender
