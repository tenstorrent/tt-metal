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
//   3. forwarding channels: when none has a packet or a packet in flight, one test skips them all; otherwise those
//      with a packet send (lowest channel first), then each with work takes completions and credits its producer
//
// One send. What differs between sender channels is of two kinds. Behavior (worker or forwarding channel, the VC's far
// ring, first-level acks, bubble flow control) is compile-time, a template parameter. Addresses (the ring, the
// completion counter, the free-slots register, the id stamped into headers) are values: a SenderChannel. The worker
// channel passes its SenderChannel as compile-time constants, which fold into immediates; the forwarding channels send
// through one send, reached by a run-time loop over the channels with a packet, their SenderChannel in their records.
// Their completions and credits stay per-channel code with compile-time addresses: they run for every channel with
// packets in flight on every pass, where run-time addressing would put a dependent load chain on each completion check
// (RESULTS §98, v4s: multicast +18). (Templated on the channel, the send was inlined once per channel: four copies of
// ~70 instructions differing in those few values.)
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
// mask of forwarding channels with packets in flight, in a table read and written once per visit. Set at init and
// connection time: each forwarding channel's SenderChannel and its producer's credit address, in the records.
// Everything else is a compile-time constant.

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

// A sender channel's addresses, as a send and a visit use them.
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

// The forwarding channels of VC0 share one visit, so they must behave alike: one VC, and no injection channel among
// them on a deadlock-avoiding ring (bubble flow control).
constexpr bool forwarding_channels_alike() {
    for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
        if (((vc0_forwarding >> ch) & 1u) != 0 &&
            (ch >= VC1_LOCAL_CHANNEL_START || (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel[ch]))) {
            return false;
        }
    }
    return true;
}
static_assert(forwarding_channels_alike());
constexpr size_t first_forwarding_channel = vc0_forwarding == 0 ? 1 : lowest_bit(vc0_forwarding);

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
    SenderChannel channel;          // set at init
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

// Sends the next packet of channel `c`. Precondition: it has a packet and the far ring has room. Returns false when
// the TX queue was busy (nothing sent; the next pass tries again). With READS_COMPLETIONS (the worker channel),
// `reported` is the channel's completion counter, read right after the send; the forwarding channels read theirs as a
// group after all their sends.
//   - read the payload size and the TX queue status as one group, and stamp the packet's source channel;
//     a busy queue yields;
//   - send the packet into the far ring's next slot, advance both slots;
//   - read the queue status and the channel's completion counter as one group while the free counts update;
//   - increment the far receiver's packets-received register (after the data, on the same queue).
template <typename FarT, bool READS_COMPLETIONS>
FORCE_INLINE bool send(
    const SenderChannel c, uint32_t& slot, FarRing& far, uint32_t& sent, uint32_t& reported, bool& attempted) {
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
    if constexpr (READS_COMPLETIONS) {
        reported = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.completion_counter);
    }
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
        uint32_t reported;
        sent = send<FarRingTraits<RX>, true>(sender_channel<0>, w.slot, far, w.sent, reported, attempted);
        if (sent) {
            completions = reported - w.completed;
        }
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

// The forwarding channels with a packet (`ready`), lowest channel first, send while the far ring has room and the TX
// queue takes them. One send serves them all (see the file header): each channel's addresses come from its record.
template <size_t RX>
FORCE_INLINE void send_forwarded(ForwardingTable& table, uint32_t ready, FarRing& far, bool& attempted) {
    using C = Channel<first_forwarding_channel>;  // their behavior (forwarding_channels_alike)
    for (; ready != 0 && far.free_slots >= C::far_slots_needed; ready &= ready - 1) {
        ForwardingRecord& r = table.records[__builtin_ctz(ready)];
        uint32_t unused;
        if (!send<FarRingTraits<RX>, false>(r.channel, r.slot, far, r.sent, unused, attempted)) {
            return;  // the TX queue is busy: the next pass tries again
        }
    }
}

// The completion counters of the forwarding channels in MASK, into reported[] (indexed by channel), lowest first.
template <uint32_t MASK>
FORCE_INLINE void read_completion_counters(uint32_t* reported) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        reported[CH] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(Channel<CH>::completion_counter);
        read_completion_counters<without_lowest_bit(MASK)>(reported);
    }
}

// Forwarding channel CH's completions (and acks), `reported` its completion counter: return them to the far ring,
// credit the upstream router, and keep the channel's bit in `in_flight` equal to "packets not yet completed".
template <size_t CH>
FORCE_INLINE void complete_forwarded(ForwardingTable& table, uint32_t reported, FarRing& far) {
    using C = Channel<CH>;
    ForwardingRecord& r = table.records[CH];
    const uint32_t completions = reported - r.completed;
    if (completions != 0) {
        far.free_slots += completions;
        r.completed = reported;
        if constexpr (!C::first_level_ack) {
            credit_upstream(r.upstream_credit_addr, completions);
        }
    }
    bool busy = r.sent != r.completed;
    if constexpr (C::first_level_ack) {
        if (r.sent != r.acked) {
            const uint32_t acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter) - r.acked;
            if (acks != 0) {
                r.acked += acks;
                credit_upstream(r.upstream_credit_addr, acks);
            }
        }
        busy = busy || r.sent != r.acked;
    }
    table.in_flight = busy ? (table.in_flight | (1u << CH)) : (table.in_flight & ~(1u << CH));
}

template <uint32_t MASK>
FORCE_INLINE void complete_forwarding_channels(ForwardingTable& table, uint32_t work, const uint32_t* reported, FarRing& far) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        if (((work >> CH) & 1u) != 0) {
            complete_forwarded<CH>(table, reported[CH], far);
        }
        complete_forwarding_channels<without_lowest_bit(MASK)>(table, work, reported, far);
    }
}

// The forwarding channels with a packet (`pending`) or packets in flight (`in_flight`): first the sends, through the
// one send; then, per channel, completions and credits, their completion counters read as one group.
template <size_t RX>
FORCE_INLINE void visit_forwarding_channels(
    ForwardingTable& table, uint32_t pending, uint32_t in_flight, FarRing& far, bool& attempted) {
    send_forwarded<RX>(table, pending & vc0_forwarding, far, attempted);
    uint32_t reported[32 - __builtin_clz(vc0_forwarding | 1u)];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    read_completion_counters<vc0_forwarding>(reported);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    complete_forwarding_channels<vc0_forwarding>(table, (pending | in_flight) & vc0_forwarding, reported, far);
}

// The producers' credit addresses of the forwarding channels, cached when each upstream router connected.
struct UpstreamCredits {
    uint64_t addr[NUM_SENDER_CHANNELS];
};

// The forwarding channels in MASK: each one's record at its first slot with nothing sent, its producer's credit
// address and its addresses.
template <uint32_t MASK>
FORCE_INLINE void init_forwarding(ForwardingTable& table, const UpstreamCredits& upstream) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        table.records[CH] = {Channel<CH>::first_slot, 0, 0, 0, upstream.addr[CH], sender_channel<CH>};
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
                        visit_forwarding_channels<VC0_RECEIVER_CHANNEL>(
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
