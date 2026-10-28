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
//   1. read the free-slots registers of VC0's channels, as one group
//   2. worker channel: send if it has a packet and the far ring has room; take completions and credit the worker;
//      check the worker connection (every pass, by rule)
//   3. forwarding channels: when none has a packet or a packet in flight, one test skips them all; otherwise visit
//      each that has work
//   4. VC1 (multi-mesh fabrics): its forwarding channels as in 3, after one group read of their registers; VC1 has no
//      worker channel, and its packets go into the far router's VC1 receiver ring
// The pass makes no calls while it sends, completes and takes upstream credits (LEDGER E0-11: forwarded 16 B 119.4 ->
// 112.3); the worker credit, once per completion batch, is out of line.
//
// The loop function takes no closures. A for_each_bit lambda capturing the loop's state by reference is an object
// holding pointers to it; whether the compiler dissolves it depends on the code around it, and when it does not, every
// test in the pass loads through those pointers and the captured state cannot live in registers (an unrelated edit to
// send once tipped it: 2D 3-hop 162 -> 188; with no closures 150, RESULTS §98). Its helpers recurse over their channel
// mask instead, with their inputs as arguments.
//
// State. Changes every pass, as locals of the loop function: the worker channel's counts and each far ring's free slots
// and next slot. Changes only when a channel is visited: each forwarding channel's record (its counts and slot) and the
// mask of forwarding channels with packets in flight (both VCs), in a table read and written once per visit. Set at
// connection time: the producers' credit addresses, in the records. Everything else (slot bounds, registers, counters)
// is a compile-time constant.

#include "router_common.hpp"
#include "router_round.hpp"

namespace fabric_router::sender {

constexpr uint8_t txq = sender_txq_id;

// Injection channels on a deadlock-avoiding ring or torus send only with two free far slots (bubble flow control): the
// worker channel, and on a torus a channel fed from the other axis (a turn).
// (It reads the untrimmed array: GCC rejects the trimmed std::array<bool, N>'s operator[] in this constant expression as
// "used before its definition".)
constexpr uint32_t far_slots_needed_by(size_t ch) {
    return (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel_all_[ch])
               ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS
               : 1;
}

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
    static constexpr uint32_t far_slots_needed = far_slots_needed_by(CH);
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
// VC1's channels are all forwarding channels (router to router; no worker connects to VC1).
constexpr uint32_t vc1_forwarding = channels_in(VC1_LOCAL_CHANNEL_START, ACTUAL_VC1_SENDER_CHANNELS);

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

// The far ring before the first send: every slot free.
template <size_t RX>
constexpr FarRing far_ring_start() {
    return {FarRingTraits<RX>::num_slots, FarRingTraits<RX>::first_slot};
}

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

// A sender channel's free-slots register counts the free slots of its ring: the producer takes one for each packet it
// writes, this router gives one back for each packet it sends. A channel holds a packet when the count is anything but
// its ring size. ("Below the ring size" would not do: a count out of range, an over-fill, must read as holding packets,
// as the generic loop reads it, or the channel is never served; LEDGER E0-10, RESULTS §93.)
template <size_t CH>
FORCE_INLINE bool holds_packet(uint32_t free_slots_register) {
    return (free_slots_register & stream_count_mask) != Channel<CH>::num_slots;
}

// The free-slots registers of the channels in MASK, into registers[CH], lowest channel first.
template <uint32_t MASK>
FORCE_INLINE void read_free_slots_registers(uint32_t* registers) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        registers[CH] = read_stream_raw<Channel<CH>::free_slots_stream>();
        read_free_slots_registers<without_lowest_bit(MASK)>(registers);
    }
}

// The channels in MASK whose free-slots register (in registers[CH]) says they hold a packet, one bit each.
template <uint32_t MASK>
FORCE_INLINE uint32_t channels_holding_packets(const uint32_t* registers) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        return (static_cast<uint32_t>(holds_packet<CH>(registers[CH])) << CH) |
               channels_holding_packets<without_lowest_bit(MASK)>(registers);
    } else {
        return 0;
    }
}

// The channels in MASK that hold a packet, one bit each. Every register is read first, as one group (register reads
// overlap while their values are unused, so the group costs about one read; the loads are volatile and kept together
// between the two fences), then tested. Only the one-word result stays live across the pass.
template <uint32_t MASK>
FORCE_INLINE uint32_t channels_with_packets() {
    uint32_t registers[MASK == 0 ? 1 : 32 - __builtin_clz(MASK)];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    read_free_slots_registers<MASK>(registers);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return channels_holding_packets<MASK>(registers);
}

// ---------------------------------------------------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------------------------------------------------

// Sends channel CH's next packet. Precondition: it has a packet and the far ring has room. Returns false when the TX
// queue was busy (nothing sent; the next pass tries again); after a send, `completions` holds the completions reported
// since `completed`, read right after the send.
//   - read the payload size and the TX queue status as one group, and stamp the packet's source channel;
//     a busy queue yields;
//   - send the packet into the far ring's next slot, advance both slots;
//   - read the queue status and the channel's completion counter as one group while the free counts update;
//   - increment the far receiver's packets-received register (after the data, on the same queue).
// The channel enters only through four values of Channel<CH>: its ring bounds (the slot advance), its completion
// counter, its free-slots register and its id (stamped into the header); the rest is the same for every channel. It is
// inlined once per channel with those values as immediates: one copy shared at run time cost forwarded packets 5 to 18
// cycles (RESULTS §98, v4s, v4s2).
template <size_t CH, typename FarT>
FORCE_INLINE bool send(
    uint32_t& slot, FarRing& far, uint32_t& sent, uint32_t completed, uint32_t& completions, bool& attempted) {
    using C = Channel<CH>;
    const uint32_t src = slot;
    auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src);

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t payload_size = header->get_payload_size_excluding_header();
    const uint32_t status = read_txq_status(txq);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if constexpr (!skip_src_ch_id_update) {
        header->src_ch_id = static_cast<uint8_t>(CH);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (txq_busy(status)) {
        attempted = true;
        return false;
    }
    const uint32_t size = payload_size + sizeof(PACKET_HEADER_TYPE);
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(CH);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(CH, static_cast<uint16_t>(size));
    }

    internal_::eth_send_packet_bytes_unsafe(txq, src, far.next_slot, size);
    far.next_slot = far.next_slot == FarT::last_slot ? FarT::first_slot : far.next_slot + channel_buffer_size;
    slot = src == C::last_slot ? C::first_slot : src + channel_buffer_size;

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t status_after = read_txq_status(txq);
    const uint32_t reported = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::completion_counter);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    far.free_slots--;
    increment_local_update_ptr_val<C::free_slots_stream>(1);
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

// The worker credit goes back once per completion batch from the far router (every few packets), or under first-level
// acks once per ack frame. Out of line, its constants are built when a credit goes out instead of being held in the
// sender loop's registers for its whole run (2D 1-hop writes 141.8 -> 130.5, RESULTS §98).

// The credit write. Blackhole emulates an inline write to L1 (the inline-write API cannot target L1 there): the value is
// stored in a scratch word and a 4-byte NoC write copies it. Here the write goes on this ERISC's write command buffer,
// which nothing else in the sender uses (its upstream credits and teardown use the atomics buffer). Its destination,
// the connected worker's credit word, holds for a whole connection, so the buffer is programmed with it when the worker
// connects and again after the base firmware has run (it may use the buffer); a credit then stores the count and
// issues the write with that state (source, address and control) instead of programming the whole command.
static_assert(!enable_read_counter_update_noc_flush);  // the scratch word may be rewritten before an earlier write read
                                                       // it: the worker then sees the newer, larger count
template <typename WorkerInterface>
__attribute__((noinline)) void preset_worker_credit(const WorkerInterface& interface) {
    noc_async_write_one_packet_set_state</*posted=*/true>(
        interface.cached_worker_semaphore_address, sizeof(uint32_t), tt::tt_fabric::worker_handshake_noc);
}

// To the worker: the absolute count of slots returned, written into the worker's L1 while it is connected and into the
// connection info for the next worker otherwise.
template <typename WorkerInterface>
__attribute__((noinline)) void credit_worker(const WorkerInterface& interface, uint32_t returned, bool connected) {
    if (connected) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(notify_worker_of_read_counter_update_src_address) = returned;
        noc_async_write_one_packet_with_state</*posted=*/true>(
            notify_worker_of_read_counter_update_src_address,
            static_cast<uint32_t>(interface.cached_worker_semaphore_address),
            tt::tt_fabric::worker_handshake_noc);
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
            preset_worker_credit(interface);
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
        sent = send<0, FarRingTraits<RX>>(w.slot, far, w.sent, w.completed, completions, attempted);
    }
    // The visit's counters, read as one group so their loads overlap: completions (unless the send read them), acks
    // (first-level-ack VCs), and the connection semaphore, which is read every pass. Completions before acks, so the
    // acks seen are never fewer than the completions seen. Whether to read each is known before any is read.
    const bool read_completions = !sent && w.sent != w.completed;
    const bool read_acks = C::first_level_ack && w.sent != w.acked;
    uint32_t reported_completions = 0;
    uint32_t reported_acks = 0;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {
        reported_completions = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::completion_counter);
    }
    if (read_acks) {
        reported_acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter);
    }
    const uint32_t semaphore = *connection_semaphore;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {
        completions = reported_completions - w.completed;
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
        if (read_acks && reported_acks != w.acked) {
            w.acked = reported_acks;
            credit_worker(interface, w.acked, connected);
        }
    }
    // The connection: one load and compare every pass; open and close out of line.
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
        did_send = send<CH, FarRingTraits<RX>>(slot, far, sent, completed, completions, attempted);
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

// Packets sent so far on every channel (a round attempted a send if this changed or a send yielded; read by the idle
// context switch only).
FORCE_INLINE uint32_t total_sent(const Worker& worker, const ForwardingTable& table) {
    return worker.sent + forwarded<vc0_forwarding | vc1_forwarding>(table);
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
    FarRing far_vc1{};
    if constexpr (vc1_forwarding != 0) {
        far_vc1 = far_ring_start<VC1_RECEIVER_CHANNEL>();
    }
    Worker worker{Channel<0>::first_slot, 0, 0, 0, unused_connection_value};
    ForwardingTable forwarding;
    init_forwarding<vc0_forwarding | vc1_forwarding>(forwarding, upstream);

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        // Router commands may have run the base firmware: program the worker credit's command buffer again.
        if (worker.expected_connection != unused_connection_value) {
            preset_worker_credit(worker_interface);
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
                if constexpr (vc1_forwarding != 0) {
                    // VC1: an idle VC1 costs the pass this group read and one test.
                    const uint32_t pending1 = channels_with_packets<vc1_forwarding>();
                    const uint32_t in_flight = forwarding.in_flight;
                    if (((pending1 | in_flight) & vc1_forwarding) != 0) {
                        visit_forwarding_channels<vc1_forwarding, VC1_RECEIVER_CHANNEL>(
                            forwarding, pending1, in_flight, far_vc1, round.attempted);
                    }
                }
            }
            const bool firmware_ran = end_round(
                round, !IDLE_CONTEXT_SWITCHING || round.attempted || total_sent(worker, forwarding) != round.sent_at_round_start);
            if (firmware_ran && worker.expected_connection != unused_connection_value) {
                preset_worker_credit(worker_interface);
            }
        }
    }
}

}  // namespace fabric_router::sender
