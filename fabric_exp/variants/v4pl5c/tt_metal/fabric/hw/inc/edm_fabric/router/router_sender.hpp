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
//   1. read the free-slots registers of VC0's channels and VC2's worker, as one group
//   2. VC0's worker channel: send if it has a packet and the far ring has room; take completions and credit the worker;
//      check the worker connection (every pass, by rule)
//   3. each VC's forwarding channels (VC0's, then VC1's on multi-mesh fabrics, after its own group read), against that
//      VC's far ring: when none has a packet or a packet in flight, one test skips them all; otherwise visit each that
//      has work
//   4. VC2 (neighbour exchange): its one channel, a worker channel, as in 2; its packets go into the far router's VC2
//      receiver ring (V4_VC2.md)
// The pass makes no calls while it sends, completes and takes upstream credits (LEDGER E0-11: forwarded 16 B 119.4 ->
// 112.3); the worker credit, once per completion batch, is out of line.
//
// The loop function takes no closures. A per-bit lambda capturing the loop's state by reference is an object
// holding pointers to it; whether the compiler dissolves it depends on the code around it, and when it does not, every
// test in the pass loads through those pointers and the captured state cannot live in registers (an unrelated edit to
// send once tipped it: 2D 3-hop 162 -> 188; with no closures 150, RESULTS §98). Its helpers recurse over their channel
// mask instead, with their inputs as arguments.
//
// State, one object per placement class (placement is per object; V4_STATE.md):
//   - left to the compiler (registers where they fit): each worker channel's slot, counts, connection semaphore address
//     and the value that semaphore holds while nothing changes (`Workers`), each VC's far ring (`FarRings`), which
//     forwarding channels have packets in flight (`in_flight`), the pass's free-slots result;
//   - in memory between visits (keep_in_memory at the end of each visit): each forwarding channel's record
//     (`ForwardingTable`: its slot and counts, its producer's credit address), touched only when its channel is
//     visited (VC0's, VC1's);
//   - in memory, set at connection time (they escape to the out-of-line connection event and worker credit): the
//     worker channels' interfaces (`WorkerConnections`);
//   - compile-time: slot bounds, registers, counters.
// Keeping the send path (slots, far rings) in memory too was measured and rejected: 3-hop and VC1 1-hop rows +5 to +16
// cycles/packet (the send's fences make the compiler store and reload memory state around each one; V4_STATE.md §11).
// The members of a channel or VC this router does not run (configuration, channel trimming) are absent (`Present`). On
// one ERISC everything is in the state record (router_single.hpp).

#include "router_common.hpp"
#include "router_round.hpp"
#include "router_telemetry.hpp"

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
    static constexpr size_t vc = CH < VC1_LOCAL_CHANNEL_START ? 0 : CH < VC2_LOCAL_CHANNEL_START ? 1 : 2;
    static constexpr size_t far_receiver =
        vc == 0 ? VC0_RECEIVER_CHANNEL : vc == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;
    static constexpr bool first_level_ack =
        vc == 0 ? ENABLE_FIRST_LEVEL_ACK_VC0 : vc == 1 ? ENABLE_FIRST_LEVEL_ACK_VC1 : false;
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
// Every VC's forwarding channels, and a channel's place among them: the index of its record (dense, compile time).
constexpr uint32_t all_forwarding = vc0_forwarding | vc1_forwarding;
constexpr size_t num_forwarding = popcount(all_forwarding);
template <size_t CH>
constexpr size_t record_index() {
    static_assert(((all_forwarding >> CH) & 1u) != 0, "not a serviced forwarding channel");
    return popcount(all_forwarding & ((1u << CH) - 1u));
}
// In Tensix mux mode the mux is the one worker and the only producer: no router forwards into this one.
static_assert(!tensix_mux_mode || (has_worker && vc0_forwarding == 0 && vc1_forwarding == 0));
// VC2's one channel, a worker channel (its own worker connects to it).
constexpr uint32_t vc2_worker = channels_in(VC2_LOCAL_CHANNEL_START, ACTUAL_VC2_SENDER_CHANNELS);
constexpr bool has_vc2_worker = vc2_worker != 0;
static_assert(popcount(vc2_worker) <= 1);
// Which far rings this router fills: a VC's, when any of its sender channels is serviced.
constexpr bool sends_vc0 = vc0_channels != 0;
constexpr bool sends_vc1 = vc1_forwarding != 0;
constexpr bool sends_vc2 = has_vc2_worker;

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

// A worker channel's state: its next slot and counts, the loop's copy of its connection semaphore's address (read
// every pass; held rather than rebuilt from a constant on each read, which costs VC1 1-hop rows ~2 cycles/packet:
// V4_STATE.md §12) and the value that semaphore holds while nothing changes.
struct Worker {
    uint32_t slot;
    uint32_t sent;
    uint32_t completed;
    uint32_t acked;  // first-level acks taken (first-level-ack VCs only)
    volatile tt_l1_ptr uint32_t* connection_semaphore;
    uint32_t expected_connection;
};

// The worker channels: VC0's channel 0 and VC2's channel, each where present.
struct Workers {
    [[no_unique_address]] Present<has_worker, Worker, 0> vc0;
    [[no_unique_address]] Present<has_vc2_worker, Worker, 2> vc2;
};

// The far rings this router fills, one per VC it sends on: VC0's (its worker and forwarding channels send into it),
// VC1's, VC2's.
struct FarRings {
    [[no_unique_address]] Present<sends_vc0, FarRing, 0> vc0;
    [[no_unique_address]] Present<sends_vc1, FarRing, 1> vc1;
    [[no_unique_address]] Present<sends_vc2, FarRing, 2> vc2;
};

// The worker channels' interfaces, the loop's own copies (connection events and the workers' credit addresses): passed
// to out-of-line calls, so in memory, apart from the state above. (Placement: see "State" above.)
template <typename Vc0Interface, typename Vc2Interface>
struct WorkerConnections {
    [[no_unique_address]] Vc0Interface vc0;
    [[no_unique_address]] Vc2Interface vc2;
};

// The forwarding channels' records, in memory between visits. (Which of them have packets in flight is a loop local of
// its own, `in_flight`: read every pass, it must not share this object.)
struct ForwardingTable {
    std::array<ForwardingRecord, num_forwarding> records;  // channel CH's at record_index<CH>()
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
// since `completed`, read right after the send, and on first-level-ack VCs `acks` the channel's ack count, read with
// them (the visit then uses it, rather than loading it where its next instruction would wait for it).
//   - read the payload size and the TX queue status as one group, and stamp the packet's source channel;
//     a busy queue yields;
//   - send the packet into the far ring's next slot, advance both slots;
//   - read the queue status and the channel's completion counter (and ack counter) as one group while the free counts
//     update;
//   - increment the far receiver's packets-received register (after the data, on the same queue).
// The channel enters only through four values of Channel<CH>: its ring bounds (the slot advance), its completion
// counter, its free-slots register and its id (stamped into the header); the rest is the same for every channel. It is
// inlined once per channel with those values as immediates: one copy shared at run time cost forwarded packets 5 to 18
// cycles (RESULTS §98, v4s, v4s2).
template <size_t CH>
FORCE_INLINE bool send(
    uint32_t& slot,
    FarRing& far,
    uint32_t& sent,
    uint32_t completed,
    uint32_t& completions,
    uint32_t& acks,
    bool& attempted) {
    using C = Channel<CH>;
    using FarT = FarRingTraits<C::far_receiver>;
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
    uint32_t reported_acks = 0;
    if constexpr (C::first_level_ack) {
        reported_acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter);
    }
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
    if constexpr (telemetry::counts_sent) {
        telemetry::count_sent(size);
    }
    completions = reported - completed;
    if constexpr (C::first_level_ack) {
        acks = reported_acks;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Credits back to producers
// ---------------------------------------------------------------------------------------------------------------------

// The worker credit goes back once per completion batch from the far router (every few packets), or under first-level
// acks once per ack frame. Out of line, its constants are built when a credit goes out instead of being held in the
// sender loop's registers for its whole run (2D 1-hop writes 141.8 -> 130.5, RESULTS §98).

// The credit write. Blackhole emulates an inline write to L1 (the inline-write API cannot target L1 there): the value is
// stored in a scratch word and a 4-byte NoC write copies it, always on this ERISC's write command buffer (the emulation
// ignores the command buffer it is given). Nothing else in the sender uses that buffer (its upstream credits and
// teardown use the atomics buffer). With one worker channel, the destination, the connected worker's credit word, holds
// for a whole connection, so the buffer is programmed with it when the worker connects and again after the base
// firmware has run (it may use the buffer); a credit then stores the count and issues the write with that state
// (source, address and control) instead of programming the whole command. With VC2's worker channel too, both workers'
// credits share the buffer, so each is a whole command (a preset would be overwritten by the other worker's credits).
constexpr bool worker_credit_preset = command_buffer_presets && !has_vc2_worker;
static_assert(!enable_read_counter_update_noc_flush);  // the scratch word may be rewritten before an earlier write read
                                                       // it: the worker then sees the newer, larger count
template <typename WorkerInterface>
__attribute__((noinline)) void preset_worker_credit(const WorkerInterface& interface) {
    if constexpr (worker_credit_preset) {
        noc_async_write_one_packet_set_state</*posted=*/true>(
            interface.cached_worker_semaphore_address, sizeof(uint32_t), tt::tt_fabric::worker_handshake_noc);
    }
}

// Where a worker channel's credits come from, per channel at compile time. VC0's worker: the scratch word while
// connected (through the preset command buffer when it is the router's only worker channel), its connection info's read
// counter while not. VC2's worker: its connection info's read counter in both cases, never the scratch word (a VC0
// credit may still be reading that word and would carry this count to the VC0 worker; main shares it, V4_VC2.md).
template <size_t CH>
struct WorkerCredit {
    static constexpr uint32_t idle_word =
        sender_connection_info_addrs[CH] + offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
    static constexpr bool preset = CH == 0 && worker_credit_preset;
    static constexpr uint32_t source = CH == 0 ? notify_worker_of_read_counter_update_src_address : idle_word;
};
// Only VC0's worker may own the preset command buffer, and only when it is the router's one worker channel.
static_assert(!(has_vc2_worker && worker_credit_preset));

// To worker channel CH's worker: the absolute count of slots returned, written into the worker's L1 while it is
// connected (a whole command, or the preset one), and into its connection info for the next worker otherwise.
template <size_t CH, typename WorkerInterface>
__attribute__((noinline)) void credit_worker(const WorkerInterface& interface, uint32_t returned, bool connected) {
    using W = WorkerCredit<CH>;
    if (connected) {
        if constexpr (W::preset) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(W::source) = returned;
            noc_async_write_one_packet_with_state</*posted=*/true>(
                W::source, static_cast<uint32_t>(interface.cached_worker_semaphore_address), tt::tt_fabric::worker_handshake_noc);
        } else {
            // A whole command, from the source word and unflushed: a later, larger count may overwrite the word before
            // this write reads it, which the worker accepts.
            noc_inline_dw_write<InlineWriteDst::L1, /*posted=*/true, /*flush=*/false>(
                interface.cached_worker_semaphore_address,
                returned,
                0xF,
                tt::tt_fabric::worker_handshake_noc,
                NOC_UNICAST_WRITE_VC,
                W::source);
        }
    } else {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(W::idle_word) = returned;
    }
}

// To an upstream router: +n on its free-slots register.
FORCE_INLINE void credit_upstream(uint64_t credit_addr, uint32_t n) {
    noc_inline_dw_write<InlineWriteDst::REG, true>(
        credit_addr, pack_value_for_inc_on_write_stream_reg_write(n), 0xf, tt::tt_fabric::worker_handshake_noc);
}

// Worker channel CH's connection semaphore changed (open or close request): handle it out of line and return the value
// it holds while nothing changes. The interface's read counter is set to the credits returned so far. VC0's worker
// presets its credit command at connection (a no-op unless it owns the preset command buffer).
template <size_t CH, typename WorkerInterface>
__attribute__((noinline)) uint32_t worker_connection_event(
    WorkerInterface& interface, uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    interface.local_read_counter.counter = returned;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            establish_worker_connection<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(interface);
            if constexpr (CH == 0) {
                preset_worker_credit(interface);
            }
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

// The worker credit a visit owes: for the completions it took, or under first-level acks for a changed ack count.
template <size_t CH, typename WorkerInterface>
FORCE_INLINE void credit_visit(
    Worker& w, WorkerInterface& interface, uint32_t completions, bool acks_read, uint32_t reported_acks, bool connected) {
    if constexpr (Channel<CH>::first_level_ack) {
        if (acks_read && reported_acks != w.acked) {
            w.acked = reported_acks;
            credit_worker<CH>(interface, w.acked, connected);
        }
    } else if (completions != 0) {
        credit_worker<CH>(interface, w.completed, connected);
    }
}

// A worker channel (VC0's channel 0, or VC2's): send, take completions (and first-level acks), credit the worker, check
// the connection.
template <size_t CH, typename WorkerInterface>
FORCE_INLINE void visit_worker(Worker& w, WorkerInterface& interface, bool has_packet, FarRing& far, bool& attempted) {
    using C = Channel<CH>;
    using namespace tt::tt_fabric::connection_interface;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool sent = false;
    if (has_packet && far.free_slots >= C::far_slots_needed) {
        sent = send<CH>(w.slot, far, w.sent, w.completed, completions, sent_acks, attempted);
    }
    // The visit's counters, read as one group so their loads overlap: completions and acks (first-level-ack VCs) unless
    // the send read them, and the connection semaphore, which is read every pass. Completions before acks, so the acks
    // seen are never fewer than the completions seen. Whether to read each is known before any is read.
    const bool read_completions = !sent && w.sent != w.completed;
    const bool read_acks = C::first_level_ack && !sent && w.sent != w.acked;
    uint32_t reported_completions = 0;
    uint32_t reported_acks = sent_acks;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {
        reported_completions = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::completion_counter);
    }
    if (read_acks) {
        reported_acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter);
    }
    const uint32_t semaphore = *w.connection_semaphore;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {
        completions = reported_completions - w.completed;
    }
    // v4pl5c (isolation): the credit and the connection check in v4pl1's sequence.
    const bool connected = w.expected_connection != unused_connection_value;
    if (completions != 0) {
        far.free_slots += completions;
        w.completed += completions;
        if constexpr (!C::first_level_ack) {
            credit_worker<CH>(interface, w.completed, connected);
        }
    }
    if constexpr (C::first_level_ack) {
        if ((sent || read_acks) && reported_acks != w.acked) {
            w.acked = reported_acks;
            credit_worker<CH>(interface, w.acked, connected);
        }
    }
    // The connection: one load and compare every pass; open and close out of line.
    if (semaphore != w.expected_connection) {
        const uint32_t returned = C::first_level_ack ? w.acked : w.completed;
        w.expected_connection = worker_connection_event<CH>(interface, semaphore, w.expected_connection, returned);
    }
}

// A forwarding channel with work: send if it has a packet and the far ring has room, take completions (and acks),
// credit the upstream router, and keep the channel's bit in `in_flight` equal to "packets not yet completed".
template <size_t CH>
FORCE_INLINE void visit_forwarding(
    ForwardingTable& table, uint32_t& in_flight, bool has_packet, FarRing& far, bool& attempted) {
    ForwardingRecord& r = table.records[record_index<CH>()];
    using C = Channel<CH>;
    uint32_t sent = r.sent;
    uint32_t completed = r.completed;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool did_send = false;
    if (has_packet && far.free_slots >= C::far_slots_needed) {
        uint32_t slot = r.slot;
        did_send = send<CH>(slot, far, sent, completed, completions, sent_acks, attempted);
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
            const uint32_t reported_acks =
                did_send ? sent_acks : *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(C::ack_counter);
            const uint32_t acks = reported_acks - r.acked;
            if (acks != 0) {
                r.acked += acks;
                credit_upstream(r.upstream_credit_addr, acks);
            }
        }
        busy = busy || sent != r.acked;
    }
    in_flight = busy ? (in_flight | (1u << CH)) : (in_flight & ~(1u << CH));
    keep_in_memory(r);
}

// ---------------------------------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------------------------------

// Visits the forwarding channels in MASK that have a packet (`pending`) or packets in flight (`in_flight`), lowest
// channel first.
template <uint32_t MASK>
FORCE_INLINE void visit_forwarding_channels(
    ForwardingTable& table, uint32_t pending, uint32_t& in_flight, FarRing& far, bool& attempted) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((pending >> CH) & 1u) != 0;
        if (has_packet || ((in_flight >> CH) & 1u) != 0) {
            visit_forwarding<CH>(table, in_flight, has_packet, far, attempted);
        }
        visit_forwarding_channels<without_lowest_bit(MASK)>(table, pending, in_flight, far, attempted);
    }
}

// One VC's forwarding channels (MASK), skipped with one test when none has a packet or a packet in flight. (Far is a
// parameter so that a VC this router does not run passes its absent far ring.)
template <uint32_t MASK, typename Far>
FORCE_INLINE void visit_forwarding_group(
    ForwardingTable& table, uint32_t pending, uint32_t& in_flight, Far& far, bool& attempted) {
    if constexpr (MASK != 0) {
        if (((pending | in_flight) & MASK) != 0) {
            visit_forwarding_channels<MASK>(table, pending, in_flight, far, attempted);
        }
    }
}

// The producers' credit addresses of the forwarding channels, cached when each upstream router connected.
struct UpstreamCredits {
    std::array<uint64_t, num_forwarding> addr;  // channel CH's at record_index<CH>()
};

// The forwarding channels in MASK: each one's record at its first slot with nothing sent, and its producer's credit
// address.
template <uint32_t MASK>
FORCE_INLINE void init_forwarding(ForwardingTable& table, const UpstreamCredits& upstream) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        table.records[record_index<CH>()] = {Channel<CH>::first_slot, 0, 0, 0, upstream.addr[record_index<CH>()]};
        init_forwarding<without_lowest_bit(MASK)>(table, upstream);
    }
}

// Packets sent so far on the forwarding channels in MASK.
template <uint32_t MASK>
FORCE_INLINE uint32_t forwarded(const ForwardingTable& table) {
    if constexpr (MASK != 0) {
        return table.records[record_index<lowest_bit(MASK)>()].sent + forwarded<without_lowest_bit(MASK)>(table);
    } else {
        return 0;
    }
}

// Whether a sender channel holds a packet (telemetry, once per round; the pass's group read of the registers).
FORCE_INLINE bool any_channel_holds_packet() {
    return (channels_with_packets<vc0_channels | vc2_worker>() | channels_with_packets<vc1_forwarding>()) != 0;
}

// Packets sent so far on every channel (a round attempted a send if this changed or a send yielded; read by the idle
// context switch only). (W and the defaulted parameters below are template parameters so that an absent channel's or
// VC's members are named only where present: see Present, router_common.hpp.)
template <typename W>
FORCE_INLINE uint32_t total_sent(const W& workers, const ForwardingTable& table) {
    uint32_t sent = forwarded<all_forwarding>(table);
    if constexpr (has_worker) {
        sent += workers.vc0.sent;
    }
    if constexpr (has_vc2_worker) {
        sent += workers.vc2.sent;
    }
    return sent;
}

// The loop's own copies of the worker channels' interfaces: after init nothing else uses them, and copies keep the loop
// from holding a pointer to the caller's for its whole run.
template <size_t CH2 = VC2_LOCAL_CHANNEL_START, typename WorkerInterfaces>
FORCE_INLINE auto worker_connections(WorkerInterfaces& interfaces) {
    using Vc0Interface = Present<has_worker, std::remove_reference_t<decltype(interfaces.template get<0>())>, 0>;
    if constexpr (has_vc2_worker) {
        using Vc2Interface = std::remove_reference_t<decltype(interfaces.template get<CH2>())>;
        return WorkerConnections<Vc0Interface, Vc2Interface>{
            present<has_worker, 0>(interfaces.template get<0>()), interfaces.template get<CH2>()};
    } else {
        return WorkerConnections<Vc0Interface, Absent<2>>{present<has_worker, 0>(interfaces.template get<0>()), {}};
    }
}

// Worker channel CH at the start of the loop: at its first slot, nothing sent, no worker connected.
template <size_t CH, typename WorkerInterface>
FORCE_INLINE Worker worker_start(const WorkerInterface& interface) {
    return {Channel<CH>::first_slot, 0, 0, 0, interface.connection_live_semaphore,
            tt::tt_fabric::connection_interface::unused_connection_value};
}

// The worker channels at the start of the loop.
template <size_t CH0 = 0, size_t CH2 = VC2_LOCAL_CHANNEL_START, typename Connections>
FORCE_INLINE Workers workers_start(const Connections& connections) {
    Workers workers{};
    if constexpr (has_worker) {
        workers.vc0 = worker_start<CH0>(connections.vc0);
    }
    if constexpr (has_vc2_worker) {
        workers.vc2 = worker_start<CH2>(connections.vc2);
    }
    return workers;
}

// The far rings before the first send: every slot free.
template <size_t RX0 = VC0_RECEIVER_CHANNEL, size_t RX1 = VC1_RECEIVER_CHANNEL, size_t RX2 = VC2_RECEIVER_CHANNEL>
FORCE_INLINE FarRings far_rings_start() {
    FarRings far{};
    if constexpr (sends_vc0) {
        far.vc0 = far_ring_start<RX0>();
    }
    if constexpr (sends_vc1) {
        far.vc1 = far_ring_start<RX1>();
    }
    if constexpr (sends_vc2) {
        far.vc2 = far_ring_start<RX2>();
    }
    return far;
}

// Router commands may have run the base firmware: program VC0's worker credit command buffer again while a worker is
// connected (a no-op unless it owns the preset command buffer).
template <typename W, typename Connections>
FORCE_INLINE void restore_worker_credit_preset(const W& workers, const Connections& connections) {
    if constexpr (has_worker) {
        if (workers.vc0.expected_connection != tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_worker_credit(connections.vc0);
        }
    }
}

// One pass over the sender channels. The body of run()'s loop, which calls it on its own locals; a single-ERISC router
// runs it as the sender half of each pass over its state in memory (router_single.hpp).
template <typename Connections>
FORCE_INLINE void pass(
    Workers& workers,
    Connections& connections,
    ForwardingTable& forwarding,
    uint32_t& in_flight,
    FarRings& far,
    bool& attempted) {
    const uint32_t pending = channels_with_packets<vc0_channels | vc2_worker>();
    if constexpr (has_worker) {
        visit_worker<0>(workers.vc0, connections.vc0, (pending & 0x1u) != 0, far.vc0, attempted);
    }
    if constexpr (vc0_forwarding != 0) {
        visit_forwarding_group<vc0_forwarding>(forwarding, pending, in_flight, far.vc0, attempted);
    }
    if constexpr (vc1_forwarding != 0) {
        // VC1's registers are read here, after VC0's visits, not with VC0's group: read at the top of the pass, a VC1
        // packet landing during VC0's visits waits a pass (VC1 forwarding rows +0.6% to +2.2%, v4sh2; V4_SENDER_SHAPE.md).
        const uint32_t pending1 = channels_with_packets<vc1_forwarding>();
        visit_forwarding_group<vc1_forwarding>(forwarding, pending1, in_flight, far.vc1, attempted);
    }
    if constexpr (has_vc2_worker) {
        visit_worker<VC2_LOCAL_CHANNEL_START>(
            workers.vc2, connections.vc2, (pending & vc2_worker) != 0, far.vc2, attempted);
    }
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds. A function
// of its own: its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
template <typename WorkerInterfaces>
__attribute__((noinline)) void run(WorkerInterfaces& interfaces, const UpstreamCredits& upstream) {
    auto connections = worker_connections(interfaces);
    FarRings far = far_rings_start();
    Workers workers = workers_start(connections);
    ForwardingTable forwarding;
    init_forwarding<all_forwarding>(forwarding, upstream);
    uint32_t in_flight = 0;  // the forwarding channels with packets not yet completed (or acked)

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_preset(workers, connections);
        while (next_round()) {
            round.attempted = false;
            round.sent_at_round_start = total_sent(workers, forwarding);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(workers, connections, forwarding, in_flight, far, round.attempted);
            }
            const bool firmware_ran = end_round(
                round, !IDLE_CONTEXT_SWITCHING || round.attempted || total_sent(workers, forwarding) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>([] { return any_channel_holds_packet(); }, [] { return false; });
            }
            if (firmware_ran) {
                restore_worker_credit_preset(workers, connections);
            }
        }
    }
}

}  // namespace fabric_router::sender
