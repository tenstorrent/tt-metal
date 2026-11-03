// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender program: sends packets from this router's sender channels over the Ethernet link into the far router's
// receiver channel of the same VC, and returns each slot to its producer once the far router completes it.
//
// Channels and VCs. What a channel does follows from its features (Channel<CH>) and its VC's (Vc<V>), never from its
// number:
//   - role: a worker channel's producer connects and disconnects at run time (a worker on this chip, or the fabric's
//     mux); its connection is checked every pass, and its credits are the absolute count of slots returned, written into
//     the worker's L1. A forwarding channel's producer is the router facing another direction, connected once at init;
//     its credits go to that router's free-slots register. A VC that takes a worker gives it its first channel (VC0's
//     and VC2's, as the host builds them; VC1 is router to router);
//   - the lead channel: the worker channel of the lead VC (VC0, the VC every router runs); its credits keep the write
//     command buffer preset;
//   - first-level acks (a VC's), and bubble flow control (a channel that injects into a ring needs two free far slots);
//   - placement class (one place below, `register_channels` / `register_vcs`): a register-class channel's state is a
//     loop local and it is visited every pass; a record-class channel's state is a record in memory and it is visited
//     when it has a packet or packets in flight.
//
// One pass:
//   1. the pass group: the free-slots registers of the VCs that read at the pass start and every worker channel's
//      connection semaphore, as one group; then every worker channel's connection, checked (every pass, by rule; a
//      change is handled out of line)
//   2. each VC in turn (visit_vc): its free-slots registers (a VC that reads them before its own visits), its
//      register-class channels, then its record-class channels behind one test ("none has a packet or packets in
//      flight"), all against the VC's far ring
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
//   - left to the compiler (registers where they fit): the register-class channels' states, the far rings of the
//     register-class VCs, which record-class channels have packets in flight (`in_flight`), the pass's free-slots result;
//   - in memory between visits (keep_in_memory at the end of each visit): the record-class channels' states (`Records`),
//     touched only when the channel is visited;
//   - in memory, set at connection time (they escape to the out-of-line connection event and worker credit): the
//     worker channels' interfaces (`Connections`);
//   - compile-time: slot bounds, registers, counters.
// Keeping the send path (slots, far rings) in memory too was measured and rejected: 3-hop and VC1 1-hop rows +5 to +16
// cycles/packet (the send's fences make the compiler store and reload memory state around each one; V4_STATE.md §11).
// A channel or VC this router does not run (configuration, channel trimming) has no state. On one ERISC everything is
// in the state record (router_single.hpp).

#include "router_common.hpp"
#include "router_round.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::sender {

constexpr uint8_t txq = sender_txq_id;

// ---------------------------------------------------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------------------------------------------------

// Injection channels on a deadlock-avoiding ring or torus send only with two free far slots (bubble flow control): the
// worker channel, and on a torus a channel fed from the other axis (a turn).
// (It reads the untrimmed array: GCC rejects the trimmed std::array<bool, N>'s operator[] in this constant expression as
// "used before its definition".)
constexpr uint32_t far_slots_needed_by(size_t ch) {
    return (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel_all_[ch])
               ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS
               : 1;
}

// The serviced sender channels among [first, first + count), one bit per channel.
constexpr uint32_t channels_in(size_t first, size_t count) {
    uint32_t mask = 0;
    for (size_t ch = first; ch < first + count && ch < NUM_SENDER_CHANNELS; ch++) {
        mask |= is_sender_channel_serviced[ch] ? (1u << ch) : 0u;
    }
    return mask;
}

// Per VC, as the host builds it: its sender channels (flat layout VC0, VC1, VC2; ct args), the far router's receiver
// channel it sends into, whether a worker connects to its first channel (erisc_datamover_builder.cpp: VC0's first
// channel and VC2's; VC1 is router to router), and whether its packets are first-level acked (VC2's never).
constexpr size_t first_channel_of_vc(size_t vc) {
    return vc == 0 ? 0 : vc == 1 ? VC1_LOCAL_CHANNEL_START : VC2_LOCAL_CHANNEL_START;
}
constexpr size_t channel_count_of_vc(size_t vc) {
    return vc == 0 ? ACTUAL_VC0_SENDER_CHANNELS : vc == 1 ? ACTUAL_VC1_SENDER_CHANNELS : ACTUAL_VC2_SENDER_CHANNELS;
}
constexpr size_t far_receiver_of_vc(size_t vc) {
    return vc == 0 ? VC0_RECEIVER_CHANNEL : vc == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;
}
constexpr std::array<bool, 3> vc_takes_worker = {true, false, true};
constexpr std::array<bool, 3> vc_first_level_ack = {ENABLE_FIRST_LEVEL_ACK_VC0, ENABLE_FIRST_LEVEL_ACK_VC1, false};
// The lead VC: the one every router runs (VC1: multi-mesh fabrics; VC2: opt-in).
constexpr size_t lead_vc = 0;

// A channel's role and whether it is the lead channel, for any channel index (init code asks about every channel,
// serviced or not; Channel<CH> exists only for a serviced one).
constexpr bool is_worker_channel(size_t ch) {
    return vc_takes_worker[vc_of_sender_channel(ch)] && ch == first_channel_of_vc(vc_of_sender_channel(ch));
}
constexpr bool is_lead_channel(size_t ch) { return is_worker_channel(ch) && vc_of_sender_channel(ch) == lead_vc; }

// VC V.
template <size_t VC>
struct Vc {
    static constexpr uint32_t channels = channels_in(first_channel_of_vc(VC), channel_count_of_vc(VC));
    static constexpr uint32_t worker = vc_takes_worker[VC] ? channels & (1u << first_channel_of_vc(VC)) : 0u;
    static constexpr size_t far_receiver = far_receiver_of_vc(VC);
    static constexpr bool first_level_ack = vc_first_level_ack[VC];
    // Where its free-slots registers are read: in the pass group when it takes a worker (the worker's connection
    // semaphore is read there every pass, and a group of its own would expose a read on every idle pass;
    // V4_RECEIVER_SHAPE.md §29-§32); right before its own visits otherwise (VC1: read at the pass start, a VC1 packet
    // landing during VC0's visits waits a pass, VC1 forwarding rows +0.6% to +2.2%, v4sh2; V4_SENDER_SHAPE.md).
    static constexpr bool reads_at_pass_start = vc_takes_worker[VC];
};

constexpr uint32_t all_channels = Vc<0>::channels | Vc<1>::channels | Vc<2>::channels;
constexpr uint32_t worker_channels = Vc<0>::worker | Vc<1>::worker | Vc<2>::worker;
constexpr uint32_t forwarding_channels = all_channels & ~worker_channels;
constexpr uint32_t lead_channels = Vc<lead_vc>::worker;  // the lead channel, where serviced
// The VCs this router sends on (bit V: VC V has a serviced sender channel).
constexpr uint32_t sending_vcs = (Vc<0>::channels != 0 ? 1u : 0u) | (Vc<1>::channels != 0 ? 2u : 0u) |
                                 (Vc<2>::channels != 0 ? 4u : 0u);
// The channels whose free-slots registers the pass group reads; the others are read before their VC's visits.
constexpr uint32_t pass_start_channels = (Vc<0>::reads_at_pass_start ? Vc<0>::channels : 0u) |
                                         (Vc<1>::reads_at_pass_start ? Vc<1>::channels : 0u) |
                                         (Vc<2>::reads_at_pass_start ? Vc<2>::channels : 0u);

// Placement, in one place: the channels whose state and the VCs whose far ring are loop locals (registers where they
// fit); every other channel's state is a record in memory. Today's rule: every worker channel's state and every far
// ring left to the compiler, the forwarding channels' records in memory (v4pl1, V4_PLACEMENT.md).
constexpr uint32_t register_channels = worker_channels;
constexpr uint32_t record_channels = all_channels & ~register_channels;
constexpr uint32_t register_vcs = sending_vcs;
// What the visits below implement: register-class worker channels (connection expected value in a register),
// record-class forwarding channels, and far rings in registers.
static_assert((register_channels & forwarding_channels) == 0 && (record_channels & worker_channels) == 0 &&
                  register_vcs == sending_vcs,
              "placement: a record-class worker channel, a register-class forwarding channel or a far ring in memory "
              "is not implemented");
static_assert((lead_channels & ~register_channels) == 0, "the lead channel's credit preset reads its register state");
// In Tensix mux mode the mux is the one worker and the only producer: no router forwards into this one.
static_assert(!tensix_mux_mode || (lead_channels != 0 && all_channels == lead_channels));

// Sender channel CH (a serviced one).
template <size_t CH>
struct Channel {
    static constexpr size_t vc = vc_of_sender_channel(CH);
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t free_slots_stream = sender_free_slots_stream<CH>;
    // Counters the far receiver writes over the link.
    static constexpr uint32_t completion_counter = to_sender_remote_completion_counters_base_address + CH * 4;
    static constexpr uint32_t ack_counter = to_sender_remote_ack_counters_base_address + CH * 4;
    // A worker channel's connection semaphore (compile time; the worker interface holds the same address).
    static constexpr uint32_t connection_semaphore = sender_connection_semaphore_addrs[CH];
    static constexpr size_t far_receiver = Vc<vc>::far_receiver;
    static constexpr bool worker = is_worker_channel(CH);
    static constexpr bool lead = is_lead_channel(CH);
    static constexpr bool first_level_ack = Vc<vc>::first_level_ack;
    static constexpr uint32_t far_slots_needed = far_slots_needed_by(CH);
    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
    static_assert(!worker || !sender_ch_live_check_skip[CH], "the host checks every worker channel's connection");
};

// The far router's receiver ring of a VC: its slots and its packets-received register (on the far router).
template <size_t RX>
struct FarRingTraits {
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[RX];
    static constexpr uint32_t first_slot = eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[RX]>::base_address;
    static constexpr uint32_t last_slot = first_slot + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[RX];
};

// ---------------------------------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------------------------------

// One T<I> for each bit I of MASK, lowest first, found by its index at compile time (of<I>()); empty where MASK is 0.
// The sender's per-channel and per-VC state and the worker interfaces are kept in these, so a channel or VC this router
// does not run has no member.
template <uint32_t MASK, template <size_t> class T>
struct Each {
    [[no_unique_address]] T<lowest_bit(MASK)> first;
    [[no_unique_address]] Each<without_lowest_bit(MASK), T> rest;

    template <size_t I>
    FORCE_INLINE T<I>& of() {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template of<I>();
        }
    }
    template <size_t I>
    FORCE_INLINE const T<I>& of() const {
        if constexpr (I == lowest_bit(MASK)) {
            return first;
        } else {
            return rest.template of<I>();
        }
    }
};
template <size_t I>
struct NotHeld : std::false_type {};
template <template <size_t> class T>
struct Each<0, T> {
    // Code that names a member exists only where that member does (if constexpr on the same mask); naming one this
    // container does not hold is an error where it is instantiated, not where it is written.
    template <size_t I>
    T<I>& of() {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
    template <size_t I>
    const T<I>& of() const {
        static_assert(NotHeld<I>::value, "no member for this index");
    }
};

// The far ring as this router fills it.
struct FarRing {
    uint32_t free_slots;
    uint32_t next_slot;
};

// A sender channel's state: its next slot and its counts, and its producer's link: a worker channel's expected
// connection value (the value its connection semaphore, Channel<CH>::connection_semaphore, holds while nothing
// changes), a forwarding channel's upstream credit address (the upstream router's free-slots register, cached when it
// connected).
template <size_t CH>
struct ChannelState {
    uint32_t slot;       // next slot to send from
    uint32_t sent;       // packets sent
    uint32_t completed;  // completions taken
    uint32_t acked;      // first-level acks taken (first-level-ack VCs only)
    [[no_unique_address]] Present<Channel<CH>::worker, uint32_t, 0> expected_connection;
    [[no_unique_address]] Present<!Channel<CH>::worker, uint64_t, 1> upstream_credit;
};

template <size_t VC>
using FarRingOf = FarRing;

template <size_t CH>
using WorkerInterface =
    tt::tt_fabric::StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY[CH]>;

// The state of each placement class (see the top of this file).
using RegisterChannels = Each<register_channels, ChannelState>;
using FarRings = Each<register_vcs, FarRingOf>;
using Records = Each<record_channels, ChannelState>;  // in memory: keep_in_memory at the end of each visit
using Connections = Each<worker_channels, WorkerInterface>;

// The far ring of VC before the first send: every slot free.
template <size_t VC>
FORCE_INLINE FarRing far_ring_start() {
    using FarT = FarRingTraits<Vc<VC>::far_receiver>;
    return {FarT::num_slots, FarT::first_slot};
}

// The producers' credit addresses of the forwarding channels, cached when each upstream router connected (init).
template <size_t CH>
constexpr size_t forwarding_index() {
    static_assert(((forwarding_channels >> CH) & 1u) != 0, "not a serviced forwarding channel");
    return popcount(forwarding_channels & ((1u << CH) - 1u));
}
struct UpstreamCredits {
    std::array<uint64_t, popcount(forwarding_channels)> addr;  // channel CH's at forwarding_index<CH>()
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

// The connection semaphores of the worker channels in WORKERS, into semaphores[CH], lowest channel first.
template <uint32_t WORKERS>
FORCE_INLINE void read_connection_semaphores(uint32_t* semaphores) {
    if constexpr (WORKERS != 0) {
        constexpr size_t CH = lowest_bit(WORKERS);
        semaphores[CH] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(Channel<CH>::connection_semaphore);
        read_connection_semaphores<without_lowest_bit(WORKERS)>(semaphores);
    }
}

// The channels in MASK that hold a packet, one bit each, and the connection semaphores of the worker channels in WORKERS
// (into semaphores[CH]). Every register and semaphore is read first, as one group (reads overlap while their values are
// unused, so the group costs about one read; the loads are volatile and kept together between the two fences), then
// the registers are tested. Only the one-word result stays live across the pass.
template <uint32_t MASK, uint32_t WORKERS = 0>
FORCE_INLINE uint32_t channels_with_packets(uint32_t* semaphores = nullptr) {
    uint32_t registers[MASK == 0 ? 1 : 32 - __builtin_clz(MASK)];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    read_free_slots_registers<MASK>(registers);
    read_connection_semaphores<WORKERS>(semaphores);
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

// The credit write: the count goes into the channel's read counter in its connection info (which therefore always holds
// the count: a worker reads it there when it opens a connection), and while a worker is connected a 4-byte NoC write
// copies that word into the worker's L1. The lead channel's write is on this ERISC's write command buffer: Blackhole
// emulates an inline write to L1 there (the inline-write API cannot target L1, and the emulation ignores the command
// buffer it is given). Its destination, the connected worker's credit word, holds for a whole connection, so the buffer
// is programmed with it when the worker connects and again after anything else used the buffer (the base firmware;
// another worker channel's connection event, whose library connect writes the read counter as an L1 inline write); a
// credit then issues the write with that state (source, address and control) instead of programming the whole command. Every
// other worker channel's write is noc_semaphore_set_remote, on the register-write command buffer, so its credits leave
// the lead's preset in place. The sender's upstream credits and teardowns use the atomics buffer.
constexpr bool worker_credit_preset = command_buffer_presets;
static_assert(!enable_read_counter_update_noc_flush);  // the read counter may be rewritten before an earlier write read
                                                       // it: the worker then sees the newer, larger count

// A channel's read counter in its connection info: the source word of its worker credits (the interface's too, for the
// write at connection; a forwarding channel's interface never uses it).
constexpr uint32_t worker_read_counter(size_t ch) {
    return sender_connection_info_addrs[ch] + offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
}

template <typename Interface>
__attribute__((noinline)) void preset_worker_credit(const Interface& interface) {
    if constexpr (worker_credit_preset) {
        noc_async_write_one_packet_set_state</*posted=*/true>(
            interface.cached_worker_semaphore_address, sizeof(uint32_t), tt::tt_fabric::worker_handshake_noc);
    }
}

// To worker channel CH's worker: the absolute count of slots returned, into the channel's read counter and, while a
// worker is connected, from there into the worker's L1 (unflushed: a later, larger count may overwrite the word before
// this write reads it, which the worker accepts).
template <size_t CH, typename Interface>
__attribute__((noinline)) void credit_worker(const Interface& interface, uint32_t returned, bool connected) {
    using C = Channel<CH>;
    constexpr uint32_t word = worker_read_counter(CH);
    if (connected) {
        if constexpr (C::lead && worker_credit_preset) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = returned;
            noc_async_write_one_packet_with_state</*posted=*/true>(
                word, static_cast<uint32_t>(interface.cached_worker_semaphore_address), tt::tt_fabric::worker_handshake_noc);
        } else if constexpr (C::lead) {
            // A whole command (the emulated inline write stores the count into the word, then copies it).
            noc_inline_dw_write<InlineWriteDst::L1, /*posted=*/true, /*flush=*/false>(
                interface.cached_worker_semaphore_address,
                returned,
                0xF,
                tt::tt_fabric::worker_handshake_noc,
                NOC_UNICAST_WRITE_VC,
                word);
        } else {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = returned;
            noc_semaphore_set_remote(word, interface.cached_worker_semaphore_address, tt::tt_fabric::worker_handshake_noc);
        }
    } else {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = returned;
    }
}

// To an upstream router: +n on its free-slots register.
FORCE_INLINE void credit_upstream(uint64_t credit_addr, uint32_t n) {
    noc_inline_dw_write<InlineWriteDst::REG, true>(
        credit_addr, pack_value_for_inc_on_write_stream_reg_write(n), 0xf, tt::tt_fabric::worker_handshake_noc);
}

// Whether channel CH's producer is connected: a worker channel's, from its expected connection value; a forwarding
// channel's always (connected at init).
template <size_t CH>
FORCE_INLINE bool producer_connected(const ChannelState<CH>& c) {
    if constexpr (Channel<CH>::worker) {
        return c.expected_connection != tt::tt_fabric::connection_interface::unused_connection_value;
    } else {
        return true;
    }
}

// Slots back to channel CH's producer, `n` more for `returned` in all: a worker, the absolute count into its L1 while
// connected, else into its connection info (out of line); an upstream router, +n on its free-slots register. (Main's
// send_credits_to_upstream_workers, by the same feature.)
template <size_t CH>
FORCE_INLINE void return_slots(
    const ChannelState<CH>& c, Connections& connections, uint32_t n, uint32_t returned, bool connected) {
    if constexpr (Channel<CH>::worker) {
        credit_worker<CH>(connections.template of<CH>(), returned, connected);
    } else {
        credit_upstream(c.upstream_credit, n);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Worker connections
// ---------------------------------------------------------------------------------------------------------------------

// Worker channel CH's connection semaphore changed (open or close request): handle it out of line and return the value
// it holds while nothing changes. The interface's read counter is set to the credits returned so far. The lead channel
// presets its credit command at connection (a no-op where command buffers are not preset).
template <size_t CH, typename Interface>
__attribute__((noinline)) uint32_t worker_connection_event(
    Interface& interface, uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    interface.local_read_counter.counter = returned;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            establish_worker_connection<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(interface);
            if constexpr (Channel<CH>::lead) {
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

// Something else used the write command buffer (router commands may have run the base firmware; another worker channel
// had a connection event): program the lead channel's credit command again while its worker is connected (a no-op
// where command buffers are not preset).
template <typename Reg, typename Conn>
FORCE_INLINE void restore_worker_credit_preset(const Reg& reg, const Conn& connections) {
    if constexpr (lead_channels != 0) {
        constexpr size_t LEAD = lowest_bit(lead_channels);
        if (reg.template of<LEAD>().expected_connection != tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_worker_credit(connections.template of<LEAD>());
        }
    }
}

// A worker channel's connection, from the semaphore the pass group read: a change (open or close request) is handled
// out of line. Returns whether it changed.
template <size_t CH>
FORCE_INLINE bool check_connection(ChannelState<CH>& c, WorkerInterface<CH>& interface, uint32_t semaphore) {
    if (semaphore != c.expected_connection) {
        const uint32_t returned = Channel<CH>::first_level_ack ? c.acked : c.completed;
        c.expected_connection = worker_connection_event<CH>(interface, semaphore, c.expected_connection, returned);
        return true;
    }
    return false;
}

// The connections of the worker channels in WORKERS (every pass, by rule), from the semaphores the pass group read
// (`semaphores[CH]`), lowest channel first. Another worker channel's connection event may have used the write command
// buffer (its library connect), so the lead's credit preset is restored after it.
template <uint32_t WORKERS, typename Reg>
FORCE_INLINE void check_connections(Reg& reg, Connections& connections, const uint32_t* semaphores) {
    if constexpr (WORKERS != 0) {
        constexpr size_t CH = lowest_bit(WORKERS);
        if (check_connection<CH>(reg.template of<CH>(), connections.template of<CH>(), semaphores[CH])) {
            if constexpr (!Channel<CH>::lead) {
                restore_worker_credit_preset(reg, connections);
            }
        }
        check_connections<without_lowest_bit(WORKERS)>(reg, connections, semaphores);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Visits
// ---------------------------------------------------------------------------------------------------------------------

// A register-class channel (visited every pass, its state in loop locals): send if it has a packet and the far ring has
// room, take completions (and first-level acks), return the slots to its producer.
template <size_t CH>
FORCE_INLINE void visit_register_channel(
    ChannelState<CH>& w, Connections& connections, bool has_packet, FarRing& far, bool& attempted) {
    using C = Channel<CH>;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool sent = false;
    if (has_packet && far.free_slots >= C::far_slots_needed) {
        sent = send<CH>(w.slot, far, w.sent, w.completed, completions, sent_acks, attempted);
    }
    // The visit's counters, unless the send read them, as one group so their loads overlap: completions, and acks
    // (first-level-ack VCs). Completions before acks, so the acks seen are never fewer than the completions seen.
    // Whether to read each is known before any is read.
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
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read_completions) {
        completions = reported_completions - w.completed;
    }
    const bool connected = producer_connected<CH>(w);
    if (completions != 0) {
        far.free_slots += completions;
        w.completed += completions;
        if constexpr (!C::first_level_ack) {
            return_slots<CH>(w, connections, completions, w.completed, connected);
        }
    }
    if constexpr (C::first_level_ack) {
        if ((sent || read_acks) && reported_acks != w.acked) {
            const uint32_t acks = reported_acks - w.acked;
            w.acked = reported_acks;
            return_slots<CH>(w, connections, acks, w.acked, connected);
        }
    }
}

// A record-class channel with work (a packet, or packets in flight): send if it has a packet and the far ring has
// room, take completions (and acks), return the slots to its producer, and keep the channel's bit in `in_flight` equal
// to "packets not yet completed (or acked)". The record's fields are taken into locals where first needed and written
// back when they change (the send's fences would make the compiler store and reload a record around each one).
template <size_t CH>
FORCE_INLINE void visit_record_channel(
    Records& records, Connections& connections, uint32_t& in_flight, bool has_packet, FarRing& far, bool& attempted) {
    ChannelState<CH>& r = records.template of<CH>();
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
            return_slots<CH>(r, connections, completions, completed, producer_connected<CH>(r));
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
                return_slots<CH>(r, connections, acks, r.acked, producer_connected<CH>(r));
            }
        }
        busy = busy || sent != r.acked;
    }
    in_flight = busy ? (in_flight | (1u << CH)) : (in_flight & ~(1u << CH));
    keep_in_memory(r);
}

// The register-class channels in MASK, every pass, lowest channel first.
template <uint32_t MASK, typename Reg>
FORCE_INLINE void visit_register_channels(
    Reg& reg, Connections& connections, uint32_t pending, FarRing& far, bool& attempted) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        visit_register_channel<CH>(reg.template of<CH>(), connections, (pending & (1u << CH)) != 0, far, attempted);
        visit_register_channels<without_lowest_bit(MASK)>(reg, connections, pending, far, attempted);
    }
}

// The record-class channels in MASK that have a packet (`pending`) or packets in flight (`in_flight`), lowest channel
// first.
template <uint32_t MASK>
FORCE_INLINE void visit_record_channels(
    Records& records, Connections& connections, uint32_t pending, uint32_t& in_flight, FarRing& far, bool& attempted) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((pending >> CH) & 1u) != 0;
        if (has_packet || ((in_flight >> CH) & 1u) != 0) {
            visit_record_channel<CH>(records, connections, in_flight, has_packet, far, attempted);
        }
        visit_record_channels<without_lowest_bit(MASK)>(records, connections, pending, in_flight, far, attempted);
    }
}

// VC V's channels against its far ring: its free-slots registers (a VC that reads them right before its visits; the
// others' are in `pending_at_start`, the pass group's), its register-class channels, then its record-class channels,
// skipped with one test when none has a packet or a packet in flight.
template <size_t VC, typename Reg, typename Far>
FORCE_INLINE void visit_vc(
    Reg& reg,
    Connections& connections,
    Records& records,
    uint32_t& in_flight,
    Far& far,
    uint32_t pending_at_start,
    bool& attempted) {
    using V = Vc<VC>;
    if constexpr (V::channels != 0) {
        uint32_t pending = pending_at_start;
        if constexpr (!V::reads_at_pass_start) {
            pending = channels_with_packets<V::channels>();
        }
        FarRing& ring = far.template of<VC>();  // a register-class VC's far ring (the only kind, today)
        visit_register_channels<V::channels & register_channels>(reg, connections, pending, ring, attempted);
        constexpr uint32_t on_demand = V::channels & record_channels;
        if constexpr (on_demand != 0) {
            if (((pending | in_flight) & on_demand) != 0) {
                visit_record_channels<on_demand>(records, connections, pending, in_flight, ring, attempted);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------------------------------------------------

// The channels in MASK at the start of the loop: at their first slot with nothing sent, no worker connected, a
// forwarding channel with its upstream router's credit address (`upstream`).
template <uint32_t MASK, typename States>
FORCE_INLINE void start_channels(States& states, const UpstreamCredits& upstream) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        if constexpr (Channel<CH>::worker) {
            states.template of<CH>() = {
                Channel<CH>::first_slot, 0, 0, 0, tt::tt_fabric::connection_interface::unused_connection_value, {}};
        } else {
            states.template of<CH>() = {Channel<CH>::first_slot, 0, 0, 0, {}, upstream.addr[forwarding_index<CH>()]};
        }
        start_channels<without_lowest_bit(MASK)>(states, upstream);
    }
}

// The far rings of the VCs in MASK before the first send.
template <uint32_t MASK>
FORCE_INLINE void start_far_rings(FarRings& far) {
    if constexpr (MASK != 0) {
        far.template of<lowest_bit(MASK)>() = far_ring_start<lowest_bit(MASK)>();
        start_far_rings<without_lowest_bit(MASK)>(far);
    }
}

// The register-class state at the start of the loop.
FORCE_INLINE FarRings far_rings_start() {
    FarRings far{};
    start_far_rings<register_vcs>(far);
    return far;
}
FORCE_INLINE RegisterChannels register_channels_start(const UpstreamCredits& upstream) {
    RegisterChannels reg{};
    start_channels<register_channels>(reg, upstream);
    return reg;
}

// The loop's own copies of the worker channels' interfaces: after init nothing else uses them, and copies keep the loop
// from holding a pointer to the caller's for its whole run.
template <uint32_t MASK, typename WorkerInterfaces>
FORCE_INLINE Each<MASK, WorkerInterface> connections_from(WorkerInterfaces& interfaces) {
    if constexpr (MASK == 0) {
        return {};
    } else {
        return {interfaces.template get<lowest_bit(MASK)>(), connections_from<without_lowest_bit(MASK)>(interfaces)};
    }
}

// Packets sent so far on the channels in MASK, added to `sent`.
template <uint32_t MASK, typename States>
FORCE_INLINE uint32_t add_sent(uint32_t sent, const States& states) {
    if constexpr (MASK != 0) {
        return add_sent<without_lowest_bit(MASK)>(sent + states.template of<lowest_bit(MASK)>().sent, states);
    } else {
        return sent;
    }
}

// Packets sent so far on every channel (a round attempted a send if this changed or a send yielded; read by the idle
// context switch only).
template <typename Reg>
FORCE_INLINE uint32_t total_sent(const Reg& reg, const Records& records) {
    return add_sent<register_channels>(add_sent<record_channels>(0, records), reg);
}

// Whether a sender channel holds a packet (telemetry, once per round; the pass's register groups).
FORCE_INLINE bool any_channel_holds_packet() {
    return (channels_with_packets<pass_start_channels>() | channels_with_packets<all_channels & ~pass_start_channels>()) !=
           0;
}

// One pass over the sender channels. The body of run()'s loop, which calls it on its own locals; a single-ERISC router
// runs it as the sender half of each pass over its state in memory (router_single.hpp).
template <typename Reg, typename Far>
FORCE_INLINE void pass(
    Reg& reg, Connections& connections, Records& records, uint32_t& in_flight, Far& far, bool& attempted) {
    uint32_t semaphores[worker_channels == 0 ? 1 : 32 - __builtin_clz(worker_channels)];
    const uint32_t pending = channels_with_packets<pass_start_channels, worker_channels>(semaphores);
    check_connections<worker_channels>(reg, connections, semaphores);
    visit_vc<0>(reg, connections, records, in_flight, far, pending, attempted);
    visit_vc<1>(reg, connections, records, in_flight, far, pending, attempted);
    visit_vc<2>(reg, connections, records, in_flight, far, pending, attempted);
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds. A function
// of its own: its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
template <typename WorkerInterfaces>
__attribute__((noinline)) void run(WorkerInterfaces& interfaces, const UpstreamCredits& upstream) {
    Connections connections = connections_from<worker_channels>(interfaces);
    FarRings far = far_rings_start();
    RegisterChannels reg = register_channels_start(upstream);
    Records records;
    start_channels<record_channels>(records, upstream);
    uint32_t in_flight = 0;  // the record-class channels with packets not yet completed (or acked)

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_preset(reg, connections);
        while (next_round()) {
            round.attempted = false;
            round.sent_at_round_start = total_sent(reg, records);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(reg, connections, records, in_flight, far, round.attempted);
            }
            const bool firmware_ran = end_round(
                round, !IDLE_CONTEXT_SWITCHING || round.attempted || total_sent(reg, records) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>([] { return any_channel_holds_packet(); }, [] { return false; });
            }
            if (firmware_ran) {
                restore_worker_credit_preset(reg, connections);
            }
        }
    }
}

}  // namespace fabric_router::sender
