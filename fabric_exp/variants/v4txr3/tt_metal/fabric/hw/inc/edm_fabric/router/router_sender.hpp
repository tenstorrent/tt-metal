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
//   - its credit command buffer (a worker channel's): a NoC command buffer of its own, kept programmed with its worker's
//     credit destination while the worker is connected;
//   - first-level acks (a VC's), and bubble flow control (a channel that injects into a ring needs two free far slots);
//   - where its state lives, by role: a worker channel's state is a loop local and it is visited every pass; a
//     forwarding channel's state stays in memory between its visits, and it is visited when it has a packet or packets
//     in flight.
//
// One pass:
//   1. the pass group: the free-slots registers of the VCs that read at the pass start and every worker channel's
//      connection semaphore, as one group; then every worker channel's connection, checked (every pass, by rule; a
//      change is handled out of line)
//   2. each VC in turn (visit_vc): its free-slots registers (a VC that reads them before its own visits), its worker
//      channel, then its forwarding channels behind one test ("none has a packet or packets in flight"), all against
//      the VC's far ring
// The pass makes no calls while it sends, completes and takes upstream credits (LEDGER E0-11: forwarded 16 B 119.4 ->
// 112.3); the worker credit, once per completion batch, is out of line.
//
// The loop function takes no closures. A per-bit lambda capturing the loop's state by reference is an object
// holding pointers to it; whether the compiler dissolves it depends on the code around it, and when it does not, every
// test in the pass loads through those pointers and the captured state cannot live in registers (an unrelated edit to
// send once tipped it: 2D 3-hop 162 -> 188; with no closures 150, RESULTS §98). Its helpers recurse over their channel
// mask instead, with their inputs as arguments.
//
// State (V4_STATE.md). Where a piece of state lives follows from how often the pass touches it, and an object is the
// unit of placement (keep_in_memory applies to a whole object), so state that lives apart is kept in separate objects:
//   - loop locals (registers where they fit): the worker channels' states (`workers`: every pass), each VC's far ring
//     (`far`: every send), which forwarding channels have packets in flight (`in_flight`), the pass's free-slots result;
//   - in memory between visits (keep_in_memory at the end of each visit): the forwarding channels' states
//     (`forwarding`), touched only when the channel is visited;
//   - at a fixed location, built once when the loop starts: the worker channels' interfaces (worker_interface<CH>():
//     connection events and the workers' credit addresses; nothing per pass reads them);
//   - compile-time: slot bounds, registers, counters.
// Keeping the send path (slots, far rings) in memory too was measured and rejected: 3-hop and VC1 1-hop rows +5 to +16
// cycles/packet (the send's fences make the compiler store and reload memory state around each one; V4_STATE.md §11).
// A channel or VC this router does not run (configuration, channel trimming) has no state. On one ERISC everything is
// in the state record (router_single.hpp).

#include <new>

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

// A channel's role, for any channel index (init code asks about every channel, serviced or not; Channel<CH> exists only
// for a serviced one).
constexpr bool is_worker_channel(size_t ch) {
    return vc_takes_worker[vc_of_sender_channel(ch)] && ch == first_channel_of_vc(vc_of_sender_channel(ch));
}

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
// The VCs this router sends on (bit V: VC V has a serviced sender channel).
constexpr uint32_t sending_vcs = (Vc<0>::channels != 0 ? 1u : 0u) | (Vc<1>::channels != 0 ? 2u : 0u) |
                                 (Vc<2>::channels != 0 ? 4u : 0u);
// The channels whose free-slots registers the pass group reads; the others are read before their VC's visits.
constexpr uint32_t pass_start_channels = (Vc<0>::reads_at_pass_start ? Vc<0>::channels : 0u) |
                                         (Vc<1>::reads_at_pass_start ? Vc<1>::channels : 0u) |
                                         (Vc<2>::reads_at_pass_start ? Vc<2>::channels : 0u);

// In Tensix mux mode the mux is the one worker and the only producer: no router forwards into this one.
static_assert(!tensix_mux_mode || (worker_channels != 0 && all_channels == worker_channels));

// The command buffers the worker channels' credits use, one per worker channel in channel order (a router has at most
// two: VC0's and VC2's). The atomics buffer carries the upstream credits and the teardowns; nothing else in the sender
// uses these two, so each stays programmed for its channel's worker (see credit_worker).
constexpr std::array<uint8_t, 2> worker_credit_cmd_bufs = {write_cmd_buf, write_reg_cmd_buf};
static_assert(popcount(worker_channels) <= worker_credit_cmd_bufs.size());

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
    // A worker channel's credit command buffer (worker_credit_cmd_bufs).
    static constexpr uint8_t credit_cmd_buf = worker_credit_cmd_bufs[worker ? popcount(worker_channels & ((1u << CH) - 1u)) : 0];
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

// The sender's state (see "State" at the top of this file).
using WorkerChannels = Each<worker_channels, ChannelState>;          // loop locals
using ForwardingChannels = Each<forwarding_channels, ChannelState>;  // in memory between visits
using FarRings = Each<sending_vcs, FarRingOf>;                       // loop locals

// The worker channels' interfaces, at a fixed location (local data memory): built when the loop starts
// (start_worker_interfaces), then reached by channel. (Storage and an explicit construction: the kernel runs no global
// constructors.)
using WorkerInterfaces = Each<worker_channels, WorkerInterface>;
struct WorkerInterfaceStorage {
    alignas(WorkerInterfaces) unsigned char bytes[sizeof(WorkerInterfaces)];
};
inline WorkerInterfaceStorage worker_interface_storage;
template <size_t CH>
FORCE_INLINE WorkerInterface<CH>& worker_interface() {
    return std::launder(reinterpret_cast<WorkerInterfaces*>(worker_interface_storage.bytes))->template of<CH>();
}

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

// What the host's observers record of a packet sent on channel CH, `size` bytes with its header (each compiled away when
// off): the channel-trimming capture (the channel was used; its smallest and largest packet), telemetry (its words; the
// round's packets are the channels' `sent` counts, read once per round).
template <size_t CH>
FORCE_INLINE void observe_sent(uint32_t size) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(CH);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(CH, static_cast<uint16_t>(size));
    }
    if constexpr (telemetry::counts_sent) {
        telemetry::count_sent(size);
    }
}

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
    observe_sent<CH>(size);
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
// the count: a worker reads it there when it opens a connection), and while a worker is connected a 4-byte posted NoC
// write copies that word into the worker's L1, on the channel's own command buffer (Channel<CH>::credit_cmd_buf). With
// two ERISCs that buffer stays programmed for the whole connection (source, destination, size: the connection event
// presets it, and it is preset again after the base firmware ran), and a credit only issues the write; with one ERISC
// both roles share the buffers and every credit is a whole command.
constexpr bool worker_credit_preset = command_buffer_presets;
static_assert(!enable_read_counter_update_noc_flush);  // the read counter may be rewritten before an earlier write read
                                                       // it: the worker then sees the newer, larger count

// A channel's read counter in its connection info: the source word of its worker credits (the interface's too, for the
// write at connection; a forwarding channel's interface never uses it).
constexpr uint32_t worker_read_counter(size_t ch) {
    return sender_connection_info_addrs[ch] + offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
}

// Programs worker channel CH's credit command buffer for its connected worker (two ERISCs).
template <size_t CH>
__attribute__((noinline)) void preset_worker_credit() {
    if constexpr (worker_credit_preset) {
        ncrisc_noc_write_set_state</*posted=*/true, /*one_packet=*/true>(
            tt::tt_fabric::worker_handshake_noc,
            Channel<CH>::credit_cmd_buf,
            worker_interface<CH>().cached_worker_semaphore_address,
            sizeof(uint32_t),
            NOC_UNICAST_WRITE_VC);
    }
}

// To worker channel CH's worker: the absolute count of slots returned, into the channel's read counter and, while a
// worker is connected, from there into the worker's L1 (unflushed: a later, larger count may overwrite the word before
// this write reads it, which the worker accepts).
template <size_t CH>
__attribute__((noinline)) void credit_worker(uint32_t returned, bool connected) {
    using C = Channel<CH>;
    constexpr uint32_t word = worker_read_counter(CH);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = returned;
    if (connected) {
        if constexpr (worker_credit_preset) {
            ncrisc_noc_write_with_state<noc_mode, /*posted=*/true, /*update_counter=*/true, /*one_packet=*/true>(
                tt::tt_fabric::worker_handshake_noc,
                C::credit_cmd_buf,
                word,
                static_cast<uint32_t>(worker_interface<CH>().cached_worker_semaphore_address));
        } else {
            while (!noc_cmd_buf_ready(tt::tt_fabric::worker_handshake_noc, C::credit_cmd_buf)) {
            }
            ncrisc_noc_fast_write<noc_mode>(
                tt::tt_fabric::worker_handshake_noc,
                C::credit_cmd_buf,
                word,
                worker_interface<CH>().cached_worker_semaphore_address,
                sizeof(uint32_t),
                NOC_UNICAST_WRITE_VC,
                /*mcast=*/false,
                /*linked=*/false,
                /*num_dests=*/1,
                /*multicast_path_reserve=*/true,
                /*posted=*/true);
        }
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
FORCE_INLINE void return_slots(const ChannelState<CH>& c, uint32_t n, uint32_t returned, bool connected) {
    if constexpr (Channel<CH>::worker) {
        credit_worker<CH>(returned, connected);
    } else {
        credit_upstream(c.upstream_credit, n);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Worker connections
// ---------------------------------------------------------------------------------------------------------------------

// Worker channel CH's connection semaphore changed (open or close request): handle it out of line and return the value
// it holds while nothing changes. The interface's read counter is set to the credits returned so far. Opening caches the
// worker's credit address, programs the channel's credit command buffer for it and sends the worker the credits returned
// so far, all on the channel's own buffer (the library's connect would send them on the write buffer).
template <size_t CH>
__attribute__((noinline)) uint32_t worker_connection_event(uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    auto& interface = worker_interface<CH>();
    interface.local_read_counter.counter = returned;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, MY_ETH_CHANNEL>();
            preset_worker_credit<CH>();
            credit_worker<CH>(returned, true);
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

// The base firmware may have used the command buffers (router commands, the context switch): program each connected
// worker channel's credit command buffer again (a no-op where command buffers are not preset).
template <uint32_t WORKERS = worker_channels, typename Workers>
FORCE_INLINE void restore_worker_credit_presets(const Workers& workers) {
    if constexpr (WORKERS != 0 && worker_credit_preset) {
        constexpr size_t CH = lowest_bit(WORKERS);
        if (producer_connected<CH>(workers.template of<CH>())) {
            preset_worker_credit<CH>();
        }
        restore_worker_credit_presets<without_lowest_bit(WORKERS)>(workers);
    }
}

// A worker channel's connection, from the semaphore the pass group read: a change (open or close request) is handled
// out of line. Returns whether it changed.
template <size_t CH>
FORCE_INLINE bool check_connection(ChannelState<CH>& c, uint32_t semaphore) {
    if (semaphore != c.expected_connection) {
        const uint32_t returned = Channel<CH>::first_level_ack ? c.acked : c.completed;
        c.expected_connection = worker_connection_event<CH>(semaphore, c.expected_connection, returned);
        return true;
    }
    return false;
}

// The connections of the worker channels in WORKERS (every pass, by rule), from the semaphores the pass group read
// (`semaphores[CH]`), lowest channel first.
template <uint32_t WORKERS, typename Workers>
FORCE_INLINE void check_connections(Workers& workers, const uint32_t* semaphores) {
    if constexpr (WORKERS != 0) {
        constexpr size_t CH = lowest_bit(WORKERS);
        check_connection<CH>(workers.template of<CH>(), semaphores[CH]);
        check_connections<without_lowest_bit(WORKERS)>(workers, semaphores);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Visits
// ---------------------------------------------------------------------------------------------------------------------

// A worker channel (visited every pass, its state in loop locals): send if it has a packet and the far ring has room,
// take completions (and first-level acks), return the slots to its worker.
template <size_t CH>
FORCE_INLINE void visit_worker(ChannelState<CH>& w, bool has_packet, FarRing& far, bool& attempted) {
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
            return_slots<CH>(w, completions, w.completed, connected);
        }
    }
    if constexpr (C::first_level_ack) {
        if ((sent || read_acks) && reported_acks != w.acked) {
            const uint32_t acks = reported_acks - w.acked;
            w.acked = reported_acks;
            return_slots<CH>(w, acks, w.acked, connected);
        }
    }
}

// A forwarding channel with work (a packet, or packets in flight): send if it has a packet and the far ring has room,
// take completions (and acks), return the slots to its upstream router, and keep the channel's bit in `in_flight` equal
// to "packets not yet completed (or acked)". Its state's fields are taken into locals where first needed and written
// back when they change (the send's fences would make the compiler store and reload memory state around each one).
template <size_t CH>
FORCE_INLINE void visit_forwarding(
    ForwardingChannels& forwarding, uint32_t& in_flight, bool has_packet, FarRing& far, bool& attempted) {
    ChannelState<CH>& r = forwarding.template of<CH>();
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
            return_slots<CH>(r, completions, completed, producer_connected<CH>(r));
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
                return_slots<CH>(r, acks, r.acked, producer_connected<CH>(r));
            }
        }
        busy = busy || sent != r.acked;
    }
    in_flight = busy ? (in_flight | (1u << CH)) : (in_flight & ~(1u << CH));
    keep_in_memory(r);
}

// The worker channels in MASK, every pass, lowest channel first.
template <uint32_t MASK, typename Workers>
FORCE_INLINE void visit_workers(Workers& workers, uint32_t pending, FarRing& far, bool& attempted) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        visit_worker<CH>(workers.template of<CH>(), (pending & (1u << CH)) != 0, far, attempted);
        visit_workers<without_lowest_bit(MASK)>(workers, pending, far, attempted);
    }
}

// The forwarding channels in MASK that have a packet (`pending`) or packets in flight (`in_flight`), lowest channel
// first.
template <uint32_t MASK>
FORCE_INLINE void visit_forwarding_channels(
    ForwardingChannels& forwarding, uint32_t pending, uint32_t& in_flight, FarRing& far, bool& attempted) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((pending >> CH) & 1u) != 0;
        if (has_packet || ((in_flight >> CH) & 1u) != 0) {
            visit_forwarding<CH>(forwarding, in_flight, has_packet, far, attempted);
        }
        visit_forwarding_channels<without_lowest_bit(MASK)>(forwarding, pending, in_flight, far, attempted);
    }
}

// VC V's channels against its far ring: its free-slots registers (a VC that reads them right before its visits; the
// others' are in `pending_at_start`, the pass group's), its worker channel, then its forwarding channels, skipped with
// one test when none has a packet or a packet in flight.
template <size_t VC, typename Workers, typename Far>
FORCE_INLINE void visit_vc(
    Workers& workers,
    ForwardingChannels& forwarding,
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
        FarRing& ring = far.template of<VC>();
        visit_workers<V::worker>(workers, pending, ring, attempted);
        constexpr uint32_t forwarders = V::channels & forwarding_channels;
        if constexpr (forwarders != 0) {
            if (((pending | in_flight) & forwarders) != 0) {
                visit_forwarding_channels<forwarders>(forwarding, pending, in_flight, ring, attempted);
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

// The loop-local state at the start of the loop.
FORCE_INLINE FarRings far_rings_start() {
    FarRings far{};
    start_far_rings<sending_vcs>(far);
    return far;
}
FORCE_INLINE WorkerChannels worker_channels_start(const UpstreamCredits& upstream) {
    WorkerChannels workers{};
    start_channels<worker_channels>(workers, upstream);
    return workers;
}

// The worker channels' interfaces in MASK, copied from the kernel's (after init nothing else uses those).
template <uint32_t MASK, typename KernelInterfaces>
FORCE_INLINE Each<MASK, WorkerInterface> interfaces_of(KernelInterfaces& interfaces) {
    if constexpr (MASK == 0) {
        return {};
    } else {
        return {interfaces.template get<lowest_bit(MASK)>(), interfaces_of<without_lowest_bit(MASK)>(interfaces)};
    }
}

// Builds the worker channels' interfaces at their fixed location (once, before the loop).
template <typename KernelInterfaces>
FORCE_INLINE void start_worker_interfaces(KernelInterfaces& interfaces) {
    new (worker_interface_storage.bytes) WorkerInterfaces{interfaces_of<worker_channels>(interfaces)};
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
template <typename Workers>
FORCE_INLINE uint32_t total_sent(const Workers& workers, const ForwardingChannels& forwarding) {
    return add_sent<worker_channels>(add_sent<forwarding_channels>(0, forwarding), workers);
}

// Whether a sender channel holds a packet (telemetry, once per round; the pass's register groups).
FORCE_INLINE bool any_channel_holds_packet() {
    return (channels_with_packets<pass_start_channels>() | channels_with_packets<all_channels & ~pass_start_channels>()) !=
           0;
}

// One pass over the sender channels. The body of run()'s loop, which calls it on its own locals; a single-ERISC router
// runs it as the sender half of each pass over its state in memory (router_single.hpp).
template <typename Workers, typename Far>
FORCE_INLINE void pass(
    Workers& workers, ForwardingChannels& forwarding, uint32_t& in_flight, Far& far, bool& attempted) {
    uint32_t semaphores[worker_channels == 0 ? 1 : 32 - __builtin_clz(worker_channels)];
    const uint32_t pending = channels_with_packets<pass_start_channels, worker_channels>(semaphores);
    check_connections<worker_channels>(workers, semaphores);
    visit_vc<0>(workers, forwarding, in_flight, far, pending, attempted);
    visit_vc<1>(workers, forwarding, in_flight, far, pending, attempted);
    visit_vc<2>(workers, forwarding, in_flight, far, pending, attempted);
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds. A function
// of its own: its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
template <typename KernelInterfaces>
__attribute__((noinline)) void run(KernelInterfaces& interfaces, const UpstreamCredits& upstream) {
    start_worker_interfaces(interfaces);
    FarRings far = far_rings_start();
    WorkerChannels workers = worker_channels_start(upstream);
    ForwardingChannels forwarding;
    start_channels<forwarding_channels>(forwarding, upstream);
    uint32_t in_flight = 0;  // the forwarding channels with packets not yet completed (or acked)

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_presets(workers);
        while (next_round()) {
            round.attempted = false;
            round.sent_at_round_start = total_sent(workers, forwarding);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(workers, forwarding, in_flight, far, round.attempted);
            }
            const uint32_t sent_this_round = total_sent(workers, forwarding) - round.sent_at_round_start;
            const bool firmware_ran = end_round(round, !IDLE_CONTEXT_SWITCHING || round.attempted || sent_this_round != 0);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>(
                    sent_this_round, [] { return any_channel_holds_packet(); }, [] { return false; });
            }
            if (firmware_ran) {
                restore_worker_credit_presets(workers);
            }
        }
    }
}

}  // namespace fabric_router::sender
