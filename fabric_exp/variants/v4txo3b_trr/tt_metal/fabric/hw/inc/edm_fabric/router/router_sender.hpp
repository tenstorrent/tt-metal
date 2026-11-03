// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The sender program: sends packets from this router's sender channels over the Ethernet link into the far router's
// receiver channel of the same VC, and returns each slot to its producer once the far router completes it.
//
// Each pass (pass, below):
//   reads     the free-slots registers of VC0's channels and VC2's worker channel, and the worker channels' connection
//             semaphores, as one read group (begin_read_group)
//   connect   each worker channel's connection, checked every pass (by rule); a change is handled out of line
//   send      VC0's worker channel: send if it holds a packet and the remote receiver channel has room; take
//             completions and credit the worker (out of line, once per completion batch)
//   forward   each VC's forwarding channels (VC0's; VC1's after a read group of its own), skipped with one test when
//             none holds a packet or has packets in flight; each one with work sends, takes completions and credits
//             its upstream router
//   VC2       its one channel, a worker channel, as VC0's (neighbour exchange; V4_VC2.md)
// The pass makes no call while it sends; the worker credit and the connection event are out of line (LEDGER E0-11).
// It takes no closures: a lambda capturing the loop's state can keep that state in memory (RESULTS §98), so the helpers
// recurse over their channel mask with their inputs as arguments.
//
// Channels and VCs. Each sender channel (SenderChannel<CH>) belongs to one VC. A VC's first channel takes a worker
// (VC0's channel 0, VC2's channel); the others are forwarding channels, written by the routers on this chip that face
// other directions. A VC's packets go into the far router's receiver channel of that VC (RemoteReceiverChannel<VC>).
//
// State, by placement class: the loop-local VC's send state (its remote receiver progress and its worker channel's
// progress and connection state) and which channels kept in memory have packets in flight, loop locals; every other
// VC's send state and each forwarding channel's state, in memory between its visits; the worker interfaces, in memory
// (they escape to out-of-line calls); slot bounds, registers and counters, compile-time. The block before pass()
// (SenderChannelStates) states the rule.
//
// Sections, in order: channels and their features; state; free-slots registers; sending; credits back to producers;
// visits; where the channel states live, and the loop.

#include "router_common.hpp"
#include "router_round.hpp"
#include "router_telemetry.hpp"

namespace fabric_router::sender {

// ---------------------------------------------------------------------------------------------------------------------
// Channels and their features
// ---------------------------------------------------------------------------------------------------------------------

constexpr uint8_t SENDER_TXQ = sender_txq_id;
// The NoC of the credits this router writes to its producers (workers and upstream routers).
constexpr uint8_t WORKER_CREDIT_NOC = tt::tt_fabric::worker_handshake_noc;

// Injection channels on a deadlock-avoiding ring or torus send only with two free far slots (bubble flow control): the
// worker channel, and on a torus a channel fed from the other axis (a turn). (It reads the untrimmed array: GCC rejects
// the trimmed std::array<bool, N>'s operator[] in this constant expression as "used before its definition".)
constexpr uint32_t remote_free_slots_to_send_by(size_t ch) {
    return (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel_all_[ch])
               ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS
               : 1;
}

// The worker channel positions of VC (the host's role, serviced or not), one bit each.
constexpr uint32_t worker_channel_positions(size_t vc) {
    uint32_t mask = 0;
    for (size_t ch = first_sender_channel_by_vc[vc];
         ch < first_sender_channel_by_vc[vc] + sender_channel_count_by_vc[vc] && ch < MAX_NUM_SENDER_CHANNELS;
         ch++) {
        mask |= is_sender_channel_worker[ch] ? (1u << ch) : 0u;
    }
    return mask;
}

// The serviced sender channels among the `count` channels from `first`, one bit each.
constexpr uint32_t channels_in(size_t first, size_t count) {
    uint32_t mask = 0;
    for (size_t ch = first; ch < first + count && ch < NUM_SENDER_CHANNELS; ch++) {
        mask |= is_sender_channel_serviced[ch] ? (1u << ch) : 0u;
    }
    return mask;
}

// Sender channel CH.
template <size_t CH>
struct SenderChannel {
    static constexpr size_t vc = vc_of_sender_channel(CH);  // ct args
    // A worker channel takes a worker (the host's role); the others are forwarding channels, which the routers on this
    // chip that face other directions write into.
    static constexpr bool worker = is_sender_channel_worker[CH];
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot_addr = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t free_slots_stream = sender_free_slots_stream<CH>;
    // Counters the far router's receiver writes over the link.
    static constexpr uint32_t completion_counter_addr =
        sender_channel_counter_addr(to_sender_remote_completion_counters_base_address, CH);
    static constexpr uint32_t first_level_ack_counter_addr =
        sender_channel_counter_addr(to_sender_remote_ack_counters_base_address, CH);
    // A worker channel's connection semaphore (compile time; the worker interface holds the same address).
    static constexpr uint32_t connection_semaphore_addr = sender_connection_semaphore_addrs[CH];
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[vc];
    static constexpr uint32_t remote_free_slots_to_send = remote_free_slots_to_send_by(CH);
    static_assert(num_slots > 0 && num_slots <= stream_count_mask);
    static_assert(sender_channel_uses_counter_credits(CH), "the sender reads its credits from L1 counters");
    static_assert(!worker || !sender_ch_live_check_skip[CH], "the host checks every worker channel's connection");
};

// VC's sender features: its serviced sender channels, its worker channel (where serviced) and its forwarding channels,
// whether its packets are first-level acked, and where its free-slots registers are read: in the pass's first read
// group when the VC has a worker channel position, whose connection semaphore that group reads anyway; right before
// its own visits otherwise, so a packet landing during the earlier VCs' visits does not wait a pass
// (V4_SENDER_SHAPE.md).
template <size_t VC>
struct SenderVc {
    static constexpr uint32_t sender_channels =
        channels_in(first_sender_channel_by_vc[VC], sender_channel_count_by_vc[VC]);
    static constexpr uint32_t worker_channels = sender_channels & worker_channel_positions(VC);
    static constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[VC];
    static constexpr bool reads_free_slots_at_pass_start = worker_channel_positions(VC) != 0;
    static_assert(popcount(worker_channel_positions(VC)) <= 1, "a VC takes at most one worker");
};

// Every VC's serviced channels, by kind; the VCs this router sends on (bit VC: VC has a serviced sender channel); the
// channels whose free-slots registers the pass's first read group reads.
static_assert(MAX_NUM_VCS == 3, "the sender's features and its pass name VCs 0, 1 and 2");
constexpr uint32_t sender_channels =
    SenderVc<0>::sender_channels | SenderVc<1>::sender_channels | SenderVc<2>::sender_channels;
constexpr uint32_t worker_channels =
    SenderVc<0>::worker_channels | SenderVc<1>::worker_channels | SenderVc<2>::worker_channels;
constexpr uint32_t forwarding_channels = sender_channels & ~worker_channels;
constexpr uint32_t sender_vcs = (SenderVc<0>::sender_channels != 0 ? 1u : 0u) |
                                (SenderVc<1>::sender_channels != 0 ? 2u : 0u) |
                                (SenderVc<2>::sender_channels != 0 ? 4u : 0u);
constexpr uint32_t channels_read_at_pass_start =
    (SenderVc<0>::reads_free_slots_at_pass_start ? SenderVc<0>::sender_channels : 0u) |
    (SenderVc<1>::reads_free_slots_at_pass_start ? SenderVc<1>::sender_channels : 0u) |
    (SenderVc<2>::reads_free_slots_at_pass_start ? SenderVc<2>::sender_channels : 0u);
// In Tensix mux mode the mux is the only producer: no router forwards into this one. (Channel trimming may leave the
// mux's worker channel unserviced too.)
static_assert(!tensix_mux_mode || forwarding_channels == 0);

// A forwarding channel's index among the serviced forwarding channels (its producer's address at init).
template <size_t CH>
constexpr size_t forwarding_index() {
    static_assert(((forwarding_channels >> CH) & 1u) != 0, "not a serviced forwarding channel");
    return popcount(forwarding_channels & ((1u << CH) - 1u));
}

// The far router's receiver channel of VC: its slots and its packets-received register (on the far router).
template <size_t VC>
struct RemoteReceiverChannel {
    static constexpr size_t rx = receiver_channel_by_vc[VC];  // the host's index of the channel
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[rx];
    static constexpr uint32_t first_slot_addr =
        eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[rx]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[rx];
};

// ---------------------------------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------------------------------

// This router's progress into VC's remote receiver channel, in one word: the slots known free there (low 16 bits) and
// the next slot to fill, its address / 16 (high 16 bits). One word, so a VC kept in memory takes and puts back its
// progress with one load and one store (V4_SENDER_OWNERSHIP.md §8.4). The encoding is for ERISC L1 only (the remote
// receiver channel is always in the far ERISC's L1); a Tensix downstream must not reuse it.
template <size_t VC>
class RemoteReceiverChannelProgress {
public:
    // Every slot free; the next slot is the first.
    static constexpr RemoteReceiverChannelProgress start() {
        return RemoteReceiverChannelProgress(
            encode(RemoteReceiverChannel<VC>::num_slots, RemoteReceiverChannel<VC>::first_slot_addr));
    }

    FORCE_INLINE bool has_room(uint32_t n) const { return free_slots() >= n; }

    // The next slot's address: the high half times 16, which is word >> 12 because the low half stays below 4096.
    FORCE_INLINE uint32_t next_slot_addr() const { return word >> 12; }

    // After a send into the next slot: the next slot advances (after the last, back to the first) and one free slot is
    // taken, in one add. The next slot is the last one exactly when the word is at least the last slot's high half.
    FORCE_INLINE void take_slot() {
        ASSERT(has_room(1));
        word += word >= last_slot_word ? wrap_add : advance_add;
    }

    // Slots the far router completed: free again.
    FORCE_INLINE void add_free_slots(uint32_t n) {
        ASSERT(free_slots() + n <= RemoteReceiverChannel<VC>::num_slots);
        word += n;
    }

    FORCE_INLINE bool operator!=(const RemoteReceiverChannelProgress& other) const { return word != other.word; }

private:
    static constexpr uint32_t encode(uint32_t free_slots, uint32_t slot_addr) { return (slot_addr / 16) << 16 | free_slots; }
    static constexpr uint32_t last_slot_word = encode(0, RemoteReceiverChannel<VC>::last_slot_addr);
    static constexpr uint32_t advance_add = encode(0, channel_buffer_size) - 1;
    static constexpr uint32_t wrap_add = encode(0, RemoteReceiverChannel<VC>::first_slot_addr) - last_slot_word - 1;  // mod 2^32
    static_assert(RemoteReceiverChannel<VC>::first_slot_addr % 16 == 0 && channel_buffer_size % 16 == 0);
    static_assert(RemoteReceiverChannel<VC>::last_slot_addr / 16 < (1u << 16), "a slot address / 16 fits in 16 bits");
    static_assert(RemoteReceiverChannel<VC>::num_slots < (1u << 12), "the free-slot count stays below bit 12");

    constexpr explicit RemoteReceiverChannelProgress(uint32_t w) : word(w) {}
    FORCE_INLINE uint32_t free_slots() const { return word & 0xFFFFu; }

    uint32_t word;
};

// A sender channel's progress through its slots: its per-packet state.
struct SenderChannelProgress {
    uint32_t next_slot_addr;     // the next slot to send from
    uint32_t sent;               // packets sent
    uint32_t completed;          // completions taken
    uint32_t first_level_acked;  // first-level acks taken (first-level-ack VCs only)
};

// A sender channel's state: its progress, and its producer's link: a worker channel's connection state (the value its
// connection semaphore holds while nothing changes), a forwarding channel's upstream router's free-slots register
// address (cached when the upstream router connected).
template <size_t CH>
struct SenderChannelState {
    SenderChannelProgress progress = {SenderChannel<CH>::first_slot_addr, 0, 0, 0};
    [[no_unique_address]] Present<SenderChannel<CH>::worker, uint32_t, 0> connection_state =
        present<SenderChannel<CH>::worker, 0>(tt::tt_fabric::connection_interface::unused_connection_value);
    [[no_unique_address]] Present<!SenderChannel<CH>::worker, uint64_t, 1> upstream_free_slots_addr{};
};

// VC's send state: its remote receiver channel's progress, and its worker channel's state where the VC has a serviced
// worker channel. (Its forwarding channels' states are SenderChannelStates::forwarding.)
template <size_t VC>
struct SenderVcState {
    RemoteReceiverChannelProgress<VC> remote = RemoteReceiverChannelProgress<VC>::start();
    [[no_unique_address]] Present<SenderVc<VC>::worker_channels != 0, SenderChannelState<first_sender_channel_by_vc[VC]>, 2>
        worker{};
};

// The sender channels' interfaces (the library's connection state, the producer's credit address), at a fixed
// location: built by the kernel at init, then reached by channel (worker_interface<CH>()). Storage and an explicit
// construction: the kernel runs no global constructors.
using WorkerInterfaces = decltype(tt::tt_fabric::EdmChannelWorkerInterfaces<WORKER_CREDIT_NOC, SENDER_NUM_BUFFERS_ARRAY>::make(
    std::make_index_sequence<NUM_SENDER_CHANNELS>{}));
struct WorkerInterfaceStorage {
    alignas(WorkerInterfaces) unsigned char bytes[sizeof(WorkerInterfaces)];
};
inline WorkerInterfaceStorage worker_interface_storage;
// Builds the interfaces at their location (once, at init); the kernel then sets each one up.
FORCE_INLINE WorkerInterfaces& start_worker_interfaces() {
    return *new (worker_interface_storage.bytes) WorkerInterfaces{};
}
template <size_t CH>
FORCE_INLINE auto& worker_interface() {
    return std::launder(reinterpret_cast<WorkerInterfaces*>(worker_interface_storage.bytes))->template get<CH>();
}

// ---------------------------------------------------------------------------------------------------------------------
// Free-slots registers
// ---------------------------------------------------------------------------------------------------------------------

// A sender channel's free-slots register counts the free slots of its ring: the producer takes one for each packet it
// writes, this router gives one back for each packet it sends. A channel holds a packet when the count is anything but
// its ring size. ("Below the ring size" would not do: a count out of range, an over-fill, must read as holding packets,
// as the generic loop reads it, or the channel is never served; LEDGER E0-10, RESULTS §93.)
template <size_t CH>
FORCE_INLINE bool holds_packet(uint32_t free_slots_register) {
    return (free_slots_register & stream_count_mask) != SenderChannel<CH>::num_slots;
}

// The free-slots registers of the channels in MASK, into registers[CH], lowest channel first.
template <uint32_t MASK>
FORCE_INLINE void read_free_slots_registers(uint32_t* registers) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        registers[CH] = read_stream_raw<SenderChannel<CH>::free_slots_stream>();
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

// The connection semaphores of the worker channels in WORKERS, into connection_semaphores[CH], lowest channel first.
template <uint32_t WORKERS>
FORCE_INLINE void read_connection_semaphores(uint32_t* connection_semaphores) {
    if constexpr (WORKERS != 0) {
        constexpr size_t CH = lowest_bit(WORKERS);
        connection_semaphores[CH] =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::connection_semaphore_addr);
        read_connection_semaphores<without_lowest_bit(WORKERS)>(connection_semaphores);
    }
}

// The channels in MASK that hold a packet, one bit each, and the connection semaphores of the worker channels in
// WORKERS (into connection_semaphores[CH]). Every register and semaphore is read first, as one read group, then the
// registers are tested. Only the one-word result stays live across the pass.
template <uint32_t MASK, uint32_t WORKERS = 0>
FORCE_INLINE uint32_t channels_with_packets(uint32_t* connection_semaphores = nullptr) {
    uint32_t registers[MASK == 0 ? 1 : 32 - __builtin_clz(MASK)];
    begin_read_group();
    read_free_slots_registers<MASK>(registers);
    read_connection_semaphores<WORKERS>(connection_semaphores);
    end_read_group();
    return channels_holding_packets<MASK>(registers);
}

// ---------------------------------------------------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------------------------------------------------

// Sends channel CH's next packet. Precondition: it has a packet and the remote receiver channel has room. Returns false
// when the TX queue was busy (nothing sent; the next pass tries again); after a send, `completions` holds the
// completions reported since `completed`, read right after the send, and on first-level-ack VCs `acks` the channel's
// ack count, read with them (the visit then uses it, rather than loading it where its next instruction would wait for
// it).
//   - read the payload size and the TX queue status as one group, and stamp the packet's source channel;
//     a busy queue yields;
//   - send the packet into the remote receiver channel's next slot, advance both slots;
//   - read the queue status and the channel's completion counter (and ack counter) as one group while the free counts
//     update;
//   - increment the far router's packets-received register (after the data, on the same queue).
// The channel enters only through four values of SenderChannel<CH>: its ring bounds (the slot advance), its completion
// counter, its free-slots register and its id (stamped into the header); the rest is the same for every channel. It is
// inlined once per channel with those values as immediates: one copy shared at run time cost forwarded packets 5 to 18
// cycles (RESULTS §98, v4s, v4s2).
template <size_t CH>
FORCE_INLINE bool send(
    uint32_t& next_slot_addr,
    RemoteReceiverChannelProgress<SenderChannel<CH>::vc>& remote,
    uint32_t& sent,
    uint32_t completed,
    uint32_t& completions,
    uint32_t& acks,
    bool& send_yielded) {
    constexpr size_t VC = SenderChannel<CH>::vc;
    const uint32_t slot_addr = next_slot_addr;
    auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(slot_addr);

    begin_read_group();
    const uint32_t payload_size = header->get_payload_size_excluding_header();
    const uint32_t status = read_txq_status(SENDER_TXQ);
    end_read_group();
    if constexpr (!skip_src_ch_id_update) {
        header->src_ch_id = static_cast<uint8_t>(CH);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (txq_busy(status)) {
        send_yielded = true;
        return false;
    }
    const uint32_t size = payload_size + sizeof(PACKET_HEADER_TYPE);
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(CH);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(CH, static_cast<uint16_t>(size));
    }

    internal_::eth_send_packet_bytes_unsafe(SENDER_TXQ, slot_addr, remote.next_slot_addr(), size);
    remote.take_slot();
    next_slot_addr = slot_addr == SenderChannel<CH>::last_slot_addr ? SenderChannel<CH>::first_slot_addr
                                                                    : slot_addr + channel_buffer_size;

    begin_read_group();
    const uint32_t status_after = read_txq_status(SENDER_TXQ);
    const uint32_t reported =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::completion_counter_addr);
    uint32_t reported_acks = 0;
    if constexpr (SenderChannel<CH>::first_level_ack) {
        reported_acks =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::first_level_ack_counter_addr);
    }
    end_read_group();

    increment_local_update_ptr_val<SenderChannel<CH>::free_slots_stream>(1);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (txq_busy(status_after)) {
        while (internal_::eth_txq_is_busy(SENDER_TXQ)) {
        }
    }
    remote_update_ptr_val<RemoteReceiverChannel<VC>::packets_received_stream, SENDER_TXQ>(1);
    sent++;
    if constexpr (telemetry::counts_sent) {
        telemetry::count_sent(size);
    }
    completions = reported - completed;
    if constexpr (SenderChannel<CH>::first_level_ack) {
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

// The credit write. VC0's worker: Blackhole emulates an inline write to L1 (the inline-write API cannot target L1
// there): the value is stored in a scratch word and a 4-byte NoC write copies it, always on this ERISC's write command
// buffer (the emulation ignores the command buffer it is given). The destination, the connected worker's credit word,
// holds for a whole connection, so the buffer is programmed with it when the worker connects and again after anything
// else used the buffer (the base firmware; a connection event of VC2's worker, whose library connect writes the
// worker's read counter as an L1 inline write); a credit then stores the count and issues the write with that state
// (source, address and control) instead of programming the whole command. VC2's worker: a 4-byte write from its own
// source word (noc_semaphore_set_remote, on the register-write command buffer), so its credits leave VC0's preset in
// place. The sender's upstream credits and teardowns use the atomics buffer.
constexpr bool worker_credit_preset = command_buffer_presets;
static_assert(!enable_read_counter_update_noc_flush);  // the scratch word may be rewritten before an earlier write read
                                                       // it: the worker then sees the newer, larger count
template <typename Interface>
__attribute__((noinline)) void preset_worker_credit(const Interface& interface) {
    if constexpr (worker_credit_preset) {
        noc_async_write_one_packet_set_state</*posted=*/true>(
            interface.cached_worker_semaphore_address, sizeof(uint32_t), WORKER_CREDIT_NOC);
    }
}

// Where a worker channel's credits come from, per channel at compile time. VC0's worker owns the write command buffer:
// the scratch word while connected (through the preset command buffer on two ERISCs), its connection info's read
// counter while not. VC2's worker: its connection info's read counter in both cases, never the scratch word (a VC0
// credit may still be reading that word and would carry this count to the VC0 worker; main shares it, V4_VC2.md).
template <size_t CH>
struct WorkerCredit {
    static constexpr uint32_t idle_word =
        sender_connection_info_addrs[CH] + offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, edm_read_counter);
    static constexpr bool owns_write_buffer = CH == 0;
    static constexpr bool preset = owns_write_buffer && worker_credit_preset;
    static constexpr uint32_t source = owns_write_buffer ? notify_worker_of_read_counter_update_src_address : idle_word;
};

// To worker channel CH's worker: the absolute count of slots returned, written into the worker's L1 while it is
// connected (a whole command, or the preset one), and into its connection info for the next worker otherwise.
template <size_t CH, typename Interface>
__attribute__((noinline)) void credit_worker(const Interface& interface, uint32_t returned, bool connected) {
    if (connected) {
        if constexpr (WorkerCredit<CH>::preset) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WorkerCredit<CH>::source) = returned;
            noc_async_write_one_packet_with_state</*posted=*/true>(
                WorkerCredit<CH>::source,
                static_cast<uint32_t>(interface.cached_worker_semaphore_address),
                WORKER_CREDIT_NOC);
        } else if constexpr (WorkerCredit<CH>::owns_write_buffer) {
            // A whole command, from the source word and unflushed: a later, larger count may overwrite the word before
            // this write reads it, which the worker accepts.
            noc_inline_dw_write<InlineWriteDst::L1, /*posted=*/true, /*flush=*/false>(
                interface.cached_worker_semaphore_address,
                returned,
                INLINE_WRITE_ALL_BYTES,
                WORKER_CREDIT_NOC,
                NOC_UNICAST_WRITE_VC,
                WorkerCredit<CH>::source);
        } else {
            // Off the write command buffer, from the source word (unflushed, as above).
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WorkerCredit<CH>::source) = returned;
            noc_semaphore_set_remote(
                WorkerCredit<CH>::source, interface.cached_worker_semaphore_address, WORKER_CREDIT_NOC);
        }
    } else {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WorkerCredit<CH>::idle_word) = returned;
    }
}

// To an upstream router: +n on its free-slots register.
FORCE_INLINE void credit_upstream(uint64_t credit_addr, uint32_t n) {
    noc_inline_dw_write<InlineWriteDst::REG, true>(
        credit_addr, pack_value_for_inc_on_write_stream_reg_write(n), INLINE_WRITE_ALL_BYTES, WORKER_CREDIT_NOC);
}

// Worker channel CH's connection semaphore changed (open or close request): handle it out of line and return the value
// it holds while nothing changes. The interface's read counter is set to the credits returned so far. VC0's worker
// presets its credit command at connection (a no-op unless it owns the preset command buffer).
template <size_t CH, typename Interface>
__attribute__((noinline)) uint32_t worker_connection_event(
    Interface& interface, uint32_t semaphore, uint32_t expected, uint32_t returned) {
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

// A worker channel's connection, checked every pass (by rule) from the semaphore the pass's read group read: a change
// (open or close request) is handled out of line. Returns whether it changed.
template <size_t CH, typename Interface>
FORCE_INLINE bool check_connection(SenderChannelState<CH>& state, Interface& interface, uint32_t semaphore) {
    if (semaphore != state.connection_state) {
        const uint32_t returned =
            SenderChannel<CH>::first_level_ack ? state.progress.first_level_acked : state.progress.completed;
        state.connection_state = worker_connection_event<CH>(interface, semaphore, state.connection_state, returned);
        return true;
    }
    return false;
}

// Slots back to channel CH's producer, `n` more for `returned` in all: to a worker, the absolute count (out of line);
// to an upstream router, +n on its free-slots register.
template <size_t CH>
FORCE_INLINE void return_slots(
    const SenderChannelState<CH>& state, uint32_t n, uint32_t returned) {
    if constexpr (SenderChannel<CH>::worker) {
        const bool connected = state.connection_state != tt::tt_fabric::connection_interface::unused_connection_value;
        credit_worker<CH>(worker_interface<CH>(), returned, connected);
    } else {
        credit_upstream(state.upstream_free_slots_addr, n);
    }
}

// The loop-local VC's worker channel, its state on the loop's copy (visited every pass): send if it has a packet and the
// remote receiver channel has room, take completions (and first-level acks), return the slots to the worker. (Its connection is checked at the
// top of the pass: check_worker_connection.)
template <size_t CH>
FORCE_INLINE void visit_worker(
    SenderChannelState<CH>& state,
    bool has_packet,
    RemoteReceiverChannelProgress<SenderChannel<CH>::vc>& remote,
    bool& send_yielded) {
    SenderChannelProgress& progress = state.progress;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool sent = false;
    if (has_packet && remote.has_room(SenderChannel<CH>::remote_free_slots_to_send)) {
        sent = send<CH>(
            progress.next_slot_addr, remote, progress.sent, progress.completed, completions, sent_acks, send_yielded);
    }
    // The visit's counters, unless the send read them, as one read group: completions, and acks (first-level-ack VCs).
    // Completions before acks, so the acks seen are never fewer than the completions seen. Whether to read each is
    // known before any is read.
    const bool read_completions = !sent && progress.sent != progress.completed;
    const bool read_acks = SenderChannel<CH>::first_level_ack && !sent && progress.sent != progress.first_level_acked;
    uint32_t reported_completions = 0;
    uint32_t reported_acks = sent_acks;
    begin_read_group();
    if (read_completions) {
        reported_completions =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::completion_counter_addr);
    }
    if (read_acks) {
        reported_acks =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::first_level_ack_counter_addr);
    }
    end_read_group();
    if (read_completions) {
        completions = reported_completions - progress.completed;
    }
    if (completions != 0) {
        remote.add_free_slots(completions);
        progress.completed += completions;
        if constexpr (!SenderChannel<CH>::first_level_ack) {
            return_slots<CH>(state, completions, progress.completed);
        }
    }
    if constexpr (SenderChannel<CH>::first_level_ack) {
        if ((sent || read_acks) && reported_acks != progress.first_level_acked) {
            const uint32_t acks = reported_acks - progress.first_level_acked;
            progress.first_level_acked = reported_acks;
            return_slots<CH>(state, acks, progress.first_level_acked);
        }
    }
}

// A channel with work whose state stays in memory (a forwarding channel, or a later VC's worker channel:
// SenderChannelStates): send if it has a packet and the remote receiver channel has room, take completions (and acks),
// return the slots to the producer, and keep the channel's bit in `channels_in_flight` equal to "packets not yet
// completed (or acked)". The state's fields are taken into locals where first needed and written back where they change
// (the send's read groups would make the compiler store and reload memory state around them).
template <size_t CH>
FORCE_INLINE void visit_channel_with_work(
    SenderChannelState<CH>& state,
    uint32_t& channels_in_flight,
    bool has_packet,
    RemoteReceiverChannelProgress<SenderChannel<CH>::vc>& remote,
    bool& send_yielded) {
    uint32_t sent = state.progress.sent;
    uint32_t completed = state.progress.completed;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool did_send = false;
    if (has_packet && remote.has_room(SenderChannel<CH>::remote_free_slots_to_send)) {
        uint32_t next_slot_addr = state.progress.next_slot_addr;
        did_send = send<CH>(next_slot_addr, remote, sent, completed, completions, sent_acks, send_yielded);
        if (did_send) {
            state.progress.next_slot_addr = next_slot_addr;
            state.progress.sent = sent;
        }
    }
    if (!did_send && sent != completed) {
        completions =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::completion_counter_addr) - completed;
    }
    if (completions != 0) {
        remote.add_free_slots(completions);
        completed += completions;
        state.progress.completed = completed;
        if constexpr (!SenderChannel<CH>::first_level_ack) {
            return_slots<CH>(state, completions, completed);
        }
    }
    bool busy = sent != completed;
    if constexpr (SenderChannel<CH>::first_level_ack) {
        if (sent != state.progress.first_level_acked) {
            const uint32_t reported_acks =
                did_send ? sent_acks
                         : *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                               SenderChannel<CH>::first_level_ack_counter_addr);
            const uint32_t acks = reported_acks - state.progress.first_level_acked;
            if (acks != 0) {
                state.progress.first_level_acked += acks;
                return_slots<CH>(state, acks, state.progress.first_level_acked);
            }
        }
        busy = busy || sent != state.progress.first_level_acked;
    }
    channels_in_flight = busy ? (channels_in_flight | (1u << CH)) : (channels_in_flight & ~(1u << CH));
    keep_in_memory(state);
}

// ---------------------------------------------------------------------------------------------------------------------
// Where the channel states live, and the loop
// ---------------------------------------------------------------------------------------------------------------------

// Every serviced VC's send state (vcs.get<VC>()) and every forwarding channel's state (forwarding.get<CH>()), and where
// they live. The loop-local VC's send state (its remote receiver progress and its worker channel's state) is copied into
// a local of run()'s loop, so the compiler can keep it in registers (loopvars.py --headroom checks). Every other VC's
// send state and every forwarding channel's state stays in memory: such a channel is visited only when the pass's reads
// show it work (it holds a packet, or has packets in flight: channels_in_flight), its fields are taken into locals where
// first needed and written back where they change, and its state is kept in memory after the visit (keep_in_memory),
// so it does not take the loop-local VC's registers on its packet path (V4_SENDER_OWNERSHIP.md §8). A one-ERISC router
// keeps every state in memory and visits the loop-local VC's in place (router_single.hpp). The accessors are templates
// on the VC so that an absent VC's state is named only in template-dependent code.
struct SenderChannelStates {
    Each<sender_vcs, SenderVcState> vcs;
    Each<forwarding_channels, SenderChannelState> forwarding;

    // The VC whose send state run() carries as a loop local: the lowest VC this router sends on (channel trimming can
    // leave a router sending on VC2 and not VC0, V4_TRIM.md). 0 where it sends on none: then nothing is visited.
    static constexpr size_t loop_local_vc = sender_vcs != 0 ? lowest_bit(sender_vcs) : 0;

    // Whether VC's channels are visited every pass (the loop-local VC's), or only when they show work.
    template <size_t VC>
    static constexpr bool visited_every_pass = VC == loop_local_vc;

    // The loop-local VC's send state in place: what run() copies at its start; a one-ERISC router visits on it in place.
    // Without a sender VC (sender_vcs == 0), the states themselves: no visit uses it.
    template <size_t VC = loop_local_vc>
    FORCE_INLINE auto& loop_local_progress() {
        if constexpr (((sender_vcs >> VC) & 1u) != 0) {
            return vcs.template get<VC>();
        } else {
            return *this;
        }
    }

    // VC's send state for its visit: the loop's copy for the loop-local VC, the state in memory otherwise.
    template <size_t VC, typename LoopCopy>
    FORCE_INLINE SenderVcState<VC>& progress_for(LoopCopy& loop_copy) {
        if constexpr (VC == loop_local_vc) {
            return loop_copy;
        } else {
            return vcs.template get<VC>();
        }
    }

    // After a later VC's visit: its remote receiver progress put back where it changed (`remote`, taken as `taken` at the
    // visit's start), and its state kept in memory. (The loop-local VC is visited on its state in place.)
    template <size_t VC>
    FORCE_INLINE void after_visit(
        SenderVcState<VC>& send_state,
        const RemoteReceiverChannelProgress<VC>& taken,
        const RemoteReceiverChannelProgress<VC>& remote) {
        static_assert(VC != loop_local_vc);
        if (remote != taken) {
            send_state.remote = remote;
        }
        keep_in_memory(send_state);
    }
};

// The forwarding channels in MASK that have a packet (`holding`) or packets in flight (`channels_in_flight`), lowest
// channel first.
template <uint32_t MASK, typename Remote>
FORCE_INLINE void visit_forwarding_channels(
    SenderChannelStates& states,
    uint32_t holding,
    uint32_t& channels_in_flight,
    Remote& remote,
    bool& send_yielded) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((holding >> CH) & 1u) != 0;
        if (has_packet || ((channels_in_flight >> CH) & 1u) != 0) {
            visit_channel_with_work<CH>(
                states.forwarding.template get<CH>(), channels_in_flight, has_packet, remote,
                send_yielded);
        }
        visit_forwarding_channels<without_lowest_bit(MASK)>(
            states, holding, channels_in_flight, remote, send_yielded);
    }
}

// VC's channels against its remote receiver progress (`remote`): its worker channel (visited every pass where VC is the
// loop-local VC, otherwise when it holds a packet or has packets in flight), then its forwarding channels, skipped with
// one test when none holds a packet or has packets in flight.
template <size_t VC, typename Remote>
FORCE_INLINE void visit_channels(
    SenderChannelStates& states,
    SenderVcState<VC>& send_state,
    Remote& remote,
    uint32_t holding,
    uint32_t& channels_in_flight,
    bool& send_yielded) {
    if constexpr (SenderVc<VC>::worker_channels != 0) {
        constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
        const bool has_packet = ((holding >> CH) & 1u) != 0;
        if constexpr (SenderChannelStates::visited_every_pass<VC>) {
            visit_worker<CH>(send_state.worker, has_packet, remote, send_yielded);
        } else if (has_packet || ((channels_in_flight >> CH) & 1u) != 0) {
            visit_channel_with_work<CH>(send_state.worker, channels_in_flight, has_packet, remote, send_yielded);
        }
    }
    if constexpr (SenderVc<VC>::forwarding_channels != 0) {
        if (((holding | channels_in_flight) & SenderVc<VC>::forwarding_channels) != 0) {
            visit_forwarding_channels<SenderVc<VC>::forwarding_channels>(
                states, holding, channels_in_flight, remote, send_yielded);
        }
    }
}

// VC's visit, on its send state where SenderChannelStates places it: its free-slots registers (when not read with the
// pass's first read group), then its channels. The loop-local VC is visited every pass on its state in place. A later
// VC is skipped with one test when none of its channels holds a packet or has packets in flight; otherwise its remote
// receiver progress is taken into a local and put back where it changed (after_visit). None where VC has no serviced
// sender channel.
template <size_t VC, typename LoopCopy>
FORCE_INLINE void visit_vc(
    LoopCopy& loop_copy,
    SenderChannelStates& states,
    uint32_t holding_at_pass_start,
    uint32_t& channels_in_flight,
    bool& send_yielded) {
    if constexpr (SenderVc<VC>::sender_channels != 0) {
        uint32_t holding = holding_at_pass_start;
        if constexpr (!SenderVc<VC>::reads_free_slots_at_pass_start) {
            holding = channels_with_packets<SenderVc<VC>::sender_channels>();
        }
        SenderVcState<VC>& send_state = states.template progress_for<VC>(loop_copy);
        if constexpr (SenderChannelStates::visited_every_pass<VC>) {
            visit_channels<VC>(states, send_state, send_state.remote, holding, channels_in_flight, send_yielded);
        } else {
            if (((holding | channels_in_flight) & SenderVc<VC>::sender_channels) == 0) {
                return;
            }
            const RemoteReceiverChannelProgress<VC> taken = send_state.remote;
            RemoteReceiverChannelProgress<VC> remote = taken;
            visit_channels<VC>(states, send_state, remote, holding, channels_in_flight, send_yielded);
            states.template after_visit<VC>(send_state, taken, remote);
        }
    }
}

// The forwarding channels in MASK: each one's upstream router's free-slots register address, as its interface cached it
// when the upstream router connected.
template <uint32_t MASK>
FORCE_INLINE void init_forwarding(SenderChannelStates& states) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        states.forwarding.template get<CH>().upstream_free_slots_addr = worker_interface<CH>().cached_worker_semaphore_address;
        init_forwarding<without_lowest_bit(MASK)>(states);
    }
}

// Packets sent so far on the forwarding channels in MASK.
template <uint32_t MASK>
FORCE_INLINE uint32_t forwarded(SenderChannelStates& states) {
    if constexpr (MASK != 0) {
        return states.forwarding.template get<lowest_bit(MASK)>().progress.sent +
               forwarded<without_lowest_bit(MASK)>(states);
    } else {
        return 0;
    }
}

// Packets sent so far on VC's worker channel (0 where it has none).
template <size_t VC, typename LoopCopy>
FORCE_INLINE uint32_t sent_by_worker(LoopCopy& loop_copy, SenderChannelStates& states) {
    if constexpr (SenderVc<VC>::worker_channels != 0) {
        return states.template progress_for<VC>(loop_copy).worker.progress.sent;
    } else {
        return 0;
    }
}

// Packets sent so far on every channel (a round attempted a send if this changed or a send yielded; read by the idle
// context switch only).
template <typename LoopCopy>
FORCE_INLINE uint32_t total_sent(LoopCopy& loop_copy, SenderChannelStates& states) {
    return forwarded<forwarding_channels>(states) + sent_by_worker<0>(loop_copy, states) +
           sent_by_worker<1>(loop_copy, states) + sent_by_worker<2>(loop_copy, states);
}

// Whether a sender channel holds a packet (telemetry, once per round; the pass's register groups).
FORCE_INLINE bool any_channel_holds_packet() {
    return (channels_with_packets<channels_read_at_pass_start>() |
            channels_with_packets<sender_channels & ~channels_read_at_pass_start>()) != 0;
}

// Something else used the write command buffer (router commands may have run the base firmware; another worker
// channel had a connection event): program the credit command of the worker channel that presets it again while its
// worker is connected (a no-op unless it owns the preset command buffer).
template <typename LoopCopy>
FORCE_INLINE void restore_worker_credit_preset(
    LoopCopy& loop_copy, SenderChannelStates& states) {
    if constexpr ((SenderVc<0>::worker_channels & 1u) != 0) {
        if (states.template progress_for<0>(loop_copy).worker.connection_state !=
            tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_worker_credit(worker_interface<0>());
        }
    }
}

// VC's worker channel's connection (every pass, by rule), from the semaphores the pass's read group read
// (`connection_semaphores[CH]`). A worker channel that does not own the credit preset may use the write command buffer
// on a connection event (its library connect), so the preset is restored after one.
template <size_t VC, typename LoopCopy>
FORCE_INLINE void check_worker_connection(
    LoopCopy& loop_copy, SenderChannelStates& states, const uint32_t* connection_semaphores) {
    if constexpr (SenderVc<VC>::worker_channels != 0) {
        constexpr size_t CH = lowest_bit(SenderVc<VC>::worker_channels);
        if (check_connection<CH>(
                states.template progress_for<VC>(loop_copy).worker,
                worker_interface<CH>(),
                connection_semaphores[CH])) {
            if constexpr (!WorkerCredit<CH>::owns_write_buffer) {
                restore_worker_credit_preset(loop_copy, states);
            }
        }
    }
}

// One pass over the sender channels: the first read group, each worker channel's connection, then one visit per VC.
// The body of run()'s loop; a single-ERISC router runs it as the sender half of each pass over its state in memory
// (router_single.hpp).
template <typename LoopCopy>
FORCE_INLINE void pass(
    LoopCopy& loop_copy,
    SenderChannelStates& states,
    uint32_t& channels_in_flight,
    bool& send_yielded) {
    uint32_t connection_semaphores[worker_channels == 0 ? 1 : 32 - __builtin_clz(worker_channels)];
    const uint32_t holding =
        channels_with_packets<channels_read_at_pass_start, worker_channels>(connection_semaphores);
    check_worker_connection<0>(loop_copy, states, connection_semaphores);
    check_worker_connection<1>(loop_copy, states, connection_semaphores);
    check_worker_connection<2>(loop_copy, states, connection_semaphores);
    visit_vc<0>(loop_copy, states, holding, channels_in_flight, send_yielded);
    visit_vc<1>(loop_copy, states, holding, channels_in_flight, send_yielded);
    visit_vc<2>(loop_copy, states, holding, channels_in_flight, send_yielded);
}

// The sender loop: rounds of passes over the sender channels, with router commands between runs of rounds. A function
// of its own: its code depends only on this body, and once-per-round work is out of line (router_round.hpp).
// ---------------------------------------------------------------------------------------------------------------------
// Headroom probe (study only, not a design): synthetic state that the guard's headroom test adds to measure what moves.
// ---------------------------------------------------------------------------------------------------------------------

// Values carried from pass to pass: in registers at each pass's end, opaque to GCC (it cannot fold them).
FORCE_INLINE void synthetic_carried(uint32_t& value) { asm volatile("" : "+r"(value)); }
FORCE_INLINE void synthetic_carried_pair(uint32_t& a, uint32_t& b) { asm volatile("" : "+r"(a), "+r"(b)); }
FORCE_INLINE void synthetic_carried_five(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e) {
    asm volatile("" : "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e));
}

__attribute__((noinline)) void run() {
    uint32_t synthetic0 = 1;
    uint32_t synthetic1 = 2;
    SenderChannelStates states;
    init_forwarding<forwarding_channels>(states);
    auto loop_copy = states.loop_local_progress();  // carried through the loop (SenderChannelStates)
    uint32_t channels_in_flight = 0;  // the channels kept in memory with packets not yet completed (or acked)

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_preset(loop_copy, states);
        while (next_round()) {
            round.send_yielded = false;
            round.sent_at_round_start = total_sent(loop_copy, states);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(loop_copy, states, channels_in_flight, round.send_yielded);
                synthetic_carried_pair(synthetic0, synthetic1);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    total_sent(loop_copy, states) != round.sent_at_round_start);
            if constexpr (telemetry::any_option) {
                telemetry::round_end<true, false>([] { return any_channel_holds_packet(); }, [] { return false; });
            }
            if (firmware_ran) {
                restore_worker_credit_preset(loop_copy, states);
            }
        }
    }
}

}  // namespace fabric_router::sender
