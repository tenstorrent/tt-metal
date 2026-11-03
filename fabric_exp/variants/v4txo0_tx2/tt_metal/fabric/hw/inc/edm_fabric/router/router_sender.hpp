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
// State, by placement class: the worker channels' progress and connection state, each VC's remote receiver progress
// and which forwarding channels have packets in flight, loop locals; each forwarding channel's progress and its
// producer's address, in memory between its visits; the worker interfaces, in memory (they escape to out-of-line
// calls); ring bounds, registers and counters, compile-time. The block before pass() states where each lives and why.
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

// Sender channel CH.
template <size_t CH>
struct SenderChannel {
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first_slot_addr = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t free_slots_stream = sender_free_slots_stream<CH>;
    // Counters the far router's receiver writes over the link.
    static constexpr uint32_t completion_counter_addr = to_sender_remote_completion_counters_base_address + CH * 4;
    static constexpr uint32_t first_level_ack_counter_addr = to_sender_remote_ack_counters_base_address + CH * 4;
    // A worker channel's connection semaphore (compile time; the worker interface holds the same address).
    static constexpr uint32_t connection_semaphore_addr = sender_connection_semaphore_addrs[CH];
    static constexpr size_t vc = CH < VC1_LOCAL_CHANNEL_START ? 0 : CH < VC2_LOCAL_CHANNEL_START ? 1 : 2;
    static constexpr bool first_level_ack = is_vc_first_level_ack_enabled[vc];
    static constexpr uint32_t remote_free_slots_to_send = remote_free_slots_to_send_by(CH);
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
// In Tensix mux mode the mux is the only producer: no router forwards into this one. (Channel trimming may leave the
// mux's worker channel unserviced too.)
static_assert(!tensix_mux_mode || all_forwarding == 0);
// VC2's one channel, a worker channel (its own worker connects to it).
constexpr uint32_t vc2_worker = channels_in(VC2_LOCAL_CHANNEL_START, ACTUAL_VC2_SENDER_CHANNELS);
constexpr bool has_vc2_worker = vc2_worker != 0;
static_assert(popcount(vc2_worker) <= 1);
// The worker channels: VC0's channel 0 and VC2's, each where serviced.
constexpr uint32_t worker_channels = (vc0_channels & 0x1u) | vc2_worker;
// Which remote receiver channels this router fills: a VC's, when any of its sender channels is serviced.
constexpr bool sends_vc0 = vc0_channels != 0;
constexpr bool sends_vc1 = vc1_forwarding != 0;
constexpr bool sends_vc2 = has_vc2_worker;

// The far router's receiver channel of VC: its slots and its packets-received register (on the far router).
template <size_t VC>
struct RemoteReceiverChannel {
    static constexpr size_t rx =  // the host's index of the channel
        VC == 0 ? VC0_RECEIVER_CHANNEL : VC == 1 ? VC1_RECEIVER_CHANNEL : VC2_RECEIVER_CHANNEL;
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[rx];
    static constexpr uint32_t first_slot_addr =
        eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[rx]>::base_address;
    static constexpr uint32_t last_slot_addr = first_slot_addr + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[rx];
};

// ---------------------------------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------------------------------

// This router's progress into a remote receiver channel: its slots known free, and the next slot to fill.
struct RemoteReceiverChannelProgress {
    uint32_t free_slots;
    uint32_t next_slot_addr;
};

// A remote receiver channel's progress before the first send: every slot free.
template <size_t VC>
constexpr RemoteReceiverChannelProgress remote_receiver_channel_progress_start() {
    return {RemoteReceiverChannel<VC>::num_slots, RemoteReceiverChannel<VC>::first_slot_addr};
}

// A forwarding channel's record.
struct ForwardingRecord {
    uint32_t next_slot_addr;            // next slot to send from
    uint32_t sent;                      // packets sent
    uint32_t completed;                 // completions taken
    uint32_t first_level_acked;         // first-level acks taken (first-level-ack VCs only)
    uint64_t upstream_free_slots_addr;  // the upstream router's free-slots register, cached when it connected
};

// A worker channel's state: its next slot and counts, and the value its connection semaphore (SenderChannel<CH>::
// connection_semaphore_addr) holds while nothing changes.
struct Worker {
    uint32_t next_slot_addr;
    uint32_t sent;
    uint32_t completed;
    uint32_t first_level_acked;  // first-level acks taken (first-level-ack VCs only)
    uint32_t connection_state;
};

// The forwarding channels' records, in memory between visits. (Which of them have packets in flight is a loop local of
// its own, `forwarding_in_flight`: read every pass, it must not share this object.)
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
    RemoteReceiverChannelProgress& remote,
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

    internal_::eth_send_packet_bytes_unsafe(SENDER_TXQ, slot_addr, remote.next_slot_addr, size);
    remote.next_slot_addr = remote.next_slot_addr == RemoteReceiverChannel<VC>::last_slot_addr
                                ? RemoteReceiverChannel<VC>::first_slot_addr
                                : remote.next_slot_addr + channel_buffer_size;
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

    remote.free_slots--;
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
template <typename WorkerInterface>
__attribute__((noinline)) void preset_worker_credit(const WorkerInterface& interface) {
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
template <size_t CH, typename WorkerInterface>
__attribute__((noinline)) void credit_worker(const WorkerInterface& interface, uint32_t returned, bool connected) {
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

// A worker channel's connection, checked every pass (by rule) from the semaphore the pass's group read: a change (open
// or close request) is handled out of line.
// Returns whether it changed.
template <size_t CH, typename WorkerInterface>
FORCE_INLINE bool check_connection(Worker& worker, WorkerInterface& interface, uint32_t semaphore) {
    if (semaphore != worker.connection_state) {
        const uint32_t returned = SenderChannel<CH>::first_level_ack ? worker.first_level_acked : worker.completed;
        worker.connection_state = worker_connection_event<CH>(interface, semaphore, worker.connection_state, returned);
        return true;
    }
    return false;
}

// A worker channel (VC0's channel 0, or VC2's): send, take completions (and first-level acks), credit the worker. (Its
// connection is checked at the top of the pass: check_worker_connections.)
template <size_t CH, typename WorkerInterface>
FORCE_INLINE void visit_worker(
    Worker& worker,
    WorkerInterface& interface,
    bool has_packet,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    using namespace tt::tt_fabric::connection_interface;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool sent = false;
    if (has_packet && remote.free_slots >= SenderChannel<CH>::remote_free_slots_to_send) {
        sent = send<CH>(
            worker.next_slot_addr, remote, worker.sent, worker.completed, completions, sent_acks, send_yielded);
    }
    // The visit's counters, unless the send read them, as one group so their loads overlap: completions, and acks
    // (first-level-ack VCs). Completions before acks, so the acks seen are never fewer than the completions seen.
    // Whether to read each is known before any is read.
    const bool read_completions = !sent && worker.sent != worker.completed;
    const bool read_acks = SenderChannel<CH>::first_level_ack && !sent && worker.sent != worker.first_level_acked;
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
        completions = reported_completions - worker.completed;
    }
    const bool connected = worker.connection_state != unused_connection_value;
    if (completions != 0) {
        remote.free_slots += completions;
        worker.completed += completions;
        if constexpr (!SenderChannel<CH>::first_level_ack) {
            credit_worker<CH>(interface, worker.completed, connected);
        }
    }
    if constexpr (SenderChannel<CH>::first_level_ack) {
        if ((sent || read_acks) && reported_acks != worker.first_level_acked) {
            worker.first_level_acked = reported_acks;
            credit_worker<CH>(interface, worker.first_level_acked, connected);
        }
    }
}

// A forwarding channel with work: send if it has a packet and the remote receiver channel has room, take completions
// (and acks), credit the upstream router, and keep the channel's bit in `forwarding_in_flight` equal to "packets not
// yet completed".
template <size_t CH>
FORCE_INLINE void visit_forwarding(
    ForwardingTable& table,
    uint32_t& forwarding_in_flight,
    bool has_packet,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    ForwardingRecord& state = table.records[record_index<CH>()];
    uint32_t sent = state.sent;
    uint32_t completed = state.completed;
    uint32_t completions = 0;
    uint32_t sent_acks = 0;  // the ack count the send read (first-level-ack VCs)
    bool did_send = false;
    if (has_packet && remote.free_slots >= SenderChannel<CH>::remote_free_slots_to_send) {
        uint32_t next_slot_addr = state.next_slot_addr;
        did_send = send<CH>(next_slot_addr, remote, sent, completed, completions, sent_acks, send_yielded);
        if (did_send) {
            state.next_slot_addr = next_slot_addr;
            state.sent = sent;
        }
    }
    if (!did_send && sent != completed) {
        completions =
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SenderChannel<CH>::completion_counter_addr) - completed;
    }
    if (completions != 0) {
        remote.free_slots += completions;
        completed += completions;
        state.completed = completed;
        if constexpr (!SenderChannel<CH>::first_level_ack) {
            credit_upstream(state.upstream_free_slots_addr, completions);
        }
    }
    bool busy = sent != completed;
    if constexpr (SenderChannel<CH>::first_level_ack) {
        if (sent != state.first_level_acked) {
            const uint32_t reported_acks =
                did_send ? sent_acks
                         : *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                               SenderChannel<CH>::first_level_ack_counter_addr);
            const uint32_t acks = reported_acks - state.first_level_acked;
            if (acks != 0) {
                state.first_level_acked += acks;
                credit_upstream(state.upstream_free_slots_addr, acks);
            }
        }
        busy = busy || sent != state.first_level_acked;
    }
    forwarding_in_flight = busy ? (forwarding_in_flight | (1u << CH)) : (forwarding_in_flight & ~(1u << CH));
    keep_in_memory(state);
}

// Visits the forwarding channels in MASK that have a packet (`holding`) or packets in flight
// (`forwarding_in_flight`), lowest channel first.
template <uint32_t MASK>
FORCE_INLINE void visit_forwarding_channels(
    ForwardingTable& table,
    uint32_t holding,
    uint32_t& forwarding_in_flight,
    RemoteReceiverChannelProgress& remote,
    bool& send_yielded) {
    if constexpr (MASK != 0) {
        constexpr size_t CH = lowest_bit(MASK);
        const bool has_packet = ((holding >> CH) & 1u) != 0;
        if (has_packet || ((forwarding_in_flight >> CH) & 1u) != 0) {
            visit_forwarding<CH>(table, forwarding_in_flight, has_packet, remote, send_yielded);
        }
        visit_forwarding_channels<without_lowest_bit(MASK)>(table, holding, forwarding_in_flight, remote, send_yielded);
    }
}

// One VC's forwarding channels (MASK), skipped with one test when none has a packet or a packet in flight. (Remote is a
// parameter so that a VC this router does not run passes its absent remote receiver progress.)
template <uint32_t MASK, typename Remote>
FORCE_INLINE void visit_forwarding_group(
    ForwardingTable& table, uint32_t holding, uint32_t& forwarding_in_flight, Remote& remote, bool& send_yielded) {
    if constexpr (MASK != 0) {
        if (((holding | forwarding_in_flight) & MASK) != 0) {
            visit_forwarding_channels<MASK>(table, holding, forwarding_in_flight, remote, send_yielded);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Where the channel states live, and the loop
// ---------------------------------------------------------------------------------------------------------------------

// Where the sender's state lives. The worker channels' progress and connection state (Workers) and each VC's remote
// receiver progress (RemoteReceiverChannelProgresses) are run()'s loop locals, kept in registers where they fit. Each
// forwarding channel's record stays in memory between its visits (ForwardingTable; keep_in_memory after each visit),
// so it does not take registers from the state every pass uses. The send path stays out of memory: its read groups
// would store and reload memory state around them (V4_STATE.md §11). The worker interfaces are the loop's own copies
// (WorkerConnections), in memory. A one-ERISC router keeps every state in its state record (router_single.hpp).

// The worker channels: VC0's channel 0 and VC2's channel, each where present.
struct Workers {
    [[no_unique_address]] Present<has_worker, Worker, 0> vc0;
    [[no_unique_address]] Present<has_vc2_worker, Worker, 2> vc2;
};

// The remote receiver channels' progress, one per VC this router sends on: VC0's (its worker and forwarding channels
// send into it), VC1's, VC2's.
struct RemoteReceiverChannelProgresses {
    [[no_unique_address]] Present<sends_vc0, RemoteReceiverChannelProgress, 0> vc0;
    [[no_unique_address]] Present<sends_vc1, RemoteReceiverChannelProgress, 1> vc1;
    [[no_unique_address]] Present<sends_vc2, RemoteReceiverChannelProgress, 2> vc2;
};

// The worker channels' interfaces, the loop's own copies (connection events and the workers' credit addresses): passed
// to out-of-line calls, so in memory.
template <typename Vc0Interface, typename Vc2Interface>
struct WorkerConnections {
    [[no_unique_address]] Vc0Interface vc0;
    [[no_unique_address]] Vc2Interface vc2;
};

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
        table.records[record_index<CH>()] = {
            SenderChannel<CH>::first_slot_addr, 0, 0, 0, upstream.addr[record_index<CH>()]};
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
    return {SenderChannel<CH>::first_slot_addr, 0, 0, 0, tt::tt_fabric::connection_interface::unused_connection_value};
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

// The remote receiver channels' progress before the first send: every slot free.
template <size_t VC0 = 0, size_t VC1 = 1, size_t VC2 = 2>
FORCE_INLINE RemoteReceiverChannelProgresses remote_progresses_start() {
    RemoteReceiverChannelProgresses remote{};
    if constexpr (sends_vc0) {
        remote.vc0 = remote_receiver_channel_progress_start<VC0>();
    }
    if constexpr (sends_vc1) {
        remote.vc1 = remote_receiver_channel_progress_start<VC1>();
    }
    if constexpr (sends_vc2) {
        remote.vc2 = remote_receiver_channel_progress_start<VC2>();
    }
    return remote;
}

// Something else used the write command buffer (router commands may have run the base firmware; VC2's worker had a
// connection event): program VC0's worker credit command buffer again while a worker is connected (a no-op unless it
// owns the preset command buffer).
template <typename W, typename Connections>
FORCE_INLINE void restore_worker_credit_preset(const W& workers, const Connections& connections) {
    if constexpr (has_worker) {
        if (workers.vc0.connection_state != tt::tt_fabric::connection_interface::unused_connection_value) {
            preset_worker_credit(connections.vc0);
        }
    }
}

// The worker channels' connections (every pass, by rule), from the semaphores the pass's group read
// (`connection_semaphores[CH]`).
// (W and the defaulted parameter are template parameters so that an absent worker's members are named only where
// present: see Present, router_common.hpp.)
template <size_t CH2 = VC2_LOCAL_CHANNEL_START, typename W, typename Connections>
FORCE_INLINE void check_worker_connections(
    W& workers, Connections& connections, const uint32_t* connection_semaphores) {
    if constexpr (has_worker) {
        check_connection<0>(workers.vc0, connections.vc0, connection_semaphores[0]);
    }
    if constexpr (has_vc2_worker) {
        if (check_connection<CH2>(workers.vc2, connections.vc2, connection_semaphores[CH2])) {
            restore_worker_credit_preset(workers, connections);
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
    uint32_t& forwarding_in_flight,
    RemoteReceiverChannelProgresses& remote,
    bool& send_yielded) {
    uint32_t connection_semaphores[worker_channels == 0 ? 1 : 32 - __builtin_clz(worker_channels)];
    const uint32_t holding = channels_with_packets<vc0_channels | vc2_worker, worker_channels>(connection_semaphores);
    check_worker_connections(workers, connections, connection_semaphores);
    if constexpr (has_worker) {
        visit_worker<0>(workers.vc0, connections.vc0, (holding & 0x1u) != 0, remote.vc0, send_yielded);
    }
    if constexpr (vc0_forwarding != 0) {
        visit_forwarding_group<vc0_forwarding>(forwarding, holding, forwarding_in_flight, remote.vc0, send_yielded);
    }
    if constexpr (vc1_forwarding != 0) {
        // VC1's registers are read here, after VC0's visits, not with VC0's group: read at the top of the pass, a VC1
        // packet landing during VC0's visits waits a pass (V4_SENDER_SHAPE.md).
        const uint32_t holding_vc1 = channels_with_packets<vc1_forwarding>();
        visit_forwarding_group<vc1_forwarding>(forwarding, holding_vc1, forwarding_in_flight, remote.vc1, send_yielded);
    }
    if constexpr (has_vc2_worker) {
        visit_worker<VC2_LOCAL_CHANNEL_START>(
            workers.vc2, connections.vc2, (holding & vc2_worker) != 0, remote.vc2, send_yielded);
    }
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

template <typename WorkerInterfaces>
__attribute__((noinline)) void run(WorkerInterfaces& interfaces, const UpstreamCredits& upstream) {
    uint32_t synthetic0 = 1;
    uint32_t synthetic1 = 2;
    auto connections = worker_connections(interfaces);
    RemoteReceiverChannelProgresses remote = remote_progresses_start();
    Workers workers = workers_start(connections);
    ForwardingTable forwarding;
    init_forwarding<all_forwarding>(forwarding, upstream);
    uint32_t forwarding_in_flight = 0;  // the forwarding channels with packets not yet completed (or acked)

    RoundCounters round;
    while (!terminating()) {
        if (!start_running()) {
            continue;
        }
        restore_worker_credit_preset(workers, connections);
        while (next_round()) {
            round.send_yielded = false;
            round.sent_at_round_start = total_sent(workers, forwarding);
            if constexpr (telemetry::any_option) {
                telemetry::round_start();
            }
            for (size_t n = 0; n < iterations_between_ctx_switch_and_teardown_checks; n++) {
                pass(workers, connections, forwarding, forwarding_in_flight, remote, round.send_yielded);
                synthetic_carried(synthetic0);
                synthetic_carried(synthetic1);
            }
            const bool firmware_ran = end_round(
                round,
                !IDLE_CONTEXT_SWITCHING || round.send_yielded ||
                    total_sent(workers, forwarding) != round.sent_at_round_start);
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
