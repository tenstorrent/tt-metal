// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Part of fabric_erisc_router_v3.cpp, which includes it once, at file scope: it uses that file's definitions and is
// not a standalone header.

// ============================================================================
// sender_erisc: ERISC 0 of a two-ERISC router (sender channels only)
// ============================================================================
// When `runs`, ERISC 0 runs run_main_loop, with its own state, in place of the generic loop. One pass:
//   1. Read VC0's free-slots registers as one fenced group (channels 0-3, as the generic loop does).
//   2. Worker channel (channel 0), specialized at compile time: send if it has a packet and the far ring has room,
//      take completions, return credits to the worker, poll the connection semaphore (open or tear down the
//      connection out of line when it changes).
//   3. VC0's forwarding channels: when none has work (an unsent packet, or packets in flight), one test skips them all
//      (all_channels_free). Otherwise each channel with work is visited, in channel order, by code specialized for it at
//      compile time and inlined here. "All slots free" is the generic loop's test for each channel (all_slots_free).
//   4. VC1 (when serviced): its free-slots registers as a second group, then the same test and visits.
// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
// and off the pass's path: worker connection open and close, and the static-connection wait at init.
//
// run_main_loop is a function of its own and holds only per-pass work. What it does once per round or rarely is in small
// functions of its own (termination_requested, next_round, end_round, run_router_command): the termination and
// router-command tests, the heartbeat, the idle context switch with its handshake, and the pause handshake. So edits to
// that code do not reach the pass loop's code (as on ERISC 1, RESULTS §97).
//
// State held across passes: per VC, the far ring's free count and slot pointer; for the worker, its slot, sent and
// completed counts, the connection semaphore pointer and the value it holds while nothing changes; one word with a
// bit per forwarding channel that has packets in flight. A forwarding channel's counts, slot and upstream credit
// address are a record its visit reads and writes. Addresses and slot bounds known at compile time are constants in
// the code, not state.
//
// Init (kernel_main): this ERISC sets up its own (sender) side. The receiver side (downstream router adapters, receiver
// channel buffers) is left to ERISC 1 (receiver_erisc::run_kernel or kernel_main), which has the same compile-time and
// runtime args and performs those writes in the same init phase. The worker credit writes (credit_worker,
// worker_connection_event) restate the worker interface's notify_worker_of_read_counter_update and
// copy_read_counter_to_worker_location_info for channel 0, with its credit count kept here: keep them in step.
//
// Grouped reads (see "Grouped reads"): the grouped free-slots read, and send_packet's pre-send and post-send groups,
// which are sender_channel_try_send's. Registers are accessed through the provided helpers: the TX queue through
// eth_txq_reg_read and eth_send_packet_bytes_unsafe, the stream credit registers through increment_local_update_ptr_val
// and remote_update_ptr_val, and the NoC through noc_inline_dw_write.
//
// Same protocol as the generic loop: every packet, register update, credit, completion and connection step, in the
// same order on each channel, channels visited in the same order within a pass. Builds this program does not cover
// keep the generic loop: single-ERISC mode, ERISC 1, the speedy path, VC2, stream-register credits, the RISC data
// cache, spin-wait sends, non-Blackhole targets, a fifth VC0 sender, a forwarding-type channel 0 and a worker-type
// forwarding channel.
namespace sender_erisc {

#if defined(FABRIC_2D_VC1_SERVICED)
constexpr bool vc1_serviced = is_2d_fabric != 0;
#else
constexpr bool vc1_serviced = false;
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
constexpr bool vc2_serviced = true;
#else
constexpr bool vc2_serviced = false;
#endif

// Channels this program visits: VC0's first two (1D) or four (2D), and VC1's when VC1 is serviced; serviced ones only.
constexpr size_t vc0_visit_limit = is_2d_fabric ? 4 : 2;
constexpr size_t vc0_end =
    ACTUAL_VC0_SENDER_CHANNELS < vc0_visit_limit ? ACTUAL_VC0_SENDER_CHANNELS : vc0_visit_limit;
constexpr size_t vc1_begin = VC1_LOCAL_CHANNEL_START;
constexpr size_t vc1_end =
    vc1_serviced ? VC1_LOCAL_CHANNEL_START + ACTUAL_VC1_SENDER_CHANNELS : VC1_LOCAL_CHANNEL_START;

constexpr bool visits_on_vc0(size_t ch) {
    return ch < NUM_SENDER_CHANNELS && is_sender_channel_serviced[ch] && ch < vc0_end;
}
constexpr bool visits_on_vc1(size_t ch) {
    return ch < NUM_SENDER_CHANNELS && is_sender_channel_serviced[ch] && ch >= vc1_begin && ch < vc1_end;
}
constexpr uint32_t forwarding_channels_on_vc0() {
    uint32_t mask = 0;
    for (size_t ch = 1; ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        mask |= visits_on_vc0(ch) ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr uint32_t forwarding_channels_on_vc1() {
    uint32_t mask = 0;
    for (size_t ch = 1; ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        mask |= visits_on_vc1(ch) ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr bool has_worker = visits_on_vc0(0);
constexpr uint32_t vc0_forwarding = forwarding_channels_on_vc0();
constexpr uint32_t vc1_forwarding = forwarding_channels_on_vc1();
// VC1's far ring (receiver channel); VC0's when VC1 has no channels here, so that the type always exists.
constexpr size_t vc1_far_ring = vc1_forwarding != 0 ? VC1_RECEIVER_CHANNEL : VC0_RECEIVER_CHANNEL;

// The generic loop's visits (run_fabric_edm_main_loop), to check that this program visits the same channels.
constexpr bool generic_loop_visits_on_vc0(size_t ch) {
    return ch == 0 || ch == 1 || (is_2d_fabric && (ch == 2 || ch == 3 || (ch == 4 && ACTUAL_VC0_SENDER_CHANNELS > 4)));
}
constexpr bool generic_loop_visits_on_vc1(size_t ch) {
    return vc1_serviced && ch >= ACTUAL_VC0_SENDER_CHANNELS &&
           ch < ACTUAL_VC0_SENDER_CHANNELS + (ACTUAL_VC1_SENDER_CHANNELS > 3 ? 4 : 3);
}

constexpr bool shape_supported = [] {
    for (size_t ch = 0; ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        const bool serviced = is_sender_channel_serviced[ch];
        if ((serviced && generic_loop_visits_on_vc0(ch)) != visits_on_vc0(ch) ||
            (serviced && generic_loop_visits_on_vc1(ch)) != visits_on_vc1(ch)) {
            return false;
        }
        if (visits_on_vc0(ch) || visits_on_vc1(ch)) {
            // Channel 0 is the worker channel (connection-checked); every other visited channel is router-to-router.
            if (sender_ch_live_check_skip[ch] != (ch != 0) || !sender_channel_uses_counter_credits(ch) ||
                SENDER_NUM_BUFFERS_ARRAY[ch] == 0 || SENDER_NUM_BUFFERS_ARRAY[ch] > stream_count_mask) {
                return false;
            }
        }
    }
    return true;
}();

// Whether this ERISC runs this program.
constexpr bool runs = MY_ERISC_ID == 0 && NUM_ACTIVE_ERISCS > 1 && !any_receiver_channel_serviced &&
                      any_sender_channel_serviced && !super_speedy_mode && is_blackhole_arch && multi_txq_enabled &&
                      !ENABLE_RISC_CPU_DATA_CACHE && !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !vc2_serviced &&
                      shape_supported;

// Injection channels under bubble flow control send only with two free far slots.
constexpr uint32_t bubble_flow_control_channels() {
    uint32_t mask = 0;
    for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
        mask |= (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel[ch]) ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr uint32_t bubble_channels = bubble_flow_control_channels();

// "Every slot of channel CH is free", exactly as the generic loop decides it, so that every value of the free-slots
// register gives the same answer as there (not only values in [0, num_slots]):
//   - forwarding channels in the grouped read that the generic loop idle-skips: channel_all_slots_free (one bit test
//     when skip_idle_bit_test, else the masked compare);
//   - other channels in the grouped read (the worker channel, forwarding channels without idle skip): the count masked
//     with stream_count_mask, compared;
//   - channels outside the grouped read (VC1): the full free-slots field (get_ptr_val), compared.
constexpr bool idle_skipped(size_t ch) {
    return ch < grouped_packet_check_channels && skip_idle_per_channel && skip_idle_channel(ch);
}
constexpr bool uses_bit_test(size_t ch) { return idle_skipped(ch) && skip_idle_bit_test; }
constexpr uint32_t free_slots_compare_mask(size_t ch) {
    return ch < grouped_packet_check_channels ? stream_count_mask : (1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1;
}
template <size_t CH>
FORCE_INLINE bool all_slots_free(uint32_t raw) {
    constexpr uint32_t n = SENDER_NUM_BUFFERS_ARRAY[CH];
    static_assert(n <= stream_count_mask);
    if constexpr (idle_skipped(CH)) {
        return channel_all_slots_free<CH>(raw);
    } else {
        return (raw & free_slots_compare_mask(CH)) == n;
    }
}
// The generic loop tests every channel of `mask` the same way (a VC's forwarding channels are all in the grouped read,
// VC0's, or all outside it, VC1's; idle skip is decided for all grouped forwarding channels at once).
constexpr bool tested_alike(uint32_t mask) {
    const size_t first = __builtin_ctz(mask);
    for (size_t ch = first; ch < NUM_SENDER_CHANNELS; ch++) {
        if (((mask >> ch) & 1) && (uses_bit_test(ch) != uses_bit_test(first) ||
                                   free_slots_compare_mask(ch) != free_slots_compare_mask(first))) {
            return false;
        }
    }
    return true;
}

template <size_t... K, typename F>
FORCE_INLINE void for_each_index(std::index_sequence<K...>, F&& f) {
    (f(std::integral_constant<size_t, K>{}), ...);
}

// Calls f(ch, k) for each channel ch = BASE + k (k < 4) whose bit is set in CHANNELS, in channel order; ch and k are
// std::integral_constant, so each call is compiled for its channel.
template <uint32_t CHANNELS, size_t BASE, typename F>
FORCE_INLINE void for_each_channel(F&& f) {
    for_each_index(std::make_index_sequence<4>{}, [&](auto k) __attribute__((always_inline)) {
        constexpr size_t CH = BASE + decltype(k)::value;
        if constexpr (((CHANNELS >> CH) & 1) != 0) {
            f(std::integral_constant<size_t, CH>{}, k);
        }
    });
}

// Whether every channel in CHANNELS (channel BASE + k's free-slots value in raw[k]) has all its slots free, as
// all_slots_free decides it for each channel, in one test. The generic loop tests them all the same way:
//   - with its bit test (one shared power-of-two slot count), each value must have that count's bit set, so their AND
//     must have it;
//   - otherwise each masked value must equal its slot count, so the OR of (value XOR slot count), masked, must be 0.
template <uint32_t CHANNELS, size_t BASE>
FORCE_INLINE bool all_channels_free(const uint32_t (&raw)[4]) {
    static_assert(tested_alike(CHANNELS));
    constexpr size_t FIRST = __builtin_ctz(CHANNELS);
    if constexpr (uses_bit_test(FIRST)) {
        uint32_t all = SENDER_NUM_BUFFERS_ARRAY[FIRST];
        for_each_channel<CHANNELS, BASE>(
            [&](auto, auto k) __attribute__((always_inline)) { all &= raw[decltype(k)::value]; });
        return all != 0;
    } else {
        uint32_t differ = 0;
        for_each_channel<CHANNELS, BASE>([&](auto ch, auto k) __attribute__((always_inline)) {
            differ |= raw[decltype(k)::value] ^ SENDER_NUM_BUFFERS_ARRAY[decltype(ch)::value];
        });
        return (differ & free_slots_compare_mask(FIRST)) == 0;
    }
}

// Channel CH's slots in its L1 ring (the values the sender channel objects are initialized with).
template <size_t CH>
struct ChannelSlots {
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last = first + (num_slots - 1) * channel_buffer_size;
};

// Counters the far receiver writes over Ethernet, one word per sender channel.
FORCE_INLINE constexpr uint32_t completion_counter_addr(uint32_t ch) {
    return to_sender_remote_completion_counters_base_address + ch * sizeof(uint32_t);
}
FORCE_INLINE constexpr uint32_t ack_counter_addr(uint32_t ch) {
    return to_sender_remote_ack_counters_base_address + ch * sizeof(uint32_t);
}

// Channel CH's free-slots stream register: the producer (worker or upstream router) adds each packet it writes, and this
// ERISC adds each slot it sends from.
template <size_t CH>
constexpr uint32_t free_slots_stream = sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)];

// The far receiver ring of one VC, shared by all of that VC's sender channels.
template <size_t RX>
struct FarRing {
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[RX];
    static constexpr uint32_t first = eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[RX]>::base_address;
    static constexpr uint32_t last = first + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_received_stream = to_receiver_packets_sent_streams[RX];  // on the far receiver

    uint32_t free_slots = num_slots;
    uint32_t next_slot = first;
};

// A forwarding (router-to-router) channel's state. Read and written only by a visit that has work; its slot bounds and
// register addresses are compile-time constants of that visit.
struct ForwardingChannel {
    uint32_t slot;       // next slot to send from
    uint32_t sent;       // packets sent
    uint32_t completed;  // completions processed
    uint32_t acked;      // first-level acks processed (first-level-ack VCs only)
    uint64_t upstream_credit_addr;  // the upstream router's free-slots register, cached at init
    // STABILITY_PROBE_STATE
};

// Per-send telemetry (bandwidth and perf telemetry), when the build has any.
constexpr bool telemetry_enabled =
    FABRIC_TELEMETRY_ANY_DYNAMIC_STAT || perf_telemetry_mode != PerfTelemetryRecorderType::NONE;
struct SendTelemetry {
    PerfTelemetryRecorder* perf;
    FabricTelemetry* local;
    bool* tx_progress;
};
struct NoSendTelemetry {};
using Telemetry = std::conditional_t<telemetry_enabled, SendTelemetry, NoSendTelemetry>;
template <typename T = Telemetry>
FORCE_INLINE T make_telemetry(PerfTelemetryRecorder& perf, FabricTelemetry& local, bool& tx_progress) {
    if constexpr (std::is_same_v<T, SendTelemetry>) {
        return T{&perf, &local, &tx_progress};
    } else {
        return T{};
    }
}

// What the main loop shares with the forwarding visits.
struct ForwardingState {
    ForwardingChannel channels[NUM_SENDER_CHANNELS];  // indexed by channel; entry 0 (the worker channel) is unused
    uint32_t in_flight;                               // bit per forwarding channel with packets not yet completed
    bool attempted;                                   // a send attempt yielded this round (the TX queue was busy)
    Telemetry telemetry;
};

// The TX queue was busy after a data command (never seen in practice): spin until it is idle.
FORCE_INLINE void wait_for_sender_txq_idle() {
    while (internal_::eth_txq_is_busy(sender_txq_id)) {
    }
}

// Sends channel `ch`'s next packet. Precondition: the channel has an unsent packet and the far ring has room. The same
// steps in the same order as sender_channel_try_send's overlapped form: read the payload size and the TX-queue status
// as one group and stamp src_ch_id; a busy queue yields (returns false, nothing sent). Otherwise issue the data
// command, advance the far slot and the channel's slot, then read the queue status and the channel's completion
// counter as one group while the free count and the channel's free-slots register are updated, and increment the far
// receiver's packets-received register on the same queue. `completions` gets the completions newly reported. A send
// attempt that yields sets `attempted`; one that sends counts in `sent` (the idle context switch test reads both).
// FREE_SLOTS_STREAM is the channel's free-slots stream register.
template <uint32_t FREE_SLOTS_STREAM, size_t RX, typename TelemetryT>
FORCE_INLINE bool send_packet(
    uint32_t ch,
    uint32_t& slot,
    uint32_t first_slot,
    uint32_t last_slot,
    FarRing<RX>& far,
    uint32_t completion_counter,
    uint32_t& sent,
    uint32_t completed,
    int32_t& completions,
    bool& attempted,
    TelemetryT& telemetry) {
    const uint32_t src_addr = slot;
    auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src_addr);

    // The TX-queue status is read as eth_txq_is_busy reads it (a CMD read first, BH-55), but inside the group.
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t raw_payload_size = pkt_header->get_payload_size_excluding_header();
    (void)eth_txq_reg_read(sender_txq_id, ETH_TXQ_CMD);
    const uint32_t pre_txq_status = eth_txq_reg_read(sender_txq_id, ETH_TXQ_STATUS);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if constexpr (!skip_src_ch_id_update) {
        pkt_header->src_ch_id = static_cast<uint8_t>(ch);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if (static_cast<int32_t>(pre_txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
        attempted = true;
        return false;
    }
    const uint32_t payload_size_bytes = raw_payload_size + sizeof(PACKET_HEADER_TYPE);
    record_sender_channel_usage(ch, payload_size_bytes);

    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, far.next_slot, payload_size_bytes);
    far.next_slot = far.next_slot == FarRing<RX>::last ? FarRing<RX>::first : far.next_slot + channel_buffer_size;
    slot = src_addr == last_slot ? first_slot : src_addr + channel_buffer_size;
    if constexpr (telemetry_enabled) {
        record_packet_send(*telemetry.perf, ch, payload_size_bytes);
    }

    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)eth_txq_reg_read(sender_txq_id, ETH_TXQ_CMD);
    const uint32_t txq_status = eth_txq_reg_read(sender_txq_id, ETH_TXQ_STATUS);
    const uint32_t completions_raw = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    far.free_slots--;
    increment_local_update_ptr_val<FREE_SLOTS_STREAM>(1);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if (static_cast<int32_t>(txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
        wait_for_sender_txq_idle();
    }
    remote_update_ptr_val<FarRing<RX>::packets_received_stream, sender_txq_id>(1);
    sent++;
    if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
        update_bw_counters(pkt_header, *telemetry.local);
    }
    if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
        *telemetry.tx_progress = true;
    }
    completions = static_cast<int32_t>(completions_raw - completed);
    return true;
}

// The far ring has room for channel `ch` of CHANNELS: one slot, or two for an injection channel under bubble flow
// control.
template <size_t RX, uint32_t CHANNELS>
FORCE_INLINE bool far_has_room(const FarRing<RX>& far, uint32_t ch) {
    if constexpr ((bubble_channels & CHANNELS) == 0) {
        return far.free_slots != 0;
    } else if constexpr ((bubble_channels & CHANNELS) == CHANNELS) {
        return far.free_slots >= BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS;
    } else {
        return far.free_slots >=
               (((bubble_channels >> ch) & 1) ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS : 1);
    }
}

// Credit back to the upstream router: +n on its free-slots register.
template <typename ChannelT>
FORCE_INLINE void credit_upstream_router(const ChannelT& channel, int32_t n) {
    noc_inline_dw_write<InlineWriteDst::REG, true>(
        channel.upstream_credit_addr,
        pack_value_for_inc_on_write_stream_reg_write(n),
        0xf,
        tt::tt_fabric::worker_handshake_noc);
}

// Credit back to the worker: the absolute count of slots returned, written into the worker's L1 while it is connected
// and into the connection info for the next worker otherwise.
template <typename WorkerInterfaceT>
FORCE_INLINE void credit_worker(const WorkerInterfaceT& worker, uint32_t credits_returned, bool connected) {
    if (connected) {
        noc_inline_dw_write<InlineWriteDst::L1, true, enable_read_counter_update_noc_flush>(
            worker.cached_worker_semaphore_address,
            credits_returned,
            0xf,
            tt::tt_fabric::worker_handshake_noc,
            NOC_UNICAST_WRITE_VC,
            notify_worker_of_read_counter_update_src_address);
    } else {
        reinterpret_cast<volatile tt_l1_ptr EDMChannelWorkerLocationInfo*>(local_sender_channel_0_connection_info_addr)
            ->edm_read_counter = credits_returned;
    }
}

// The worker connection semaphore holds `expected` while nothing changes: unused_connection_value with no worker
// connected, open_connection_value with one. Any other value comes here: open on a request (open, or open then close
// already), tear down on a close request. Returns the new expected value. The worker interface's read counter is set
// to the credits returned so far, the value it holds in the generic loop.
template <typename WorkerInterfaceT>
__attribute__((noinline)) uint32_t worker_connection_event(
    WorkerInterfaceT& worker, uint32_t semaphore, uint32_t expected, uint32_t credits_returned) {
    worker.local_read_counter.counter = credits_returned;
    if (expected == tt::tt_fabric::connection_interface::unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            establish_worker_connection<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(worker);
            return tt::tt_fabric::connection_interface::open_connection_value;
        }
        return tt::tt_fabric::connection_interface::unused_connection_value;
    }
    if (semaphore == tt::tt_fabric::connection_interface::close_connection_request_value) {
        worker.template teardown_worker_connection<true, ENABLE_RISC_CPU_DATA_CACHE>();
        return tt::tt_fabric::connection_interface::unused_connection_value;
    }
    return tt::tt_fabric::connection_interface::open_connection_value;
}
static_assert(
    tt::tt_fabric::connection_interface::open_connection_value !=
    tt::tt_fabric::connection_interface::unused_connection_value);

// One visit to forwarding channel CH, which has work (the generic loop's run_forwarding_sender_step): send if it has a
// packet and the far ring has room, take completions (and first-level acks), credit the upstream router, and keep the
// channel's bit in `in_flight` equal to "packets not yet completed (or acked)". Inline and specialized for CH: slot
// bounds, register and counter addresses and src_ch_id are constants; the channel's counts and slot come from its
// record, read only here.
template <size_t CH, size_t RX, bool FIRST_LEVEL_ACK>
FORCE_INLINE void visit_forwarding_channel(bool has_unsent_packet, FarRing<RX>& far, ForwardingState& state) {
    constexpr uint32_t bit = 1u << CH;
    ForwardingChannel& channel = state.channels[CH];
    uint32_t sent = channel.sent;
    uint32_t completed = channel.completed;
    int32_t completions = 0;
    bool sent_packet = false;
    if (far_has_room<RX, bit>(far, CH) && has_unsent_packet) {
        uint32_t slot = channel.slot;
        sent_packet = send_packet<free_slots_stream<CH>, RX>(
            CH,
            slot,
            ChannelSlots<CH>::first,
            ChannelSlots<CH>::last,
            far,
            completion_counter_addr(CH),
            sent,
            completed,
            completions,
            state.attempted,
            state.telemetry);
        if (sent_packet) {
            channel.slot = slot;
            channel.sent = sent;
        }
    }
    if (!sent_packet && sent != completed) {
        completions = static_cast<int32_t>(
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(CH)) - completed);
    }
    if (completions) {
        far.free_slots += completions;
        completed += completions;
        channel.completed = completed;
        if constexpr (!FIRST_LEVEL_ACK) {
            credit_upstream_router(channel, completions);
        }
    }
    bool busy = sent != completed;
    if constexpr (FIRST_LEVEL_ACK) {
        uint32_t acked = channel.acked;
        if (sent != acked) {
            const uint32_t acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_counter_addr(CH)) - acked;
            if (acks > 0) {
                acked += acks;
                channel.acked = acked;
                credit_upstream_router(channel, acks);
            }
        }
        busy = busy || sent != acked;
    }
    state.in_flight = busy ? (state.in_flight | bit) : (state.in_flight & ~bit);
}

// The forwarding channels of one VC (bits of CHANNELS), whose free-slots values are raw[k] for channel BASE + k. A
// channel has work when it has an unsent packet or packets not yet completed. When no channel has work, which is every
// pass on a router whose traffic all comes from its worker, one test skips them all; otherwise each channel with work
// is visited, in channel order.
template <size_t RX, uint32_t CHANNELS, size_t BASE, bool FIRST_LEVEL_ACK>
FORCE_INLINE void visit_forwarding_channels(const uint32_t (&raw)[4], FarRing<RX>& far, ForwardingState& state) {
    static_assert(
        (CHANNELS >> BASE) < 16 && (CHANNELS & ((1u << BASE) - 1)) == 0,
        "a VC's forwarding channels must be in its group of four reads");
    static_assert(
        (bubble_channels & CHANNELS) == 0 || FIRST_LEVEL_ACK,
        "Bubble flow control and first level ack must be set to the same values");

    if (all_channels_free<CHANNELS, BASE>(raw) && (state.in_flight & CHANNELS) == 0) {
        return;
    }
    for_each_channel<CHANNELS, BASE>([&](auto ch, auto k) __attribute__((always_inline)) {
        constexpr size_t CH = decltype(ch)::value;
        const bool has_unsent_packet = !all_slots_free<CH>(raw[decltype(k)::value]);
        if (has_unsent_packet || (state.in_flight & (1u << CH)) != 0) {
            visit_forwarding_channel<CH, RX, FIRST_LEVEL_ACK>(has_unsent_packet, far, state);
        }
    });
}

// The VC1 channels' free-slots registers, read back to back after VC0's work (four at most, so the reads overlap).
template <uint32_t CHANNELS, size_t BASE>
FORCE_INLINE void read_free_slots_group(uint32_t (&raw)[4]) {
    for_each_channel<CHANNELS, BASE>([&](auto ch, auto k) __attribute__((always_inline)) {
        raw[decltype(k)::value] = read_free_slots_raw<decltype(ch)::value>();
    });
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Wait for each router-to-router channel's upstream router to connect, then cache its credit address (the generic
// path's wait_for_static_connection_to_ready, with one poll loop shared by all channels).
template <bool INSTANTIATED_ON_USE = true>
__attribute__((noinline)) void wait_for_connect_request(
    volatile tt_l1_ptr uint32_t* connection_live_semaphore,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    uint32_t count = 0;
    while (!connect_is_requested(*connection_live_semaphore) &&
           !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        if (count++ == CONTEXT_SWITCH_TIMEOUT_SHORT) {
            count = 0;
            run_routing();
        }
    }
}

template <typename EdmChannelWorkerIFs>
FORCE_INLINE void wait_for_static_connections(
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    tuple_for_each_constexpr(
        local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
            if constexpr (is_sender_channel_serviced[idx] && sender_ch_wait_static_connection[idx]) {
                wait_for_connect_request(interface.connection_live_semaphore, termination_signal_ptr);
                interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
            }
        });
}

// The main loop's once-per-round and rare work, out of line (see run_main_loop).
// The termination test: the termination signal, or the launch mailbox's exit flag.
__attribute__((noinline)) bool termination_requested() {
    return got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr));
}

// At the top of a round: run it unless termination is requested or, on the context-switch master, a router command
// other than RUN is pending.
__attribute__((noinline)) bool next_round(tt_l1_ptr RouterStateManager* state_manager_l1) {
    return continue_running_main_run_loop<ENABLE_RISC_CPU_DATA_CACHE>(
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr), state_manager_l1);
}

// At the end of a round: the heartbeat, then the context switch. As the context-switch master this ERISC hands the core
// to the base firmware once more than SWITCH_INTERVAL rounds in a row passed without a send attempt
// (IDLE_CONTEXT_SWITCHING; otherwise every SWITCH_INTERVAL rounds); otherwise it follows the master's handshake.
__attribute__((noinline)) void end_round(
    uint16_t& fabric_heartbeat_counter, size_t& did_nothing_count, bool attempted_send) {
    auto termination_signal_ptr = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    if ((++fabric_heartbeat_counter & 0x3F) == 0) {
        *reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR) = 0xDCBA0000 | fabric_heartbeat_counter;
    }
    if constexpr (enable_context_switch) {
        // shouldn't do noc counter sync since we are not incrementing them
        if constexpr (IDLE_CONTEXT_SWITCHING) {
            if (attempted_send) {
                did_nothing_count = 0;
            } else if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
            }
        } else {
            if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
            }
        }
    } else {
        run_routing_without_noc_sync_coordinated_as_non_master(termination_signal_ptr);
    }
}

// A router command other than RUN, as read by the caller: PAUSE runs the pause handshake as the context-switch master.
__attribute__((noinline)) void run_router_command(tt_l1_ptr RouterStateManager* state_manager_l1, RouterCommand command) {
    auto termination_signal_ptr = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    switch (command) {
        case RouterCommand::PAUSE:
            coordinated_context_switch_start_as_master(termination_signal_ptr);
            execute_pause_command(state_manager_l1, termination_signal_ptr);
            coordinated_context_switch_finish_as_master(termination_signal_ptr);
            break;
        default: ASSERT(false); break;
    }
}

// The sending ERISC's main loop: router commands, rounds (each followed by end_round) and, in each round,
// iterations_between_ctx_switch_and_teardown_checks passes over the sender channels.
//
// A function of its own, which owns the channel state: its code depends only on this body, not on kernel_main's init
// and teardown. The work it does once per round or rarely is in the small functions above, so that this function holds
// only per-pass code; the worker connection check stays in every pass, with only its rare open and close out of line
// (worker_connection_event).
template <typename EdmChannelWorkerIFs>
__attribute__((noinline)) void run_main_loop(
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);

    FabricTelemetry local_fabric_telemetry{};
    auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetry*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);
    PerfTelemetryRecorder perf_telemetry_recorder = build_perf_telemetry_recorder<perf_telemetry_mode>();
    auto local_perf_telemetry_buffer =
        build_perf_telemetry_buffer(reinterpret_cast<uint32_t*>(perf_telemetry_buffer_addr));
    bool tx_progress = false;

    FarRing<VC0_RECEIVER_CHANNEL> far_vc0;
    [[maybe_unused]] FarRing<vc1_far_ring> far_vc1;

    // The worker channel. The interface copy serves connection events and holds the worker's credit address.
    constexpr bool worker_first_level_ack = ENABLE_FIRST_LEVEL_ACK_VC0;
    auto worker = local_sender_channel_worker_interfaces.template get<0>();
    volatile tt_l1_ptr uint32_t* const worker_semaphore = worker.connection_live_semaphore;
    uint32_t worker_slot = ChannelSlots<0>::first;
    uint32_t worker_sent = 0;
    uint32_t worker_completed = 0;
    uint32_t worker_acked = 0;
    uint32_t worker_expected = tt::tt_fabric::connection_interface::unused_connection_value;

    // The forwarding channels.
    ForwardingState state{};
    state.telemetry = make_telemetry(perf_telemetry_recorder, local_fabric_telemetry, tx_progress);
    tuple_for_each_constexpr(
        local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
            constexpr size_t CH = decltype(idx)::value;
            if constexpr ((((vc0_forwarding | vc1_forwarding) >> CH) & 1) != 0) {
                ForwardingChannel& channel = state.channels[CH];
                channel.slot = ChannelSlots<CH>::first;
                channel.upstream_credit_addr = interface.cached_worker_semaphore_address;
            }
        });

    size_t did_nothing_count = 0;
    uint16_t fabric_heartbeat_counter = 0;

    // Packets sent on every channel so far. A round attempted a send if this changed or an attempt yielded.
    auto packets_sent = [&]() __attribute__((always_inline)) {
        uint32_t total = worker_sent;
        for_each_index(std::make_index_sequence<NUM_SENDER_CHANNELS>{}, [&](auto c) __attribute__((always_inline)) {
            constexpr size_t CH = decltype(c)::value;
            if constexpr ((((vc0_forwarding | vc1_forwarding) >> CH) & 1) != 0) {
                total += state.channels[CH].sent;
            }
        });
        return total;
    };

    auto execute_main_loop = [&]() __attribute__((always_inline)) {
        // STABILITY_PROBE_PRELOOP
        while (next_round(state_manager_l1)) {
            state.attempted = false;
            const uint32_t round_packets_sent = packets_sent();
            tx_progress = false;
            open_perf_recording_window(perf_telemetry_recorder);
            uint64_t loop_start_cycles;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                loop_start_cycles = get_timestamp();
            }

            for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
                // STABILITY_PROBE_LOOP
                uint32_t free_slots_raw[4] = {0, 0, 0, 0};
                read_grouped_free_slots(free_slots_raw);

                if constexpr (has_worker) {
                    constexpr uint32_t num_slots = ChannelSlots<0>::num_slots;
                    const bool has_unsent_packet = (free_slots_raw[0] & stream_count_mask) != num_slots;
                    int32_t completions = 0;
                    bool sent_packet = false;
                    if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && has_unsent_packet) {
                        sent_packet = send_packet<free_slots_stream<0>, VC0_RECEIVER_CHANNEL>(
                            0,
                            worker_slot,
                            ChannelSlots<0>::first,
                            ChannelSlots<0>::last,
                            far_vc0,
                            completion_counter_addr(0),
                            worker_sent,
                            worker_completed,
                            completions,
                            state.attempted,
                            state.telemetry);
                    }
                    if (!sent_packet && worker_sent != worker_completed) {
                        completions = static_cast<int32_t>(
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(0)) -
                            worker_completed);
                    }
                    const bool connected =
                        worker_expected != tt::tt_fabric::connection_interface::unused_connection_value;
                    if (completions) {
                        far_vc0.free_slots += completions;
                        worker_completed += completions;
                        if constexpr (!worker_first_level_ack) {
                            credit_worker(worker, worker_completed, connected);
                        }
                    }
                    if constexpr (worker_first_level_ack) {
                        if (worker_sent != worker_acked) {
                            const uint32_t acks =
                                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_counter_addr(0)) - worker_acked;
                            if (acks > 0) {
                                worker_acked += acks;
                                credit_worker(worker, worker_acked, connected);
                            }
                        }
                    }
                    // Worker connection open and teardown, polled on every pass.
                    const uint32_t semaphore = *worker_semaphore;
                    if (semaphore != worker_expected) {
                        worker_expected = worker_connection_event(
                            worker,
                            semaphore,
                            worker_expected,
                            worker_first_level_ack ? worker_acked : worker_completed);
                    }
                }

                if constexpr (vc0_forwarding != 0) {
                    visit_forwarding_channels<
                        VC0_RECEIVER_CHANNEL,
                        vc0_forwarding,
                        0,
                        ENABLE_FIRST_LEVEL_ACK_VC0>(free_slots_raw, far_vc0, state);
                }
                if constexpr (vc1_forwarding != 0) {
                    uint32_t vc1_free_slots_raw[4] = {0, 0, 0, 0};
                    read_free_slots_group<vc1_forwarding, vc1_begin>(vc1_free_slots_raw);
                    visit_forwarding_channels<
                        vc1_far_ring,
                        vc1_forwarding,
                        vc1_begin,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(vc1_free_slots_raw, far_vc1, state);
                }
            }

            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                uint64_t loop_end_cycles = get_timestamp();
                uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
                update_bw_cycles(loop_delta_cycles, tx_progress, false, local_fabric_telemetry);
            }
            if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
                update_telemetry(
                    local_sender_channel_free_slots_stream_ids,
                    tx_progress,
                    false,
                    local_fabric_telemetry,
                    fabric_telemetry);
            }

            // A round attempted a send if a send attempt yielded or any channel's sent count changed (read only by
            // the idle context switch).
            end_round(
                fabric_heartbeat_counter,
                did_nothing_count,
                !IDLE_CONTEXT_SWITCHING || state.attempted || packets_sent() != round_packets_sent);

            close_perf_recording_window(perf_telemetry_recorder);
            if constexpr (perf_telemetry_mode != PerfTelemetryRecorderType::NONE) {
                if (captured_an_event(perf_telemetry_recorder) ||
                    any_sender_channels_active(local_sender_channel_free_slots_stream_ids)) {
                    write_perf_recording_window_results(perf_telemetry_recorder, local_perf_telemetry_buffer);
                }
            }
        }
    };

    while (!termination_requested()) {
        const RouterCommand command = reinterpret_cast<volatile tt_l1_ptr RouterStateManager*>(state_manager_l1)->command;
        if (command == RouterCommand::RUN) {
            state_manager_l1->state = RouterState::RUNNING;
            execute_main_loop();
        } else {
            run_router_command(state_manager_l1, command);
        }
    }
}

}  // namespace sender_erisc
