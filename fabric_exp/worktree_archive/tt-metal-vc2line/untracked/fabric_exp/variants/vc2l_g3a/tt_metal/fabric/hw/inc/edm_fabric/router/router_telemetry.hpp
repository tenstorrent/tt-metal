// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Telemetry: the host's two options (V4_TELEMETRY.md), both off by default and then compiled away.
//   perf recorder  ("bandwidth telemetry" in test_tt_fabric; PERF_TELEMETRY_MODE): the sender side adds each round's
//                  busy cycles, words (16 B) and packets to an L1 buffer, for rounds that sent a packet or ended with one
//                  waiting in a sender channel
//   dynamic stats  (FABRIC_TELEMETRY_*; TT_METAL_FABRIC_TELEMETRY): bandwidth (packets, 4 B words, elapsed and active
//                  cycles per side), the router state and the TX / RX heartbeats, in the FabricTelemetry block
// Packets and words are counted in the send and receive paths; the rest happens once per round, out of line. The counts
// live in one global rather than in the loops' locals, so an option adds memory updates per packet without changing a
// loop's register state. A round is one run of passes between end_round calls (router_round.hpp), as main's.
// The call sites guard each call with `if constexpr` on the option, so a build with both options off carries no call,
// argument or lambda: dead inline calls alone shifted register allocation in the spill-heavy loops (VC1 senders,
// VC2 and speedy receivers; offline check 2026-10-29).

#include "router_common.hpp"
#include "router_round.hpp"

namespace fabric_router::telemetry {

constexpr bool perf_recorder = perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH;
constexpr bool dynamic_stats = FABRIC_TELEMETRY_ANY_DYNAMIC_STAT;
constexpr bool any_option = perf_recorder || dynamic_stats;
constexpr bool counts_sent = any_option && any_sender_channel_serviced;
constexpr bool counts_received = dynamic_stats && any_receiver_channel_serviced;

// The running counts of this ERISC, and where the current round started.
struct Counters {
    uint64_t tx_packets;
    uint64_t tx_words;      // 4 B words (dynamic bandwidth)
    uint64_t tx_eth_words;  // 16 B words, truncated per packet (perf recorder)
    uint64_t rx_packets;
    uint64_t rx_words;
    uint64_t tx_elapsed_cycles;
    uint64_t tx_active_cycles;
    uint64_t rx_elapsed_cycles;
    uint64_t rx_active_cycles;
    uint64_t tx_heartbeat;
    uint64_t rx_heartbeat;
    uint64_t round_wall_clock;  // the round's start: the perf recorder's clock
    uint64_t round_timestamp;   // and the dynamic bandwidth's
    uint64_t round_tx_packets;
    uint64_t round_tx_eth_words;
    uint64_t round_rx_packets;
};
inline Counters counters;  // used only with an option on; zeroed in init()

FORCE_INLINE volatile tt_l1_ptr FabricTelemetry* telemetry_block() {
    return reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
}

// At init, after the FabricTelemetry block's own init: the counts, and the perf recorder's buffer on the sender side.
// (The out-of-line functions here are templates so that a build with both options off emits none of them.)
template <int = 0>
__attribute__((noinline)) void init_impl() {
    if constexpr (any_option) {
        counters = {};
    }
    if constexpr (perf_recorder && any_sender_channel_serviced) {
        auto* buffer = reinterpret_cast<volatile tt_l1_ptr LowResolutionBandwidthTelemetry*>(perf_telemetry_buffer_addr);
        buffer->timestamp_start.full = 0;
        buffer->timestamp_end.full = 0;
        buffer->num_words_sent = 0;
        buffer->num_packets_sent = 0;
    }
}

FORCE_INLINE void init() {
    if constexpr (any_option) {
        init_impl();
    }
}

// A packet sent over the link, `bytes` with its header.
FORCE_INLINE void count_sent(uint32_t bytes) {
    if constexpr (counts_sent) {
        counters.tx_packets++;
        if constexpr (perf_recorder) {
            counters.tx_eth_words += bytes_to_eth_words_truncated(bytes);
        }
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            counters.tx_words += (bytes + 3) >> 2;
        }
    }
}

// A packet taken off the link (forwarded, delivered or both), `bytes` with its header.
FORCE_INLINE void count_received(uint32_t bytes) {
    if constexpr (counts_received) {
        counters.rx_packets++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            counters.rx_words += (bytes + 3) >> 2;
        }
    }
}

template <int = 0>
__attribute__((noinline)) void round_start_impl() {
    counters.round_wall_clock = eth_read_wall_clock();
    counters.round_timestamp = get_timestamp();
    counters.round_tx_packets = counters.tx_packets;
    counters.round_tx_eth_words = counters.tx_eth_words;
    counters.round_rx_packets = counters.rx_packets;
}

// Before a round's passes.
FORCE_INLINE void round_start() {
    if constexpr (any_option) {
        round_start_impl();
    }
}

FORCE_INLINE void store_bandwidth(
    volatile BandwidthTelemetry& to, uint64_t elapsed, uint64_t active, uint64_t words, uint64_t packets) {
    to.elapsed_active_cycles.full = active;
    to.elapsed_cycles.full = elapsed;
    to.num_words_sent = words;
    to.num_packets_sent = packets;
}

// After a round (its passes and end_round), for the sides this loop runs (SENDER, RECEIVER). any_held(): whether a
// sender channel holds a packet; rx_register_empty(): whether VC0's packets-received register reads 0; each is read only
// when needed.
template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
__attribute__((noinline)) void round_end_impl(AnyHeld any_held, RxRegisterEmpty rx_register_empty) {
    Counters& c = counters;
    const bool tx_progress = c.tx_packets != c.round_tx_packets;
    const bool rx_progress = c.rx_packets != c.round_rx_packets;
    if constexpr (SENDER && perf_recorder) {
        const uint64_t now = eth_read_wall_clock();
        if (tx_progress || any_held()) {
            // As main: the busy cycles accumulate in timestamp_start (idle rounds left out); timestamp_end is unused.
            auto* buffer =
                reinterpret_cast<volatile tt_l1_ptr LowResolutionBandwidthTelemetry*>(perf_telemetry_buffer_addr);
            buffer->timestamp_start.full = buffer->timestamp_start.full + (now - c.round_wall_clock);
            buffer->num_words_sent = buffer->num_words_sent + (c.tx_eth_words - c.round_tx_eth_words);
            buffer->num_packets_sent = buffer->num_packets_sent + (c.tx_packets - c.round_tx_packets);
        }
    }
    if constexpr (dynamic_stats) {
        auto* block = telemetry_block();
        auto& erisc = block->dynamic_info.erisc[MY_ERISC_ID];
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            const uint64_t cycles = get_timestamp() - c.round_timestamp;
            if constexpr (SENDER) {
                c.tx_elapsed_cycles += cycles;
                c.tx_active_cycles += tx_progress ? cycles : 0;
                store_bandwidth(
                    block->dynamic_info.tx_bandwidth, c.tx_elapsed_cycles, c.tx_active_cycles, c.tx_words, c.tx_packets);
            }
            if constexpr (RECEIVER) {
                c.rx_elapsed_cycles += cycles;
                c.rx_active_cycles += rx_progress ? cycles : 0;
                store_bandwidth(
                    block->dynamic_info.rx_bandwidth, c.rx_elapsed_cycles, c.rx_active_cycles, c.rx_words, c.rx_packets);
            }
        }
        if constexpr (FABRIC_TELEMETRY_ROUTER_STATE) {
            erisc.router_state = router_state()->state;
        }
        // The heartbeats move while their side makes progress or has nothing to do, and stop when it is stuck.
        if constexpr (SENDER && FABRIC_TELEMETRY_HEARTBEAT_TX) {
            if (tx_progress || !any_held()) {
                erisc.tx_heartbeat.full = ++c.tx_heartbeat;
            }
        }
        if constexpr (RECEIVER && FABRIC_TELEMETRY_HEARTBEAT_RX) {
            if (rx_progress || rx_register_empty()) {
                erisc.rx_heartbeat.full = ++c.rx_heartbeat;
            }
        }
    }
}

// After a round's passes and its end_round.
template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
FORCE_INLINE void round_end(AnyHeld any_held, RxRegisterEmpty rx_register_empty) {
    if constexpr (any_option) {
        round_end_impl<SENDER, RECEIVER>(any_held, rx_register_empty);
    }
}

}  // namespace fabric_router::telemetry
