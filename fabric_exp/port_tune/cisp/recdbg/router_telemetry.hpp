// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Telemetry: the host's two options, both off by default and then compiled away.
//   perf recorder  ("bandwidth telemetry" in test_tt_fabric; PERF_TELEMETRY_MODE): the sender side adds each round's
//                  busy cycles, words (16 B) and packets to an L1 buffer, for rounds that sent a packet or ended with
//                  one waiting in a sender channel
//   dynamic stats  (FABRIC_TELEMETRY_*; TT_METAL_FABRIC_TELEMETRY): bandwidth (packets, 4 B words, elapsed and active
//                  cycles per side), the router state and the TX / RX heartbeats, in the FabricTelemetry block
// Packets and words are counted in the send and receive paths; the rest happens once per round, out of line. The counts
// live in one global rather than in the loops' locals, so an option adds memory updates per packet without changing a
// loop's register state. Each call site is under `if constexpr` on its option: with both off, the loops carry no call.

#include "router_common.hpp"
#include "router_round.hpp"

namespace fabric_router::telemetry {

constexpr bool perf_recorder = perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH;
constexpr bool dynamic_stats = FABRIC_TELEMETRY_ANY_DYNAMIC_STAT;
constexpr bool any_option = perf_recorder || dynamic_stats;
constexpr bool counts_sent = any_option && any_sender_channel_serviced;
constexpr bool counts_received = dynamic_stats && any_receiver_channel_serviced;

// The running counts of this ERISC, and where the current round started. The sender's totals are folded in once per
// round: its packets from its channels' sent counts (round_end's tx_packets), its words from this round's counts.
struct Counters {
    uint64_t tx_packets;
    uint64_t tx_words;            // 4 B words (dynamic bandwidth)
    uint64_t tx_eth_words;        // 16 B words, truncated per packet (perf recorder)
    uint32_t round_tx_words;      // this round's (a round sends fewer than 2^32 words)
    uint32_t round_tx_eth_words;  // this round's
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
    uint64_t round_rx_packets;
};
inline Counters counters;  // used only with an option on; zeroed in init()

// `bytes` in 4 B words, rounded up (the dynamic bandwidth's unit).
FORCE_INLINE uint32_t words_of(uint32_t bytes) { return (bytes + 3) >> 2; }

// The perf recorder's buffer in L1 (the host reads it).
FORCE_INLINE volatile tt_l1_ptr LowResolutionBandwidthTelemetry* perf_buffer() {
    return reinterpret_cast<volatile tt_l1_ptr LowResolutionBandwidthTelemetry*>(perf_telemetry_buffer_addr);
}

// At init, after the FabricTelemetry block's own init: the counts, and the perf recorder's buffer on the sender side.
// (The out-of-line functions here are templates so that a build with both options off emits none of them.)
template <int = 0>
__attribute__((noinline)) void init_impl() {
    if constexpr (any_option) {
        counters = {};
    }
    if constexpr (perf_recorder && any_sender_channel_serviced) {
        auto* buffer = perf_buffer();
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

// A packet sent over the link, `bytes` with its header: its words, into this round's counts. (Its count comes from the
// channel's sent count, once per round: round_end.)
FORCE_INLINE void note_sent(uint32_t bytes) {
    if constexpr (counts_sent) {
        if constexpr (perf_recorder) {
            counters.round_tx_eth_words += bytes_to_eth_words_truncated(bytes);
        }
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            counters.round_tx_words += words_of(bytes);
        }
    }
}

// A packet taken off the link (forwarded, delivered or both), `bytes` with its header.
FORCE_INLINE void note_received(uint32_t bytes) {
    if constexpr (counts_received) {
        counters.rx_packets++;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            counters.rx_words += words_of(bytes);
        }
    }
}

// LOCAL (fabric_exp/port_tune/cisp/recdbg, the profiler stall study; never committed): per iteration, the perf
// recorder's counted rounds as the sender ERISC saw them, into a ring in L1 the routers of the NX rows leave unused
// (their channel buffers end at 371104; the link-sync block starts at 454880). An iteration starts at the first counted
// round after the host cleared the perf buffer. The host never reads it: hang_capture.py dumps L1 after the run.
constexpr uint32_t recdbg_ring_addr = 0x60000;  // header (128 B: [0] iterations, [1] magic), then 128 B entries
constexpr uint32_t recdbg_entries = 320;
struct RecDbgEntry {
    uint64_t busy;            // counted rounds' wall-clock sum (the perf buffer's cycles)
    uint64_t max_wall;        // the longest counted round: wall-clock delta
    uint64_t max_ts;          //   get_timestamp() delta over the same round (read right after each wall-clock read)
    uint64_t max_start;       //   its start (wall clock)
    uint64_t max_end;         //   its end (wall clock)
    uint64_t first_send;      // the start of the first counted round with tx_packets > 0
    uint64_t zero_tx_sum;     // counted rounds with tx_packets == 0: summed wall-clock duration
    uint64_t max_gap;         // the largest step between consecutive wall-clock reads (round start / end), any round
    uint32_t max_tx;          //   the longest round's tx_packets
    uint32_t max_held;        //   its any_held()
    uint32_t max_fw;          //   whether end_round ran the base firmware's link check in it
    uint32_t zero_tx_rounds;  // counted rounds with tx_packets == 0
    uint32_t counted;         // counted rounds
    uint32_t backwards;       // wall-clock reads below the previous one
    uint32_t max_free[7];     //   the longest round's sender channels' free-slots registers (raw)
    uint32_t huge_steps;      // steps between consecutive reads above 2^31 cycles
    uint32_t seq;             // this entry's iteration number (1-based)
    uint32_t magic;           // 0x5EC0DB61
};
static_assert(sizeof(RecDbgEntry) == 128);
struct RecDbgState {
    uint64_t last_wall;
    uint32_t iteration;
    RecDbgEntry e;
};
inline RecDbgState recdbg;

FORCE_INLINE void recdbg_step(uint64_t now) {
    if (now < recdbg.last_wall) {
        recdbg.e.backwards++;
    } else {
        const uint64_t d = now - recdbg.last_wall;
        if (recdbg.last_wall != 0 && d > recdbg.e.max_gap) {
            recdbg.e.max_gap = d;
        }
        if (recdbg.last_wall != 0 && d > (1ull << 31)) {
            recdbg.e.huge_steps++;
        }
    }
    recdbg.last_wall = now;
}

template <int = 0>
__attribute__((noinline)) void round_start_impl() {
    counters.round_wall_clock = eth_read_wall_clock();
    counters.round_timestamp = get_timestamp();
    if constexpr (perf_recorder && any_sender_channel_serviced) {
        recdbg_step(counters.round_wall_clock);
    }
    counters.round_tx_words = 0;
    counters.round_tx_eth_words = 0;
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
// sender channel holds a packet; rx_register_empty(): whether VC0's packets-received register reads 0; each is read
// only when needed. tx_packets: the packets the sender's channels sent this round (their sent counts).
template <bool SENDER, bool RECEIVER, typename AnyHeld, typename RxRegisterEmpty>
__attribute__((noinline)) void round_end_impl(
    AnyHeld any_held, RxRegisterEmpty rx_register_empty, uint32_t tx_packets) {
    Counters& c = counters;
    const bool tx_progress = tx_packets != 0;
    if constexpr (SENDER) {
        c.tx_packets += tx_packets;
        c.tx_words += c.round_tx_words;
        c.tx_eth_words += c.round_tx_eth_words;
    }
    const bool rx_progress = c.rx_packets != c.round_rx_packets;
    if constexpr (SENDER && perf_recorder) {
        const uint64_t now = eth_read_wall_clock();
        const uint64_t ts_now = get_timestamp();  // LOCAL (recdbg)
        recdbg_step(now);                          // LOCAL (recdbg)
        const bool fw_ran = recdbg_fw_ran;         // LOCAL (recdbg)
        recdbg_fw_ran = false;                     // LOCAL (recdbg)
        if (tx_progress || any_held()) {
            // The busy cycles accumulate in timestamp_start (idle rounds left out); timestamp_end is unused.
            auto* buffer = perf_buffer();
            // LOCAL (recdbg): a cleared perf buffer starts the next iteration's entry
            if (buffer->timestamp_start.full == 0 && recdbg.iteration < recdbg_entries) {
                recdbg.iteration++;
                const uint64_t keep_last = recdbg.last_wall;
                recdbg.e = {};
                recdbg.e.seq = recdbg.iteration;
                recdbg.e.magic = 0x5EC0DB61;
                recdbg.last_wall = keep_last;
                auto* hdr = reinterpret_cast<volatile uint32_t*>(recdbg_ring_addr);
                hdr[0] = recdbg.iteration;
                hdr[1] = 0x5EC0DB61;
            }
            buffer->timestamp_start.full = buffer->timestamp_start.full + (now - c.round_wall_clock);
            buffer->num_words_sent = buffer->num_words_sent + c.round_tx_eth_words;
            buffer->num_packets_sent = buffer->num_packets_sent + tx_packets;
            // LOCAL (recdbg): this round into the iteration's entry
            if (recdbg.iteration != 0) {
                RecDbgEntry& e = recdbg.e;
                const uint64_t dur = now - c.round_wall_clock;
                e.busy += dur;
                e.counted++;
                if (tx_packets == 0) {
                    e.zero_tx_rounds++;
                    e.zero_tx_sum += dur;
                } else if (e.first_send == 0) {
                    e.first_send = c.round_wall_clock;
                }
                if (dur > e.max_wall) {
                    e.max_wall = dur;
                    e.max_ts = ts_now - c.round_timestamp;
                    e.max_start = c.round_wall_clock;
                    e.max_end = now;
                    e.max_tx = tx_packets;
                    e.max_held = any_held() ? 1 : 0;
                    e.max_fw = fw_ran ? 1 : 0;
                    constexpr uint32_t ids[7] = {sender_channel_0_free_slots_stream_id,
                                                 sender_channel_1_free_slots_stream_id,
                                                 sender_channel_2_free_slots_stream_id,
                                                 sender_channel_3_free_slots_stream_id,
                                                 sender_channel_4_free_slots_stream_id,
                                                 sender_channel_5_free_slots_stream_id,
                                                 sender_channel_6_free_slots_stream_id};
                    for (uint32_t i = 0; i < 7; i++) {
                        e.max_free[i] = i < NUM_SENDER_CHANNELS
                                            ? NOC_STREAM_READ_REG(ids[i], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX)
                                            : 0xFFFFFFFF;
                    }
                }
                auto* dst = reinterpret_cast<volatile uint32_t*>(recdbg_ring_addr + 128 * recdbg.iteration);
                const uint32_t* srcw = reinterpret_cast<const uint32_t*>(&e);
                for (uint32_t i = 0; i < sizeof(RecDbgEntry) / 4; i++) {
                    dst[i] = srcw[i];
                }
            }
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
                    block->dynamic_info.tx_bandwidth,
                    c.tx_elapsed_cycles,
                    c.tx_active_cycles,
                    c.tx_words,
                    c.tx_packets);
            }
            if constexpr (RECEIVER) {
                c.rx_elapsed_cycles += cycles;
                c.rx_active_cycles += rx_progress ? cycles : 0;
                store_bandwidth(
                    block->dynamic_info.rx_bandwidth,
                    c.rx_elapsed_cycles,
                    c.rx_active_cycles,
                    c.rx_words,
                    c.rx_packets);
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
FORCE_INLINE void round_end(AnyHeld any_held, RxRegisterEmpty rx_register_empty, uint32_t tx_packets = 0) {
    if constexpr (any_option) {
        round_end_impl<SENDER, RECEIVER>(any_held, rx_register_empty, tx_packets);
    }
}

}  // namespace fabric_router::telemetry
