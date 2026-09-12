// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "api/debug/assert.h"
#include "tools/profiler/sync/eth_ptp_link.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;
namespace link = tt::tt_metal::eth_ptp::link;

#if defined(PROFILE_KERNEL) && defined(PROFILE_STREAMING) && defined(D2D_HW_TS)
#define LINK_HW 1
static eth_ptp::LinkSession g_hw;
static uint32_t g_slot_base = 0;  // the kBurstFrames sync words behind the channel region
#endif
static uint32_t g_round = 0;  // the sender numbers the rounds; the receiver reads the number from the frame

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

static constexpr uint32_t HANDSHAKE_ADDR = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

static constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(0);
static constexpr uint32_t NUM_MESSAGES = get_compile_time_arg_val(1);
static constexpr uint32_t MESSAGE_SIZE = get_compile_time_arg_val(2);

template <bool MEASURE>
FORCE_INLINE void run_loop_iteration(
    std::array<uint32_t, NUM_CHANNELS> const& channel_addrs,
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> const& channel_sync_addrs,
    uint32_t full_payload_size,
    uint32_t full_payload_size_eth_words) {
    if constexpr (MEASURE) {
        // The round's number rides in the frame's sync word, so the receiver stamps the same round under the same
        // number. Software stamps are read as instants at their events and recorded once the round's exchanges are
        // done, so nothing but clock reads sits between a stamp and the frame it stamps.
        const uint32_t round = g_round++;
#if !defined(PROFILE_STREAMING)
        DeviceZoneScopedN("SYNC-ZONE-SENDER");  // legacy DRAM-profiler fit reads this; the streaming link
                                                // half is the PP_CLOCK(LINK) stamps below, not the zone
#endif
        const bool emit_round = link::room(2);  // t0 + t2, or neither: a whole round is all-or-none
        const eth_ptp::Instant t0 = eth_ptp::read_instant();
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            channel_sync_addrs[i]->bytes_sent = 1;
            channel_sync_addrs[i]->receiver_ack = 0;
            channel_sync_addrs[i]->reserved_2 = round;
            eth_send_bytes_over_channel_payload_only(
                channel_addrs[i], channel_addrs[i], full_payload_size, full_payload_size, full_payload_size_eth_words);
        }
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            while (channel_sync_addrs[i]->bytes_sent != 0) {
                invalidate_l1_cache();
            }
        }
        const eth_ptp::Instant t2 = eth_ptp::read_instant();
        if (emit_round) {
            link::record_sw(t0, round, link::kRoleT0);
            link::record_sw(t2, round, link::kRoleT2);
        }
    } else {
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            channel_sync_addrs[i]->bytes_sent = 1;
            channel_sync_addrs[i]->receiver_ack = 0;
            eth_send_bytes_over_channel_payload_only(
                channel_addrs[i], channel_addrs[i], full_payload_size, full_payload_size, full_payload_size_eth_words);
        }
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            while (channel_sync_addrs[i]->bytes_sent != 0) {
                invalidate_l1_cache();
            }
        }
    }
}

#if defined(LINK_HW)
// The resident hardware-stamped sync (eth_ptp_link.hpp): kBurstsPerRound bursts a round, pace_ticks / kBurstsPerRound
// apart on the refclk, each frame at its phase of the stamp tick. A burst starts by taking the previous burst's echo
// stamps; a round's records go out at the start of the next round's first burst, once its last echoes are in.
static void hw_sender_loop(uint32_t stop_addr, uint32_t pace_ticks) {
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    const uint32_t burst_ticks = pace_ticks / eth_ptp::kBurstsPerRound;
    const eth_ptp::Instant start = eth_ptp::read_instant();
    eth_ptp::Instant at_burst = start;
    // Wall cycles per refclk tick, x16, from the previous burst's interval: the frames' phases of the tick are spun
    // in wall cycles, and a grid scaled by a stale AICLK covers more or less than the tick, which biases the
    // stamps' rounding by stamp kind. 1.25 GHz until measured.
    uint32_t c16 = 400;
    eth_ptp::Pacer pacer;
    pacer.calibrate();
    eth_ptp::StopDiag diag;
    eth_ptp::HwRound rnd;
    uint32_t round = 0;
    bool emit = false, ok = false;
    eth_ptp::Instant t0{}, t2{};
    const auto issue = [](volatile eth_channel_sync_t* s) {
        const uint32_t addr = reinterpret_cast<uint32_t>(s);
        internal_::eth_send_packet(eth_ptp::kLinkTxq, addr >> 4, addr >> 4, sizeof(eth_channel_sync_t) >> 4);
    };
    const auto close_round = [&]() {
        if (emit) {
            link::record_sw(t0, round, link::kRoleT0);
            link::record_sw(t2, round, link::kRoleT2);
            if (ok && g_hw.timer_ok && rnd.complete(eth_ptp::kTripsPerRound)) {
                link::record_hw(rnd.tx.q(g_hw), round, link::kRoleT0);
                link::record_hw(rnd.rx.q(g_hw), round, link::kRoleT2);
            }
        }
        diag.note_round(rnd, ok);
    };
    uint64_t slot_cfr = start.refclk + eth_ptp::kFrameTicks;
    bool stop = false;
    for (uint64_t b = 0; !stop; b++, slot_cfr += burst_ticks) {
        while (eth_ptp::read_cfr() < slot_cfr) {
            if (*stopw != 0) {
                stop = true;
                break;
            }
            invalidate_l1_cache();
        }
        if (stop) {
            break;
        }
        const eth_ptp::Instant now = eth_ptp::read_instant();
        const uint32_t hold0 = now.wall_lo;
        if (b != 0) {
            c16 = (static_cast<uint32_t>(now.wall() - at_burst.wall()) * 16u) /
                  static_cast<uint32_t>(now.refclk - at_burst.refclk);
        }
        at_burst = now;
        eth_ptp::rx_stamps_drain(g_hw, [&](uint64_t ts) { rnd.rx.add(ts); });
        const uint32_t j0 = static_cast<uint32_t>(b % eth_ptp::kBurstsPerRound) * eth_ptp::kBurstFrames;
        if (j0 == 0) {
            if (b != 0) {
                close_round();
            }
            round = g_round++;
            rnd.begin(round);
            emit = link::room(4);
            ok = true;
        }
        eth_ptp::stamps_arm(g_hw, eth_ptp::kGapTag);
        const uint64_t tag = 0x5000'0000'0000'0000ull | b;
        // The burst's frames sit kFrameTicks apart from the first, each at its phase of the tick, all in wall cycles
        // from one reading: the slot wait's exit shifts the whole burst by the same amount, which the grid does not
        // mind.
        const uint32_t spacing = (eth_ptp::kFrameTicks * c16) >> 4;
        const uint32_t phase0 = eth_ptp::frame_phase_cycles(j0, c16);
        const uint32_t w0 = eth_ptp::rd(eth_ptp::kWallClockLo) + 32 + phase0;
        for (uint32_t i = 0; i < eth_ptp::kBurstFrames; i++) {
            const uint32_t j = j0 + i;
            volatile eth_channel_sync_t* s = eth_ptp::slot(g_slot_base, i);
            pacer.until(w0 + i * spacing + eth_ptp::frame_phase_cycles(j, c16) - phase0);
            s->reserved_2 = round;
            s->bytes_sent = eth_ptp::frame_key(round, j);
            if (j == 0) {
                t0 = eth_ptp::read_instant();
            }
            issue(s);
            if (i == 0) {
                eth_ptp::stamps_retag(g_hw, tag);
            }
            if (j == 0) {
                for (uint32_t spin = 0; s->bytes_sent != 0; spin++) {
                    if (spin == eth_ptp::kEchoSpins) {
                        ok = false;
                        break;
                    }
                    invalidate_l1_cache();
                }
                t2 = eth_ptp::read_instant();
            }
        }
        if (!eth_ptp::collect_burst(g_hw, static_cast<uint32_t>(tag), rnd.tx)) {
            ok = false;
        }
        diag.note_hold(eth_ptp::rd(eth_ptp::kWallClockLo) - hold0);
    }
    g_hw.end();
    const eth_ptp::Instant end = eth_ptp::read_instant();
    diag.timer = g_hw.timer_ok ? 1u : 2u;
    diag.span_wall = end.wall() - start.wall();
    diag.span_refclk = end.refclk - start.refclk;
    diag.write(stop_addr);
}
#endif

void kernel_main() {
    const uint32_t message_size_eth_words = MESSAGE_SIZE >> 4;

    const uint32_t full_payload_size = MESSAGE_SIZE + sizeof(eth_channel_sync_t);
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    ASSERT(NUM_CHANNELS * 2 <= 8);

    std::array<uint32_t, NUM_CHANNELS> channel_addrs;
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> channel_sync_addrs;
    {
        uint32_t channel_addr = HANDSHAKE_ADDR + sizeof(eth_channel_sync_t);
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
            channel_addrs[i] = channel_addr;
            channel_addr += MESSAGE_SIZE;
            channel_sync_addrs[i] = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
            channel_addr += sizeof(eth_channel_sync_t);
        }
#if defined(LINK_HW)
        g_slot_base = channel_addr;
        for (uint32_t j = 0; j < eth_ptp::kBurstFrames; j++) {
            volatile eth_channel_sync_t* s = eth_ptp::slot(g_slot_base, j);
            s->bytes_sent = 0;
            s->receiver_ack = 0;
            s->src_id = 0;
            s->reserved_2 = 0;
        }
#endif
    }

#if defined(LINK_HW)
    g_hw.begin();
#endif
    eth_setup_handshake(HANDSHAKE_ADDR, true);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words);
#if defined(PROFILE_STREAMING)
    // The streaming backend always runs resident. Runtime args carry the stop word and the round period in refclk
    // ticks: positional compile args past index 2 do not reach this kernel (Kernel::compute_hash ignores them), so
    // these must be runtime args. Teardown stops the sender first, so the receiver still echoes through this final
    // round.
    const uint32_t stop_addr = get_arg_val<uint32_t>(0);
    const uint32_t pace_ticks = get_arg_val<uint32_t>(1);
#if defined(LINK_HW)
    hw_sender_loop(stop_addr, pace_ticks);
#else
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    const eth_ptp::Instant start = eth_ptp::read_instant();
    uint64_t target = start.refclk + pace_ticks;
    eth_ptp::StopDiag diag;
    while (*stopw == 0) {
        uint64_t rc = eth_ptp::read_cfr();
        while (rc < target && *stopw == 0) {
            invalidate_l1_cache();
            rc = eth_ptp::read_cfr();
        }
        target = rc + pace_ticks;
        if (*stopw != 0) {
            break;
        }
        while (eth_txq_is_busy()) {
        }
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words);
        diag.rounds++;
    }
    const eth_ptp::Instant end = eth_ptp::read_instant();
    diag.span_wall = end.wall() - start.wall();
    diag.span_refclk = end.refclk - start.refclk;
    diag.write(stop_addr);
#endif
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr + 4) = 1;  // done, host polls this
#else
    for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
        while (eth_txq_is_busy()) {
            // Start on an empty q (don't let separate loop iterations interfere with each other)
        }
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words);
    }
#endif
}
