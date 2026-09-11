// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "debug/debug.h"

#if defined(PROFILE_KERNEL) && defined(PROFILE_STREAMING)
#include "tools/profiler/kernel_profiler.hpp"
static constexpr uint32_t kRoleT0 = kernel_profiler::ppfmt::CLOCK_ROLE_T0,
                          kRoleT1 = kernel_profiler::ppfmt::CLOCK_ROLE_T1,
                          kRoleT1B = kernel_profiler::ppfmt::CLOCK_ROLE_T1B,
                          kRoleT2 = kernel_profiler::ppfmt::CLOCK_ROLE_T2;
// Link half of the d2d sync: one PP_CLOCK(CLOCK_LINK_REFCLK) sample per stamp -- this core's refclk against its
// wall clock, exactly the local tracker's record with the link kind, plus the round's number and the stamp's place
// in it. The sender numbers the rounds and carries the number to the receiver inside each exchange frame, so the
// host pairs the two ends by identity and fits refclk against refclk: DVFS on either chip's wall clock cannot enter
// the link solve. Streaming backend only: the DRAM profiler's build of this kernel gains nothing but the tag.
// Room for n words in this core's SPSC ring, non-blocking. The burst runs during bring-up before the host
// receiver drains, so the pusher cannot free the ring mid-burst and a blocking reserve would deadlock the
// pair; a round a side has no room for is one the host never completes, and nothing behind it shifts.
FORCE_INLINE uint64_t link_refclk64() {
    volatile uint32_t* lop = reinterpret_cast<volatile uint32_t*>(0xFFB98850);
    volatile uint32_t* hip = reinterpret_cast<volatile uint32_t*>(0xFFB98854);
    const uint32_t h1 = *hip;
    uint32_t l = *lop;
    const uint32_t h2 = *hip;
    if (h1 != h2) {
        l = *lop;
    }
    return (static_cast<uint64_t>(h2) << 32) | l;
}
FORCE_INLINE bool link_clock_room(uint32_t n) {
    invalidate_l1_cache();
    return (kernel_profiler::wIndex - kernel_profiler::profiler_control_buffer[kernel_profiler::HEAD_INDEX]) <=
           (kernel_profiler::RING_USABLE - n);
}
// A software stamp is taken in two steps so nothing but the clock reads sits at the instant: read, then record.
struct LinkInstant {
    uint32_t wlo, whi;
    uint64_t refclk;
};
FORCE_INLINE LinkInstant link_clock_read() {
    LinkInstant t;
    t.wlo = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);  // reading L latches H: L first
    t.whi = *reinterpret_cast<volatile uint32_t*>(0xFFB121F8);
    t.refclk = link_refclk64();
    return t;
}
FORCE_INLINE void link_clock_record(const LinkInstant& t, uint32_t round, uint32_t role) {
    kernel_profiler::ring_write_sticky_timer(t.whi);
    kernel_profiler::ring_write_word(
        kernel_profiler::ppfmt::clock_w0(kernel_profiler::ppfmt::CLOCK_LINK_REFCLK, t.refclk));
    kernel_profiler::ring_write_word(t.wlo);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w2(t.refclk));
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w3(round, role));
    kernel_profiler::publish_tail();
}
FORCE_INLINE void link_clock_stamp(uint32_t round, uint32_t role) { link_clock_record(link_clock_read(), round, role); }
#if defined(D2D_HW_TS)
#define LINK_HW 1
#include "tools/profiler/sync/eth_ptp_link.hpp"
static tt::tt_metal::eth_ptp::LinkHwState g_hw;
FORCE_INLINE void link_hw_record(uint32_t round, uint32_t role, uint64_t value) {
    const uint32_t wlo = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);  // reading L latches H: L first
    const uint32_t whi = *reinterpret_cast<volatile uint32_t*>(0xFFB121F8);
    kernel_profiler::ring_write_sticky_timer(whi);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w0(kernel_profiler::ppfmt::CLOCK_LINK_PTP, value));
    kernel_profiler::ring_write_word(wlo);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w2(value));
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w3(round, role));
    kernel_profiler::publish_tail();
}
#endif
static uint32_t g_round = 0;
FORCE_INLINE uint32_t link_round_begin() { return g_round++; }
#else
FORCE_INLINE uint64_t link_refclk64() { return 0; }
FORCE_INLINE bool link_clock_room(uint32_t) { return false; }
struct LinkInstant {};
FORCE_INLINE LinkInstant link_clock_read() { return {}; }
FORCE_INLINE void link_clock_record(const LinkInstant&, uint32_t, uint32_t) {}
FORCE_INLINE void link_clock_stamp(uint32_t, uint32_t) {}
static constexpr uint32_t kRoleT0 = 0, kRoleT1 = 0, kRoleT1B = 0, kRoleT2 = 0;
FORCE_INLINE uint32_t link_round_begin() { return 0; }
#endif

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
        // number.
        const uint32_t round = link_round_begin();
#if defined(LINK_HW)
        // Hardware-stamped round: kTripsPerRound exchanges back to back, the MAC's egress stamps of our frames and
        // the RX classifier's ingress stamps of the echoes averaged on this side, as the round's t0 and t2. The
        // software stamps of the first trip go out too, as their own stream.
        static_assert(NUM_CHANNELS == 1);
        channel_sync_addrs[0]->reserved_2 = round;
        const bool hw_emit_round = link_clock_room(20);
        int64_t sum_t0 = 0, sum_t2 = 0;
        bool stamps_ok = true;
        for (uint32_t trip = 0; trip < tt::tt_metal::eth_ptp::kTripsPerRound; trip++) {
            channel_sync_addrs[0]->bytes_sent = 1;
            channel_sync_addrs[0]->receiver_ack = 0;
            const uint64_t t0h = tt::tt_metal::eth_ptp::link_hw_send(
                0x5000'0000'0000'0000ull | g_hw.round, channel_addrs[0], channel_addrs[0], full_payload_size_eth_words, [&] {
                    if (hw_emit_round && trip == 0) {
                        link_clock_stamp(round, kRoleT0);
                    }
                });
            // Only our labelled frames reach the RX stamp FIFO (no-match keep-timestamp is off), so the echo's stamp
            // waits there and is popped once the echo is seen; popping inside the wait made every iteration a few
            // register reads long and put that much jitter on the software t2.
            tt::tt_metal::eth_ptp::LinkHwRx rx;
            while (channel_sync_addrs[0]->bytes_sent != 0) {
                invalidate_l1_cache();
            }
            if (hw_emit_round && trip == 0) {
                link_clock_stamp(round, kRoleT2);
            }
            const uint64_t t2h = tt::tt_metal::eth_ptp::link_hw_rx_take(rx);
            stamps_ok = stamps_ok && t0h != 0 && t2h != 0;
            sum_t0 += static_cast<int64_t>(t0h);
            sum_t2 += static_cast<int64_t>(t2h);
            g_hw.round++;
        }
        if (hw_emit_round && stamps_ok) {
            link_hw_record(
                round, kRoleT0, tt::tt_metal::eth_ptp::link_hw_q(g_hw, sum_t0, tt::tt_metal::eth_ptp::kTripsPerRound));
            link_hw_record(
                round, kRoleT2, tt::tt_metal::eth_ptp::link_hw_q(g_hw, sum_t2, tt::tt_metal::eth_ptp::kTripsPerRound));
        }
        return;
#endif
#if !defined(PROFILE_STREAMING)
        DeviceZoneScopedN("SYNC-ZONE-SENDER");  // legacy DRAM-profiler fit reads this; the streaming link
                                                // half is the PP_CLOCK(LINK) stamps below, not the zone
#endif
        const bool emit_round = link_clock_room(10);  // t0 + t2, or neither: a whole round is all-or-none
        if (emit_round) {
            link_clock_stamp(round, kRoleT0);
        }
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
        if (emit_round) {
            link_clock_stamp(round, kRoleT2);
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
    }

#if defined(LINK_HW)
    tt::tt_metal::eth_ptp::link_hw_begin(g_hw);
#endif
    eth_setup_handshake(HANDSHAKE_ADDR, true);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words);
#if defined(PROFILE_STREAMING)
    // The streaming backend always runs resident at 1 kHz. Runtime args carry the stop word and the pace interval:
    // positional compile args past index 2 do not reach this kernel (Kernel::compute_hash ignores them), so these
    // must be runtime args. Pace in DVFS-immune refclk; each round's PP_CLOCK(LINK) t0/t2 stamps drain through the
    // idle pusher. Teardown stops the sender first, so the receiver still echoes through this final round.
    const uint32_t stop_addr = get_arg_val<uint32_t>(0);
    const uint32_t pace_ticks = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    const uint64_t rc0 = link_refclk64();
    uint64_t target = rc0 + pace_ticks;
    uint32_t rounds = 0;
    while (*stopw == 0) {
        uint64_t rc = link_refclk64();
        while (rc < target && *stopw == 0) {
            invalidate_l1_cache();
            rc = link_refclk64();
        }
        target = rc + pace_ticks;
        if (*stopw != 0) {
            break;
        }
        while (eth_txq_is_busy()) {
        }
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words);
        rounds++;
    }
#if defined(LINK_HW)
    tt::tt_metal::eth_ptp::link_hw_end(g_hw);
#endif
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr + 8) = rounds;  // round count (diagnostic)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr + 12) =
        static_cast<uint32_t>((link_refclk64() - rc0) / 50000);          // sync duration in ms (diagnostic)
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
