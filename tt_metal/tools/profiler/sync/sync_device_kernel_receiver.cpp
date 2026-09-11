// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "api/debug/assert.h"

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
#else
FORCE_INLINE bool link_clock_room(uint32_t) { return false; }
struct LinkInstant {};
FORCE_INLINE LinkInstant link_clock_read() { return {}; }
FORCE_INLINE void link_clock_record(const LinkInstant&, uint32_t, uint32_t) {}
FORCE_INLINE void link_clock_stamp(uint32_t, uint32_t) {}
static constexpr uint32_t kRoleT0 = 0, kRoleT1 = 0, kRoleT1B = 0, kRoleT2 = 0;
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
#if defined(PROFILE_STREAMING)
// The streaming backend always runs resident; the stop word address arrives as a runtime arg (positional
// compile args past index 2 do not reach this kernel). Set in kernel_main, read in the message waits.
static uint32_t g_stop_addr = 0;
#endif

template <bool MEASURE>
FORCE_INLINE bool run_loop_iteration(
    std::array<uint32_t, NUM_CHANNELS> const& channel_addrs,
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> const& channel_sync_addrs) {
    if constexpr (MEASURE) {
#if defined(LINK_HW)
        // Hardware-stamped round: kTripsPerRound messages, each echoed; the ingress stamps average into t1 and the
        // echo-egress stamps into t1b, the receiver's own two ends of the round. The software t1 and t1b of the
        // first trip go out as their own stream. The round's number is the sender's, read from the frame.
        static_assert(NUM_CHANNELS == 1);
        static int64_t sum_t1 = 0, sum_t1b = 0;
        static uint32_t trip = 0, round = 0;
        static bool round_ok = true, round_emit = false;
        static LinkInstant sw_t1{}, sw_t1b{};
        volatile tt_l1_ptr uint32_t* hw_stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr);
        tt::tt_metal::eth_ptp::LinkHwRx rx;
        while (channel_sync_addrs[0]->bytes_sent == 0 && *hw_stopw == 0) {
            invalidate_l1_cache();
        }
        if (*hw_stopw != 0) {
            return false;
        }
        if (trip == 0) {
            sw_t1 = link_clock_read();
            round = channel_sync_addrs[0]->reserved_2;
            round_emit = link_clock_room(20);
            round_ok = true;
            sum_t1 = 0;
            sum_t1b = 0;
        }
        const uint64_t t1h = tt::tt_metal::eth_ptp::link_hw_rx_take(rx);
        channel_sync_addrs[0]->bytes_sent = 0;
        channel_sync_addrs[0]->receiver_ack = 0;
        const uint32_t sync_addr = reinterpret_cast<uint32_t>(channel_sync_addrs[0]);
        const uint64_t t1bh = tt::tt_metal::eth_ptp::link_hw_send(
            0x5200'0000'0000'0000ull | g_hw.round, sync_addr, sync_addr, sizeof(eth_channel_sync_t) >> 4, [&] {
                if (trip == 0) {
                    sw_t1b = link_clock_read();
                }
            });
        if (trip == 0 && round_emit) {
            link_clock_record(sw_t1, round, kRoleT1);
            link_clock_record(sw_t1b, round, kRoleT1B);
        }
        round_ok = round_ok && t1h != 0 && t1bh != 0;
        sum_t1 += static_cast<int64_t>(t1h);
        sum_t1b += static_cast<int64_t>(t1bh);
        g_hw.round++;
        if (++trip == tt::tt_metal::eth_ptp::kTripsPerRound) {
            trip = 0;
            if (round_emit && round_ok) {
                link_hw_record(
                    round,
                    kRoleT1,
                    tt::tt_metal::eth_ptp::link_hw_q(g_hw, sum_t1, tt::tt_metal::eth_ptp::kTripsPerRound));
                link_hw_record(
                    round,
                    kRoleT1B,
                    tt::tt_metal::eth_ptp::link_hw_q(g_hw, sum_t1b, tt::tt_metal::eth_ptp::kTripsPerRound));
            }
        }
        return true;
#endif
#if defined(PROFILE_STREAMING)
        // Resident receiver: break the message wait on the host's stop word (the sender stopped first, so no further
        // message is coming) and exit without echoing. Non-streaming builds compile this out entirely.
        volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr);
        while (channel_sync_addrs[0]->bytes_sent == 0 && *stopw == 0) {
            invalidate_l1_cache();
        }
        if (*stopw != 0) {
            return false;
        }
#else
        while (channel_sync_addrs[0]->bytes_sent == 0) {
            invalidate_l1_cache();
        }
#endif

        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
#if defined(PROFILE_STREAMING)
            while (channel_sync_addrs[i]->bytes_sent == 0 && *stopw == 0) {
                invalidate_l1_cache();
            }
            if (*stopw != 0) {
                return false;
            }
#else
            while (channel_sync_addrs[i]->bytes_sent == 0) {
                invalidate_l1_cache();
            }
#endif
#if !defined(PROFILE_STREAMING)
            DeviceZoneScopedN("SYNC-ZONE-RECEIVER");
#endif
            // t1 at the frame's detection, t1b right before the echo is issued: the receiver's midpoint mirrors the
            // sender's, so the turnaround cancels out of the software round. Both records are written after the echo.
            const LinkInstant t1 = link_clock_read();
            const uint32_t round = channel_sync_addrs[i]->reserved_2;
            const bool emit_round = link_clock_room(10);

            channel_sync_addrs[i]->bytes_sent = 0;
            channel_sync_addrs[i]->receiver_ack = 0;

            const LinkInstant t1b = link_clock_read();
            // wait for txq to be ready, otherwise we'll
            // hit a context switch in the send command
            eth_send_bytes_over_channel_payload_only(
                reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t) >> 4);
            if (emit_round) {
                link_clock_record(t1, round, kRoleT1);
                link_clock_record(t1b, round, kRoleT1B);
            }
        }
    } else {
        while (channel_sync_addrs[0]->bytes_sent == 0) {
            invalidate_l1_cache();
        }

        {
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                while (channel_sync_addrs[i]->bytes_sent == 0) {
                    invalidate_l1_cache();
                }

                channel_sync_addrs[i]->bytes_sent = 0;
                channel_sync_addrs[i]->receiver_ack = 0;

                eth_send_bytes_over_channel_payload_only(
                    reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                    reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                    sizeof(eth_channel_sync_t),
                    sizeof(eth_channel_sync_t),
                    sizeof(eth_channel_sync_t) >> 4);
            }
        }
    }
    return true;
}

static constexpr uint32_t MAX_CHANNELS = 8;
void kernel_main() {
    std::array<uint32_t, NUM_CHANNELS> channel_addrs;
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> channel_sync_addrs;
    {
        uint32_t channel_addr = HANDSHAKE_ADDR + sizeof(eth_channel_sync_t);
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
            channel_addrs[i] = channel_addr;
            channel_addr += MESSAGE_SIZE;
            channel_sync_addrs[i] = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
            channel_sync_addrs[i]->bytes_sent = 0;
            channel_sync_addrs[i]->receiver_ack = 0;
            channel_addr += sizeof(eth_channel_sync_t);
        }
    }

#if defined(LINK_HW)
    tt::tt_metal::eth_ptp::link_hw_begin(g_hw);
#endif
    eth_setup_handshake(HANDSHAKE_ADDR, false);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs);
#if defined(PROFILE_STREAMING)
    g_stop_addr = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr);
    while (*stopw == 0) {
        if (!run_loop_iteration<true>(channel_addrs, channel_sync_addrs)) {
            break;  // stopped mid-wait
        }
    }
#if defined(LINK_HW)
    tt::tt_metal::eth_ptp::link_hw_end(g_hw);
#endif
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr + 4) = 1;  // done, host polls this
#else
    for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs);
    }
#endif
}
