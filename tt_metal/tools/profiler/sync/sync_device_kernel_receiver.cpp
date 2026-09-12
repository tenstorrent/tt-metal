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
static uint32_t g_trip = 0;  // the MAC FIFO tag of the next stamped frame
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
static uint64_t g_busy_ticks = 0;
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
        static eth_ptp::HwRound rnd;
        static uint32_t trip = 0;
        static eth_ptp::Instant sw_t1{}, sw_t1b{};
        static uint64_t span0 = 0;
        volatile tt_l1_ptr uint32_t* hw_stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr);
        eth_ptp::RxStamps rx;
        while (channel_sync_addrs[0]->bytes_sent == 0 && *hw_stopw == 0) {
            invalidate_l1_cache();
        }
        if (*hw_stopw != 0) {
            return false;
        }
        if (trip == 0) {
            sw_t1 = eth_ptp::read_instant();
            span0 = sw_t1.refclk;
            rnd.begin(channel_sync_addrs[0]->reserved_2, link::room(4) && g_hw.timer_ok);
        }
        const uint64_t t1h = rx.take<eth_ptp::LinkSession>();
        channel_sync_addrs[0]->bytes_sent = 0;
        channel_sync_addrs[0]->receiver_ack = 0;
        const uint32_t sync_addr = reinterpret_cast<uint32_t>(channel_sync_addrs[0]);
        const uint64_t t1bh = eth_ptp::send_and_stamp(
            g_hw, 0x5200'0000'0000'0000ull | g_trip++, sync_addr, sync_addr, sizeof(eth_channel_sync_t) >> 4, [&] {
                if (trip == 0) {
                    sw_t1b = eth_ptp::read_instant();
                }
            });
        if (trip == 0 && rnd.emit) {
            link::record_sw(sw_t1, rnd.id, link::kRoleT1);
            link::record_sw(sw_t1b, rnd.id, link::kRoleT1B);
        }
        rnd.add(t1h, t1bh);
        if (++trip == eth_ptp::kTripsPerRound) {
            trip = 0;
            g_busy_ticks += eth_ptp::read_cfr() - span0;
            if (rnd.complete()) {
                link::record_hw(rnd.q_a(g_hw), rnd.id, link::kRoleT1);
                link::record_hw(rnd.q_b(g_hw), rnd.id, link::kRoleT1B);
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
            const eth_ptp::Instant t1 = eth_ptp::read_instant();
            const uint32_t round = channel_sync_addrs[i]->reserved_2;
            const bool emit_round = link::room(2);

            channel_sync_addrs[i]->bytes_sent = 0;
            channel_sync_addrs[i]->receiver_ack = 0;

            const eth_ptp::Instant t1b = eth_ptp::read_instant();
            // wait for txq to be ready, otherwise we'll
            // hit a context switch in the send command
            eth_send_bytes_over_channel_payload_only(
                reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t) >> 4);
            if (emit_round) {
                link::record_sw(t1, round, link::kRoleT1);
                link::record_sw(t1b, round, link::kRoleT1B);
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
    g_hw.begin();
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
    // The host logs whether this end's 1588 timer ran (0 no hardware path, 1 ran, 2 never acknowledged its rate).
    uint32_t timer_word = 0;
#if defined(LINK_HW)
    g_hw.end();
    timer_word = g_hw.timer_ok ? 1u : 2u;
#endif
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr + 16) = timer_word;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr + 20) = static_cast<uint32_t>(g_busy_ticks);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr + 24) = static_cast<uint32_t>(g_busy_ticks >> 32);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr + 4) = 1;  // done, host polls this
#else
    for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs);
    }
#endif
}
