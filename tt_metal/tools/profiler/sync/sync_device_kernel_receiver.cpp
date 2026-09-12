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

#if defined(LINK_HW)
// The receiving end of the resident hardware-stamped sync (eth_ptp_link.hpp): every frame is echoed from the slot it
// arrived in, a burst's echo stamps are collected after its last echo, and a round's records go out when the next
// round's first frame arrives. The round's number is the sender's, read from the frame.
static void hw_receiver_loop(uint32_t stop_addr) {
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    const eth_ptp::Instant start = eth_ptp::read_instant();
    eth_ptp::StopDiag diag;
    eth_ptp::HwRound rnd;
    uint32_t round = 0, expect = 0;
    bool started = false, emit = false, ok = false;
    eth_ptp::Instant t1{}, t1b{};
    volatile eth_channel_sync_t* const first = eth_ptp::slot(g_slot_base, 0);
    // A round closes once the next round's first frame has been echoed, so the close's record writes do not sit
    // inside that frame's turnaround; the closed round's state is carried over as a copy.
    struct Closed {
        eth_ptp::HwRound rnd;
        uint32_t round;
        bool emit, ok;
        eth_ptp::Instant t1, t1b;
    };
    const auto close_round = [&](const Closed& c) {
        if (c.emit) {
            link::record_sw(c.t1, c.round, link::kRoleT1);
            link::record_sw(c.t1b, c.round, link::kRoleT1B);
            if (c.ok && g_hw.timer_ok && c.rnd.complete(eth_ptp::kTripsPerRound)) {
                link::record_hw(c.rnd.rx.q(g_hw), c.round, link::kRoleT1);
                link::record_hw(c.rnd.tx.q(g_hw), c.round, link::kRoleT1B);
            }
        }
        diag.note_round(c.rnd, c.ok);
    };
    bool stop = false;
    while (!stop) {
        // The next frame in order, or a round's first frame in slot 0 if the order broke.
        volatile eth_channel_sync_t* s = eth_ptp::slot(g_slot_base, expect);
        uint32_t key;
        for (;;) {
            key = s->bytes_sent;
            if (key != 0) {
                break;
            }
            if (s != first) {
                key = first->bytes_sent;
                if ((key & eth_ptp::kTripMask) == 1) {
                    s = first;
                    break;
                }
            }
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
        const uint32_t j = (key & eth_ptp::kTripMask) - 1;
        Closed closed{};
        bool close_after_echo = false;
        if (j == 0) {
            if (started) {
                closed = Closed{rnd, round, emit, ok, t1, t1b};
                close_after_echo = true;
            }
            started = true;
            round = s->reserved_2;
            rnd.begin(round);
            emit = link::room(4);
            ok = true;
            t1 = now;
        } else if (key != eth_ptp::frame_key(round, j)) {
            s->bytes_sent = 0;  // a frame of a round already given up
            continue;
        } else if (j != expect) {
            ok = false;
        }
        eth_ptp::rx_stamps_drain(g_hw, [&](uint64_t ts) { rnd.rx.add(ts); });
        const uint32_t i = j % eth_ptp::kBurstFrames;
        const uint64_t tag = 0x5200'0000'0000'0000ull | (round * eth_ptp::kBurstsPerRound + j / eth_ptp::kBurstFrames);
        if (i == 0) {
            eth_ptp::stamps_arm(g_hw, eth_ptp::kGapTag);
        }
        s->bytes_sent = 0;
        if (j == 0) {
            t1b = eth_ptp::read_instant();
        }
        const uint32_t addr = reinterpret_cast<uint32_t>(s);
        internal_::eth_send_packet(eth_ptp::kLinkTxq, addr >> 4, addr >> 4, sizeof(eth_channel_sync_t) >> 4);
        if (i == 0) {
            eth_ptp::stamps_retag(g_hw, tag);
        }
        if (close_after_echo) {
            close_round(closed);
        }
        if (i == eth_ptp::kBurstFrames - 1 && !eth_ptp::collect_burst(g_hw, static_cast<uint32_t>(tag), rnd.tx)) {
            ok = false;
        }
        expect = j + 1;
        diag.note_hold(eth_ptp::rd(eth_ptp::kWallClockLo) - now.wall_lo);
    }
    g_hw.end();
    const eth_ptp::Instant end = eth_ptp::read_instant();
    diag.timer = g_hw.timer_ok ? 1u : 2u;
    diag.span_wall = end.wall() - start.wall();
    diag.span_refclk = end.refclk - start.refclk;
    diag.write(stop_addr);
}
#endif

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
    eth_setup_handshake(HANDSHAKE_ADDR, false);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs);
#if defined(PROFILE_STREAMING)
    g_stop_addr = get_arg_val<uint32_t>(0);
#if defined(LINK_HW)
    hw_receiver_loop(g_stop_addr);
#else
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr);
    const eth_ptp::Instant start = eth_ptp::read_instant();
    eth_ptp::StopDiag diag;
    while (*stopw == 0) {
        if (!run_loop_iteration<true>(channel_addrs, channel_sync_addrs)) {
            break;  // stopped mid-wait
        }
        diag.rounds++;
    }
    const eth_ptp::Instant end = eth_ptp::read_instant();
    diag.span_wall = end.wall() - start.wall();
    diag.span_refclk = end.refclk - start.refclk;
    diag.write(g_stop_addr);
#endif
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_stop_addr + 4) = 1;  // done, host polls this
#else
    for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs);
    }
#endif
}
