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
static eth_ptp::SenderLink<true> g_link;
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
#endif
    }

#if defined(LINK_HW)
    g_link.open();
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
    g_link.start(g_slot_base, pace_ticks);
    volatile tt_l1_ptr uint32_t* stopw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    while (*stopw == 0) {
        g_link.step();
        invalidate_l1_cache();
    }
    g_link.stop(stop_addr);
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
