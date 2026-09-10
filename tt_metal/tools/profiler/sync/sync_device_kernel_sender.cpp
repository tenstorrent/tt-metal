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
// Link half of the d2d sync: one PP_CLOCK(CLOCK_LINK_REFCLK) sample per stamp -- this core's refclk low 24 bits
// against its wall clock, exactly the local tracker's record with the link kind. The host pairs the two ends of
// a round by index and fits refclk against refclk, so DVFS on either chip's wall clock cannot enter the link
// solve. Streaming backend only: the DRAM profiler's build of this kernel is untouched.
// Room for n words in this core's SPSC ring, non-blocking. The burst runs during bring-up before the host
// receiver drains, so the pusher cannot free the ring mid-burst and a blocking reserve would deadlock the
// pair; room only shrinks, so each side keeps a contiguous prefix of rounds that the host pairs by index.
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
FORCE_INLINE void link_clock_stamp() {
    const uint32_t wlo = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);  // reading L latches H: L first
    const uint32_t whi = *reinterpret_cast<volatile uint32_t*>(0xFFB121F8);
    volatile uint32_t* rlop = reinterpret_cast<volatile uint32_t*>(0xFFB98850);
    volatile uint32_t* rhip = reinterpret_cast<volatile uint32_t*>(0xFFB98854);
    const uint32_t h1 = *rhip;
    uint32_t rlo = *rlop;
    if (*rhip != h1) {  // no latch on the refclk pair: guard a 2^32 splice
        rlo = *rlop;
    }
    kernel_profiler::ring_write_sticky_timer(whi);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w0(kernel_profiler::ppfmt::CLOCK_LINK_REFCLK, rlo));
    kernel_profiler::ring_write_word(wlo);
    kernel_profiler::publish_tail();
}
#else
FORCE_INLINE uint64_t link_refclk64() { return 0; }
FORCE_INLINE bool link_clock_room(uint32_t) { return false; }
FORCE_INLINE void link_clock_stamp() {}
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
#if !defined(PROFILE_STREAMING)
        DeviceZoneScopedN("SYNC-ZONE-SENDER");  // legacy DRAM-profiler fit reads this; the streaming link
                                                // half is the PP_CLOCK(LINK) stamps below, not the zone
#endif
        const bool emit_round = link_clock_room(6);  // t0 + t2, or neither: a whole round is all-or-none
        if (emit_round) {
            link_clock_stamp();  // round start (t0)
        }
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
        if (emit_round) {
            link_clock_stamp();  // round end (t2); the host takes the midpoint of t0 and t2
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
