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
    // Reserve first: ring_write_word stores without checking room, so every caller must. Blocking is right
    // here -- a one-shot boot kernel waits for the idle-eth pusher to drain, as the zone macros do.
    kernel_profiler::ring_ensure_room(3);
    kernel_profiler::ring_write_sticky_timer(whi);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w0(kernel_profiler::ppfmt::CLOCK_LINK_REFCLK, rlo));
    kernel_profiler::ring_write_word(wlo);
    kernel_profiler::publish_tail();
}
#else
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
        DeviceZoneScopedN("SYNC-ZONE-SENDER");
        link_clock_stamp();  // round start
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
        link_clock_stamp();  // round end: the host takes the midpoint of start and end
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
    {
        uint32_t i = 0;
        for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
            while (eth_txq_is_busy()) {
                // Start on an empty q (don't let separate loop iterations interfere with each other)
            }

            run_loop_iteration<true>(channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words);
        }
    }
}
