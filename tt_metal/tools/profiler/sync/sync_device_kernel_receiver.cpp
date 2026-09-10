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
// Link half of the d2d sync: one PP_CLOCK(CLOCK_LINK_REFCLK) sample per stamp -- this core's refclk low 24 bits
// against its wall clock, exactly the local tracker's record with the link kind. The host pairs the two ends of
// a round by index and fits refclk against refclk, so DVFS on either chip's wall clock cannot enter the link
// solve. Streaming backend only: the DRAM profiler's build of this kernel is untouched.
// Room for n words in this core's SPSC ring, non-blocking. The burst runs during bring-up before the host
// receiver drains, so the pusher cannot free the ring mid-burst and a blocking reserve would deadlock the
// pair; room only shrinks, so each side keeps a contiguous prefix of rounds that the host pairs by index.
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
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> const& channel_sync_addrs) {
    if constexpr (MEASURE) {
        while (channel_sync_addrs[0]->bytes_sent == 0) {
            invalidate_l1_cache();
        }

        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            while (channel_sync_addrs[i]->bytes_sent == 0) {
                invalidate_l1_cache();
            }
#if !defined(PROFILE_STREAMING)
            DeviceZoneScopedN("SYNC-ZONE-RECEIVER");
#endif
            if (link_clock_room(3)) {
                link_clock_stamp();  // arrival (t1)
            }

            channel_sync_addrs[i]->bytes_sent = 0;
            channel_sync_addrs[i]->receiver_ack = 0;

            // wait for txq to be ready, otherwise we'll
            // hit a context switch in the send command
            eth_send_bytes_over_channel_payload_only(
                reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t) >> 4);
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

    eth_setup_handshake(HANDSHAKE_ADDR, false);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs);
    {
        uint32_t i = 0;
        for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
            run_loop_iteration<true>(channel_addrs, channel_sync_addrs);
        }
    }
}
