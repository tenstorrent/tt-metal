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

#if defined(PROFILE_STREAMING)
static eth_ptp::ReceiverLink<> g_link;
static uint32_t g_slot_base = 0;  // the link's L1 (pilot, slots) behind the channel region
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
    const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
    const std::array<volatile eth_channel_sync_t*, NUM_CHANNELS>& channel_sync_addrs) {
    if constexpr (MEASURE) {
        while (channel_sync_addrs[0]->bytes_sent == 0) {
            invalidate_l1_cache();
        }

        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            while (channel_sync_addrs[i]->bytes_sent == 0) {
                invalidate_l1_cache();
            }
            DeviceZoneScopedN("SYNC-ZONE-RECEIVER");

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
#if defined(PROFILE_STREAMING)
        g_slot_base = channel_addr;
#endif
    }

#if defined(PROFILE_STREAMING)
    g_link.open();
#endif
    eth_setup_handshake(HANDSHAKE_ADDR, false);

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs);
#if defined(PROFILE_STREAMING)
    // Resident for the session; the stop word is a runtime arg (positional compile args past index 2 do not reach
    // this kernel).
    const uint32_t stop_addr = get_arg_val<uint32_t>(0);
    g_link.start(g_slot_base, stop_addr);
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    while (*ctl != eth_ptp::kCtlStop) {
        g_link.step();
    }
    g_link.stop();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr + 4) = 1;  // done, host polls this
#else
    for (uint32_t i = 0; i < NUM_MESSAGES; i++) {
        run_loop_iteration<true>(channel_addrs, channel_sync_addrs);
    }
#endif
}
