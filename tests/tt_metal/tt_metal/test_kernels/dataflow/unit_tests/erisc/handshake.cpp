// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "debug/waypoint.h"

typedef struct router_state {
    uint32_t sync_in;
    uint32_t padding_in[3];
    uint32_t sync_out;
    uint32_t padding_out[3];
    uint32_t scratch[4];
} router_state_t;

router_state_t router_state __attribute__((aligned(16), section(".fabric_router_data")));
router_state_t fvc_consumer_state
    __attribute__((aligned(16), section(".fabric_router_data")));  // replicate for each fvc
router_state_t fvc_producer_state
    __attribute__((aligned(16), section(".fabric_router_data")));  // replicate for each fvc
router_state_t fvcc_inbound_state
    __attribute__((aligned(16), section(".fabric_router_data")));  // inbound fabric virtual control channel
router_state_t fvcc_outbound_state
    __attribute__((aligned(16), section(".fabric_router_data")));  // outbound fabric virtual control channel
volatile router_state_t local_pull_request_temp
    __attribute__((aligned(16), section(".fabric_router_data")));        // replicate for each fvc
volatile router_state_t* local_pull_request = &local_pull_request_temp;  // replicate for each fvc

void kernel_main() {
    uint32_t handshake_value = get_arg_val<uint32_t>(0);
    uint32_t sync_in_addr = get_arg_val<uint32_t>(1);
    uint32_t sync_out_addr = get_arg_val<uint32_t>(2);
    uint32_t scratch_addr = get_arg_val<uint32_t>(3);

    bool src_ready = false;
    bool dest_ready = false;

    volatile tt_l1_ptr uint32_t* sync_in_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_in_addr);
    volatile tt_l1_ptr uint32_t* sync_out_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_out_addr);
    volatile tt_l1_ptr uint32_t* scratch_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);

    scratch_addr_ptr[0] = handshake_value;

    while (!src_ready or !dest_ready) {
        invalidate_l1_cache();
        if (sync_out_addr_ptr[0] != handshake_value) {
            internal_::eth_send_packet(0, scratch_addr >> 4, sync_in_addr >> 4, 1);
        } else {
            dest_ready = true;
        }

        WAYPOINT("DRDY");

        if (!src_ready && sync_in_addr_ptr[0] == handshake_value) {
            internal_::eth_send_packet(0, sync_in_addr >> 4, sync_out_addr >> 4, 1);
            src_ready = true;
        }

        WAYPOINT("SRDY");
    }

    WAYPOINT("NAVY");

    src_ready = false;
    dest_ready = false;

    constexpr uint32_t packet_word_size_bytes = 16;
    uint32_t sync_in_addr2 = ((uint32_t)&router_state.sync_in) / packet_word_size_bytes;
    uint32_t sync_out_addr2 = ((uint32_t)&router_state.sync_out) / packet_word_size_bytes;

    uint32_t scratch_addr2 = ((uint32_t)&router_state.scratch) / packet_word_size_bytes;

    router_state.scratch[0] = handshake_value;

    WATCHER_RING_BUFFER_PUSH((uint32_t)&router_state);
    WATCHER_RING_BUFFER_PUSH((uint32_t)&fvc_consumer_state);
    WATCHER_RING_BUFFER_PUSH((uint32_t)&fvc_producer_state);
    WATCHER_RING_BUFFER_PUSH((uint32_t)&fvcc_inbound_state);
    WATCHER_RING_BUFFER_PUSH((uint32_t)&fvcc_outbound_state);
    WATCHER_RING_BUFFER_PUSH((uint32_t)&local_pull_request_temp);

    while (!src_ready or !dest_ready) {
        invalidate_l1_cache();
        if (router_state.sync_out != handshake_value) {
            internal_::eth_send_packet(0, scratch_addr2, sync_in_addr2, 1);
        } else {
            dest_ready = true;
        }

        WAYPOINT("DDY2");

        if (!src_ready && router_state.sync_in == handshake_value) {
            internal_::eth_send_packet(0, sync_in_addr2, sync_out_addr2, 1);
            src_ready = true;
        }

        WAYPOINT("SDY2");
    }

    WAYPOINT("LOVE");
}
